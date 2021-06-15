from abc import ABC, abstractmethod
import torch
import numpy as np
import contextlib
import pandas as pd

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
from babyai.utils.demos import load_demos, mission_to_demos, Demos
from babyai.utils.format import DemoPreprocessor, to_emergent_text


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = ParallelEnv(envs)
        # Load demos
        self.demos = Demos(load_demos("./demos/BabyAI-GoToLocal-v0.pkl"))
        self.demo_preproc = DemoPreprocessor()
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs


        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.instrs = [None]*(shape[0])
        self.instrs_lens = [None]*(shape[0])

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def generate_instructions(self, mask=None):
        """
        Mask here is 0 if environment is done and we need to resample.
        """
        obs_missions = []
        demos = []
        if mask is None:
            mask = [0 for _ in self.obs]
        for mask_value, obs in zip(mask, self.obs):
            if mask_value:
                continue
            mission = obs["mission"]
            obs_missions.append(mission)

            # Choose random demo
            randi = np.random.randint(len(self.demos.missions2demos[mission]))
            demo = self.demos.missions2demos[mission][randi]
            demos.append(demo)

        # Preprocess demos
        demos = self.demo_preproc(demos, device=self.device)

        instr, instr_len = self.acmodel.sample_from_teacher(demos)
        return instr, instr_len

    def detach_old(self):
        self.memory = self.memory.detach()
        self.memories = self.memories.detach()
        self.log_probs = self.log_probs.detach()
        self.advantages = self.advantages.detach()
        self.values = self.values.detach()

    def collect_experiences(self, mode):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Params
        ------
        mode : str (one of 'student', 'teacher')
            which mode to run in

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        if mode not in ['student', 'teacher']:
            raise ValueError(f"unknown mode {mode}")

        self.detach_old()

        # if mode == 'student', no gradients calculated, since we're
        # just collecting experience.  if mode == 'teacher', we're
        # explicitly trying to optimize teacher parameters. student
        # params are frozen but gradients are calculated all the way
        # through.
        context = torch.no_grad if mode == 'student' else contextlib.nullcontext
        log_instrs = []

        if self.acmodel.gen_instr:
            # Generate instructions, fixing teacher parameters.
            with context():
                instr, instr_len = self.generate_instructions()
                # Add to instructions
                instr_text = to_emergent_text(instr.argmax(2), eos=2, join=True)
                gt_text = [o["mission"] for o in self.obs]
                log_instrs.extend(zip(instr_text, gt_text))
        else:
            instr = None
            instr_len = None

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            # Load demos.
            # TODO: this should be abstracted
            # into the environment, not obs.
            # FIXME - I *believe* obs should be the same each time.

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with context():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1), instr=instr, instr_len=instr_len)
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            if mode == 'student':
                action = dist.sample()
            else:
                # Deterministic
                action = dist.logits.argmax(1)

            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)

            # Update experiences values

            self.obss[i] = self.obs
            self.instrs[i] = instr
            self.instrs_lens[i] = instr_len

            self.memories[i] = self.memory
            self.memory = memory

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            mask_bool = self.mask.bool()

            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

            if not mask_bool.all():
                # Resample instructions where done
                with context():
                    new_instr, new_instr_len = self.generate_instructions(mask=mask_bool)
                    new_instr_padded = torch.zeros_like(instr)
                    new_instr_padded[~mask_bool] = new_instr
                    new_instr_len_padded = torch.zeros_like(instr_len)
                    new_instr_len_padded[~mask_bool] = new_instr_len

                    instr = torch.where(mask_bool.unsqueeze(1).unsqueeze(1), instr, new_instr_padded)
                    instr_len = torch.where(mask_bool, instr_len, new_instr_len_padded)

                    # Log new instructions
                    new_instr_text = to_emergent_text(new_instr.argmax(2), eos=2, join=True)
                    new_gt_text = [o["mission"] for o, m in zip(self.obs, mask_bool) if not m]
                    assert len(new_instr_text) == len(new_gt_text)
                    log_instrs.extend(zip(new_instr_text, new_gt_text))

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with context():
            next_value = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1), instr=instr, instr_len=instr_len)['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        # L = sequence length
        # T x P x L x D -> P x T x L x D -> (P * T) x L x D
        exps.instr = torch.stack(self.instrs).transpose(0, 1).reshape(-1, instr.shape[1], instr.shape[2])
        # T x P -> P x T -> P * T
        exps.instr_len = torch.stack(self.instrs_lens).transpose(0, 1).reshape(-1)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log_instrs_df = pd.DataFrame.from_records(log_instrs, columns=["emergent", "gt"])
        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
            "lang": log_instrs_df,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_teacher_parameters(self):
        pass

    @abstractmethod
    def update_student_parameters(self):
        pass
