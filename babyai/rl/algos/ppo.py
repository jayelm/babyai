import numpy
import torch
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


from babyai.rl.algos.base import BaseAlgo


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None, teacher_lr=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         aux_info)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        if teacher_lr is None:
            teacher_lr = lr
        self.student_optimizer = torch.optim.Adam(self.acmodel.student_parameters(), lr, (beta1, beta2), eps=adam_eps)
        self.teacher_optimizer = torch.optim.Adam(self.acmodel.teacher_parameters(), teacher_lr, (beta1, beta2), eps=adam_eps)
        self.batch_num = 0

    def update_teacher_parameters(self):
        self.acmodel.freeze_student()
        # Collect experiences

        # Here we train teacher generation network to maximize student reward,
        # assuming fixed student reward func. This means we do backprop on
        # entire batches of instructions, rather than on batches of frames.
        # TODO: probably make batch size/epochs smaller.
        for epoch_i in range(self.epochs):
            exps, logs = self.collect_experiences('teacher')
            # Initialize log values

            #  log_entropies = []
            #  log_values = []
            #  log_policy_losses = []
            #  log_value_losses = []
            log_grad_norms = []
            log_losses = []

            teacher_loss = -exps.returnn.mean()
            self.teacher_optimizer.zero_grad()
            teacher_loss.backward()
            grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.teacher_parameters() if p.grad is not None) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.acmodel.teacher_parameters(), self.max_grad_norm)
            self.teacher_optimizer.step()

            # Update log values
            # Is it that the obs are partial...? need to detach obs? make sure everything is detached has no grad fn

            #  log_entropies.append(batch_entropy)
            #  log_values.append(batch_value)
            #  log_policy_losses.append(batch_policy_loss)
            #  log_value_losses.append(batch_value_loss)
            log_grad_norms.append(grad_norm.item())
            log_losses.append(teacher_loss.item())

        # Log some values
        logs["loss"] = numpy.mean(log_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)

        return logs

    def update_student_parameters(self):
        self.acmodel.freeze_teacher()
        # Collect experiences

        exps, logs = self.collect_experiences('student')
        '''
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            batch_idxs = self._get_batches_starting_indexes()
            for batch_i, inds in enumerate(batch_idxs):
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss
                    model_results = self.acmodel(
                        sb.obs,
                        memory * sb.mask,
                        instr=sb.instr,
                        instr_len=sb.instr_len,
                    )
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']
                    extra_predictions = model_results['extra_predictions']

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.student_optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.student_parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.student_parameters(), self.max_grad_norm)
                self.student_optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
