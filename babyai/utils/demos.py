import os
import pickle

from .. import utils
import blosc
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
import torch


def make_vocab(items, str_format=False):
    w2i = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
    }
    for item in items:
        if str_format:
            item = item.split(" ")
        for tok in item:
            tok = str(tok)
            if tok not in w2i:
                w2i[tok] = len(w2i)
    return w2i


class Demos:
    def __init__(self, demos):
        self.obs = []
        self.obs_lens = []
        self.langs = []
        self.lang_lens = []
        self.dirs = []
        self.acts = []

        missions, packed_obs, directions, actions = zip(*demos)

        self.lang_w2i = make_vocab(missions, str_format=True)
        self.dirs_w2i = make_vocab(directions)
        self.acts_w2i = make_vocab(actions)

        self.lang_i2w = {v: k for k, v in self.lang_w2i.items()}
        self.dirs_i2w = {v: k for k, v in self.dirs_w2i.items()}
        self.acts_i2w = {v: k for k, v in self.acts_w2i.items()}

        self.missions2demos = defaultdict(list)

        for orig_mission, packed_obs, dirs, acts in demos:
            missions_i = [1, *(self.lang_w2i[t] for t in orig_mission.split(" ")), 2]
            # src doesn't need start of sentence token.
            dirs_i = [self.dirs_w2i[str(t)] for t in dirs]
            acts_i = [self.acts_w2i[str(t)] for t in acts]

            missions_i = np.array(missions_i, dtype=np.int64)
            dirs_i = np.array(dirs_i, dtype=np.int64)
            acts_i = np.array(acts_i, dtype=np.int64)

            obs = blosc.unpack_array(packed_obs)
            # Transpose - channels first
            obs = np.transpose(obs, (0, 3, 1, 2))

            demo_processed = {
                "obs": torch.from_numpy(obs),
                "obs_len": obs.shape[0],
                "lang": torch.from_numpy(missions_i),
                "lang_len": len(missions_i),
                "dirs": torch.from_numpy(dirs_i),
                "acts": torch.from_numpy(acts_i),
            }
            self.missions2demos[orig_mission].append(demo_processed)

            self.obs.append(obs)
            self.obs_lens.append(obs.shape[0])
            self.langs.append(missions_i)
            self.lang_lens.append(len(missions_i))
            self.dirs.append(dirs_i)
            self.acts.append(acts_i)

        self.missions2demos = dict(self.missions2demos)

    def lang_to_onehot(self, lang):
        assert lang.ndim < 3, "weird dimensions"
        return F.one_hot(
            lang,
            num_classes=len(self.lang_w2i)
        ).float()

    def dirs_to_onehot(self, dirs):
        assert dirs.ndim < 3, "weird dimensions"
        return F.one_hot(
            dirs,
            num_classes=len(self.dirs_w2i)
        )

    def acts_to_onehot(self, acts):
        assert acts.ndim < 3, "weird dimensions"
        return F.one_hot(
            acts,
            num_classes=len(self.acts_w2i)
        )

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.obs[i]),
            torch.from_numpy(self.dirs[i]),
            torch.from_numpy(self.acts[i]),
            self.obs_lens[i],
            self.lang_to_onehot(torch.from_numpy(self.langs[i])),
            self.lang_lens[i],
        )

    def __len__(self):
        return len(self.langs)


def get_demos_path(demos=None, env=None, origin=None, valid=False):
    valid_suff = '_valid' if valid else ''
    demos_path = (demos + valid_suff
                  if demos
                  else env + "_" + origin + valid_suff) + '.pkl'
    return os.path.join(utils.storage_dir(), 'demos', demos_path)


def load_demos(path, raise_not_found=True):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No demos found at {}".format(path))
        else:
            return []


def save_demos(demos, path):
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))


def synthesize_demos(demos):
    print('{} demonstrations saved'.format(len(demos)))
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    if len(demos) > 0:
        print('Demo num frames: {}'.format(num_frames_per_episode))


def transform_demos(demos):
    '''
    takes as input a list of demonstrations in the format generated with `make_agent_demos` or `make_human_demos`
    i.e. each demo is a tuple (mission, blosc.pack_array(np.array(images)), directions, actions)
    returns demos as a list of lists. Each demo is a list of (obs, action, done) tuples
    '''
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]

        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        for i in range(n_observations):
            obs = {'image': all_images[i],
                   'direction': directions[i],
                   'mission': mission}
            action = actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos


def mission_to_demos(demos):
    missions = defaultdict(list)
    for demo in demos:
        mission = demo[0]
        missions[mission].append(demo)
    return dict(missions)
