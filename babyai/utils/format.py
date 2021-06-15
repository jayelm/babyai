import os
import json
import numpy
import re
import torch
import babyai.rl
from torch.nn.utils.rnn import pad_sequence

from .. import utils


def get_vocab_path(model_name):
    return os.path.join(utils.get_model_dir(model_name), "vocab.json")


class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        self.max_size = 100
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))
        else:
            self.vocab = {}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self, path=None):
        if path is None:
            path = self.path
        utils.create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)


class InstructionsPreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        # FIXME - you may need to make this more like the emcomm vocab (SOS, EOS, etc)
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def __call__(self, obss, device=None):
        raw_instrs = []
        max_instr_len = 0

        for obs in obss:
            tokens = re.findall("([a-z]+)", obs["mission"].lower())
            instr = numpy.array([self.vocab[token] for token in tokens])
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, :len(instr)] = instr

        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs


def demo_collate(batch):
    obs, dirs, acts, obs_lens, langs, lang_lens = zip(*batch)
    obs_lens = torch.tensor(obs_lens)
    lang_lens = torch.tensor(lang_lens)
    obs_pad = pad_sequence(obs, batch_first=True, padding_value=0)
    dirs_pad = pad_sequence(dirs, batch_first=True, padding_value=0)
    acts_pad = pad_sequence(acts, batch_first=True, padding_value=0)
    langs_pad = pad_sequence(langs, batch_first=True, padding_value=0)
    return {
        "obs": obs_pad,
        "dirs": dirs_pad,
        "acts": acts_pad,
        "obs_lens": obs_lens,
        "langs": langs_pad,
        "lang_lens": lang_lens,
    }


class DemoPreprocessor(object):
    def __init__(self):
        pass

    def __call__(self, demos, device=None):
        demo_tpls = []
        for demo in demos:
            demo_tpl = (
                demo["obs"],
                demo["dirs"],
                demo["acts"],
                demo["obs_len"],
                demo["lang"],
                demo["lang_len"],
            )
            demo_tpls.append(demo_tpl)

        demo_proc = demo_collate(demo_tpls)
        demo_proc = {k: t.to(device) for k, t in demo_proc.items()}
        return demo_proc


def to_emergent_text(idxs, join=False, eos=None):
    texts = []
    for lang in idxs:
        toks = []
        for i in lang:
            i_item = i.item()
            i = str(i_item)
            toks.append(i)
            if eos is not None and i_item == eos:
                break
        if join:
            texts.append(" ".join(toks))
        else:
            texts.append(toks)
    return texts


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        images = torch.tensor(images, device=device, dtype=torch.float)
        return images


class IntImagePreprocessor(object):
    def __init__(self, num_channels, max_high=255):
        self.num_channels = num_channels
        self.max_high = max_high
        self.offsets = numpy.arange(num_channels) * max_high
        self.max_size = int(num_channels * max_high)

    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        # The padding index is 0 for all the channels
        images = (images + self.offsets) * (images > 0)
        images = torch.tensor(images, device=device, dtype=torch.long)
        return images


class ObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class IntObssPreprocessor(object):
    def __init__(self, model_name, obs_space, load_vocab_from=None):
        image_obs_space = obs_space.spaces["image"]
        self.image_preproc = IntImagePreprocessor(image_obs_space.shape[-1],
                                                  max_high=image_obs_space.high.max())
        self.instr_preproc = InstructionsPreprocessor(load_vocab_from or model_name)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": self.image_preproc.max_size,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_
