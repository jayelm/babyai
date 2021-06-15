"""
Teacher that goes from demonstrations -> language
"""

import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class DemonstrationEncoder(nn.Module):
    def __init__(
        self, n_dirs, n_acts, obs_dim=(7, 7), dir_dim=50, act_dim=100, hidden_dim=512
    ):
        super().__init__()

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )
        n = obs_dim[0]
        m = obs_dim[1]
        self.obs_dim = obs_dim
        self.obs_hidden_dim = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.dir_dim = dir_dim
        self.act_dim = act_dim

        self.n_dirs = n_dirs
        self.n_acts = n_acts

        self.dir_embedding = nn.Embedding(n_dirs, dir_dim, padding_idx=0)
        self.act_embedding = nn.Embedding(n_acts, act_dim, padding_idx=0)

        self.input_dim = self.obs_hidden_dim + dir_dim + act_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)

    def forward(self, obs, dirs, acts, obs_lens, hidden=None):
        obs_shape = obs.shape
        obs_flat = obs.view(obs_shape[0] * obs_shape[1], *obs_shape[2:])
        obs_flat_enc = self.obs_encoder(obs_flat)
        obs_enc = obs_flat_enc.view(obs_shape[0], obs_shape[1], -1)

        dirs_enc = self.dir_embedding(dirs)
        acts_enc = self.act_embedding(acts)

        inp_enc = torch.cat((obs_enc, dirs_enc, acts_enc), 2)

        packed = pack_padded_sequence(
            inp_enc, obs_lens.cpu(), enforce_sorted=False, batch_first=True
        )

        _, hidden = self.gru(packed, hidden)
        return hidden[-1]


class LanguageDecoder(nn.Module):
    def __init__(self, vocab, embedding_dim=300, hidden_dim=512):
        super().__init__()
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.dense = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, enc_out, tgt, tgt_len):
        # embed your sequences
        embed_tgt = tgt @ self.embedding.weight

        # assume 1-layer gru - hidden state expansion
        enc_out = enc_out.unsqueeze(0)

        # No need to decode at the end position, so decrement tgt_len
        tgt_len = tgt_len - 1
        packed_input = pack_padded_sequence(
            embed_tgt, tgt_len.cpu(), enforce_sorted=False, batch_first=True
        )

        # shape = (tgt_len, batch, hidden_dim)
        packed_output, _ = self.gru(packed_input, enc_out)
        logits = self.dense(packed_output.data)

        return logits

    def sample(self, states, max_len=50, greedy=False, uniform_weight=0.0, softmax_temp=1.0, trim=False):
        batch_size = states.shape[0]  # 0th dim is singleton for GRU
        states = states.unsqueeze(0)
        # This contains are series of sampled onehot vectors
        lang = torch.zeros((batch_size, max_len, self.vocab_size), dtype=torch.float32).to(states.device)
        # Set all pad indices
        lang[:, :, self.vocab["<PAD>"]] = 1.0

        # Vector lengths
        lang_length = torch.ones(batch_size, dtype=torch.int64).to(states.device)
        # Binary indicator that we're done sampling
        done_sampling = torch.zeros(batch_size, dtype=torch.uint8).to(states.device)

        # first input is SOS token
        # (batch_size, n_vocab)
        inputs_onehot = torch.zeros(batch_size, self.vocab_size).to(states.device)
        inputs_onehot[:, self.vocab["<SOS>"]] = 1.0

        # (batch_size, len, n_vocab)
        inputs_onehot = inputs_onehot.unsqueeze(1)

        lang[:, 0, :] = inputs_onehot[:, 0, :]

        # compute embeddings
        # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
        inputs = inputs_onehot @ self.embedding.weight

        for i in range(max_len - 2):  # Have room for SOS, EOS if never sampled
            # FIXME: This is inefficient since I do sampling even if we've
            # finished generating language.
            if done_sampling.all():
                break
            outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
            outputs = self.dense(outputs)  # outputs: (B,V)

            if greedy:
                predicted = outputs.max(1)[1]
                predicted = predicted.unsqueeze(1)
                raise NotImplementedError
            else:
                # Normalize first
                outputs = torch.log_softmax(outputs, -1)

                if uniform_weight != 0.0:
                    uniform_outputs = torch.full_like(outputs, np.log(1 / outputs.shape[1]))
                    # Weighted average of logits and uniform distribution in log space
                    combined_outputs = torch.stack([
                        uniform_outputs + np.log(uniform_weight),
                        outputs + np.log(1 - uniform_weight),
                    ], 2)
                    outputs = torch.logsumexp(combined_outputs, 2)

                if softmax_temp != 1.0:
                    outputs = outputs / softmax_temp

                predicted_onehot = F.gumbel_softmax(outputs, tau=1.0, hard=True)

                # Zero out predicted where done sampling
                predicted_onehot[done_sampling.bool(), :, self.vocab["<PAD>"]] = 0.0
                # ADD ONE since we filled 0 in already
                lang[:, i + 1, :] = predicted_onehot[:, 0, :]

            predicted_n = predicted_onehot.argmax(2).squeeze(1)

            # If not done sampling, increment length by 1
            lang_length += (1 - done_sampling)
            # If we have sampled EOS, set done sampling
            done_sampling = done_sampling | (predicted_n == self.vocab["<EOS>"])

            inputs = predicted_onehot @ self.embedding.weight
        else:
            # We never broke, so need to add EOS for some trailing samples
            eos_onehot = torch.zeros(batch_size, self.vocab_size).to(states.device)
            eos_onehot[:, self.vocab["<EOS>"]] = 1.0
            lang[:, -1, :] = eos_onehot

            lang_length += (1 - done_sampling)

        # Concat tokens and trim
        if trim:
            max_lang_len = lang_length.max()
            lang = lang[:, :max_lang_len, :]

        return lang, lang_length


def make_vocab(vocab_size):
    vocab = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
    }
    for i in range(vocab_size - 3):
        tok = str(i)
        vocab[tok] = len(vocab)

    rev_vocab = {v: k for k, v in vocab.items()}
    return vocab, rev_vocab


class Teacher(nn.Module):
    def __init__(self, n_dirs, n_acts, vocab_size):
        super().__init__()
        self.encoder = DemonstrationEncoder(n_dirs, n_acts)
        self.vocab_size = vocab_size
        self.vocab, self.rev_vocab = make_vocab(vocab_size)
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self.decoder = LanguageDecoder(self.vocab)

    def forward(self, obs, dirs, acts, obs_lens, langs, lang_lens, hidden=None):
        demos_enc = self.encoder(obs, dirs, acts, obs_lens)

        langs_pred = self.decoder(demos_enc, langs, lang_lens)
        return langs_pred

    def sample(self, obs, dirs, acts, obs_lens, **kwargs):
        demos_enc = self.encoder(obs, dirs, acts, obs_lens)
        langs_pred = self.decoder.sample(demos_enc, **kwargs)
        return langs_pred
