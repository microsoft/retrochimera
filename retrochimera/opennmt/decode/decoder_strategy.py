"""
MIT License

Copyright (c) 2017-Present OpenNMT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Original Source: https://github.com/OpenNMT/OpenNMT-py/blob/v3.5.1/onmt/translate/decode_strategy.py
"""
import random
from copy import deepcopy

import numpy as np
import torch


def tile(x: torch.Tensor, count: int, dim: int = 0):
    """Tiles x on dimension dim count times.

    Args:
        x (torch.Tensor): The tensor to tile.
        memory_bank as an example x, shape (padded_src_len, batch_size, hidden_dim), where dim=1.
        count (int): The number of times to tile x.
        dim (int): The dimension to tile x on. Default is 0.

    Returns:
        x (torch.Tensor): The tiled tensor.
        For memory_bank as an example x, shape (padded_src_len, batch_size * count/beam_size, hidden_dim), where dim=1.

    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()

    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )

    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def set_random_seed(seed, is_cuda):
    """Sets the random seed."""
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True
        # This one is needed for various tranfroms
        np.random.seed(seed)

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)


def report_matrix(row_label, column_label, matrix):
    header_format = "{:>10.10} " + "{:>10.7} " * len(row_label)
    row_format = "{:>10.10} " + "{:>10.7f} " * len(row_label)
    output = header_format.format("", *row_label) + "\n"
    for word, row in zip(column_label, matrix):
        max_index = row.index(max(row))
        row_format = row_format.replace("{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
        row_format = row_format.replace("{:*>10.7f} ", "{:>10.7f} ", max_index)
        output += row_format.format(word, *row) + "\n"
        row_format = "{:>10.10} " + "{:>10.7f} " * len(row_label)
    return output


class DecodeStrategy(object):
    """Base class for generation strategies.

    Args:
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        unk (int): Magic integer in output vocab.
        batch_size (int): Current batch size.
        parallel_paths (int): Decoding strategies like beam search
            use parallel paths. Each batch is repeated ``parallel_paths``
            times in relevant state tensors.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        max_length (int): Longest acceptable sequence, not counting
            begin-of-sentence (presumably there has been no EOS
            yet if max_length is used as a cutoff).
        ban_unk_token (Boolean): Whether unk token is forbidden
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[int]): If a gram contains any of these
            tokens, it may repeat.
        return_attention (bool): Whether to work with attention too. If this
            is true, it is assumed that the decoder is attentional.

    Attributes:
        pad (int): See above.
        bos (int): See above.
        eos (int): See above.
        unk (int): See above.
        predictions (list[list[LongTensor]]): For each batch, holds a
            list of beam prediction sequences.
        scores (list[list[FloatTensor]]): For each batch, holds a
            list of scores.
        attention (list[list[FloatTensor or list[]]]): For each
            batch, holds a list of attention sequence tensors
            (or empty lists) having shape ``(step, inp_seq_len)`` where
            ``inp_seq_len`` is the length of the sample (not the max
            length of all inp seqs).
        alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
            This sequence grows in the ``step`` axis on each call to
            :func:`advance()`.
        is_finished (ByteTensor or NoneType): Shape
            ``(B, parallel_paths)``. Initialized to ``None``.
        alive_attn (FloatTensor or NoneType): If tensor, shape is
            ``(step, B x parallel_paths, inp_seq_len)``, where ``inp_seq_len``
            is the (max) length of the input sequence.
        target_prefix (LongTensor or NoneType): If tensor, shape is
            ``(B x parallel_paths, prefix_seq_len)``, where ``prefix_seq_len``
            is the (max) length of the pre-fixed prediction.
        min_length (int): See above.
        max_length (int): See above.
        ban_unk_token (Boolean): See above.
        block_ngram_repeat (int): See above.
        exclusion_tokens (set[int]): See above.
        return_attention (bool): See above.
        done (bool): See above.
    """

    def __init__(
        self,
        pad: int = 1,
        bos: int = 2,
        eos: int = 3,
        unk: int = 0,
        batch_size: int = 32,
        parallel_paths: int = 10,  # beam size
        min_length: int = 0,
        max_length: int = 512,
        block_ngram_repeat: int = 0,
        exclusion_tokens: set[int] = set(),
        ban_unk_token: bool = False,
        return_attention: bool = False,
    ):
        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.unk = unk

        self.batch_size = batch_size
        self.parallel_paths = parallel_paths
        self.beam_size = parallel_paths

        self.min_length = min_length
        self.max_length = max_length
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens
        self.ban_unk_token = ban_unk_token
        self.return_attention = return_attention

        # result caching
        self.predictions: list = [[] for _ in range(batch_size)]
        self.scores: list = [[] for _ in range(batch_size)]
        self.attention: list = [[] for _ in range(batch_size)]
        self.hypotheses: list = [[] for _ in range(batch_size)]

        self.alive_attn = None
        self.done = False

        n_paths = batch_size * parallel_paths
        self.forbidden_tokens: list = [dict() for _ in range(n_paths)]

    def get_device_from_memory_bank(self, memory_bank):
        if isinstance(memory_bank, tuple):
            mb_device = memory_bank[0].device
        else:
            mb_device = memory_bank.device
        return mb_device

    def initialize_tile(
        self, memory_bank: torch.Tensor, src_lengths, src_map=None, target_prefix=None
    ):
        """Tile everything for the beam search loop.

        Args:
            memory_bank (FloatTensor): The encoder memory bank. shape (padded_src_len, batch_size, hidden_dim).
            src_lengths (LongTensor): The source lengths. shape (batch_size,).
        """

        def fn_map_state(state, dim):
            return tile(state, self.beam_size, dim=dim)

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, self.beam_size, dim=1) for x in memory_bank)
        elif memory_bank is not None:
            memory_bank = tile(memory_bank, self.beam_size, dim=1)
        if src_map is not None:
            src_map = tile(src_map, self.beam_size, dim=1)

        self.memory_lengths = tile(src_lengths, self.beam_size)  # shape: (batch_size * beam_size,)
        if target_prefix is not None:
            target_prefix = tile(target_prefix, self.beam_size, dim=1)

        return fn_map_state, memory_bank, src_map, target_prefix

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None, target_prefix=None):
        """DecodeStrategy subclasses should override :func:`initialize()`.

        Args:
            memory_bank (FloatTensor): The encoder memory bank. shape (padded_src_len, batch_size * beam_size, hidden_dim).
            src_lengths (LongTensor): The source lengths. shape (batch_size * beam_size,).

        `initialize` should be called before all actions.
        used to prepare necessary ingredients for decode.
        """
        if device is None:
            device = torch.device("cpu")
        self.alive_seq = torch.full(
            [self.batch_size * self.parallel_paths, 1], self.bos, dtype=torch.long, device=device
        )
        self.is_finished = torch.zeros(
            [self.batch_size, self.parallel_paths], dtype=torch.uint8, device=device
        )
        if target_prefix is not None:
            seq_len, batch_size, n_feats = target_prefix.size()
            assert (
                batch_size == self.batch_size * self.parallel_paths
            ), "forced target_prefix should've extend to same number of path!"
            target_prefix_words = target_prefix[:, :, 0].transpose(0, 1)
            target_prefix = target_prefix_words[:, 1:]  # remove bos

            # fix length constraint and remove eos from count
            prefix_non_pad = target_prefix.ne(self.pad).sum(dim=-1).tolist()
            self.max_length += max(prefix_non_pad) - 1
            self.min_length += min(prefix_non_pad) - 1

        self.target_prefix = target_prefix  # NOTE: forced prefix words
        return None, memory_bank, src_lengths, src_map

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_unk_removed(self, log_probs):
        if self.ban_unk_token:
            log_probs[:, self.unk] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def block_ngram_repeats(self, log_probs):
        """
        We prevent the beam from going in any direction that would repeat any
        ngram of size <block_ngram_repeat> more thant once.

        The way we do it: we maintain a list of all ngrams of size
        <block_ngram_repeat> that is updated each time the beam advances, and
        manually put any token that would lead to a repeated ngram to 0.

        This improves on the previous version's complexity:
           - previous version's complexity: batch_size * beam_size * len(self)
           - current version's complexity: batch_size * beam_size

        This improves on the previous version's accuracy;
           - Previous version blocks the whole beam, whereas here we only
            block specific tokens.
           - Before the translation would fail when all beams contained
            repeated ngrams. This is sure to never happen here.
        """

        # we don't block nothing if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # we can't block nothing beam's too short
        if len(self) < self.block_ngram_repeat:
            return

        n = self.block_ngram_repeat - 1
        for path_idx in range(self.alive_seq.shape[0]):
            # we check paths one by one

            current_ngram = tuple(self.alive_seq[path_idx, -n:].tolist())
            forbidden_tokens = self.forbidden_tokens[path_idx].get(current_ngram, None)
            if forbidden_tokens is not None:
                log_probs[path_idx, list(forbidden_tokens)] = -10e20

    def maybe_update_forbidden_tokens(self):
        """We complete and reorder the list of forbidden_tokens"""

        # we don't forbid nothing if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # we can't forbid nothing if beam's too short
        if len(self) < self.block_ngram_repeat:
            return

        n = self.block_ngram_repeat

        forbidden_tokens = list()
        for path_idx, seq in zip(self.select_indices, self.alive_seq):

            # Reordering forbidden_tokens following beam selection
            # We rebuild a dict to ensure we get the value and not the pointer
            forbidden_tokens.append(deepcopy(self.forbidden_tokens[path_idx]))

            # Grabing the newly selected tokens and associated ngram
            current_ngram = tuple(seq[-n:].tolist())

            # skip the blocking if any token in current_ngram is excluded
            if set(current_ngram) & self.exclusion_tokens:
                continue

            forbidden_tokens[-1].setdefault(current_ngram[:-1], set())
            forbidden_tokens[-1][current_ngram[:-1]].add(current_ngram[-1])

        self.forbidden_tokens = forbidden_tokens

    def target_prefixing(self, log_probs):
        """Fix the first part of predictions with `self.target_prefix`.

        Args:
            log_probs (FloatTensor): logits of size ``(B, vocab_size)``.

        Returns:
            log_probs (FloatTensor): modified logits in ``(B, vocab_size)``.
        """
        _B, vocab_size = log_probs.size()
        step = len(self)
        if self.target_prefix is not None and step <= self.target_prefix.size(1):
            pick_idx = self.target_prefix[:, step - 1].tolist()  # (B)
            pick_coo = [
                [path_i, pick]
                for path_i, pick in enumerate(pick_idx)
                if pick not in [self.eos, self.pad]
            ]
            mask_pathid = [
                path_i for path_i, pick in enumerate(pick_idx) if pick in [self.eos, self.pad]
            ]
            if len(pick_coo) > 0:
                pick_coo = torch.tensor(pick_coo).to(self.target_prefix)
                pick_fill_value = torch.ones([pick_coo.size(0)], dtype=log_probs.dtype)
                # pickups: Tensor where specified index were set to 1, others 0
                pickups = torch.sparse_coo_tensor(
                    pick_coo.t(), pick_fill_value, size=log_probs.size(), device=log_probs.device
                ).to_dense()
                # dropdowns: opposite of pickups, 1 for those shouldn't pick
                dropdowns = torch.ones_like(pickups) - pickups
                if len(mask_pathid) > 0:
                    path_mask = torch.zeros(_B).to(self.target_prefix)
                    path_mask[mask_pathid] = 1
                    path_mask = path_mask.unsqueeze(1).to(dtype=bool)
                    dropdowns = dropdowns.masked_fill(path_mask, 0)
                # Minus dropdowns to log_probs making probabilities of
                # unspecified index close to 0
                log_probs -= 10000 * dropdowns
        return log_probs

    def maybe_update_target_prefix(self, select_index):
        """We update / reorder `target_prefix` for alive path."""
        if self.target_prefix is None:
            return
        # prediction step have surpass length of given target_prefix,
        # no need to further change this attr
        if len(self) > self.target_prefix.size(1):
            return
        self.target_prefix = self.target_prefix.index_select(0, select_index)

    def advance(self, log_probs, attn):
        """DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """

        raise NotImplementedError()

    def update_finished(self):
        """DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """

        raise NotImplementedError()

    @property
    def current_predictions(self):
        raise NotImplementedError()
