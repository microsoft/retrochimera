"""
MIT License

Copyright (c) 2017-Present OpenNMT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Original Source: https://github.com/OpenNMT/OpenNMT-py/blob/v3.5.1/onmt/translate/beam_search.py
Modifications:
1. Simplified the structure by removing the `BeamSearchLM` and `GNMTGlobalScorer` class, and merged the `BeamSearchBase` class into the `BeamSearch` class.
2. Introduced the `customised_beam_search` attribute and corresponding logic to the `BeamSearch` class, enabling optimized beam search for retrosynthesis prediction.
"""
import torch

from retrochimera.opennmt.decode.decoder_strategy import DecodeStrategy


class BeamSearch(DecodeStrategy):
    """Beam search for seq2seq/encoder-decoder models.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        unk (int): See base.
        batch_size (int): See base.
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        n_best (int): Don't stop until at least this many beams have reached EOS.
        min_length (int): See base.
        max_length (int): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        ratio (float).
        ban_unk_token (bool): See base.
        return_attention (bool): See base.

    Attributes:
        top_beam_finished (ByteTensor): Shape (batch_size,).
        _batch_offset (LongTensor): Shape (batch_size,).
        _beam_offset (LongTensor): Shape (batch_size * beam_size,).
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape (batch_size, beam_size,). These are the scores used for the topk operation.
        memory_lengths (LongTensor): Lengths of encodings. Used for masking attentions.
        select_indices (LongTensor or NoneType): Shape (batch_size * beam_size,). This is just a flat view of the ``_batch_index``.
        topk_scores (FloatTensor): Shape (batch_size, beam_size)``. These are the scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__(
        self,
        pad: int = 1,
        bos: int = 2,
        eos: int = 3,
        unk: int = 0,
        batch_size: int = 32,
        beam_size: int = 10,
        n_best: int = 10,
        min_length: int = 0,
        max_length: int = 512,
        block_ngram_repeat: int = 0,
        exclusion_tokens: set[int] = set(),
        ratio: float = 0,
        ban_unk_token: bool = False,
        return_attention: bool = False,
        customised_beam_search: bool = False,
    ):
        super(BeamSearch, self).__init__(
            pad,
            bos,
            eos,
            unk,
            batch_size,
            beam_size,
            min_length,
            max_length,
            block_ngram_repeat,
            exclusion_tokens,
            ban_unk_token,
            return_attention,
        )

        # beam parameters
        self.beam_size = beam_size
        self.n_best = n_best
        self.ratio = ratio

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        # BoolTensor was introduced in pytorch 1.2
        try:
            self.top_beam_finished = self.top_beam_finished.bool()
        except AttributeError:
            pass
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)

        self.select_indices = None
        self.done = False
        # "global state" of the old beam
        self.memory_lengths = None

        self.customised_beam_search = customised_beam_search

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None, target_prefix=None):
        """Initialize for decoding.
        Repeat src objects `beam_size` times.

        Args:
            memory_bank (FloatTensor): The memory bank to perform attention. shape (padded_src_len, batch_size, hidden_dim).
            src_lengths (LongTensor): The length of each source sequence. shape (batch_size,).
        """

        (fn_map_state, memory_bank, src_map, target_prefix) = self.initialize_tile(
            memory_bank, src_lengths, src_map, target_prefix
        )
        # shape of memory_bank: (padded_src_len, batch_size * beam_size, hidden_dim)
        # shape of self.memory_lengths: (batch_size * beam_size,)

        if device is None:
            device = self.get_device_from_memory_bank(memory_bank)

        self.initialize_(memory_bank, self.memory_lengths, src_map, device, target_prefix)

        return fn_map_state, memory_bank, self.memory_lengths, src_map

    def initialize_(self, memory_bank, memory_lengths, src_map, device, target_prefix):
        """Initialize for decoding.

        Args:
            memory_bank (FloatTensor): The memory bank to perform attention. shape (padded_src_len, batch_size * beam_size, hidden_dim).
            memory_lengths (LongTensor): The length of each source sequence. shape (batch_size * beam_size,).
        """
        super(BeamSearch, self).initialize(
            memory_bank, memory_lengths, src_map, device, target_prefix
        )

        self.best_scores = torch.full(
            [self.batch_size], -1e10, dtype=torch.float, device=device
        )  # only when ratio > 0
        self._beam_offset = torch.arange(
            0,
            self.batch_size * self.beam_size,
            step=self.beam_size,
            dtype=torch.long,
            device=device,
        )  # shape: (batch_size,)
        self.topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (self.beam_size - 1), device=device)
            .repeat(self.batch_size)
            .reshape(self.batch_size, self.beam_size)
        )
        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.float, device=device
        )
        self.topk_ids = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.long, device=device
        )
        self._batch_index = torch.empty(
            [self.batch_size, self.beam_size], dtype=torch.long, device=device
        )

    @property
    def current_predictions(self) -> torch.Tensor:
        return self.alive_seq[:, -1]

    @property
    def batch_offset(self):
        return self._batch_offset

    def _pick(self, log_probs, out=None):
        """Take a token pick decision for a step.

        Args:
            log_probs (FloatTensor): (B * beam_size, vocab_size)
            out (Tensor, LongTensor): output buffers to reuse, optional.

        Returns:
            topk_scores (FloatTensor): (B, beam_size)
            topk_ids (LongTensor): (B, beam_size)
        """
        vocab_size = log_probs.size(-1)
        # maybe fix some prediction at this step by modifying log_probs
        log_probs = self.target_prefixing(log_probs)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs.reshape(-1, self.beam_size * vocab_size)
        if out is not None:
            torch.topk(curr_scores, self.beam_size, dim=-1, out=out)
            return
        topk_scores, topk_ids = torch.topk(curr_scores, self.beam_size, dim=-1)
        return topk_scores, topk_ids

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance

        if not self.customised_beam_search:
            self.topk_log_probs.masked_fill_(
                self.is_finished, -1e10
            )  # shape: (batch_size, beam_size)

        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to("cpu")
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        attention = (
            self.alive_attn.view(step - 1, _B_old, self.beam_size, self.alive_attn.size(-1))
            if self.alive_attn is not None
            else None
        )
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):  # Batch level
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero(as_tuple=False).view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:  # Beam level: finished beam j in batch i
                if self.ratio > 0:
                    s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s
                if not self.customised_beam_search:
                    self.hypotheses[b].append(
                        (
                            self.topk_scores[i, j],
                            predictions[i, j, 1:],  # Ignore start_token.
                            attention[:, i, j, : self.memory_lengths[i]]
                            if attention is not None
                            else None,
                        )
                    )
                else:
                    if predictions[i, j, 1:].size(-1) == 0 or predictions[i, j, -2] != self.eos:
                        self.hypotheses[b].append(
                            (
                                self.topk_scores[i, j],
                                predictions[i, j, 1:],  # Ignore start_token.
                                attention[:, i, j, : self.memory_lengths[i]]
                                if attention is not None
                                else None,
                            )
                        )
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self.memory_lengths[i] * self.ratio
                finish_flag = (
                    (self.topk_scores[i, 0] / pred_len) <= self.best_scores[b]
                ) or self.is_finished[i].all()
            else:
                if not self.customised_beam_search:
                    finish_flag = self.top_beam_finished[i] != 0
                else:
                    finish_flag = self.is_finished[i].all()  # shape: (beam_size,)

            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)  # ``(batch, n_best,)``
                    self.attention[b].append(attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        self.remove_finished_batches(_B_new, _B_old, non_finished, predictions, attention, step)

    def remove_finished_batches(self, _B_new, _B_old, non_finished, predictions, attention, step):
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0, non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished).view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
        self.maybe_update_target_prefix(self.select_indices)
        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention.index_select(1, non_finished).view(
                step - 1, _B_new * self.beam_size, inp_seq_len
            )

    def advance(self, log_probs, attn):
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)
        self.ensure_unk_removed(log_probs)

        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = 1.0

        curr_scores = log_probs / length_penalty

        # Avoid any direction that would repeat unwanted ngrams
        self.block_ngram_repeats(curr_scores)

        # Pick up candidate token by curr_scores
        self._pick(curr_scores, out=(self.topk_scores, self.topk_ids))

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        self._batch_index = self.topk_ids // vocab_size
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)
        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [
                self.alive_seq.index_select(0, self.select_indices),
                self.topk_ids.view(_B * self.beam_size, 1),
            ],
            -1,
        )  # shape: (B * beam_size, step + 1)

        self.maybe_update_forbidden_tokens()

        if self.return_attention:
            current_attn = attn.index_select(1, self.select_indices)
            if step == 1:
                self.alive_attn = current_attn
                # update global state (step == 1)

            else:
                self.alive_attn = self.alive_attn.index_select(1, self.select_indices)
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)
                # update global state (step > 1)

        self.is_finished = self.topk_ids.eq(self.eos)  # shape: (B, beam_size)
        self.ensure_max_length()
