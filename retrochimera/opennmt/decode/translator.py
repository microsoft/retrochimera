#!/usr/bin/env python
"""
MIT License

Copyright (c) 2017-Present OpenNMT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Original Source: https://github.com/OpenNMT/OpenNMT-py/blob/v3.5.1/onmt/translate/translator.py
Modifications:
1. Simplified the structure by removing the `build_translator` function, and merged the `Inference` class into the `Translator` class.
2. Refactored the `_run_encoder`, `_decode_and_generate`, and `_translate_batch_with_strategy` methods to align with the `retrochimera.models.smiles_transformer.SmilesTransformerModel` class.
3. Introduced the `customised_beam_search` attribute and corresponding logic to the `Translator` class, enabling optimized beam search for retrosynthesis prediction.
"""
from typing import Any, Optional

import torch
from torch.nn.functional import log_softmax

from retrochimera.data.smiles_tokenizer import Tokenizer
from retrochimera.models.smiles_transformer import SmilesTransformerModel
from retrochimera.opennmt.decode.beam_search import BeamSearch
from retrochimera.opennmt.decode.decoder_strategy import DecodeStrategy, set_random_seed


class Translator(object):
    """Translate a batch of sentences with a saved model and tokenizer.

    Args:
        model: Encoder-decoder model to use.
        tokenizer: Tokenizer.
        gpu: GPU device. Set to negative for no GPU.
        n_best: How many beams to wait for.
        min_length: See class .decode_strategy.DecodeStrategy.
        max_length: See class .decode_strategy.DecodeStrategy.
        ratio: See class .decode_strategy.DecodeStrategy.
        beam_size: Number of beams.
        block_ngram_repeat: See class .decode_strategy.DecodeStrategy.
        ignore_when_blocking: See class .decode_strategy.DecodeStrategy.
        replace_unk: Replace unknown token.
        ban_unk_token: Ban unknown token.
        tgt_prefix: Force the predictions begin with provided -tgt.
        data_type: Source data type.
        verbose: Print/log every translation.
        seed: Random seed.
    """

    def __init__(
        self,
        model: SmilesTransformerModel,
        tokenizer: Tokenizer,
        gpu: int = 0,
        n_best: int = 10,
        min_length: int = 0,
        max_length: int = 512,
        ratio: float = 0.0,
        beam_size: int = 10,
        block_ngram_repeat: int = 0,
        ignore_when_blocking: set = set(),
        replace_unk: bool = False,
        ban_unk_token: bool = False,
        tgt_prefix: bool = False,
        data_type: str = "text",
        verbose: bool = False,
        seed: int = 1,
        customised_beam_search: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self._tgt_eos_idx = self.tokenizer.end_token_id
        self._tgt_pad_idx = self.tokenizer.pad_token_id
        self._tgt_bos_idx = self.tokenizer.begin_token_id
        self._tgt_unk_idx = self.tokenizer.unk_token_id

        self._tgt_vocab = tokenizer.vocab_dict  # token -> id
        self._tgt_vocab_len = len(self._tgt_vocab)

        self.gpu = gpu
        self.use_cuda = gpu > -1
        self.device = torch.device("cuda", self.gpu) if self.use_cuda else torch.device("cpu")

        self.n_best = n_best
        self.min_length = min_length
        self.max_length = max_length

        self.ratio = ratio
        self.beam_size = beam_size

        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {self._tgt_vocab[t] for t in self.ignore_when_blocking}
        self.replace_unk = replace_unk
        if self.replace_unk:
            raise ValueError("replace_unk requires an attentional decoder.")
        self.ban_unk_token = ban_unk_token

        self.tgt_prefix = tgt_prefix
        self.data_type = data_type
        self.verbose = verbose
        self.customised_beam_search = customised_beam_search

        set_random_seed(seed, self.use_cuda)

    def translate_batch(self, batch: dict[str, Any], attn_debug: bool = False) -> dict[str, Any]:
        """Translate a batch of sentences.

        Args:
            batch: dict, keys:
            - src: Tuple(src, src_lengths)
                - src (torch.Tensor): shape of src: (padded_src_len, batch_size, 1)
                - src_lengths (torch.Tensor): shape of src_lengths: (batch_size,)
            - batch_size: int
        """
        with torch.no_grad():
            # TODO: support these blacklisted features
            decode_strategy = BeamSearch(
                pad=self._tgt_pad_idx,
                bos=self._tgt_bos_idx,
                eos=self._tgt_eos_idx,
                unk=self._tgt_unk_idx,
                batch_size=batch["batch_size"],
                beam_size=self.beam_size,
                n_best=self.n_best,
                min_length=self.min_length,
                max_length=self.max_length,
                block_ngram_repeat=self.block_ngram_repeat,
                exclusion_tokens=self._exclusion_idxs,
                ratio=self.ratio,
                ban_unk_token=self.ban_unk_token,
                return_attention=attn_debug or self.replace_unk,
                customised_beam_search=self.customised_beam_search,
            )

            return self._translate_batch_with_strategy(batch, decode_strategy)

    def _translate_batch_with_strategy(
        self, batch: dict[str, Any], decode_strategy: DecodeStrategy
    ) -> dict[str, Any]:
        """Translate a batch of sentences step by step using cache.

        Args:
            batch (dict), keys:
            - src: Tuple(src, src_lengths)
                - src (torch.Tensor): shape of src: (padded_src_len, batch_size, 1)
                - src_lengths (torch.Tensor): shape of src_lengths: (batch_size,)
            - batch_size: int

            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = False
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch["batch_size"]

        # (1) Run the encoder on the src.
        src, encoder_embedding, memory_bank, src_lengths = self._run_encoder(batch)
        # shape of src: (padded_src_len, batch_size, 1)
        # shape of encoder_embedding: (padded_src_len, batch_size, hidden_dim)
        # shape of memory_bank: (padded_src_len, batch_size, hidden_dim)
        # shape of src_lengths: (batch_size,)

        # self.model.decoder.init_state(src, memory_bank, encoder_embedding)
        self.model.decoder.init_state(src, None, None)

        gold_score = [0] * batch_size

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = batch["src_map"] if use_src_map else None  # None

        target_prefix = batch["tgt"] if self.tgt_prefix else None  # None

        (fn_map_state, memory_bank, memory_lengths, src_map,) = decode_strategy.initialize(
            memory_bank, src_lengths, src_map, target_prefix=target_prefix
        )
        # [returns]
        # shape of memory_bank: (padded_src_len, batch_size * beam_size, hidden_dim)
        # shape of memory_lengths: (batch_size * beam_size,)

        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state, only_map_src=True)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(
                1, -1, 1
            )  # type: ignore[override]
            # shape of decoder_input: (1, self.batch_size * self.parallel_paths, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=decode_strategy.batch_offset,  # type: ignore
            )  # (batch_size * beam_size, vocab)

            if self.customised_beam_search:
                # modify the log_probs for finished sentences.
                _, vocab_size = tuple(log_probs.shape)
                bad_token_log_prob = -1e5

                complete_seq_log_prob = (torch.ones((1, vocab_size)) * bad_token_log_prob).to(
                    log_probs.device
                )  # shape: (1, vocab_size)
                complete_seq_log_prob[:, self._tgt_eos_idx] = 0.0  # shape: (1, vocab_size)

                # Use this vector in the output for sequences which are complete.
                is_end_token = (
                    decoder_input.squeeze() == self._tgt_eos_idx
                )  # shape: (batch_size * beam_size,)
                log_prob_mask = is_end_token.unsqueeze(1)  # shape: (batch_size * beam_size, 1)
                log_probs = (log_prob_mask * complete_seq_log_prob) + (
                    ~log_prob_mask * log_probs
                )  # shape: (batch_size * beam_size, vocab_size)

                assert log_probs.dim() == 2, f"Expected 2D tensor, got {log_probs.dim()}"
                assert log_probs.size(0) == decoder_input.size(
                    1
                ), f"Expected {batch_size * parallel_paths}, got {log_probs.size(0)}"
                assert (
                    log_probs.size(1) == vocab_size
                ), f"Expected {vocab_size}, got {log_probs.size(1)}"

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices  # type: ignore

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices) for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        return self.report_results(
            gold_score,
            batch,
            batch_size,
            decode_strategy,
        )

    def _run_encoder(self, batch: dict[str, Any]) -> tuple:
        src, src_lengths = (
            batch["src"] if isinstance(batch["src"], tuple) else (batch["src"], None)
        )  # src: (padded_src_len, batch_size, 1), src_lengths: (batch_size,)

        encoder_embedding = self.model.construct_input(
            src.transpose(0, 1).contiguous(), is_encoder=True
        )  # shape: (batch_size, padded_src_len, hidden_dim)
        encoder_padding_mask = (
            src.transpose(0, 1).contiguous().squeeze(2).eq(self._tgt_pad_idx)
        )  # shape: (batch_size, padded_src_len)

        memory_bank = (
            self.model.encoder.forward(
                src=encoder_embedding, src_key_padding_mask=encoder_padding_mask
            )
            .transpose(0, 1)
            .contiguous()
        )  # shape: (padded_src_len, batch_size, hidden_dim)

        if src_lengths is None:
            assert not isinstance(
                memory_bank, tuple
            ), "Ensemble decoding only supported for text data"
            src_lengths = (
                torch.Tensor(batch["batch_size"])
                .type_as(memory_bank)
                .long()
                .fill_(memory_bank.size(0))
            )  # shape: (batch_size,)
        # shape of src: (padded_src_len, batch_size, 1)
        # shape of encoder_embedding.transpose(0,1).contiguous(): (padded_src_len, batch_size, hidden_dim)
        # shape of memory_bank: (padded_src_len, batch_size, hidden_dim)
        # shape of src_lengths: (batch_size,)
        return src, encoder_embedding.transpose(0, 1).contiguous(), memory_bank, src_lengths

    def _decode_and_generate(
        self,
        decoder_in: torch.Tensor,
        memory_bank: torch.Tensor,
        batch: dict[str, Any],
        memory_lengths: torch.Tensor,
        src_map=None,
        step: Optional[int] = None,
        batch_offset: torch.LongTensor = None,
    ):
        """Decode and generate one step.

        Args:
            decoder_in (torch.Tensor): shape: (1, batch_size * beam_size, 1), due to kv_cache mechanism
            memory_bank (torch.Tensor): shape: (padded_src_len, batch_size * beam_size, hidden_dim)
            batch (dict), keys:
            - src: Tuple(src, src_lengths)
                - src (torch.Tensor): shape of src: (padded_src_len, batch_size, 1)
                - src_lengths (torch.Tensor): shape of src_lengths: (batch_size,)
            - batch_size: int
            memory_lengths (torch.Tensor): shape: (batch_size * beam_size,)
            src_map (torch.Tensor): None
            step (int): current step
            batch_offset (int): batch offset

        Returns:
            log_probs (torch.Tensor): shape: (batch_size * beam_size, vocab)
            attn (torch.Tensor): shape: (batch_size * beam_size, tgt_len, src_len)
        """
        # dec_out, dec_attn = self.model.decoder(
        #     decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        # )

        # # Generator forward.
        # if "std" in dec_attn:
        #     attn = dec_attn["std"]
        # else:
        #     attn = None
        # log_probs = self.model.generator(dec_out.squeeze(0))

        # return log_probs, attn

        # Decoder forward, takes [batch, tgt_len, nfeats] as input
        # and [batch, src_len, hidden] as enc_out
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch

        decoder_embedding = self.model.construct_input(
            decoder_in.transpose(0, 1).contiguous(), step=step, is_encoder=False
        )  # shape: (batch_size * beam_size, 1, hidden_dim)
        assert decoder_embedding.dim() == 3, f"Expected 3D tensor, got {decoder_embedding.dim()}"

        decoder_padding_mask = decoder_in.eq(self._tgt_pad_idx).squeeze(
            2
        )  # shape: (1, batch_size * beam_size)
        memory_padding_mask = torch.arange(
            0, max(memory_lengths), device=memory_bank.device
        ) >= memory_lengths.unsqueeze(
            1
        )  # shape: (batch_size * beam_size, padded_src_len)

        dec_out, dec_attn = self.model.decoder.forward(
            tgt=decoder_embedding,  # shape: (batch_size * beam_size, 1, hidden_dim)
            tgt_key_padding_mask=decoder_padding_mask.transpose(
                0, 1
            ).contiguous(),  # shape: (batch_size * beam_size, 1)
            enc_out=memory_bank.transpose(
                0, 1
            ).contiguous(),  # shape: (batch_size * beam_size, padded_src_len, hidden_dim)
            enc_key_padding_mask=memory_padding_mask,  # shape: (batch_size * beam_size, padded_src_len)
            step=step,
        )  # shape: (batch_size, tgt_len-1, hidden_dim)

        # Generator forward.
        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        # scores = self.model.generator(dec_out.squeeze(1))
        scores = self.model.token_fc(
            dec_out.squeeze(1)
        )  # shape: (batch_size * beam_size, vocab_size)
        log_probs = log_softmax(scores, dim=-1)  # shape: (batch_size * beam_size, vocab_size)
        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [batch_size, tgt_len, vocab ] when full sentence

        return log_probs, attn

    def report_results(
        self,
        gold_score,
        batch,
        batch_size,
        decode_strategy,
    ):
        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": gold_score,
        }

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["alignment"] = [[] for _ in range(batch_size)]
        return results
