"""
MIT License

Copyright (c) 2017-Present OpenNMT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Original Source: https://github.com/OpenNMT/OpenNMT-py/blob/v3.5.1/onmt/encoders/transformer.py
Modifications:
1. Removed the `parallel_gpu` argument from all four classes: `TransformerDecoderLayerBase`, `TransformerDecoderLayer`, `TransformerDecoderBase`, and `TransformerDecoder`, as Chimera typically uses Pytorch-Lightning for distributed training.
2. Removed the `embeddings` argument from the `TransformerDecoderBase` and `TransformerDecoder` constructors, as it is now built in the `retrochimera.models.smiles_transformer.SmilesTransformerModel` class.
3. Removed two unused classes: `TransformerLMDecoderLayer` and `TransformerLMDecoder`.
3. Refactored the `forward` method of the `TransformerDecoder` class to align with the `retrochimera.models.smiles_transformer.SmilesTransformerModel` class.
"""
from typing import Optional

import torch
import torch.nn as nn

from retrochimera.opennmt.modules.average_attention import AverageAttention
from retrochimera.opennmt.modules.moe import MoE
from retrochimera.opennmt.modules.multi_head_attention import MultiHeadedAttention
from retrochimera.opennmt.modules.position_ffn import ActivationFunction, PositionwiseFeedForward
from retrochimera.opennmt.modules.rmsnorm import RMSNorm


def sequence_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None,
) -> torch.Tensor:
    """Creates a boolean mask from sequence lengths."""
    assert max_len >= lengths.max(), f"{max_len} should be equal to {lengths.max()}"
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device) >= lengths.unsqueeze(1)


class TransformerDecoderLayerBase(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled_dot",
        max_relative_positions=0,
        relative_positions_buckets=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        num_kv=0,
        add_ffnbias=True,
        parallel_residual=False,
        shared_layer_norm=False,
        layer_norm="standard",
        norm_eps=1e-6,
        use_ckpting=[],
        sliding_window=0,
        rotary_interleave=True,
        rotary_theta=1e4,
        rotary_dim=0,
        num_experts=0,
        num_experts_per_tok=2,
    ):
        """
        Args:
            d_model (int): the dimension of keys/values/queries in MultiHeadedAttention,
            also the input size of the first-layer of PositionwiseFeedForward.
            heads (int): the number of heads for MultiHeadedAttention.
            d_ff (int): the second-layer of PositionwiseFeedForward.
            dropout (float): dropout in residual, self-attn(dot) and feed-forward.
            attention_dropout (float): dropout in context_attn  (and self-attn(avg)).
            self_attn_type (string): type of self-attention scaled-dot, flash-scaled-dot, average.
            max_relative_positions (int): max distance between inputs in relative positions representations.
            relative_positions_buckets (int): relative position bias see
            https://github.com/google-research/text-to-text-transfer-transformer.
            aan_useffn (bool): Turn on the FFN layer in the AAN decoder.
            full_context_alignment (bool): whether enable an extra full context decoder forward for alignment.
            alignment_heads (int): number of cross attention heads to use for alignment guiding.
            pos_ffn_activation_fn (ActivationFunction): activation function choice for PositionwiseFeedForward layer
            add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear.
            num_kv (int): number of heads for KV when different vs Q (multiquery).
            add_ffnbias (bool): whether to add bias to the FF nn.Linear.
            parallel_residual (bool): use parallel residual connections in each layer block, as used by the GPT-J and GPT-NeoX models.
            shared_layer_norm (bool): when using parallel residual, share the input and post attention layer norms.
            layer_norm (string): type of layer normalization standard/rms.
            norm_eps (float): layer norm epsilon.
            use_ckpting (List): layers for which we checkpoint for backward.
            sliding_window (int): width of the band mask and KV cache (cf Mistral Model).
            rotary_interleave (bool): interleave the head dimensions when rotary embeddings are applied.
            rotary_theta (float): rotary base theta.
            rotary_dim (int): in some cases the rotary dim is lower than head dim.
            num_experts (int): number of experts for MoE.
            num_experts_per_tok (int): number of experts choice per token.
        """
        super(TransformerDecoderLayerBase, self).__init__()

        if self_attn_type in ["scaled-dot", "scaled-dot-flash"]:
            self.self_attn = MultiHeadedAttention(
                heads,
                d_model,
                dropout=attention_dropout,
                max_relative_positions=max_relative_positions,
                relative_positions_buckets=relative_positions_buckets,
                rotary_interleave=rotary_interleave,
                rotary_theta=rotary_theta,
                rotary_dim=rotary_dim,
                attn_type="self",
                self_attn_type=self_attn_type,
                add_qkvbias=add_qkvbias,
                num_kv=num_kv,
                use_ckpting=use_ckpting,
            )
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(
                d_model, dropout=attention_dropout, aan_useffn=aan_useffn
            )

        if num_experts > 0:
            self.feed_forward = MoE(
                num_experts,
                num_experts_per_tok,
                d_model,
                d_ff,
                dropout,
                pos_ffn_activation_fn,
                add_ffnbias,
                parallel_residual,
                layer_norm,
                norm_eps,
                use_ckpting=use_ckpting,
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                d_model,
                d_ff,
                dropout,
                pos_ffn_activation_fn,
                add_ffnbias,
                parallel_residual,
                layer_norm,
                norm_eps,
                use_ckpting=use_ckpting,
            )
        self.parallel_residual = parallel_residual
        self.shared_layer_norm = shared_layer_norm
        if layer_norm == "standard":
            self.layer_norm_1 = nn.LayerNorm(d_model, eps=norm_eps)
            if parallel_residual and not shared_layer_norm:
                self.layer_norm_res = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm_1 = RMSNorm(d_model, eps=norm_eps)
            if parallel_residual and not shared_layer_norm:
                self.layer_norm_res = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads
        self.sliding_window = sliding_window
        self.self_attn_type = self_attn_type

    def forward(self, *args, **kwargs):
        """Extend `_forward` for (possibly) multiple decoder pass:
        Always a default (future masked) decoder forward pass,
        Possibly a second future aware decoder pass for joint learn
        full context alignement, :cite:`garg2019jointly`.

        Args:
            * All arguments of _forward, of which
            with_align (bool): needed to compute attn_align.
            return_attn (bool): to force MHA to return attns.

        Returns:
            layer_out (FloatTensor): shape (batch_size, T, hidden_dim).
            top_attn (FloatTensor): shape (batch_size, T, src_len).
            attn_align (FloatTensor or None): shape (batch_size, T, src_len).
        """
        with_align = kwargs.pop("with_align", False)
        layer_out, attns = self._forward(*args, **kwargs)
        top_attn = None if attns is None else attns[:, 0, :, :].contiguous()
        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads > 0:
                attns = attns[:, : self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return layer_out, top_attn, attn_align

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_dec_mask(self, tgt_pad_mask, future):
        tgt_len = tgt_pad_mask.size(-1)
        if not future:
            # Add triangular future_mask and pad_mask, result mask in (B, T, T).
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8,
            )
            future_mask = future_mask.tril_(0)
            if self.sliding_window > 0:
                future_mask = future_mask.triu_(-self.sliding_window)
            future_mask = future_mask.bool()
            future_mask = ~future_mask.view(1, tgt_len, tgt_len)
            # Patch for scaled dot product attention.
            patch_mask = ~torch.all(tgt_pad_mask + future_mask, dim=2, keepdim=True).expand_as(
                tgt_pad_mask + future_mask
            )
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            dec_mask = torch.logical_and(dec_mask, patch_mask)
        else:
            # Only mask padding, result mask in (B, 1, T).
            dec_mask = tgt_pad_mask
        return dec_mask

    def _forward_self_attn(self, norm_layer_in, dec_mask, step, return_attn=False):
        if self.self_attn_type in ["scaled-dot", "scaled-dot-flash"]:
            return self.self_attn(
                norm_layer_in,
                norm_layer_in,
                norm_layer_in,
                mask=dec_mask,
                sliding_window=self.sliding_window,
                step=step,
                return_attn=return_attn,
            )
        elif self.self_attn_type == "average":
            return self.self_attn(norm_layer_in, mask=dec_mask, step=step)
        else:
            raise ValueError(f"self attention {type(self.self_attn)} not supported")


class TransformerDecoderLayer(TransformerDecoderLayerBase):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    See https://tunz.kr/post/4 and :cite:`DeeperTransformer`.
    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        max_relative_positions=0,
        relative_positions_buckets=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        add_qkvbias=False,
        num_kv=0,
        add_ffnbias=True,
        parallel_residual=False,
        shared_layer_norm=False,
        layer_norm="standard",
        norm_eps=1e-6,
        use_ckpting=[],
        sliding_window=0,
        rotary_interleave=True,
        rotary_theta=1e4,
        rotary_dim=0,
        num_experts=0,
        num_experts_per_tok=2,
    ):
        """
        Args:
            See TransformerDecoderLayerBase
        """
        super(TransformerDecoderLayer, self).__init__(
            d_model,
            heads,
            d_ff,
            dropout,
            attention_dropout,
            self_attn_type,
            max_relative_positions,
            relative_positions_buckets,
            aan_useffn,
            full_context_alignment,
            alignment_heads,
            pos_ffn_activation_fn=pos_ffn_activation_fn,
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            add_ffnbias=add_ffnbias,
            parallel_residual=parallel_residual,
            shared_layer_norm=shared_layer_norm,
            layer_norm=layer_norm,
            norm_eps=norm_eps,
            use_ckpting=use_ckpting,
            sliding_window=sliding_window,
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            rotary_dim=rotary_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )
        self.context_attn = MultiHeadedAttention(
            heads,
            d_model,
            dropout=attention_dropout,
            attn_type="context",
            self_attn_type=self.self_attn_type,
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            use_ckpting=use_ckpting,
        )
        if layer_norm == "standard":
            self.layer_norm_2 = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm_2 = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")

    def update_dropout(self, dropout, attention_dropout):
        super(TransformerDecoderLayer, self).update_dropout(dropout, attention_dropout)
        self.context_attn.update_dropout(attention_dropout)

    def _forward(
        self,
        layer_in,
        enc_out,
        src_pad_mask,
        tgt_pad_mask,
        step=None,
        future=False,
        return_attn=False,
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            layer_in (FloatTensor): ``(batch_size, T, hidden_dim)``
            enc_out (FloatTensor): ``(batch_size, src_len, hidden_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.
            return_attn (bool) : if set True requires attns output

        Returns:
            (FloatTensor, FloatTensor):

            * layer_out ``(batch_size, T, hidden_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None
        src_pad_mask = src_pad_mask.unsqueeze(1)  # [B,1,1,slen]

        if layer_in.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)
            dec_mask = dec_mask.unsqueeze(1)
            dec_mask = dec_mask.expand(-1, -1, dec_mask.size(3), -1)
            src_pad_mask = src_pad_mask.expand(-1, -1, dec_mask.size(3), -1)
            # mask now are (batch x 1 x tlen x s or t len)
            # 1 = heads to be expanded in MHA

        norm_layer_in = self.layer_norm_1(layer_in)

        self_attn, _ = self._forward_self_attn(
            norm_layer_in, dec_mask, step, return_attn=return_attn
        )
        if self.dropout_p > 0:
            self_attn = self.dropout(self_attn)
        if self.parallel_residual:
            ctx_attn, attns = self.context_attn(
                enc_out,
                enc_out,
                norm_layer_in,
                mask=src_pad_mask,
                return_attn=return_attn,
            )
            # feed_forward applies residual, so we remove and apply residual with un-normed
            layer_out = (
                self.feed_forward(norm_layer_in) - norm_layer_in + layer_in + self_attn + ctx_attn
            )
        else:
            query = self_attn + layer_in
            norm_query = self.layer_norm_2(query)
            ctx_attn, attns = self.context_attn(
                enc_out, enc_out, norm_query, mask=src_pad_mask, return_attn=return_attn
            )
            if self.dropout_p > 0:
                ctx_attn = self.dropout(ctx_attn)
            layer_out = self.feed_forward(ctx_attn + query)

        return layer_out, attns


class TransformerDecoderBase(nn.Module):
    def __init__(self, d_model, copy_attn, alignment_layer, layer_norm, norm_eps):
        super(TransformerDecoderBase, self).__init__()

        # Decoder State
        self.state = {}

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        if layer_norm == "standard":
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")

        self.alignment_layer = alignment_layer

    def init_state(self, src, enc_out, enc_final_hs):
        """Initialize decoder state."""
        self.state["src"] = src  # torch.Tensor: (padded_src_len, batch_size, 1)

    def map_state(self, fn, only_map_src=False):
        if self.state["src"] is not None:
            self.state["src"] = fn(self.state["src"], 1)

        if not only_map_src:
            for layer in self.transformer_layers:
                if hasattr(layer, "context_attn"):
                    if layer.context_attn.layer_cache[1]["keys"].numel() != 0:
                        x = fn(layer.context_attn.layer_cache[1]["keys"], 0)
                        y = fn(layer.context_attn.layer_cache[1]["values"], 0)
                        layer.context_attn.layer_cache = True, {"keys": x, "values": y}
                if isinstance(layer.self_attn, AverageAttention):
                    if layer.self_attn.layer_cache[1]["prev_g"].numel() != 0:
                        x = fn(layer.self_attn.layer_cache[1]["prev_g"], 0)
                        layer.self_attn.layer_cache = True, {"prev_g": x}
                else:
                    if layer.self_attn.layer_cache[1]["keys"].numel() != 0:
                        x = fn(layer.self_attn.layer_cache[1]["keys"], 0)
                        y = fn(layer.self_attn.layer_cache[1]["values"], 0)
                        if layer.self_attn.layer_cache[1].get("key_pad_mask", None) is not None:
                            z = fn(layer.self_attn.layer_cache[1]["key_pad_mask"], 0)
                        else:
                            z = None
                        layer.self_attn.layer_cache = True, {
                            "keys": x,
                            "values": y,
                            "key_pad_mask": z,
                        }

    def detach_state(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)


class TransformerDecoder(TransformerDecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        num_layers (int): number of decoder layers.
        d_model (int): size of the model.
        heads (int): number of heads.
        d_ff (int): size of the inner FF layer.
        copy_attn (bool): if using a separate copy attention.
        self_attn_type (str): type of self-attention scaled-dot, scaled-dot-flash, average.
        dropout (float): dropout in residual, self-attn(dot) and feed-forward.
        attention_dropout (float): dropout in context_attn (and self-attn(avg)).
        max_relative_positions (int): enable relative position encoding: -1 to enable Rotary Embeddings,
        > 0 (e.g., 16, 32) to use max distance between inputs in relative positions representations
        relative_positions_buckets (int): number of buckets when using relative position bias.
        aan_useffn (bool): turn on the FFN layer in the AAN decoder.
        full_context_alignment (bool): whether enable an extra full context decoder forward for alignment.
        alignment_layer (int): N° Layer to supervise with for alignment guiding.
        alignment_heads (int): N. of cross attention heads to use for alignment guiding.
        pos_ffn_activation_fn (str): activation function choice for PositionwiseFeedForward layer.
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear.
        num_kv (int): number of heads for KV when different vs Q (multiquery).
        add_ffnbias (bool): whether to add bias to the FF nn.Linear.
        parallel_residual (bool): Use parallel residual connections in each layer block, as used by the GPT-J and GPT-NeoX models.
        shared_layer_norm (bool): When using parallel residual, share the input and post attention layer norms.
        layer_norm (string): type of layer normalization standard/rms.
        norm_eps (float): layer norm epsilon.
        use_ckpting (List): layers for which we checkpoint for backward.
        sliding_window (int): Width of the band mask and KV cache (cf Mistral Model).
        rotary_interleave (bool): Interleave the head dimensions when rotary embeddings are applied.
        rotary_theta (int): rotary base theta.
        rotary_dim (int): in some cases the rotary dim is lower than head dim.
        num_experts (int): Number of experts for MoE.
        num_experts_per_tok (int): Number of experts choice per token.
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 256,
        heads: int = 8,
        d_ff: int = 2048,
        copy_attn: bool = False,
        self_attn_type: str = "scaled-dot",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_relative_positions: int = 0,
        relative_positions_buckets: int = 0,
        aan_useffn: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: int = 0,
        alignment_heads: int = 0,
        pos_ffn_activation_fn: str = ActivationFunction.relu,
        add_qkvbias: bool = False,
        num_kv: int = 0,
        add_ffnbias: bool = True,
        parallel_residual: bool = False,
        shared_layer_norm: bool = False,
        layer_norm: str = "standard",
        norm_eps: float = 1e-6,
        use_ckpting: list = [],
        sliding_window: int = 0,
        rotary_interleave: bool = True,
        rotary_theta: float = 1e4,
        rotary_dim: int = 0,
        num_experts: int = 0,
        num_experts_per_tok: int = 2,
    ):
        super(TransformerDecoder, self).__init__(
            d_model, copy_attn, alignment_layer, layer_norm, norm_eps
        )
        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                    relative_positions_buckets=relative_positions_buckets,
                    aan_useffn=aan_useffn,
                    full_context_alignment=full_context_alignment,
                    alignment_heads=alignment_heads,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    add_qkvbias=add_qkvbias,
                    num_kv=num_kv,
                    add_ffnbias=add_ffnbias,
                    parallel_residual=parallel_residual,
                    shared_layer_norm=shared_layer_norm,
                    layer_norm=layer_norm,
                    norm_eps=norm_eps,
                    use_ckpting=use_ckpting,
                    sliding_window=sliding_window,
                    rotary_interleave=rotary_interleave,
                    rotary_theta=rotary_theta,
                    rotary_dim=rotary_dim,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                )
                for i in range(num_layers)
            ]
        )

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        enc_out: torch.Tensor,
        enc_key_padding_mask: torch.Tensor,
        step: Optional[int] = None,
        **kwargs,
    ):
        """
        During training, step is always None, when decoding, step increases.

        Args:
        tgt (torch.Tensor): shape (batch_size, padded_tgt_len/1, hidden_dim).
        tgt_key_padding_mask (torch.Tensor, bool: 0/1): shape (batch_size, padded_tgt_len/1).
        enc_out (torch.Tensor): encoder output (batch_size, padded_src_len, hidden_dim).
        enc_key_padding_mask (torch.Tensor, bool: 0/1): encoder padding mask (batch_size, padded_src_len).

        Returns:
        dec_out (torch.Tensor): decoder output (batch_size, padded_tgt_len/1, hidden_dim).
        attns (dict): attention weights.

        Note: padded_tgt_len is 1 when using kv cache during inference.

        """

        assert enc_out is not None
        if step == 0:
            self._init_cache(enc_out)
        elif step is None:
            for layer in self.transformer_layers:
                if isinstance(layer.self_attn, AverageAttention):
                    layer.self_attn.layer_cache = False, {"prev_g": torch.tensor([])}
                else:
                    layer.self_attn.layer_cache = (
                        False,
                        {"keys": torch.tensor([]), "values": torch.tensor([])},
                    )
                layer.context_attn.layer_cache = (
                    False,
                    {"keys": torch.tensor([]), "values": torch.tensor([])},
                )

        dec_out = tgt
        src_len = enc_key_padding_mask.eq(0).sum(dim=1).long()  # shape: (batch_size,)
        src_max_len = self.state["src"].shape[
            0
        ]  # self.state["src"] shape: (padded_src_len, batch_size, 1)

        src_pad_mask = sequence_mask(src_len, src_max_len).unsqueeze(
            1
        )  # shape: (batch_size, 1, padded_src_len)
        tgt_pad_mask = tgt_key_padding_mask.unsqueeze(1)  # shape: (batch_size, 1, padded_tgt_len/1)

        with_align = kwargs.pop("with_align", False)
        return_attn = with_align or self._copy or kwargs.pop("return_attn", False)
        attn_aligns = []

        for layer in self.transformer_layers:
            dec_out, attn, attn_align = layer(
                dec_out,
                enc_out,
                src_pad_mask,
                tgt_pad_mask,
                step=step,
                with_align=with_align,
                return_attn=return_attn,
            )
            if attn_align is not None:
                attn_aligns.append(attn_align)

        dec_out = self.layer_norm(dec_out)  # shape: (batch_size, padded_tgt_len/1, hidden_dim)

        attns = {
            "std": attn
        }  # shape of attn: (batch_size, heads, padded_tgt_len/1, padded_src_len)
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        return dec_out, attns

    def _init_cache(self, enc_out: torch.Tensor):
        """Initialize the cache for the transformer layers.

        Args:
        enc_out (torch.Tensor): encoder output (batch_size, padded_src_len, hidden_dim).

        """
        batch_size = enc_out.size(0)
        depth = enc_out.size(-1)

        for layer in self.transformer_layers:
            # first value set to True triggered by the beginning of decoding
            # layer_cache becomes active in the MultiHeadedAttention fwd
            layer.context_attn.layer_cache = (
                True,
                {
                    "keys": torch.tensor([], device=enc_out.device),
                    "values": torch.tensor([], device=enc_out.device),
                },
            )
            if isinstance(layer.self_attn, AverageAttention):
                layer.self_attn.layer_cache = True, {
                    "prev_g": torch.zeros((batch_size, 1, depth), device=enc_out.device).to(
                        enc_out.dtype
                    )
                }
            else:
                layer.self_attn.layer_cache = (
                    True,
                    {
                        "keys": torch.tensor([], device=enc_out.device),
                        "values": torch.tensor([], device=enc_out.device),
                    },
                )
                if hasattr(layer.self_attn, "rope"):
                    layer.self_attn.rope = layer.self_attn.rope.to(enc_out.device)
                    layer.self_attn.cos = layer.self_attn.cos.to(enc_out.device)
                    layer.self_attn.sin = layer.self_attn.sin.to(enc_out.device)
