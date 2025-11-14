"""
MIT License

Copyright (c) 2017-Present OpenNMT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Original Source: https://github.com/OpenNMT/OpenNMT-py/blob/v3.5.1/onmt/encoders/transformer.py
Modifications:
1. Removed the `parallel_gpu` argument from the `TransformerEncoderLayer` and `TransformerEncoder` constructors, as Chimera typically uses Pytorch-Lightning for distributed training.
2. Removed the `embeddings` argument from the `TransformerEncoder` constructor, as it is now built in the `retrochimera.models.smiles_transformer.SmilesTransformerModel` class.
3. Refactored the `forward` method of the `TransformerEncoder` class to align with the `retrochimera.models.smiles_transformer.SmilesTransformerModel` class.
"""
import torch
import torch.nn as nn

from retrochimera.opennmt.modules.multi_head_attention import MultiHeadedAttention
from retrochimera.opennmt.modules.position_ffn import ActivationFunction, PositionwiseFeedForward
from retrochimera.opennmt.modules.rmsnorm import RMSNorm


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in MultiHeadedAttention, also the input size of
        the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0) used in any other place besides attention layer.
        attention_dropout (float): dropout probability(0-1.0) used in attention layer.
        max_relative_positions (int): enable relative position encoding: -1 to enable Rotary Embeddings,
        > 0 (e.g., 16, 32) to use maximum distance between inputs in relative positions representations
        relative_positions_buckets (int): number of buckets when using relative position bias.
        pos_ffn_activation_fn (str): activation function choice for PositionwiseFeedForward layer.
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear.
        num_kv (int): number of heads for KV when different vs Q (multiquery).
        add_ffnbias (bool): whether to add bias to the FF nn.Linear.
        parallel_residual (bool): Use parallel residual connections in each layer block, as used by the GPT-J and GPT-NeoX models.
        layer_norm (string): type of layer normalization standard/rms.
        norm_eps (float): layer norm epsilon.
        use_ckpting (List): layers for which we checkpoint for backward.
        rotary_interleave (bool): Interleave the head dimensions when rotary embeddings are applied.
        rotary_theta (float): rotary base theta.
        rotary_dim (int): rotary dim when different to dim per head.
    """

    def __init__(
        self,
        d_model: int = 256,
        heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_relative_positions: int = 0,
        relative_positions_buckets: int = 0,
        pos_ffn_activation_fn: str = ActivationFunction.relu,
        add_qkvbias: bool = False,
        num_kv: int = 0,
        add_ffnbias: bool = True,
        parallel_residual: bool = False,
        layer_norm: str = "standard",
        norm_eps: float = 1e-6,
        use_ckpting: list = [],
        rotary_interleave: bool = True,
        rotary_theta: float = 1e4,
        rotary_dim: int = 0,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads,
            d_model,
            dropout=attention_dropout,
            is_decoder=False,
            max_relative_positions=max_relative_positions,
            relative_positions_buckets=relative_positions_buckets,
            rotary_interleave=rotary_interleave,
            rotary_theta=rotary_theta,
            rotary_dim=rotary_dim,
            attn_type="self",
            add_qkvbias=add_qkvbias,
            num_kv=num_kv,
            use_ckpting=use_ckpting,
        )
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
        if layer_norm == "standard":
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        layer_in: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            layer_in (FloatTensor): shape (batch_size, src_len, hidden_dim).
            mask (LongTensor): shape (batch_size, 1, src_len).

        Returns:
            layer_out (FloatTensor): shape (batch_size, src_len, hidden_dim).
        """
        norm_layer_in = self.layer_norm(layer_in)
        context, _ = self.self_attn(norm_layer_in, norm_layer_in, norm_layer_in, mask=mask)
        if self.dropout_p > 0:
            context = self.dropout(context)
        if self.parallel_residual:
            # feed_forward applies residual, so we remove and apply residual with un-normed
            layer_out = self.feed_forward(norm_layer_in) - norm_layer_in + layer_in + context
        else:
            layer_out = context + layer_in
            layer_out = self.feed_forward(layer_out)

        return layer_out

    def update_dropout(
        self,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(nn.Module):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        num_layers (int): number of encoder layers.
        d_model (int): size of the model.
        heads (int): number of heads.
        d_ff (int): size of the inner FF layer.
        dropout (float): dropout parameters.
        pos_ffn_activation_fn (str): activation function for PositionwiseFeedForward layer.
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 256,
        heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_relative_positions: int = 0,
        relative_positions_buckets: int = 0,
        pos_ffn_activation_fn: str = ActivationFunction.relu,
        add_qkvbias: bool = False,
        num_kv: int = 0,
        add_ffnbias: bool = True,
        parallel_residual: bool = False,
        layer_norm: str = "standard",
        norm_eps: float = 1e-6,
        use_ckpting: list = [],
        rotary_interleave: bool = True,
        rotary_theta: float = 1e4,
        rotary_dim: int = 0,
    ):
        super(TransformerEncoder, self).__init__()

        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    max_relative_positions=max_relative_positions,
                    relative_positions_buckets=relative_positions_buckets,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    add_qkvbias=add_qkvbias,
                    num_kv=num_kv,
                    add_ffnbias=add_ffnbias,
                    parallel_residual=parallel_residual,
                    layer_norm=layer_norm,
                    norm_eps=norm_eps,
                    use_ckpting=use_ckpting,
                    rotary_interleave=rotary_interleave,
                    rotary_theta=rotary_theta,
                    rotary_dim=rotary_dim,
                )
                for i in range(num_layers)
            ]
        )
        if layer_norm == "standard":
            self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        elif layer_norm == "rms":
            self.layer_norm = RMSNorm(d_model, eps=norm_eps)
        else:
            raise ValueError(f"{layer_norm} layer norm type is not supported")

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor):
        """
        Args:
            src (torch.Tensor) - the sequence to the encoder (required). (batch_size, padded_src_len, hidden_dim).
            src_key_padding_mask (torch.Tensor) - the mask for the src keys per batch (required). (batch_size, padded_src_len).

        Returns:
            enc_out (torch.Tensor) - the output of the encoder. (batch_size, padded_src_len, hidden_dim).
        """

        enc_out = src  # shape: (batch_size, padded_src_len, hidden_dim)

        # 1. Extend src_key_padding_mask to (batch_size, 1, padded_src_len, padded_src_len) to be compatible with MHA.
        # 1 in the above shape will be expanded to number of heads in MHA.
        assert src_key_padding_mask.dtype == torch.bool
        src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(
            1
        )  # shape: (batch_size, 1, 1, padded_src_len)
        src_key_padding_mask = src_key_padding_mask.expand(
            -1, -1, src_key_padding_mask.size(3), -1
        )  # shape: (batch_size, 1, padded_src_len, padded_src_len)

        # 2. Loop through the encoder layers.
        for layer in self.transformer:
            enc_out = layer(
                enc_out, src_key_padding_mask
            )  # shape: (batch_size, padded_src_len, hidden_dim)

        # 3. Layer normalization before LM head.
        enc_out = self.layer_norm(enc_out)  # shape: (batch_size, padded_src_len, hidden_dim)

        return enc_out

    def update_dropout(
        self,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
