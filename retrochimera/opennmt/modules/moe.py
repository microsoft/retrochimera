"""
MIT License

Copyright (c) 2017-Present OpenNMT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Original Source: https://github.com/OpenNMT/OpenNMT-py/blob/v3.5.1/onmt/modules/moe.py
Modifications:
1. Removed the `self.parallel_gpu` attribute and corresponding logic from the `MoE` class, including calls to `torch.distributed.all_reduce`.
"""
import torch
import torch.nn as nn

from retrochimera.opennmt.modules.position_ffn import PositionwiseFeedForward


class MoE(nn.Module):
    def __init__(
        self,
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
        use_ckpting=[],
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                PositionwiseFeedForward(
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
                for i in range(num_experts)
            ]
        )
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        scores = self.gate(x)
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)
        flat_expert_indices = expert_indices.view(-1)

        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            if torch.any(flat_expert_indices == i):
                y[flat_expert_indices == i] = expert(x[flat_expert_indices == i].unsqueeze(0))
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)
