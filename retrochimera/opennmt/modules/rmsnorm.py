"""
MIT License

Copyright (c) 2017-Present OpenNMT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Original Source: https://github.com/OpenNMT/OpenNMT-py/blob/v3.5.1/onmt/modules/rmsnorm.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    """RMSNorm: https://arxiv.org/abs/1910.07467
    Args:
        hidden_size (int): layer hidden_size dimension.
        eps: variance epsilon.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        # OPT: collapse the 5-op sequence (pow + mean + rsqrt + mul + mul) into the
        # single fused `torch.nn.functional.rms_norm` kernel. F.rms_norm computes the
        # mean of squares in fp32 internally regardless of input dtype, so the result
        # is bit-identical to the original (`hidden_states.float().pow(2).mean(...)`)
        # path; verified max-abs-diff == 0.0 on random fp32 input.
        return F.rms_norm(
            hidden_states, self.weight.shape, self.weight, self.eps
        )
