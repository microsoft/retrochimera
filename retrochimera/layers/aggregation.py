import torch
from torch import nn
from torch_geometric.nn.glob import GlobalAttention


class MultiheadAttentionalAggregation(nn.Module):
    """Multi-head global attention-based aggregation layer.

    Each head is implemented as a `torch_geometric.nn.glob.GlobalAttention` layer."""

    def __init__(self, in_channels: int, num_heads: int, out_channels: int) -> None:
        super().__init__()

        if out_channels % num_heads != 0:
            raise ValueError(
                f"Number of output channels {out_channels} has to be divisible by the number of attention heads {num_heads}"
            )

        out_channels_per_head = out_channels // num_heads
        self.heads = nn.ModuleList(
            [
                GlobalAttention(
                    gate_nn=nn.Linear(in_channels, 1, bias=False),
                    nn=nn.Linear(in_channels, out_channels_per_head),
                )
                for _ in range(num_heads)
            ]
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x, batch) for head in self.heads], dim=-1)
