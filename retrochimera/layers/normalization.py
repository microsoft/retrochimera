import torch
from torch import nn


class NormWrapper(nn.Module):
    """Wrapper fixing the behaviour of some of the normalization layers in PyTorch Geometric.

    The GNN models in PyTorch Geometric process batched graphs as if it was a single large graph,
    without knowing the batch assignment. However, some normalization layers (e.g. `GraphNorm`) need
    batch assignment as they normalize *within* each input graph. If such a layer is plugged into a
    model from `torch_geometric.nn.models`, it is silently used incorrectly, normalizing *across*
    graphs, which can have terrible consequences at inference time when batch size is different from
    training (see https://github.com/pyg-team/pytorch_geometric/discussions/7856 for more details).
    This class hacks PyG-s design and provides the batch assignment "through the back door".
    """

    # Classes in `torch_geometric.nn.norm` that need to be wrapped.
    NORMS_REQUIRING_BATCH_ASSIGNMENT = [
        "GraphNorm",
        "GraphSizeNorm",
        "InstanceNorm",
        "LayerNorm",
        "MeanSubtractionNorm",
        "PairNorm",
    ]

    def __init__(self, norm: nn.Module) -> None:
        super().__init__()

        self.norm = norm
        self.batch = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.batch is not None
        return self.norm(x, batch=self.batch)
