from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChemEncoderConfig:
    fingerprint_dim: int = 16387  # Fingerprint dimension


@dataclass
class GNNEncoderConfig:
    # Hyperparams that control the output dimensions
    atom_out_channels: int = 128
    bond_out_channels: Optional[int] = None
    mol_out_channels: int = 512

    # Hyperparams for mapping categorical features to dense embeddings
    atom_categorical_features_channels: Optional[int] = None

    # Which GNN from `pytorch_geometric` to use, and with what hyperparameters
    gnn_class: str = "PNA"
    gnn_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(
            hidden_channels=64,
            num_layers=3,
            dropout=0.0,
            norm="GraphNorm",
            # Use Jumping Knowledge i.e. concatenate node-level outputs across all GNN layers:
            jk="cat",
            # Use all aggregators and scalers supported by the `PNAConv` layer:
            aggregators=["sum", "mean", "min", "max", "var", "std"],
            scalers=["identity", "amplification", "attenuation", "linear", "inverse_linear"],
            # Degree histogram used for normalization computed over products in USPTO-50K train set:
            deg=[4, 191634, 516869, 307522, 23296, 0, 15],
        )
    )

    # Aggregation hyperparameters
    aggregation_num_heads: int = 8
    aggregation_dropout: float = 0.5

    # Featurizer to map from raw inputs to graphs
    featurizer_class: str = "DGLLifeMoleculeFeaturizer"
    featurizer_kwargs: dict[str, Any] = field(default_factory=lambda: {})
