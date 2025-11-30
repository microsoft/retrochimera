from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Batch, Data

import retrochimera.encoders
from retrochimera.encoders.base import Encoder, EncoderOutput, InputType
from retrochimera.encoders.featurizers import RawGraph, RawRewriteGraph
from retrochimera.layers.aggregation import MultiheadAttentionalAggregation
from retrochimera.layers.normalization import NormWrapper
from retrochimera.utils.misc import lookup_by_name


class GNNEncoder(Encoder[InputType, RawGraph, Batch]):
    """Implements an encoder backed by a GNN.

    This encoder depends on a featurizer that maps chemical objects to annotated graphs, such as
    those available in `dgllife.utils`. It can provide atom-, bond-, and molecule-level outputs:
    - atom-level outputs are directly produced by the GNN backbone
    - bond-level outputs are computed from atom-level outputs for endpoints and raw bond features
    - molecule-level outputs are computed by aggregating all atom-level outputs.
    """

    def __init__(
        self,
        atom_out_channels: int,
        bond_out_channels: Optional[int],
        mol_out_channels: int,
        gnn_class: str,
        gnn_kwargs: dict[str, Any],
        aggregation_num_heads: int,
        aggregation_dropout: float,
        featurizer_class: str,
        featurizer_kwargs: Optional[dict[str, Any]] = None,
        atom_categorical_features_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Handle the default value of `None`, which is here for backwards compatibility.
        featurizer_kwargs = featurizer_kwargs or {}

        self.featurizer = lookup_by_name(
            retrochimera.encoders.featurizers, featurizer_class
        ).get_default(**featurizer_kwargs)

        self._atom_out_channels = atom_out_channels
        self._bond_out_channels = bond_out_channels
        self._mol_out_channels = mol_out_channels

        num_total_atom_features = self.featurizer.num_atom_features
        if atom_categorical_features_channels is None:
            self.atom_categorical_features_emb = None
            num_total_atom_features += sum(self.featurizer.num_atom_categorical_features)
        else:
            self.atom_categorical_features_emb = nn.ModuleList(
                [
                    nn.Embedding(num_classes, atom_categorical_features_channels)
                    for num_classes in self.featurizer.num_atom_categorical_features
                ]
            )
            num_total_atom_features += atom_categorical_features_channels

        if gnn_class == "GPS_PNA":
            # Use the GPS layer from `graphium`.
            try:
                import graphium
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "You are using a model configuration that requires the `graphium` library. "
                    "Install it via `pip install syntheseus-graphium` or choose different settings."
                )

            if bond_out_channels is not None:
                raise ValueError("The GPS implementation does not support bond-level outputs")

            self.gnn = graphium.nn.architectures.FullGraphMultiTaskNetwork(
                gnn_kwargs={
                    "in_dim": gnn_kwargs["hidden_channels"],
                    "in_dim_edges": self.featurizer.num_bond_features,
                    "out_dim": atom_out_channels,
                    "hidden_dims": gnn_kwargs["hidden_channels"],
                    "depth": gnn_kwargs["num_layers"],
                    "activation": "gelu",
                    "last_activation": None,
                    "dropout": gnn_kwargs["dropout"],
                    "normalization": "layer_norm",
                    "last_normalization": "layer_norm",
                    "residual_type": gnn_kwargs.get("residual_type", "none"),
                    "virtual_node": gnn_kwargs.get("virtual_node"),
                    "layer_type": "pyg:gps",
                    "layer_kwargs": dict(
                        mpnn_type="pyg:pna-msgpass",
                        mpnn_kwargs={key: gnn_kwargs[key] for key in ["aggregators", "scalers"]},
                    ),
                },
                pre_nn_kwargs={
                    "in_dim": num_total_atom_features,
                    "out_dim": gnn_kwargs["hidden_channels"],
                    "hidden_dims": gnn_kwargs["hidden_channels"],
                    "depth": 1,
                    "activation": "relu",
                    "last_activation": "none",
                    "dropout": 0.0,
                },
            )

            self._norm_needs_batch_assignment = False
            self._use_graphium = True
        else:
            # Use a layer from `torch_geometric.nn.models`.
            norm = lookup_by_name(torch_geometric.nn.norm, gnn_kwargs["norm"])(
                in_channels=gnn_kwargs["hidden_channels"]
            )

            # We need to wrap some of the normalization layers, see `NormWrapper` for details.
            if gnn_kwargs["norm"] in NormWrapper.NORMS_REQUIRING_BATCH_ASSIGNMENT:
                self._norm_needs_batch_assignment = True
                norm = NormWrapper(norm)
            else:
                self._norm_needs_batch_assignment = False

            # Make a few adjustments to the kwargs.
            gnn_kwargs = {**gnn_kwargs, "norm": norm, "deg": torch.as_tensor(gnn_kwargs["deg"])}

            self.gnn = lookup_by_name(torch_geometric.nn.models, gnn_class)(
                in_channels=num_total_atom_features,
                out_channels=atom_out_channels,
                edge_dim=self.featurizer.num_bond_features,
                **gnn_kwargs,
            )
            self._use_graphium = False

        if bond_out_channels is None:
            self.edge_proj = None
        else:
            self.edge_proj = nn.Linear(
                2 * atom_out_channels + self.featurizer.num_bond_features,
                bond_out_channels,
            )

        self.aggregation = MultiheadAttentionalAggregation(
            in_channels=atom_out_channels,
            num_heads=aggregation_num_heads,
            out_channels=mol_out_channels,
        )
        self.aggregation_dropout = nn.Dropout(p=aggregation_dropout)

    @property
    def preprocess(self) -> Callable[[InputType], RawGraph]:
        return self.featurizer.__call__

    def collate(self, inputs: list[RawGraph]) -> Batch:
        return Batch.from_data_list([self.tensorize(input) for input in inputs])

    def tensorize(self, data: RawGraph) -> Data:
        if isinstance(data, RawRewriteGraph):
            extra_data = {"node_in_lhs": torch.as_tensor(data.node_in_lhs)}
        else:
            extra_data = {}

        if data.atom_categorical_features is not None:
            extra_data["categorical_feat"] = torch.as_tensor(
                data.atom_categorical_features, dtype=torch.long
            )

        return Data(
            feat=torch.as_tensor(data.atom_features, dtype=torch.float),
            edge_index=torch.as_tensor(data.bonds, dtype=torch.long).T,
            edge_feat=torch.as_tensor(data.bond_features, dtype=torch.float),
            **extra_data,
        )

    def forward(self, batch: Batch) -> EncoderOutput:
        if hasattr(batch, "categorical_feat"):
            combined_embeddings = None
            for idx, feat in enumerate(batch.categorical_feat.T):
                if self.atom_categorical_features_emb is None:
                    embeddings = nn.functional.one_hot(
                        feat, num_classes=self.featurizer.num_atom_categorical_features[idx]
                    )
                    embeddings = embeddings.float().to(batch.feat.device)
                else:
                    embeddings = self.atom_categorical_features_emb[idx](feat)

                if combined_embeddings is None:
                    combined_embeddings = embeddings
                else:
                    # Aggregate embeddings for all categorical features through a simple sum.
                    combined_embeddings += embeddings

            batch.feat = torch.cat([batch.feat, combined_embeddings], dim=-1)
        else:
            assert self.atom_categorical_features_emb is None

        if self._use_graphium:
            atom_outputs = self.gnn(batch).feat
        else:
            atom_features, edge_index, bond_features, batch_assignment = (
                batch.feat,
                batch.edge_index,
                batch.edge_feat,
                batch.batch,
            )

            if self._norm_needs_batch_assignment:
                for norm in self.gnn.norms:
                    norm.batch = batch_assignment

            atom_outputs = self.gnn(atom_features, edge_index, edge_attr=bond_features)

        if self.edge_proj is not None:
            # Compute edge outputs by concatenating endpoint outputs with original bond features...
            bond_outputs = torch.cat(
                (bond_features, atom_outputs[edge_index.T].reshape(bond_features.shape[0], -1)),
                dim=-1,
            )

            # ...and projecting into the desired dimensionality.
            bond_outputs = self.edge_proj(bond_outputs)
        else:
            bond_outputs = None

        if self.aggregation is not None:
            # Compute graph outputs by appropriately aggregating node outputs.
            mol_outputs = self.aggregation(atom_outputs, batch.batch)
            mol_outputs = self.aggregation_dropout(mol_outputs)
        else:
            mol_outputs = None

        return EncoderOutput(
            atom_outputs=atom_outputs, bond_outputs=bond_outputs, mol_outputs=mol_outputs
        )

    def move_batch_to_device(self, batch: Batch, device: str) -> Batch:
        return batch.to(device)

    @property
    def atom_out_channels(self) -> int:
        return self._atom_out_channels

    @property
    def bond_out_channels(self) -> Optional[int]:
        return self._bond_out_channels

    @property
    def mol_out_channels(self) -> int:
        return self._mol_out_channels
