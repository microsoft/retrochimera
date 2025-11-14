from typing import Callable, Optional

import torch
from syntheseus.interface.molecule import Molecule
from torch import Tensor

from retrochimera.chem.descriptors import ChemData, MoleculeEmbedder
from retrochimera.encoders.base import Encoder, EncoderOutput

BatchType = tuple[Tensor, Tensor]


class ChemEncoder(Encoder[Molecule, ChemData, BatchType]):
    """Implements a molecule encoder based on fixed chemical features such as fingerprints."""

    def __init__(self, fingerprint_dim: int, sparse: bool = True, log_x: bool = True) -> None:
        super().__init__()

        self.fingerprint_dim = fingerprint_dim
        self.embedder = MoleculeEmbedder(size=fingerprint_dim, sparse=sparse, log_x=log_x)

    @property
    def preprocess(self) -> Callable[[Molecule], ChemData]:
        return self.embedder.convert_mol

    def collate(self, inputs: list[ChemData]) -> BatchType:
        embeddings, masks = self.embedder._postprocess(
            X=[input.embedding for input in inputs], mask=[input.valid for input in inputs]
        )
        return (
            torch.as_tensor(embeddings.todense(), dtype=torch.float),
            torch.as_tensor(masks.flatten()),
        )

    def forward(self, batch: BatchType) -> EncoderOutput:
        mol_outputs, _ = batch  # The second element is a validity mask, which we ignore for now.
        return EncoderOutput(atom_outputs=None, bond_outputs=None, mol_outputs=mol_outputs)

    def move_batch_to_device(self, batch: BatchType, device: str) -> BatchType:
        outputs, validity_mask = batch
        return outputs.to(device), validity_mask.to(device)

    @property
    def atom_out_channels(self) -> Optional[int]:
        return None

    @property
    def bond_out_channels(self) -> Optional[int]:
        return None

    @property
    def mol_out_channels(self) -> int:
        return self.fingerprint_dim
