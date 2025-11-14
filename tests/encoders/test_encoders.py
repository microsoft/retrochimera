import dataclasses
import os
from typing import Optional, cast

import pytest
import pytorch_lightning as pl
import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import NumAromaticRings
from syntheseus.interface.molecule import Molecule
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from retrochimera.encoders.chem import ChemEncoder
from retrochimera.encoders.configs import GNNEncoderConfig
from retrochimera.encoders.gnn import GNNEncoder


def assert_shape(t: Optional[Tensor], expected_shape: list[int]) -> None:
    assert t is not None
    assert isinstance(t, Tensor)
    assert list(t.shape) == expected_shape


class Model(pl.LightningModule):
    def __init__(self, encoder) -> None:
        super().__init__()

        self.encoder = encoder
        self.proj = nn.Linear(encoder.mol_out_channels, 1)

    def forward(self, batch: Batch):
        return self.proj(self.encoder(batch).mol_outputs).squeeze(-1)

    def ttv_step(self, batch, step_name):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)

        self.log(f"{step_name}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.ttv_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.ttv_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.ttv_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


@pytest.fixture
def train_mols() -> list[Molecule]:
    return [
        Molecule(smiles.rstrip())
        for smiles in open(
            os.path.join(os.path.dirname(__file__), "..", "testdata", "guacamol_100_smiles.txt")
        )
    ]


def get_default_chem_encoder() -> ChemEncoder:
    return ChemEncoder(fingerprint_dim=2048)


def get_default_gnn_config() -> GNNEncoderConfig:
    config = GNNEncoderConfig(
        atom_out_channels=16, bond_out_channels=24, mol_out_channels=32, aggregation_dropout=0.0
    )
    config.gnn_kwargs["hidden_channels"] = 8
    config.gnn_kwargs["num_layers"] = 1

    return config


def get_default_gnn_encoder() -> GNNEncoder:
    return GNNEncoder(**dataclasses.asdict(get_default_gnn_config()))


def get_graphium_gnn_encoder() -> GNNEncoder:
    config = get_default_gnn_config()
    config.gnn_class = "GPS_PNA"
    config.bond_out_channels = None

    return GNNEncoder(**dataclasses.asdict(config))


@pytest.mark.parametrize(
    "encoder_fn", [get_default_chem_encoder, get_default_gnn_encoder, get_graphium_gnn_encoder]
)
class TestEncoders:
    def test_encode_mols(self, encoder_fn) -> None:
        mols = [Molecule(smiles) for smiles in ["c1ccccc1N=O", "CCC"]]

        encoder = encoder_fn()
        output = encoder.forward_raw(mols)

        rdkit_mols = [mol.rdkit_mol for mol in mols]
        num_atoms = sum(mol.GetNumAtoms() for mol in rdkit_mols)
        num_bonds = sum(mol.GetNumBonds() for mol in rdkit_mols)
        num_mols = len(mols)

        # How many outputs we expect for each "kind" of output.
        num_outputs_per_level = {"atom": num_atoms, "bond": 2 * num_bonds, "mol": num_mols}

        for level, expected_num_outputs in num_outputs_per_level.items():
            num_channels = getattr(encoder, f"{level}_out_channels")

            if num_channels is not None:
                output_for_level = getattr(output, f"{level}_outputs")
                assert_shape(output_for_level, [expected_num_outputs, num_channels])

    def test_train(self, encoder_fn, train_mols: list[Chem.Mol]):
        for _ in range(20):
            encoder = encoder_fn()
            train_mols_processed = [encoder.preprocess(mol) for mol in train_mols]

            # Set the targets to the number of aromatic rings, which should be learnable by any encoder.
            train_targets = torch.as_tensor(
                [NumAromaticRings(mol.rdkit_mol) for mol in train_mols],
                dtype=torch.float32,
            )

            # A list implements the `Dataset` interface via duck typing.
            dataset = cast(Dataset, list(zip(train_mols_processed, train_targets)))

            # Compute the loss for a baseline that always predicts the mean of all the targets.
            predict_the_mean_loss = torch.nn.functional.mse_loss(
                train_targets, train_targets.mean()
            )

            def collate_fn(samples: list[tuple]):
                inputs = [input for input, _ in samples]
                targets = [target for _, target in samples]
                return encoder.collate(inputs), torch.as_tensor(targets)

            model = Model(encoder)
            data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

            trainer = pl.Trainer(max_epochs=20, logger=False, enable_checkpointing=False)

            # Before training the predictions should be no better than a naive baseline unless we
            # got lucky with initialization. In the latter case we stay in the loop and retry.
            [results_before] = trainer.test(model, dataloaders=data_loader)
            if results_before["test_loss"] > predict_the_mean_loss:
                break

        trainer.fit(model, train_dataloaders=data_loader)
        model.eval()

        # After training the predictions should be better than the baseline.
        [results_after] = trainer.test(model, dataloaders=data_loader)
        assert results_after["test_loss"] < 0.75 * predict_the_mean_loss

    def test_batching(self, encoder_fn) -> None:
        encoder = encoder_fn()
        mols = [Molecule(smiles) for smiles in ["CN", "CC"]]

        encoder.eval()
        outputs_unbatched = [encoder.forward_raw([mol]) for mol in mols]
        outputs_batched = encoder.forward_raw(mols)

        for level in ["atom", "bond", "mol"]:
            if getattr(encoder, f"{level}_out_channels") is None:
                continue

            encodings_unbatched = [
                getattr(output, f"{level}_outputs") for output in outputs_unbatched
            ]
            encodings_batched = getattr(outputs_batched, f"{level}_outputs")

            # Encodings should depend on the molecule.
            assert not torch.allclose(encodings_unbatched[0], encodings_unbatched[1], atol=1e-4)

            # Encodings should not depend on whether there were other molecules in the batch.
            assert torch.allclose(torch.cat(encodings_unbatched), encodings_batched, atol=1e-4)
