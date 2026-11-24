import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from syntheseus import Molecule
from syntheseus.reaction_prediction.chem.utils import molecule_bag_from_smiles

from retrochimera.chem.rewrite import Rewrite
from retrochimera.chem.rules import RuleBase
from retrochimera.data.dataset import DataFold, ReactionDataset
from retrochimera.data.preprocessing import template_classification, template_localization
from retrochimera.data.processed_dataset import ProcessedDataModule
from retrochimera.data.template_reaction_sample import (
    TemplateApplicationResult,
    TemplateReactionSample,
)
from retrochimera.encoders.chem import ChemEncoder
from retrochimera.encoders.configs import GNNEncoderConfig
from retrochimera.encoders.featurizers import DGLLifeMoleculeFeaturizer, DGLLifeRewriteFeaturizer
from retrochimera.encoders.gnn import GNNEncoder
from retrochimera.utils.training import preprocess_and_save


def test_preprocessing_retains_classification_and_real_atom_targets_with_synthetic_hydrogens() -> (
    None
):
    product = Molecule("CCO")
    [result] = Rewrite.from_rdkit(
        "([C:1](-[#1&^0:2])-[O:3]-[#1&^0:4])>>([C:1]=[O:3].[#1:2]-[#1:4])"
    ).apply(product)
    assert result.metadata["localizations"] == [[1, None, 2, None]]

    sample = template_localization._process_sample(
        (product, result.mols, 17, [result]),
        input_preprocess_fn=DGLLifeMoleculeFeaturizer.get_default(),
    )
    assert sample is not None, "Synthetic H rows must not discard the classification sample"
    graph, target, loc_target = sample
    assert len(graph.atom_features) == product.rdkit_mol.GetNumAtoms()
    assert target == 17
    # Synthetic H rows have no encoder atom; retain real-atom targets without normalizing H rows.
    np.testing.assert_array_equal(
        loc_target,
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "template,reactants,accepted",
    [
        ("[N+:1]>>[N+:1]", "C[NH3+]", True),
        ("[N+:1]>>[N+:1]", "C[NH3+].[Cl-]", True),
        ("[N+:1]>>[N+:1]", "C[NH3+].[Br-]", False),
        ("[N+:1]>>[NH2+0:1]", "CN", True),
        ("[N+:1]>>[NH2+0:1]", "CN.[Cl-]", True),
        ("([N+:1].[Cl-:2])>>[N+:1]", "C[NH3+]", True),
        ("([N+:1].[Cl-:2])>>[N+:1]", "C[NH3+].[Cl-]", False),
    ],
)
def test_preprocessing_matches_reference_with_or_without_spectator_counterions(
    template: str, reactants: str, accepted: bool, tmp_path: Path
) -> None:
    rulebase = RuleBase()
    rulebase.add_rule(template)
    rulebase.save_to_file(tmp_path)
    product = Molecule("C[NH3+].[Cl-]")
    sample = TemplateReactionSample.from_reaction_smiles_strict(
        f"{reactants}>>{product.smiles}",
        mapped=False,
        template_application_results=[TemplateApplicationResult(template_id=0)],
    )

    processed = list(
        template_localization.preprocess_samples(
            [sample],
            tmp_path,
            input_encoder=GNNEncoder(**asdict(GNNEncoderConfig())),
            num_processes=1,
        )
    )
    assert len(processed) == int(accepted)
    if accepted:
        assert processed[0].target == 0
        assert (
            processed[0].loc_target.sum()
            == Rewrite.from_rdkit(template).rdkit_lhs_mol.GetNumAtoms()
        )


def test_preprocessing_does_not_overweight_repeated_hydrogen_expanded_localizations() -> None:
    product = Molecule("CCCl")
    [result] = Rewrite.from_rdkit("([C:1]-[#1,#17:2])>>([C:1]-[*:2])").apply(product)
    processed = template_localization._process_sample(
        (product, molecule_bag_from_smiles("CCCl"), 0, [result]),
        input_preprocess_fn=DGLLifeMoleculeFeaturizer.get_default(),
    )
    assert processed is not None
    np.testing.assert_allclose(processed[2], [[1 / 3, 2 / 3, 0], [0, 0, 1]])


def run_preprocessing_with_dummy_model(model, dataset: ReactionDataset):
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir, "data.h5")
        preprocess_and_save(
            save_path=save_path,
            dataset=dataset,
            model=model,
            num_processes=1,
        )

        datamodule = ProcessedDataModule(
            h5_path=save_path, data_loader_kwargs={"collate_fn": lambda x: x, "shuffle": False}
        )

        train_data: list = sum(datamodule.train_dataloader(), [])
        test_data: list = sum(datamodule.test_dataloader(), [])

        assert len(train_data) == dataset.get_num_samples(DataFold.TRAIN)
        assert len(test_data) == dataset.get_num_samples(DataFold.TEST)

        return train_data, test_data


@pytest.fixture
def rulebase_dir(reaction_dataset: ReactionDataset) -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        reaction_dataset.rulebase.save_to_file(dir=temp_dir)
        yield temp_dir


def test_preprocess_classification(reaction_dataset: ReactionDataset, rulebase_dir: str):
    class DummyModel:
        def __init__(self):
            self.encoder = ChemEncoder(fingerprint_dim=16)

        def preprocess(self, samples: list[TemplateReactionSample], num_processes: int):
            return template_classification.preprocess_samples(
                samples=samples,
                rulebase_dir=rulebase_dir,
                encoder=self.encoder,
                num_processes=num_processes,
            )

    train_data, test_data = run_preprocessing_with_dummy_model(
        model=DummyModel(), dataset=reaction_dataset
    )

    for processed_sample, raw_sample in zip(train_data, reaction_dataset[DataFold.TRAIN]):
        assert processed_sample.target == raw_sample.template_application_results[0].template_id

    for processed_sample, raw_sample in zip(test_data, reaction_dataset[DataFold.TEST]):
        assert processed_sample.target == raw_sample.template_application_results[0].template_id


def test_preprocess_localization(reaction_dataset: ReactionDataset, rulebase_dir: str):
    rulebase = reaction_dataset.rulebase

    class DummyModel:
        def __init__(self):
            input_encoder_config = GNNEncoderConfig()
            rewrite_encoder_config = GNNEncoderConfig(
                featurizer_class="DGLLifeRewriteFeaturizer",
                featurizer_kwargs=DGLLifeRewriteFeaturizer.prepare_kwargs(
                    rewrites=(rule.rxn for rule in rulebase.rules.values())
                ),
            )

            self.input_encoder = GNNEncoder(**asdict(input_encoder_config))
            self.rewrite_encoder = GNNEncoder(**asdict(rewrite_encoder_config))

        def preprocess(self, samples: list[TemplateReactionSample], num_processes: int):
            return template_localization.preprocess_samples(
                samples=samples,
                rulebase_dir=rulebase_dir,
                input_encoder=self.input_encoder,
                num_processes=num_processes,
            )

    train_data, test_data = run_preprocessing_with_dummy_model(
        model=DummyModel(), dataset=reaction_dataset
    )

    for sample in train_data + test_data:
        # Localization target should have the correct shape.
        assert sample.loc_target.shape == (
            rulebase[sample.target].rxn.rdkit_lhs_mol.GetNumAtoms(),
            len(sample.input.atom_features),
        )

        # Localization target for each rewrite node should be a valid distribution.
        assert np.all(sample.loc_target >= 0.0)
        assert np.all(sample.loc_target <= 1.0)
        np.testing.assert_allclose(sample.loc_target.sum(axis=1), 1.0)
