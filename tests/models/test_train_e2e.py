import dataclasses
import tempfile
from pathlib import Path
from typing import Any, Union

import pytest
from syntheseus.cli.eval_single_step import get_results

from retrochimera.cli.train import (
    ModelClass,
    TemplateClassificationChemConfig,
    TemplateClassificationGNNConfig,
    TrainConfig,
    train,
)
from retrochimera.data.dataset import DataFold, ReactionDataset
from retrochimera.encoders.configs import GNNEncoderConfig
from retrochimera.inference.template_classification import TemplateClassificationModel
from retrochimera.models.template_classification import MCCModel
from retrochimera.utils.training import preprocess_and_save
from tests.chem.test_rules import check_results_against_targets


@pytest.mark.parametrize(
    "encoder_class,encoder_kwargs",
    [
        ("ChemEncoder", {"fingerprint_dim": 1024}),
        ("GNNEncoder", dataclasses.asdict(GNNEncoderConfig()) | {"aggregation_dropout": 0.0}),
    ],
)
def test_workflow_end_to_end(
    encoder_class: str, encoder_kwargs: dict[str, Any], reaction_dataset: ReactionDataset
) -> None:
    """Test the full workflow of training an `MCCModel`, loading it, and running inference."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TrainConfig(
            processed_data_dir=tmp_dir,
            model_class=ModelClass[f"TemplateClassification{encoder_class[:-7]}"],
            checkpoint_dir=tmp_dir,
        )

        model_config: Union[TemplateClassificationChemConfig, TemplateClassificationGNNConfig]
        if encoder_class == "ChemEncoder":
            model_config = config.template_classification_chem_config
        elif encoder_class == "GNNEncoder":
            model_config = config.template_classification_gnn_config
        else:
            raise ValueError(f"Encoder class {encoder_class} not recognized")

        model_config.mlp.dropout = 0.0
        model_config.mlp.hidden_dim = 128
        model_config.training.n_epochs = 30
        model_config.training.accelerator = "cpu"

        model = MCCModel(
            encoder_class=encoder_class,
            encoder_kwargs=encoder_kwargs,
            hidden_dim=model_config.mlp.hidden_dim,
            n_classes=len(reaction_dataset.rulebase),
            n_hidden_layers=model_config.mlp.n_layers,
        )
        reaction_dataset.rulebase.save_to_file(dir=config.checkpoint_dir)
        model.set_rulebase(rulebase=reaction_dataset.rulebase, rulebase_dir=config.checkpoint_dir)

        # Run data preprocessing and training.
        processed_data_path = str(Path(tmp_dir) / "data.h5")

        preprocess_and_save(
            save_path=processed_data_path, dataset=reaction_dataset, model=model, num_processes=0
        )
        results = train(
            model=model, data_path=processed_data_path, config=config, model_config=model_config
        )

        # Check that the training has worked.
        assert len(results.test_results) > 0
        assert results.test_results[0]["test_acc"] > 0.75

        # Build the wrapper, and exit the context (the temporary dir will not be needed unless the
        # rule application processes need to be restarted which shouldn't happen here).
        model_wrapper = TemplateClassificationModel(model_dir=tmp_dir, device="cpu")

    train_reactant_smiles = []
    train_product_mols = []

    for datapoint in reaction_dataset[DataFold.TRAIN]:
        train_reactant_smiles.append(datapoint.reactants_str)
        train_product_mols.append(list(datapoint.products)[0])

    # Run inference using a lower-level API specific to `TemplateClassificationModel`.
    inf_results = model_wrapper.predict(train_product_mols)
    check_results_against_targets(inf_results, train_reactant_smiles)

    # Run inference via the generic `get_results`.
    results_from_eval = get_results(model_wrapper, inputs=train_product_mols, num_results=10)
    check_results_against_targets(results_from_eval.results, train_reactant_smiles)
