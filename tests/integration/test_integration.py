"""Integration tests for the reaction prediction pipeline.

These tests should be executed **in this order**, and use the shared temporary directory to avoid
duplicating work (e.g. data preprocessing). One downside of separating interconnected bits of
testing is that if one test fails, subsequent ones will likely fail as well; however, we also get
cleaner error reporting than if everything was one big test.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pytest
from syntheseus import Bag, Molecule

from retrochimera.utils.misc import convert_camel_to_snake

NUM_SAMPLES = 30  # For truncating training data.


def run_with_python(path: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--source", "./retrochimera/", "-p", path] + args,
        capture_output=True,
        check=True,
    )


def get_num_lines(path: Path) -> int:
    with open(path, "r") as f:
        return len(f.readlines())


def get_num_files(dir: Path) -> int:
    return len([f for f in os.listdir(dir)])


def test_deduplicate_dataset(deduplicate_output_path: Path) -> None:
    """Test deduplication of reactions in the dataset."""
    raw_data_path = Path("./tests/testdata/uspto_reactions.smi")
    assert get_num_lines(raw_data_path) == 630

    run_with_python(
        path="./retrochimera/cli/deduplicate_dataset.py",
        args=[f"input_path={raw_data_path}", f"output_path={deduplicate_output_path}"],
    )

    # Test data contains 5 artificially inserted duplicates.
    assert get_num_lines(deduplicate_output_path) == 625


def test_split_dataset(split_output_dir: Path, deduplicate_output_path: Path) -> None:
    """Test splitting the dataset into training, validation, and test sets."""
    run_with_python(
        path="./retrochimera/cli/split_dataset.py",
        args=[
            f"input_path={deduplicate_output_path}",
            f"output_dir={split_output_dir}",
            "val_frac=0.05",
            "test_frac=0.1",
            "max_reactants_num=4",
            "min_product_atoms=3",
            "max_product_atoms=60",
        ],
    )

    # Test data contains real reactions from USPTO-50K plus 5 extra ones that get filtered out:
    # - [C:1].C.C.C.C>>[C:1]CCCCC    too many reactants
    # - [C:1]-[C:1]>>[C:1]CCCC       double mapped left-hand side
    # - [C:1].[O:2]>>[C:1][O:2]      product too small
    # - [C:1].CCC...>>[C:1]CCC...    product too large
    # - [C:1]CCCC.O>>[C:1]CCCC       product among the reactants

    assert get_num_lines(split_output_dir / "pista_train.smi") == 527
    assert get_num_lines(split_output_dir / "pista_val.smi") == 31
    assert get_num_lines(split_output_dir / "pista_test.smi") == 62


def test_extract_templates(split_output_dir: Path, processed_output_dir: Path) -> None:
    """Test extraction of templates from the dataset."""
    run_with_python(
        path="./retrochimera/cli/extract_templates.py",
        args=[f"data_dir={split_output_dir}", f"output_dir={processed_output_dir}"],
    )

    EXPECTED_SIZE_TRAIN = 527
    EXPECTED_SIZE_VAL = 31
    EXPECTED_SIZE_TEST = 62

    assert get_num_lines(processed_output_dir / "template_lib.json") == 22
    assert get_num_lines(processed_output_dir / "train.jsonl") == EXPECTED_SIZE_TRAIN
    assert get_num_lines(processed_output_dir / "val.jsonl") == EXPECTED_SIZE_VAL
    assert get_num_lines(processed_output_dir / "test.jsonl") == EXPECTED_SIZE_TEST


def test_augment_rsmiles(processed_output_dir: Path, augmented_output_dir: Path) -> None:
    """Test R-SMILES data augmentation (without using the augmented data for training)."""
    NUM_AUGMENTATIONS = 10

    run_with_python(
        path="./retrochimera/cli/augment_rsmiles.py",
        args=[
            f"data_dir={processed_output_dir}",
            f"augmented_data_dir={augmented_output_dir}",
            f"augmentation={NUM_AUGMENTATIONS}",
        ],
    )

    for (path_input, path_output) in [
        (processed_output_dir / "train.jsonl", augmented_output_dir / "train" / "train.jsonl"),
        (processed_output_dir / "val.jsonl", augmented_output_dir / "val" / "val.jsonl"),
    ]:
        with open(path_input, "r") as f_in, open(path_output, "r") as f_out:
            rxns_input = [json.loads(line) for line in f_in.readlines()]
            rxns_output = [json.loads(line) for line in f_out.readlines()]

        assert len(rxns_output) == NUM_AUGMENTATIONS * len(rxns_input)

        for idx, rxn in enumerate(rxns_input):
            rxns_augmented = rxns_output[idx * NUM_AUGMENTATIONS : (idx + 1) * NUM_AUGMENTATIONS]

            # Check augmentation produced varying outputs.
            for key in ["reactants", "products"]:
                values = [tuple(d["smiles"] for d in r[key]) for r in rxns_augmented]
                assert len(set(values)) >= NUM_AUGMENTATIONS // 2

                # Check that all outputs match the original when canonicalized.
                for value in values:
                    assert Bag(Molecule(smiles) for smiles in value) == Bag(
                        [Molecule(d["smiles"]) for d in rxn[key]]
                    )

            # Check mapped reaction SMILES is the same across augmentations.
            for rxn_augmented in rxns_augmented:
                assert rxn_augmented["mapped_reaction_smiles"] == rxn["mapped_reaction_smiles"]

    # Use a small sample of training data as all folds to let the model easily overfit. We do this
    # *after* testing augmentation to still cover the case of augmenting varied folds.
    processed_train_set_path = processed_output_dir / "train.jsonl"

    with open(processed_train_set_path, "rt") as f:
        lines = f.readlines()
        lines = lines[:NUM_SAMPLES]

    with open(processed_train_set_path, "wt") as f:
        f.writelines(lines)

    for fold in ["val", "test"]:
        shutil.copy(processed_train_set_path, processed_output_dir / f"{fold}.jsonl")


@pytest.mark.parametrize(
    ("testcase", "extra_model_kwargs"),
    [
        ("TemplateClassificationChem", {}),
        ("TemplateClassificationGNN", {}),
        ("TemplateLocalization[softmax]", {}),
        ("TemplateLocalization[sigmoid]", {"classification_loss_type": "sigmoid"}),
        ("SmilesTransformer", {}),
    ],
)
def test_preprocess_and_train(
    testcase: str,
    extra_model_kwargs: dict[str, Any],
    tmp_path: Path,
    processed_output_dir: Path,
) -> None:
    """Test the end-to-end reaction prediction pipeline for various model classes."""
    model_class = testcase.split("[")[0]

    # Prepare kwargs that define a small GNN model to reuse across several model classes.
    small_gnn_kwargs = [
        f"encoder.{k}"
        for k in [
            "atom_out_channels=16",
            "mol_out_channels=512",
            "atom_categorical_features_channels=null",
            "aggregation_num_heads=1",
            "aggregation_dropout=0.0",
            "gnn_kwargs.hidden_channels=64",
            "gnn_kwargs.num_layers=1",
            "gnn_kwargs.dropout=0.0",
        ]
    ]

    # Prepare kwargs for all models (mostly making them smaller).
    model_class_to_kwargs = {
        "TemplateClassificationChem": ["encoder.fingerprint_dim=509"],
        "TemplateClassificationGNN": small_gnn_kwargs,
        "TemplateLocalization": sum([[f"input_{k}", f"rewrite_{k}"] for k in small_gnn_kwargs], [])
        + [
            "classification_label_smoothing=0.1",
            "classification_space_dim=256",
            "free_rewrite_embedding_dim=0",
            "num_negative_rewrites_in_localization=0",
            "negative_to_positive_targets_ratio=1.5",
            "rewrite_encoder_num_epochs=null",
            "training.batch_size=8",
        ],
        "SmilesTransformer": [
            "hidden_dim=32",
            "feedforward_dim=32",
            "n_layers=1",
            "n_heads=2",
            "schedule=constant",
            "warm_up_steps=5",
            "dropout=0.0",
        ],
    }

    train_config_key = f"{convert_camel_to_snake(model_class)}_config"

    featurized_output_dir = tmp_path / f"featurized_{testcase}"
    checkpoint_dir = tmp_path / f"checkpoint_{testcase}"
    log_dir = tmp_path / f"logs_{testcase}"
    eval_results_dir = tmp_path / f"eval_results_{testcase}"

    if model_class.startswith("TemplateClassification"):
        eval_model_class = "TemplateClassification"
    else:
        eval_model_class = model_class

    # Assemble model kwargs which have to be consistent between preprocessing and training.
    model_kwargs = [f"model_class={model_class}"] + [
        f"{train_config_key}.{k}" for k in model_class_to_kwargs[model_class]
    ]

    # Append any extra, test-specific kwargs.
    model_kwargs.extend([f"{train_config_key}.{k}={v}" for k, v in extra_model_kwargs.items()])

    preprocess_args = model_kwargs + [
        f"data_dir={processed_output_dir}",
        f"processed_data_dir={featurized_output_dir}",
        "rulebase_min_rule_support=2",
        "num_processes_preprocessing=1",
    ]

    n_epochs = 450 if model_class == "SmilesTransformer" else 200
    train_args = model_kwargs + [
        f"processed_data_dir={featurized_output_dir}",
        f"checkpoint_dir={checkpoint_dir}",
        f"log_dir={log_dir}",
        f"{train_config_key}.training.n_epochs={n_epochs}",
        f"{train_config_key}.training.learning_rate=0.001",
        f"{train_config_key}.training.learning_rate_decay_step_size={n_epochs // 2}",
        f"{train_config_key}.training.check_val_every_n_epoch=50",
        f"{train_config_key}.training.num_checkpoints_for_averaging=2",
        f"{train_config_key}.training.accelerator=cpu",
        f"{train_config_key}.training.gradient_clip_val=50.0",
        f"{train_config_key}.training.accumulate_grad_batches=1",
    ]

    extra_args = []
    if model_class == "SmilesTransformer":
        featurized_output_dir.mkdir()
        vocab_output_path = featurized_output_dir / "vocab.txt"

        run_with_python(
            path="./retrochimera/cli/build_tokenizer.py",
            args=[f"data_dir={processed_output_dir}", f"output_vocab_path={vocab_output_path}"],
        )

        assert vocab_output_path.exists()

        vocab_arg = f"{train_config_key}.vocab_path={vocab_output_path}"
        train_args += [vocab_arg]
        preprocess_args += [vocab_arg]

        # We did not apply augmentation to the training data, so turn it off for eval.
        extra_args += ["model_kwargs={'augmentation_size': 1}"]

    run_with_python(path="./retrochimera/cli/preprocess.py", args=preprocess_args)

    # Test data is composed of reactions for three most common templates in USPTO-50K, plus some
    # extra reactions using uncommon templates; the latter should get filtered out.
    assert get_num_lines(featurized_output_dir / "template_lib.json") == 3
    assert (featurized_output_dir / "data.h5").exists()

    run_with_python(path="./retrochimera/cli/train.py", args=train_args)

    # As many best checkpoints as specified via `num_checkpoints_for_averaging`.
    assert get_num_files(checkpoint_dir / "best") == 2
    assert get_num_files(checkpoint_dir / "last") == 1
    assert (checkpoint_dir / "combined.ckpt").exists()

    check_model_eval(
        eval_model_class=eval_model_class,
        model_dir=checkpoint_dir,
        data_dir=processed_output_dir,
        eval_results_dir=eval_results_dir,
        extra_args=extra_args,
    )


def check_model_eval(
    eval_model_class: str,
    model_dir: Path,
    data_dir: Path,
    eval_results_dir: Path,
    extra_args: Optional[list[str]] = None,
) -> None:
    """Run model evaluation and check the results."""
    eval_args = [
        f"model_class={eval_model_class}",
        f"model_dir={model_dir}",
        f"data_dir={data_dir}",
        f"results_dir={eval_results_dir}",
        "num_gpus=0",
    ]

    run_with_python(path="./retrochimera/cli/eval.py", args=eval_args + (extra_args or []))

    assert get_num_files(eval_results_dir) == 1
    [eval_results_path] = os.listdir(eval_results_dir)

    with open(eval_results_dir / eval_results_path, "r") as f:
        eval_results = json.load(f)

    assert eval_results["eval_args"]["model_class"] == f"{eval_model_class}Model"
    assert eval_results["eval_args"]["fold"] == "TEST"
    assert eval_results["num_samples"] == NUM_SAMPLES

    # In this example, all template-based are capped at ~90% due to a small fraction of examples
    # using out-of-library templates. They should however be able to overfit the rest.
    assert eval_results["top_k"][0] > 0.8


def test_optimize_ensembles(tmp_path: Path, processed_output_dir: Path) -> None:
    """Test the creation of a model ensemble."""
    # Copy over model results for ensembling.
    ensembling_dir = tmp_path / "ensembling"

    model_classes = ["TemplateLocalization", "SmilesTransformer"]
    testcases = ["TemplateLocalization[softmax]", "SmilesTransformer"]

    # Ensembling is designed to be robust to longer names (e.g. `template_localization_v1`), which
    # should then get shortened (to e.g. `template_localization`) when saving ensemble checkpoint.
    model_class_to_long_lowercase_name = {
        model_class: convert_camel_to_snake(model_class) + f"_v{i}"
        for i, model_class in enumerate(model_classes)
    }

    for model_class, testcase in zip(model_classes, testcases):
        # Ensembling needs both validation and test outputs. We only ran the latter, but since the
        # folds are the same, we just copy the results twice and adjust the fold key as needed.

        for fold in ["validation", "test"]:
            target_dir = (
                ensembling_dir
                / fold
                / model_class_to_long_lowercase_name[model_class]
                / "eval_results_test"
            )
            target_dir.parent.mkdir(parents=True, exist_ok=True)

            shutil.copytree(tmp_path / f"eval_results_{testcase}", target_dir)

            if fold == "validation":
                [results_path] = os.listdir(target_dir)

                with open(target_dir / results_path, "r") as f:
                    results = json.load(f)

                results["eval_args"]["fold"] = "VALIDATION"

                with open(target_dir / results_path, "w") as f:
                    json.dump(results, f)

        shutil.copytree(
            tmp_path / f"checkpoint_{testcase}",
            ensembling_dir / "input_checkpoints" / model_class_to_long_lowercase_name[model_class],
        )

    ensemble_checkpoint_dir = ensembling_dir / "output_checkpoints"

    run_with_python(
        path="./retrochimera/cli/optimize_ensembles.py",
        args=[
            f"data_dir={processed_output_dir}",
            f"results_dir={ensembling_dir}",
            f"output_dir={ensembling_dir}",
            f"input_checkpoints_dir={ensembling_dir / 'input_checkpoints'}",
            f"output_checkpoints_dir={ensemble_checkpoint_dir}",
            "dataset=test",
            "num_repeats=2",
        ],
    )

    with open(ensembling_dir / "individual.json", "rt") as f:
        individual_results = json.load(f)
        assert individual_results.keys() == set(
            model_class_to_long_lowercase_name[model_class] for model_class in model_classes
        )

    with open(ensembling_dir / "ensembles_optimized.json", "rt") as f:
        ensembling_results = json.load(f)
        assert len(ensembling_results) == 1

        chimera_result = ensembling_results["smiles_transformer_v1 template_localization_v0"]
        assert len(chimera_result) == 3

        # Check that ensembling weights look as expected.
        assert len(chimera_result["weights"]) == 2
        assert len(chimera_result["weights"][0]) == 100
        assert len(chimera_result["weights"][1]) == 100

        # Check that the ensemble achieves good accuracy.
        for fold in ["validation", "test"]:
            assert chimera_result[f"{fold}_accuracies"][0] > 0.85

    # Check that ensemble checkpoint was saved correctly.
    [ensemble_ckpt_dir] = os.listdir(ensemble_checkpoint_dir)
    assert ensemble_ckpt_dir == "ensemble_smiles_transformer_v1_template_localization_v0"

    ensemble_checkpoint_full_dir = ensemble_checkpoint_dir / ensemble_ckpt_dir
    assert get_num_files(ensemble_checkpoint_full_dir) == 3
    assert (ensemble_checkpoint_full_dir / "template_localization").is_dir()
    assert (ensemble_checkpoint_full_dir / "smiles_transformer").is_dir()

    with open(ensemble_checkpoint_full_dir / "models.json", "rt") as f:
        config = json.load(f)

        assert len(config) == 2
        for model_class in model_classes:
            (model_class_saved, weights) = config[convert_camel_to_snake(model_class)]

            assert model_class_saved == f"{model_class}Model"
            assert len(weights) == 100

    check_model_eval(
        eval_model_class="Ensemble",
        model_dir=ensemble_checkpoint_full_dir,
        data_dir=processed_output_dir,
        eval_results_dir=tmp_path / "eval_results_ensemble",
    )
