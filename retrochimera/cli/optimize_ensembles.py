"""Script for evaluating model ensembles using automatically optimized ensembling weights.

In contrast to `eval.py` this script is "offline": it uses pre-generated outputs from the former to
rapidly try many ensembling configurations and hyperparameters.
"""

import glob
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import MISSING
from syntheseus import Bag, Molecule, SingleProductReaction
from syntheseus.reaction_prediction.data.dataset import DataFold, DiskReactionDataset
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from syntheseus.reaction_prediction.utils.metrics import TopKMetricsAccumulator
from syntheseus.reaction_prediction.utils.misc import set_random_seed

from retrochimera.inference.ensemble import combine_results
from retrochimera.utils.logging import get_logger
from retrochimera.utils.misc import convert_camel_to_snake

logger = get_logger(__name__)


@dataclass
class EvalEnsemblesConfig:
    data_dir: str = MISSING  # Directory containing the data
    results_dir: str = MISSING  # Directory containing results from evaluating all of the models
    output_dir: str = MISSING  # Directory to use for saving the result
    dataset: str = "uspto"  # Dataset name used to look up "eval_results_{dataset}" subdirectories
    num_repeats: int = 5  # Number of repeats of the weight optimization process to be averaged out

    # Options below are for saving ensemble checkpoints, which are constructed by copying original
    # model checkpoints and saving them in a separate subdirectory together with a `models.json`
    # file containing the ensembling weights. Either both or neither of these paths should be set.
    input_checkpoints_dir: Optional[str] = None  # Directory containing model checkpoints
    output_checkpoints_dir: Optional[str] = None  # Directory for saving ensemble checkpoints
    shorten_model_names: bool = True  # Whether to shorten model names when constructing checkpoints


def make_mol(smiles: str) -> Molecule:
    try:
        return Molecule(smiles, make_rdkit_mol=False)
    except Exception:
        return Molecule(smiles, canonicalize=False, make_rdkit_mol=False)


def load_ground_truths(
    data_dir: str, folds: list[DataFold]
) -> dict[DataFold, list[tuple[Bag[Molecule], Bag[Molecule]]]]:
    """Load ground truth outputs for all reactions in the dataset in order."""
    dataset = DiskReactionDataset(data_dir, sample_cls=ReactionSample)
    return {fold: [(r.products, r.reactants) for r in dataset[fold]] for fold in folds}


def load_predicted_reactants(data: dict[str, Any]) -> list[list[list[Molecule]]]:
    raw_reactants: list[list[list[Molecule]]] = []
    for predictions in data["predictions"]:
        if isinstance(predictions, dict):
            predictions = predictions["predictions"]  # Handle older output format.

        raw_reactants.append(  # Handle older output format.
            [
                [make_mol(r["smiles"]) for r in p.get("reactants", p.get("output"))]
                for p in predictions
            ]
        )

    return raw_reactants


def eval_model(
    results: list[list[Bag[Molecule]]], ground_truths: list[Bag[Molecule]]
) -> list[float]:
    metrics = TopKMetricsAccumulator(max_num_results=100)

    assert len(results) == len(ground_truths)
    for result, ground_truth in zip(results, ground_truths):
        metrics.add([r == ground_truth for r in result])

    return metrics.top_k


def eval_ensemble(
    model_results: list[list[list[Bag[Molecule]]]],
    ground_truths: list[Bag[Molecule]],
    model_weights: list[list[float]],
) -> list[float]:
    metrics = TopKMetricsAccumulator(max_num_results=100)

    dummy_product = Molecule("")  # Product is not important here.
    for results, ground_truth in zip(zip(*model_results), ground_truths):
        results_combined = combine_results(
            input=dummy_product,
            model_results=[
                [SingleProductReaction(product=dummy_product, reactants=r) for r in result]
                for result in results
            ],
            model_weights=model_weights,
            num_results=100,
        )

        metrics.add([r.reactants == ground_truth for r in results_combined])

    return metrics.top_k


def get_weights_from_params(params: torch.Tensor) -> torch.Tensor:
    weights = torch.exp(params)  # Transform into non-negative.
    weights = torch.cumsum(weights, dim=1)  # Transform into increasing.
    weights = torch.cumsum(weights, dim=1)  # Transform into convex.
    weights = weights.flip(dims=[1])
    return weights / weights.mean()


def optimize_ensemble_weights(
    model_results: list[list[list[Bag[Molecule]]]],
    ground_truths: list[Bag[Molecule]],
    num_results: int = 100,
    initial_learning_rate: float = 0.1,
    initial_temperature: float = 0.1,
    num_steps: int = 1000,
    decay_every_n_steps: int = 25,
    decay_factor: float = 0.9,
    margin: float = 1e-4,
    regularization_loss_weight: float = 0.2,
) -> torch.Tensor:
    num_models = len(model_results)

    rank_list_pairs: list[torch.Tensor] = []
    for results, ground_truth in zip(zip(*model_results), ground_truths):
        result_to_ranks: dict[Bag[Molecule], list[int]] = defaultdict(
            lambda: [num_results] * num_models
        )

        for model_idx, results in enumerate(results):
            for rank, r in enumerate(results):
                result_to_ranks[r][model_idx] = rank

        if ground_truth not in result_to_ranks:
            # If the ground-truth was not recovered at all then there's nothing to optimize.
            continue

        ranks_ground_truth = result_to_ranks[ground_truth]
        for r, ranks in result_to_ranks.items():
            if (
                r == ground_truth
                or all([pi <= ni for (pi, ni) in zip(ranks_ground_truth, ranks)])
                or all([pi >= ni for (pi, ni) in zip(ranks_ground_truth, ranks)])
            ):
                # If one of the results has better rank from all models than another result, then
                # the ensemble will always rank them in the same way; nothing to optimize.
                continue

            # Stack the ranks into a tensor and shift to make them valid indices into an unraveled
            # ensembling weight matrix.
            rank_list_pairs.append(
                torch.as_tensor([ranks_ground_truth, ranks])
                + (num_results + 1) * torch.arange(num_models)
            )

    rank_list_pairs_stacked = torch.stack(rank_list_pairs, dim=0)
    params = torch.nn.parameter.Parameter(data=torch.zeros((num_models, num_results)))

    def compute_loss(temperature: float) -> torch.Tensor:
        weights = get_weights_from_params(params)
        all_weight_ratios = [
            weights[model_idx_1] / weights[model_idx_2]
            for model_idx_1 in range(num_models)
            for model_idx_2 in range(num_models)
            if model_idx_1 != model_idx_2
        ]

        weights = torch.cat([weights, torch.zeros(num_models, 1)], dim=-1).ravel()

        # Map ranks to scores, and sum over the models.
        scores = weights[rank_list_pairs_stacked].sum(dim=-1)

        # Subtract the positive scores (first column) from the negative ones (second column).
        scores[:, 0] *= -1
        scores = scores.sum(dim=-1)

        # Compute the ranking loss.
        loss_ranking = torch.sigmoid((scores + margin) / temperature).sum() / len(ground_truths)

        # Compute the regularization loss.
        loss_regularization = 0.0
        for weight_ratios in all_weight_ratios:
            loss_regularization += (weight_ratios[:-1] - weight_ratios[1:]).abs().mean()

        loss_regularization = loss_regularization / len(all_weight_ratios)
        return loss_ranking + regularization_loss_weight * loss_regularization

    optimizer = torch.optim.Adam([params], lr=initial_learning_rate)
    temperature = initial_temperature

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = compute_loss(temperature)
        loss.backward()
        optimizer.step()

        if (step + 1) % decay_every_n_steps == 0 and step != num_steps - 1:
            temperature *= decay_factor
            for g in optimizer.param_groups:
                g["lr"] *= decay_factor

    return get_weights_from_params(params).detach()


def merge_top_k(
    top_k_1: list[float], num_samples_1: int, top_k_2: list[float], num_samples_2: int
) -> list[float]:
    # Go back from accuracies to number of correct answers (which can be mapped to an integer).
    [num_correct_1, num_correct_2] = [
        [round(v * num_samples) for v in top_k]
        for (top_k, num_samples) in [(top_k_1, num_samples_1), (top_k_2, num_samples_2)]
    ]

    num_correct_total = [v_1 + v_2 for (v_1, v_2) in zip(num_correct_1, num_correct_2)]
    num_samples_total = num_samples_1 + num_samples_2

    return [v / num_samples_total for v in num_correct_total]


def run_from_config(config: EvalEnsemblesConfig) -> None:
    set_random_seed(0)

    if (config.input_checkpoints_dir is None) != (config.output_checkpoints_dir is None):
        raise ValueError(
            "Both or neither of `input_checkpoints_dir` and `output_checkpoints_dir` should be set"
        )

    output_dir = Path(config.output_dir)

    logger.info(f"Loading dataset ground truths from {config.data_dir}")
    ground_truths = {
        fold: [reactants for _, reactants in samples]
        for fold, samples in load_ground_truths(
            data_dir=config.data_dir, folds=[DataFold.VALIDATION, DataFold.TEST]
        ).items()
    }

    # If the results for some of the models are split into subdirectories, make sure these are
    # processed in correct order (we assume a naming scheme of the form `...1`, `...2`, etc).
    result_paths = sorted(
        glob.glob(f"{config.results_dir}/**/eval_results_{config.dataset}/*.json", recursive=True)
    )

    result_paths_joined = "\n".join(result_paths)
    logger.info(f"Found {len(result_paths)} results files:\n{result_paths_joined}")

    model_to_fold_to_results: dict[str, dict[DataFold, list[list[Bag[Molecule]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    model_to_fold_to_top_k: dict[str, dict[DataFold, list[float]]] = defaultdict(dict)
    model_to_short_name: dict[str, str] = {}
    model_to_model_class: dict[str, str] = {}
    fold_name_to_fold = {f.name.lower(): f for f in DataFold}

    for path in result_paths:
        path_parts = path.split("/")

        [(fold, model_class)] = [
            (fold_name_to_fold[fold_name], model_class)
            for (fold_name, model_class) in zip(path_parts, path_parts[1:])
            if fold_name in fold_name_to_fold
        ]

        if fold == DataFold.TRAIN:
            continue

        logger.info(f"Loading data for model {model_class} and fold {fold}")
        with open(path, "rt") as f:
            data = json.load(f)

            if not data["predictions"]:
                continue

            args = data["eval_args"]

            # Convert case (e.g. "TemplateLocalizationModel" -> "template_localization").
            short_name = convert_camel_to_snake(args["model_class"].removesuffix("Model"))

            # Make sure the model class and fold that we parsed out of the path matches the file.
            assert model_class.startswith(short_name)

            model_to_short_name[model_class] = short_name
            model_to_model_class[model_class] = args["model_class"]

            # Results for Transformer have an incorrect fold recorded in the output files.
            if "smiles_transformer" not in model_class:
                assert args["fold"] == fold.name

            top_k = [float(value) for value in data["top_k"]]

            if fold not in model_to_fold_to_top_k[model_class]:
                model_to_fold_to_top_k[model_class][fold] = top_k
            else:
                model_to_fold_to_top_k[model_class][fold] = merge_top_k(
                    top_k_1=model_to_fold_to_top_k[model_class][fold],
                    num_samples_1=len(model_to_fold_to_results[model_class][fold]),
                    top_k_2=top_k,
                    num_samples_2=len(data["predictions"]),
                )

            for reactants in load_predicted_reactants(data):
                model_to_fold_to_results[model_class][fold].append([Bag(r) for r in reactants])

    models_to_ensemble: list[str] = []
    for model, model_results in model_to_fold_to_results.items():
        if set(model_results.keys()) < {DataFold.VALIDATION, DataFold.TEST}:
            logger.warning(
                f"Model {model} will not be used for ensembling as not all folds were found"
            )
        else:
            assert set(model_results.keys()) == {DataFold.VALIDATION, DataFold.TEST}

            for fold in model_results:
                assert len(ground_truths[fold]) == len(model_results[fold])

            models_to_ensemble.append(model)

    if config.input_checkpoints_dir is not None:
        for model in models_to_ensemble:
            model_ckpt_dir = Path(config.input_checkpoints_dir) / model
            if not model_ckpt_dir.is_dir():
                raise ValueError(
                    f"Ensemble checkpoint saving is on, but {model_ckpt_dir} does not exist"
                )

        if config.shorten_model_names and len(models_to_ensemble) != len(
            set(model_to_short_name[m] for m in models_to_ensemble)
        ):
            raise ValueError(
                "Ensemble checkpoint saving with model name shortening is on, but the model names "
                "are not unique when shortened (i.e. two checkpoints of the same underlying model)."
            )

        if len(models_to_ensemble) > 2:
            logger.warning(
                "Ensemble checkpoint saving is on, so we will save a checkpoint for each "
                f"combination. As {len(models_to_ensemble)} models are being ensembled, this may "
                "use a lot of disk space (we copy all checkpoints involved in each ensemble)."
            )

    model_to_test_top_k = {
        model: model_to_fold_to_top_k[model][DataFold.TEST] for model in model_to_fold_to_top_k
    }

    with open(output_dir / "individual.json", "wt") as f:
        json.dump(model_to_test_top_k, f)

    logger.info("Verifying results for models tested in isolation")
    for model in model_to_fold_to_results:
        for fold in model_to_fold_to_results[model]:
            individual_results = eval_model(
                results=model_to_fold_to_results[model][fold],
                ground_truths=ground_truths[fold],
            )

            # Verify that we get the same accuracies as what was reported.
            assert np.allclose(individual_results, model_to_fold_to_top_k[model][fold])

    models_public = []
    models_private = []
    for model in models_to_ensemble:
        if "smiles_transformer" in model or "template_localization" in model:
            models_private.append(model)
        else:
            models_public.append(model)

    # First, ensemble all model pairs.
    model_subsets_to_ensemble = [list(t) for t in combinations(models_public + models_private, 2)]

    if config.dataset == "uspto":
        # On USPTO-50K RetroKNN's adapter was trained on validation set, inflating the model's
        # perceived performance, and making the ensembling algorithm put excessive weight on
        # RetroKNN. We thus exclude it from the set for the larger ensembles.
        models_public = [model for model in models_public if model != "retro_knn"]

    # Second, also add ensembles containing all public models and a subset of private models.
    for num_private in range(len(models_private) + 1):
        if len(models_public) + num_private < 2:
            continue

        for t in combinations(models_private, num_private):
            new_subset = models_public + list(t)
            if new_subset not in model_subsets_to_ensemble:
                model_subsets_to_ensemble.append(new_subset)

    model_subsets_to_ensemble = [sorted(model_subset) for model_subset in model_subsets_to_ensemble]

    model_subsets_joined = "\n".join([" ".join(ms) for ms in model_subsets_to_ensemble])
    logger.info(f"Ensembling the following subsets of models:\n{model_subsets_joined}")

    all_results: dict[str, dict[str, list]] = defaultdict(dict)
    for model_subset in model_subsets_to_ensemble:
        key = " ".join(model_subset)
        logger.info(f"Running ensembling for {key}")

        all_weights = [
            optimize_ensemble_weights(
                model_results=[
                    model_to_fold_to_results[model][DataFold.VALIDATION] for model in model_subset
                ],
                ground_truths=ground_truths[DataFold.VALIDATION],
            )
            for _ in range(config.num_repeats)
        ]

        results: dict[str, Any] = {}
        results["weights"] = (sum(all_weights) / len(all_weights)).numpy().tolist()

        for fold in [DataFold.VALIDATION, DataFold.TEST]:
            results[f"{fold.name.lower()}_accuracies"] = eval_ensemble(
                model_results=[model_to_fold_to_results[model][fold] for model in model_subset],
                ground_truths=ground_truths[fold],
                model_weights=results["weights"],
            )

        # Dump under a string key for JSON saving to work.
        all_results[key] = results

        if config.output_checkpoints_dir is not None:
            assert config.input_checkpoints_dir is not None

            checkpoint_dir = (
                Path(config.output_checkpoints_dir) / f"ensemble_{'_'.join(model_subset)}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving ensemble checkpoint to {checkpoint_dir}")

            model_config: dict[str, tuple[str, list[float]]] = {}
            for model in model_subset:
                short_name = model_to_short_name[model]
                subdir_name = short_name if config.shorten_model_names else model

                input_checkpoint_dir = Path(config.input_checkpoints_dir) / model
                output_checkpoint_dir = checkpoint_dir / subdir_name
                output_checkpoint_dir.mkdir(parents=True, exist_ok=True)

                for item in input_checkpoint_dir.iterdir():
                    if item.is_file():
                        shutil.copy(item, output_checkpoint_dir)

                model_config[short_name] = (
                    model_to_model_class[model],
                    results["weights"][model_subset.index(model)],
                )

            with open(checkpoint_dir / "models.json", "wt") as f:
                json.dump(model_config, f)

    with open(output_dir / "ensembles_optimized.json", "wt") as f:
        json.dump(all_results, f)


def main(argv: Optional[list[str]]) -> None:
    config: EvalEnsemblesConfig = cli_get_config(argv=argv, config_cls=EvalEnsemblesConfig)
    run_from_config(config)


if __name__ == "__main__":
    main(argv=None)
