import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import omegaconf
import torch
from syntheseus import BackwardReactionModel, Bag, Molecule, SingleProductReaction
from syntheseus.interface.reaction import ReactionMetaData
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel

from retrochimera import inference
from retrochimera.utils.misc import lookup_by_name


class RetroChimeraModel(ExternalBackwardReactionModel):
    """Wrapper for the RetroChimera model."""

    def __init__(
        self,
        *args,
        model_dir: Optional[Union[str, Path]] = None,
        probability_from_score_temperature: float = 8.0,
        **kwargs,
    ) -> None:
        """Initializes the RetroChimera model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the ensembling config in the `models.json` file
        - `model_dir/[SUBMODEL]` contains the checkpoint files for model `SUBMODEL`
        """

        model_dir = Path(model_dir or self.get_default_model_dir())
        model_data = self._load_model_data_from_dir(Path(model_dir))

        model_kwargs = {}
        for key in model_data:
            if key in kwargs:
                # If the RetroChimera constructor receives an argument whose name matches one of the
                # constituent models, pass through the value as model kwargs downstream.
                value = kwargs[key]
                if not isinstance(value, (dict, omegaconf.DictConfig)):
                    raise RuntimeError(
                        f"Value for {key} should be the model kwargs for {model_data[key][0]}; "
                        f"found value of type {type(value)} instead of dict"
                    )

                model_kwargs[key] = value
                del kwargs[key]

        # Having removed kwargs meant for constituent models we can now call base class `__init__`.
        super().__init__(*args, model_dir=model_dir, **kwargs)

        self._init_from_dir(model_dir=model_dir, model_data=model_data, model_kwargs=model_kwargs)
        self.probability_from_score_temperature = probability_from_score_temperature

    def _load_model_data_from_dir(self, model_dir: Path) -> dict[str, tuple[str, list[float]]]:
        with open(model_dir / "models.json") as f:
            return json.load(f)

    def _init_from_dir(
        self,
        model_dir: Path,
        model_data: dict[str, tuple[str, list[float]]],
        model_kwargs: dict[str, Any],
    ) -> None:
        self._models: list[BackwardReactionModel] = []
        self._model_names: list[str] = []
        self._model_weights: list[list[float]] = []

        for model_subdir, (model_class_name, model_weights) in model_data.items():
            model_class = lookup_by_name(inference, model_class_name)

            self._models.append(
                model_class(
                    model_dir=model_dir / model_subdir,
                    device=self.device,
                    **model_kwargs.get(model_subdir, {}),
                )
            )
            self._model_names.append(model_subdir)
            self._model_weights.append(model_weights)

    def get_parameters(self):
        return itertools.chain(*(model.get_parameters() for model in self._models))

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        model_batch_results: list[list[Sequence[SingleProductReaction]]] = [
            model(inputs=inputs, num_results=num_results) for model in self._models
        ]

        return [
            combine_results(
                input=input,
                model_results=list(model_results),
                model_weights=self._model_weights,
                num_results=num_results,
                model_names=self._model_names,
                probability_from_score_temperature=self.probability_from_score_temperature,
            )
            for input, model_results in zip(inputs, zip(*model_batch_results))
        ]


def combine_results(
    input: Molecule,
    model_results: list[Sequence[SingleProductReaction]],
    model_weights: list[list[float]],
    num_results: int,
    model_names: Optional[list[str]] = None,
    probability_from_score_temperature: Optional[float] = None,
) -> list[SingleProductReaction]:
    # Generate default model names if not provided.
    model_names = model_names or [f"model_{i}" for i in range(len(model_results))]

    result_to_ranks: dict[Bag[Molecule], list[Optional[int]]] = defaultdict(
        lambda: [None] * len(model_results)
    )
    result_to_metadata: dict[Bag[Molecule], list[Optional[ReactionMetaData]]] = defaultdict(
        lambda: [None] * len(model_results)
    )

    for model_idx, results in enumerate(model_results):
        for rank, r in enumerate(results):
            result_to_ranks[r.reactants][model_idx] = rank
            result_to_metadata[r.reactants][model_idx] = r.metadata

    result_to_score: dict[Bag[Molecule], float] = {}
    for result, ranks in result_to_ranks.items():
        result_to_score[result] = 0.0
        for rk, weights in zip(ranks, model_weights):
            if rk is not None:
                if rk < len(weights):
                    result_to_score[result] += weights[rk]
                else:
                    # Extend the weights sequence at the end for tie-breaking.
                    result_to_score[result] += weights[-1] / (rk - len(weights) + 2)

    combined_results = sorted(
        result_to_score.keys(), key=lambda r: result_to_score[r], reverse=True
    )[:num_results]

    metadata_list = []
    for rct in combined_results:
        metadata_list.append(
            ReactionMetaData(  # type: ignore[typeddict-unknown-key]
                score=result_to_score[rct],
                individual_ranks={
                    model_name: result_to_ranks[rct][idx]
                    for idx, model_name in enumerate(model_names)
                },
                individual_metadata={
                    model_name: result_to_metadata[rct][idx]
                    for idx, model_name in enumerate(model_names)
                },
            )
        )

    if probability_from_score_temperature is not None:
        # Divide the scores by the sum of rank-1 scores across all models (so that the best possible
        # score is normalized to 1), then multiply by the given temperature before applying softmax.
        factor = probability_from_score_temperature / sum(weights[0] for weights in model_weights)
        scores = [result_to_score[r] * factor for r in combined_results]
        probabilities = torch.nn.functional.softmax(torch.as_tensor(scores), dim=-1)

        for metadata, probability in zip(metadata_list, probabilities.numpy().tolist()):
            metadata["probability"] = probability

    return [
        SingleProductReaction(product=input, reactants=reactants, metadata=metadata)
        for reactants, metadata in zip(combined_results, metadata_list)
    ]
