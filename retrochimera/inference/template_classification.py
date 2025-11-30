from pathlib import Path
from typing import Any, Optional, Sequence, Union

from syntheseus import BackwardReactionModel, Molecule, SingleProductReaction
from syntheseus.reaction_prediction.utils.inference import get_unique_file_in_dir

from retrochimera.chem.rules import RuleBasedRetrosynthesizer, RulePrediction


class TemplateClassificationModel(RuleBasedRetrosynthesizer, BackwardReactionModel):
    """Wrapper for a model performing template classification backed by a given molecule encoder."""

    def __init__(
        self,
        *args,
        model_dir: Optional[Union[str, Path]],
        device: str = "cuda:0",
        model: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Initializes the TemplateClassification model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.ckpt` file
        """

        if (model_dir is None) == (model is None):
            raise ValueError(
                "Exactly one of `model_dir` and `model` attributes has to be specified."
            )

        from retrochimera.models.template_classification import MCCModel

        if model:
            assert model.rulebase_dir is not None

            self.model = model
            super().__init__(*args, **(kwargs | {"rulebase_dir": model.rulebase_dir}))
        else:
            assert model_dir is not None

            super().__init__(*args, **(kwargs | {"rulebase_dir": model_dir}))
            self.model = MCCModel.load_from_checkpoint(
                get_unique_file_in_dir(model_dir, pattern="*.ckpt")
            )
            self.model.to(device)

        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def get_model_info(self) -> dict[str, Any]:
        return self.model.hparams

    def get_parameters(self):
        return self.model.parameters()

    def _predict_ranked_rules(
        self, targets: list[Molecule], top_k: int = 50
    ) -> list[list[RulePrediction]]:
        from retrochimera.utils.pytorch import get_sorted_ids_and_probs

        mol_outputs = self.model.encoder.forward_raw(targets, device=self.device).mol_outputs
        assert mol_outputs is not None

        return [
            [RulePrediction(id=id, prob=prob) for id, prob in zip(rule_ids, rule_probs)]
            for rule_ids, rule_probs in get_sorted_ids_and_probs(
                self.model.forward(mol_outputs), k=top_k
            )
        ]

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        return [
            prediction_list[:num_results]
            for prediction_list in self.predict(inputs, top_k=num_results)
        ]
