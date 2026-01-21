from typing import Any, Sequence

from syntheseus import Molecule, SingleProductReaction
from syntheseus.reaction_prediction.inference_base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import get_unique_file_in_dir

from retrochimera.chem.rules import RuleBasedRetrosynthesizer, RulePrediction


class TemplateClassificationModel(RuleBasedRetrosynthesizer, ExternalBackwardReactionModel):
    """Wrapper for a model performing template classification backed by a given molecule encoder."""

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the TemplateClassification model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.ckpt` file
        """

        from retrochimera.models.template_classification import MCCModel

        super().__init__(*args, **kwargs)
        self.start_server(rulebase_dir=self.model_dir)

        self.model = MCCModel.load_from_checkpoint(
            get_unique_file_in_dir(self.model_dir, pattern="*.ckpt")
        )
        self.model.to(self.device)
        self.model.eval()

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
