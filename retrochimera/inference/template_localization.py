import math
from typing import Any, Optional, Sequence

from syntheseus import Molecule, SingleProductReaction
from syntheseus.interface.reaction import ReactionMetaData
from syntheseus.reaction_prediction.inference_base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import get_unique_file_in_dir

from retrochimera.chem.rewrite import RewriteResult
from retrochimera.chem.rules import RuleBasedRetrosynthesizer, RulePrediction


class TemplateLocalizationModel(RuleBasedRetrosynthesizer, ExternalBackwardReactionModel):
    """Wrapper for the template localization model."""

    def __init__(
        self,
        *args,
        localization_score_weight: float = 2.25,
        classification_temperature: Optional[float] = 30.0,
        output_temperature: float = 1.0,
        **kwargs,
    ) -> None:
        """Initializes the TemplateLocalization model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.ckpt` file
        """
        import torch

        from retrochimera.models.template_localization import (
            TemplateLocalizationModel as LocalizationModel,
        )

        super().__init__(*args, **kwargs)
        self.start_server(rulebase_dir=self.model_dir)

        self.model = LocalizationModel.load_from_checkpoint(
            get_unique_file_in_dir(self.model_dir, pattern="*.ckpt")
        )
        self._localization_score_weight = localization_score_weight
        self._classification_temperature = classification_temperature
        self._output_temperature = output_temperature

        self.model.to(self.device)
        self.model.eval()

        # These should be set if the model was loaded from a trained checkpoint.
        assert self.model.all_rewrites_atom_outputs is not None
        assert self.model.all_rewrites_batch_idx is not None

        # Unpack the rewrite atom outputs by rule ID. The `batch_idx` tensor is sorted, so we find
        # group boundaries with `unique_consecutive` and then split.
        _, counts = torch.unique_consecutive(self.model.all_rewrites_batch_idx, return_counts=True)
        self.all_rewrites_atom_outputs_list: list[torch.Tensor] = list(
            torch.split(self.model.all_rewrites_atom_outputs, counts.tolist())
        )

    @property
    def _rule_application_server_kwargs(self) -> dict[str, Any]:
        return {"return_localization": True}

    def get_model_info(self) -> dict[str, Any]:
        return dict(self.model.hparams)

    def get_parameters(self):
        return self.model.parameters()

    def _predict_ranked_rules(
        self, targets: list[Molecule], top_k: int = 50
    ) -> list[list[RulePrediction]]:
        import torch

        from retrochimera.utils.pytorch import get_sorted_ids_and_probs, tensor_to_list

        # Preprocess and batch the inputs.
        input_graphs = self.model.input_encoder.collate(
            [self.model.input_encoder.preprocess(target) for target in targets]
        )
        input_graphs = self.model.input_encoder.move_batch_to_device(input_graphs, self.device)
        input_graphs_enc = self.model.input_encoder.forward(input_graphs)

        assert input_graphs_enc.mol_outputs is not None
        assert self.model.all_rewrites_mol_outputs is not None

        # Run the classification branch to get logits for all templates.
        batch_rule_logits = self.model.forward_classification(
            input_reprs=input_graphs_enc.mol_outputs,
            rewrite_reprs=self.model.all_rewrites_mol_outputs,
            rewrite_ids=torch.arange(self.model.n_classes, device=self.device),
            temperature=self._classification_temperature,
        )

        input_graphs_atom_outputs = input_graphs_enc.atom_outputs
        assert input_graphs_atom_outputs is not None

        results = []
        for idx, (rule_ids, rule_probs) in enumerate(
            get_sorted_ids_and_probs(batch_rule_logits, k=top_k)
        ):
            input_graph_atom_outputs = torch.t(input_graphs_atom_outputs[input_graphs.batch == idx])

            lhs_chunks = [self.all_rewrites_atom_outputs_list[rule_id] for rule_id in rule_ids]
            chunk_sizes = [c.size(0) for c in lhs_chunks]

            all_scores = torch.mm(torch.cat(lhs_chunks, dim=0), input_graph_atom_outputs)
            all_scores = torch.nn.functional.log_softmax(all_scores, dim=-1)
            all_scores_list = tensor_to_list(all_scores)

            offset = 0
            graph_results = []
            for rule_id, rule_prob, chunk_size in zip(rule_ids, rule_probs, chunk_sizes):
                graph_results.append(
                    RulePrediction(
                        id=rule_id,
                        prob=rule_prob,
                        localization_scores=all_scores_list[offset : offset + chunk_size],
                    )
                )
                offset += chunk_size

            results.append(graph_results)

        return results

    def _fill_prediction_probability(self, results_metadata: list[ReactionMetaData]) -> None:
        import torch

        scaled_scores = [
            metadata["combined_score"] / self._output_temperature for metadata in results_metadata  # type: ignore[typeddict-item]
        ]
        probabilities = torch.nn.functional.softmax(torch.as_tensor(scaled_scores), dim=-1)

        for metadata, probability in zip(results_metadata, probabilities.numpy().tolist()):
            metadata["probability"] = probability

    def _rerank_results(self, results: list[RewriteResult]) -> None:
        for rewrite_result in results:
            classification_score = math.log(rewrite_result.metadata["template_probability"])
            localization_score = score_localization(rewrite_result)
            combined_score = (
                classification_score + self._localization_score_weight * localization_score
            )

            rewrite_result.metadata.update(
                {
                    "classification_score": classification_score,
                    "localization_score": localization_score,
                    "combined_score": combined_score,
                }
            )

        results.sort(key=lambda t: t.metadata["combined_score"], reverse=True)

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        return [
            prediction_list[:num_results]
            for prediction_list in self.predict(inputs, top_k=num_results)
        ]


def score_localization(result: RewriteResult) -> float:
    localizations = result.metadata["localizations"]
    localization_scores = result.metadata["localization_scores"]

    total_logprob = 0.0
    for index_rewrite_lhs, indices_input_mol in enumerate(zip(*localizations)):
        prob = 0.0
        for index_input_mol in set(indices_input_mol):
            prob += math.exp(localization_scores[index_rewrite_lhs][index_input_mol])

        total_logprob += math.log(prob)

    # A sum of log probabilities is more mathematically correct, but an average empirically seems
    # to work a bit better, so we do the latter.
    return total_logprob / len(localizations[0])
