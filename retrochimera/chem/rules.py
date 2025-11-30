from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from rdkit import RDLogger
from syntheseus import Bag, Molecule, SingleProductReaction
from syntheseus.interface.molecule import molecule_bag_to_smiles
from syntheseus.interface.reaction import ReactionMetaData

from retrochimera.chem.rewrite import Rewrite, RewriteResult
from retrochimera.utils.logging import get_logger

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

logger = get_logger(__name__)


@dataclass
class ReactionRule:
    rxn: Rewrite
    smarts: str  # Reaction SMARTS
    rule_id: int  # ID for the rule (e.g. for template classification)
    rule_hash: Optional[str]  # Hash of the rule (useful for database lookup)
    name: Optional[str]  # Name of the rule, if available
    n_support: int = -1  # Number of times the rule appeared in the dataset it was extracted from


class RuleBase:
    DEFAULT_FILE_NAME = "template_lib.json"

    def __init__(self) -> None:
        self.rules: dict[int, ReactionRule] = {}
        self.hash_dict: dict[str, int] = {}

    def add_rule(
        self,
        smarts: str,
        rule_id: Optional[int] = None,
        rule_hash: Optional[str] = None,
        name: Optional[str] = None,
        n_support: int = -1,
    ) -> None:
        """Add a new rule to the rule base.

        Args:
            rxn_smarts: RDKit reaction smarts.
            rule_hash: Hash code of the rule.
            rule_id: ID of the rule, usually the label for Multi Class Classification.
            name: Name of the rule.
            n_support: Support of the rule i.e. number of times it is contained in the dataset it
                was extracted from.
        """

        if rule_id is None:
            rule_id = len(self.rules)

        rule = ReactionRule(
            rxn=Rewrite.from_rdkit(smarts),
            smarts=smarts,
            rule_id=rule_id,
            rule_hash=rule_hash,
            name=name,
            n_support=n_support,
        )

        self.rules[rule_id] = rule

        if rule_hash is not None:
            self.hash_dict[rule_hash] = rule_id

    def __contains__(self, item) -> bool:
        return item in self.rules

    def __getitem__(self, item) -> ReactionRule:
        return self.rules[item]

    def __len__(self) -> int:
        return len(self.rules)

    def get_rule(self, rule_id: int) -> Optional[ReactionRule]:
        return self.rules.get(rule_id)

    def get_rule_from_hash(self, hashcode: str) -> Optional[ReactionRule]:
        rule_id = self.hash_dict.get(hashcode)

        if rule_id is not None:
            return self.rules.get(rule_id)
        else:
            return None

    def save_to_file(self, dir: Union[str, Path], filename: str = DEFAULT_FILE_NAME) -> None:
        """Save all rules into a file using the JSONLines format."""
        output_dir = Path(dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / filename, mode="wt") as f:
            for rule in self.rules.values():
                rule_dict = {
                    key: value for (key, value) in dataclasses.asdict(rule).items() if key != "rxn"
                }
                f.write(json.dumps(rule_dict) + "\n")

    @staticmethod
    def load_from_file(
        dir: Union[str, Path],
        filename: str = DEFAULT_FILE_NAME,
        min_rule_support: Optional[int] = None,
        max_num_rules: Optional[int] = None,
        rule_ids: Optional[Iterable[int]] = None,
    ) -> RuleBase:
        """Load a `RuleBase` saved through `save_to_file`.

        Args:
            dir: Directory under which to look for the `RuleBase` file.
            filename: Expected name of the `RuleBase` file.
            min_rule_support: Minimum `n_support` for a rule to be kept.
            max_num_rules: Maximum number of lowest ID rules to be kept.
            rule_ids: Explicit IDs of rules to be kept.
        """
        rulebase = RuleBase()

        rule_ids_set: Optional[set[int]] = None
        if rule_ids is not None:
            rule_ids_set = set(rule_ids)

        with open(Path(dir) / filename) as f:
            lines = list(f)
            orig_size = len(lines)

            for line in lines:
                data = json.loads(line)

                if min_rule_support is not None and data["n_support"] < min_rule_support:
                    break

                if max_num_rules is not None and len(rulebase) == max_num_rules:
                    break

                if rule_ids_set is not None and data["rule_id"] not in rule_ids_set:
                    continue

                rulebase.add_rule(**data)

        logger.info(
            f"Loaded rulebase "
            f"(min_rule_support = {min_rule_support}, max_num_rules = {max_num_rules}); "
            f"original size: {orig_size}, final size: {len(rulebase)}"
        )

        return rulebase

    @staticmethod
    def load_rule_ids_from_file(
        dir: Union[str, Path], filename: str = DEFAULT_FILE_NAME
    ) -> list[int]:
        """Similar to `load_from_file`, but only loads the rule IDs, and hence is much faster."""
        return [json.loads(line)["rule_id"] for line in open(Path(dir) / filename)]


@dataclass
class RulePrediction:
    id: int  # ID of the rule that the model proposes to apply.
    prob: float  # Probability assigned by the model.

    # The `RulePrediction` object lives from the point when the rule is proposed in
    # `_predict_ranked_rules`, and until it is applied. Models may wish to retain some data computed
    # during the former to use during the latter; these are stored below.
    localization_scores: Optional[list[list[float]]] = None


class RuleBasedRetrosynthesizer:
    def __init__(
        self,
        rulebase_dir: Union[str, Path],
        *,
        num_templates_per_result: int = 10,
        min_num_rules_to_apply: int = 5,
        max_cumulative_prob: float = 1.0,
        apply_rules_timeout: Optional[float] = None,
        include_all_metadata: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a rule-based model.

        Args:
            rulebase_dir: Directory containing the rulebase file.
            num_templates_per_result: Ratio of the maximum number of templates we should try
                applying to the number of results requested by the caller.
            min_num_rules_to_apply: Minimum number of rules to apply before potentially breaking due
                to accumulating a large total probability mass.
            max_cumulative_prob: If the total probability of the rules reaches this threshold, and
                there were at least `min_num_rules_to_apply` applied, then the application will be
                terminated, even if `rules_to_apply` has not been exhausted.
            apply_rules_timeout: Timeout passed to the rule application server; this is the maximum
                amount of time (in seconds) we will wait for the server to apply a *single* rule. If
                not set, we fallback to the default setting in the server.
            include_all_metadata: If set, model outputs will include detailed metadata (may require
                substantial disk space for saving the results).
        """
        super().__init__(**kwargs)  # In case this is one of several base classes in the MRO.

        # Local to avoid circular import.
        from retrochimera.chem.rule_application_server import RuleApplicationServer

        self._server = RuleApplicationServer(
            rulebase_dir=rulebase_dir, rule_application_kwargs=self._rule_application_server_kwargs
        )
        self._num_templates_per_result = num_templates_per_result
        self._min_num_rules_to_apply = min_num_rules_to_apply
        self._max_cumulative_prob = max_cumulative_prob
        self._apply_rules_timeout = apply_rules_timeout
        self._include_all_metadata = include_all_metadata

    @property
    def _rule_application_server_kwargs(self) -> dict[str, Any]:
        return {}

    def predict(
        self, targets: list[Molecule], top_k: int = 50
    ) -> list[Sequence[SingleProductReaction]]:
        """Batched single step retrosynthesis prediction.

        Args:
            targets: Molecules to perform single step prediction on.
            top_k: Number of rules to apply. Each rule can potentially match in multiple locations,
                so number of predictions can be larger than `top_k` (this is intended behaviour).

        Returns:
            A list of sequences of `SingleProductReaction`s, where `i`-th entry corresponds the `i`-th
            input molecule.
        """
        return self.apply_top_rules(
            inputs=targets,
            batch_rules_to_apply=self._predict_ranked_rules(
                targets, top_k=self._num_templates_per_result * top_k
            ),
        )

    def _predict_ranked_rules(
        self, targets: list[Molecule], top_k=50
    ) -> list[list[RulePrediction]]:
        """Predict the `top_k` rule IDs to apply to the each molecule in the batch."""
        return [
            [
                RulePrediction(id=rule_id, prob=1.0 / len(self._server.rule_ids))
                for rule_id in self._server.rule_ids
            ]
            for _ in targets
        ]

    def apply_top_rules(
        self, inputs: list[Molecule], batch_rules_to_apply: list[list[RulePrediction]]
    ) -> list[Sequence[SingleProductReaction]]:
        """Apply the top rules provided.

        Args:
            inputs: Molecules to apply the rules to.
            batch_rules_to_apply: Rules to apply for each input (their IDs and probabilities), in
                the order they should be applied.

        Returns:
            A sequence of `SingleProductReaction`s containing several sets of predicted precursors for
            the input molecule.
        """
        # Truncate the rules to apply based on total probability.
        batch_rule_ids_to_apply: list[list[int]] = [[] for _ in inputs]
        for rules_to_apply, rule_ids_to_apply in zip(batch_rules_to_apply, batch_rule_ids_to_apply):
            total_prob = 0.0
            for rule_prediction in rules_to_apply:
                if (
                    len(rule_ids_to_apply) >= self._min_num_rules_to_apply
                    and total_prob >= self._max_cumulative_prob
                ):
                    break

                total_prob += rule_prediction.prob
                rule_ids_to_apply.append(rule_prediction.id)

        apply_rules_kwargs = (
            {"timeout": self._apply_rules_timeout} if self._apply_rules_timeout else {}
        )

        batch_raw_results = self._server.apply_rules(
            inputs=inputs, rule_ids_to_apply=batch_rule_ids_to_apply, **apply_rules_kwargs
        )

        batch_results: list[Sequence[SingleProductReaction]] = []
        for input, raw_results, rule_predictions in zip(
            inputs, batch_raw_results, batch_rules_to_apply
        ):
            results: list[RewriteResult] = []
            for raw_result, rule_prediction in zip(raw_results, rule_predictions):
                for r in raw_result:
                    r.metadata.update(
                        {
                            "reaction_id": rule_prediction.id,
                            "template_probability": rule_prediction.prob,
                        }
                    )

                    if rule_prediction.localization_scores is not None:
                        r.metadata["localization_scores"] = rule_prediction.localization_scores

                    results.append(r)

            self._rerank_results(results)

            if not self._include_all_metadata:
                # Some metadata cannot be removed earlier as it's needed for e.g. `_rerank_results`.
                keys_to_remove = ["localizations", "localization_scores"]
                for r in results:
                    for key in keys_to_remove:
                        if key in r.metadata:
                            del r.metadata[key]

            seen_outputs: set[Bag[Molecule]] = set()
            results_mols: list[Bag[Molecule]] = []
            results_metadata: list[ReactionMetaData] = []

            for r in results:
                if r.mols in seen_outputs:
                    continue

                seen_outputs.add(r.mols)

                metadata: ReactionMetaData = r.metadata  # type: ignore
                metadata["reaction_smiles"] = f"{molecule_bag_to_smiles(r.mols)}>>{input.smiles}"

                results_mols.append(r.mols)
                results_metadata.append(metadata)

            self._fill_prediction_probability(results_metadata)

            batch_results.append(
                [
                    SingleProductReaction(product=input, reactants=mols, metadata=metadata)
                    for mols, metadata in zip(results_mols, results_metadata)
                ]
            )

        return batch_results

    def _fill_prediction_probability(self, results_metadata: list[ReactionMetaData]) -> None:
        """Fill in the `probability` value for predictions for a single input."""
        for metadata in results_metadata:
            metadata["probability"] = metadata["template_probability"]  # type: ignore[typeddict-item]

    def _rerank_results(self, results: list[RewriteResult]) -> None:
        """Optionally reorder the predictions."""
        pass


def get_products(
    mol: Molecule,
    rule: Rewrite,
    cutoff: float = 0.2,
    return_localization: bool = False,
) -> list[RewriteResult]:
    """Applies the rule to the input molecule.

    Args:
        mol: Reactant bag as single `Molecule` object.
        rule: `Rewrite` object representing the rule to apply.
        cutoff: Only keep a result if its SMILES length is at least a `cutoff` fraction of the
            SMILES length of `mol`.
        return_localization: Whether to also extract the corresponding localization indices.

    Returns:
        A set of possible outcomes for applying this rule to the input mol.
    """

    result_smiles_min_length = len(mol.smiles) * cutoff
    results = rule.apply(mol, return_localization=return_localization)

    results_filtered = []
    for result in results:
        combined_smiles = molecule_bag_to_smiles(result.mols)
        if len(combined_smiles) > result_smiles_min_length:
            results_filtered.append(result)
        else:
            logger.debug(f"Result filtered out: {mol.smiles}>>{combined_smiles}")

    return results_filtered
