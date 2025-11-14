from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from numpy.random import default_rng
from syntheseus import BackwardReactionModel, Molecule, SingleProductReaction

from retrochimera.chem.rules import RuleBasedRetrosynthesizer, RulePrediction


class NullModel(RuleBasedRetrosynthesizer, BackwardReactionModel):
    """Skips the top templates then randomly applies remaining templates without any checking."""

    def __init__(
        self,
        *args,
        template_lib_dir: Optional[Union[str, Path]],
        skip: int = 4000,
        max_rules: int = 5000,
        **kwargs,
    ) -> None:
        self.skip = skip
        self.max_rules = max_rules

        super().__init__(*args, **(kwargs | {"rulebase_dir": template_lib_dir}))

        self.num_rules = len(self._server.rule_ids)

    def _predict_ranked_rules(
        self, targets: list[Molecule], top_k: int = 50
    ) -> list[list[RulePrediction]]:

        rng = default_rng(1337)

        ids = np.arange(self.skip, self.num_rules)

        numbers = rng.choice(ids, size=self.max_rules, replace=False)

        return [
            [RulePrediction(id=i, prob=1.0 / self.max_rules) for i in list(numbers)]
            for _ in targets
        ]

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        return [
            prediction_list[:num_results]
            for prediction_list in self.predict(inputs, top_k=num_results)
        ]
