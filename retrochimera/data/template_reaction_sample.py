from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample, ReactionType
from syntheseus.reaction_prediction.utils.misc import undictify_bag_of_molecules

from retrochimera.chem.rewrite import RewriteResult


@dataclass(frozen=True)
class TemplateApplicationResult:
    """Result from applying a fixed template on a given product.

    If `results` is `None`, that means the template application result was not recorded, and a
    downstream user should recompute it if needed. If `results` is an empty list, then the template
    was applied, but it did not match.

    The `ground_truth` flag is set if the template was produced by running template extraction on a
    data sample during preprocessing; usually this means that `results` contains a result with the
    ground-truth reactant bag, although in some situations this may not be the case (due to e.g
    stereochemistry issues).
    """

    template_id: int  # Corresponding template ID
    ground_truth: bool = True  # Whether this template was produced by running extraction
    results: Optional[list[RewriteResult]] = None  # Results from applying the template

    @classmethod
    def from_dict(
        cls: type[TemplateApplicationResult], data: dict[str, Any]
    ) -> TemplateApplicationResult:
        if data["results"] is not None:
            for r in data["results"]:
                r["mols"] = undictify_bag_of_molecules(r["mols"])

            data["results"] = [RewriteResult(**r) for r in data["results"]]

        return cls(**data)


@dataclass(frozen=True)
class TemplateReactionSample(ReactionSample):
    """Reaction sample containing results from applying one or many templates."""

    template: Optional[str] = field(default=None, hash=False, compare=False)
    template_application_results: list[TemplateApplicationResult] = field(
        default_factory=lambda: []
    )

    @classmethod
    def from_dict(cls: type[ReactionType], data: dict[str, Any]) -> ReactionType:
        data["template_application_results"] = [
            TemplateApplicationResult.from_dict(d) for d in data["template_application_results"]
        ]

        return super().from_dict(data)
