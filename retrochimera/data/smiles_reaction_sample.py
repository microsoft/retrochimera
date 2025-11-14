from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from syntheseus.interface.molecule import SMILES_SEPARATOR
from syntheseus.reaction_prediction.data.reaction_sample import ReactionType

from retrochimera.data.template_reaction_sample import TemplateReactionSample


@dataclass(frozen=True)
class SmilesReactionSample(TemplateReactionSample):
    """Reaction sample that also loads the SMILES with no canonicalization or standardization."""

    # We need to provide default values due to inheritance, but never expect these to be kept.
    raw_products_smiles: str = field(default="", hash=True, compare=True)
    raw_reactants_smiles: str = field(default="", hash=True, compare=True)

    @classmethod
    def from_dict(cls: type[ReactionType], data: dict[str, Any]) -> ReactionType:
        for key in ["reactants", "products"]:
            raw_key = f"raw_{key}_smiles"
            if raw_key not in data:
                data[raw_key] = SMILES_SEPARATOR.join([d["smiles"] for d in data[key]])

        return super().from_dict(data)
