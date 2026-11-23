import math

import pytest
from syntheseus import Bag

from retrochimera.chem.rewrite import RewriteResult
from retrochimera.inference.template_localization import score_localization


@pytest.mark.parametrize(
    "localizations,expected",
    [
        ([[0, 1]], (math.log(0.2) + math.log(0.5)) / 2),
        ([[0, None, 1], [1, None, 1]], math.log(0.4) / 2),
        ([[None]], 0.0),
    ],
)
def test_localization_scores_only_real_encoder_atoms(localizations, expected) -> None:
    scores = [[math.log(p) for p in row] for row in [[0.2, 0.8], [0.5, 0.5], [0.6, 0.4]]]
    result = RewriteResult(Bag([]), {"localizations": localizations, "localization_scores": scores})
    assert score_localization(result) == pytest.approx(expected)
