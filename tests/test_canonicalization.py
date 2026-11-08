import pytest
from loky import get_reusable_executor

from retrochimera.inference.smiles_transformer import SmilesTransformerModel
from retrochimera.inference.smiles_transformer_forward import SmilesTransformerForwardModel


@pytest.mark.parametrize("model_cls", [SmilesTransformerModel, SmilesTransformerForwardModel])
def test_canonicalization_chunks_preserve_order_and_output(model_cls) -> None:
    # Exercise the real worker pool without loading a transformer checkpoint.
    model = object.__new__(model_cls)
    model._canonicalization_processes = 2
    lines = [("C(C)O", 0.1), ("not-a-smiles", 0.2), ("N.CC", 0.3), ("[CH3:1][OH:2]", 0.4)] * 7
    expected = [("CCO", "CCO", 0.1), ("", "", 0.2), ("CC.N", "CC", 0.3), ("CO", "CO", 0.4)] * 7

    try:
        for chunksize in (1, 8, 32):
            model._canonicalization_chunksize = chunksize
            assert model._canonicalize_predictions(lines) == expected
            assert model._canonicalize_predictions([]) == []
    finally:
        get_reusable_executor().shutdown(wait=True)


@pytest.mark.parametrize("model_cls", [SmilesTransformerModel, SmilesTransformerForwardModel])
@pytest.mark.parametrize("chunksize", [0, -1, True, 1.5, "8", None])
def test_canonicalization_rejects_invalid_chunksize(tmp_path, model_cls, chunksize) -> None:
    with pytest.raises(ValueError, match="canonicalization_chunksize must be a positive integer"):
        model_cls(
            model_dir=tmp_path,
            device="cpu",
            canonicalization_chunksize=chunksize,
        )
