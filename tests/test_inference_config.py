from pathlib import Path

import pytest
from omegaconf import OmegaConf
from syntheseus.reaction_prediction.utils.config import get_config
from syntheseus.reaction_prediction.utils.model_loading import get_model

from retrochimera import RetroChimeraModel
from retrochimera.cli.eval import EvalConfig
from retrochimera.cli.run_search import SearchConfig

PROFILE = Path(__file__).resolve().parents[1] / "docs/configs/fast-fp16.yaml"


@pytest.mark.parametrize("config_cls", [EvalConfig, SearchConfig])
def test_fast_profile_matches_eval_and_search_schemas(config_cls) -> None:
    config = get_config(["--config", str(PROFILE), "model_dir=/unused"], config_cls)
    assert config.model_class.value is RetroChimeraModel
    assert OmegaConf.to_container(config.model_kwargs) == {
        "call_submodels_in_parallel": True,
        "smiles_transformer": {
            "inference_precision": "float16",
            "beam_size": 20,
            "augmentation_size": 10,
            "filter_duplicate_augmentations": True,
            "canonicalization_processes": 16,
            "canonicalization_chunksize": 8,
            "use_cache": False,
        },
        "template_localization": {"num_processes": 16, "use_cache": False},
    }


@pytest.mark.parametrize(
    "config_cls,overrides,loader_kwargs",
    [
        (EvalConfig, ["model_kwargs.use_cache=False"], {"remove_duplicates": True}),
        (SearchConfig, [], {"use_cache": False, "default_num_results": 20}),
    ],
)
def test_fast_profile_does_not_overlap_native_loader_kwargs(
    monkeypatch, config_cls, overrides, loader_kwargs
) -> None:
    config = get_config(["--config", str(PROFILE), "model_dir=/unused", *overrides], config_cls)
    received = {}

    def record_init(self, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(RetroChimeraModel, "__init__", record_init)
    model = get_model(config, num_gpus=0, **loader_kwargs)
    assert isinstance(model, RetroChimeraModel)
    assert received["use_cache"] is False
    assert received["device"] == "cpu"
    assert received["smiles_transformer"]["inference_precision"] == "float16"
    assert received["smiles_transformer"]["canonicalization_chunksize"] == 8
    assert received["template_localization"]["num_processes"] == 16
