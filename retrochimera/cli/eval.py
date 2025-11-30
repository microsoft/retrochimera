"""Script for evaluating a model on a given dataset.

Each of the model types can be loaded from a *single directory*, possibly containing several files
(e.g. checkpoint, config, etc). See individual model wrappers for the model directory formats.

Example invocation:
    python ./retrochimera/cli/eval.py \
        data_dir=[DATA_DIR] \
        model_class=TemplateLocalization \
        model_dir=[MODEL_DIR]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from omegaconf import MISSING
from syntheseus.cli import eval_single_step as eval_syntheseus
from syntheseus.reaction_prediction.inference.config import ModelConfig
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config

from retrochimera.inference.retrochimera import RetroChimeraModel
from retrochimera.inference.smiles_transformer import SmilesTransformerModel
from retrochimera.inference.smiles_transformer_forward import SmilesTransformerForwardModel
from retrochimera.inference.template_classification import TemplateClassificationModel
from retrochimera.inference.template_localization import TemplateLocalizationModel


class BackwardModelClass(Enum):
    RetroChimera = RetroChimeraModel
    SmilesTransformer = SmilesTransformerModel
    SmilesTransformerForward = SmilesTransformerForwardModel
    TemplateClassification = TemplateClassificationModel
    TemplateLocalization = TemplateLocalizationModel


@dataclass
class BackwardModelConfig(ModelConfig):
    """Config for loading one of the supported backward models."""

    model_class: BackwardModelClass = MISSING


@dataclass
class EvalConfig(BackwardModelConfig, eval_syntheseus.BaseEvalConfig):
    """Config for running evaluation on a given dataset."""

    pass


def main(argv: Optional[list[str]]) -> None:
    config: EvalConfig = cli_get_config(argv=argv, config_cls=EvalConfig)
    eval_syntheseus.run_from_config(config, extra_steps=[])  # type: ignore


if __name__ == "__main__":
    main(argv=None)
