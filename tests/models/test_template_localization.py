import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from syntheseus import Molecule

from retrochimera.chem.rules import RuleBase
from retrochimera.cli.train import ModelClass, ModelTrainingConfig, build_model_from_config
from retrochimera.data.preprocessing.template_localization import ProcessedSample
from retrochimera.models.template_localization import TemplateLocalizationModel


@pytest.mark.parametrize("num_negative_rewrites", [0, 1])
def test_localization_loss_masks_synthetic_hydrogens_with_label_smoothing(
    tmp_path, monkeypatch, num_negative_rewrites: int
) -> None:
    rulebase = RuleBase()
    rulebase.add_rule(smarts="([C:1](-[#1&^0:2])-[O:3]-[#1&^0:4])>>([C:1]=[O:3].[#1:2]-[#1:4])")
    rulebase.add_rule(smarts="[C:1]>>[C:1]")
    config = OmegaConf.structured(ModelTrainingConfig(model_class=ModelClass.TemplateLocalization))
    config.template_localization_config.num_negative_rewrites_in_localization = (
        num_negative_rewrites
    )
    config.template_localization_config.negative_to_positive_targets_ratio = 1.0
    for encoder in (
        config.template_localization_config.input_encoder,
        config.template_localization_config.rewrite_encoder,
    ):
        encoder.gnn_kwargs.hidden_channels = 8
        encoder.gnn_kwargs.num_layers = 1
    model, _ = build_model_from_config(config, rulebase, tmp_path)
    assert isinstance(model, TemplateLocalizationModel)
    model._prepare_rewrite_graphs()
    batch = model.collate(
        [
            ProcessedSample(
                input=model.input_encoder.preprocess(Molecule("CCO")),
                target=0,
                loc_target=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32),
            )
        ]
    )
    rows, columns = 4 + num_negative_rewrites, 3 + bool(num_negative_rewrites)
    logits = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    logits.requires_grad_()
    monkeypatch.setattr(model, "forward_localization", lambda **kwargs: [logits])
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "retrochimera.models.template_localization.log_classification_metrics",
        lambda **kwargs: None,
    )

    loss = model.ttv_step(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[[1, 3]]) == 0
    assert torch.count_nonzero(logits.grad[[0, 2]]) > 0
    if num_negative_rewrites:
        assert torch.count_nonzero(logits.grad[4:]) > 0
