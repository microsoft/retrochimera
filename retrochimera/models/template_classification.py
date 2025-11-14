from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Iterable, Optional, Union

import torch
from torch.nn import functional as f

from retrochimera.chem.rules import RuleBase
from retrochimera.data.preprocessing.template_classification import (
    ProcessedSample,
    preprocess_samples,
)
from retrochimera.data.template_reaction_sample import TemplateReactionSample
from retrochimera.encoders.base import BatchType, get_encoder_by_name
from retrochimera.layers.mlp import MLP
from retrochimera.models.lightning import AbstractModel, log_classification_metrics


@dataclass
class Batch(Generic[BatchType]):
    inputs: BatchType
    targets: Any


class MCCModel(AbstractModel[TemplateReactionSample, ProcessedSample, Batch]):
    """Model performing template classification on top of an arbitrary encoder.

    This model simply takes the molecule-level output from a backbone `Encoder`, and passes that
    through an MLP to compute template probabilities.
    """

    def __init__(
        self,
        encoder_class: str,
        encoder_kwargs: dict[str, Any],
        hidden_dim: int,
        n_classes: int,
        n_hidden_layers: int = 1,
        dropout: float = 0.2,
        label_smoothing: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_hidden_layers = n_hidden_layers
        self.encoder_class = encoder_class
        self.encoder_kwargs = encoder_kwargs
        self.dropout = dropout
        self._label_smoothing = label_smoothing
        self.rulebase_dir: Optional[Union[str, Path]] = None

        self.rebuild()

    def set_rulebase(self, rulebase: RuleBase, rulebase_dir: Union[str, Path]) -> None:
        self.rulebase_dir = rulebase_dir

    def rebuild(self):
        self.encoder = get_encoder_by_name(self.encoder_class)(**self.encoder_kwargs).to(
            self.device
        )
        self.input_dim = self.encoder.mol_out_channels
        self.mlp = MLP(
            self.input_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.n_classes,
            n_layers=self.n_hidden_layers,
            dropout=self.dropout,
        ).to(self.device)

    def forward(self, x):
        return self.mlp(x)

    def ttv_step(self, batch: Batch, step_name: str):
        batch_encoded = self.encoder(batch.inputs)
        batch_size = batch_encoded.mol_outputs.shape[0]

        y_hat = self.forward(batch_encoded.mol_outputs)
        loss = f.cross_entropy(y_hat, batch.targets, label_smoothing=self._label_smoothing)

        self.log(f"{step_name}_loss", loss, batch_size=batch_size)
        log_classification_metrics(
            log_fn=self.log,
            step_name=step_name,
            batch_preds=y_hat,
            batch_targets=batch.targets,
            batch_size=batch_size,
        )

        return loss

    def preprocess(
        self, samples: Iterable[TemplateReactionSample], num_processes: int = 0
    ) -> Iterable[ProcessedSample]:
        assert self.rulebase_dir is not None
        yield from preprocess_samples(
            samples=samples,
            rulebase_dir=self.rulebase_dir,
            encoder=self.encoder,
            num_processes=num_processes,
        )

    def collate(self, samples: list[ProcessedSample]) -> Batch:
        return Batch(
            inputs=self.encoder.collate([sample.input for sample in samples]),
            targets=torch.as_tensor([sample.target for sample in samples], dtype=torch.long),
        )
