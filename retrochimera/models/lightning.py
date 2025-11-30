from abc import ABCMeta, abstractmethod
from typing import Callable, Generic, Iterable, TypeVar

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.utilities import grad_norm
from torch.nn import functional as f

from retrochimera.utils.logging import get_logger

logger = get_logger(__name__)

SampleType = TypeVar("SampleType")
ProcessedSampleType = TypeVar("ProcessedSampleType")
BatchType = TypeVar("BatchType")


class AbstractLightningModel(pl.LightningModule, metaclass=ABCMeta):
    """Base class for models based on PyTorch Lightning."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        learning_rate_decay_step_size: int = 100,
        learning_rate_decay_rate: float = 0.1,
        optimizer_betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=self.hyperparameters_excluded_from_checkpoint)

        self.learning_rate = learning_rate
        self.learning_rate_decay_step_size = learning_rate_decay_step_size
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.optimizer_betas = optimizer_betas

    @property
    def hyperparameters_excluded_from_checkpoint(self) -> list[str]:
        return []

    def log(self, name, value, *args, **kwargs) -> None:
        if name.startswith("val_") or name.startswith("test_"):
            # Sync validation and test metrics across devices.
            kwargs["sync_dist"] = True

        super().log(name, value.to(self.device), *args, **kwargs)

    @abstractmethod
    def ttv_step(self, batch, step_name):
        pass

    def training_step(self, batch, batch_idx):
        return self.ttv_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.ttv_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.ttv_step(batch, "test")

    def setup(self, *args, **kwargs):
        # If running multi-GPU, make sure different GPUs have different seeds.
        if torch.distributed.is_initialized():
            initial_seed = torch.random.initial_seed()
            rank = torch.distributed.get_rank()
            new_seed = initial_seed + rank

            logger.info(f"Shifting random seed for GPU {rank} from {initial_seed} to {new_seed}")
            torch.manual_seed(new_seed)

        super().setup(*args, **kwargs)

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, betas=self.optimizer_betas
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.learning_rate_decay_step_size,
            gamma=self.learning_rate_decay_rate,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class AbstractModel(
    AbstractLightningModel, Generic[SampleType, ProcessedSampleType, BatchType], metaclass=ABCMeta
):
    """Base class for models that are trainable via `cli/train.py`."""

    @abstractmethod
    def preprocess(
        self, samples: Iterable[SampleType], num_processes: int = 0
    ) -> Iterable[ProcessedSampleType]:
        pass

    @abstractmethod
    def collate(self, samples: list[ProcessedSampleType]) -> BatchType:
        pass


def log_classification_metrics(
    log_fn: Callable,
    step_name: str,
    batch_preds: torch.Tensor,
    batch_targets: torch.Tensor,
    batch_size: int,
) -> None:
    log_fn(
        f"{step_name}_acc",
        torchmetrics.functional.accuracy(preds=batch_preds, target=batch_targets),
        batch_size=batch_size,
    )

    for k in [5, 10, 50]:
        if k < batch_preds.shape[-1]:
            log_fn(
                f"{step_name}_top_{k}_acc",
                torchmetrics.functional.accuracy(preds=batch_preds, target=batch_targets, top_k=k),
                batch_size=batch_size,
            )

    if step_name != "train":
        num_classes = batch_preds.shape[-1]

        # The `retrieval_reciprocal_rank` function is unbatched and somewhat slow. We call it on
        # each sample separately, but only during validation/testing.
        reciprocal_ranks = [
            torchmetrics.functional.retrieval_reciprocal_rank(preds=preds, target=target)
            for preds, target in zip(batch_preds, f.one_hot(batch_targets, num_classes=num_classes))
        ]
        log_fn(
            f"{step_name}_mrr",
            torch.mean(torch.as_tensor(reciprocal_ranks)),
            batch_size=batch_size,
        )
