from typing import Any

import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.callbacks import Callback

from retrochimera.utils.logging import get_logger

logger = get_logger(__name__)


class OptLRMonitor(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, trainer, *args, **kwargs):
        # Only support one optimizer
        opt = trainer.optimizers[0]

        # Only support one param group
        stats = {"lr-Adam": opt.param_groups[0]["lr"]}
        trainer.logger.log_metrics(stats, step=trainer.global_step)


class ModelCheckpoint(pl_callbacks.ModelCheckpoint):
    """Custom `ModelCheckpoint` callback which fixes checkpoint paths upon directory change.

    Note that we keep the original class name for backward compatibility.
    """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)

        if self.dirpath != dirpath_from_ckpt:
            logger.warning(
                f"Current checkpoint directory {self.dirpath} differs from {dirpath_from_ckpt} "
                "found in the checkpoint. Updating all paths with the new prefix."
            )
            assert self.dirpath is not None

            for key in [
                "dirpath",
                "best_model_path",
                "last_model_path",
                "best_k_models",
                "kth_best_model_path",
            ]:
                value = state_dict[key]

                if not value:
                    continue

                if isinstance(value, str):
                    assert value.startswith(dirpath_from_ckpt)
                    new_value = self.dirpath + value.removeprefix(dirpath_from_ckpt)
                else:
                    assert isinstance(value, dict)
                    assert all(k.startswith(dirpath_from_ckpt) for k in value)
                    new_value = {
                        self.dirpath + k.removeprefix(dirpath_from_ckpt): v
                        for k, v in value.items()
                    }

                logger.info(f"Updated {key} from {value} to {new_value}")
                state_dict[key] = new_value

        super().load_state_dict(state_dict)


class UnfreezeCallback(Callback):
    """Callback to unfreeze pretrained parameters after warmup epochs."""

    def __init__(self, frozen_param_names: set[str], warmup_epochs: int) -> None:
        self.frozen_param_names = frozen_param_names
        self.warmup_epochs = warmup_epochs
        self._unfrozen = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._unfrozen:
            return

        if trainer.current_epoch >= self.warmup_epochs:
            unfrozen_count = 0
            for name, param in pl_module.named_parameters():
                if name in self.frozen_param_names:
                    param.requires_grad = True
                    unfrozen_count += param.numel()

            logger.info(
                f"Epoch {trainer.current_epoch}: Unfroze {unfrozen_count:,} pretrained parameters"
            )
            self._unfrozen = True
