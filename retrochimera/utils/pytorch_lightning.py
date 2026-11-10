import math
import random
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl
import torch
from more_itertools import batched
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import isolate_rng

from syntheseus.interface.models import BackwardReactionModel
from syntheseus.reaction_prediction.data.dataset import DataFold
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.misc import cpu_count

from retrochimera.data.dataset import SingleProductReactionDataset
from retrochimera.models.lightning import AbstractModel
from retrochimera.utils.logging import get_logger

logger = get_logger(__name__)


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


class ReactionMetricsCallback(Callback):
    """Logs end-to-end top-k accuracy and MRR on raw validation and test reactions.

    Mirrors `cli/eval.py` (syntheseus `eval_single_step`), but runs the live model through its
    inference wrapper on a fixed random subset of each evaluation fold, sharded across ranks.
    """

    TOP_K = (1, 3, 5, 10, 50)
    NUM_RESULTS = 100  # Same as the `num_top_results` default in `cli/eval.py`
    MAX_NUM_SAMPLES = 5000  # todo: remove and just use a smaller dataset

    def __init__(self, data_dir: Union[str, Path], batch_size: int = 32) -> None:
        self._data_dir = data_dir
        self._batch_size = batch_size

        self._samples: list[ReactionSample] = []
        self._inference_model: Optional[BackwardReactionModel] = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if stage == "predict":
            return
        # Lightning cannot detect that these reaction metrics are already globally reduced.
        warnings.filterwarnings("ignore", message="It is recommended to use .*sync_dist=True")
        # Share half the host CPUs across local ranks, leaving the rest for data loading.
        self._num_processes = min(16, max(1, cpu_count() // (2 * trainer.num_devices)))
        dataset = SingleProductReactionDataset(
            self._data_dir, sample_cls=ReactionSample, num_processes=self._num_processes
        )
        fold = DataFold.TEST if stage == "test" else DataFold.VALIDATION
        samples = list(dataset[fold])
        if len(samples) > self.MAX_NUM_SAMPLES:
            samples = random.Random(0).sample(samples, self.MAX_NUM_SAMPLES)
        self._samples = samples[trainer.global_rank :: trainer.world_size]
        logger.info(f"Computing reaction metrics on {len(samples)} {fold.value} reactions")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.sanity_checking:
            self._log_reaction_metrics(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        top_k_acc = self._log_reaction_metrics(trainer, pl_module, "test")
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_table(
                key="test_top_k_acc",
                columns=["k", "acc"],
                data=[[k, acc.item()] for k, acc in zip(self.TOP_K, top_k_acc)],
            )

    def _log_reaction_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, step_name: str
    ) -> list[torch.Tensor]:
        assert isinstance(pl_module, AbstractModel)
        ranks: list[float] = []
        # Fix augmentations across epochs without perturbing the training RNG streams.
        # Epoch-end hooks run outside autocast; match the precision of cached evaluation encodings.
        with isolate_rng(include_cuda=False), trainer.precision_plugin.forward_context():
            random.seed(0)
            if self._inference_model is None:
                self._inference_model = pl_module.build_inference_model(self._num_processes)
            for batch in batched(self._samples, self._batch_size):
                products = [product for sample in batch for product in sample.products]
                predictions = self._inference_model(products, num_results=self.NUM_RESULTS)
                for sample, preds in zip(batch, predictions):
                    matches = (i for i, p in enumerate(preds, 1) if p.reactants == sample.reactants)
                    ranks.append(next(matches, math.inf))

        # Sum over ranks before normalizing, as shards may differ in size.
        stats = torch.tensor(
            [sum(r <= k for r in ranks) for k in self.TOP_K]
            + [sum(1 / r for r in ranks), len(ranks)],
            dtype=torch.float32,
            device=pl_module.device,
        )
        stats = trainer.strategy.reduce(stats, reduce_op="sum")
        *top_k_acc, mrr, _ = stats / stats[-1].clamp(min=1)

        metrics = {f"{step_name}_top_{k}_acc": value for k, value in zip(self.TOP_K, top_k_acc)}
        pl_module.log_dict(metrics | {f"{step_name}_mrr": mrr}, sync_dist=False)
        return top_k_acc

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self._inference_model = None
