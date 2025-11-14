from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import torch
from tqdm import tqdm

from retrochimera.data.dataset import DataFold, ReactionDataset
from retrochimera.models.lightning import AbstractModel
from retrochimera.models.smiles_transformer import SmilesTransformerModel
from retrochimera.utils.logging import get_logger
from retrochimera.utils.misc import save_into_h5py

logger = get_logger(__name__)


def preprocess_and_save(
    save_path: Union[str, Path],
    dataset: ReactionDataset,
    model: AbstractModel,
    num_processes: int,
    max_num_samples: Optional[int] = None,
) -> None:
    """Preprocesses raw data for any model type and saves as one HDF5 file per fold."""
    preprocess_fn = partial(model.preprocess, num_processes=num_processes)

    logger.info(f"Starting to save data under {save_path}")
    with h5py.File(save_path, mode="w", libver="latest") as file:
        for fold in DataFold:
            logger.info(f"Saving fold {fold.value}")

            raw_data_iterator = dataset[fold]
            num_samples = dataset.get_num_samples(fold)

            if max_num_samples is not None:
                logger.info(f"Truncating the fold to {max_num_samples} samples")

                raw_data_iterator = islice(raw_data_iterator, max_num_samples)
                num_samples = min(num_samples, max_num_samples)

            group = file.create_group(fold.value)
            for idx, data in enumerate(preprocess_fn(tqdm(raw_data_iterator, total=num_samples))):
                save_into_h5py(group, key=str(idx), data=data)

            if isinstance(model, SmilesTransformerModel):
                model.tokenizer.print_unknown_tokens()


def average_checkpoints(input_paths: list[Union[str, Path]], output_path: Union[str, Path]) -> None:
    """Compute an averaged checkpoint by uniformly averaging out a given list of checkpoints.

    Args:
        input_paths: Paths to the checkpoints to use for averaging.
        output_path: Path under which to save the final combined checkpoint.
    """
    combined_checkpoint: Optional[dict[str, Any]] = None
    num_combined = 0

    for path in input_paths:
        with open(path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

            if combined_checkpoint is None:
                combined_checkpoint = checkpoint
            else:
                for key, weights in checkpoint["state_dict"].items():
                    if key.endswith(".num_batches_tracked"):
                        # Batch counter in `BatchNorm` may differ slightly accross checkpoints but
                        # it's not needed during inference.
                        continue

                    if weights.dtype.is_floating_point:
                        combined_checkpoint["state_dict"][key] += weights
                    else:
                        if (combined_checkpoint["state_dict"][key] != weights).any():
                            raise ValueError(
                                f"Checkpoints to combine differ on key {key} "
                                f"which cannot be averaged due to having type {weights.dtype}"
                            )

            num_combined += 1

    assert combined_checkpoint is not None

    if num_combined > 1:
        for key, weights in combined_checkpoint["state_dict"].items():
            if weights.dtype.is_floating_point:
                weights /= float(num_combined)

    with open(output_path, "wb") as f:
        torch.save(combined_checkpoint, f)
