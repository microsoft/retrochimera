import abc
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, Union

import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from retrochimera.data.dataset import DataFold
from retrochimera.utils.misc import load_from_h5py

ProcessedSampleType = TypeVar("ProcessedSampleType")


class AbstractProcessedDataset(Dataset[ProcessedSampleType], Generic[ProcessedSampleType], abc.ABC):
    def __init__(self, h5_path: Union[str, Path], fold: DataFold):
        self.h5_path = h5_path
        self.fold = fold


class ProcessedDataset(AbstractProcessedDataset[ProcessedSampleType], Generic[ProcessedSampleType]):
    def __init__(self, h5_path: Union[str, Path], fold: DataFold):
        super().__init__(h5_path, fold)

        self._data = None

        with h5py.File(self.h5_path, mode="r", libver="latest") as f:
            self._len = len(f[self.fold.value])

    def __getitem__(self, idx: int) -> ProcessedSampleType:
        # Dataloader does not work with hdf5 when `num_workers > 1`. Solution below comes from
        # https://github.com/pytorch/pytorch/issues/11929.
        if self._data is None:
            self._data = h5py.File(self.h5_path, mode="r", libver="latest")[self.fold.value]

        assert self._data is not None
        return load_from_h5py(self._data[str(idx)])

    def __len__(self) -> int:
        return self._len


class ProcessedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path: Union[str, Path],
        data_loader_kwargs: Optional[dict[str, Any]] = None,
        processed_dataset_cls: type[AbstractProcessedDataset] = ProcessedDataset,
    ):
        super().__init__()
        self.prepare_data_per_node = False

        self._h5_path = h5_path
        self._data_loader_kwargs = data_loader_kwargs or {}
        self.processed_dataset_cls = processed_dataset_cls

    def _dataloader(self, fold: DataFold, shuffle: bool) -> DataLoader:
        kwargs = {"shuffle": shuffle} | self._data_loader_kwargs
        return DataLoader(
            dataset=self.processed_dataset_cls(h5_path=self._h5_path, fold=fold), **kwargs
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(fold=DataFold.TRAIN, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(fold=DataFold.VALIDATION, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(fold=DataFold.TEST, shuffle=False)
