from abc import abstractproperty
from pathlib import Path
from typing import Iterable, Optional, TypeVar, Union

from syntheseus.reaction_prediction.data import dataset as dataset_syntheseus
from syntheseus.reaction_prediction.data.dataset import DataFold
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample

from retrochimera.chem.rules import RuleBase

SampleType = TypeVar("SampleType", bound=ReactionSample)


class ReactionDataset(dataset_syntheseus.ReactionDataset[SampleType]):
    """Dataset holding raw reactions split into folds (together with a rulebase, if applicable).

    Depending on implementation, the reactions can either be held on disk or in memory.
    """

    @abstractproperty
    def rulebase(self) -> RuleBase:
        pass


class DiskReactionDataset(dataset_syntheseus.DiskReactionDataset[SampleType], ReactionDataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_cls: type[SampleType],
        num_processes: int = 0,
        rulebase_min_rule_support: Optional[int] = None,
        rulebase_max_num_rules: Optional[int] = None,
    ):
        super().__init__(data_dir, sample_cls=sample_cls, num_processes=num_processes)

        self._rulebase: Optional[RuleBase] = None
        self._rulebase_min_rule_support = rulebase_min_rule_support
        self._rulebase_max_num_rules = rulebase_max_num_rules

    @property
    def rulebase(self) -> RuleBase:
        if self._rulebase is None:
            self._rulebase = RuleBase.load_from_file(
                dir=self._data_dir,
                min_rule_support=self._rulebase_min_rule_support,
                max_num_rules=self._rulebase_max_num_rules,
            )

        return self._rulebase


class SingleProductReactionDataset(dataset_syntheseus.DiskReactionDataset[SampleType]):
    """Exclude multi-product reactions from validation and test folds only."""

    def __getitem__(self, fold: DataFold) -> Iterable[SampleType]:
        samples = super().__getitem__(fold)
        if fold == DataFold.TRAIN:
            return samples
        return (sample for sample in samples if len(sample.products) == 1)

    def get_num_samples(self, fold: DataFold) -> int:
        if fold == DataFold.TRAIN:
            return super().get_num_samples(fold)
        if fold not in self._num_samples:
            self._num_samples[fold] = sum(1 for _ in self[fold])
        return self._num_samples[fold]


class InMemoryReactionDataset(ReactionDataset[SampleType]):
    def __init__(
        self, samples: dict[DataFold, list[SampleType]], rulebase: Optional[RuleBase] = None
    ):
        self._samples = samples
        self._rulebase = rulebase
        self._num_samples = {fold: len(samples_list) for fold, samples_list in samples.items()}

    def __getitem__(self, fold: DataFold) -> Iterable[SampleType]:
        return self._samples.get(fold, [])

    def get_num_samples(self, fold: DataFold) -> int:
        return self._num_samples.get(fold, 0)

    @property
    def rulebase(self) -> RuleBase:
        if self._rulebase is None:
            raise ValueError("Rulebase was not provided")

        return self._rulebase
