"""Script for preprocessing a raw dataset.

Running preprocessing requires knowing the model class that will be used during training; see
`train.py` for the currently available options.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import MISSING
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from syntheseus.reaction_prediction.utils.misc import cpu_count, set_random_seed

from retrochimera.cli.train import ModelClass, ModelTrainingConfig, build_model_from_config
from retrochimera.data.dataset import DataFold, DiskReactionDataset
from retrochimera.data.smiles_reaction_sample import SmilesReactionSample
from retrochimera.data.template_reaction_sample import TemplateReactionSample
from retrochimera.utils.logging import get_logger
from retrochimera.utils.training import preprocess_and_save

logger = get_logger(__name__)


@dataclass
class PreprocessConfig(ModelTrainingConfig):
    """Config for preprocessing a raw dataset."""

    data_dir: str = MISSING  # Directory containing raw data
    processed_data_dir: str = MISSING  # Directory for saving preprocessed data
    seed: int = 0  # Seed to use for all sources of randomness (Python, numpy, torch)
    max_num_samples: Optional[int] = None  # Maximum number of samples to truncate each fold to

    num_processes_preprocessing: int = cpu_count()
    rulebase_min_rule_support: Optional[int] = None
    rulebase_max_num_rules: Optional[int] = None


def main() -> None:
    config: PreprocessConfig = cli_get_config(argv=None, config_cls=PreprocessConfig)
    logger.info(f"Running preprocessing with the following config: {config}")

    set_random_seed(config.seed)

    processed_data_dir = Path(config.processed_data_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_path = processed_data_dir / "data.h5"

    if processed_data_path.exists():
        logger.info(f"Processed data file {processed_data_path} exists, nothing to do")
        return

    if config.rulebase_min_rule_support is None and config.rulebase_max_num_rules is None:
        logger.warning("Neither `rulebase_min_rule_support` nor `rulebase_max_num_rules` is set")

    reaction_sample_cls: type[ReactionSample]
    if config.model_class is ModelClass.SmilesTransformer:
        reaction_sample_cls = SmilesReactionSample
    else:
        reaction_sample_cls = TemplateReactionSample

    dataset = DiskReactionDataset(
        data_dir=config.data_dir,
        sample_cls=reaction_sample_cls,
        num_processes=config.num_processes_preprocessing,
        rulebase_min_rule_support=config.rulebase_min_rule_support,
        rulebase_max_num_rules=config.rulebase_max_num_rules,
    )

    logger.info(f"Saving the modified rulebase under {config.processed_data_dir}")
    dataset.rulebase.save_to_file(dir=processed_data_dir)

    model, _ = build_model_from_config(
        config=config, rulebase=dataset.rulebase, rulebase_dir=processed_data_dir
    )

    logger.info(
        f"Preprocessing datapoints: "
        f"{dataset.get_num_samples(DataFold.TRAIN)} train, "
        f"{dataset.get_num_samples(DataFold.VALIDATION)} validation, "
        f"{dataset.get_num_samples(DataFold.TEST)} test"
    )

    preprocess_and_save(
        save_path=processed_data_path,
        dataset=dataset,
        model=model,
        num_processes=config.num_processes_preprocessing,
        max_num_samples=config.max_num_samples,
    )


if __name__ == "__main__":
    main()
