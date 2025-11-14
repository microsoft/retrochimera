from collections import defaultdict
from dataclasses import dataclass

from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from syntheseus.reaction_prediction.utils.misc import cpu_count

from retrochimera.data.processing import parse_samples
from retrochimera.data.utils import load_raw_reactions_file
from retrochimera.utils.logging import get_logger
from retrochimera.utils.misc import silence_rdkit_warnings

logger = get_logger(__name__)


@dataclass
class DeduplicateDatasetConfig:
    """Config for deduplicating a given dataset."""

    input_path: str  # Path to raw dataset, either Pistachio format (.smi) or USPTO format (.csv).
    output_path: str = "pistachio_deduplicated.smi"  # Path to a *.smi file for saving the output.
    num_processes: int = cpu_count()  # Number of processes to use for processing the SMILES.


def deduplicate_dataset(config: DeduplicateDatasetConfig) -> None:
    path = config.input_path
    raw_data = load_raw_reactions_file(path)
    unique_samples: dict[str, set[str]] = defaultdict(set)

    logger.info(f"Processing {len(raw_data)} samples")
    data = parse_samples(raw_data, num_processes=config.num_processes)
    logger.info(f"Parsed {len(data)} valid samples\n")

    for d in data:
        unique_samples[d.reaction_smiles].add(d.metadata["original_str"])  # type: ignore
    logger.info(f"Number of unique reaction SMILES: {len(unique_samples)}")

    # Save unique reaction SMILES.
    with open(config.output_path, "wt") as f:
        for original_str in unique_samples.values():
            f.write(f"{original_str.pop()}\n")


def main() -> None:
    silence_rdkit_warnings()

    config: DeduplicateDatasetConfig = cli_get_config(
        argv=None, config_cls=DeduplicateDatasetConfig
    )

    logger.info(f"Removing duplicates with the following config: {config}")

    deduplicate_dataset(config)


if __name__ == "__main__":
    main()
