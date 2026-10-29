from dataclasses import dataclass

from omegaconf import MISSING
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from syntheseus.reaction_prediction.utils.misc import cpu_count

from retrochimera.data.smiles_tokenizer import Tokenizer
from retrochimera.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BuildTokenizerConfig:
    """Config for building a tokenizer for training SMILES-based models."""

    data_dir: str = MISSING  # Directory containing raw data
    output_vocab_path: str = MISSING  # Path for saving the resulting vocabulary file
    num_processes: int = cpu_count()  # Number of parallel processes to use


def main() -> None:
    config: BuildTokenizerConfig = cli_get_config(argv=None, config_cls=BuildTokenizerConfig)
    logger.info(f"Building a tokenizer with the following config: {config}")

    tokenizer = Tokenizer.from_reactions(
        reaction_data_path=config.data_dir, num_processes=config.num_processes
    )
    tokenizer.save_vocab(vocab_path=config.output_vocab_path)


if __name__ == "__main__":
    main()
