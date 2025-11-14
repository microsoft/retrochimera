from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from retrochimera.cli.eval import BackwardModelConfig
from syntheseus.cli import search


@dataclass
class SearchConfig(BackwardModelConfig, search.BaseSearchConfig):
    """Config for running search for given search targets."""

    pass


def main(argv: Optional[list[str]]) -> None:
    search.main(argv, config_cls=SearchConfig)


if __name__ == "__main__":
    main(argv=None)
