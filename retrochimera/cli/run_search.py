from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from syntheseus.cli import search

from retrochimera.cli.eval import BackwardModelConfig


@dataclass
class SearchConfig(BackwardModelConfig, search.BaseSearchConfig):
    """Config for running search for given search targets."""

    pass


def main(argv: Optional[list[str]]) -> None:
    search.main(argv, config_cls=SearchConfig)


if __name__ == "__main__":
    main(argv=None)
