from pathlib import Path
from typing import Union

import pandas as pd

from retrochimera.data.dataset import DataFold


def load_raw_reactions_file(path: Union[str, Path]) -> list[str]:
    if str(path).endswith(".csv"):
        # USPTO-style format.
        data = pd.read_csv(path)
        column_name = "reactants>reagents>production"

        if column_name not in data.columns:
            raise ValueError(f"Column {column_name} not found in {path}")

        return data[column_name].values
    elif str(path).endswith(".smi"):
        # Generic format with one reaction SMILES per line, with optionally extra into after \t.
        with open(path) as f:
            return [line.rstrip() for line in f]
    else:
        raise ValueError(f"Unrecognized file extension in {path}")


def load_raw_reactions_files(dir: Union[str, Path]) -> dict[DataFold, list[str]]:
    """Load reactions for each fold from files in the given directory.

    Args:
        dir: Directory containing the reaction files.

    Returns:
        Dictionary mapping folds to lists of reactions.
    """
    fold_to_path: dict[DataFold, Path] = {}
    for fold in DataFold:
        matching_paths = sum(
            [list(Path(dir).glob(f"*{fold.value}*.{ext}")) for ext in ["csv", "smi"]], []
        )

        if not matching_paths:
            raise ValueError(f"No files found for fold {fold.value}")

        if len(matching_paths) > 1:
            raise ValueError(
                f"Multiple files found for fold {fold.value}: {[str(f) for f in matching_paths]}"
            )

        fold_to_path[fold] = matching_paths[0]

    if len(set(path.suffix for path in fold_to_path.values())) > 1:
        raise ValueError("Files for different folds have inconsistent formats")

    return {fold: load_raw_reactions_file(path) for fold, path in fold_to_path.items()}
