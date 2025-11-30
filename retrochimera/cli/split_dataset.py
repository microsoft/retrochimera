"""Script for running initial processing of a dataset and splitting it into folds.

The processing hyperparameters can be overriden, but by default they map to the following settings:
1. Remove reactions with more than 4 reactants.
2. Remove reactions with zero or several main products (defined as those having at least 5 atoms and
   not being among "common products", i.e. products appearing in the data at least 1000 times).
3. Remove reactions with product having more than 100 atoms or with ratio between number of reactant
   and product atoms being more than 20.
4. Remove reactions where the main product is also present among the reactants.
5. Refine reactions by removing atom mapping numbers appearing only on one side, and dropping
   reactants with no mapped atoms. Remove reactions with an invalid mapping or no mapping left.

Split into folds is random after grouping by product. Optionally, a `dataset_to_follow_dir` argument
allows for splitting in accordance with an external (already split) dataset.
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from syntheseus.reaction_prediction.utils.misc import cpu_count, set_random_seed
from tqdm import tqdm

from retrochimera.data.dataset import DataFold
from retrochimera.data.processing import (
    AtomMappingProcessingStep,
    NumAtomsProcessingStep,
    NumReactantsProcessingStep,
    OneMainProductProcessingStep,
    ProductAmongReactantsProcessingStep,
    parse_samples,
)
from retrochimera.data.utils import load_raw_reactions_file, load_raw_reactions_files
from retrochimera.utils.logging import get_logger
from retrochimera.utils.misc import silence_rdkit_warnings

logger = get_logger(__name__)


@dataclass
class SplitDatasetConfig:
    """Config for splitting a given dataset."""

    input_path: str  # path to the dataset to be split (either Pistachio or USPTO format)
    output_dir: str  # output directory for saving the resulting folds
    num_processes: int = cpu_count()  # number of processes to use for processing the SMILES

    # Directory containing the folds of a pre-split datset. These are used to define products that
    # should preferrably be placed into the train/validation/test set. This setting is useful if one
    # wants to train on one dataset and validate on another pre-split one while minimizing leakage.
    dataset_to_follow_dir: Optional[str] = None

    # Fraction of samples to place into each fold.
    val_frac: float = 0.05
    test_frac: float = 0.05

    # Hyperparamters for filtering some bad samples.
    max_reactants_num: int = 4
    min_product_atoms: int = 5
    max_product_occurrences: int = 1000
    max_product_atoms: int = 100
    max_reactants_to_product_ratio: int = 20


def filter_dataset(data: list[ReactionSample], config: SplitDatasetConfig) -> list[ReactionSample]:
    final_data = []

    changed_samples_dir = Path(config.output_dir) / "changed_samples"
    changed_samples_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Drop samples with too many reactants.
    data = list(
        NumReactantsProcessingStep(
            name="1", output_dir=changed_samples_dir, max_reactants_num=config.max_reactants_num
        ).process_samples(data)
    )

    # Create and save occurrence count of product molecules.
    common_products: Counter = Counter()
    for sample in data:
        common_products.update(sample.products)

    common_products = Counter(
        {p: cnt for p, cnt in common_products.items() if cnt >= config.max_product_occurrences}
    )

    with open(changed_samples_dir / "common_products.txt", "w") as f:
        for product, count in common_products.most_common():
            f.write(f"{product.smiles} {count}\n")

    # Step 2: Drop samples with more than one main product.
    # A non-main product either has fewer than `min_product_atoms` atoms or occurs at least
    # `max_product_occurrences` times (over the entire dataset).
    one_main_product_step = OneMainProductProcessingStep(
        name="2",
        output_dir=changed_samples_dir,
        min_product_atoms=config.min_product_atoms,
        common_products=set(common_products),
    )

    # Step 3: Drop samples with large product or large ratio between reactants and product.
    num_atoms_step = NumAtomsProcessingStep(
        name="3",
        output_dir=changed_samples_dir,
        max_product_atoms=config.max_product_atoms,
        max_reactants_to_product_ratio=config.max_reactants_to_product_ratio,
    )

    # Step 4: Drop samples where the main product is also present among the reactants.
    product_among_reactants_step = ProductAmongReactantsProcessingStep(
        name="4", output_dir=changed_samples_dir
    )

    # Step 5: Fix samples with minor mapping issues and drop those that cannot be fixed.
    atom_mapping_step = AtomMappingProcessingStep(name="5", output_dir=changed_samples_dir)

    data_iterable: Iterable[ReactionSample] = data
    for step in [
        one_main_product_step,
        num_atoms_step,
        product_among_reactants_step,
        atom_mapping_step,
    ]:
        data_iterable = step.process_samples(data_iterable)

    for sample in tqdm(data_iterable, total=len(data)):
        # Clear rdkit molecules to save memory.
        for product in sample.products:
            if "rdkit_mol" in product.metadata:
                del product.metadata["rdkit_mol"]
        for reactant in sample.reactants:
            if "rdkit_mol" in reactant.metadata:
                del reactant.metadata["rdkit_mol"]

        if "repaired_str" in sample.metadata:
            sample.metadata["original_str"] = sample.metadata["repaired_str"]  # type: ignore[typeddict-item]

        final_data.append(sample)

    return final_data


def split_dataset(config: SplitDatasetConfig) -> None:
    raw_data = load_raw_reactions_file(config.input_path)
    logger.info(f"Read {len(raw_data)} raw samples to split")

    data = parse_samples(raw_data, num_processes=config.num_processes)
    logger.info(f"Left with {len(data)} valid samples\n")

    data = filter_dataset(data, config)
    logger.info(f"Left with {len(data)} samples after filtering")

    fold_target_size: dict[DataFold, int] = {
        DataFold.VALIDATION: int(config.val_frac * len(data)),
        DataFold.TEST: int(config.test_frac * len(data)),
    }
    fold_target_size[DataFold.TRAIN] = len(data) - sum(fold_target_size.values())

    fold_sizes_joined = "\n".join([f"{fold.value}: {fold_target_size[fold]}" for fold in DataFold])
    logger.info(f"Fold target sizes:\n{fold_sizes_joined}")

    assert min(fold_target_size.values()) >= 0, "Target fold sizes must be positive"

    data_grouped: dict[Bag[Molecule], list[ReactionSample]] = defaultdict(list)
    num_products_count: dict[int, int] = defaultdict(int)
    num_products_example: dict[int, str] = {}

    for d in data:
        data_grouped[d.products].append(d)

        num_products = len(d.products)
        if num_products > 1:
            num_products_count[num_products] += 1

            if num_products not in num_products_example:
                assert d.mapped_reaction_smiles is not None
                num_products_example[num_products] = d.mapped_reaction_smiles

    for num_products, count in sorted(num_products_count.items()):
        logger.warning(
            f"Found {count} samples with {num_products} products "
            f"(example reaction SMILES: {num_products_example[num_products]})"
        )

    logger.info(f"Got {len(data_grouped)} groups after grouping by product\n")

    fold_to_follow_to_products: dict[DataFold, set[Molecule]] = defaultdict(set)
    if config.dataset_to_follow_dir is not None:
        raw_data_to_follow = load_raw_reactions_files(config.dataset_to_follow_dir)

        for fold in DataFold:
            logger.info(
                f"Read {len(raw_data_to_follow[fold])} samples to follow for fold {fold.value}"
            )

            data_to_follow = parse_samples(
                raw_data_to_follow[fold], num_processes=config.num_processes
            )
            logger.info(f"Left with {len(data_to_follow)} valid samples")

            for d in data_to_follow:
                if len(d.products) > 1:
                    raise ValueError(
                        "Dataset to follow has side-products, which can cause weird results."
                    )

                [product] = list(d.products)
                fold_to_follow_to_products[fold].add(product)

    data_split: dict[DataFold, list[ReactionSample]] = defaultdict(list)
    data_groups_left: list[tuple[Bag[Molecule], list[ReactionSample]]] = []

    for products, d_group in data_grouped.items():
        # Go through folds in reversed order. If one side-product occurrs in the pre-split dataset
        # in test, and another one in train, then the sample goes into test fold of the new dataset.
        for fold in reversed(DataFold):
            if any(product in fold_to_follow_to_products[fold] for product in products):
                data_split[fold].extend(d_group)
                break
        else:
            data_groups_left.append((products, d_group))

    if config.dataset_to_follow_dir is not None:
        fold_sizes_joined = "\n".join(
            [f"{fold.value}: {len(data_split[fold])}" for fold in DataFold]
        )
        logger.info(f"Fold sizes after including the overlapping samples:\n{fold_sizes_joined}")
    else:
        assert not data_split

    if any(len(data_split[fold]) > fold_target_size[fold] for fold in DataFold):
        raise ValueError(
            "After including overlapping sizes some folds are already bigger than desired"
        )

    random.shuffle(data_groups_left)

    folds = list(DataFold)
    current_fold = DataFold.TRAIN

    for _, d_group in data_groups_left:
        if (
            current_fold != DataFold.TEST
            and len(data_split[current_fold]) + len(d_group) > fold_target_size[current_fold]
        ):
            # If the current fold is full progress to the next one.
            current_fold = folds[folds.index(current_fold) + 1]

        data_split[current_fold].extend(d_group)

    fold_sizes_joined = "\n".join(
        [
            f"{fold.value}: {len(data_split[fold])} (target {fold_target_size[fold]})"
            for fold in DataFold
        ]
    )
    logger.info(f"Final fold sizes:\n{fold_sizes_joined}")

    logger.info(f"Saving data under {config.output_dir}")
    for fold, datapoints in data_split.items():
        random.shuffle(datapoints)

        with open(Path(config.output_dir) / f"pista_{fold.value}.smi", "wt") as f:
            for datapoint in datapoints:
                original_str = datapoint.metadata["original_str"]  # type: ignore[typeddict-item]
                f.write(f"{original_str}\n")


def main() -> None:
    set_random_seed(0)
    silence_rdkit_warnings()

    config: SplitDatasetConfig = cli_get_config(argv=None, config_cls=SplitDatasetConfig)
    logger.info(f"Splitting dataset with the following config: {config}")

    split_dataset(config)


if __name__ == "__main__":
    main()
