import dataclasses
from abc import abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from rdkit import Chem
from syntheseus import Bag, Molecule
from syntheseus.interface.molecule import SMILES_SEPARATOR
from syntheseus.interface.reaction import REACTION_SEPARATOR
from syntheseus.reaction_prediction.chem.utils import ATOM_MAPPING_PROP_NAME, remove_atom_mapping
from syntheseus.reaction_prediction.data.reaction_sample import ReactionSample
from syntheseus.reaction_prediction.utils.misc import parallelize
from tqdm import tqdm

from retrochimera.chem.utils import get_all_atom_mapping_nums, get_mapped_atoms
from retrochimera.utils.logging import get_logger

logger = get_logger(__name__)


def parse_sample(sample: tuple[str, dict[str, Any]]) -> Optional[ReactionSample]:
    """Parse a single sample (reaction SMILES, with its metadata)."""
    reaction_smiles, metadata = sample

    return ReactionSample.from_reaction_smiles(
        reaction_smiles=reaction_smiles, mapped=True, metadata=metadata
    )


def parse_samples(
    samples: list[str], num_processes: int, source: Optional[str] = None
) -> list[ReactionSample]:
    """Parse a list of samples (input strings) in parallel.

    Args:
        samples: List of input strings to parse.
        num_processes: Number of parallel processes to use.
        source: An optional tag to include into the metadata.

    Returns:
        List of samples (`ReactionSample`), excluding those that could not be parsed. Each parsed
        sample includes metadata such as the original string, `source`, and its index in `samples`.
    """
    # Parse samples in parallel; turn off chunking as we store all outputs in memory anyway. Here,
    # we first extract reaction SMILES from input strings (handling Pistachio-style format that
    # contains extra info); this is a no-op for other formats that only have the reaction SMILES.
    results = tqdm(
        parallelize(
            parse_sample,
            (
                (
                    sample.split("\t")[0].split(" ")[0],
                    {"original_str": sample, "source": source, "idx_in_source": idx},
                )
                for idx, sample in enumerate(samples)
            ),
            num_processes=num_processes,
            num_chunks_per_process_per_segment=None,
        ),
        total=len(samples),
    )
    return [result for result in results if result is not None]


class ProcessingStep:
    def __init__(self, name: str, output_dir: Path) -> None:
        self._name = name
        self._output_dir = output_dir
        self._logs_pending: str = ""
        self._samples_saved: dict[str, list[str]] = defaultdict(list)
        self._sample_counts: Counter[str] = Counter()

    def save_sample(
        self, sample: ReactionSample, prefix: str = "filtered", suffix: str = ""
    ) -> None:
        self._samples_saved[f"{prefix}_by_step_{self._name}{suffix}"].append(
            sample.metadata["original_str"]  # type: ignore[typeddict-item]
        )

    def process_samples(self, samples: Iterable[ReactionSample]) -> Iterable[ReactionSample]:
        num_samples = 0
        for sample in samples:
            processed_sample = self(sample)
            if processed_sample is not None:
                num_samples += 1
                yield processed_sample

        self._logs_pending += f"\n{'=' * 30} Step {self._name} {'=' * 30}\n"

        for key, count in self._sample_counts.items():
            self._logs_pending += f"Counted {count} samples {key}\n"

        for key, original_strings in self._samples_saved.items():
            filename = f"{key}.txt"
            self._logs_pending += f"Saving {len(original_strings)} samples under {filename}\n"

            with open(self._output_dir / filename, "w") as f:
                for original_string in original_strings:
                    f.write(str(original_string) + "\n")

        self.flush_additional_data()
        self._logs_pending += f"Left with {num_samples} samples\n"

        logger.info(self._logs_pending)

        self._logs_pending = ""
        self._samples_saved = defaultdict(list)
        self._sample_counts = Counter()

    def flush_additional_data(self) -> None:
        """Flush any extra stats or metadata after processing a stream of samples."""
        pass

    @abstractmethod
    def __call__(self, sample: ReactionSample) -> Optional[ReactionSample]:
        """Process a sample and return its (repaired) version or `None` if should be discarded."""
        pass


def replace_reaction_smiles(sample: ReactionSample, new_reaction_smiles: str) -> ReactionSample:
    assert sample.mapped_reaction_smiles is not None

    old_str: str = sample.metadata.get("repaired_str", sample.metadata["original_str"])  # type: ignore
    new_str = old_str.replace(sample.mapped_reaction_smiles, new_reaction_smiles)

    sample = dataclasses.replace(sample, mapped_reaction_smiles=new_reaction_smiles)
    sample.metadata["repaired_str"] = new_str  # type: ignore[typeddict-item]

    return sample


class NumReactantsProcessingStep(ProcessingStep):
    """Removes samples with too many reactants."""

    def __init__(self, max_reactants_num: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_num_reactants = max_reactants_num

    def __call__(self, sample: ReactionSample) -> Optional[ReactionSample]:
        if len(sample.reactants) > self._max_num_reactants:
            self.save_sample(sample)
            return None

        return sample


class OneMainProductProcessingStep(ProcessingStep):
    """Tries to select main product for each sample and removes if not possible."""

    def __init__(
        self, min_product_atoms: int, common_products: set[Molecule], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._min_product_atoms = min_product_atoms
        self._common_products = common_products

    def __call__(self, sample: ReactionSample) -> Optional[ReactionSample]:
        main_products = []
        for product in sample.products:
            if (
                product not in self._common_products
                and product.rdkit_mol.GetNumAtoms() >= self._min_product_atoms
            ):
                main_products.append(product)

        assert len(sample.products) > 0
        if len(sample.products) == 1:
            self._sample_counts["originally with 1 product"] += 1
        elif len(sample.products) == 2:
            self._sample_counts["originally with 2 products"] += 1
        else:
            self._sample_counts["originally with 3 or more products"] += 1

        if len(main_products) != 1:  # keep samples where there is exactly one main product
            if len(sample.products) == 1:
                self.save_sample(sample, suffix="_originally_with_1_product")
            elif len(sample.products) == 2:
                self.save_sample(sample, suffix="_originally_with_2_products")
            else:
                self.save_sample(sample, suffix="_originally_with_3_or_more_products")
            return None

        [main_product] = main_products

        assert sample.mapped_reaction_smiles is not None
        mapped_reactants, reagents, mapped_products = sample.mapped_reaction_smiles.split(
            REACTION_SEPARATOR
        )

        for mapped_product in mapped_products.split(SMILES_SEPARATOR):
            if Molecule(remove_atom_mapping(mapped_product)) == main_product:
                mapped_main_product = mapped_product
                break
        else:
            assert False, "Could not extract mapped main product SMILES"

        return replace_reaction_smiles(
            dataclasses.replace(sample, products=Bag([main_product])),
            new_reaction_smiles=REACTION_SEPARATOR.join(
                [mapped_reactants, reagents, mapped_main_product]
            ),
        )


class NumAtomsProcessingStep(ProcessingStep):
    """Removes samples with large product or large ratio between reactants and product."""

    def __init__(
        self, max_product_atoms: int, max_reactants_to_product_ratio: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._max_product_atoms = max_product_atoms
        self._max_reactants_to_product_ratio = max_reactants_to_product_ratio

    def __call__(self, sample: ReactionSample) -> Optional[ReactionSample]:
        reactant_atoms = sum(r.rdkit_mol.GetNumAtoms() for r in sample.reactants)
        product_atoms = sum(p.rdkit_mol.GetNumAtoms() for p in sample.products)

        if product_atoms > self._max_product_atoms:
            self.save_sample(sample, suffix="_product_too_large")
            return None

        if reactant_atoms / product_atoms > self._max_reactants_to_product_ratio:
            self.save_sample(sample, suffix="_reactant_to_product_ratio_too_large")
            return None

        return sample


class ProductAmongReactantsProcessingStep(ProcessingStep):
    """Removes samples where a product molecule is one of the reactants."""

    def __call__(self, sample: ReactionSample) -> Optional[ReactionSample]:
        if set(sample.reactants) & set(sample.products):
            self.save_sample(sample)
            return None

        return sample


class AtomMappingProcessingStep(ProcessingStep):
    """Tries to fix atom mapping information and removes the sample if not possible."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reactants_removed: set[str] = set()

    def flush_additional_data(self) -> None:
        self._logs_pending += f"Removed {len(self._reactants_removed)} unmapped reactant sets\n"

        with open(self._output_dir / f"reactants_removed_by_step_{self._name}.txt", "w") as f:
            for reactant in self._reactants_removed:
                f.write(reactant + "\n")

        self._reactants_removed = set()

    def __call__(self, sample: ReactionSample) -> Optional[ReactionSample]:
        assert sample.mapped_reaction_smiles is not None
        mapped_reactants, reagents, mapped_product = sample.mapped_reaction_smiles.split(
            REACTION_SEPARATOR
        )

        reactants = [
            Molecule(reactant_smiles, canonicalize=False)
            for reactant_smiles in mapped_reactants.split(SMILES_SEPARATOR)
        ]
        product = Molecule(mapped_product, canonicalize=False)

        # Create sets of atom mapping numbers for reactants and product.
        atom_mapping_numbers_in_raw_reactants = set()
        for reactant in reactants:
            atom_mapping_numbers_in_raw_reactants |= set(
                get_all_atom_mapping_nums(reactant.rdkit_mol)
            )

        atom_mapping_numbers_in_raw_main_product = set(get_all_atom_mapping_nums(product.rdkit_mol))

        # Refine samples by clearing the atom mapping numbers that appear on only one side.
        repaired_flag = False
        atom_mapping_numbers_in_each_repaired_reactant = []
        atom_mapping_numbers_in_repaired_reactants = set()
        num_mapped_atoms_in_repaired_reactants = 0
        for reactant in reactants:
            reactant_atom_mapping_numbers = set()
            for reactant_atom in get_mapped_atoms(reactant.rdkit_mol):
                if reactant_atom.GetAtomMapNum() not in atom_mapping_numbers_in_raw_main_product:
                    reactant_atom.ClearProp(ATOM_MAPPING_PROP_NAME)
                    repaired_flag = True
                else:
                    reactant_atom_mapping_numbers.add(reactant_atom.GetAtomMapNum())
                    atom_mapping_numbers_in_repaired_reactants.add(reactant_atom.GetAtomMapNum())
                    num_mapped_atoms_in_repaired_reactants += 1
            atom_mapping_numbers_in_each_repaired_reactant.append(reactant_atom_mapping_numbers)

        atom_mapping_numbers_in_repaired_main_product = set()
        num_mapped_atoms_in_repaired_main_product = 0
        for product_atom in get_mapped_atoms(product.rdkit_mol):
            if product_atom.GetAtomMapNum() not in atom_mapping_numbers_in_raw_reactants:
                product_atom.ClearProp(ATOM_MAPPING_PROP_NAME)
                repaired_flag = True
            else:
                atom_mapping_numbers_in_repaired_main_product.add(product_atom.GetAtomMapNum())
                num_mapped_atoms_in_repaired_main_product += 1

        if repaired_flag:
            self.save_sample(sample, prefix="repaired", suffix="_removed_lone_atom_mapping_numbers")

        # Drop samples with double mapped atoms in the main product.
        if (
            len(atom_mapping_numbers_in_repaired_main_product)
            != num_mapped_atoms_in_repaired_main_product
        ):
            self.save_sample(sample, suffix="_double_mapped_product")
            return None

        # Drop samples with double mapped atoms in the reactants.
        if (
            len(atom_mapping_numbers_in_repaired_reactants)
            != num_mapped_atoms_in_repaired_reactants
        ):
            self.save_sample(sample, suffix="_double_mapped_reactants")
            return None

        # Drop samples without atom mapping numbers.
        if (
            len(atom_mapping_numbers_in_repaired_main_product) == 0
            or len(atom_mapping_numbers_in_repaired_reactants) == 0
        ):
            self.save_sample(sample, suffix="_no_atom_mapping")
            return None

        # Refine samples by removing reactants that don't contribute atoms to the product.
        refined_mapped_reactants_smiles_list: list[str] = []
        removed_reactants: list[str] = []
        remaining_unmapped_reactant_mols: list[Molecule] = []
        for i, (reactant, unmapped_reactant_mol) in enumerate(zip(reactants, sample.reactants)):
            if len(
                atom_mapping_numbers_in_each_repaired_reactant[i]
                - atom_mapping_numbers_in_repaired_main_product
            ) < len(atom_mapping_numbers_in_each_repaired_reactant[i]):
                refined_mapped_reactants_smiles_list.append(Chem.MolToSmiles(reactant.rdkit_mol))
                remaining_unmapped_reactant_mols.append(unmapped_reactant_mol)
            else:
                removed_reactants.append(Chem.MolToSmiles(reactant.rdkit_mol))

        refined_mapped_reactants_smiles = SMILES_SEPARATOR.join(
            refined_mapped_reactants_smiles_list
        )

        if removed_reactants:
            self.save_sample(sample, prefix="repaired", suffix="_removed_unmapped_reactants")
            self._reactants_removed.add(SMILES_SEPARATOR.join(removed_reactants))

        return replace_reaction_smiles(
            dataclasses.replace(sample, reactants=Bag(remaining_unmapped_reactant_mols)),
            new_reaction_smiles=f"{refined_mapped_reactants_smiles}>{reagents}>{Chem.MolToSmiles(product.rdkit_mol)}",
        )
