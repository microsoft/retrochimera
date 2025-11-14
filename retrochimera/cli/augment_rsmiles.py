"""Script for augmenting reaction data with the approach used by the R-SMILES model.

This script is a modified version of `preprocessing/generate_PtoR_data.py` from
https://github.com/otori-bird/retrosynthesis/. Some of the utilities are imported from the original
script, but the main logic is cleaned up and simplified by removing features we do not need.
"""

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING
from rdkit import Chem, RDLogger
from syntheseus.reaction_prediction.data.dataset import DataFold
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from tqdm import tqdm

from retrochimera.utils.logging import get_logger
from retrochimera.utils.root_aligned import get_product_roots

RDLogger.DisableLog("rdApp.*")

logger = get_logger(__name__)


@dataclass
class AugmentRSMILESConfig:
    data_dir: str = MISSING  # Directory containing the data
    augmented_data_dir: str = MISSING  # Directory for saving the augmented data
    folds: list[DataFold] = field(
        default_factory=lambda: [DataFold.TRAIN, DataFold.VALIDATION]
    )  # Folds to augment
    augmentation: int = 1  # Number of augmentations per reaction
    seed: int = 33  # Random seed for reproducibility
    use_random_smiles: bool = False  # If set, use random SMILES instead of root-aligned


def multi_process(data: dict):
    """Multi-process function to process the data."""
    from root_aligned.preprocessing.generate_PtoR_data import (
        clear_map_canonical_smiles,
        get_cano_map_number,
        get_root_id,
    )

    reaction = data["reaction"]
    augmentation = data["augmentation"]

    reactant, _, product = reaction["mapped_reaction_smiles"].split(">")
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)

    # Check data quality.
    return_status = {
        "status": 0,
        "src_smi": [],
        "tgt_smi": [],
        "reaction": reaction,
    }

    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if len(pro_mol.GetAtoms()) == 1:
        return_status["status"] = "small_p"
    if not all([a.HasProp("molAtomMapNumber") for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"

    if return_status["status"] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if not data["use_random_smiles"]:
            product_roots = get_product_roots(
                product_atom_ids=pro_atom_map_numbers, num_augmentations=augmentation
            )

            for pro_root_atom_map in product_roots:
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [
                    list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant
                ]
                used_indices = []

                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        if map_number in rea_map_number:
                            rea_root = get_root_id(
                                Chem.MolFromSmiles(reactant[i]), root_map_number=map_number
                            )
                            rea_smi = clear_map_canonical_smiles(
                                reactant[i], canonical=True, root=rea_root
                            )
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(
                    list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1]
                )
                aligned_reactants = [item[0] for item in sorted_reactants]
                reactant_smi = ".".join(aligned_reactants)

                # Save the resulted smiles.
                return_status["src_smi"].append(pro_smi)
                return_status["tgt_smi"].append(reactant_smi)

            assert len(return_status["src_smi"]) == data["augmentation"]
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join(
                [
                    clear_map_canonical_smiles(rea)
                    for rea in reactant
                    if set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(pro_atom_map_numbers)
                ]
            )
            return_status["src_smi"].append(cano_product)
            return_status["tgt_smi"].append(cano_reactanct)

            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]

            for i in range(int(augmentation - 1)):
                pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status["src_smi"].append(pro_smi)
                return_status["tgt_smi"].append(rea_smi)

    return return_status


def preprocess(
    augmented_data_dir: str,
    reaction_list: list,
    fold: DataFold,
    augmentation: int = 1,
    use_random_smiles: bool = False,
) -> None:
    """Preprocess reaction data to extract graph adjacency matrix and features."""

    if not os.path.exists(augmented_data_dir):
        os.makedirs(augmented_data_dir)

    data = [
        {
            "reaction": reaction_list[i],
            "augmentation": augmentation,
            "use_random_smiles": use_random_smiles,
        }
        for i in range(len(reaction_list))
    ]

    skip_dict = {
        "invalid_p": 0,
        "invalid_r": 0,
        "small_p": 0,
        "small_r": 0,
        "error_mapping": 0,
        "error_mapping_p": 0,
        "empty_p": 0,
        "empty_r": 0,
    }

    results = [multi_process(sample) for sample in tqdm(data)]

    with open(os.path.join(augmented_data_dir, f"{fold.value}.jsonl"), "w") as f:
        for result in tqdm(results):
            if result["status"] != 0:
                skip_dict[result["status"]] += 1
                continue

            for src, tgt in zip(result["src_smi"], result["tgt_smi"]):
                reactants_smiles = []
                for reactant in tgt.split("."):
                    reactants_smiles.append({"smiles": reactant})
                assert "." not in src
                products_smiles = [{"smiles": src}]

                updated_reaction = result["reaction"]
                updated_reaction["reactants"] = reactants_smiles
                updated_reaction["products"] = products_smiles
                f.write("{}\n".format(json.dumps(updated_reaction)))

    for key, value in skip_dict.items():
        logger.info(f"{value} samples marked as {key} ({100. * value / len(reaction_list)}%)")


def run_from_config(config: AugmentRSMILESConfig) -> None:
    logger.info(f"Preprocessing dataset from {config.data_dir}")

    random.seed(config.seed)

    data_dir = config.data_dir
    augmented_data_dir = config.augmented_data_dir

    if not os.path.exists(augmented_data_dir):
        os.makedirs(augmented_data_dir)

    for fold in config.folds:
        logger.info(f"Processing fold {fold}")

        with open(os.path.join(data_dir, f"{fold.value}.jsonl"), "r") as f:
            reaction_list = [json.loads(line) for line in f.readlines()]

        logger.info(f"Fold size: {len(reaction_list)}")

        product_smarts_list = list(
            map(lambda x: x["mapped_reaction_smiles"].split(">")[2], reaction_list)
        )

        augmented_data_fold_dir = os.path.join(augmented_data_dir, fold.value)

        # Prevent reactions with multiple products.
        multiple_product_indices = [
            i for i in range(len(product_smarts_list)) if "." in product_smarts_list[i]
        ]
        assert len(multiple_product_indices) == 0, "Multiple products are not supported."

        preprocess(
            augmented_data_fold_dir,
            reaction_list,
            fold,
            config.augmentation,
            use_random_smiles=config.use_random_smiles,
        )


def main(argv: Optional[list[str]]) -> None:
    config: AugmentRSMILESConfig = cli_get_config(argv=argv, config_cls=AugmentRSMILESConfig)
    run_from_config(config)


if __name__ == "__main__":
    main(argv=None)
