"""RootAligned utilities

Code for clear_map_canonical_smiles, get_cano_map_number, and get_root_id from
https://github.com/otori-bird/retrosynthesis/blob/main/preprocessing/generate_PtoR_data.py
"""

import random
from typing import Optional

import numpy as np
from rdkit import Chem

AUGMENTATION_SEED_METADATA_KEY = "retrochimera_augmentation_seed"


def get_product_roots(
    product_atom_ids: list[int],
    num_augmentations: int,
    rng: Optional[random.Random] = None,
) -> list[int]:
    product_roots = [-1]

    if len(product_atom_ids) < num_augmentations:
        product_roots.extend(product_atom_ids)
        if rng is None:
            extra_roots = random.choices(
                product_roots,
                k=num_augmentations - len(product_roots),
            )
        else:
            extra_roots = rng.choices(
                product_roots,
                k=num_augmentations - len(product_roots),
            )
        product_roots.extend(extra_roots)
    else:
        if rng is None:
            sampled_roots = random.sample(product_atom_ids, num_augmentations - 1)
        else:
            sampled_roots = rng.sample(product_atom_ids, num_augmentations - 1)
        product_roots.extend(sampled_roots)

    assert len(product_roots) == num_augmentations
    return product_roots


def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


def get_cano_map_number(smi, root=-1):
    atommap_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi, root=root))
    cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
    correct_mapped = [
        canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol()
        for i, index in enumerate(cano2atommapIdx)
    ]
    atom_number = len(canonical_mol.GetAtoms())
    if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
        cano2atommapIdx = [0] * atom_number
        atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)
        if len(atommap2canoIdx) != atom_number:
            return None
        for i, index in enumerate(atommap2canoIdx):
            cano2atommapIdx[index] = i
    id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]

    return [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]


def get_root_id(mol, root_map_number):
    root = -1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            root = i
            break
    return root
