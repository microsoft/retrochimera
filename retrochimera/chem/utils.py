from rdkit import Chem
from rdkit.Chem import AllChem
from syntheseus.reaction_prediction.chem.utils import ATOM_MAPPING_PROP_NAME


def get_mapped_atoms(mol: Chem.Mol) -> list[Chem.Atom]:
    """Extract all atoms with atom mapping.

    Args:
        mol: Molecule to extract the atoms from.

    Returns:
        List of atoms that have atom mapping.
    """
    return [atom for atom in mol.GetAtoms() if atom.HasProp(ATOM_MAPPING_PROP_NAME)]


def get_atom_mapping(mol: Chem.Mol) -> dict[int, int]:
    """Extract a mapping from atom mapping ID to atom ID.

    Args:
        mol: Molecule to extract the atom mapping from.

    Returns:
        Dictionary mapping IDs assigned under the atom mapping to RDKit atom IDs.
    """
    atom_mapping_num_to_idx = {}

    for atom in get_mapped_atoms(mol):
        atom_mapping_num_to_idx[atom.GetAtomMapNum()] = atom.GetIdx()

    return atom_mapping_num_to_idx


def get_all_atom_mapping_nums(mol: Chem.Mol) -> list[int]:
    """Extract all atom mapping numbers.

    Args:
        mol: Molecule to extract the numbers from.

    Returns:
        List of atom mapping IDs.
    """
    return [atom.GetAtomMapNum() for atom in get_mapped_atoms(mol)]


def normalize_atom_mapping(reaction: AllChem.ChemicalReaction) -> None:
    """Normalize atom mapping in a reaction in-place to use `[1, 2, ...]`."""

    def get_mapped_atoms(mol: Chem.Mol) -> list[Chem.Atom]:
        return [
            (atom, atom.GetAtomMapNum())
            for atom in mol.GetAtoms()
            if atom.HasProp(ATOM_MAPPING_PROP_NAME)
        ]

    reactant_mapped_atoms: list[Chem.Atom] = sum(
        [get_mapped_atoms(mol) for mol in reaction.GetReactants()], []
    )
    product_mapped_atoms: list[Chem.Atom] = sum(
        [get_mapped_atoms(mol) for mol in reaction.GetProducts()], []
    )

    reactant_atom_ids = set(id for _, id in reactant_mapped_atoms)
    product_atom_ids = set(id for _, id in product_mapped_atoms)

    if len(reactant_atom_ids) < len(reactant_mapped_atoms):
        raise ValueError("Reactant side of the reaction contains duplicate mapping IDs")

    if len(product_atom_ids) < len(product_mapped_atoms):
        raise ValueError("Product side of the reaction contains duplicate mapping IDs")

    id_mapping = {
        id_old: id_new + 1
        for id_new, id_old in enumerate(sorted(reactant_atom_ids & product_atom_ids))
    }

    for atom, id in reactant_mapped_atoms + product_mapped_atoms:
        if id in id_mapping:
            atom.SetAtomMapNum(id_mapping[id])
        else:
            atom.ClearProp(ATOM_MAPPING_PROP_NAME)


def map_removed_lhs_atoms_to_dummies(
    reaction: AllChem.ChemicalReaction,
) -> tuple[AllChem.ChemicalReaction, int]:
    """Add a dummy atom to the right-hand side for each unmapped atom on the left-hand side.

    Returns:
        A modified reaction and the largest atom mapping ID that is *not* mapped to a dummy atom.
    """

    reactant_unmapped_atoms: list[Chem.Atom] = []
    max_atom_mapping_num = -1

    for mol in reaction.GetReactants():
        for atom in mol.GetAtoms():
            if atom.HasProp(ATOM_MAPPING_PROP_NAME):
                max_atom_mapping_num = max(max_atom_mapping_num, atom.GetAtomMapNum())
            else:
                reactant_unmapped_atoms.append(atom)

    reactants = list(reaction.GetReactants())
    products = list(reaction.GetProducts())

    product_new = Chem.RWMol(products[0])
    curr_atom_mapping_num = max_atom_mapping_num + 1

    for atom in reactant_unmapped_atoms:
        atom.SetAtomMapNum(curr_atom_mapping_num)

        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetAtomMapNum(curr_atom_mapping_num)
        product_new.AddAtom(new_atom)

        curr_atom_mapping_num += 1

    products[0] = Chem.Mol(product_new)

    # Create an empty reaction and add back all the reactant/product templates.
    reaction_new = AllChem.ReactionFromSmarts(">>")

    for reactant in reactants:
        reaction_new.AddReactantTemplate(reactant)

    for product in products:
        reaction_new.AddProductTemplate(product)

    return reaction_new, max_atom_mapping_num
