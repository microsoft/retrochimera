"""Functions to fix broken molecules, in particular heterocycles.

Code was adapted from RDKit cookbook:
https://rdkit.readthedocs.io/en/latest/Cookbook.html#cleaning-up-heterocycles
"""

from typing import Optional

from rdkit import Chem, RDLogger

from retrochimera.utils.logging import get_logger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

logger = get_logger(__name__)


def reverse_rxn(smi):
    r, a, p = smi.split(">")
    return f"{p}>{a}>{r}"


def frag_indices_to_mol(o_mol, indices):
    em = Chem.EditableMol(Chem.Mol())

    newIndices = {}
    for i, idx in enumerate(indices):
        em.AddAtom(o_mol.GetAtomWithIdx(idx))
        newIndices[idx] = i

    for i, idx in enumerate(indices):
        at = o_mol.GetAtomWithIdx(idx)
        for bond in at.GetBonds():
            if bond.GetBeginAtomIdx() == idx:
                oidx = bond.GetEndAtomIdx()
            else:
                oidx = bond.GetBeginAtomIdx()
            # make sure every bond only gets added once:
            if oidx < idx:
                continue
            em.AddBond(newIndices[idx], newIndices[oidx], bond.GetBondType())
    res = em.GetMol()
    res.ClearComputedProps()
    Chem.GetSymmSSSR(res)
    res.UpdatePropertyCache(False)
    res._idxMap = newIndices
    return res


def _recursively_modify_Ns(mol, matches, indices=None):
    if indices is None:
        indices = []
    res = None
    while len(matches) and res is None:
        tIndices = indices[:]
        nextIdx = matches.pop(0)
        tIndices.append(nextIdx)
        nm = Chem.Mol(mol.ToBinary())
        nm.GetAtomWithIdx(nextIdx).SetNoImplicit(True)
        nm.GetAtomWithIdx(nextIdx).SetNumExplicitHs(1)
        cp = Chem.Mol(nm.ToBinary())
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            res, indices = _recursively_modify_Ns(nm, matches, indices=tIndices)
        else:
            indices = tIndices
            res = cp
    return res, indices


def adjust_aromatic_Ns(m: Chem.Mol, nitrogenPattern: str = "[n&D2&H0;r5,r6]") -> Optional[Chem.Mol]:
    """Fix issues with aromatic nitrogens in rings.

    Default pattern matches Ns in 5-rings and 6-rings in order to be able to fix `O=c1ccncc1`.
    """
    Chem.GetSymmSSSR(m)
    m.UpdatePropertyCache(False)

    # break non-ring bonds linking rings:
    em = Chem.EditableMol(m)
    linkers = m.GetSubstructMatches(Chem.MolFromSmarts("[r]!@[r]"))
    plsFix = set()
    for a, b in linkers:
        em.RemoveBond(a, b)
        plsFix.add(a)
        plsFix.add(b)
    nm = em.GetMol()
    for at in plsFix:
        at = nm.GetAtomWithIdx(at)
        if at.GetIsAromatic() and at.GetAtomicNum() == 7:
            at.SetNumExplicitHs(1)
            at.SetNoImplicit(True)

    # build molecules from the fragments:
    fragLists = Chem.GetMolFrags(nm)
    frags = [frag_indices_to_mol(nm, x) for x in fragLists]

    # loop through the fragments in turn and try to aromatize them:
    for frag in frags:
        cp = Chem.Mol(frag.ToBinary())
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            matches = [x[0] for x in frag.GetSubstructMatches(Chem.MolFromSmarts(nitrogenPattern))]
            lres, indices = _recursively_modify_Ns(frag, matches)
            if not lres:
                return None
            else:
                revMap = {}
                for k, v in frag._idxMap.items():
                    revMap[v] = k
                for idx in indices:
                    oatom = m.GetAtomWithIdx(revMap[idx])
                    oatom.SetNoImplicit(True)
                    oatom.SetNumExplicitHs(1)

    return m


def fix_mol(mol: Chem.Mol) -> Optional[str]:
    """Fix weird heterocycles.

    Args:
        mol: RDKit molecule to fix which may be in a broken state.

    Returns:
        SMILES of the fixed molecule, or `None` if it could not have been fixed.
    """
    try:
        mol.UpdatePropertyCache(False)
        cp = Chem.Mol(mol.ToBinary())
        Chem.SanitizeMol(cp)
        mol = cp
        return Chem.MolToSmiles(mol)
    except ValueError:
        logger.debug(f"fix_mol: {Chem.MolToSmiles(mol)}")

        mol_adjusted = adjust_aromatic_Ns(mol)
        if mol_adjusted is None:
            return None

        try:
            Chem.SanitizeMol(mol_adjusted)
            return Chem.MolToSmiles(mol_adjusted)
        except ValueError:
            return None
