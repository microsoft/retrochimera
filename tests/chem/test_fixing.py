from rdkit import Chem

from retrochimera.chem.fixing import fix_mol, reverse_rxn


def test_fix_mol_possible() -> None:
    smi = "c1cc2cc(ccc2n1)-c1ncno1"
    out = fix_mol(Chem.MolFromSmiles(smi, sanitize=False))
    tgt = Chem.CanonSmiles("N1C=CC2=CC(=CC=C12)C1=NC=NO1")
    assert out == tgt


def test_fix_mol_impossible() -> None:
    # First of the SMILES below is already rejected by `adjust_aromatic_Ns`, second one is processed
    # successfully but at the end still cannot be sanitized.
    broken_smiles = ["nnn", "COc1ccccc1Oc1nc(Nc2cc(C)[nH](C3CCCCO3)n2)cc2ccccc12"]

    for smiles in broken_smiles:
        assert fix_mol(Chem.MolFromSmiles(smiles, sanitize=False)) is None


def test_reverse() -> None:
    assert reverse_rxn("A>B>C.D") == "C.D>B>A"
    assert reverse_rxn("A>>C1OC1") == "C1OC1>>A"
