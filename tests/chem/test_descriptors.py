from rdkit import Chem
from rdkit.Chem import AllChem

from retrochimera.chem.descriptors import (
    calc_ap,
    calc_morgan,
    calc_rdkfp,
    calc_rxn_fp,
    calc_rxn_morgan,
    fp_to_dense_np,
    fp_to_sparse_array,
    tanimoto,
    tanimoto_sparse,
)


def test_fp_to_dense() -> None:
    mol = Chem.MolFromSmiles("CCCCOCCCF")

    fpr = calc_morgan(mol)
    a = fp_to_dense_np(fp=fpr, size=128)
    assert a.sum() >= 0


def test_fp_to_sparse() -> None:
    mol = Chem.MolFromSmiles("CCCCOCCCF")

    fpr = calc_morgan(mol)
    a = fp_to_sparse_array(fp=fpr, size=128)
    assert a.sum() >= 0


def test_rdkit_rxnfp() -> None:
    r1 = AllChem.ReactionFromSmarts("OCCCO>>FCCCF")
    r2 = AllChem.ReactionFromSmarts("FOOF.CCCCl>>CCCO")
    r3 = AllChem.ReactionFromSmarts("OCCCCO>>FCCCCF")

    fpr1 = calc_rxn_morgan(r1)
    fpr2 = calc_rxn_morgan(r2)
    fpr3 = calc_rxn_morgan(r3)

    a = fp_to_dense_np(fp=fpr1, size=128)
    b = fp_to_dense_np(fp=fpr2, size=128)
    c = fp_to_dense_np(fp=fpr3, size=128)

    assert a.sum() == 0
    assert b.sum() != 0

    res = tanimoto(a.reshape(1, -1), c.reshape(1, -1))
    assert res.shape == (1,)
    assert 0.0 < res.item() < 1.0


def test_custom_rxnfp() -> None:
    r1 = AllChem.ReactionFromSmarts(
        "OC(=O)C1=CC=CC=C1.C1CCNC1>>O=C(N1CCCC1)C1=CC=CC=C1", useSmiles=1
    )
    r2 = AllChem.ReactionFromSmarts("FOOF.CCCCl>>CCCO", useSmiles=1)
    r3 = AllChem.ReactionFromSmarts(
        "CC1=CC=CC=C1C(O)=O.C1CCNC1>>CC1=C(C=CC=C1)C(=O)N1CCCC1", useSmiles=1
    )

    a = calc_rxn_fp(r1, fp_fn=calc_morgan)
    b = calc_rxn_fp(r2, fp_fn=calc_morgan)
    c = calc_rxn_fp(r3, fp_fn=calc_morgan)

    assert b.sum() != 0
    res = tanimoto_sparse(a.reshape(1, -1), c.reshape(1, -1))
    assert res.shape == (1,)
    assert 0.0 < res.item() < 1.0


def test_custom_ap_rxnfp() -> None:
    r1 = AllChem.ReactionFromSmarts(
        "OC(=O)C1=CC=CC=C1.C1CCNC1>>O=C(N1CCCC1)C1=CC=CC=C1", useSmiles=1
    )
    r2 = AllChem.ReactionFromSmarts(
        "CC1=CC=CC=C1C(O)=O.C1CCNC1>>CC1=C(C=CC=C1)C(=O)N1CCCC1", useSmiles=1
    )

    a = calc_rxn_fp(r1, fp_fn=calc_ap, size=1000)
    b = calc_rxn_fp(r2, fp_fn=calc_ap, size=1000)

    res = tanimoto_sparse(a, b)
    assert res.shape == (1,)
    assert 0.0 < res.item() < 1.0


def test_custom_rdk_rxnfp() -> None:
    r1 = AllChem.ReactionFromSmarts(
        "OC(=O)C1=CC=CC=C1.C1CCNC1>>O=C(N1CCCC1)C1=CC=CC=C1", useSmiles=1
    )
    r2 = AllChem.ReactionFromSmarts(
        "CC1=CC=CC=C1C(O)=O.C1CCNC1>>CC1=C(C=CC=C1)C(=O)N1CCCC1", useSmiles=1
    )

    a = calc_rxn_fp(r1, fp_fn=calc_rdkfp)
    b = calc_rxn_fp(r2, fp_fn=calc_rdkfp)

    res = tanimoto_sparse(a, b)
    assert res.shape == (1,)
    assert 0.0 < res.item() < 1.0
