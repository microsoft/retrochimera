"""Fingerprint methods to featurize molecules and reactions."""

import copy
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Iterator, Union

import joblib
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdmolops
from rdkit.Chem.rdChemReactions import (
    CreateDifferenceFingerprintForReaction,
    FingerprintType,
    ReactionFingerprintParams,
)
from scipy import sparse as sp
from syntheseus.interface.molecule import Molecule


def fp_to_dense_np(fp, size: int = 16384) -> NDArray:
    """Convert rdkit counted fingerprint to dense numpy array, with modulo-folding to given size.

    Args:
        fp: RDKit count fingerprint.
        size: Length to fold the fingerprint to (via mod).

    Returns:
        Fingerprint as a dense `numpy` array.
    """

    array = np.zeros(size)

    for i, j in fp.GetNonzeroElements().items():
        k = i % size
        array[k] += j

    return array


def fp_to_sparse_array(fp, size: int = 2048) -> sp.dok_matrix:
    """Converts `rdkit` count fingerprint to `scipy` sparse array.

    The fingerprint is modulo-folded to the given size. The returned array, which is in
    `dok_matrix` form, should be converted into the needed array later on, e.g. CSR matrix.

    Args:
        fp: RDKit count fingerprint.
        size: Length to fold the fingerprint to (via mod).

    Returns:
        Fingerprint as a sparse `scipy.sparse.dok_matrix` array.
    """
    array = sp.dok_matrix((1, size), dtype="i")

    for i, j in fp.GetNonzeroElements().items():
        k = i % size
        array[0, k] += j

    return array


def tanimoto(fp1: NDArray, fp2: NDArray) -> NDArray:
    assert fp1.shape == fp2.shape
    assert len(fp1.shape) == 2

    ab = (fp1 * fp2).sum(axis=-1)
    a = (fp1 * fp1).sum(axis=-1)
    b = (fp2 * fp2).sum(axis=-1)

    c = a + b - ab

    return np.divide(a, c, out=np.zeros_like(ab), where=c != 0)


def tanimoto_sparse(fp1, fp2):
    assert fp1.shape == fp2.shape
    assert len(fp1.shape) == 2

    ab = np.asarray(fp1.multiply(fp2).sum(axis=-1)).flatten()
    a = np.asarray(fp1.multiply(fp1).sum(axis=-1)).flatten()
    b = np.asarray(fp2.multiply(fp2).sum(axis=-1)).flatten()

    return ab / (a + b - ab)


def calc_morgan(mol: Chem.Mol, radius: int = 2):
    Chem.rdmolops.GetSSSR(mol)
    return AllChem.GetMorganFingerprint(mol, radius)


def calc_ap(mol: Chem.Mol, max_path: int = 3, n_bits: int = 1001107):
    return AllChem.GetHashedAtomPairFingerprint(mol, maxLength=max_path, nBits=n_bits)


def calc_rdkfp(mol: Chem.Mol, max_path: int = 8, branched: bool = True):
    return rdmolops.UnfoldedRDKFingerprintCountBased(mol, maxPath=max_path, branchedPaths=branched)


def calc_count_fp(
    mol: Chem.Mol, size: int = 32768, radius: int = 2, fold_mod: int = 32749
) -> NDArray:
    assert fold_mod <= size

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)

    fingerprint = np.zeros((size,), dtype=np.float32)
    for index, count in fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements().items():
        fingerprint[index % fold_mod] += count

    return fingerprint


def calc_rxn_morgan(
    rxn: AllChem.ChemicalReaction, size: int = 4096, fp_type=FingerprintType.MorganFP
):
    params = ReactionFingerprintParams(False, 0, 1, 0, size, fp_type)
    return CreateDifferenceFingerprintForReaction(rxn, params)


def calc_rxn_fp(rxn: AllChem.ChemicalReaction, fp_fn, size: int = 8192):
    output = sp.dok_matrix((1, size), dtype="i")

    for mol in rxn.GetProducts():
        mol.UpdatePropertyCache(False)
        fp = fp_fn(mol)
        output += fp_to_sparse_array(fp, size=size)

    for mol in rxn.GetReactants():
        mol.UpdatePropertyCache(False)
        fp = fp_fn(mol)
        output -= fp_to_sparse_array(fp, size=size)

    return output.tocsr()


def sum_fps(fps):
    summed_fp = copy.deepcopy(fps[0])
    for fp in fps[1:]:
        summed_fp += fp
    return summed_fp


@dataclass
class ChemData:
    embedding: Union[NDArray, sp.dok_matrix]
    valid: bool


class ChemicalEmbedder:
    """Abstract class to featurize chemical entities (molecules and reactions) in batch."""

    def __init__(self, size: int = 1024, sparse: bool = True):
        self.size = size
        self.sparse = sparse

    def embed_list(self, smiles_list: list[str], n_jobs: int = -1) -> Iterator[ChemData]:
        pool = joblib.Parallel(n_jobs=n_jobs)
        jobs = (joblib.delayed(self._convert)(s) for s in smiles_list)

        return iter(pool(jobs))

    def embed(self, smiles_list: list[str], n_jobs: int = -1) -> tuple[NDArray, NDArray]:
        """Featurize a list of molecules or reactions.

        Args:
            smiles_list: Chemical entities to featurize (molecules or reactions).
            n_jobs: Number of parallel processes to use.

        Returns:
            Two arrays of the same length as `smiles_list`. The first contains the feature vectors,
            the second is a binary mask which indicates whether the featurization worked.
        """
        embeddings = []
        mask = []

        for result in self.embed_list(smiles_list, n_jobs):
            embeddings.append(result.embedding)
            mask.append(result.valid)

        return self._postprocess(embeddings, mask)

    def _postprocess(self, X, mask):
        if self.sparse:
            return self._post_proc_sparse(sp.vstack(X, format="csr")), np.array(mask)
        else:
            return self._post_proc_dense(np.vstack(X)), np.array(mask)

    def _post_proc_sparse(self, X):
        return X

    def _post_proc_dense(self, X):
        return X

    @abstractmethod
    def _convert(self, smiles):
        pass


class MoleculeEmbedder(ChemicalEmbedder):
    """Class to featurize molecules (fingerprint hardcoded as Morgan2 / ECFP4)."""

    def __init__(self, size: int = 1024, sparse: bool = True, log_x: bool = True) -> None:
        super().__init__(size, sparse)
        self.log_x = log_x

    def _convert(self, smiles: str) -> ChemData:
        mol = Chem.MolFromSmiles(smiles)

        if mol:
            fp = calc_morgan(mol, 2)
            return self._return_fp(fp)
        else:
            return ChemData(embedding=np.zeros(self.size), valid=False)

    def convert_mol(self, mol: Molecule) -> ChemData:
        return self._convert(mol.smiles)

    def _return_fp(self, fp) -> ChemData:
        if self.sparse:
            return ChemData(embedding=fp_to_sparse_array(fp, self.size), valid=True)
        else:
            return ChemData(embedding=fp_to_dense_np(fp, self.size), valid=True)

    def _post_proc_dense(self, X):
        if self.log_x:
            return np.log(X + 1)
        else:
            return X

    def _post_proc_sparse(self, X):
        if self.log_x:
            X.data = np.log(X.data + 1)
            return X
        else:
            return X


class FPType(Enum):
    ecfp = 1
    apfp = 2
    pathfp = 3


def get_fp_function(fptype: FPType, max_path: int = 3, radius: int = 2, branched: bool = True):
    assert fptype in FPType

    if fptype == FPType.ecfp:
        return partial(calc_morgan, radius=radius)
    if fptype == FPType.apfp:
        return partial(calc_rdkfp, max_path=max_path, branched=branched)
    if fptype == FPType.pathfp:
        return partial(calc_ap, max_path=max_path)
