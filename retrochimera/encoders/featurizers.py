"""Classes used for converting molecules and rewrites into graphs compatible with `GNNEncoder`."""

from __future__ import annotations

import hashlib
from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from syntheseus.interface.molecule import Molecule

from retrochimera.chem.rewrite import Rewrite
from retrochimera.dgllife.featurizers import (
    BaseAtomFeaturizer,
    BaseBondFeaturizer,
    CanonicalBondFeaturizer,
    ConcatFeaturizer,
    WeaveAtomFeaturizer,
    atom_chiral_tag_one_hot,
    atom_degree_one_hot,
    atom_formal_charge,
    atom_hybridization_one_hot,
    atom_implicit_valence_one_hot,
    atom_is_aromatic,
    atom_num_radical_electrons,
    atom_total_num_H_one_hot,
    atom_type_one_hot,
)

AtomFeaturizer = Union[BaseAtomFeaturizer, WeaveAtomFeaturizer]


@dataclass
class RawGraph:
    """Class to store a featurized graph.

    This is similar to `torch_geometric.data.Data` but uses numpy arrays instead of torch tensors.
    """

    atom_features: NDArray
    bond_features: NDArray
    bonds: NDArray

    atom_categorical_features: Optional[NDArray]


@dataclass
class RawRewriteGraph(RawGraph):
    """Featurized graph which has its nodes divided into two parts: 'left' and 'right'.

    Technically this is not a bipartite graph, as nodes within each group can be connected.
    """

    node_in_lhs: NDArray


class Featurizer(metaclass=ABCMeta):
    @abstractproperty
    def num_atom_features(self) -> int:
        pass

    @abstractproperty
    def num_bond_features(self) -> int:
        pass

    @property
    def num_atom_categorical_features(self) -> list[int]:
        return []


class MoleculeFeaturizer(Featurizer):
    @abstractmethod
    def __call__(self, mol: Molecule) -> RawGraph:
        pass


class RewriteFeaturizer(Featurizer):
    @abstractmethod
    def __call__(self, rewrite: Rewrite) -> RawGraph:
        pass

    @staticmethod
    def prepare_kwargs(rewrites: Iterable[Rewrite]) -> dict[str, Any]:
        return {}


def featurize_with_dgl_life(
    mol: Chem.Mol, atom_featurizer: AtomFeaturizer, bond_featurizer: BaseBondFeaturizer
) -> RawGraph:
    """Calls the DGL-Life atom/bond featurizers on a given `Chem.Mol` object.

    Importantly, this function does not perform any sanitization on `mol`; it does not have to be a
    valid molecule as long as it is accepted by the provided featurizers. `DGLLifeRewriteFeaturizer`
    exploits this, passing a `mol` constructed from a SMARTS string.
    """
    atom_features = atom_featurizer(mol)["h"]

    if len(mol.GetBonds()) > 0:
        bonds_list = []
        for bond in mol.GetBonds():
            bonds_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bonds_list.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

        bonds = np.asarray(bonds_list)
        bond_features = bond_featurizer(mol)["e"]
    else:
        # The `dgllife` featurizer returns an empty dictionary if there are no bonds, which
        # would lead the above to fail; we handle that case separately.
        bonds = np.zeros((0, 2))
        bond_features = np.zeros((0, bond_featurizer.feat_size()))

    return RawGraph(
        atom_features=atom_features,
        bond_features=bond_features,
        bonds=bonds,
        atom_categorical_features=None,
    )


class DGLLifeMoleculeFeaturizer(MoleculeFeaturizer):
    """A molecule featurizer wrapping around classes provided by DGL-Life."""

    @staticmethod
    def get_default(atom_types: Optional[list[str]] = None) -> DGLLifeMoleculeFeaturizer:
        """Creates a default featurizer.

        Args:
            atom_types: List of atom types to consider. If set to `None`, will use the same
                extensive list of atom types as used in the LocalRetro model.
        """
        # fmt: off
        DEFAULT_ATOM_TYPES = [
            "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
            "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
            "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr",
            "Cr", "Pt", "Hg", "Pb", "W", "Ru", "Nb", "Re", "Te", "Rh", "Tc", "Ba", "Bi",
            "Hf", "Mo", "U", "Sm", "Os", "Ir", "Ce", "Gd", "Ga", "Cs", "Ta"
        ]
        # fmt: on

        atom_featurizer = WeaveAtomFeaturizer(atom_types=atom_types or DEFAULT_ATOM_TYPES)
        bond_featurizer = CanonicalBondFeaturizer(self_loop=False)

        return DGLLifeMoleculeFeaturizer(atom_featurizer, bond_featurizer)

    def __init__(
        self, atom_featurizer: AtomFeaturizer, bond_featurizer: BaseBondFeaturizer
    ) -> None:
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

    @staticmethod
    def _featurize_single(
        mol: Molecule, atom_featurizer: AtomFeaturizer, bond_featurizer: BaseBondFeaturizer
    ) -> RawGraph:
        return featurize_with_dgl_life(
            mol=mol.rdkit_mol,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
        )

    def __call__(self, mol: Molecule) -> RawGraph:
        return DGLLifeMoleculeFeaturizer._featurize_single(
            mol,
            atom_featurizer=self.atom_featurizer,
            bond_featurizer=self.bond_featurizer,
        )

    @property
    def num_atom_features(self) -> int:
        return self.atom_featurizer.feat_size()

    @property
    def num_bond_features(self) -> int:
        return self.bond_featurizer.feat_size()


class DGLLifeRewriteFeaturizer(RewriteFeaturizer):
    """A rewrite featurizer wrapping around classes provided by DGL-Life.

    Even though the featurizers provided by DGL-Life are designed for molecules (i.e. SMILES), some
    of them (generally, the more basic ones) also work on rewrites (i.e. SMARTS). This class
    exploits that to featurize both sides of the rewrite, which are glued using the atom mapping.
    """

    @staticmethod
    def get_default(atom_smarts_vocab: Optional[list[str]]) -> DGLLifeRewriteFeaturizer:
        """Creates a default featurizer.

        Args:
            atom_smarts_vocab: Vocabulary of supported atom SMARTS, used for one-hot encoding. If
                set to `None`, the atom SMARTS will instead be featurized with random vectors of a
                fixed size. This option is useful for hashing, where the embeddings do not have to
                be informative or easy to separate.
        """
        # Same as `CanonicalAtomFeaturizer` but with `atom_chiral_tag_one_hot` added.
        atom_featurizer = BaseAtomFeaturizer(
            featurizer_funcs={
                "h": ConcatFeaturizer(
                    [
                        atom_chiral_tag_one_hot,
                        atom_degree_one_hot,
                        atom_formal_charge,
                        atom_hybridization_one_hot,
                        atom_implicit_valence_one_hot,
                        atom_is_aromatic,
                        atom_num_radical_electrons,
                        atom_total_num_H_one_hot,
                        atom_type_one_hot,
                    ]
                )
            }
        )
        bond_featurizer = CanonicalBondFeaturizer(self_loop=False)

        return DGLLifeRewriteFeaturizer(atom_featurizer, bond_featurizer, atom_smarts_vocab)

    def __init__(
        self,
        atom_featurizer: AtomFeaturizer,
        bond_featurizer: BaseBondFeaturizer,
        atom_smarts_vocab: Optional[list[str]],
    ) -> None:
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

        self._atom_smarts_to_id: Optional[dict[str, int]]
        if atom_smarts_vocab is not None:
            self._atom_smarts_to_id = {
                atom_smarts: id for id, atom_smarts in enumerate(atom_smarts_vocab)
            }
        else:
            self._atom_smarts_to_id = None

    @staticmethod
    def _smarts_to_mol(smarts: str) -> Chem.Mol:
        mol = Chem.MolFromSmarts(smarts)
        mol.UpdatePropertyCache(strict=False)
        return mol

    @staticmethod
    def _get_atom_smarts(mol: Chem.Mol) -> list[str]:
        return [atom.GetSmarts() for atom in mol.GetAtoms()]

    @staticmethod
    def prepare_kwargs(rewrites: Iterable[Rewrite]) -> dict[str, Any]:
        atom_smarts_vocab = set()

        for rewrite in rewrites:
            for smarts in [rewrite.lhs, rewrite.rhs]:
                mol = DGLLifeRewriteFeaturizer._smarts_to_mol(smarts)
                atom_smarts_vocab |= set(DGLLifeRewriteFeaturizer._get_atom_smarts(mol))

        return {"atom_smarts_vocab": sorted(list(atom_smarts_vocab))}

    @staticmethod
    def _embed_atom_smarts(smarts: str, atom_smarts_to_id: Optional[dict[str, int]]) -> int:
        if atom_smarts_to_id is not None:
            return atom_smarts_to_id[smarts]
        else:
            return int(hashlib.shake_256(smarts.encode()).hexdigest(4), 16)

    @staticmethod
    def _featurize_single(
        rewrite: Rewrite,
        atom_featurizer: AtomFeaturizer,
        bond_featurizer: BaseBondFeaturizer,
        atom_smarts_to_id: Optional[dict[str, int]],
    ) -> RawRewriteGraph:
        graphs = []
        atom_smarts = []

        # First, featurize each side separately using DGL-Life.
        for smarts in [rewrite.lhs, rewrite.rhs]:
            mol = DGLLifeRewriteFeaturizer._smarts_to_mol(smarts)
            atom_smarts += DGLLifeRewriteFeaturizer._get_atom_smarts(mol)
            graphs.append(
                featurize_with_dgl_life(
                    mol=mol, atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer
                )
            )

        # TODO(krmaziar): Consider handling bond SMARTS in a similar way.
        atom_smarts_embeddings = np.stack(
            [
                DGLLifeRewriteFeaturizer._embed_atom_smarts(
                    smarts, atom_smarts_to_id=atom_smarts_to_id
                )
                for smarts in atom_smarts
            ]
        )

        [num_atoms_lhs, num_atoms_rhs] = [len(graph.atom_features) for graph in graphs]

        # Build an extra binary feature to differentiate left from right.
        which_side_feature = np.concatenate(
            [np.ones((num_atoms_lhs, 1)), np.zeros((num_atoms_rhs, 1))], axis=0
        )

        # Grab bonds from both sides, shift the atom ids for the rhs.
        [bonds_lhs, bonds_rhs] = [graph.bonds for graph in graphs]

        combined_graph = RawRewriteGraph(
            atom_features=np.concatenate([graph.atom_features for graph in graphs], axis=0),
            bond_features=np.concatenate([graph.bond_features for graph in graphs], axis=0),
            bonds=np.concatenate([bonds_lhs, bonds_rhs + num_atoms_lhs], axis=0),
            atom_categorical_features=atom_smarts_embeddings[:, np.newaxis],
            # Useful downstream if we only care about the lhs representations:
            node_in_lhs=np.squeeze(which_side_feature, axis=-1),
        )

        # Concatenate the lhs-vs-rhs feature into the atom features.
        combined_graph.atom_features = np.concatenate(
            [combined_graph.atom_features, which_side_feature], axis=1
        )

        num_bonds_original = len(combined_graph.bonds)

        # Prepare the atom mapping edges that will connect the two components.
        atom_mapping_bonds_list = [
            (atom_idx_lhs, atom_idx_rhs + num_atoms_lhs)
            for (atom_idx_lhs, atom_idx_rhs) in rewrite.mapping
        ]

        # Include the new edges in both directions.
        atom_mapping_bonds_list += [
            (idx_rhs, idx_lhs) for (idx_lhs, idx_rhs) in atom_mapping_bonds_list
        ]

        if atom_mapping_bonds_list:
            atom_mapping_bonds = np.asarray(atom_mapping_bonds_list)
        else:
            atom_mapping_bonds = np.zeros((0, 2))

        combined_graph.bonds = np.concatenate([combined_graph.bonds, atom_mapping_bonds], axis=0)
        combined_graph.bond_features = np.concatenate(
            [
                combined_graph.bond_features,
                np.zeros((len(atom_mapping_bonds_list), combined_graph.bond_features.shape[1])),
            ],
            axis=0,
        )

        # Add an extra binary feature to differentiate real bonds from the atom mapping.
        is_atom_mapping_feature = np.concatenate(
            [np.zeros((num_bonds_original, 1)), np.ones((len(atom_mapping_bonds_list), 1))], axis=0
        )
        combined_graph.bond_features = np.concatenate(
            [combined_graph.bond_features, is_atom_mapping_feature], axis=1
        )

        return combined_graph

    def __call__(self, rewrite: Rewrite) -> RawGraph:
        return DGLLifeRewriteFeaturizer._featurize_single(
            rewrite,
            atom_featurizer=self.atom_featurizer,
            bond_featurizer=self.bond_featurizer,
            atom_smarts_to_id=self._atom_smarts_to_id,
        )

    @property
    def num_atom_features(self) -> int:
        return self.atom_featurizer.feat_size() + 1

    @property
    def num_bond_features(self) -> int:
        return self.bond_featurizer.feat_size() + 1

    @property
    def num_atom_categorical_features(self) -> list[int]:
        assert self._atom_smarts_to_id is not None
        return [len(self._atom_smarts_to_id)]
