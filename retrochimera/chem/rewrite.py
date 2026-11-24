from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional, cast

from rdkit import Chem
from rdkit.Chem import AllChem
from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.chem.utils import (
    molecule_bag_from_smiles,
    remove_atom_mapping_from_mol,
)

from retrochimera.chem.explicit_hydrogen import hydrogen_inputs
from retrochimera.chem.fixing import fix_mol, restore_counterions
from retrochimera.chem.utils import (
    get_atom_mapping,
    map_removed_lhs_atoms_to_dummies,
    normalize_atom_mapping,
)


@dataclass
class RewriteResult:
    mols: Bag[Molecule]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, order=False)
class Rewrite:
    """Wrapper around a SMARTS string representing a graph rewrite."""

    lhs: str  # Left-hand side of the rewrite, represented as SMARTS.
    rhs: str  # Right-hand side of the rewrite, represented as SMARTS.
    mapping: list[tuple[int, int]]  # Atom mapping connecting both sides of the rewrite.

    # Other objects which encode the same information as above, just in a different form:
    smarts: str = field(repr=False, compare=False)
    rdkit_lhs_mol: Chem.Mol = field(repr=False, compare=False)
    rdkit_rhs_mol: Chem.Mol = field(repr=False, compare=False)
    rdkit_reaction: AllChem.ChemicalReaction = field(repr=False, compare=False)

    # Extra data to remember which rhs atoms are dummies (see below).
    _max_non_dummy_atom_mapping_num: int = field(repr=False, compare=False)

    @staticmethod
    def from_rdkit(template_smarts: str) -> Rewrite:
        template = AllChem.ReactionFromSmarts(template_smarts)
        normalize_atom_mapping(template)

        # If the rewrite removes some atoms present in the lhs, then this would prevent us from
        # being able to compute the localization below. We thus add dummies to make all the atoms
        # from the lhs appear in the rhs.
        template, max_atom_mapping_num = map_removed_lhs_atoms_to_dummies(template)

        # Make sure the template looks as expected.
        assert len(template.GetReactants()) == 1
        assert len(template.GetProducts()) == 1

        template_lhs_mol = Chem.Mol(template.GetReactantTemplate(0))
        template_rhs_mol = Chem.Mol(template.GetProductTemplate(0))

        mapping_num_to_id_lhs = get_atom_mapping(template_lhs_mol)
        mapping_num_to_id_rhs = get_atom_mapping(template_rhs_mol)

        template_mapping = [
            (mapping_num_to_id_lhs[id], mapping_num_to_id_rhs[id])
            for id in mapping_num_to_id_lhs.keys() & mapping_num_to_id_rhs.keys()
        ]

        for mol in [template_lhs_mol, template_rhs_mol]:
            remove_atom_mapping_from_mol(mol)

        return Rewrite(
            lhs=Chem.MolToSmarts(template_lhs_mol),
            rhs=Chem.MolToSmarts(template_rhs_mol),
            mapping=template_mapping,
            smarts=template_smarts,
            rdkit_lhs_mol=template_lhs_mol,
            rdkit_rhs_mol=template_rhs_mol,
            rdkit_reaction=template,
            _max_non_dummy_atom_mapping_num=max_atom_mapping_num,
        )

    def __getstate__(self) -> str:
        """Convert to easily serializable state, used by `pickle`."""
        return self.smarts

    def __setstate__(self, state: str) -> None:
        """Restore from state returned by `__getstate__`."""
        object.__setattr__(self, "__dict__", Rewrite.from_rdkit(state).__dict__)

    def __deepcopy__(self, memo) -> Rewrite:
        """Override `copy.deepcopy`s behaviour to never copy a `Rewrite` as it is immutable."""
        memo[id(self)] = self
        return self

    @cached_property
    def _localization_plan(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (atom.GetIdx(), atom.GetAtomMapNum(), self.mapping[atom.GetAtomMapNum() - 1][0])
            for atom in self.rdkit_reaction.GetProductTemplate(0).GetAtoms()
            if atom.HasProp("molAtomMapNumber")
        )

    def _extract_localization(self, mol: Chem.Mol) -> tuple[list[int], list[int]]:
        localization: list[Optional[int]] = [None] * self.rdkit_lhs_mol.GetNumAtoms()
        removed = []
        try:
            for rhs_idx, map_num, lhs_idx in self._localization_plan:
                atom = mol.GetAtomWithIdx(rhs_idx)
                if atom.GetIntProp("old_mapno") != map_num:
                    raise KeyError("Unexpected product atom order")
                localization[lhs_idx] = atom.GetIntProp("react_atom_idx")
                if map_num > self._max_non_dummy_atom_mapping_num:
                    removed.append(rhs_idx)
        except (KeyError, RuntimeError):
            # Fall back to provenance scanning if RDKit changes the product atom order.
            localization = [None] * self.rdkit_lhs_mol.GetNumAtoms()
            removed = []
            for atom in mol.GetAtoms():
                if atom.HasProp("old_mapno"):
                    map_num = atom.GetIntProp("old_mapno")
                    localization[self.mapping[map_num - 1][0]] = atom.GetIntProp("react_atom_idx")
                    if map_num > self._max_non_dummy_atom_mapping_num:
                        removed.append(atom.GetIdx())
        assert None not in localization
        return cast(list[int], localization), removed

    @cached_property
    def _needs_hydrogens(self) -> bool:
        # does this rewrite require explicit hydrogens on the input molecule?
        return Chem.HasQueryHs(self.rdkit_lhs_mol)[0]

    def apply(
        self, input: Molecule, fix_result: bool = True, return_localization: bool = True
    ) -> list[RewriteResult]:
        results: list[RewriteResult] = []
        result_to_index: dict[Bag[Molecule], int] = {}
        smiles_to_index: dict[str, Optional[int]] = {}
        has_dummies = self._max_non_dummy_atom_mapping_num < len(self.mapping)

        inputs = (
            hydrogen_inputs(input.rdkit_mol, self.rdkit_lhs_mol)
            if self._needs_hydrogens
            else [(input.rdkit_mol, None)]
        )
        outcomes = (
            (result, indices)
            for mol, indices in inputs
            for result in self.rdkit_reaction.RunReactants((mol,))
        )
        for raw_result, source_indices in outcomes:
            # Make sure the result is a single molecule (should hold due to the asserts above).
            assert len(raw_result) == 1
            (result_rdkit_mol,) = raw_result

            localization: Optional[list[int]] = None
            atom_ids_to_remove: list[int] = []

            if return_localization and has_dummies:
                localization, atom_ids_to_remove = self._extract_localization(result_rdkit_mol)

            if atom_ids_to_remove:
                # Remove dummy atoms that we may have added in `map_removed_lhs_atoms_to_dummies`;
                # proceed in reversed order to prevent IDs shifting from messing things up.
                result_rdkit_mol = Chem.RWMol(result_rdkit_mol)
                for atom_id in reversed(atom_ids_to_remove):
                    result_rdkit_mol.RemoveAtom(atom_id)

            if fix_result:
                smiles = fix_mol(result_rdkit_mol)
            else:
                try:
                    smiles = Chem.MolToSmiles(result_rdkit_mol)
                except Exception:
                    smiles = None

            if smiles is None:
                continue

            # make sure we carry through any spectator counterions from the input molecule
            if "." in input.smiles:
                if localization is None:
                    localization, _ = self._extract_localization(result_rdkit_mol)
                matched_atoms = (
                    localization
                    if source_indices is None
                    else [source_indices[i] for i in localization]
                )
                smiles = restore_counterions(
                    smiles, input.rdkit_mol, {i for i in matched_atoms if i is not None}
                )

            # Identical products can come from different atom matches: reuse parsing,
            # but keep every localization for scoring below.
            if smiles not in smiles_to_index:
                # `fix_mol` already sanitized the molecule and produced canonical SMILES,
                # except explicit/mapped H atoms require a SMILES round-trip to fold.
                canonicalize = not fix_result or "[H]" in smiles or "[H:" in smiles
                mols = molecule_bag_from_smiles(
                    smiles,
                    canonicalize=canonicalize,
                    make_rdkit_mol=canonicalize,
                )
                index = None
                if mols is not None:
                    index = result_to_index.get(mols)
                    if index is None:
                        index = len(results)
                        result_to_index[mols] = index
                        metadata: dict[str, Any] = (
                            {"localizations": []} if return_localization else {}
                        )
                        results.append(RewriteResult(mols=mols, metadata=metadata))
                smiles_to_index[smiles] = index
            index = smiles_to_index[smiles]
            if index is not None and return_localization:
                # Defer provenance work until the outcome has survived fixing and parsing.
                if localization is None:
                    localization, _ = self._extract_localization(result_rdkit_mol)
                results[index].metadata["localizations"].append(
                    [source_indices[i] for i in localization]
                    if source_indices is not None
                    else localization
                )

        return results
