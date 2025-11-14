from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, cast

from rdkit import Chem
from rdkit.Chem import AllChem
from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.chem.utils import (
    molecule_bag_from_smiles,
    remove_atom_mapping_from_mol,
)

from retrochimera.chem.fixing import fix_mol
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

    def apply(
        self, input: Molecule, fix_result: bool = True, return_localization: bool = True
    ) -> list[RewriteResult]:
        results: list[RewriteResult] = []
        result_to_index: dict[Bag[Molecule], int] = {}

        for raw_result in self.rdkit_reaction.RunReactants((input.rdkit_mol,)):
            # Make sure the result is a single molecule (should hold due to the asserts above).
            assert len(raw_result) == 1
            (result_rdkit_mol,) = raw_result

            localization: Optional[list[int]] = None
            atom_ids_to_remove: list[int] = []

            if return_localization:
                localization_tmp = [None] * self.rdkit_lhs_mol.GetNumAtoms()

                for atom in result_rdkit_mol.GetAtoms():
                    atom_props = atom.GetPropsAsDict()

                    if "old_mapno" in atom_props:
                        atom_mapping_idx = atom_props["old_mapno"]
                        atom_idx = self.mapping[atom_mapping_idx - 1][0]
                        localization_tmp[atom_idx] = atom_props["react_atom_idx"]

                        if atom_mapping_idx > self._max_non_dummy_atom_mapping_num:
                            atom_ids_to_remove.append(atom.GetIdx())

                assert None not in localization_tmp
                localization = cast(list[int], localization_tmp)

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

            if smiles is not None:
                mols = molecule_bag_from_smiles(smiles)
                if mols is not None:
                    index = result_to_index.get(mols)
                    if index is not None:
                        if return_localization:
                            current_localizations = results[index].metadata["localizations"]
                            assert localization is not None

                            current_localizations.append(localization)
                    else:
                        result_to_index[mols] = len(results)

                        if localization is not None:
                            metadata = {"localizations": [localization]}
                        else:
                            metadata = {}

                        results.append(RewriteResult(mols=mols, metadata=metadata))

        return results
