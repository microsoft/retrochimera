import random
from typing import Sequence

import numpy as np
from syntheseus import Bag, Molecule, Reaction
from syntheseus.interface.molecule import SMILES_SEPARATOR
from syntheseus.interface.reaction import ReactionMetaData
from syntheseus.reaction_prediction.inference_base import ExternalForwardReactionModel
from syntheseus.reaction_prediction.utils.inference import process_raw_smiles_outputs_forwards

from retrochimera.inference.smiles_transformer import AbstractSmilesTransformerModel


class SmilesTransformerForwardModel(
    AbstractSmilesTransformerModel[Bag[Molecule], Reaction], ExternalForwardReactionModel
):
    def _augment_input(self, input: Bag[Molecule]) -> list[str]:
        from root_aligned.preprocessing.generate_PtoR_data import clear_map_canonical_smiles

        augmented_input = []

        # Obtain `reactant_roots`
        all_reactants_atom_ids = [
            [i for i in range(reactant.rdkit_mol.GetNumAtoms())] for reactant in input
        ]
        max_times = np.prod(
            [len(per_reactant_atom_ids) for per_reactant_atom_ids in all_reactants_atom_ids]
        )
        reactant_roots = [[-1 for _ in input]]

        while len(reactant_roots) < min(self.augmentation_size, max_times):
            roots = [random.choice(all_reactants_atom_ids[k]) for k in range(len(input))]
            if roots not in reactant_roots:
                reactant_roots.append(roots)

        if len(reactant_roots) < self.augmentation_size:
            reactant_roots.extend(
                random.choices(reactant_roots, k=self.augmentation_size - len(reactant_roots))
            )

        for k in range(self.augmentation_size):
            tmp = list(zip(input, reactant_roots[k]))
            reactant_k, reactant_roots_k = [i[0] for i in tmp], [i[1] for i in tmp]

            aligned_reactants = []
            for i, reactant in enumerate(reactant_k):
                reactant_root = reactant_roots_k[i]
                reactant_smi = clear_map_canonical_smiles(
                    reactant.smiles, canonical=True, root=reactant_root
                )
                aligned_reactants.append(reactant_smi)

            augmented_input.append(SMILES_SEPARATOR.join(sorted(aligned_reactants)))

        return augmented_input

    def _process_raw_smiles_outputs(
        self, input: Bag[Molecule], output_list: list[str], metadata_list: list[ReactionMetaData]
    ) -> Sequence[Reaction]:
        return process_raw_smiles_outputs_forwards(input, output_list, metadata_list)
