import argparse
import math
import multiprocessing
from abc import abstractmethod
from typing import Any, Generic, Sequence, TypeVar

import torch
from syntheseus import Molecule, Reaction, SingleProductReaction
from syntheseus.interface.reaction import ReactionMetaData
from syntheseus.reaction_prediction.inference_base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
)
from syntheseus.reaction_prediction.utils.misc import suppress_outputs

from retrochimera.models.smiles_transformer import SmilesTransformerModel as TransformerModel
from retrochimera.opennmt.decode.translator import Translator
from retrochimera.utils.logging import get_logger
from retrochimera.utils.root_aligned import get_product_roots

logger = get_logger(__name__)


InputType = TypeVar("InputType")
ReactionType = TypeVar("ReactionType", bound=Reaction)


class AbstractSmilesTransformerModel(Generic[InputType, ReactionType]):
    def __init__(
        self,
        *args,
        beam_size: int = 20,
        augmentation_size: int = 10,
        max_generated_seq_len: int = 512,
        probability_from_score_temperature: float = 3.0,
        filter_duplicate_augmentations: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the SmilesTransformer model wrapper.

        This base class is shared between the backward and forward model variants.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.ckpt` file
        - `model_dir` contains the vocab path as the only `*.txt` file
        """

        super().__init__(*args, **kwargs)

        if not (hasattr(self, "model_dir") and hasattr(self, "device")):
            raise ValueError(
                "Class based on `AbstractSmilesTransformerModel` should extended `ReactionModel`"
            )

        # There should be exaclty one `*.ckpt` file under `model_dir`.
        chkpt_path = get_unique_file_in_dir(self.model_dir, pattern="*.ckpt")
        vocab_path = get_unique_file_in_dir(self.model_dir, pattern="*.txt")

        with suppress_outputs():
            self.model = TransformerModel.load_from_checkpoint(chkpt_path, vocab_path=vocab_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Configure the inference parameters.
        self.beam_size = beam_size
        self.augmentation_size = augmentation_size
        self.max_generated_seq_len = max_generated_seq_len
        assert (
            self.max_generated_seq_len <= self.model.max_seq_len
        ), f"The maximum generated sequence length should be less than or equal to the maximum sequence length supported by the model (e.g., for position encoding): {self.model.max_seq_len}."

        self.probability_from_score_temperature = probability_from_score_temperature
        self.filter_duplicate_augmentations = filter_duplicate_augmentations

        self.model.tokenizer.log_vocab_info()
        logger.info(f"Beam size: {self.beam_size}")
        logger.info(f"Augmentation size: {self.augmentation_size}")
        logger.info(f"Maximum generated sequence length: {self.max_generated_seq_len}")
        logger.info(f"Filter duplicate augmentations: {self.filter_duplicate_augmentations}")

    def get_parameters(self):
        return self.model.parameters()

    def _build_kwargs_from_scores(
        self, scores: list[float], additional_info: dict
    ) -> list[ReactionMetaData]:
        from syntheseus.reaction_prediction.inference.root_aligned import RootAlignedModel

        kwargs_list = RootAlignedModel.build_prediction_kwargs_from_scores(
            scores,
            num_augmentations=self.augmentation_size,
            beam_size=self.beam_size,
            probability_from_score_temperature=self.probability_from_score_temperature,
        )

        position_prob_tuples = additional_info["position_prob_tuples"]
        adaptive_augmentation_size = additional_info["adaptive_augmentation_size"]
        effective_augmentation_size = additional_info["effective_augmentation_size"]

        assert len(scores) == len(
            position_prob_tuples
        ), f"Number of scores ({len(scores)}) does not match number of info entries ({len(position_prob_tuples)})"

        for kwargs, position_prob_tuple in zip(kwargs_list, position_prob_tuples):
            kwargs.update(
                {  # type: ignore[misc]
                    "position_prob_tuple": position_prob_tuple,
                    "adaptive_augmentation_size": adaptive_augmentation_size,
                    "effective_augmentation_size": effective_augmentation_size,
                }
            )

        return kwargs_list

    def _smiles_to_batch(self, smiles: list[str]) -> dict[str, Any]:
        token_ids, padding_mask = self.model.tokenizer.encode(smiles, pad=True, right_padding=True)

        # Convert inputs to the model to tensors.
        return {
            "encoder_input": torch.tensor(token_ids),  # shape: (batch_size, src_len)
            "encoder_padding_mask": torch.tensor(
                padding_mask, dtype=torch.bool
            ),  # shape: (batch_size, src_len)
        }

    @abstractmethod
    def _augment_input(self, input: InputType) -> list[str]:
        pass

    @abstractmethod
    def _process_raw_smiles_outputs(
        self, input: InputType, output_list: list[str], metadata_list: list[ReactionMetaData]
    ) -> Sequence[ReactionType]:
        pass

    def _get_reactions(
        self, inputs: list[InputType], num_results: int
    ) -> list[Sequence[ReactionType]]:
        from retrochimera.utils.root_aligned_score import (
            canonicalize_smiles_clear_map,
            compute_rank,
        )

        # Step 1: Perform data augmentation on the input side (and convert to SMILES along the way).
        augmented_inputs: list[str] = []

        # Set up `opt` for the `compute_rank` function, originally from `root_aligned.score`.
        opt = argparse.Namespace()
        setattr(opt, "synthon", False)
        setattr(opt, "beam_size", self.beam_size)

        adaptive_augmentation_sizes = []
        for input in inputs:
            augmented_input = self._augment_input(input)

            if self.filter_duplicate_augmentations:
                augmented_input = list(dict.fromkeys(augmented_input))

            augmented_inputs.extend(augmented_input)
            adaptive_augmentation_sizes.append(len(augmented_input))

        # Step 2: Map from `InputType` class to `torch.Tensor`s.
        augmented_batch = self._smiles_to_batch(augmented_inputs)

        augmented_batch_to_device = {
            key: val.to(self.device) if type(val) == torch.Tensor else val  # type: ignore[attr-defined]
            for key, val in augmented_batch.items()
        }

        # We have to set `num_beams` as an attribute of the model.
        self.model.num_beams = self.beam_size
        translator = Translator(
            model=self.model,
            tokenizer=self.model.tokenizer,
            n_best=self.beam_size,
            max_length=self.max_generated_seq_len,
            beam_size=self.beam_size,
        )

        batch = {}
        src = (
            augmented_batch_to_device["encoder_input"].transpose(0, 1).contiguous().unsqueeze(2),
            torch.sum(~augmented_batch_to_device["encoder_padding_mask"], dim=1),
        )
        batch_size = src[0].size(1)
        assert src[0].size(1) == src[1].size(0)
        batch["src"] = src  # tuple[Tensor, Tensor]: (padded_src_len, batch_size, 1), (batch_size,)
        batch["batch_size"] = batch_size

        translate_results = translator.translate_batch(batch, attn_debug=False)
        augmented_batch_output_token_ids = translate_results[
            "predictions"
        ]  # list[list[LongTensor]]: For each batch, holds a list of beam prediction sequences
        augmented_batch_smiles = augmented_batch_output_token_ids

        # Obtain probabilities of the predictions.
        augmented_batch_log_probs = translate_results["scores"]  # list[list[float]]
        augmented_batch_scores = [
            [math.exp(log_prob) for log_prob in log_probs]
            for log_probs in augmented_batch_log_probs
        ]  # shape: (input_size * adaptive_augmentation_size, num_beams)

        lines = []  # shape: (input_size * adaptive_augmentation_size * num_beams)
        for i in range(len(augmented_batch_smiles)):
            for j in range(len(augmented_batch_smiles[i])):
                line = self.model.tokenizer.decode([augmented_batch_smiles[i][j].tolist()])
                assert isinstance(line, list)
                assert len(line) == 1
                assert isinstance(line[0], str)
                lines.append((line[0], augmented_batch_scores[i][j]))

        raw_predictions = []
        pool = multiprocessing.Pool(4)

        raw_predictions = pool.map(
            func=canonicalize_smiles_clear_map, iterable=lines
        )  # canonicalize reactants and modify illegal reactants into empty strings
        pool.close()
        pool.join()

        predictions = []
        left_index = 0
        for augmentation_size in adaptive_augmentation_sizes:
            curr_input_predictions = []
            for i in range(augmentation_size):
                curr_input_predictions.append(
                    raw_predictions[left_index : left_index + self.beam_size]
                )
                left_index += self.beam_size
            predictions.append(curr_input_predictions)

        # Rank legal reactants from all augmentations and beams.
        ranked_results = []
        ranked_scores = []
        ranked_additional_info = []

        for i in range(len(predictions)):
            (
                rank,
                position_prob_info,
                adaptive_augmentation_size,
                effective_augmentation_size,
            ) = compute_rank(predictions[i], opt=opt)
            # `adaptive_augmentation_size` is the number of augmentations performed on the input product molecule.
            # `effective_augmentation_size` is the number of augmentations where the transformer model returns at least one valid SMILES prediction in the beam.
            # `rank` is a dictionary with (reactant_smiles, max_frag_smiles) as keys and their computed scores for ranking as values.
            rank_list = list(zip(rank.keys(), rank.values()))
            rank_list.sort(key=lambda x: x[1], reverse=True)
            rank_list = rank_list[:num_results]  # truncate to `num_results` results
            ranked_results.append([item[0][0] for item in rank_list])  # output reactant SMILES
            ranked_scores.append([item[1] for item in rank_list])  # output scores used for ranking
            ranked_additional_info.append(
                {
                    "position_prob_tuples": [position_prob_info[item[0]] for item in rank_list],
                    "adaptive_augmentation_size": adaptive_augmentation_size,
                    "effective_augmentation_size": effective_augmentation_size,
                }
            )

        return [
            self._process_raw_smiles_outputs(
                input, outputs, self._build_kwargs_from_scores(scores, additional_info)
            )
            for input, outputs, scores, additional_info in zip(
                inputs, ranked_results, ranked_scores, ranked_additional_info
            )
        ]

    def compute_probs(
        self, reaction_smiles: list[str], minibatch_size: int = 32
    ) -> tuple[list[float], list[float]]:
        """Compute total and average probabilities for a list of reaction SMILES strings."""
        return self.model.compute_probs(reaction_smiles, minibatch_size=minibatch_size)


class SmilesTransformerModel(
    AbstractSmilesTransformerModel[Molecule, SingleProductReaction], ExternalBackwardReactionModel
):
    def _augment_input(self, input: Molecule) -> list[str]:
        from root_aligned.preprocessing.generate_PtoR_data import clear_map_canonical_smiles

        augmented_input = []

        product_roots = get_product_roots(
            product_atom_ids=[i + 1 for i in range(input.rdkit_mol.GetNumAtoms())],
            num_augmentations=self.augmentation_size,
        )

        for pro_root_atom_id in product_roots:
            pro_root = pro_root_atom_id - 1
            if pro_root_atom_id <= 0:
                pro_root = -1

            augmented_input.append(
                clear_map_canonical_smiles(input.smiles, canonical=True, root=pro_root)
            )

        return augmented_input

    def _process_raw_smiles_outputs(
        self, input: Molecule, output_list: list[str], metadata_list: list[ReactionMetaData]
    ) -> Sequence[SingleProductReaction]:
        return process_raw_smiles_outputs_backwards(input, output_list, metadata_list)
