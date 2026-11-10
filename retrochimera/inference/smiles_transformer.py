import argparse
import math
from abc import abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Generic, Literal, Optional, Sequence, TypeVar, Union

import torch
from loky import get_reusable_executor
from syntheseus import Molecule, Reaction, SingleProductReaction
from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.reaction import ReactionMetaData
from syntheseus.reaction_prediction.inference_base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
)
from syntheseus.reaction_prediction.utils.misc import cpu_count, suppress_outputs

from retrochimera.models.smiles_transformer import SmilesTransformerModel as TransformerModel
from retrochimera.opennmt.decode.translator import Translator
from retrochimera.utils.logging import get_logger
from retrochimera.utils.root_aligned import clear_map_canonical_smiles, get_product_roots

logger = get_logger(__name__)


InputType = TypeVar("InputType")
ReactionType = TypeVar("ReactionType", bound=Reaction)


class AbstractSmilesTransformerModel(Generic[InputType, ReactionType]):
    def __init__(
        self,
        model: TransformerModel,
        *,
        beam_size: int = 20,
        augmentation_size: int = 10,
        max_generated_seq_len: int = 512,
        probability_from_score_temperature: float = 3.0,
        filter_duplicate_augmentations: bool = True,
        canonicalization_processes: int = min(16, max(1, cpu_count() // 2)),
        inference_precision: Literal["float32", "float16", "bfloat16"] = "float32",
        **kwargs,
    ) -> None:
        """Adapt an existing model without changing its device or mode.

        Decoding is shared between the backward and forward model variants.
        """
        super().__init__(**kwargs)
        self.model = model
        self.device = str(model.device)

        if inference_precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported inference_precision {inference_precision}")
        self._autocast_dtype = getattr(torch, inference_precision)

        if canonicalization_processes <= 0:
            raise ValueError("canonicalization_processes must be positive")
        self._canonicalization_processes = canonicalization_processes


        # Configure the inference parameters.
        self.beam_size = beam_size
        self.augmentation_size = augmentation_size
        self.max_generated_seq_len = max_generated_seq_len
        assert (
            self.max_generated_seq_len <= self.model.max_seq_len
        ), f"The maximum generated sequence length should be less than or equal to the maximum sequence length supported by the model (e.g., for position encoding): {self.model.max_seq_len}."

        self.probability_from_score_temperature = probability_from_score_temperature
        self.filter_duplicate_augmentations = filter_duplicate_augmentations


    def _autocast_context(self):
        if self._autocast_dtype == torch.float32 or not self.device.startswith("cuda"):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)

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
            seed=0,  # Leave RNG seeding and backend determinism to the caller.
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

        with self._autocast_context():
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

        # Keep workers reusable across batches; Loky retires them after five idle minutes
        # and restarts them on demand, so do not shut down the shared executor here.
        executor = get_reusable_executor(max_workers=self._canonicalization_processes, timeout=300)
        raw_predictions = list(
            executor.map(canonicalize_smiles_clear_map, lines, chunksize=32)
        )  # canonicalize reactants and modify illegal reactants into empty strings

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
        with torch.inference_mode(), self._autocast_context():
            return self.model.compute_probs(reaction_smiles, minibatch_size=minibatch_size)


class CheckpointSmilesTransformerModel:
    """Load a checkpoint before initializing the shared inference adapter."""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        model_dir = Path(model_dir or self.get_default_model_dir())
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        chkpt_path = get_unique_file_in_dir(model_dir, pattern="*.ckpt")
        vocab_path = get_unique_file_in_dir(model_dir, pattern="*.txt")
        with suppress_outputs():
            model = TransformerModel.load_from_checkpoint(chkpt_path, vocab_path=vocab_path)
        model = model.to(device)
        model.eval()

        super().__init__(model=model, model_dir=model_dir, device=device, **kwargs)

        self.model.tokenizer.log_vocab_info()
        logger.debug(f"Beam size: {self.beam_size}")
        logger.debug(f"Augmentation size: {self.augmentation_size}")
        logger.debug(f"Maximum generated sequence length: {self.max_generated_seq_len}")
        logger.debug(f"Filter duplicate augmentations: {self.filter_duplicate_augmentations}")
        logger.debug(f"Canonicalization processes: {self._canonicalization_processes}")
        logger.debug(
            f"Inference precision: {self._autocast_dtype}"
            if self._autocast_dtype != torch.float32 and self.device.startswith("cuda")
            else "Inference precision: inherited from caller"
        )


class LiveSmilesTransformerModel(
    AbstractSmilesTransformerModel[Molecule, SingleProductReaction], BackwardReactionModel
):
    def _augment_input(self, input: Molecule) -> list[str]:
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


class SmilesTransformerModel(
    CheckpointSmilesTransformerModel, LiveSmilesTransformerModel, ExternalBackwardReactionModel
):
    """Checkpoint-backed backward model; the directory contains one checkpoint and vocabulary."""
