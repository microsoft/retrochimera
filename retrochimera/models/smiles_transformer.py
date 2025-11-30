import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from more_itertools import batched
from torch import nn
from torch.nn.init import xavier_uniform_, zeros_
from torch.optim.lr_scheduler import OneCycleLR

from retrochimera.chem.rules import RuleBase
from retrochimera.data.preprocessing.smiles_transformer import ProcessedSample, preprocess_samples
from retrochimera.data.smiles_reaction_sample import SmilesReactionSample
from retrochimera.data.smiles_tokenizer import Tokenizer
from retrochimera.models.lightning import AbstractModel
from retrochimera.opennmt.modules.transformer_decoder import TransformerDecoder
from retrochimera.opennmt.modules.transformer_encoder import TransformerEncoder
from retrochimera.utils.logging import get_logger
from retrochimera.utils.pytorch import FuncLR

logger = get_logger(__name__)


@dataclass
class Batch:
    encoder_input: torch.Tensor  # (batch_size, src_len), dtype=torch.int
    encoder_padding_mask: torch.Tensor  # (batch_size, src_len), dtype=torch.bool, True/False
    decoder_input: torch.Tensor  # (batch_size, tgt_len-1), dtype=torch.int
    decoder_padding_mask: torch.Tensor  # (batch_size, tgt_len-1), dtype=torch.bool, True/False
    target: torch.Tensor  # (batch_size, tgt_len-1), dtype=torch.int
    target_mask: torch.Tensor  # (batch_size, tgt_len-1), dtype=torch.bool, True/False


class SmilesTransformerModel(AbstractModel[SmilesReactionSample, ProcessedSample, Batch]):
    """Transformer model operating on product/reactants in SMILES form."""

    DEFAULT_VOCAB_FILE_NAME = "vocab.txt"

    def __init__(
        self,
        vocab_path: str,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        feedforward_dim: int = 2048,
        activation: str = "gelu",
        max_seq_len: int = 512,
        dropout: float = 0.1,
        positional_encoding_type: str = "SinusoidalInterleaved",
        label_smoothing: float = 0.0,
        share_encoder_decoder_input_embedding: bool = True,
        initialization: str = "xavier",
        add_qkvbias: bool = False,
        layer_norm: str = "standard",
        num_kv: int = 0,
        parallel_residual: bool = False,
        schedule: str = "noam",
        warm_up_steps: int = 8000,
        n_classes: int = 0,
        total_steps: int = 100000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Build the tokenizer.
        logger.info(f"Loaded tokenizer from {vocab_path}")
        self.tokenizer = Tokenizer.from_vocab_file(vocab_path)
        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.begin_token_id = self.tokenizer.begin_token_id
        self.end_token_id = self.tokenizer.end_token_id

        # Model hyperparameters.
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.feedforward_dim = feedforward_dim
        self.activation = activation
        self.dropout = dropout
        self.positional_encoding_type = positional_encoding_type
        self.label_smoothing = label_smoothing
        self.share_encoder_decoder_input_embedding = share_encoder_decoder_input_embedding
        self.initialization = initialization
        self.add_qkvbias = add_qkvbias
        self.layer_norm = layer_norm
        self.num_kv = num_kv
        self.parallel_residual = parallel_residual

        self.max_seq_len = max_seq_len
        self.n_classes = n_classes  # number of classes for template prediction
        self.total_steps = total_steps  # for some lr schedulers
        self.dropout_layer = nn.Dropout(dropout)

        # 1. Token embedding and positional encoding.
        if self.share_encoder_decoder_input_embedding:
            self.token_embedding = nn.Embedding(
                self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id
            )
        else:
            self.encoder_token_embedding = nn.Embedding(
                self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id
            )
            self.decoder_token_embedding = nn.Embedding(
                self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id
            )

        self.register_buffer(
            "positional_encoding", self.absolute_positional_encoding()
        )  # fixed sinusoidal APE (Vaswani et al., 2017)

        # 2. Encoder and decoder.
        self.encoder = TransformerEncoder(
            num_layers=self.n_layers,
            d_model=self.hidden_dim,
            heads=self.n_heads,
            d_ff=self.feedforward_dim,
            dropout=self.dropout,
            attention_dropout=self.dropout,
            pos_ffn_activation_fn=self.activation,
            add_qkvbias=self.add_qkvbias,
            add_ffnbias=True,
            layer_norm=self.layer_norm,  # "standard" or "rms"
            num_kv=self.num_kv,
            parallel_residual=self.parallel_residual,
        )

        self.decoder = TransformerDecoder(
            num_layers=self.n_layers,
            d_model=self.hidden_dim,
            heads=self.n_heads,
            d_ff=self.feedforward_dim,
            dropout=self.dropout,
            attention_dropout=self.dropout,
            pos_ffn_activation_fn=self.activation,
            add_qkvbias=self.add_qkvbias,
            add_ffnbias=True,
            layer_norm=self.layer_norm,  # "standard" or "rms"
            num_kv=self.num_kv,
            parallel_residual=self.parallel_residual,
        )

        self.token_fc = nn.Linear(self.hidden_dim, self.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.pad_token_id, label_smoothing=self.label_smoothing
        )

        self._init_params()

        # Training hyperparameters.
        self.learning_rate = kwargs["learning_rate"]
        self.optimizer_betas = kwargs["optimizer_betas"]
        self.schedule = schedule
        self.warm_up_steps = warm_up_steps  # for some lr schedulers

        # Inference/sampling hyperparameters.
        self.num_beams = 10
        self.bad_token_log_prob = -1e5

    @property
    def hyperparameters_excluded_from_checkpoint(self) -> list[str]:
        return ["vocab_path"]

    def forward(self, batch_input: Batch) -> dict[str, torch.Tensor]:
        """Run a batch of inputs through the model.

        Args:
            batch_input: Batch from `collate` which contains the following attributes:
                encoder_input: Integer tensor of token IDs with shape `(batch_size, src_len)`.
                encoder_padding_mask: Boolean tensor of padding masks with shape
                    `(batch_size, src_len)`.
                decoder_input: Integer tensor of decoder token IDs with shape
                    `(batch_size, tgt_len-1)`.
                decoder_padding_mask: Boolean tensor of decoder padding masks with shape
                    `(batch_size, tgt_len-1)`.
                target: Unused in this function.
                target_mask: Unused in this function.

        Returns:
            Dictionary with keys "token_output" and "model_output".
        """
        # Encoder
        encode_dict = {
            "encoder_input": batch_input.encoder_input,  # shape: (batch_size, src_len)
            "encoder_padding_mask": batch_input.encoder_padding_mask,  # shape: (batch_size, src_len)
        }
        memory = self.encode(encode_dict)  # shape: (batch_size, src_len, hidden_dim)

        self.decoder.init_state(
            batch_input.encoder_input.transpose(0, 1).contiguous().unsqueeze(2), None, None
        )

        # Decoder
        decode_dict = {
            "decoder_input": batch_input.decoder_input,
            "decoder_padding_mask": batch_input.decoder_padding_mask,
            "memory_input": memory,
            "memory_padding_mask": batch_input.encoder_padding_mask.clone(),
        }
        model_output, token_output, token_log_probs = self.decode(decode_dict)
        output = {
            "model_output": model_output,
            "token_output": token_output,
            "token_log_probs": token_log_probs,
        }

        return output

    def ttv_step(self, batch: Batch, step_name: str):
        """Training, validation, and test step."""
        if step_name == "train":
            self.train()
        else:
            self.eval()

        batch_size = batch.encoder_input.shape[0]

        model_output = self.forward(batch)
        loss = self.calculate_loss(batch, model_output)
        with torch.no_grad():
            token_acc = self.calculate_token_accuracy(batch, model_output)
            perplexity = self.calculate_perplexity(batch, model_output)

        self.log(
            f"{step_name}_loss",
            loss,
            prog_bar=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{step_name}_token_acc",
            token_acc,
            prog_bar=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{step_name}_perplexity",
            perplexity,
            prog_bar=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def construct_input(
        self, token_ids: torch.Tensor, step: Optional[int] = None, is_encoder: bool = True
    ) -> torch.Tensor:
        """Construct an input embedding for a given token id tensor.

        Args:
            token_ids: Integer tensor of token IDs with shape (batch_size, seq_len) or (batch_size, seq_len, 1).
            step: Step for positional encoding.
            is_encoder: Whether the input is for the encoder or decoder.

        Returns:
            embs: Tensor of input embeddings with shape `(batch_size, seq_len, hidden_dim)`.
        """
        if token_ids.dim() == 3:  # the case of beam search for decoder input
            assert token_ids.size(2) == 1
            token_ids = token_ids.squeeze(2)
        elif token_ids.dim() != 2:
            raise ValueError(
                f"Expected token_ids to have shape (batch_size, seq_len) or (batch_size, seq_len, 1), got {token_ids.shape}"
            )

        _, seq_len = tuple(token_ids.size())

        if (
            self.share_encoder_decoder_input_embedding
        ):  # shared token embedding for encoder and decoder
            token_embs = self.token_embedding(token_ids)
        else:  # separate embedding for encoder and decoder
            if is_encoder:
                token_embs = self.encoder_token_embedding(
                    token_ids
                )  # shape: (batch_size, seq_len, hidden_dim)
            else:
                token_embs = self.decoder_token_embedding(
                    token_ids
                )  # shape: (batch_size, seq_len, hidden_dim)

        assert token_embs.size(1) == seq_len and token_embs.size(2) == self.hidden_dim

        token_embs = token_embs * math.sqrt(
            self.hidden_dim
        )  # scaling the embeddings like this is done in other transformer libraries

        # Start from step.
        step = step or 0
        assert (
            self.positional_encoding.shape[0] >= step + token_embs.shape[1]
        ), f"Sequence is {step + token_embs.shape[1]} but PositionalEncoding is {self.positional_encoding.shape[0]}. See max_len argument."

        positional_embs = self.positional_encoding[step : step + seq_len, :].unsqueeze(
            0
        )  # shape: (1, seq_len, hidden_dim)
        embs = token_embs + positional_embs  # shape: (batch_size, seq_len, hidden_dim)
        embs = self.dropout_layer(embs)

        return embs

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Construct a memory embedding for an encoder input.

        Args:
            batch: Dictionary containing two keys:
                "encoder_input": Integer tensor of token IDs with shape (batch_size, src_len)
                "encoder_padding_mask": Boolean tensor of padding masks with shape (batch_size, src_len)

        Returns:
            Tensor of memory embedding with shape `(src_len, batch_size, hidden_dim)`.
        """
        encoder_input = batch["encoder_input"]  # shape: (batch_size, src_len)
        encoder_padding_mask = batch["encoder_padding_mask"]  # shape: (batch_size, src_len)
        encoder_embs = self.construct_input(
            encoder_input, is_encoder=True
        )  # shape: (batch_size, src_len, hidden_dim)
        model_output = self.encoder.forward(
            src=encoder_embs, src_key_padding_mask=encoder_padding_mask
        )  # shape: (batch_size, src_len, hidden_dim)

        return model_output

    def decode(
        self,
        batch: dict[str, torch.Tensor],
        start_pos: int = 0,
        beam_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct an output from a given decoder input.

        Args:
            batch: Dictionary containing four keys:
                "decoder_input": Integer tensor of decoder token IDs with shape (batch_size, tgt_len-1)
                "decoder_padding_mask": Boolean tensor of decoder padding masks with shape (batch_size, tgt_len-1)
                "memory_input": Integer memory embedding tensor with shape (batch_size, src_len, hidden_dim)
                "memory_padding_mask": Boolean tensor of memory padding masks with shape (batch_size, src_len)
            start_pos: Starting position of decoder input for attention caching.

        Returns:
            Three decoder output tensors, each with shape (tgt_len-1, batch_size, hidden_dim),
            corresponding to raw output, token logits, and token log-probabilities, respectively.
        """
        decoder_input = batch["decoder_input"]
        decoder_padding_mask = batch["decoder_padding_mask"]  # shape: (batch_size, tgt_len-1)
        memory_input = batch["memory_input"]  # shape: (batch_size, src_len, hidden_dim)
        memory_padding_mask = batch["memory_padding_mask"]  # shape: (batch_size, src_len)

        decoder_embs = self.construct_input(
            decoder_input, is_encoder=False
        )  # shape: (batch_size, tgt_len-1, hidden_dim)

        _, seq_len, _ = tuple(decoder_embs.size())

        model_output, _ = self.decoder.forward(
            tgt=decoder_embs,
            tgt_key_padding_mask=decoder_padding_mask,
            enc_out=memory_input,
            enc_key_padding_mask=memory_padding_mask,
        )  # shape: (batch_size, tgt_len-1, hidden_dim)
        token_output = self.token_fc(model_output)  # shape: (batch_size, tgt_len - 1, hidden_dim)
        assert (
            token_output.size(1) == seq_len and token_output.size(2) == self.vocab_size
        ), f"{token_output.size()}"

        token_log_probs = self.log_softmax(
            token_output
        )  # shape: (batch_size, tgt_len - 1, vocab_size)

        return model_output, token_output, token_log_probs

    def absolute_positional_encoding(self) -> torch.Tensor:
        """Construct a tensor of positional embeddings from sine/cosine waves of varying wavelength.

        Returns:
            Tensor of shape `(self.max_seq_len, self.hidden_dim)` filled with positional embeddings.
        """
        if self.hidden_dim % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd dim {self.hidden_dim}"
            )

        if self.positional_encoding_type == "SinusoidalInterleaved":
            pe = torch.zeros(self.max_seq_len, self.hidden_dim)  # shape: (max_seq_len, hidden_dim)
            position = torch.arange(0, self.max_seq_len).unsqueeze(1)  # shape: (max_seq_len, 1)
            div_term = torch.exp(
                (
                    torch.arange(0, self.hidden_dim, 2, dtype=torch.float)
                    * -(math.log(10000.0) / self.hidden_dim)
                )
            )  # shape: (hidden_dim // 2)
            pe[:, 0::2] = torch.sin(
                position.float() * div_term
            )  # shape: (max_seq_len, hidden_dim//2)
            pe[:, 1::2] = torch.cos(
                position.float() * div_term
            )  # shape: (max_seq_len, hidden_dim//2)
        elif self.positional_encoding_type == "SinusoidalConcat":
            half_dim = self.hidden_dim // 2
            pe = math.log(10000) / (half_dim - 1)
            pe = torch.exp(
                torch.arange(half_dim, dtype=torch.float) * -pe
            )  # shape: (hidden_dim//2)
            pe = torch.arange(self.max_seq_len, dtype=torch.float).unsqueeze(1) * pe.unsqueeze(
                0
            )  # shape: (max_seq_len, hidden_dim//2)
            pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=1).view(
                self.max_seq_len, -1
            )  # shape: (max_seq_len, hidden_dim)
        else:
            raise ValueError(
                "Position encoding should be SinusoidalInterleaved or SinusoidalConcat."
            )
        return pe

    def compute_probs(
        self, reaction_smiles: list[str], minibatch_size: int = 32
    ) -> tuple[list[float], list[float]]:
        """Compute probabilities for a list of reaction SMILES strings.

        Args:
            reaction_smiles: List of reaction SMILES strings, where each reaction SMILES string has the format:
                reactant1.reactant2>reagent1.reagent2>product1.product2
        Returns:
            Lists of probabilities (total and average) of the reactions.
        """
        return_total_probs = []
        return_avg_probs = []
        for reaction_smiles_minibatch in batched(reaction_smiles, minibatch_size):
            # 1. Obtain the product and reactant SMILES strings in the minibatch.
            product_smi = [smi.split(">")[2] for smi in reaction_smiles_minibatch]
            reactants_smi = [smi.split(">")[0] for smi in reaction_smiles_minibatch]

            # 2. Tokenize the product and reactant SMILES strings and their masks.
            products_token_ids, products_masks = self.tokenizer.encode(
                product_smi, pad=True, right_padding=True
            )
            reactants_token_ids, reactants_masks = self.tokenizer.encode(
                reactants_smi, pad=True, right_padding=True, add_begin_token=True
            )

            products_token_ids = torch.tensor(products_token_ids)
            products_masks = torch.tensor(products_masks, dtype=torch.bool)
            reactants_token_ids = torch.tensor(reactants_token_ids)
            reactants_masks = torch.tensor(reactants_masks, dtype=torch.bool)

            batch = Batch(
                encoder_input=products_token_ids.to(self.device),
                encoder_padding_mask=products_masks.to(self.device),
                decoder_input=reactants_token_ids[:, :-1].to(self.device),
                decoder_padding_mask=reactants_masks[:, :-1].to(self.device),
                target=reactants_token_ids.clone()[:, 1:].to(self.device),
                target_mask=reactants_masks.clone()[:, 1:].to(self.device),
            )

            model_output = self.forward(batch)
            total_probs, avg_probs = self.calculate_probs(batch, model_output)
            return_total_probs.extend(total_probs.tolist())
            return_avg_probs.extend(avg_probs.tolist())

        return return_total_probs, return_avg_probs

    def calculate_loss(
        self, batch_input: Batch, model_output: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate the loss function to update the model.

        Args:
            batch_input: Input given to the model.
            model_output: Output from the model.

        Returns:
            Resulting loss.
        """

        tokens = batch_input.target.transpose(0, 1)
        padding_mask = batch_input.target_mask.transpose(0, 1)
        token_output = model_output["token_output"].transpose(0, 1)

        return self._calculate_mask_loss(token_output, tokens, padding_mask)

    def _calculate_mask_loss(
        self, token_output: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss function.

        Args:
            token_output: Tensor of shape `(tgt_len-1, batch_size, vocab_size)` containing token
                outputs from the transformer.
            target: Tensor of shape `(tgt_len-1, batch_size)` containing original (unmasked) SMILES
                token IDs from the tokenizer.
            target_mask: Tensor of shape `(tgt_len-1, batch_size)` containing padding mask for
                target tokens.

        Returns:
            Loss computed using cross-entropy.
        """

        seq_len, batch_size = tuple(target.size())
        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def calculate_probs(
        self, batch_input: Batch, model_output: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the probabilities (including total probability and average probability of a target sequence)."""
        target_ids = batch_input.target.transpose(0, 1)
        target_mask = batch_input.target_mask.transpose(0, 1)
        token_log_probs = model_output["token_log_probs"].transpose(0, 1)

        inv_target_mask = ~(target_mask > 0)
        log_probs = token_log_probs.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask

        total_log_probs = log_probs.sum(dim=0)
        total_probs = torch.exp(total_log_probs)

        num_tokens = inv_target_mask.sum(dim=0)
        probs = torch.exp(log_probs) * inv_target_mask
        probs = probs.sum(dim=0)
        avg_probs = probs / num_tokens.float()

        return total_probs, avg_probs

    def calculate_perplexity(
        self, batch_input: Batch, model_output: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate the perplexity."""
        target_ids = batch_input.target.transpose(0, 1)  # shape: (tgt_len-1, batch_size)
        target_mask = batch_input.target_mask.transpose(0, 1)  # shape: (tgt_len-1, batch_size)
        token_log_probs = model_output["token_log_probs"].transpose(
            0, 1
        )  # shape: (tgt_len-1, batch_size, vocab_size)

        inv_target_mask = ~(target_mask > 0)
        log_probs = token_log_probs.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)  # shape: (batch_size,)

        seq_lengths = inv_target_mask.sum(dim=0)  # shape: (batch_size,)
        perp = torch.exp(-log_probs / seq_lengths)

        return perp.mean()

    def calculate_token_accuracy(
        self, batch_input: Batch, model_output: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate the token accuracy."""
        token_ids = batch_input.target.transpose(0, 1)  # shape: (tgt_len-1, batch_size)
        target_mask = batch_input.target_mask.transpose(0, 1)  # shape: (tgt_len-1, batch_size)
        token_output = model_output["token_output"].transpose(
            0, 1
        )  # shape: (tgt_len-1, batch_size, vocab_size)

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()
        accuracy = num_correct / total

        return accuracy

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        logger.info("Configuring optimizer")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Betas: {self.optimizer_betas}")
        logger.info(f"Schedule: {self.schedule}")

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, betas=self.optimizer_betas
        )

        if self.schedule == "noam":
            logger.info("Using original transformer schedule (noam)")
            lr_scheduler = FuncLR(optimizer, lr_lambda=self._noam_lr)
        elif self.schedule == "cosine":
            logger.info("Using cosine LR schedule")
            lr_scheduler = FuncLR(optimizer, lr_lambda=self._warmup_cosine_lr)
        elif self.schedule == "constant":
            logger.info("Using constant LR schedule")
            lr_scheduler = FuncLR(optimizer, lr_lambda=self._constant_lr)
        elif self.schedule == "cycle":
            logger.info("Using cyclical LR schedule")
            lr_scheduler = OneCycleLR(optimizer, self.learning_rate, total_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown schedule {self.schedule}")
        sch = {"scheduler": lr_scheduler, "interval": "step"}

        return [optimizer], [sch]

    def _constant_lr(self, step: int) -> float:
        if step < self.warm_up_steps:
            return (self.learning_rate / int(self.warm_up_steps)) * step

        return self.learning_rate

    def _noam_lr(self, step: int) -> float:
        mult = self.hidden_dim**-0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step**-0.5, step * (self.warm_up_steps**-1.5))

        return self.learning_rate * mult * lr

    def _warmup_cosine_lr(self, step: int, linear_decay: bool = False) -> float:
        if step < self.warm_up_steps:
            return (self.learning_rate / int(self.warm_up_steps)) * step
        else:
            if linear_decay:
                return max(
                    0.0,
                    float(self.total_steps - step)
                    / float(max(1.0, self.total_steps - self.warmup_num_steps)),
                )

            return 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * float(step - self.warmup_num_steps)
                    / float(max(1.0, self.total_steps - self.warmup_num_steps))
                )
            )

    def _init_params(self) -> None:
        """Apply Xavier uniform initialisation of learnable weights."""

        # Analyze model parameters.
        logger.info(f"Total parameter size: {sum(p.numel() for p in self.parameters())}")
        for name, param in self.named_parameters():
            logger.debug(f"{name}: {param.size()}")

        # Initialize model parameters.
        if self.initialization == "xavier":
            logger.info("Using Xavier uniform initialization")
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
        elif self.initialization == "opennmt_xavier":
            logger.info("Using OpenNMT-py Xavier uniform initialization")
            for param_name, param in self.named_parameters():
                if param_name.endswith("weight") and param.dim() > 1:
                    xavier_uniform_(param)
                elif param_name.endswith("bias"):
                    zeros_(param)
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")

    def collate(self, samples: list[ProcessedSample]) -> Batch:
        """Collate a list of samples into a batch."""
        products_token_ids, products_masks = self.tokenizer.pad_token_ids_list(
            [sample.encoded_products for sample in samples], right_padding=True
        )
        products_token_ids = torch.tensor(products_token_ids)
        products_masks = torch.tensor(products_masks, dtype=torch.bool)

        decoder_token_ids, decoder_masks = self.tokenizer.pad_token_ids_list(
            [sample.encoded_reactants for sample in samples], right_padding=True
        )
        decoder_token_ids = torch.tensor(decoder_token_ids)
        decoder_masks = torch.tensor(decoder_masks, dtype=torch.bool)

        return Batch(
            encoder_input=products_token_ids,  # shape: (batch_size, src_len)
            encoder_padding_mask=products_masks,  # shape: (batch_size, src_len)
            decoder_input=decoder_token_ids[:, :-1],  # shape: (batch_size, tgt_len-1)
            decoder_padding_mask=decoder_masks[:, :-1],  # shape: (batch_size, tgt_len-1)
            target=decoder_token_ids.clone()[:, 1:],  # shape: (batch_size, tgt_len-1)
            target_mask=decoder_masks.clone()[:, 1:],  # shape: (batch_size, tgt_len-1)
        )

    def preprocess(
        self, samples: Iterable[SmilesReactionSample], num_processes: int = 0
    ) -> Iterable[ProcessedSample]:
        yield from preprocess_samples(
            samples=samples,
            rulebase_dir=self.rulebase_dir,
            tokenizer=self.tokenizer,
            num_processes=num_processes,
        )

    def set_rulebase(self, rulebase: RuleBase, rulebase_dir: Union[str, Path]) -> None:
        # Inherited from other template-based models (Not used in this model now)
        self.rulebase_dir = rulebase_dir
