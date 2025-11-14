import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import torch
from more_itertools import batched
from syntheseus.reaction_prediction.utils.misc import parallelize
from torch import nn
from torch.nn import functional as f
from torch_geometric.data import Batch as BatchType

from retrochimera.chem.rules import RuleBase
from retrochimera.data.preprocessing.template_localization import (
    ProcessedSample,
    preprocess_samples,
)
from retrochimera.data.template_reaction_sample import TemplateReactionSample
from retrochimera.encoders.base import get_encoder_by_name
from retrochimera.encoders.featurizers import RawRewriteGraph
from retrochimera.models.lightning import AbstractModel, log_classification_metrics
from retrochimera.utils.pytorch import split_select


@dataclass
class Batch:
    inputs: BatchType
    rewrites: BatchType
    targets: torch.Tensor
    targets_in_batch: torch.Tensor
    loc_targets: list[torch.Tensor]


class TemplateLocalizationModel(AbstractModel[TemplateReactionSample, ProcessedSample, Batch]):
    def __init__(
        self,
        input_encoder_class: str,
        rewrite_encoder_class: str,
        input_encoder_kwargs: dict[str, Any],
        rewrite_encoder_kwargs: dict[str, Any],
        classification_label_smoothing: float,
        localization_label_smoothing: float,
        classification_space_dim: Optional[int],
        free_rewrite_embedding_dim: int,
        classification_loss_type: str,
        classification_min_temperature: Optional[float],
        classification_max_temperature: Optional[float],
        negative_to_positive_targets_ratio: float,
        num_negative_rewrites_in_localization: int,
        n_classes: int,
        num_total_rewrite_lhs_atoms: int,
        rewrite_encoder_num_epochs: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert classification_loss_type in ["softmax", "sigmoid"]

        self.input_encoder = get_encoder_by_name(input_encoder_class)(**input_encoder_kwargs)
        self.rewrite_encoder = get_encoder_by_name(rewrite_encoder_class)(**rewrite_encoder_kwargs)

        if (classification_min_temperature is None) != (classification_max_temperature is None):
            raise ValueError(
                "Either both or neither of classification_{min,max}_temperature should be set"
            )

        self.classification_loss_type = classification_loss_type
        self.classification_min_temperature = classification_min_temperature
        self.classification_max_temperature = classification_max_temperature
        self.negative_to_positive_targets_ratio = negative_to_positive_targets_ratio
        self.num_negative_rewrites_in_localization = num_negative_rewrites_in_localization
        self.rewrite_encoder_num_epochs = rewrite_encoder_num_epochs

        if classification_loss_type == "softmax":
            self.classification_raw_temperature = None
        elif classification_loss_type == "sigmoid":
            # Initialize following "Sigmoid Loss for Language Image Pre-Training", unless a
            # temperature range is set, in which case initialize to the middle of that range.
            if classification_min_temperature is None:
                init_raw_temperature = math.log(10)
            else:
                init_raw_temperature = 0.0

            self.classification_raw_temperature = nn.Parameter(
                torch.full((1,), init_raw_temperature)
            )
            self.classification_bias = nn.Parameter(torch.full((1,), -10.0))
        else:
            raise ValueError(f"Unrecognized loss type {classification_loss_type}")

        assert self.input_encoder.mol_out_channels is not None
        assert self.rewrite_encoder.mol_out_channels is not None

        if free_rewrite_embedding_dim > 0:
            self.free_rewrite_embeddings = nn.Parameter(
                torch.zeros(n_classes, free_rewrite_embedding_dim)
            )
            rewrite_embedding_dim_total = (
                self.rewrite_encoder.mol_out_channels + free_rewrite_embedding_dim
            )
        else:
            self.free_rewrite_embeddings = None
            rewrite_embedding_dim_total = self.rewrite_encoder.mol_out_channels

        if classification_space_dim is not None:
            self.input_proj = nn.Linear(
                self.input_encoder.mol_out_channels, classification_space_dim
            )
            self.rewrite_proj = nn.Linear(rewrite_embedding_dim_total, classification_space_dim)
        else:
            assert self.input_encoder.mol_out_channels == rewrite_embedding_dim_total

            self.input_proj = nn.Identity()
            self.rewrite_proj = nn.Identity()

        if num_negative_rewrites_in_localization > 0:
            self.no_localization_embedding = nn.Parameter(
                torch.zeros(1, self.input_encoder.atom_out_channels)
            )
        else:
            self.no_localization_embedding = None

        self._classification_label_smoothing = classification_label_smoothing
        self._localization_label_smoothing = localization_label_smoothing
        self.n_classes = n_classes
        self.rulebase: Optional[RuleBase] = None
        self.rulebase_dir: Optional[Union[str, Path]] = None

        self.all_rewrites_batch_idx = nn.Parameter(
            torch.zeros(num_total_rewrite_lhs_atoms), requires_grad=False
        )
        self.all_rewrites_mol_outputs = nn.Parameter(
            torch.zeros(n_classes, self.rewrite_encoder.mol_out_channels), requires_grad=False
        )
        self.all_rewrites_atom_outputs = nn.Parameter(
            torch.zeros(num_total_rewrite_lhs_atoms, self.rewrite_encoder.atom_out_channels),
            requires_grad=False,
        )
        self._rewrite_encodings_up_to_date = False

        # Delay computing the rewrite graphs until we know we actually need it; we don't need it for
        # inference as we would have cached the results when saving the checkpoint.
        self._all_rewrite_graphs: Optional[list[RawRewriteGraph]] = None

    def set_rulebase(self, rulebase: RuleBase, rulebase_dir: Union[str, Path]) -> None:
        self.rulebase = rulebase
        self.rulebase_dir = rulebase_dir

    def _prepare_rule_weights(self) -> None:
        if self.classification_loss_type == "sigmoid":
            assert self.rulebase is not None

            positive_rule_frequency = torch.as_tensor(
                [rule.n_support for rule in self.rulebase.rules.values()], dtype=torch.float32
            )
            negative_rule_frequency = torch.ones_like(positive_rule_frequency)

            positive_rule_frequency /= positive_rule_frequency.sum()
            negative_rule_frequency /= negative_rule_frequency.sum()
            combined_rule_frequency = (
                positive_rule_frequency
                + self.negative_to_positive_targets_ratio * negative_rule_frequency
            )

            # Compute rule weights which are inversely proportional to frequency and average to 1.
            inverse_rule_frequency = 1 / combined_rule_frequency
            self._rule_weight = (
                inverse_rule_frequency / inverse_rule_frequency.sum() * self.n_classes
            )

    def _prepare_rewrite_graphs(self) -> None:
        if self._all_rewrite_graphs is None:
            assert self.rulebase is not None

            all_rewrites = [rule.rxn for rule in self.rulebase.rules.values()]
            self._all_rewrite_graphs = list(
                parallelize(self.rewrite_encoder.preprocess, all_rewrites, num_processes=0)
            )

    def on_fit_start(self) -> None:
        assert self.rulebase is not None

        self._prepare_rule_weights()
        self._prepare_rewrite_graphs()

        del self.rulebase

    def get_temperature(self) -> torch.Tensor:
        if self.classification_min_temperature is None:
            return torch.exp(self.classification_raw_temperature)
        else:
            assert self.classification_max_temperature is not None

            return self.classification_min_temperature + (
                self.classification_max_temperature - self.classification_min_temperature
            ) * torch.sigmoid(self.classification_raw_temperature)

    def forward_classification(
        self,
        input_reprs: torch.Tensor,
        rewrite_reprs: torch.Tensor,
        rewrite_ids: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        if self.free_rewrite_embeddings is not None:
            rewrite_reprs = torch.cat(
                [rewrite_reprs, self.free_rewrite_embeddings[rewrite_ids]], dim=-1
            )

        input_reprs = self.input_proj(input_reprs)
        rewrite_reprs = self.rewrite_proj(rewrite_reprs)

        if self.classification_loss_type == "sigmoid":
            input_reprs = f.normalize(input_reprs)
            rewrite_reprs = f.normalize(rewrite_reprs)

        logits = torch.mm(input_reprs, torch.t(rewrite_reprs))

        if self.classification_loss_type == "sigmoid":
            if temperature is None:
                temperature = self.get_temperature()

            logits *= temperature
            logits += self.classification_bias

        return logits

    def forward_localization(
        self,
        input_graphs: BatchType,
        rewrite_graphs_node_in_lhs: torch.Tensor,
        rewrite_graphs_batch: torch.Tensor,
        input_graphs_atom_outputs: torch.Tensor,
        rewrite_graphs_atom_outputs: torch.Tensor,
        targets_in_batch: torch.Tensor,
        negative_targets_in_batch: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        # Extract the representations of the lhs atoms, as these are the ones we need to localize.
        node_in_lhs_mask = rewrite_graphs_node_in_lhs.bool()
        rewrite_graphs_lhs_batch = torch.masked_select(rewrite_graphs_batch, node_in_lhs_mask)
        rewrite_graphs_lhs_atom_outputs = torch.masked_select(
            rewrite_graphs_atom_outputs, node_in_lhs_mask.unsqueeze(-1)
        ).view(-1, rewrite_graphs_atom_outputs.shape[-1])

        # Unbatch the atom representations to prepare for the localization step.
        input_atom_reprs_per_graph = split_select(input_graphs_atom_outputs, input_graphs.batch)
        rewrite_lhs_atom_reprs_per_graph = split_select(
            rewrite_graphs_lhs_atom_outputs, rewrite_graphs_lhs_batch
        )

        dot_prods_per_graph = []
        for graph_idx, graph_input_atom_reprs in enumerate(input_atom_reprs_per_graph):
            targets = [targets_in_batch[graph_idx]]

            if negative_targets_in_batch is not None:
                assert self.no_localization_embedding is not None
                graph_input_atom_reprs_ext = torch.cat(
                    [graph_input_atom_reprs, self.no_localization_embedding], dim=0
                )
                targets.extend(list(negative_targets_in_batch[graph_idx]))
            else:
                graph_input_atom_reprs_ext = graph_input_atom_reprs

            graph_rewrite_atom_reprs = torch.cat(
                [rewrite_lhs_atom_reprs_per_graph[idx] for idx in targets], dim=0
            )

            dot_prods = torch.mm(
                graph_rewrite_atom_reprs, torch.t(graph_input_atom_reprs_ext)
            )  # [num_loc_targets, num_atoms]
            dot_prods_per_graph.append(dot_prods)

        return dot_prods_per_graph

    def ttv_step(self, batch: Batch, step_name: str):
        input_graphs_enc = self.input_encoder(batch.inputs)

        batch_size = input_graphs_enc.mol_outputs.shape[0]
        device = input_graphs_enc.mol_outputs.device

        if step_name != "train" or (
            self.classification_loss_type == "softmax"
            and self.rewrite_encoder_num_epochs is not None
            and self.current_epoch >= self.rewrite_encoder_num_epochs
        ):
            if not self._rewrite_encodings_up_to_date:
                self._update_rewrite_encodings()

            # Use cached rewrite encodings for the entire rulebase.
            rewrite_graphs_mol_outputs = self.all_rewrites_mol_outputs
            rewrite_graphs_atom_outputs = self.all_rewrites_atom_outputs
            rewrite_graphs_node_in_lhs = torch.ones(
                self.all_rewrites_batch_idx.shape[0], device=device
            )
            rewrite_graphs_batch = self.all_rewrites_batch_idx

            # Include all templates in the loss.
            targets = torch.arange(self.n_classes, device=device)
            targets_in_batch = batch.targets[batch.targets_in_batch]

            # Call metrics "classification_loss", "acc", etc.
            metric_prefix = step_name
        else:
            self._rewrite_encodings_up_to_date = False

            # Call the rewrite encoder afresh as it is currently being trained.
            rewrite_graphs_enc = self.rewrite_encoder(batch.rewrites)
            rewrite_graphs_mol_outputs = rewrite_graphs_enc.mol_outputs
            rewrite_graphs_atom_outputs = rewrite_graphs_enc.atom_outputs
            rewrite_graphs_node_in_lhs = batch.rewrites.node_in_lhs
            rewrite_graphs_batch = batch.rewrites.batch

            # Include only the templates in the batch in the loss.
            targets = batch.targets
            targets_in_batch = batch.targets_in_batch

            # Call metrics "batch_classification_loss", "batch_acc", etc.
            metric_prefix = f"{step_name}_batch"

        classification_logits = self.forward_classification(
            input_reprs=input_graphs_enc.mol_outputs,
            rewrite_reprs=rewrite_graphs_mol_outputs,
            rewrite_ids=targets,
        )

        # `classification_targets` contains ones for correct input-rewrite combinations and zeros
        # elsewhere. Note that some columns (rewrites) may have multiple ones.
        targets_in_batch_flat = (
            targets_in_batch
            + torch.arange(batch_size, device=device) * classification_logits.shape[-1]
        )
        classification_targets = torch.zeros_like(classification_logits).ravel()
        classification_targets[targets_in_batch_flat] = 1.0
        classification_targets = classification_targets.reshape(classification_logits.shape)

        if self.classification_loss_type == "softmax":
            # Assymetric loss based on `https://arxiv.org/abs/2103.00020` (a symmetric loss would
            # not make sense as some rewrites may be negative samples with no matching inputs).
            classification_loss = f.cross_entropy(
                classification_logits,
                classification_targets,
                label_smoothing=self._classification_label_smoothing,
            )
        elif self.classification_loss_type == "sigmoid":
            # Loss variant based on `https://arxiv.org/abs/2303.15343`.
            # First compute appropriate weights for the pairwise losses.
            rule_weight = self._rule_weight.to(device)[targets]
            loss_weight = classification_targets + (
                1 - classification_targets
            ) * rule_weight.expand_as(classification_targets)

            # Convert {0, 1} to {-1, 1}.
            classification_targets = 2 * classification_targets - torch.ones_like(
                classification_targets
            )
            classification_pairwise_losses = -f.logsigmoid(
                classification_logits * classification_targets
            )

            if self._classification_label_smoothing > 0.0:
                # Label smoothing does not transfer directly to the sigmoid loss, but we do
                # something roughly similar: we make the "target" loss value a small positive
                # constant instead of 0.
                target_loss_value = -0.1 * math.log(1.0 - self._classification_label_smoothing)
                classification_pairwise_losses = torch.abs(
                    classification_pairwise_losses - target_loss_value
                )

            classification_loss = (loss_weight * classification_pairwise_losses).sum() / batch_size
        else:
            assert False

        assert input_graphs_enc.atom_outputs is not None
        assert input_graphs_enc.mol_outputs is not None

        if self.num_negative_rewrites_in_localization > 0:
            # Mask out the logits for the correct targets.
            classification_logits_masked = torch.clone(classification_logits.ravel())
            classification_logits_masked[targets_in_batch_flat] = -1e9
            classification_logits_masked = classification_logits_masked.reshape(
                classification_targets.shape
            )

            # Extract top *incorrect* targets for each input.
            negative_targets_in_batch = torch.topk(
                classification_logits_masked, k=self.num_negative_rewrites_in_localization
            ).indices
        else:
            negative_targets_in_batch = None

        # Run template localization for the ground-truth templates.
        dot_prods_per_graph = self.forward_localization(
            input_graphs=batch.inputs,
            rewrite_graphs_node_in_lhs=rewrite_graphs_node_in_lhs,
            rewrite_graphs_batch=rewrite_graphs_batch,
            input_graphs_atom_outputs=input_graphs_enc.atom_outputs,
            rewrite_graphs_atom_outputs=rewrite_graphs_atom_outputs,
            targets_in_batch=targets_in_batch,
            negative_targets_in_batch=negative_targets_in_batch,
        )

        localization_loss = torch.as_tensor(0.0, device=device)
        for graph_dot_prods, graph_loc_targets in zip(dot_prods_per_graph, batch.loc_targets):
            if self.num_negative_rewrites_in_localization > 0:
                graph_loc_targets_ext = torch.cat(
                    [graph_loc_targets, torch.zeros(len(graph_loc_targets), 1, device=device)],
                    dim=-1,
                )
                graph_loc_targets_negative = torch.zeros(
                    len(graph_dot_prods) - len(graph_loc_targets),
                    graph_loc_targets_ext.shape[-1],
                    device=device,
                )
                graph_loc_targets_negative[:, -1] = 1.0
                graph_loc_targets_ext = torch.cat(
                    [graph_loc_targets_ext, graph_loc_targets_negative], dim=0
                )
            else:
                graph_loc_targets_ext = graph_loc_targets

            graph_loc_targets_uniform = (
                torch.ones_like(graph_loc_targets_ext) / graph_loc_targets_ext.shape[-1]
            )
            graph_loc_targets_smooth = (
                (1.0 - self._localization_label_smoothing) * graph_loc_targets_ext
                + self._localization_label_smoothing * graph_loc_targets_uniform
            )
            localization_loss += f.kl_div(
                f.log_softmax(graph_dot_prods, dim=-1), graph_loc_targets_smooth, reduction="sum"
            )

        localization_loss /= batch_size * (self.num_negative_rewrites_in_localization + 1)
        loss = classification_loss + localization_loss

        if step_name == "train":
            if self.classification_loss_type == "sigmoid":
                self.log(
                    "classification_temperature", self.get_temperature(), batch_size=batch_size
                )
                self.log("classification_bias", self.classification_bias, batch_size=batch_size)

        self.log(f"{metric_prefix}_classification_loss", classification_loss, batch_size=batch_size)
        self.log(f"{metric_prefix}_localization_loss", localization_loss, batch_size=batch_size)
        self.log(f"{metric_prefix}_loss", loss, batch_size=batch_size)
        log_classification_metrics(
            log_fn=self.log,
            step_name=metric_prefix,
            batch_preds=classification_logits,
            batch_targets=targets_in_batch,
            batch_size=batch_size,
        )

        return loss

    def preprocess(
        self, samples: Iterable[TemplateReactionSample], num_processes: int = 0
    ) -> Iterable[ProcessedSample]:
        assert self.rulebase_dir is not None
        yield from preprocess_samples(
            samples=samples,
            rulebase_dir=self.rulebase_dir,
            input_encoder=self.input_encoder,
            num_processes=num_processes,
        )

    def collate(self, samples: list[ProcessedSample]) -> Batch:
        assert self._all_rewrite_graphs is not None

        targets = torch.as_tensor([sample.target for sample in samples], dtype=torch.long)
        targets_unique, targets_pos = torch.unique(targets, return_inverse=True)

        num_negative_targets = min(
            int(self.negative_to_positive_targets_ratio * len(targets_unique)),
            self.n_classes - len(targets_unique),
        )
        if num_negative_targets > 0:
            negative_target_weight = torch.ones(self.n_classes)
            negative_target_weight[targets_unique] = 0.0

            targets_unique = torch.cat(
                [
                    targets_unique,
                    torch.multinomial(negative_target_weight, num_samples=num_negative_targets),
                ]
            )

        return Batch(
            self.input_encoder.collate([sample.input for sample in samples]),
            self.rewrite_encoder.collate(
                [self._all_rewrite_graphs[target] for target in targets_unique]
            ),
            targets_unique,
            targets_pos,
            [torch.as_tensor(sample.loc_target, dtype=torch.float32) for sample in samples],
        )

    def _update_rewrite_encodings(self) -> None:
        assert self._all_rewrite_graphs is not None
        assert not self._rewrite_encodings_up_to_date
        self._rewrite_encodings_up_to_date = True

        # To get the best possible rewrite embeddings, we need to make sure we compute them in eval
        # mode, so that e.g. dropout is turned off. We restore the mode afterwards for transparency.
        orig_mode = self.training

        if orig_mode:
            self.eval()

        batch_idx_list: list[torch.Tensor] = []
        mol_outputs_list: list[torch.Tensor] = []
        atom_outputs_list: list[torch.Tensor] = []

        batch_idx_offset = 0
        device = next(self.parameters()).device

        REWRITES_ENCODING_BATCH_SIZE = 4096
        for graphs in batched(self._all_rewrite_graphs, REWRITES_ENCODING_BATCH_SIZE):
            batch = self.rewrite_encoder.collate(graphs)
            batch = self.rewrite_encoder.move_batch_to_device(batch, device=device)

            with torch.no_grad():
                batch_encoding = self.rewrite_encoder(batch)

            batch_idx_list.append(batch.batch.detach() + batch_idx_offset)
            mol_outputs_list.append(batch_encoding.mol_outputs.detach())
            atom_outputs_list.append(batch_encoding.atom_outputs.detach())

            batch_idx_offset += REWRITES_ENCODING_BATCH_SIZE

            del graphs, batch, batch_encoding

        if orig_mode:
            self.train()

        all_batch_idx = torch.cat(batch_idx_list)
        all_atom_outputs = torch.cat(atom_outputs_list)

        all_node_in_lhs = torch.cat(
            [torch.as_tensor(graph.node_in_lhs) for graph in self._all_rewrite_graphs]
        )
        all_node_in_lhs = all_node_in_lhs.bool().to(device)

        self.all_rewrites_batch_idx.data = torch.masked_select(all_batch_idx, all_node_in_lhs)
        self.all_rewrites_mol_outputs.data = torch.cat(mol_outputs_list)
        self.all_rewrites_atom_outputs.data = torch.masked_select(
            all_atom_outputs, all_node_in_lhs.unsqueeze(-1)
        ).view(-1, all_atom_outputs.shape[-1])

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if not self._rewrite_encodings_up_to_date:
            self._update_rewrite_encodings()

            # If we updated encodings *after* `checkpoint` was assembled then refresh them here.
            for key in [
                "all_rewrites_batch_idx",
                "all_rewrites_mol_outputs",
                "all_rewrites_atom_outputs",
            ]:
                checkpoint[key] = getattr(self, key).detach().cpu()
