"""Script for training a new model on a given dataset.

Currently supported model classes:
- TemplateClassificationChem: template classification on top of the `ChemEncoder`
- TemplateClassificationGNN: template classification on top of the `GNNEncoder`
- TemplateLocalization: template classification and localization using two `GNNEncoder`s
- SmilesTransformer: smiles based transformer models with better model flexibility and faster beam search
"""

import datetime
import math
import shutil
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Union, cast

import pytorch_lightning as pl
import torch
from omegaconf import MISSING, OmegaConf
from syntheseus.reaction_prediction.utils.misc import set_random_seed

import retrochimera
from retrochimera.chem.rules import RuleBase
from retrochimera.data.processed_dataset import ProcessedDataModule
from retrochimera.encoders.configs import ChemEncoderConfig, GNNEncoderConfig
from retrochimera.models.lightning import AbstractModel
from retrochimera.models.smiles_transformer import SmilesTransformerModel
from retrochimera.models.template_classification import MCCModel
from retrochimera.models.template_localization import TemplateLocalizationModel
from retrochimera.utils.logging import get_logger
from retrochimera.utils.misc import convert_camel_to_snake, lookup_by_name
from retrochimera.utils.pytorch_lightning import ModelCheckpoint, OptLRMonitor
from retrochimera.utils.training import average_checkpoints

logger = get_logger(__name__)


HELP = """\
Train the model with a chosen preset and optional config overrides.

Required arguments:
  model_class         Specifies the model class to use. Must be one of 'TemplateClassificationChem',
                      'TemplateClassificationGNN', 'TemplateLocalization' or 'SmilesTransformer'.
  preset              Selects a configuration from 'pistachio' (default), 'uspto_full', 'uspto_50k'.
  processed_data_dir  Directory containing the processed data.
  checkpoint_dir      Directory where model checkpoints will be saved.

Optionally you can provide one or several paths to YAML config files to override the preset via the
'config' argument. Any config key can also be overridden directly from the CLI with key=value pairs.
"""


class ModelClass(Enum):
    TemplateClassificationChem = auto()
    TemplateClassificationGNN = auto()
    TemplateLocalization = auto()
    SmilesTransformer = auto()


@dataclass
class MLPConfig:
    """Stores hyperparameters configuring the MLP head."""

    n_layers: int = 0  # Number of extra hidden layers (not counting input and output layers)
    dropout: float = 0.2
    hidden_dim: int = 256


@dataclass
class TrainingConfig:
    """Stores hyperparameters controlling the training process."""

    batch_size: int = 128  # Batch size to use
    learning_rate: float = 1e-3  # Starting learning rate
    learning_rate_decay_step_size: int = 100  # Frequency (in epochs) for decaying the learning rate
    learning_rate_decay_rate: float = 0.1  # Coefficient for a single learning rate decay step
    optimizer_betas: tuple[float, float] = (0.9, 0.999)  # Momentum coefficients for Adam optimizer
    n_epochs: int = 100  # Maximum number of training epochs
    check_val_every_n_epoch: int = 1  # Frequency of running validation (in training epochs)
    num_checkpoints_for_averaging: int = 1  # How many best checkpoints to take an average of
    accelerator: str = "gpu"  # Whether to use a CPU, GPU or TPU
    n_devices: int = 1  # Number of devices
    gradient_clip_val: float = 50.0  # Maximum gradient norm to clip to.
    accumulate_grad_batches: int = 1  # Number of batches between gradient steps.


@dataclass
class TemplateClassificationChemConfig:
    """Configures the template classification model backed by the chem encoder."""

    encoder: ChemEncoderConfig = ChemEncoderConfig()
    mlp: MLPConfig = MLPConfig()
    label_smoothing: float = 0.0
    training: TrainingConfig = TrainingConfig()


@dataclass
class TemplateClassificationGNNConfig:
    """Configures the template classification model backed by the GNN encoder."""

    encoder: GNNEncoderConfig = GNNEncoderConfig()
    mlp: MLPConfig = MLPConfig(dropout=0.4, hidden_dim=128)
    label_smoothing: float = 0.0
    training: TrainingConfig = TrainingConfig(n_epochs=150)


@dataclass
class TemplateLocalizationConfig:
    """Configures the template localization model."""

    # Note: the encoders here do not have to be GNN-based, but they need to be able to provide
    # atom-level outputs, thus `ChemEncoder` would not suffice. The configuration below can be
    # generalized once we introduce other suitable encoders, e.g. based on graph transformers.

    input_encoder: GNNEncoderConfig = GNNEncoderConfig(
        aggregation_dropout=0.4
    )  # Encoder to use for input graphs
    rewrite_encoder: GNNEncoderConfig = GNNEncoderConfig(
        aggregation_dropout=0.4, featurizer_class="DGLLifeRewriteFeaturizer"
    )  # Encoder to use for rewrites
    classification_label_smoothing: float = 0.1
    localization_label_smoothing: float = 0.1
    classification_space_dim: Optional[int] = 256
    free_rewrite_embedding_dim: int = 0
    classification_loss_type: str = "softmax"  # can be either "softmax" or "sigmoid"
    classification_min_temperature: Optional[float] = None
    classification_max_temperature: Optional[float] = None
    negative_to_positive_targets_ratio: float = 0.0
    num_negative_rewrites_in_localization: int = 0
    rewrite_encoder_num_epochs: Optional[int] = None
    training: TrainingConfig = TrainingConfig(n_epochs=600, learning_rate_decay_step_size=550)


@dataclass
class SmilesTransformerConfig:
    """Configures the transformer model."""

    vocab_path: str = MISSING
    hidden_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    feedforward_dim: int = 2048
    activation: str = "gelu"
    max_seq_len: int = 512
    dropout: float = 0.1
    label_smoothing: float = 0.0
    schedule: str = "cycle"
    warm_up_steps: int = 8000
    share_encoder_decoder_input_embedding: bool = True
    initialization: str = "xavier"
    add_qkvbias: bool = False
    layer_norm: str = "standard"
    num_kv: int = 0
    parallel_residual: bool = False
    training: TrainingConfig = TrainingConfig(n_epochs=200, batch_size=128)


@dataclass
class ModelTrainingConfig:
    """Base config for configuring a model to be trained."""

    model_class: ModelClass  # Which model type to train

    # Fields relevant to specific model types
    template_classification_chem_config: TemplateClassificationChemConfig = (
        TemplateClassificationChemConfig()
    )
    template_classification_gnn_config: TemplateClassificationGNNConfig = (
        TemplateClassificationGNNConfig()
    )
    template_localization_config: TemplateLocalizationConfig = TemplateLocalizationConfig()
    smiles_transformer_config: SmilesTransformerConfig = SmilesTransformerConfig()


@dataclass
class TrainConfig(ModelTrainingConfig):
    """Config for running training on a given dataset."""

    processed_data_dir: str = MISSING  # Directory for saving preprocessed data
    checkpoint_dir: str = MISSING  # Directory to store pytorch lightning model checkpoints
    log_dir: str = "."  # Directory to store TensorBoard logs
    seed: int = 0  # Seed to use for all sources of randomness (Python, numpy, torch)

    num_processes_training: int = 1  # Number of processes to use for training


@dataclass
class TrainingResult:
    model: AbstractModel
    test_results: list[dict[str, Any]]


def train(
    model: AbstractModel, data_path: Union[str, Path], config: TrainConfig, model_config: Any
) -> TrainingResult:
    data_loader_kwargs = {
        "batch_size": model_config.training.batch_size,
        "num_workers": config.num_processes_training,
        "collate_fn": model.collate,
        "persistent_workers": config.num_processes_training > 0,
        "pin_memory": True,
    }

    # Add a timeout if we are using data loading worker processes
    if data_loader_kwargs["num_workers"]:
        data_loader_kwargs["timeout"] = 600

    datamodule = ProcessedDataModule(h5_path=data_path, data_loader_kwargs=data_loader_kwargs)

    # Prepare a callback to save the checkpoint at the end of each epoch to recover from
    # preemptions; this is different from the callback that keeps track of the best checkpoint(s).
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="{epoch}-last",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    if isinstance(model, SmilesTransformerModel):
        batches_per_gpu = math.ceil(
            len(datamodule.train_dataloader()) / float(model_config.training.n_devices)
        )
        total_steps = (
            math.ceil(batches_per_gpu / model_config.training.accumulate_grad_batches)
            * model_config.training.n_epochs
        )
        model.total_steps = total_steps
        logger.info(f"Total training steps: {total_steps}")

        best_checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="{epoch}-best",
            monitor="epoch",
            mode="max",
            save_top_k=model_config.training.num_checkpoints_for_averaging,
            every_n_epochs=1,
        )
        lr_monitor_callback = OptLRMonitor()

        callbacks = [best_checkpoint_callback, last_checkpoint_callback, lr_monitor_callback]
    else:
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="{epoch}-best",
            monitor="val_mrr",
            save_top_k=model_config.training.num_checkpoints_for_averaging,
            mode="max",
        )

        callbacks = [best_checkpoint_callback, last_checkpoint_callback]

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir_contents = list(checkpoint_dir.iterdir())
    logger.info(f"Contents of the checkpoint directory: {checkpoint_dir_contents}")

    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    wandb_logger = pl.loggers.wandb.WandbLogger(save_dir=config.log_dir)

    if (
        isinstance(model, TemplateLocalizationModel)
        and config.template_localization_config.rewrite_encoder_num_epochs is not None
    ):
        # TODO(krmaziar): Handle this more directly.
        strategy_kwargs = {"find_unused_parameters": True}
    else:
        strategy_kwargs = {}

    trainer = pl.Trainer(
        strategy=pl.strategies.DDPStrategy(
            timeout=datetime.timedelta(seconds=36000), **strategy_kwargs
        ),  # Increase timeout to give nodes enough time to load the rulebase
        enable_checkpointing=True,
        max_epochs=model_config.training.n_epochs,
        check_val_every_n_epoch=model_config.training.check_val_every_n_epoch,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator=model_config.training.accelerator,
        devices=model_config.training.n_devices,
        gradient_clip_val=model_config.training.gradient_clip_val,
        accumulate_grad_batches=model_config.training.accumulate_grad_batches,
        num_sanity_val_steps=0,
    )

    best_checkpoint_path = checkpoint_dir / "combined.ckpt"

    if best_checkpoint_path.exists():
        logger.info("Found a combined checkpoint which means training has already completed")
    else:
        last_checkpoint_paths = [
            path for path in checkpoint_dir_contents if str(path).endswith("-last.ckpt")
        ]
        if len(last_checkpoint_paths) > 1:
            logger.warning("Ambiguous state of the checkpoint directory")
            last_checkpoint_path = max(
                last_checkpoint_paths,
                key=lambda path: int(str(path).removesuffix("-last.ckpt").split("=")[-1]),
            )
        elif len(last_checkpoint_paths) == 1:
            [last_checkpoint_path] = last_checkpoint_paths
        else:
            last_checkpoint_path = None

        if last_checkpoint_path is not None:
            logger.info(f"Resuming from {last_checkpoint_path}")
            fit_kwargs = {"ckpt_path": last_checkpoint_path}
        else:
            logger.info("No checkpoint found, starting training from scratch")
            fit_kwargs = {}

        trainer.fit(model, datamodule=datamodule, **fit_kwargs)

        if trainer.global_rank == 0:
            checkpoint_paths_for_averaging = list(best_checkpoint_callback.best_k_models.keys())
            logger.info(
                f"Averaging out the following checkpoints: {checkpoint_paths_for_averaging}"
            )

            average_checkpoints(
                input_paths=checkpoint_paths_for_averaging, output_path=best_checkpoint_path
            )

    logger.info(f"Best checkpoint: {best_checkpoint_path}")

    last_checkpoints_dir = checkpoint_dir / "last"
    best_checkpoints_dir = checkpoint_dir / "best"

    model.eval()

    if last_checkpoints_dir.exists() and best_checkpoints_dir.exists():
        logger.info("Judging from checkpoint directory organization testing was already done")
        test_results = []
    else:
        test_results = trainer.test(ckpt_path=best_checkpoint_path, datamodule=datamodule)

    if trainer.global_rank == 0:
        last_checkpoints_dir.mkdir(exist_ok=True)
        best_checkpoints_dir.mkdir(exist_ok=True)

        for path in checkpoint_dir.iterdir():
            if str(path).endswith(".ckpt"):
                if str(path).endswith("-last.ckpt"):
                    logger.info(f"Moving {path} under {last_checkpoints_dir}")
                    path.rename(last_checkpoints_dir / path.name)
                elif path.name != best_checkpoint_path.name:
                    logger.info(f"Moving {path} under {best_checkpoints_dir}")
                    path.rename(best_checkpoints_dir / path.name)
        logger.info(
            f"Contents of the checkpoint directory after training: {list(checkpoint_dir.iterdir())}"
        )

    return TrainingResult(model=model, test_results=test_results)


def build_model_from_config(
    config: ModelTrainingConfig, rulebase: RuleBase, rulebase_dir: Union[str, Path]
) -> tuple[AbstractModel, Any]:
    model_class: Any = None
    model_config: Any = None
    model_kwargs: dict[str, Any] = {}

    def add_encoder_kwargs(
        prefix: str,
        encoder_class: str,
        encoder_config: Union[ChemEncoderConfig, GNNEncoderConfig],
    ) -> None:
        model_kwargs[f"{prefix}encoder_class"] = encoder_class
        model_kwargs[f"{prefix}encoder_kwargs"] = OmegaConf.to_container(encoder_config)

    if config.model_class == ModelClass.TemplateClassificationChem:
        model_class = MCCModel
        model_config = config.template_classification_chem_config

        add_encoder_kwargs(
            prefix="", encoder_class="ChemEncoder", encoder_config=model_config.encoder
        )
    elif config.model_class == ModelClass.TemplateClassificationGNN:
        model_class = MCCModel
        model_config = config.template_classification_gnn_config

        add_encoder_kwargs(
            prefix="", encoder_class="GNNEncoder", encoder_config=model_config.encoder
        )
    elif config.model_class == ModelClass.TemplateLocalization:
        model_class = TemplateLocalizationModel
        model_config = config.template_localization_config

        add_encoder_kwargs(
            prefix="input_", encoder_class="GNNEncoder", encoder_config=model_config.input_encoder
        )
        add_encoder_kwargs(
            prefix="rewrite_",
            encoder_class="GNNEncoder",
            encoder_config=model_config.rewrite_encoder,
        )

        rewrite_featurizer_class = lookup_by_name(
            retrochimera.encoders.featurizers,
            model_kwargs["rewrite_encoder_kwargs"]["featurizer_class"],
        )

        # Look through the rulebase to build the featurizer kwargs (e.g. atom SMARTS vocabulary).
        rewrite_featurizer_kwargs = rewrite_featurizer_class.prepare_kwargs(
            rewrites=(rule.rxn for rule in rulebase.rules.values())
        )
        model_kwargs["rewrite_encoder_kwargs"]["featurizer_kwargs"] = rewrite_featurizer_kwargs
    elif config.model_class == ModelClass.SmilesTransformer:
        model_class = SmilesTransformerModel
        model_config = config.smiles_transformer_config
    else:
        raise ValueError(f"Model class {config.model_class} not recognized")

    # Add model kwargs that are shared across all models classes.
    model_kwargs.update(
        {
            "n_classes": len(rulebase),
            "learning_rate": model_config.training.learning_rate,
            "learning_rate_decay_step_size": model_config.training.learning_rate_decay_step_size,
            "learning_rate_decay_rate": model_config.training.learning_rate_decay_rate,
            "optimizer_betas": list(model_config.training.optimizer_betas),
        }
    )

    if config.model_class == ModelClass.TemplateLocalization:
        keys = [
            "classification_label_smoothing",
            "localization_label_smoothing",
            "classification_space_dim",
            "free_rewrite_embedding_dim",
            "classification_loss_type",
            "classification_min_temperature",
            "classification_max_temperature",
            "negative_to_positive_targets_ratio",
            "num_negative_rewrites_in_localization",
            "rewrite_encoder_num_epochs",
        ]
        model_kwargs.update({key: getattr(model_config, key) for key in keys})
        model_kwargs["num_total_rewrite_lhs_atoms"] = sum(
            rule.rxn.rdkit_lhs_mol.GetNumAtoms() for rule in rulebase.rules.values()
        )
    elif config.model_class == ModelClass.SmilesTransformer:
        keys = [
            "vocab_path",
            "hidden_dim",
            "n_layers",
            "n_heads",
            "feedforward_dim",
            "activation",
            "max_seq_len",
            "dropout",
            "label_smoothing",
            "schedule",
            "warm_up_steps",
            "share_encoder_decoder_input_embedding",
            "initialization",
            "add_qkvbias",
            "layer_norm",
            "num_kv",
            "parallel_residual",
        ]
        model_kwargs.update({key: getattr(model_config, key) for key in keys})
    else:
        model_kwargs.update(
            {
                "hidden_dim": model_config.mlp.hidden_dim,
                "n_hidden_layers": model_config.mlp.n_layers,
                "dropout": model_config.mlp.dropout,
            }
        )

    if config.model_class in [
        ModelClass.TemplateClassificationChem,
        ModelClass.TemplateClassificationGNN,
    ]:
        model_kwargs["label_smoothing"] = model_config.label_smoothing

    logger.info(f"Building the model {config.model_class} with kwargs {model_kwargs}")

    model = model_class(**model_kwargs)
    model.set_rulebase(rulebase=rulebase, rulebase_dir=rulebase_dir)

    return model, model_config


def parse_training_config(argv: list[str]) -> TrainConfig:
    config = OmegaConf.from_dotlist(argv)
    preset = config.pop("preset", "pistachio")

    # Use preset and model class to determine the config file to load.
    config_paths = [
        Path(__file__).parent
        / "config"
        / preset
        / (convert_camel_to_snake(config.model_class) + ".yaml")
    ]

    if "config" in config:
        config_paths.append(config.pop("config"))

    combined_config = OmegaConf.merge(
        OmegaConf.structured(TrainConfig), *[OmegaConf.load(c) for c in config_paths], config
    )
    return cast(TrainConfig, combined_config)


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")

    argv = sys.argv[1:]

    # If no arguments were provided or `--help`/`-h` is present, print the help message and exit.
    if not argv or "--help" in argv or "-h" in argv:
        print(HELP)
        sys.exit(0)

    config = parse_training_config(argv)
    logger.info(f"Running training with the following config: {config}")

    set_random_seed(config.seed)

    processed_data_dir = Path(config.processed_data_dir)
    processed_rulebase_path = processed_data_dir / RuleBase.DEFAULT_FILE_NAME

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rulebase_path = checkpoint_dir / RuleBase.DEFAULT_FILE_NAME

    # Copy the rulebase into the checkpoint directory, which is only done by the first process
    # (other training processes will be spawned in `train`).
    if not checkpoint_rulebase_path.exists():
        logger.info(f"Copying the rulebase into {config.checkpoint_dir}")
        shutil.copy(processed_rulebase_path, checkpoint_rulebase_path)

    if config.model_class is ModelClass.SmilesTransformer:
        checkpoint_vocab_path = checkpoint_dir / SmilesTransformerModel.DEFAULT_VOCAB_FILE_NAME

        if not checkpoint_vocab_path.exists():
            # For the SMILES Transformer we also copy the vocab file.
            logger.info(f"Copying the vocab file into {config.checkpoint_dir}")
            shutil.copy(config.smiles_transformer_config.vocab_path, checkpoint_vocab_path)

    model, model_config = build_model_from_config(
        config=config,
        rulebase=RuleBase.load_from_file(dir=checkpoint_dir),
        rulebase_dir=checkpoint_dir,
    )

    logger.info("Starting training")
    train(
        model=model,
        data_path=str(processed_data_dir / "data.h5"),
        config=config,
        model_config=model_config,
    )


if __name__ == "__main__":
    main()
