from __future__ import annotations

import copy
import datetime
import json
import math
import pickle
import queue
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Optional, Sequence, cast

from omegaconf import DictConfig, OmegaConf
from syntheseus import BackwardReactionModel, Molecule, SingleProductReaction
from syntheseus.cli import search
from syntheseus.reaction_prediction.utils.config import get_config as cli_get_config
from syntheseus.reaction_prediction.utils.misc import cpu_count, set_random_seed
from syntheseus.reaction_prediction.utils.model_loading import get_model
from syntheseus.search.mol_inventory import SmilesListInventory

from retrochimera import inference
from retrochimera.chem.rules import RuleBasedRetrosynthesizer
from retrochimera.cli.eval import BackwardModelConfig
from retrochimera.utils.misc import lookup_by_name
from retrochimera.utils.root_aligned import AUGMENTATION_SEED_METADATA_KEY


@dataclass
class SearchConfig(BackwardModelConfig, search.BaseSearchConfig):
    """Config for running search for given search targets."""

    max_active_searches: int = 32
    inference_batch_size: int = 16
    inference_batch_wait_s: float = 0.5
    inference_replicas: int = 1
    num_routes_to_plot: int = 0
    seed: int = 0


@dataclass
class _InferenceTicket:
    input: Molecule
    num_results: int
    queued_at: float
    future: Future[Sequence[SingleProductReaction]]


class _InferenceBroker:
    """Single-owner model executor which batches molecule tickets."""

    def __init__(
        self,
        model: BackwardReactionModel,
        batch_size: int,
        batch_wait_s: float,
        max_queue_size: int,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._batch_wait_s = batch_wait_s
        self._queue: queue.Queue[Optional[_InferenceTicket]] = queue.Queue(max_queue_size)
        self.batch_sizes: list[int] = []
        self._thread = threading.Thread(target=self._run, name="retrochimera-inference")
        self._thread.start()

    def __enter__(self) -> _InferenceBroker:
        return self

    def __exit__(self, *args: object) -> None:
        self._queue.put(None)
        self._thread.join()

    def submit(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Future[Sequence[SingleProductReaction]]]:
        futures = [Future[Sequence[SingleProductReaction]]() for _ in inputs]
        for input, future in zip(inputs, futures):
            self._queue.put(_InferenceTicket(input, num_results, time.monotonic(), future))
        return futures

    def _run(self) -> None:
        pending: Optional[_InferenceTicket] = None
        failure: Optional[BaseException] = None
        stop = False

        while not stop:
            ticket = pending if pending is not None else self._queue.get()
            pending = None
            if ticket is None:
                break

            batch = [ticket]
            try:
                if failure is None:
                    deadline = ticket.queued_at + self._batch_wait_s
                    while len(batch) < self._batch_size:
                        remaining = deadline - time.monotonic()
                        try:
                            next_ticket = (
                                self._queue.get(timeout=remaining)
                                if remaining > 0
                                else self._queue.get_nowait()
                            )
                        except queue.Empty:
                            break

                        if next_ticket is None:
                            stop = True
                            break
                        if next_ticket.num_results != ticket.num_results:
                            pending = next_ticket
                            break
                        batch.append(next_ticket)

                    outputs = self._model(
                        [item.input for item in batch], num_results=ticket.num_results
                    )
                    if len(outputs) != len(batch):
                        raise RuntimeError(
                            f"Model returned {len(outputs)} outputs for {len(batch)} inputs"
                        )
                    outputs = [copy.deepcopy(output) for output in outputs]
                    self.batch_sizes.append(len(batch))
                    for item, output in zip(batch, outputs):
                        item.future.set_result(output)
                else:
                    for item in batch:
                        item.future.set_exception(failure)
            except BaseException as error:
                failure = failure or error
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(failure)


class _BrokeredBackwardReactionModel(BackwardReactionModel):
    def __init__(
        self,
        broker: _InferenceBroker,
        cancel_event: Optional[threading.Event] = None,
        augmentation_rng: Optional[random.Random] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("remove_duplicates", False)
        super().__init__(**kwargs)
        self._broker = broker
        self._cancel_event = cancel_event or threading.Event()
        self._augmentation_rng = augmentation_rng

    def num_calls(self, count_cache: Optional[bool] = None) -> int:
        return (
            sys.maxsize
            if self._cancel_event.is_set()
            else super().num_calls(count_cache=count_cache)
        )

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        if self._cancel_event.is_set():
            raise CancelledError()
        broker_inputs = inputs
        if self._augmentation_rng is not None:
            broker_inputs = []
            for input in inputs:
                metadata = input.metadata.copy()
                metadata[AUGMENTATION_SEED_METADATA_KEY] = self._augmentation_rng.getrandbits(64)
                broker_inputs.append(
                    Molecule(
                        input.smiles,
                        identifier=input.identifier,
                        canonicalize=False,
                        make_rdkit_mol=False,
                        metadata=metadata,
                    )
                )

        outputs = [future.result() for future in self._broker.submit(broker_inputs, num_results)]
        if self._cancel_event.is_set():
            raise CancelledError()
        return outputs


def _validate_config(config: SearchConfig) -> None:
    if config.max_active_searches <= 0:
        raise ValueError("max_active_searches must be positive")
    if config.inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    if config.inference_replicas <= 0:
        raise ValueError("inference_replicas must be positive")
    if not math.isfinite(config.inference_batch_wait_s) or config.inference_batch_wait_s < 0:
        raise ValueError("inference_batch_wait_s must be finite and non-negative")
    if config.num_routes_to_plot != 0:
        raise ValueError("Route plotting is not supported by concurrent search")


def _get_search_targets(config: SearchConfig) -> list[str]:
    config_dict = cast(DictConfig, config)
    target = config_dict.get("search_target")
    targets_file = config_dict.get("search_targets_file")
    if not ((target is None) ^ (targets_file is None)):
        raise ValueError("Exactly one search target source must be provided")
    if target is not None:
        return [target]

    with open(targets_file, "rt") as file:
        targets = [line.strip() for line in file if line.strip()]
    if not targets:
        raise ValueError("Search targets file is empty")
    return targets


def _configure_replica_processes(config: SearchConfig, replica_count: int) -> SearchConfig:
    if replica_count == 1:
        return config

    model_kwargs = OmegaConf.to_container(config.model_kwargs)
    assert isinstance(model_kwargs, dict)

    if config.model_class.name == "RetroChimera":
        with open(Path(config.model_dir) / "models.json") as file:
            model_data = json.load(file)
        rule_model_keys = [
            key
            for key, (class_name, _) in model_data.items()
            if issubclass(
                lookup_by_name(inference, class_name),
                RuleBasedRetrosynthesizer,
            )
        ]
        default_processes = max(
            1, cpu_count() // (2 * replica_count * max(1, len(rule_model_keys)))
        )
        for key in rule_model_keys:
            submodel_kwargs = model_kwargs.setdefault(key, {})
            if not isinstance(submodel_kwargs, dict):
                raise ValueError(f"Model kwargs for {key} must be a dictionary")
            submodel_kwargs.setdefault("num_processes", default_processes)
    elif issubclass(config.model_class.value, RuleBasedRetrosynthesizer):
        model_kwargs.setdefault("num_processes", max(1, cpu_count() // (2 * replica_count)))

    return OmegaConf.merge(config, {"model_kwargs": model_kwargs})


def _run_target(
    index: int,
    smiles: str,
    config: SearchConfig,
    broker: _InferenceBroker,
    cancel_event: threading.Event,
    inventory: SmilesListInventory,
    results_dir: Path,
) -> tuple[int, dict[str, Any]]:
    model = _BrokeredBackwardReactionModel(
        broker,
        cancel_event,
        augmentation_rng=random.Random(f"{config.seed}:{index}"),
        use_cache=config.reaction_model_use_cache,
        default_num_results=config.num_top_results,
    )
    algorithm = config.search_algorithm.value(
        reaction_model=model,
        mol_inventory=inventory,
        **search.search_algorithm_config_to_kwargs(config),
    )

    start = time.monotonic()
    graph, _ = algorithm.run_from_mol(Molecule(smiles))
    if cancel_event.is_set():
        raise CancelledError()
    stats = {
        "index": index,
        "smiles": smiles,
        "solved": bool(graph.root_node.has_solution),
        "elapsed_s": time.monotonic() - start,
        "rxn_model_calls_used": model.num_calls(),
        "num_nodes_in_final_tree": len(graph),
    }

    target_dir = results_dir / str(index)
    target_dir.mkdir()
    if config.save_graph:
        with open(target_dir / "graph.pkl", "wb") as file:
            pickle.dump(graph, file)
    with open(target_dir / "stats.json", "wt") as file:
        json.dump(stats, file, indent=2)
    return index, stats


def run_from_config(config: SearchConfig) -> Path:
    _validate_config(config)
    targets = _get_search_targets(config)
    set_random_seed(config.seed)

    dirname = config.model_class.name
    if config.append_timestamp_to_dir:
        dirname += f"_{datetime.datetime.now().isoformat(timespec='seconds')}"
    results_dir = Path(config.results_dir) / dirname
    if results_dir.exists():
        raise FileExistsError(f"Results directory already exists: {results_dir}")

    active_searches = min(len(targets), config.max_active_searches)
    replica_count = min(
        config.inference_replicas,
        max(1, active_searches // config.inference_batch_size),
    )
    model_config = _configure_replica_processes(config, replica_count)

    backends = [
        cast(
            BackwardReactionModel,
            get_model(
                model_config,
                batch_size=config.inference_batch_size,
                num_gpus=int(config.use_gpu),
                use_cache=False,
                default_num_results=config.num_top_results,
            ),
        )
        for _ in range(replica_count)
    ]
    inventory = SmilesListInventory.load_from_file(
        config.inventory_smiles_file, canonicalize=config.canonicalize_inventory
    )

    results_dir.mkdir(parents=True)

    start = time.monotonic()
    stats_by_index: dict[int, dict[str, Any]] = {}
    cancel_event = threading.Event()
    wait_s = (
        0.0 if min(len(targets), config.max_active_searches) == 1 else config.inference_batch_wait_s
    )
    brokers = [
        _InferenceBroker(
            backend,
            config.inference_batch_size,
            wait_s,
            config.max_active_searches * config.inference_batch_size,
        )
        for backend in backends
    ]

    with ExitStack() as stack:
        for broker in brokers:
            stack.enter_context(broker)
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=config.max_active_searches))
        target_iter = iter(enumerate(targets))

        def submit(target):
            index = target[0]
            return executor.submit(
                _run_target,
                *target,
                config,
                brokers[index % replica_count],
                cancel_event,
                inventory,
                results_dir,
            )

        futures = {submit(target) for target in islice(target_iter, config.max_active_searches)}
        try:
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    index, stats = future.result()
                    stats_by_index[index] = stats
                    try:
                        target = next(target_iter)
                    except StopIteration:
                        continue
                    futures.add(submit(target))
        except BaseException:
            cancel_event.set()
            for future in futures:
                future.cancel()
            raise

    ordered_stats = [stats_by_index[index] for index in range(len(targets))]
    batch_sizes = [size for broker in brokers for size in broker.batch_sizes]
    summary = {
        "num_targets": len(targets),
        "num_solved_targets": sum(item["solved"] for item in ordered_stats),
        "elapsed_s": time.monotonic() - start,
        "inference_replicas": replica_count,
        "num_inference_batches": len(batch_sizes),
        "inference_batch_sizes": batch_sizes,
        "average_inference_batch_size": (
            sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
        ),
        "targets": ordered_stats,
    }
    with open(results_dir / "stats.json", "wt") as file:
        json.dump(summary, file, indent=2)
    return results_dir


def main(argv: Optional[list[str]]) -> Path:
    return run_from_config(cli_get_config(argv=argv, config_cls=SearchConfig))


if __name__ == "__main__":
    main(argv=None)
