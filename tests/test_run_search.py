import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
from omegaconf import OmegaConf
from syntheseus import BackwardReactionModel, Bag, Molecule, SingleProductReaction

from retrochimera.cli import run_search
from retrochimera.cli.eval import BackwardModelClass
from retrochimera.inference.smiles_transformer import SmilesTransformerModel
from retrochimera.utils.root_aligned import AUGMENTATION_SEED_METADATA_KEY


class RecordingModel(BackwardReactionModel):
    def __init__(self, fail: bool = False) -> None:
        super().__init__(use_cache=False)
        self.fail = fail
        self.calls: list[tuple[list[str], int]] = []
        self.active_calls = 0
        self.max_active_calls = 0
        self.lock = threading.Lock()

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        with self.lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self.calls.append(([input.smiles for input in inputs], num_results))

        try:
            time.sleep(0.01)
            if self.fail:
                raise RuntimeError("inference failed")
            return [
                [
                    SingleProductReaction(
                        product=input,
                        reactants=Bag([Molecule("C")]),
                        metadata={"probability": 1.0, "source": input.smiles},
                    )
                ]
                for input in inputs
            ]
        finally:
            with self.lock:
                self.active_calls -= 1


def _call_together(
    models: list[run_search._BrokeredBackwardReactionModel],
    molecules: list[Molecule],
    num_results: list[int],
):
    barrier = threading.Barrier(len(models))

    def call(index: int):
        barrier.wait()
        return models[index]([molecules[index]], num_results=num_results[index])

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        return list(executor.map(call, range(len(models))))


def test_broker_batches_requests_and_preserves_order() -> None:
    backend = RecordingModel()
    molecules = [Molecule("C" * length) for length in range(2, 10)]

    with run_search._InferenceBroker(backend, 8, 0.2, 8) as broker:
        models = [run_search._BrokeredBackwardReactionModel(broker) for _ in molecules]
        outputs = _call_together(models, molecules, [3] * len(molecules))

    assert len(backend.calls) == 1
    assert set(backend.calls[0][0]) == {molecule.smiles for molecule in molecules}
    assert backend.max_active_calls == 1
    assert broker.batch_sizes == [8]
    assert [output[0][0].product for output in outputs] == molecules


def test_independent_brokers_use_multiple_model_replicas() -> None:
    backends = [RecordingModel(), RecordingModel()]
    molecules = [Molecule("C" * length) for length in range(2, 6)]

    with (
        run_search._InferenceBroker(backends[0], 2, 0.2, 2) as first,
        run_search._InferenceBroker(backends[1], 2, 0.2, 2) as second,
    ):
        models = [
            run_search._BrokeredBackwardReactionModel([first, second][index % 2])
            for index in range(len(molecules))
        ]
        outputs = _call_together(models, molecules, [3] * len(molecules))

    assert sorted(len(inputs) for backend in backends for inputs, _ in backend.calls) == [2, 2]
    assert [output[0][0].product for output in outputs] == molecules


def test_broker_flushes_partial_batch() -> None:
    backend = RecordingModel()

    with run_search._InferenceBroker(backend, 8, 0.01, 1) as broker:
        model = run_search._BrokeredBackwardReactionModel(broker)
        output = model([Molecule("CC")])

    assert output[0][0].product == Molecule("CC")
    assert broker.batch_sizes == [1]


def test_broker_splits_multi_input_calls_at_batch_boundary() -> None:
    backend = RecordingModel()
    molecules = [Molecule("C" * length) for length in range(2, 9)]

    with run_search._InferenceBroker(backend, 3, 0.01, 7) as broker:
        model = run_search._BrokeredBackwardReactionModel(broker)
        outputs = model(molecules)

    assert max(len(inputs) for inputs, _ in backend.calls) == 3
    assert [output[0].product for output in outputs] == molecules


def test_broker_keeps_incompatible_requests_separate() -> None:
    backend = RecordingModel()

    with run_search._InferenceBroker(backend, 8, 0.05, 2) as broker:
        models = [run_search._BrokeredBackwardReactionModel(broker) for _ in range(2)]
        _call_together(models, [Molecule("CC"), Molecule("CCC")], [1, 2])

    assert sorted(num_results for _, num_results in backend.calls) == [1, 2]
    assert broker.batch_sizes == [1, 1]


def test_broker_propagates_failure_to_all_callers() -> None:
    backend = RecordingModel(fail=True)

    with run_search._InferenceBroker(backend, 2, 0.2, 5) as broker:
        models = [run_search._BrokeredBackwardReactionModel(broker) for _ in range(5)]
        barrier = threading.Barrier(5)

        def call(index: int) -> None:
            barrier.wait()
            with pytest.raises(RuntimeError, match="inference failed"):
                models[index]([Molecule("C" * (index + 2))])

        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(call, range(5)))

        with pytest.raises(RuntimeError, match="inference failed"):
            models[0]([Molecule("CCCCCC")])

    assert len(backend.calls) == 1
    assert len(backend.calls[0][0]) == 2


def test_proxies_have_independent_caches_and_results() -> None:
    backend = RecordingModel()
    molecule = Molecule("CC")

    with run_search._InferenceBroker(backend, 2, 0.05, 2) as broker:
        models = [
            run_search._BrokeredBackwardReactionModel(broker, use_cache=True) for _ in range(2)
        ]
        outputs = _call_together(models, [molecule, molecule], [1, 1])

        assert models[0]([molecule], num_results=1) == outputs[0]
        assert models[0].num_calls() == models[1].num_calls() == 1
        models[0].reset()
        assert models[0].num_calls() == 0
        models[0]([molecule], num_results=1)

    outputs[0][0][0].metadata["source"] = "changed"
    next(iter(outputs[0][0][0].reactants)).metadata["changed"] = True
    assert outputs[1][0][0].metadata["source"] == molecule.smiles
    assert "changed" not in next(iter(outputs[1][0][0].reactants)).metadata
    assert len(backend.calls) == 2


def test_target_augmentation_rng_is_independent_of_scheduling() -> None:
    class SeedRecordingModel(RecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.seeds: list[tuple[int, int]] = []

        def _get_reactions(self, inputs, num_results):
            self.seeds.extend(
                (
                    int(input.identifier),
                    input.metadata[AUGMENTATION_SEED_METADATA_KEY],
                )
                for input in inputs
            )
            return super()._get_reactions(inputs, num_results)

    def collect(order: list[int]) -> dict[int, list[int]]:
        backend = SeedRecordingModel()
        with run_search._InferenceBroker(backend, 1, 0.0, 1) as broker:
            models = [
                run_search._BrokeredBackwardReactionModel(
                    broker,
                    augmentation_rng=random.Random(f"17:{target_index}"),
                )
                for target_index in range(2)
            ]
            for target_index in order:
                models[target_index](
                    [Molecule("CC", identifier=target_index)],
                    num_results=1,
                )

        seeds: dict[int, list[int]] = {0: [], 1: []}
        for target_index, seed in backend.seeds:
            seeds[target_index].append(seed)
        return seeds

    assert collect([0, 1, 0, 1]) == collect([1, 1, 0, 0])


def test_seeded_augmentation_does_not_consume_global_random_state() -> None:
    model: Any = object.__new__(SmilesTransformerModel)
    model.augmentation_size = 4
    molecule = Molecule(
        "CCCO",
        metadata={AUGMENTATION_SEED_METADATA_KEY: 1234},
    )

    random.seed(99)
    expected_next_random = random.random()
    random.seed(99)

    first = model._augment_input(molecule)
    second = model._augment_input(molecule)

    assert first == second
    assert random.random() == expected_next_random


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_active_searches", 0, "max_active_searches"),
        ("inference_batch_size", 0, "inference_batch_size"),
        ("inference_replicas", 0, "inference_replicas"),
        ("inference_batch_wait_s", -1.0, "inference_batch_wait_s"),
        ("inference_batch_wait_s", float("inf"), "inference_batch_wait_s"),
        ("num_routes_to_plot", 1, "Route plotting"),
    ],
)
def test_validate_config(field: str, value, message: str) -> None:
    config = SimpleNamespace(
        max_active_searches=8,
        inference_batch_size=8,
        inference_batch_wait_s=0.05,
        inference_replicas=2,
        num_routes_to_plot=0,
    )
    setattr(config, field, value)

    with pytest.raises(ValueError, match=message):
        run_search._validate_config(config)  # type: ignore[arg-type]


def test_replica_process_config_handles_model_shapes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_search, "cpu_count", lambda: 64)

    standalone = OmegaConf.create(
        run_search.SearchConfig(
            model_class=BackwardModelClass.TemplateLocalization,
            model_dir="unused",
        )
    )
    configured = run_search._configure_replica_processes(standalone, 2)
    assert configured.model_kwargs.num_processes == 16

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "models.json").write_text(
        json.dumps({"smiles_transformer": ["SmilesTransformerModel", [1.0]]})
    )
    ensemble = OmegaConf.create(
        run_search.SearchConfig(
            model_class=BackwardModelClass.RetroChimera,
            model_dir=str(model_dir),
        )
    )
    configured = run_search._configure_replica_processes(ensemble, 2)
    assert "template_localization" not in configured.model_kwargs


def test_run_from_config_batches_targets(monkeypatch, tmp_path: Path) -> None:
    target_path = tmp_path / "targets.smi"
    target_path.write_text("CC\nCCC\nCCCC\nCCCCC\n")
    inventory_path = tmp_path / "inventory.smi"
    inventory_path.write_text("C\n")
    backend = RecordingModel()
    load_calls = 0

    def load_model(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return backend

    monkeypatch.setattr(run_search, "get_model", load_model)
    config = OmegaConf.create(
        run_search.SearchConfig(
            model_class=BackwardModelClass.RetroChimera,
            model_dir="unused",
            search_targets_file=str(target_path),
            inventory_smiles_file=str(inventory_path),
            results_dir=str(tmp_path / "results"),
            append_timestamp_to_dir=False,
            use_gpu=False,
            save_graph=False,
            limit_iterations=1,
            max_active_searches=4,
            inference_batch_size=4,
            inference_batch_wait_s=0.2,
        )
    )

    results_dir = run_search.run_from_config(config)  # type: ignore[arg-type]

    assert load_calls == 1
    assert any(len(inputs) > 1 for inputs, _ in backend.calls)
    assert [path.name for path in sorted(results_dir.iterdir())] == [
        "0",
        "1",
        "2",
        "3",
        "stats.json",
    ]
    summary = json.loads((results_dir / "stats.json").read_text())
    assert [target["smiles"] for target in summary["targets"]] == [
        "CC",
        "CCC",
        "CCCC",
        "CCCCC",
    ]
    assert summary["num_targets"] == summary["num_solved_targets"] == 4
    assert summary["inference_replicas"] == 1
    assert summary["average_inference_batch_size"] > 1


def test_run_from_config_bounds_active_searches(monkeypatch, tmp_path: Path) -> None:
    target_path = tmp_path / "targets.smi"
    target_path.write_text("\n".join(["CC"] * 10))
    inventory_path = tmp_path / "inventory.smi"
    inventory_path.write_text("C\n")
    active = 0
    max_active = 0
    lock = threading.Lock()

    def run_target(index, smiles, config, broker, cancel_event, inventory, results_dir):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.01)
            return index, {"index": index, "smiles": smiles, "solved": True}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(run_search, "get_model", lambda *args, **kwargs: RecordingModel())
    monkeypatch.setattr(run_search, "_run_target", run_target)
    config = OmegaConf.create(
        run_search.SearchConfig(
            model_class=BackwardModelClass.RetroChimera,
            model_dir="unused",
            search_targets_file=str(target_path),
            inventory_smiles_file=str(inventory_path),
            results_dir=str(tmp_path / "results"),
            append_timestamp_to_dir=False,
            use_gpu=False,
            save_graph=False,
            max_active_searches=3,
        )
    )

    run_search.run_from_config(config)  # type: ignore[arg-type]

    assert max_active == 3


def test_run_from_config_loads_needed_replicas(monkeypatch, tmp_path: Path) -> None:
    target_path = tmp_path / "targets.smi"
    target_path.write_text("\n".join(["CC"] * 32))
    inventory_path = tmp_path / "inventory.smi"
    inventory_path.write_text("C\n")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "models.json").write_text(
        json.dumps({"template_localization": ["TemplateLocalizationModel", [1.0]]})
    )
    load_calls = 0
    loaded_num_processes = []
    broker_by_index = {}

    def load_model(config, *args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        loaded_num_processes.append(config.model_kwargs.template_localization.num_processes)
        return RecordingModel()

    def run_target(index, smiles, config, broker, cancel_event, inventory, results_dir):
        broker_by_index[index] = id(broker)
        return index, {"index": index, "smiles": smiles, "solved": True}

    monkeypatch.setattr(run_search, "get_model", load_model)
    monkeypatch.setattr(run_search, "_run_target", run_target)
    monkeypatch.setattr(run_search, "cpu_count", lambda: 64)
    config = OmegaConf.create(
        run_search.SearchConfig(
            model_class=BackwardModelClass.RetroChimera,
            model_dir=str(model_dir),
            search_targets_file=str(target_path),
            inventory_smiles_file=str(inventory_path),
            results_dir=str(tmp_path / "results"),
            append_timestamp_to_dir=False,
            use_gpu=False,
            save_graph=False,
        )
    )

    results_dir = run_search.run_from_config(config)  # type: ignore[arg-type]

    assert load_calls == 2
    assert loaded_num_processes == [16, 16]
    assert len(set(broker_by_index.values())) == 2
    assert len({broker_by_index[index] for index in range(0, 32, 2)}) == 1
    assert len({broker_by_index[index] for index in range(1, 32, 2)}) == 1
    assert json.loads((results_dir / "stats.json").read_text())["inference_replicas"] == 2


def test_run_from_config_cancels_active_searches(monkeypatch, tmp_path: Path) -> None:
    target_path = tmp_path / "targets.smi"
    target_path.write_text("CC\nCCC\nCCCC\n")
    inventory_path = tmp_path / "inventory.smi"
    inventory_path.write_text("C\n")
    barrier = threading.Barrier(3)
    cancelled: list[int] = []

    def run_target(index, smiles, config, broker, cancel_event, inventory, results_dir):
        barrier.wait()
        if index == 0:
            raise RuntimeError("search failed")
        if cancel_event.wait(timeout=1):
            cancelled.append(index)
        return index, {"index": index, "smiles": smiles, "solved": False}

    monkeypatch.setattr(run_search, "get_model", lambda *args, **kwargs: RecordingModel())
    monkeypatch.setattr(run_search, "_run_target", run_target)
    config = OmegaConf.create(
        run_search.SearchConfig(
            model_class=BackwardModelClass.RetroChimera,
            model_dir="unused",
            search_targets_file=str(target_path),
            inventory_smiles_file=str(inventory_path),
            results_dir=str(tmp_path / "results"),
            append_timestamp_to_dir=False,
            use_gpu=False,
            save_graph=False,
            max_active_searches=3,
        )
    )

    with pytest.raises(RuntimeError, match="search failed"):
        run_search.run_from_config(config)  # type: ignore[arg-type]

    assert sorted(cancelled) == [1, 2]
