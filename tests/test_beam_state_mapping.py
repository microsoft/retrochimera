from types import SimpleNamespace
from typing import Any

import pytest
import torch

from retrochimera.opennmt.decode.beam_search import BeamSearch
from retrochimera.opennmt.decode.decoder_strategy import tile
from retrochimera.opennmt.decode.translator import Translator
from retrochimera.opennmt.modules.average_attention import AverageAttention
from retrochimera.opennmt.modules.transformer_decoder import TransformerDecoder


def _decoder(self_attn_type: str = "scaled-dot") -> TransformerDecoder:
    decoder = TransformerDecoder(
        num_layers=1,
        d_model=4,
        heads=2,
        d_ff=8,
        self_attn_type=self_attn_type,
        dropout=0.0,
        attention_dropout=0.0,
    )
    decoder.eval()
    decoder.init_state(torch.tensor([[[10], [10], [20], [20]]]), None, None)
    return decoder


@pytest.mark.parametrize("self_attn_type", ["scaled-dot", "average"])
def test_map_state_can_reorder_only_self_attention(self_attn_type: str) -> None:
    decoder = _decoder(self_attn_type)
    layer = decoder.transformer_layers[0]
    context_keys = torch.arange(16).view(4, 2, 2)
    context_values = context_keys + 100
    layer.context_attn.layer_cache = True, {
        "keys": context_keys,
        "values": context_values,
    }

    if isinstance(layer.self_attn, AverageAttention):
        self_state = torch.arange(16).view(4, 1, 4)
        layer.self_attn.layer_cache = True, {"prev_g": self_state}
    else:
        self_keys = torch.arange(16).view(4, 2, 2)
        self_values = self_keys + 100
        key_pad_mask = torch.tensor([[False, True], [True, False], [False, False], [True, True]])
        layer.self_attn.layer_cache = True, {
            "keys": self_keys,
            "values": self_values,
            "key_pad_mask": key_pad_mask,
        }

    src = decoder.state["src"]
    select_indices = torch.tensor([1, 0, 3, 2])
    decoder.map_state(
        lambda state, dim: state.index_select(dim, select_indices),
        map_src=False,
        map_context=False,
        map_self=True,
    )

    assert decoder.state["src"] is src
    assert layer.context_attn.layer_cache[1]["keys"] is context_keys
    assert layer.context_attn.layer_cache[1]["values"] is context_values
    if isinstance(layer.self_attn, AverageAttention):
        assert torch.equal(
            layer.self_attn.layer_cache[1]["prev_g"],
            self_state.index_select(0, select_indices),
        )
    else:
        assert torch.equal(
            layer.self_attn.layer_cache[1]["keys"],
            self_keys.index_select(0, select_indices),
        )
        assert torch.equal(
            layer.self_attn.layer_cache[1]["values"],
            self_values.index_select(0, select_indices),
        )
        assert torch.equal(
            layer.self_attn.layer_cache[1]["key_pad_mask"],
            key_pad_mask.index_select(0, select_indices),
        )


def test_optimized_mapping_matches_full_mapping_outputs() -> None:
    torch.manual_seed(1)
    full_mapping = _decoder()
    optimized_mapping = _decoder()
    optimized_mapping.load_state_dict(full_mapping.state_dict())
    memory = torch.randn(2, 3, 4).repeat_interleave(2, dim=0)
    memory_mask = torch.tensor([[False, False, True], [False, False, False]]).repeat_interleave(
        2, dim=0
    )
    target = torch.randn(4, 1, 4)
    target_mask = torch.zeros(4, 1, dtype=torch.bool)
    full_mapping(
        target,
        target_mask,
        memory,
        memory_mask,
        step=0,
        return_attn=True,
    )
    optimized_mapping(
        target,
        target_mask,
        memory,
        memory_mask,
        step=0,
        return_attn=True,
    )
    within_source = torch.tensor([1, 0, 3, 2])

    def map_within_source(state, dim):
        return state.index_select(dim, within_source)

    full_mapping.map_state(map_within_source)
    optimized_mapping.map_state(
        map_within_source,
        map_src=False,
        map_context=False,
        map_self=True,
    )

    next_target = torch.randn(4, 1, 4)
    full_output, full_attention = full_mapping(
        next_target,
        target_mask,
        memory.index_select(0, within_source),
        memory_mask.index_select(0, within_source),
        step=1,
        return_attn=True,
    )
    optimized_output, optimized_attention = optimized_mapping(
        next_target,
        target_mask,
        memory,
        memory_mask,
        step=1,
        return_attn=True,
    )
    assert torch.equal(optimized_output, full_output)
    assert torch.equal(optimized_attention["std"], full_attention["std"])

    remaining_source = torch.tensor([2, 3])

    def map_remaining_source(state, dim):
        return state.index_select(dim, remaining_source)

    full_mapping.map_state(map_remaining_source)
    optimized_mapping.map_state(
        map_remaining_source,
        map_src=True,
        map_context=True,
        map_self=True,
    )
    compact_memory = memory.index_select(0, remaining_source)
    compact_mask = memory_mask.index_select(0, remaining_source)
    compact_target = torch.randn(2, 1, 4)
    compact_target_mask = torch.zeros(2, 1, dtype=torch.bool)
    full_output, full_attention = full_mapping(
        compact_target,
        compact_target_mask,
        compact_memory,
        compact_mask,
        step=2,
        return_attn=True,
    )
    optimized_output, optimized_attention = optimized_mapping(
        compact_target,
        compact_target_mask,
        compact_memory,
        compact_mask,
        step=2,
        return_attn=True,
    )
    assert torch.equal(optimized_output, full_output)
    assert torch.equal(optimized_attention["std"], full_attention["std"])


def _beam_search(customised_beam_search: bool) -> BeamSearch:
    beam = BeamSearch(
        pad=0,
        bos=1,
        eos=2,
        unk=3,
        batch_size=2,
        beam_size=2,
        n_best=1,
        max_length=5,
        customised_beam_search=customised_beam_search,
    )
    beam.initialize(torch.zeros(2, 2, 1), torch.tensor([2, 2]))
    return beam


def test_beam_search_reports_source_compaction_without_changing_results() -> None:
    beam = _beam_search(customised_beam_search=False)
    log_probs = torch.full((4, 5), -100.0)
    log_probs[0] = torch.tensor([-10.0, -10.0, 0.0, -1.0, -2.0])
    log_probs[2] = torch.tensor([-10.0, -10.0, -2.0, 0.0, -1.0])

    beam.advance(log_probs, attn=None)
    assert beam.update_finished()

    assert beam.batch_offset.tolist() == [1]
    assert beam.select_indices is not None
    assert beam.select_indices.tolist() == [2, 2]
    assert beam.predictions[0][0].tolist() == [beam.eos]
    assert beam.scores[0][0].item() == 0.0
    assert beam.predictions[1] == []


def test_beam_search_does_not_report_compaction_for_single_finished_beam() -> None:
    beam = _beam_search(customised_beam_search=True)
    log_probs = torch.full((4, 5), -100.0)
    log_probs[0] = torch.tensor([-10.0, -10.0, 0.0, -1.0, -2.0])
    log_probs[2] = torch.tensor([-10.0, -10.0, -2.0, 0.0, -1.0])

    beam.advance(log_probs, attn=None)
    assert not beam.update_finished()

    assert beam.batch_offset.tolist() == [0, 1]
    assert beam.select_indices is not None
    assert beam.select_indices.tolist() == [0, 0, 2, 2]


def test_beam_search_does_not_report_compaction_when_all_batches_finish() -> None:
    beam = _beam_search(customised_beam_search=False)
    log_probs = torch.full((4, 5), -100.0)
    log_probs[0] = torch.tensor([-10.0, -10.0, 0.0, -1.0, -2.0])
    log_probs[2] = torch.tensor([-10.0, -10.0, 0.0, -1.0, -2.0])

    beam.advance(log_probs, attn=None)
    assert not beam.update_finished()
    assert beam.done


class _RecordingDecoder:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, dict[str, bool]]] = []

    def init_state(self, src, enc_out, enc_final_hs) -> None:
        pass

    def map_state(self, fn, only_map_src=False, **kwargs) -> None:
        self.calls.append((only_map_src, kwargs))


class _ScriptedStrategy:
    parallel_paths = 2
    max_length = 3
    done = False
    scores: list[list[Any]] = [[], []]
    predictions: list[list[Any]] = [[], []]
    attention: list[list[Any]] = [[], []]

    def __init__(self) -> None:
        self.step = -1
        self.active_rows = 4
        self.batch_offset = torch.tensor([0, 1])
        self.select_indices = None
        self.is_finished = torch.zeros(2, 2, dtype=torch.bool)

    @property
    def current_predictions(self) -> torch.Tensor:
        return torch.ones(self.active_rows, dtype=torch.long)

    def initialize(self, memory_bank, src_lengths, src_map, target_prefix=None):
        memory_bank = tuple(tile(x, self.parallel_paths, dim=1) for x in memory_bank)
        memory_lengths = tile(src_lengths, self.parallel_paths)
        return None, memory_bank, memory_lengths, src_map

    def advance(self, log_probs, attn) -> None:
        self.step += 1
        if self.step == 0:
            self.select_indices = torch.tensor([1, 0, 3, 2])
            self.is_finished = torch.tensor([[True, False], [False, False]])
        elif self.step == 1:
            self.select_indices = torch.tensor([2, 3])
            self.is_finished = torch.tensor([[True, True], [False, False]])
        else:
            self.select_indices = torch.tensor([1, 0])
            self.is_finished = torch.zeros(1, 2, dtype=torch.bool)

    def update_finished(self) -> bool:
        if self.step == 0:
            return False
        self.active_rows = 2
        self.batch_offset = torch.tensor([1])
        return True


def test_translator_maps_source_rows_only_after_compaction() -> None:
    decoder = _RecordingDecoder()
    translator: Any = object.__new__(Translator)
    translator.model = SimpleNamespace(decoder=decoder)
    translator.tgt_prefix = False
    translator.customised_beam_search = False
    translator._tgt_pad_idx = 0
    translator._run_encoder = lambda batch: (
        batch["src"][0],
        None,
        (
            torch.tensor([[[10.0], [20.0]], [[11.0], [21.0]]]),
            torch.tensor([[[30.0], [40.0]], [[31.0], [41.0]]]),
        ),
        batch["src"][1],
    )
    observed_memory: list[
        tuple[tuple[torch.Tensor, ...], tuple[int, ...], torch.Tensor, torch.Tensor]
    ] = []

    def decode(decoder_input, memory_bank, batch, **kwargs):
        observed_memory.append(
            (
                tuple(x.clone() for x in memory_bank),
                tuple(x.data_ptr() for x in memory_bank),
                kwargs["memory_lengths"].clone(),
                kwargs["memory_padding_mask"].clone(),
            )
        )
        return torch.zeros(decoder_input.size(1), 5), None

    translator._decode_and_generate = decode
    batch = {
        "src": (torch.ones(2, 2, 1, dtype=torch.long), torch.tensor([1, 2])),
        "batch_size": 2,
    }
    translator._translate_batch_with_strategy(batch, _ScriptedStrategy())

    assert decoder.calls == [
        (
            False,
            {"map_src": False, "map_context": False, "map_self": True},
        ),
        (
            False,
            {"map_src": True, "map_context": True, "map_self": True},
        ),
        (
            False,
            {"map_src": False, "map_context": False, "map_self": True},
        ),
    ]
    assert observed_memory[0][1] == observed_memory[1][1]
    assert torch.equal(observed_memory[0][0][0], observed_memory[1][0][0])
    assert observed_memory[2][0][0][:, :, 0].tolist() == [[20.0, 21.0], [20.0, 21.0]]
    assert observed_memory[2][2].tolist() == [2, 2]
    assert observed_memory[2][3].tolist() == [[False, False], [False, False]]
