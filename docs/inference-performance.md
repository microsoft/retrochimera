# Fast inference

[fast-fp16.yaml](configs/fast-fp16.yaml) is an opt-in CUDA profile for the full Pistachio
ensemble. FP32 remains the default; validate accuracy before adopting FP16.

From this checkout's CUDA environment, select an available GPU with `CUDA_VISIBLE_DEVICES`.
Set `MODEL_DIR` to the ensemble checkpoint, `DATA_DIR` to a preprocessed dataset, and
`RESULTS_DIR` to a fresh output directory. The CLI below was tested with Syntheseus `7905e45`.

```sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1

python -m retrochimera.cli.eval \
    --config docs/configs/fast-fp16.yaml \
    model_dir="$MODEL_DIR" \
    data_dir="$DATA_DIR" \
    results_dir="$RESULTS_DIR" \
    model_kwargs.use_cache=False \
    num_gpus=1 fold=TEST batch_size=16 num_top_results=50 \
    float32_matmul_precision=highest \
    resumable=True
```

Outputs under `RESULTS_DIR/resumable_TEST/`: `stats.json` and `predictions.jsonl`.
Completed results are reused; a fresh measurement needs a new directory.

## Notes

- Submodel keys must match the checkpoint's `models.json`.
- Tune batch size and workers for your hardware. The benchmarked FP16 profile reserved
  19.50 GiB of GPU memory despite a 7.01 GiB peak allocation.
- `canonicalization_chunksize` must be a positive integer, not a Boolean; eight is the default,
  one restores unchunked scheduling. Chunking preserves output order and scores.
- Caches are disabled for uncached measurements, not because disabling them is always fastest.
- For search, pass the same `--config` to `python -m retrochimera.cli.run_search` and use
  [Syntheseus's concurrency options](https://github.com/microsoft/syntheseus/blob/main/docs/cli/search.md#concurrent-targets-and-batched-inference).
  Omit `model_kwargs.use_cache` (search manages the backend cache) and
  `float32_matmul_precision` (evaluation-only CLI field). No feasibility filter is supplied.
- Follow the [model-use limitations](../README.md#checkpoints-for-retrochimera-1).

See the [benchmark record](benchmarks/inference-2026-10-28.md) for runtime/timing conditions.
The 1.753×/2.604× experimental cases require unmerged code; this preset does not enable them.
