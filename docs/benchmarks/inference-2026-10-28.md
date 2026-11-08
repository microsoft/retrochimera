# Inference benchmarks

**Full Pistachio ensemble**, one RTX A6000; Python 3.12.13, Torch 2.8.0+cu128, RDKit 2025.09.2;
Syntheseus `7905e45`.

## Isolated #45 contribution — 2026-10-30

FP16 and `matmul=highest` on both sides; 1,984 reactions/pass, batch 16, top-50. Two passes/version
in baseline → PR → PR → baseline order. Only the canonicalization wrapper differs at runtime;
all other model/runtime settings match those below.

| FP16 configuration | Chunk size | Two model-call times (s) | Mean (s) | Throughput vs FP16 baseline |
|---|---:|---|---:|---:|
| Without #45 (`09f2dc7`, PR base) | 1 | 905.473, 905.478 | 905.48 | 1.000× |
| With #45 (`4b50a39`) | 8 | 839.478, 837.660 | 838.57 | 1.080× |

**7.98% more throughput; 7.39% less model-call time.** Inputs, augmentation/RNG, checkpoints,
cache policy and workers matched. All per-record top-1/3/5/10/50 correctness flags matched;
maximum within-version timing spread was 0.217%, with no observed GPU co-tenancy.

## Earlier combined configurations — 2026-10-28

RetroChimera baseline `09f2dc7`, chunking `aeb4647`, experimental port `68d1793` (unpublished).

Shared model settings: [fast-fp16.yaml](../configs/fast-fp16.yaml), except precision/chunking below.
Weights stayed FP32. Runtime: eight Torch threads, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`,
`OPENBLAS_NUM_THREADS=1`, CPU affinity 0–15 and 64–79, cuDNN TF32 disabled.
“Matmul” is Torch's process-wide `float32_matmul_precision`; `high` allows TF32 independently
of transformer autocast.

| Profile | Matmul | Chunk size | Additional code | Call time | Throughput vs FP32 |
|---|---|---:|---|---:|---:|
| FP32 baseline | `highest` | 1 | None | 1,302.21 s | 1.000× |
| FP16 preset | `highest` | 8 | None | 834.97 s | 1.560× |
| FP16 experimental | `high` | 8 | CPU bookkeeping | 743.02 s | 1.753× |
| BF16 experimental | `high` | 8 | Bookkeeping, fused attention, CUDA KV cache | 500.02 s | 2.604× |

Experimental options: `attention_backend=fused`; `decoder_cache_backend=legacy` (FP16) or `cuda`
(BF16). Unpublished code only; FP16 fell back to legacy attention/cache. These kwargs are unsupported here.

## Oct28 inference protocol

- 1,984 matched test reactions; batch 16, top-50, `skip_repeats=True`; parent/submodel caches off.
- Seed 0 before model loading; one untimed warmup on `Oc1ccc(OCc2ccccc2)c(Br)c1`, requesting 50
  results. Retain native RNG behavior; do not reseed each batch.
- CUDA synchronization brackets each model call. Times exclude loading, compilation, warmup,
  dataset/metric processing and output writes; standard CLI wall time is not equivalent.
- Two FP32 controls, one pass per other profile; control spread **0.253%**, no GPU co-tenancy.
  Saved predictions were checked against official accuracy statistics.

The Oct28 speedups combine precision and other changes; they are not #45-only or cold-start/AOT gains.

Absolute top-1 / top-50 accuracy: **FP32 and FP16 46.9758% / 93.0948%**;
**BF16 46.8750% / 92.9940%** (−0.1008 percentage points each versus FP32).
Versus FP32, FP16 top-3/top-5 changed −0.0504 pp and top-10 +0.0504 pp. Rankings were not identical.

## Native Syntheseus search — 2026-10-28

Two opposite-order sweeps of **200 targets** per profile:

- `max_active_searches=32`, `inference_batch_size=16`, `inference_batch_wait_s=0.01`.
- Retro*, depth 20; `time_limit_s=30`, `limit_reaction_model_calls=1000`, `limit_iterations=100000`.
  Other settings follow [native RetroChimera defaults](https://github.com/microsoft/syntheseus/blob/7905e45ef7de5d2fac2597f46dd0ad60412015e9/syntheseus/cli/search_config.yml).
- Top-20 plus a separate GINE-PNA feasibility model, threshold 0.55; **not supplied by the preset**.
- 812,020-entry inventory, canonicalized once then loaded with `canonicalize_inventory=False`.
- Backend/submodel caches off; `reaction_model_use_cache=True` for per-target facades.
- Timing includes per-target statistics/graph writes, excludes setup and graph readback.

| Profile | Solved / 200, two sweeps | Median search wall |
|---|---:|---:|
| FP32 baseline | 24, 22 | 321.67 s |
| FP16 preset | 29, 29 | 227.14 s |
| FP16 experimental | 47, 46 | 262.90 s |
| BF16 experimental | 64, 68 | 240.95 s |

Statistics and graphs were read back for all 1,600 target runs. Time-limited solves reflect
different work, not fixed-work latency or chemical accuracy; in-flight inference can exceed the budget.
These are A6000 observations, not production/A100 certification. Test data and feasibility
artifacts are not bundled. See the [usage guide](../inference-performance.md) for the runnable preset.
