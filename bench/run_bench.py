"""Benchmark RetroChimera inference end-to-end and per-branch.

Usage:
    python run_bench.py --n 32 --out outputs.json [--branch loc|transformer|both]
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

TEST_PATH = "/data/projects/reaction_prediction/data/preprocessed/pistachio_2023Q2_v2/test.jsonl"
CKPT_DIR = "/datahdd/model_paper_results/pistachio/checkpoints/ensemble"


def load_inputs(n: int):
    from syntheseus import Molecule
    mols = []
    with open(TEST_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            d = json.loads(line)
            mols.append(Molecule(d["products"][0]["smiles"]))
    return mols


def serialize_predictions(predictions):
    """Serialize predictions deterministically to compare across runs."""
    out = []
    for ranked in predictions:
        per_input = []
        for r in ranked:
            try:
                rxnts = sorted(m.smiles for m in r.reactants)
            except Exception:
                rxnts = sorted(str(m) for m in r.reactants)
            md = r.metadata or {}
            per_input.append({
                "reactants": rxnts,
                "score": float(md.get("score", 0.0)),
                "probability": float(md.get("probability", 0.0)),
            })
        out.append(per_input)
    return out


def time_runs(model, inputs, num_results, batch_size, n_warmup=1, n_repeat=1):
    """Run inference; returns (predictions_serialized, mean_total_ms_per_mol, peak_mem_mb)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup batches not counted.
    for _ in range(n_warmup):
        for i in range(0, min(len(inputs), batch_size), batch_size):
            chunk = inputs[i:i + batch_size]
            _ = model(chunk, num_results=num_results)
    torch.cuda.synchronize()

    times = []
    last_preds = []
    for rep in range(n_repeat):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_preds = []
        for i in range(0, len(inputs), batch_size):
            chunk = inputs[i:i + batch_size]
            preds = model(chunk, num_results=num_results)
            all_preds.extend(preds)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / len(inputs))
        last_preds = all_preds

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return serialize_predictions(last_preds), min(times), peak_mem_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--num-results", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--branch", choices=["loc", "transformer", "ensemble"], default="ensemble")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    inputs = load_inputs(args.n)
    print(f"Loaded {len(inputs)} molecules.")

    device = "cuda:0"

    if args.branch == "ensemble":
        from retrochimera import RetroChimeraModel
        model = RetroChimeraModel(model_dir=CKPT_DIR, device=device)
    elif args.branch == "loc":
        from retrochimera.inference.template_localization import TemplateLocalizationModel
        model = TemplateLocalizationModel(
            model_dir=os.path.join(CKPT_DIR, "template_localization"), device=device
        )
    elif args.branch == "transformer":
        from retrochimera.inference.smiles_transformer import SmilesTransformerModel
        model = SmilesTransformerModel(
            model_dir=os.path.join(CKPT_DIR, "smiles_transformer"), device=device
        )
    else:
        raise SystemExit(args.branch)

    print(f"Model loaded: {type(model).__name__}")

    preds, mean_ms, peak_mb = time_runs(
        model, inputs, args.num_results,
        batch_size=args.batch_size,
        n_warmup=args.warmup, n_repeat=args.repeat,
    )

    result = {
        "branch": args.branch,
        "n_inputs": len(inputs),
        "num_results": args.num_results,
        "batch_size": args.batch_size,
        "mean_latency_ms_per_mol": mean_ms,
        "peak_mem_mb": peak_mb,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "predictions": preds,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)

    print(f"Branch={args.branch} N={len(inputs)} mean={mean_ms:.2f} ms/mol peak={peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
