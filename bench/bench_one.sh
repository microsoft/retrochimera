#!/usr/bin/env bash
# Helper: run quick (N=32) bench for one branch and compare to its baseline.
# Usage: bench_one.sh <branch> <out_path> [--n N] [--batch B]
set -euo pipefail
export PYTHONUSERBASE=/datahdd/agent/retrochimera/.userbase
export PATH=$PYTHONUSERBASE/bin:$PATH
BRANCH=${1:?branch}
OUT=${2:?out_path}
shift 2
N_ARG=${N_ARG:-32}
BS_ARG=${BS_ARG:-8}
cd /datahdd/agent/retrochimera/RetroChimera
python3 bench/run_bench.py --branch "$BRANCH" --n "$N_ARG" --batch-size "$BS_ARG" --out "$OUT" "$@"
case "$BRANCH" in
  loc) BASE=bench/baseline_quick_loc.json;;
  transformer) BASE=bench/baseline_quick_transformer.json;;
  ensemble) BASE=bench/baseline_quick_ensemble.json;;
esac
python3 bench/compare.py "$BASE" "$OUT"
