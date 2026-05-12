"""Compare two prediction JSONs for relaxed-equality vs. a baseline.

Relaxed acceptance rule (single source of truth for all experiments):

1. Pool every (input_idx, rank_idx) slot across the eval set into one flat sequence.
   The predicted reactant-SMILES at the corresponding slot before vs. after the
   change must match in at least `min_slot_match_rate` of all slots
   (default 0.999, i.e. at most 0.1% of slots may differ).

2. For slots where the SMILES match, the per-slot probability must agree within
   `rtol` relative tolerance (and `atol` absolute floor). Mismatches on this
   probability rule count as slot mismatches too (a probability divergence on
   an otherwise-matching slot is treated as a divergence).

Both `n_inputs` and per-input rank counts must still match exactly: we never
allow shrinking the prediction set.
"""
import argparse
import json
import math


def _slot_matches(b, c, rtol, atol):
    """Return (smiles_match, prob_ok). prob_ok only meaningful when smiles_match."""
    if b["reactants"] != c["reactants"]:
        return False, False
    bv, cv = b.get("probability", 0.0), c.get("probability", 0.0)
    if math.isnan(bv) and math.isnan(cv):
        return True, True
    if math.isnan(bv) or math.isnan(cv):
        return True, False
    prob_ok = abs(bv - cv) <= atol + rtol * abs(bv)
    return True, prob_ok


def compare(base, cand, rtol=1e-3, atol=1e-5, min_slot_match_rate=0.999):
    """Returns dict(ok, total_slots, slot_mismatches, slot_match_rate,
                    smiles_mismatches, prob_only_mismatches,
                    max_prob_abs_delta, max_prob_rel_delta, msg)."""
    if base["n_inputs"] != cand["n_inputs"]:
        return {
            "ok": False,
            "msg": f"n_inputs differ: {base['n_inputs']} vs {cand['n_inputs']}",
        }
    bp, cp = base["predictions"], cand["predictions"]
    if len(bp) != len(cp):
        return {
            "ok": False,
            "msg": f"prediction list lengths differ: {len(bp)} vs {len(cp)}",
        }

    total_slots = 0
    smiles_mismatches = 0
    prob_only_mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    first_diff = None

    for i, (b_list, c_list) in enumerate(zip(bp, cp)):
        if len(b_list) != len(c_list):
            return {
                "ok": False,
                "msg": f"input {i}: pred count differs {len(b_list)} vs {len(c_list)}",
            }
        for j, (b, c) in enumerate(zip(b_list, c_list)):
            total_slots += 1
            smi_ok, prob_ok = _slot_matches(b, c, rtol, atol)
            if not smi_ok:
                smiles_mismatches += 1
                if first_diff is None:
                    first_diff = (i, j, "smiles", b["reactants"], c["reactants"])
            elif not prob_ok:
                prob_only_mismatches += 1
                bv = float(b.get("probability", 0.0))
                cv = float(c.get("probability", 0.0))
                ad = abs(bv - cv)
                rd = ad / abs(bv) if bv != 0 else float("inf")
                if ad > max_abs:
                    max_abs = ad
                if rd > max_rel:
                    max_rel = rd
                if first_diff is None:
                    first_diff = (i, j, "probability", bv, cv)
            else:
                # also track abs/rel deltas on matched slots for diagnostics
                bv = float(b.get("probability", 0.0))
                cv = float(c.get("probability", 0.0))
                ad = abs(bv - cv)
                rd = ad / abs(bv) if bv != 0 else 0.0
                if ad > max_abs:
                    max_abs = ad
                if rd > max_rel:
                    max_rel = rd

    mismatches = smiles_mismatches + prob_only_mismatches
    rate = 1.0 - (mismatches / total_slots) if total_slots else 1.0
    ok = rate >= min_slot_match_rate

    msg = (
        f"slots={total_slots} mismatches={mismatches} "
        f"(smiles={smiles_mismatches}, prob_only={prob_only_mismatches}) "
        f"match_rate={rate:.6f} threshold={min_slot_match_rate} "
        f"max_abs_prob_delta={max_abs:.3e} max_rel_prob_delta={max_rel:.3e}"
    )
    if not ok and first_diff is not None:
        msg += f"  first_diff={first_diff}"

    return {
        "ok": ok,
        "total_slots": total_slots,
        "slot_mismatches": mismatches,
        "smiles_mismatches": smiles_mismatches,
        "prob_only_mismatches": prob_only_mismatches,
        "slot_match_rate": rate,
        "max_prob_abs_delta": max_abs,
        "max_prob_rel_delta": max_rel,
        "rtol": rtol,
        "atol": atol,
        "min_slot_match_rate": min_slot_match_rate,
        "msg": msg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--min-slot-match-rate", type=float, default=0.999)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    with open(args.base) as f:
        base = json.load(f)
    with open(args.cand) as f:
        cand = json.load(f)
    res = compare(base, cand, args.rtol, args.atol, args.min_slot_match_rate)
    if args.json:
        print(json.dumps(res))
    else:
        print("EQUALITY_OK" if res["ok"] else "EQUALITY_FAIL", res["msg"])
        bms = base.get("mean_latency_ms_per_mol")
        cms = cand.get("mean_latency_ms_per_mol")
        bmem = base.get("peak_mem_mb")
        cmem = cand.get("peak_mem_mb")
        if bms and cms:
            print(f"latency: {bms:.2f} -> {cms:.2f} ms/mol  ({(cms-bms)/bms*100:+.1f}%)")
        if bmem and cmem:
            print(f"peak mem: {bmem:.1f} -> {cmem:.1f} MB  ({(cmem-bmem)/bmem*100:+.1f}%)")


if __name__ == "__main__":
    main()
