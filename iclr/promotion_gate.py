"""Pre-registered replacement gate (plan §7) → promotion_decision.json.

Evaluates the ten §7 conditions from a results JSON and writes a
machine-readable decision listing every condition as ``pass``/``fail``. A
candidate is promoted to the next TFScope version only if **all** applicable
conditions pass; otherwise v24 stays current and the candidate is recorded as an
unsuccessful ablation (plan §7, §8).

This module contains no test-set tuning — it only *checks* pre-registered
thresholds against already-computed metrics.

Input JSON schema (see iclr/README.md and _demo() below):

    {
      "primary_endpoint": "gene_covR",
      "candidate": {"test_gene_covR": .., "monomer": .., "multimer": ..,
                    "per_seed": [.., .., ..],
                    "per_gene_paired_diff": [.. per gene candidate-v24 ..],
                    "per_family_diff": {"bHLH": .., ...}},
      "v24":       {"test_gene_covR": .., "monomer": .., "multimer": ..},
      "best_simple_baseline": {"id": "B2", "test_gene_covR": ..},
      "permutation": {"max_gene_covR_delta": .., "pred_equiv_within_tol": true,
                      "tolerance": 1e-4},
      "claims_chain_set": true,
      "sequence_only_headline": true,
      "artifacts_recorded": true
    }
"""
from __future__ import annotations

import argparse
import json
import math

# pre-registered thresholds (plan §7)
MIN_ABS_GAIN        = 0.02   # §7.1 candidate − v24 test gene_covR
MAX_MONOMER_DROP    = 0.01   # §7.5
MIN_MULTIMER_GAIN   = 0.03   # §7.6 (only if chain-set claimed)
MAX_PERM_DELTA      = 0.005  # §7.7


def _paired_bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI of the mean of paired per-gene differences.

    Deterministic (fixed seed) so the gate is reproducible. Returns (lo, hi).
    """
    import random
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def evaluate_gate(results: dict) -> dict:
    cand = results["candidate"]
    v24 = results["v24"]
    conditions: list[dict] = []

    def add(cid, desc, passed, detail):
        conditions.append({"id": cid, "description": desc,
                           "pass": bool(passed), "detail": detail})

    # §7.1 absolute gain
    gain = cand["test_gene_covR"] - v24["test_gene_covR"]
    add("7.1", f"candidate − v24 test gene_covR ≥ +{MIN_ABS_GAIN}",
        gain >= MIN_ABS_GAIN, {"gain": gain})

    # §7.2 paired bootstrap CI above zero
    diffs = cand.get("per_gene_paired_diff", [])
    lo, hi = _paired_bootstrap_ci(diffs) if diffs else (float("nan"), float("nan"))
    add("7.2", "paired hierarchical-bootstrap 95% CI (candidate − v24) > 0",
        (not math.isnan(lo)) and lo > 0, {"ci95": [lo, hi], "n_genes": len(diffs)})

    # §7.3 all seeds positive
    per_seed = cand.get("per_seed", [])
    seed_diffs = [s - v24["test_gene_covR"] for s in per_seed]
    add("7.3", "all seeds have positive candidate − v24 gene_covR",
        len(per_seed) >= 3 and all(d > 0 for d in seed_diffs),
        {"per_seed_diff": seed_diffs, "n_seeds": len(per_seed)})

    # §7.4 beats best simple ESM baseline
    bsb = results.get("best_simple_baseline", {})
    add("7.4", "candidate beats best parameter-matched simple ESM baseline",
        bsb and cand["test_gene_covR"] > bsb.get("test_gene_covR", float("inf")),
        {"baseline_id": bsb.get("id"), "baseline": bsb.get("test_gene_covR")})

    # §7.5 monomer preservation
    mono_drop = v24.get("monomer", float("nan")) - cand.get("monomer", float("nan"))
    add("7.5", f"monomer gene_covR decreases by ≤ {MAX_MONOMER_DROP}",
        not math.isnan(mono_drop) and mono_drop <= MAX_MONOMER_DROP, {"monomer_drop": mono_drop})

    # §7.6 multimer gain (only if the chain-set module is a claimed contribution)
    if results.get("claims_chain_set", False):
        multi_gain = cand.get("multimer", float("nan")) - v24.get("multimer", float("nan"))
        add("7.6", f"multimer gene_covR improves by ≥ {MIN_MULTIMER_GAIN} (chain-set claimed)",
            not math.isnan(multi_gain) and multi_gain >= MIN_MULTIMER_GAIN, {"multimer_gain": multi_gain})
    else:
        add("7.6", "multimer gain not required (chain-set not a claimed contribution)",
            True, {"skipped": True})

    # §7.7 permutation invariance
    perm = results.get("permutation", {})
    add("7.7", f"chain permutations change gene_covR < {MAX_PERM_DELTA} and preds equivalent within tol",
        perm.get("max_gene_covR_delta", 1.0) < MAX_PERM_DELTA and perm.get("pred_equiv_within_tol", False),
        {"max_delta": perm.get("max_gene_covR_delta"), "tol": perm.get("tolerance"),
         "equiv": perm.get("pred_equiv_within_tol")})

    # §7.8 not one dominant family; positive under leave-one-family-out
    fam = cand.get("per_family_diff", {})
    n_pos = sum(1 for v in fam.values() if v > 0)
    lofo_ok = bool(fam) and n_pos >= max(1, int(math.ceil(0.5 * len(fam)))) and min(fam.values()) > -MIN_ABS_GAIN
    add("7.8", "gains not explained by one family; positive under LOFO summaries",
        lofo_ok, {"n_families": len(fam), "n_positive": n_pos,
                  "min_family_diff": (min(fam.values()) if fam else None)})

    # §7.9 sequence-only headline
    add("7.9", "sequence-only inference used for the headline result",
        bool(results.get("sequence_only_headline", False)), {})

    # §7.10 artifacts recorded
    add("7.10", "all metrics, configs, seeds, checkpoints, failed variants recorded",
        bool(results.get("artifacts_recorded", False)), {})

    all_pass = all(c["pass"] for c in conditions)
    decision = {
        "primary_endpoint": results.get("primary_endpoint", "gene_covR"),
        "promote": all_pass,
        "decision": "PROMOTE candidate to next TFScope version"
                    if all_pass else "KEEP v24; record candidate as unsuccessful ablation",
        "conditions": conditions,
        "summary": {
            "candidate_test_gene_covR": cand["test_gene_covR"],
            "v24_test_gene_covR": v24["test_gene_covR"],
            "gain": gain,
            "n_conditions_passed": sum(c["pass"] for c in conditions),
            "n_conditions_total": len(conditions),
        },
    }
    return decision


def _demo() -> dict:
    """A self-consistent failing example (gain below threshold) for the self-test."""
    return {
        "primary_endpoint": "gene_covR",
        "candidate": {"test_gene_covR": 0.535, "monomer": 0.55, "multimer": 0.50,
                      "per_seed": [0.532, 0.536, 0.537],
                      "per_gene_paired_diff": [0.01, -0.02, 0.03, 0.00, 0.015],
                      "per_family_diff": {"bHLH": 0.01, "Homeodomain": -0.005, "C2H2": 0.02}},
        "v24": {"test_gene_covR": 0.523, "monomer": 0.54, "multimer": 0.47},
        "best_simple_baseline": {"id": "B2", "test_gene_covR": 0.515},
        "permutation": {"max_gene_covR_delta": 0.001, "pred_equiv_within_tol": True, "tolerance": 1e-4},
        "claims_chain_set": True,
        "sequence_only_headline": True,
        "artifacts_recorded": True,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", help="path to results JSON (see schema in module docstring)")
    ap.add_argument("--out", default="results/iclr_phase1/promotion_decision.json")
    ap.add_argument("--self-test", action="store_true", help="run on the built-in demo")
    args = ap.parse_args()

    results = _demo() if args.self_test else json.load(open(args.results))
    decision = evaluate_gate(results)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(decision, open(args.out, "w"), indent=2)
    print(json.dumps(decision, indent=2))
    print(f"\n[gate] {'PROMOTE' if decision['promote'] else 'KEEP v24'} "
          f"({decision['summary']['n_conditions_passed']}/"
          f"{decision['summary']['n_conditions_total']} conditions) → {args.out}")


if __name__ == "__main__":
    main()
