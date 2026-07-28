"""Paired v24-vs-B0 statistics from the frozen unified-evaluator per-sample output.

Answers: is the Panel-A content gain (and the Panel-B covR difference)
statistically supported? Uses gene-level pairing (equal-weight genes) and a
*hierarchical* bootstrap (resample genes, then rows within gene) for the 95% CI.
Also reports leave-one-family-out and the drop-{p53,POU} sensitivity.

No model inference; reads:
  results/iclr_phase1_apples_to_apples/unified_models.json    (v24 = B8_v24)
  results/iclr_phase1_apples_to_apples/unified_baselines.json (B0)
"""
from __future__ import annotations
import json, os, random
import numpy as np

ROOT = "results/iclr_phase1_apples_to_apples"
N_BOOT = 10000
SEED = 0


def _load_per_sample(path, key):
    d = json.load(open(path))[key]
    return {s["filename"]: s for s in d["per_sample"]}


import sys
MODEL_KEY = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "B8_v24"
BASE_KEY = sys.argv[sys.argv.index("--baseline") + 1] if "--baseline" in sys.argv else "B0"
OUT_SUFFIX = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else "B0"


def _gene_table(ps_a, ps_b, metric):
    """Return {gene: {"family":.., "a":[rows], "b":[rows]}} over shared filenames."""
    genes = {}
    for fn, sa in ps_a.items():
        sb = ps_b.get(fn)
        if sb is None:
            continue
        g = sa["gene"]
        e = genes.setdefault(g, {"family": sa.get("family", "NA"), "a": [], "b": []})
        e["a"].append(sa[metric]); e["b"].append(sb[metric])
    return genes


def _mean_delta(genes, gene_subset=None):
    gs = gene_subset if gene_subset is not None else list(genes.keys())
    da = [np.mean(genes[g]["a"]) for g in gs]
    db = [np.mean(genes[g]["b"]) for g in gs]
    return float(np.mean(da) - np.mean(db)), float(np.mean(da)), float(np.mean(db))


def _hier_bootstrap_ci(genes, rng, n_boot=N_BOOT):
    gs = list(genes.keys()); n = len(gs)
    deltas = []
    for _ in range(n_boot):
        boot_genes = [gs[rng.randrange(n)] for _ in range(n)]
        per_gene_delta = []
        for g in boot_genes:
            a = genes[g]["a"]; b = genes[g]["b"]; m = len(a)
            idx = [rng.randrange(m) for _ in range(m)]      # resample rows within gene
            per_gene_delta.append(np.mean([a[i] for i in idx]) - np.mean([b[i] for i in idx]))
        deltas.append(np.mean(per_gene_delta))
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]; hi = deltas[int(0.975 * n_boot)]
    frac_pos = float(np.mean([d > 0 for d in deltas]))
    return float(lo), float(hi), frac_pos


def analyze(metric):
    ps_v = _load_per_sample(f"{ROOT}/unified_models.json", MODEL_KEY)
    ps_b = _load_per_sample(f"{ROOT}/unified_baselines.json", BASE_KEY)
    genes = _gene_table(ps_v, ps_b, metric)
    rng = random.Random(SEED)
    delta, mean_a, mean_b = _mean_delta(genes)
    lo, hi, frac_pos = _hier_bootstrap_ci(genes, rng)

    # per-gene sign
    per_gene = {g: float(np.mean(genes[g]["a"]) - np.mean(genes[g]["b"])) for g in genes}
    n_pos = sum(1 for v in per_gene.values() if v > 0)

    # leave-one-family-out
    fams = sorted(set(genes[g]["family"] for g in genes))
    lofo = {}
    for f in fams:
        subset = [g for g in genes if genes[g]["family"] != f]
        lofo[f] = _mean_delta(genes, subset)[0]

    # drop p53/POU
    subset = [g for g in genes if genes[g]["family"] not in ("p53", "POU")]
    drop_delta, da, db = _mean_delta(genes, subset)

    return {
        "metric": metric,
        "mean_v24": mean_a, "mean_B0": mean_b, "gene_delta_v24_minus_B0": delta,
        "hier_bootstrap_95ci": [lo, hi], "bootstrap_frac_delta_gt0": frac_pos,
        "ci_excludes_zero": (lo > 0) or (hi < 0),
        "n_genes": len(genes), "n_genes_v24_better": n_pos,
        "leave_one_family_out_delta": lofo,
        "drop_p53_POU": {"delta": drop_delta, "mean_v24": da, "mean_B0": db,
                         "n_genes": len(subset)},
    }


def main():
    out = {"note": "v24(B8 seed42) vs B0(family-avg by family_id). Single seed; "
                   "B0 uses family_id taxonomy (p53->Other, POU->Homeodomain).",
           "panelA_content_r": analyze("A_content_r"),
           "panelB_covR": analyze("B_covR")}
    out["model_key"] = MODEL_KEY; out["baseline_key"] = BASE_KEY
    os.makedirs(ROOT, exist_ok=True)
    json.dump(out, open(f"{ROOT}/paired_stats_v24_vs_{OUT_SUFFIX}.json", "w"), indent=2)
    for k in ["panelA_content_r", "panelB_covR"]:
        r = out[k]
        print(f"\n=== {k} ===")
        print(f"  v24={r['mean_v24']:.4f}  B0={r['mean_B0']:.4f}  Δ(gene)={r['gene_delta_v24_minus_B0']:+.4f}")
        print(f"  95% CI [{r['hier_bootstrap_95ci'][0]:+.4f}, {r['hier_bootstrap_95ci'][1]:+.4f}]  "
              f"excludes 0: {r['ci_excludes_zero']}  P(Δ>0)={r['bootstrap_frac_delta_gt0']:.3f}")
        print(f"  genes v24-better: {r['n_genes_v24_better']}/{r['n_genes']}")
        print(f"  drop p53/POU: Δ={r['drop_p53_POU']['delta']:+.4f} (n_genes={r['drop_p53_POU']['n_genes']})")
        print(f"  LOFO Δ range: {min(r['leave_one_family_out_delta'].values()):+.4f} .. "
              f"{max(r['leave_one_family_out_delta'].values()):+.4f}")
    print(f"\nsaved {ROOT}/paired_stats_v24_vs_B0.json")


if __name__ == "__main__":
    main()
