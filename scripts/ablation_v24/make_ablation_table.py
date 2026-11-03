#!/usr/bin/env python
"""Turn the scored v24 ablations into the paper table + a machine-readable summary.

Two numbers per variant, both gene-balanced over the 291-row structure test set:

  Panel A gene_content_r  (GT length, oracle shift/RC)  — motif content
  Panel B gene_covR       (own predicted gate span)     — end-to-end

and, against the same-harness full-v24 reference (A0_full), a *paired* delta with a
hierarchical bootstrap 95% CI (resample genes, then rows within gene) — the same
estimator `iclr/paired_stats.py` uses for v24-vs-B0. Pairing matters here: the raw
seed spread of v24 on this benchmark is large (content_r 0.529-0.665 across five
seeds, sd 0.055), so an unpaired single-seed difference below ~0.11 is unreadable.
The paired CI removes the shared per-gene difficulty and is what should be quoted.

Seeds are pooled two ways and both are printed, because they answer different questions:
  * mean +/- sd over seeds        — "how big is the effect relative to seed noise"
  * seed-matched paired delta     — "for a fixed seed, what does removing X cost"

    scripts/ablation_v24/make_ablation_table.py
    scripts/ablation_v24/make_ablation_table.py --metric A_content_r --boot 10000
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

IN = "results/ablation_v24/unified_ablation.json"
OUT_JSON = "results/ablation_v24/ablation_summary.json"
OUT_MD = "results/ablation_v24/ablation_table.md"
REF = "A0_full"
REF_FALLBACK = "A0_shipped"   # the shipped 6-GPU seed-42 v24, scored as ICLR B8_v24

# display order + one-line "what was removed", kept in sync with train_ablation_v24.sh
VARIANTS = [
    ("A0_shipped",     "shipped v24 seed-42 (6-GPU DDP; = ICLR B8_v24)"),
    ("A0_full",        "complete v24 (same-harness reference)"),
    ("A1_deeppbsfold", "byte-identical DeepPBS fold 0 + DeepPBS masked-PWM labels"),
    ("A1b_structonly",  "train on structure-backed rows only (DeepPBS data regime)"),
    ("A2_nodistill",   "- 2-D contact-distillation KL"),
    ("A3_noattnsup",   "- 1-D recognition-residue attention supervision"),
    ("A4_nobias",      "- learnable contact bias at the head"),
    ("A5_shufcontact", "contact terms kept, residue targets permuted (control)"),
    ("A6_nolora",      "- LoRA (ESM-2 fully frozen)"),
    ("A7_singlechain", "- N-chain protomer input"),
]


def load(path):
    d = json.load(open(path))
    runs = defaultdict(dict)          # tag -> seed -> per_sample dict
    for key, res in d.items():
        m = re.match(r"^(.*)_s(\d+)$", key)
        if not m:
            continue
        tag, seed = m.group(1), int(m.group(2))
        runs[tag][seed] = {s["filename"]: s for s in res["per_sample"]}
    return runs


def gene_mean(ps, metric, genes=None):
    """Equal-weight-over-genes mean of `metric`."""
    by = defaultdict(list)
    for s in ps.values():
        by[s["gene"]].append(s[metric])
    gs = genes if genes is not None else list(by)
    vals = [np.mean(by[g]) for g in gs if g in by]
    return float(np.mean(vals)) if vals else float("nan")


def paired_delta(ps_a, ps_ref, metric, n_boot, rng):
    """mean_gene(a) - mean_gene(ref) over shared filenames + hierarchical bootstrap CI."""
    by = defaultdict(lambda: {"a": [], "b": []})
    for fn, sa in ps_a.items():
        sb = ps_ref.get(fn)
        if sb is None:
            continue
        by[sa["gene"]]["a"].append(sa[metric])
        by[sa["gene"]]["b"].append(sb[metric])
    genes = sorted(by)
    if not genes:
        return None
    def stat(gs, resample_rows):
        da, db = [], []
        for g in gs:
            a, b = by[g]["a"], by[g]["b"]
            if resample_rows and len(a) > 1:
                idx = rng.integers(0, len(a), len(a))
                da.append(np.mean([a[i] for i in idx])); db.append(np.mean([b[i] for i in idx]))
            else:
                da.append(np.mean(a)); db.append(np.mean(b))
        return float(np.mean(da) - np.mean(db))
    point = stat(genes, False)
    boots = np.array([stat([genes[i] for i in rng.integers(0, len(genes), len(genes))], True)
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    wins = sum(1 for g in genes if np.mean(by[g]["a"]) > np.mean(by[g]["b"]))
    return {"delta": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_genes": len(genes), "wins": wins,
            "significant": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=IN)
    ap.add_argument("--metric", default="both", choices=["both", "A_content_r", "B_covR"])
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leak-free", action="store_true",
                    help="drop the 16 test genes that appear in DeepPBS fold 0 from EVERY "
                         "model before scoring (226 rows / 35 genes). The folds were made "
                         "disjoint from the 20 struct-having test genes, not from all 51, "
                         "so A1_deeppbsfold is otherwise flattered by leakage.")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit(f"no scored runs at {args.inp} — run scripts/ablation_v24/score_ablation.py")
    runs = load(args.inp)
    leaky = set()
    if args.leak_free:
        lp = "data/processed/splits/ablation_v24/deeppbs_fold0_leaky_genes.json"
        if not os.path.exists(lp):
            raise SystemExit(f"--leak-free needs {lp} "
                             f"(scripts/ablation_v24/build_deeppbs_fold_dataset.py)")
        leaky = set(json.load(open(lp))["leaky_test_genes"])
        # filter every model identically so the paired comparison stays like-for-like
        runs = {t: {s_: {fn: r for fn, r in ps.items()
                         if str(r["gene"]).upper() not in leaky}
                    for s_, ps in by_seed.items()}
                for t, by_seed in runs.items()}
    rng = np.random.default_rng(args.seed)
    metrics = ["A_content_r", "B_covR"] if args.metric == "both" else [args.metric]

    ref = REF if REF in runs else (REF_FALLBACK if REF_FALLBACK in runs else None)

    summary, lines = {}, []
    lines.append("# v24 ablation study\n")
    lines.append(f"Benchmark: 291-row structure test set of `data/processed/splits/train_v22/split.json`, "
                 f"gene-balanced, sequence-only inference. Reference = `{ref}`.\n")
    if leaky:
        lines.append(f"**Leak-free view**: the {len(leaky)} test genes present in DeepPBS "
                     f"fold 0 are dropped from every model "
                     f"({', '.join(sorted(leaky))}), leaving 226 rows / 35 genes.\n")
    if ref is None:
        lines.append("> **No reference run scored yet** — deltas are omitted.\n")
    elif ref == REF_FALLBACK:
        lines.append(f"> **Interim reference**: `{REF}` (same-harness single-GPU re-run of the v24 "
                     f"recipe) has not finished, so deltas are paired against the shipped 6-GPU "
                     f"seed-42 checkpoint. Re-run this script once `{REF}` lands.\n")

    for metric in metrics:
        name = "Panel A — gene_content_r (GT length, oracle registration)" if metric == "A_content_r" \
               else "Panel B — gene_covR (own predicted gate span)"
        lines.append(f"\n## {name}\n")
        lines.append("| variant | what changed | seeds | mean ± sd | per-seed | paired Δ vs full v24 (95% CI) | genes won |")
        lines.append("|---|---|---|---|---|---|---|")
        for tag, desc in VARIANTS:
            if tag not in runs:
                lines.append(f"| `{tag}` | {desc} | – | *not run* | | | |")
                continue
            seeds = sorted(runs[tag])
            vals = {s: gene_mean(runs[tag][s], metric) for s in seeds}
            mu, sd = float(np.mean(list(vals.values()))), float(np.std(list(vals.values()), ddof=0))
            per_seed = " ".join(f"{s}:{vals[s]:.3f}" for s in seeds)

            dtxt, wtxt = "", ""
            if ref is not None and tag != ref:
                # seed-matched where possible; a variant seed with no matching reference
                # seed falls back to reference seed 42 (noted in the table footnote)
                shared = [s for s in seeds if s in runs[ref]] or \
                         ([42] if 42 in runs[ref] and 42 in seeds else [])
                if shared:
                    ds = [paired_delta(runs[tag][s], runs[ref][s], metric, args.boot, rng)
                          for s in shared]
                    ds = [d for d in ds if d]
                    if ds:
                        dd = float(np.mean([d["delta"] for d in ds]))
                        lo = float(np.mean([d["ci_lo"] for d in ds]))
                        hi = float(np.mean([d["ci_hi"] for d in ds]))
                        star = "**" if (lo > 0 or hi < 0) else ""
                        dtxt = f"{star}{dd:+.3f}{star} [{lo:+.3f}, {hi:+.3f}]"
                        wtxt = f"{int(np.mean([d['wins'] for d in ds]))}/{ds[0]['n_genes']}"
                        summary.setdefault(tag, {})[f"{metric}_paired"] = {
                            "delta": dd, "ci_lo": lo, "ci_hi": hi,
                            "seeds_matched": shared, "significant": bool(lo > 0 or hi < 0)}
                else:
                    dtxt = "*(no seed-matched reference)*"
            lines.append(f"| `{tag}` | {desc} | {len(seeds)} | {mu:.3f} ± {sd:.3f} | {per_seed} | "
                         f"{dtxt} | {wtxt} |")
            summary.setdefault(tag, {})[metric] = {
                "mean": mu, "sd": sd, "per_seed": vals, "n_seeds": len(seeds)}

    lines.append(
        "\n**Reading the table.** `mean ± sd` is across training seeds and carries the full "
        "seed noise of this benchmark (v24's own five-seed sd is 0.055 on Panel A), so it is "
        "the conservative view. The paired Δ removes shared per-gene difficulty and is the "
        "estimator to quote; **bold** = 95% CI excludes zero. `A5_shufcontact` is the control "
        "for the contact block: if A2/A3/A4 each cost something but A5 does not, the contact "
        "terms are acting as a regulariser rather than as structural knowledge.\n")

    out_md = OUT_MD.replace(".md", "_leakfree.md") if leaky else OUT_MD
    out_json = OUT_JSON.replace(".json", "_leakfree.json") if leaky else OUT_JSON
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w").write("\n".join(lines) + "\n")
    json.dump(summary, open(out_json, "w"), indent=1)
    print("\n".join(lines))
    print(f"\n-> {out_md}\n-> {out_json}")


if __name__ == "__main__":
    main()
