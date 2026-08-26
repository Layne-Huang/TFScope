#!/usr/bin/env python
"""Phase A: per-family scores for the 5-seed v24 ENSEMBLE, with every row labelled by
how contaminated it is for this model.

The canonical `train_v22` split cannot answer "how good is v24 on family F": under the
corrected taxonomy 24 of 43 families have zero test rows, including the two largest.
So instead of pretending one number exists, every row of the v23 table is scored and
tagged with its provenance:

  test      held out from training AND from checkpoint selection          -> CLEAN
  excluded  never in any split; gene-disjoint from train and val          -> CLEAN
  val       gene-disjoint from train, but ckpt_best was SELECTED on its
            oracle-r (train.py:1337)                                      -> SELECTION-CONTAMINATED
  train     fitted                                                        -> LEAKED (ceiling only)

Metric is the unified evaluator (iclr/unified_eval), identical to the protocol behind
the published v24 / DeepPBS numbers: Panel A = oracle shift+RC registered content r
over the IC-trimmed GT core; Panel B = covR (r x coverage) at the model's own predicted
gate span. Aggregation is gene-balanced within each (family, surface) cell.

  python scripts/lofo/eval_v24_ens_per_family.py --surfaces test excluded val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from iclr.baselines import _decode_pwm                      # noqa: E402
from iclr.score_v24_ensemble import predict_ensemble        # noqa: E402
from iclr.unified_eval import panel_A, panel_B, trimmed_core  # noqa: E402
from reclassify_tf_families import classify                 # noqa: E402

DATA = "data/processed/tf_pwm_training_v23.parquet"
CANON = "data/processed/splits/train_v22/split.json"
# Absolute, because the two read-only checkpoint roots live under DIFFERENT /data1 trees
# and the repo's `checkpoints` symlink points at only one of them (and, inside the LOFO
# mirror, at the writable LOFO tree instead).
CKPTS = ["/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt"] + \
        [f"/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/seed{s}/ckpt_best.pt"
         for s in (1, 7, 13, 23)]
OUT = "results/family_lofo/v24_ens_per_family.json"

CONTAMINATION = {"test": "clean", "excluded": "clean",
                 "val": "selection_contaminated", "train": "leaked"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_taxonomy(df):
    df = df.copy()
    df["g"] = df.gene_symbol.fillna("").astype(str).str.upper()
    seed = {r.g: r.family_name for r in
            df[df.family_name.isin(["C2H2_short", "C2H2_medium", "C2H2_long"])].itertuples()}
    fam = df.g.map(lambda x: classify(x, seed))
    df["fam_lofo"] = fam.map(lambda f: "C2H2" if str(f).startswith("C2H2") else f)
    return df


def gene_balanced(rows, key, n_boot=2000, seed=0):
    """Mean over per-gene means, with a gene-level bootstrap CI."""
    by_gene = {}
    for r in rows:
        by_gene.setdefault(r["gene"], []).append(r[key])
    gm = np.array([np.nanmean(v) for v in by_gene.values()], float)
    if gm.size == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    rng = np.random.RandomState(seed)
    boot = [np.nanmean(rng.choice(gm, gm.size, replace=True)) for _ in range(n_boot)] \
        if gm.size > 1 else [gm[0]]
    return (float(np.nanmean(gm)),
            (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            int(gm.size))


def score_rows(ckpts, data_path, filenames, df, device):
    """Run the ensemble over an arbitrary row set and return per-row scores."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"test": sorted(filenames)}, tmp); tmp.close()
    try:
        preds, gate_lens = predict_ensemble(ckpts, data_path, tmp.name, device)
    finally:
        os.unlink(tmp.name)

    sub = df[df.filename.isin(preds)].set_index("filename")
    out, skipped = [], 0
    for fn, pred in preds.items():
        r = sub.loc[fn]
        core = trimmed_core(_decode_pwm(r.pwm))
        if core is None:
            skipped += 1
            continue
        A = panel_A(pred, core)
        B = panel_B(pred, core, gate_lens.get(fn))
        out.append({"filename": fn, "gene": r.g, "family": r.fam_lofo,
                    "content_r": A["content_r"], "covR": B["covR"],
                    "coverage": B["coverage"], "len_bias": B["len_bias"]})
    return out, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surfaces", nargs="+", default=["test", "excluded", "val"],
                    choices=["test", "excluded", "val", "train"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--min-genes", type=int, default=2,
                    help="suppress per-family cells backed by fewer genes than this")
    args = ap.parse_args()

    missing = [c for c in CKPTS if not os.path.exists(c)]
    if missing:
        sys.exit(f"ABORT: missing ensemble checkpoints:\n  " + "\n  ".join(missing))

    df = add_taxonomy(pd.read_parquet(DATA))
    df["filename"] = df.filename.astype(str)
    canon = json.load(open(CANON))
    log(f"ensemble: {len(CKPTS)} members | surfaces: {args.surfaces}")

    per_surface, all_rows = {}, []
    for surf in args.surfaces:
        fns = set(canon[surf]) & set(df.filename)
        log(f"--- {surf}: {len(fns)} rows ({CONTAMINATION[surf]}) ---")
        t0 = time.time()
        rows, skipped = score_rows(CKPTS, DATA, fns, df, args.device)
        for r in rows:
            r["surface"] = surf
            r["contamination"] = CONTAMINATION[surf]
        all_rows += rows
        log(f"    scored {len(rows)} rows (skipped {skipped}) in {time.time()-t0:.0f}s")

        m, ci, ng = gene_balanced(rows, "content_r")
        mb, cib, _ = gene_balanced(rows, "covR")
        per_surface[surf] = {
            "contamination": CONTAMINATION[surf], "n_rows": len(rows), "n_genes": ng,
            "gene_content_r": round(m, 4), "content_r_ci95": [round(c, 4) for c in ci],
            "gene_covR": round(mb, 4), "covR_ci95": [round(c, 4) for c in cib],
        }
        log(f"    {surf}: content_r={m:.4f} [{ci[0]:.4f},{ci[1]:.4f}]  covR={mb:.4f}  "
            f"({ng} genes)")

    # ── per family x contamination class ──────────────────────────────────────
    clean = [r for r in all_rows if r["contamination"] == "clean"]
    cells = {}
    for fam in sorted({r["family"] for r in all_rows}):
        entry = {}
        for label, subset in [("clean", clean),
                              ("selection_contaminated",
                               [r for r in all_rows if r["contamination"] == "selection_contaminated"]),
                              ("leaked", [r for r in all_rows if r["contamination"] == "leaked"])]:
            rows = [r for r in subset if r["family"] == fam]
            if not rows:
                continue
            m, ci, ng = gene_balanced(rows, "content_r")
            mb, _, _ = gene_balanced(rows, "covR")
            if ng < args.min_genes:
                entry[label] = {"n_rows": len(rows), "n_genes": ng,
                                "suppressed": f"fewer than {args.min_genes} genes"}
                continue
            entry[label] = {"n_rows": len(rows), "n_genes": ng,
                            "content_r": round(m, 4),
                            "content_r_ci95": [round(c, 4) for c in ci],
                            "covR": round(mb, 4)}
        if entry:
            cells[fam] = entry

    res = {"model": "v24_ens5", "members": CKPTS, "data": DATA,
           "taxonomy": "reclassify_tf_families.py, C2H2_* merged",
           "metric": "iclr/unified_eval; gene-balanced; CI = gene-level bootstrap x2000",
           "surfaces": per_surface, "per_family": cells}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    pd.DataFrame(all_rows).round(4).to_csv(args.out.replace(".json", "_rows.csv"), index=False)

    print("\n=== v24-ensemble per family, CLEAN surface (test + excluded) ===")
    print(f"{'family':<20} {'genes':>6} {'rows':>6} {'content_r':>10} {'95% CI':>18} {'covR':>7}")
    print("-" * 74)
    for fam, e in sorted(cells.items(), key=lambda kv: -(kv[1].get("clean", {}).get("content_r") or -9)):
        c = e.get("clean")
        if not c or "content_r" not in c:
            continue
        ci = c["content_r_ci95"]
        print(f"{fam:<20} {c['n_genes']:>6} {c['n_rows']:>6} {c['content_r']:>10.4f} "
              f"{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>18} {c['covR']:>7.4f}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
