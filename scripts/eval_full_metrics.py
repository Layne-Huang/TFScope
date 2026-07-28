#!/usr/bin/env python
"""Full metric panel for v17 / v18a / DeepPBS on the trimmed informative core.

Every model's prediction is aligned (offset + reverse-complement) to each test
TF's trimmed core, then the complete panel is computed identically over the
aligned overlap columns:
  mean Pearson r, median r, IC-weighted r, MAE, RMSE, cross-entropy, KL,
  top-1 base accuracy, AUC (macro OvR), F1 (macro), MCC.

This is the fair motif-level comparison (registration freedom granted to all).
Also prints the deployable canonical-fixed mean/median r for reference.
"""
import os, sys, json, argparse
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from eval_canonical_registration import infer, deeppbs_preds, CKPTS, DATA, DATA_OF, canonicalize


def ic_bits(p):
    p = np.clip(p, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)


def trimmed_core(T_i, M_i, thresh=0.25):
    idx = M_i.astype(bool)
    if not idx.any(): return None
    t = np.clip(T_i[:, idx], 1e-8, 1.0)
    ic = ic_bits(t); inf = np.where(ic >= thresh)[0]
    if len(inf) == 0: return None
    c = t[:, inf[0]:inf[-1] + 1]
    return c / c.sum(0, keepdims=True)


def aligned_cols(pred, core, max_shift=10, min_overlap=2):
    """Align pred to core; return (aligned core-frame pred, covered col indices)."""
    aligned, shift, orient, score = align_pwm(
        pred,
        core,
        max_shift=max_shift,
        consider_revcomp=True,
        min_overlap=min_overlap,
    )
    if score <= -1.5:
        return aligned, np.array([], dtype=int), score
    o = revcomp_pwm_np(pred) if orient == "rc" else pred
    cols = [i + shift for i in range(o.shape[1]) if 0 <= i + shift < core.shape[1]]
    return aligned, np.array(sorted(cols)), score


def panel_full(core, aligned, cols, pred_ncols=None):
    """Coverage-aware panel: scores ALL ground-truth columns, not just the overlap.

    Why this exists
    ---------------
    `panel()` scores only `cols` (where the aligned prediction overlaps the
    core), so a prediction covering 3 of 14 core columns is scored on those 3
    alone. A perfect 3-column fragment therefore gets r=1.000 / top1=1.00 /
    mae=0.000 -- identical to a perfect full-length prediction -- and a RANDOM
    3-column prediction scores r=0.57 vs 0.27 for a random full-length one,
    because a short prediction also enjoys more alignment freedom per column.
    At eval time the gate chooses this width (train.py: `active = gate > 0.5`),
    so the old panel actively rewards gate collapse.

    Here an uncovered core column is treated as what it actually is: no
    prediction, i.e. a uniform 0.25 column. It contributes zero correlation and
    its MAE against the target, so coverage is paid for rather than forgiven.

    Adds (does not replace) keys; `panel()` is untouched for reproducibility.
    """
    Lg = core.shape[1]
    if Lg == 0:
        return None
    d = {}
    cols = np.asarray(cols, dtype=int)
    cov = len(cols) / Lg
    d["coverage"] = cov
    d["len_gt"] = Lg
    d["len_pred"] = int(pred_ncols) if pred_ncols is not None else len(cols)
    d["len_mae"] = abs(d["len_pred"] - Lg)

    # per-column r: covered columns get their true r, uncovered contribute 0
    rs = np.zeros(Lg, dtype=float)
    if len(cols):
        t = core[:, cols]
        p = np.clip(aligned[:, cols], 1e-8, 1.0)
        p = p / p.sum(0, keepdims=True)
        for k, j in enumerate(cols):
            if t[:, k].std() == 0 or p[:, k].std() == 0:
                rs[j] = 0.0
            else:
                rs[j] = pearsonr(t[:, k], p[:, k])[0]
    d["r_full"] = float(np.nanmean(rs))

    # MAE / top-1 with uncovered columns filled by a uniform prediction
    full = np.full_like(core, 0.25, dtype=float)
    if len(cols):
        pp = np.clip(aligned[:, cols], 1e-8, 1.0)
        full[:, cols] = pp / pp.sum(0, keepdims=True)
    d["mae_full"] = float(np.abs(full - core).mean())
    tc = core.argmax(0)
    hit = (full.argmax(0) == tc).astype(float)
    uncovered = np.ones(Lg, dtype=bool); uncovered[cols] = False
    hit[uncovered] = 0.25          # uniform column -> chance-level, not a free win
    d["top1_full"] = float(hit.mean())

    # coverage-scaled version of the legacy overlap-only r, for comparison
    base = panel(core, aligned, cols)
    d["r_overlap"] = base["r"] if base else float("nan")
    d["r_cov"] = (base["r"] * cov) if base else float("nan")
    return d


def uniform_floor(core):
    """Score of a no-information (uniform) full-length prediction on this core.

    Report alongside every number: with oracle alignment the floor is far above
    zero (uniform-random reaches panel-r ~0.42 on this benchmark), so a raw
    0.65 is not '0.65 of the way to perfect'.
    """
    u = np.full_like(core, 0.25, dtype=float)
    cols = np.arange(core.shape[1])
    return panel_full(core, u, cols, pred_ncols=core.shape[1])


def aligned_cols_fixed(pred, core):
    """Fixed-frame alignment: no shift, no reverse-complement.

    Matches how the model is trained when latent_registration=False, and is the
    honest lower bound next to the oracle-aligned numbers.
    """
    Lg = core.shape[1]
    aligned = np.full((4, Lg), 0.25, dtype=float)
    n = min(pred.shape[1], Lg)
    aligned[:, :n] = pred[:, :n]
    return aligned, np.arange(n), None


def panel(core, aligned, cols):
    """Full metric dict over the covered core columns."""
    if len(cols) < 2: return None
    t = core[:, cols]; p = np.clip(aligned[:, cols], 1e-8, 1.0)
    p = p / p.sum(0, keepdims=True)
    d = {}
    d["r"]   = np.nanmean([pearsonr(t[:, j], p[:, j])[0] for j in range(t.shape[1])])
    d["mae"] = np.abs(p - t).mean()
    d["rmse"] = np.sqrt(((p - t) ** 2).mean())
    d["ce"]  = -(t * np.log(p)).sum(0).mean()
    d["kl"]  = (t * (np.log(t) - np.log(p))).sum(0).mean()
    ic = 2.0 + (t * np.log2(t)).sum(0); w = ic / (ic.sum() + 1e-8)
    mp = (w * p).sum(); mt = (w * t).sum()
    cov = (w * (p - mp) * (t - mt)).sum()
    sp = np.sqrt((w * (p - mp) ** 2).sum()); st = np.sqrt((w * (t - mt) ** 2).sum())
    d["icr"] = cov / (sp * st + 1e-8)
    pc, tc = p.argmax(0), t.argmax(0)
    d["top1"] = (pc == tc).mean()
    a = []
    for b in range(4):
        isb = (tc == b).astype(int)
        if isb.sum() >= 1 and (1 - isb).sum() >= 1:
            try: a.append(roc_auc_score(isb, p[b]))
            except: pass
    d["auc"] = np.mean(a) if a else np.nan
    d["f1"] = f1_score(tc, pc, average="macro", zero_division=0)
    try: d["mcc"] = matthews_corrcoef(tc, pc)
    except: d["mcc"] = np.nan
    return d


def canon_fixed_r(core, pred):
    cp = canonicalize(np.clip(pred, 1e-8, 1.0).astype(np.float32))
    cp = cp / cp.sum(0, keepdims=True).clip(1e-8)
    Lt = core.shape[1]; pp = np.full((4, Lt), 0.25, np.float32)
    L = min(cp.shape[1], Lt); pp[:, :L] = cp[:, :L]
    rs = [0.0 if (core[:, j].std() < 1e-8 or pp[:, j].std() < 1e-8)
          else np.corrcoef(core[:, j], pp[:, j])[0, 1] for j in range(Lt)]
    return float(np.mean(rs))


def grouped_r_cov(rows, metadata, column):
    """Equal-weight groups: mean within group, then expose each group value."""
    if metadata is None or column not in metadata.columns:
        return {}
    lookup = metadata.drop_duplicates("filename").set_index("filename")[column].to_dict()
    grouped = {}
    for row in rows:
        value = lookup.get(row["filename"], "unknown")
        grouped.setdefault(str(value), []).append(row["r_cov"])
    return {
        key: float(np.nanmean(values))
        for key, values in sorted(grouped.items())
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v17", "v18a", "deeppbs"])
    ap.add_argument("--ic-thresh", type=float, default=0.25)
    ap.add_argument("--metadata", default=None,
                    help="optional parquet with filename/gene_symbol/family_name")
    args = ap.parse_args()

    tfs = [m for m in args.models if m != "deeppbs"]
    preds, tgt_ref = {}, None
    for name in tfs:
        if not os.path.exists(CKPTS[name]): print(f"[skip] {name}"); continue
        print(f"infer {name} ...", flush=True)
        p, tg = infer(CKPTS[name], DATA_OF.get(name, DATA))
        preds[name] = p
        if tgt_ref is None: tgt_ref = tg
    fns = list(tgt_ref.keys())
    dpp = deeppbs_preds(fns) if "deeppbs" in args.models else {}
    cores = {fn: trimmed_core(*tgt_ref[fn], args.ic_thresh) for fn in fns}
    cores = {fn: c for fn, c in cores.items() if c is not None}
    shared = [fn for fn in cores if fn in dpp] if "deeppbs" in args.models else list(cores)
    print(f"shared TFs: {len(shared)}")

    def get_pred(name, fn):
        if name == "DeepPBS": return dpp.get(fn)
        return preds[name].get(fn)

    names = [n for n in tfs if n in preds] + (["DeepPBS"] if "deeppbs" in args.models else [])
    metadata = None
    if args.metadata:
        import pandas as pd
        metadata = pd.read_parquet(args.metadata)
    agg = {n: {} for n in names}
    agg_full = {n: {} for n in names}
    per_sample = {n: [] for n in names}
    rmed = {n: [] for n in names}; cfix = {n: [] for n in names}
    for n in names:
        rows, full_rows = [], []
        for fn in shared:
            pv = get_pred(n, fn)
            if pv is None or pv.shape[1] == 0: continue
            aligned, cols, _ = aligned_cols(pv, cores[fn])
            d = panel(cores[fn], aligned, cols)
            if d: rows.append(d); rmed[n].append(d["r"])
            full = panel_full(cores[fn], aligned, cols, pred_ncols=pv.shape[1])
            if full:
                full_rows.append(full)
                per_sample[n].append({"filename": fn, **full})
            cfix[n].append(canon_fixed_r(cores[fn], pv))
        if rows:
            for k in rows[0]:
                agg[n][k] = np.nanmean([r[k] for r in rows])
        if full_rows:
            for k in full_rows[0]:
                agg_full[n][k] = np.nanmean([r[k] for r in full_rows])

    keys = [("Mean Pearson r","r"),("Median Pearson r","med"),("IC-weighted r","icr"),
            ("MAE","mae"),("RMSE","rmse"),("Cross-entropy","ce"),("KL divergence","kl"),
            ("Top-1 accuracy","top1"),("AUC (macro OvR)","auc"),("F1 (macro)","f1"),("MCC","mcc"),
            ("Canonical-fixed r (deployable)","cfix")]
    lower_better = {"mae","rmse","ce","kl"}
    print(f"\n=== Full metric panel — trimmed core (IC>={args.ic_thresh}), offset+RC aligned, "
          f"{len(shared)} TFs ===")
    hdr = f"{'Metric':<32}" + "".join(f"{n:>10}" for n in names) + "   best"
    print(hdr); print("-" * len(hdr))
    for label, k in keys:
        vals = {}
        for n in names:
            if k == "med": vals[n] = np.nanmedian(rmed[n])
            elif k == "cfix": vals[n] = np.nanmean(cfix[n])
            else: vals[n] = agg[n][k]
        best = min(vals, key=vals.get) if k in lower_better else max(vals, key=vals.get)
        print(f"{label:<32}" + "".join(f"{vals[n]:>10.4f}" for n in names) + f"   {best}")

    print("\n=== Coverage-aware full-core metrics ===")
    for n in names:
        f = agg_full[n]
        print(
            f"{n:<16} r_overlap={f.get('r_overlap', float('nan')):.4f}  "
            f"coverage={f.get('coverage', float('nan')):.4f}  "
            f"r_cov={f.get('r_cov', float('nan')):.4f}  "
            f"r_full={f.get('r_full', float('nan')):.4f}  "
            f"len_mae={f.get('len_mae', float('nan')):.3f}"
        )

    os.makedirs("results/full_metrics", exist_ok=True)
    payload = {}
    for n in names:
        gene_groups = grouped_r_cov(per_sample[n], metadata, "gene_symbol")
        payload[n] = {
            "legacy_overlap": {k: float(v) for k, v in agg[n].items()},
            "coverage_aware": {k: float(v) for k, v in agg_full[n].items()},
            "canonical_fixed_r": float(np.nanmean(cfix[n])),
            "gene_balanced_r_cov": (
                float(np.nanmean(list(gene_groups.values())))
                if gene_groups else None
            ),
            "by_gene_r_cov": gene_groups,
            "by_family_r_cov": grouped_r_cov(
                per_sample[n], metadata, "family_name"
            ),
            "per_sample": per_sample[n],
        }
    json.dump(payload, open("results/full_metrics/panel.json", "w"), indent=2)
    print("\nSaved results/full_metrics/panel.json")


if __name__ == "__main__":
    main()
