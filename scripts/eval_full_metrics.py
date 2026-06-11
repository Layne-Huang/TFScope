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


def aligned_cols(pred, core, max_shift=10):
    """Align pred to core; return (aligned core-frame pred, covered col indices)."""
    aligned, shift, orient, score = align_pwm(pred, core, max_shift=max_shift, consider_revcomp=True)
    o = revcomp_pwm_np(pred) if orient == "rc" else pred
    cols = [i + shift for i in range(o.shape[1]) if 0 <= i + shift < core.shape[1]]
    return aligned, np.array(sorted(cols)), score


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v17", "v18a", "deeppbs"])
    ap.add_argument("--ic-thresh", type=float, default=0.25)
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
    agg = {n: {} for n in names}; rmed = {n: [] for n in names}; cfix = {n: [] for n in names}
    for n in names:
        rows = []
        for fn in shared:
            pv = get_pred(n, fn)
            if pv is None or pv.shape[1] == 0: continue
            aligned, cols, _ = aligned_cols(pv, cores[fn])
            d = panel(cores[fn], aligned, cols)
            if d: rows.append(d); rmed[n].append(d["r"])
            cfix[n].append(canon_fixed_r(cores[fn], pv))
        for k in rows[0]:
            agg[n][k] = np.nanmean([r[k] for r in rows])

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

    os.makedirs("results/full_metrics", exist_ok=True)
    json.dump({n: {k: float(v) for k, v in agg[n].items()} for n in names},
              open("results/full_metrics/panel.json", "w"), indent=2)
    print("\nSaved results/full_metrics/panel.json")


if __name__ == "__main__":
    main()
