#!/usr/bin/env python
"""Leave-family-out evaluation.

For each completed LFO family, loads ckpt_best.pt, runs inference on the
held-out test split, and reports full oracle-r metrics + per-family breakdown.
Compares to in-distribution cluster40 per-family oracle-r numbers.

Usage:
  python scripts/eval_lofo.py                        # all completed families
  python scripts/eval_lofo.py --families C2H2_long Homeodomain
  python scripts/eval_lofo.py --full-metrics         # also print 11-metric panel
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from torch.utils.data import DataLoader

CKPT_ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/lofo_v18a"
SPLIT_ROOT = "data/processed/splits/lofo_v2"
DATA       = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
device     = "cuda" if torch.cuda.is_available() else "cpu"

# c40 in-distribution per-family oracle-r (from cluster40 honest benchmark)
C40_FAMILY_R = {
    "bZIP":            0.72,
    "Homeodomain":     0.68,
    "Nuclear_Receptor":0.66,
    "Forkhead":        0.60,
    "bHLH":            0.50,
    "C2H2_medium":     0.50,
    "Other":           0.49,
    "C2H2_long":       0.43,
    "ETS":             0.40,
    "C2H2_short":      0.39,
}

ALL_FAMILIES = list(C40_FAMILY_R.keys())


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


def infer_family(family):
    ckpt = os.path.join(CKPT_ROOT, family, "ckpt_best.pt")
    split = os.path.join(SPLIT_ROOT, f"{family}.json")
    if not os.path.exists(ckpt):
        print(f"[skip] {family}: checkpoint not found"); return None
    if not os.path.exists(split):
        print(f"[skip] {family}: split not found"); return None

    cfg = TFScopeConfig()
    cfg_path = os.path.join(CKPT_ROOT, family, "config.json")
    if os.path.exists(cfg_path):
        for k, v in json.load(open(cfg_path)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass

    m = TFScopeModel(cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd["model"], strict=False)
    m.eval()

    ds = TFDataset(cfg, DATA, split, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                    collate_fn=collate_variable_length)

    P, T, M, G, fns = [], [], [], [], []
    with torch.no_grad():
        for b in ld:
            b = {k: (v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                     if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            gate_logits, pw, aux = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                                     retrieved_pwms=b.get("retrieved_pwms"),
                                     retrieved_masks=b.get("retrieved_masks"),
                                     retrieved_sims=b.get("retrieved_sims"),
                                     recog_prior=b.get("recog_prior"))
            P.append(F.softmax(pw, 1).cpu().numpy())
            T.append(b["target_pwm"].cpu().numpy())
            M.append(b["pwm_mask"].cpu().numpy())
            G.append(torch.sigmoid(gate_logits).cpu().numpy())
    P = np.concatenate(P); T = np.concatenate(T)
    M = np.concatenate(M); G = np.concatenate(G)
    fns = ds.filenames
    return P, T, M, G, fns


def panel_per_tf(P, T, M, G, fns, ic_thresh=0.25, max_shift=10):
    """Returns list of per-TF result dicts."""
    results = []
    skipped = 0
    for i, fn in enumerate(fns):
        # gate-selected active columns
        gate_mask = G[i] > 0.5
        if not gate_mask.any():
            gate_mask = M[i].astype(bool)
        pred = P[i][:, gate_mask]

        # trimmed core from target
        core = trimmed_core(T[i], M[i], ic_thresh)
        if core is None or core.shape[1] < 2 or pred.shape[1] < 1:
            skipped += 1; continue

        aligned, shift, orient, oracle_r = align_pwm(pred, core, max_shift=max_shift,
                                                      consider_revcomp=True)
        o = revcomp_pwm_np(pred) if orient == "rc" else pred
        cols = np.array([j + shift for j in range(o.shape[1])
                         if 0 <= j + shift < core.shape[1]])
        if len(cols) < 2:
            skipped += 1; continue

        ct = core[:, cols]
        cp = np.clip(aligned[:, cols], 1e-8, 1.0)
        cp /= cp.sum(0, keepdims=True)

        d = {"fn": fn, "oracle_r": oracle_r}
        d["r"]    = np.nanmean([pearsonr(ct[:, j], cp[:, j])[0] for j in range(ct.shape[1])])
        d["mae"]  = float(np.abs(cp - ct).mean())
        d["rmse"] = float(np.sqrt(((cp - ct) ** 2).mean()))
        d["ce"]   = float(-(ct * np.log(cp)).sum(0).mean())
        d["kl"]   = float((ct * (np.log(ct) - np.log(cp))).sum(0).mean())
        # IC-weighted Pearson r: per-column r weighted by column IC
        per_col_r = np.array([pearsonr(ct[:, j], cp[:, j])[0] for j in range(ct.shape[1])])
        ic_col = 2.0 + (ct * np.log2(ct)).sum(0)
        ic_w = np.clip(ic_col, 0, None); ic_w /= (ic_w.sum() + 1e-8)
        d["icr"]  = float(np.dot(ic_w, per_col_r))
        pc, tc = cp.argmax(0), ct.argmax(0)
        d["top1"] = float((pc == tc).mean())
        a = []
        for b in range(4):
            isb = (tc == b).astype(int)
            if isb.sum() >= 1 and (1 - isb).sum() >= 1:
                try: a.append(roc_auc_score(isb, cp[b]))
                except: pass
        d["auc"]  = float(np.mean(a)) if a else np.nan
        d["f1"]   = float(f1_score(tc, pc, average="macro", zero_division=0))
        try:    d["mcc"] = float(matthews_corrcoef(tc, pc))
        except: d["mcc"] = np.nan
        results.append(d)

    return results, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--ic-thresh", type=float, default=0.25)
    ap.add_argument("--max-shift", type=int, default=10)
    ap.add_argument("--full-metrics", action="store_true")
    args = ap.parse_args()

    if args.families:
        families = args.families
    else:
        # auto-detect completed families (ckpt_best.pt exists)
        families = [f for f in ALL_FAMILIES
                    if os.path.exists(os.path.join(CKPT_ROOT, f, "ckpt_best.pt"))]
        print(f"Auto-detected {len(families)} completed families: {families}")

    all_results = {}
    for fam in families:
        print(f"\n--- {fam} ---", flush=True)
        out = infer_family(fam)
        if out is None: continue
        P, T, M, G, fns = out
        rows, sk = panel_per_tf(P, T, M, G, fns, args.ic_thresh, args.max_shift)
        all_results[fam] = rows
        rs = [r["oracle_r"] for r in rows]
        print(f"  n_test={len(rows)} (skipped={sk})  "
              f"mean_r={np.mean(rs):.4f}  median_r={np.median(rs):.4f}")

    # summary table
    print("\n" + "=" * 75)
    print(f"{'Family':<20} {'n':>5} {'LFO mean r':>12} {'LFO med r':>11} "
          f"{'c40 mean r':>12} {'delta':>8}")
    print("-" * 75)
    all_lfo_r = []
    for fam in families:
        if fam not in all_results: continue
        rows = all_results[fam]
        rs = [r["oracle_r"] for r in rows]
        lfo_mean = np.mean(rs); lfo_med = np.median(rs)
        c40 = C40_FAMILY_R.get(fam, np.nan)
        delta = lfo_mean - c40
        all_lfo_r.extend(rs)
        print(f"{fam:<20} {len(rows):>5} {lfo_mean:>12.4f} {lfo_med:>11.4f} "
              f"{c40:>12.4f} {delta:>+8.4f}")
    if all_lfo_r:
        print("-" * 75)
        print(f"{'ALL (macro avg)':<20} {len(all_lfo_r):>5} "
              f"{np.mean(all_lfo_r):>12.4f} {np.median(all_lfo_r):>11.4f}")
    print("=" * 75)

    if args.full_metrics and all_results:
        keys = [("oracle_r","oracle-r"),("r","col Pearson r"),("icr","IC-weighted r"),
                ("mae","MAE"),("rmse","RMSE"),("ce","Cross-entropy"),("kl","KL divergence"),
                ("top1","Top-1 acc"),("auc","AUC"),("f1","F1"),("mcc","MCC")]
        print(f"\n=== Full metric panel (macro avg over all evaluated TFs) ===")
        all_rows = [r for rows in all_results.values() for r in rows]
        for k, label in keys:
            vals = [r[k] for r in all_rows if not np.isnan(r.get(k, np.nan))]
            print(f"  {label:<22}  {np.mean(vals):.4f}")

    # save
    os.makedirs("results/lofo", exist_ok=True)
    flat = {fam: [{"fn": r["fn"], "oracle_r": r["oracle_r"]} for r in rows]
            for fam, rows in all_results.items()}
    json.dump(flat, open("results/lofo/per_tf_oracle_r.json", "w"), indent=2)
    print("\nSaved results/lofo/per_tf_oracle_r.json")


if __name__ == "__main__":
    main()
