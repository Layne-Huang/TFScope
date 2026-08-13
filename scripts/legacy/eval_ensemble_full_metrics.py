#!/usr/bin/env python
"""Full 11-metric panel for 5-fold ensemble (last checkpoint) vs DeepPBS.

Alignment: oracle offset + RC on trimmed informative core (IC >= 0.25 bits).
Metrics per TF: mean Pearson r, median r, IC-weighted r, MAE, RMSE,
                cross-entropy, KL, top-1 accuracy, AUC, F1, MCC.
Also reports canonical-fixed r (deployable metric, no alignment freedom).

Usage:
  python scripts/eval_ensemble_full_metrics.py --mode fulldata_v18a
  python scripts/eval_ensemble_full_metrics.py --mode both_v18a
  python scripts/eval_ensemble_full_metrics.py --mode fulldata_v18a --use-best
"""
import os, sys, json, glob, argparse
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")

from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from canonicalize_pwms import canonicalize
from torch.utils.data import DataLoader

SPLIT    = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
DATA_RAW = "data/processed/tf_pwm_deeppbs_only.parquet"
IC_THRESH = 0.25
MAX_SHIFT = 10
N_FOLDS   = 5
device    = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = {
    "deeppbs_v18a": {
        "ckpt_root": "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_5fold_v18a",
        "data":      "data/processed/tf_pwm_deeppbs_only_canon.parquet",
        "label":     "TFScope-DeepPBS-v18a",
        "out_dir":   "results/ensemble_deeppbs_v18a",
    },
    "fulldata_v18a": {
        "ckpt_root": "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_5fold_v18a",
        "data":      "data/processed/tf_pwm_aug_dbd_canon.parquet",
        "label":     "TFScope-FullData-v18a",
        "out_dir":   "results/ensemble_fulldata_v18a",
    },
}


# ── checkpoint loading ────────────────────────────────────────────────────────

def last_epoch_ckpt(fold_dir):
    files = sorted(glob.glob(os.path.join(fold_dir, "ckpt_epoch*.pt")))
    return files[-1] if files else None


def infer_fold(fold, ckpt_root, data, use_best=False):
    fold_dir = os.path.join(ckpt_root, f"fold_{fold}")
    if use_best:
        ckpt = os.path.join(fold_dir, "ckpt_best.pt")
    else:
        ckpt = last_epoch_ckpt(fold_dir)
        if ckpt:
            print(f"  fold {fold}: {os.path.basename(ckpt)}")
    if not ckpt or not os.path.exists(ckpt):
        print(f"  [skip] fold {fold}"); return None

    cfg = TFScopeConfig()
    cfg_path = os.path.join(fold_dir, "config.json")
    if os.path.exists(cfg_path):
        for k, v in json.load(open(cfg_path)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass

    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False)
    m.eval()

    ds = TFDataset(cfg, data, SPLIT, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                    collate_fn=collate_variable_length)
    P, T, M, G = [], [], [], []
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            gate, pw, _ = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                            retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None,
                            recog_prior=b.get("recog_prior"))
            P.append(F.softmax(pw, 1).cpu().numpy())
            T.append(b["target_pwm"].cpu().numpy())
            M.append(b["pwm_mask"].cpu().numpy())
            G.append((gate.sigmoid() > 0.5).cpu().numpy())

    P = np.concatenate(P); T = np.concatenate(T)
    M = np.concatenate(M); G = np.concatenate(G)
    # Use predicted gate mask — honest deployable setting.
    return {fn: P[i][:, G[i].astype(bool)] for i, fn in enumerate(ds.filenames)}, \
           {fn: (T[i], M[i]) for i, fn in enumerate(ds.filenames)}


def build_ensemble(fold_preds, fns):
    available = sorted(fold_preds.keys())
    out = {}
    for fn in fns:
        arrays = [fold_preds[f][fn] for f in available if fn in fold_preds.get(f, {})]
        if not arrays: continue
        max_L = max(a.shape[1] for a in arrays)
        padded = [np.pad(a, ((0,0),(0,max_L-a.shape[1])), constant_values=0.25) for a in arrays]
        avg = np.mean(padded, axis=0)
        out[fn] = avg / avg.sum(0, keepdims=True).clip(1e-8)
    return out


# ── targets / DeepPBS ─────────────────────────────────────────────────────────

def trimmed_core(T_i, M_i):
    idx = M_i.astype(bool)
    if not idx.any(): return None
    t = np.clip(T_i[:, idx], 1e-8, 1.0)
    ic = 2.0 + (t * np.log2(t)).sum(0)
    inf = np.where(ic >= IC_THRESH)[0]
    if len(inf) == 0: return None
    c = t[:, inf[0]:inf[-1]+1]
    return c / c.sum(0, keepdims=True)


def deeppbs_preds(fns):
    df = pd.read_parquet(DATA_RAW)
    fn2gene = dict(zip(df["filename"], df["gene_symbol"]))
    dpbs = np.load("results/deeppbs_blind_benchmark/gene_preds.npz", allow_pickle=True)
    ci = {k.upper(): k for k in dpbs.files}
    out = {}
    for fn in fns:
        gk = ci.get(str(fn2gene.get(fn, "")).upper())
        if gk is not None:
            out[fn] = np.clip(np.array(dpbs[gk]).T, 1e-8, 1.0)
    return out


# ── metrics ───────────────────────────────────────────────────────────────────

def aligned_cols(pred, core):
    aligned, shift, orient, score = align_pwm(pred, core, max_shift=MAX_SHIFT, consider_revcomp=True)
    o = revcomp_pwm_np(pred) if orient == "rc" else pred
    cols = np.array(sorted([i+shift for i in range(o.shape[1]) if 0 <= i+shift < core.shape[1]]))
    return aligned, cols, score


def compute_panel(core, aligned, cols):
    if len(cols) < 2: return None
    t = core[:, cols]
    p = np.clip(aligned[:, cols], 1e-8, 1.0)
    p = p / p.sum(0, keepdims=True)
    d = {}
    d["r"]    = np.nanmean([pearsonr(t[:,j], p[:,j])[0] for j in range(t.shape[1])])
    d["mae"]  = np.abs(p - t).mean()
    d["rmse"] = np.sqrt(((p - t)**2).mean())
    d["ce"]   = -(t * np.log(p)).sum(0).mean()
    d["kl"]   = (t * (np.log(t) - np.log(p))).sum(0).mean()
    ic = 2.0 + (t * np.log2(t)).sum(0); w = ic / (ic.sum() + 1e-8)
    mp = (w * p).sum(); mt = (w * t).sum()
    cov = (w * (p - mp) * (t - mt)).sum()
    sp = np.sqrt((w*(p-mp)**2).sum()); st = np.sqrt((w*(t-mt)**2).sum())
    d["icr"]  = cov / (sp * st + 1e-8)
    pc, tc = p.argmax(0), t.argmax(0)
    d["top1"] = (pc == tc).mean()
    aucs = []
    for b in range(4):
        isb = (tc == b).astype(int)
        if isb.sum() >= 1 and (1-isb).sum() >= 1:
            try: aucs.append(roc_auc_score(isb, p[b]))
            except: pass
    d["auc"] = np.mean(aucs) if aucs else np.nan
    d["f1"]  = f1_score(tc, pc, average="macro", zero_division=0)
    try: d["mcc"] = matthews_corrcoef(tc, pc)
    except: d["mcc"] = np.nan
    return d


def canon_fixed_r(core, pred):
    cp = canonicalize(np.clip(pred, 1e-8, 1.0).astype(np.float32))
    cp = cp / cp.sum(0, keepdims=True).clip(1e-8)
    Lt = core.shape[1]; pp = np.full((4, Lt), 0.25, np.float32)
    L = min(cp.shape[1], Lt); pp[:, :L] = cp[:, :L]
    rs = [0.0 if (core[:,j].std()<1e-8 or pp[:,j].std()<1e-8)
          else np.corrcoef(core[:,j], pp[:,j])[0,1] for j in range(Lt)]
    return float(np.mean(rs))


def score_all(preds_dict, cores, shared):
    rows, r_all, cfix_all = [], [], []
    for fn in shared:
        pv = preds_dict.get(fn)
        if pv is None or pv.shape[1] == 0: continue
        core = cores[fn]
        aligned, cols, _ = aligned_cols(pv, core)
        d = compute_panel(core, aligned, cols)
        if d:
            rows.append(d); r_all.append(d["r"])
            cfix_all.append(canon_fixed_r(core, pv))
    agg = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
    agg["med"] = np.nanmedian(r_all)
    agg["cfix"] = np.nanmean(cfix_all)
    return agg


# ── main ──────────────────────────────────────────────────────────────────────

def eval_one(cfg_key, cores, shared, dpp, use_best=False):
    cfg = CONFIGS[cfg_key]
    tag = "[best]" if use_best else "[last]"
    print(f"\nLoading {cfg['label']} {tag} ...", flush=True)

    fold_preds = {}
    fns = list(cores.keys())
    for fold in range(N_FOLDS):
        result = infer_fold(fold, cfg["ckpt_root"], cfg["data"], use_best=use_best)
        if result is None: continue
        p, _ = result
        fold_preds[fold] = p

    if not fold_preds: print("  No checkpoints."); return None
    ensemble = build_ensemble(fold_preds, fns)
    print(f"  {len(fold_preds)}/{N_FOLDS} folds, {len(ensemble)} TFs in ensemble")
    return score_all(ensemble, cores, shared)


def print_table(results, names):
    METRICS = [
        ("Mean Pearson r",           "r",    False),
        ("Median Pearson r",         "med",  False),
        ("IC-weighted r",            "icr",  False),
        ("MAE",                      "mae",  True),
        ("RMSE",                     "rmse", True),
        ("Cross-entropy",            "ce",   True),
        ("KL divergence",            "kl",   True),
        ("Top-1 accuracy",           "top1", False),
        ("AUC (macro OvR)",          "auc",  False),
        ("F1 (macro)",               "f1",   False),
        ("MCC",                      "mcc",  False),
        ("Canonical-fixed r",        "cfix", False),
    ]
    col_w = 14
    hdr = f"{'Metric':<28}" + "".join(f"{n:>{col_w}}" for n in names) + "   best"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    wins = {n: 0 for n in names}
    for label, k, lower in METRICS:
        vals = {n: results[n].get(k, float("nan")) for n in names}
        best = min(vals, key=vals.get) if lower else max(vals, key=vals.get)
        wins[best] += 1
        print(f"{label:<28}" + "".join(f"{vals[n]:>{col_w}.4f}" for n in names) + f"   {best}")
    print("-" * len(hdr))
    print(f"{'Wins':<28}" + "".join(f"{wins[n]:>{col_w}}" for n in names))
    print("=" * len(hdr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["deeppbs_v18a", "fulldata_v18a", "both_v18a"],
                    default="fulldata_v18a")
    ap.add_argument("--use-best", action="store_true",
                    help="Use best-val checkpoint (default: last epoch checkpoint)")
    args = ap.parse_args()

    modes = ["deeppbs_v18a", "fulldata_v18a"] if args.mode == "both_v18a" else [args.mode]

    # Load targets from fulldata parquet (covers all test TFs)
    print("Loading test targets ...")
    tmp_cfg = TFScopeConfig()
    ds_ref = TFDataset(tmp_cfg, CONFIGS["fulldata_v18a"]["data"],
                       SPLIT, split="test", max_seq_len=1024)
    ld_ref = DataLoader(ds_ref, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_variable_length)
    T_all, M_all = [], []
    for b in ld_ref:
        T_all.append(b["target_pwm"].numpy()); M_all.append(b["pwm_mask"].numpy())
    T_all = np.concatenate(T_all); M_all = np.concatenate(M_all)
    fns_ref = ds_ref.filenames

    cores = {fn: trimmed_core(T_all[i], M_all[i]) for i, fn in enumerate(fns_ref)}
    cores = {fn: c for fn, c in cores.items() if c is not None}

    dpp = deeppbs_preds(fns_ref)
    shared = [fn for fn in cores if fn in dpp]
    print(f"Shared TFs with DeepPBS: {len(shared)}")

    results = {"DeepPBS": score_all(dpp, cores, shared)}
    for mode in modes:
        r = eval_one(mode, cores, shared, dpp, use_best=args.use_best)
        if r: results[CONFIGS[mode]["label"]] = r

    names = [CONFIGS[m]["label"] for m in modes if CONFIGS[m]["label"] in results] + ["DeepPBS"]
    print_table(results, names)

    os.makedirs("results/full_metrics", exist_ok=True)
    tag = "best" if args.use_best else "last"
    fname = f"results/full_metrics/panel_ensemble_{args.mode}_{tag}.json"
    json.dump({n: {k: float(v) for k, v in results[n].items()} for n in results},
              open(fname, "w"), indent=2)
    print(f"\nSaved {fname}")


if __name__ == "__main__":
    main()
