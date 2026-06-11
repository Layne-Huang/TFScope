#!/usr/bin/env python
"""Evaluate 5-fold noRAG ensemble on the 130 DeepPBS blind test set.

Canonical-registration scoring (deployable, no oracle alignment):
  - Each fold model outputs a PWM for every test TF
  - Predictions are averaged in probability space (mean of softmax)
  - Canonical transform applied to averaged prediction
  - Fixed per-column Pearson r vs canonical target
  - Missing folds are skipped (partial ensemble reported)

Usage:
  python scripts/eval_ensemble.py                    # DeepPBS-exact folds
  python scripts/eval_ensemble.py --mode fulldata    # Full-data folds
  python scripts/eval_ensemble.py --mode both        # Both, side-by-side
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")

from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from canonicalize_pwms import canonicalize
from torch.utils.data import DataLoader

CONFIGS = {
    "deeppbs": {
        "ckpt_root": "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_5fold_norag",
        "data":      "data/processed/tf_pwm_deeppbs_only_canon.parquet",
        "label":     "TFScope-DeepPBS-exact",
        "out_dir":   "results/ensemble_deeppbs",
    },
    "fulldata": {
        "ckpt_root": "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/fulldata_5fold",
        "data":      "data/processed/tf_pwm_aug_dbd_canon.parquet",
        "label":     "TFScope-FullData",
        "out_dir":   "results/ensemble_fulldata",
    },
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

DATA_RAW = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT    = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
N_FOLDS  = 5
device   = "cuda" if torch.cuda.is_available() else "cpu"


def infer_fold(fold: int, ckpt_root: str, data: str):
    ckpt = os.path.join(ckpt_root, f"fold_{fold}", "ckpt_best.pt")
    if not os.path.exists(ckpt):
        print(f"  [skip] fold {fold}: not found"); return None, None

    cfg = TFScopeConfig()
    cfg_path = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.exists(cfg_path):
        for k, v in json.load(open(cfg_path)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass

    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False
    )
    m.eval()

    ds = TFDataset(cfg, data, SPLIT, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                    collate_fn=collate_variable_length)
    P, T, M = [], [], []
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device,
                         dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            _, pw, _ = m(
                b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None,
                recog_prior=b.get("recog_prior"),
            )
            P.append(F.softmax(pw, 1).cpu().numpy())
            T.append(b["target_pwm"].cpu().numpy())
            M.append(b["pwm_mask"].cpu().numpy())

    P = np.concatenate(P)
    T = np.concatenate(T)
    M = np.concatenate(M)
    preds = {fn: P[i][:, M[i].astype(bool)] for i, fn in enumerate(ds.filenames)}
    tgts  = {fn: (T[i], M[i])               for i, fn in enumerate(ds.filenames)}
    print(f"  fold {fold}: {len(preds)} test TFs")
    return preds, tgts


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


def canon_fixed_r(target_core: np.ndarray, pred: np.ndarray) -> float:
    cp = canonicalize(np.clip(pred, 1e-8, 1.0).astype(np.float32))
    cp = cp / cp.sum(0, keepdims=True).clip(1e-8)
    Lt = target_core.shape[1]
    pp = np.full((4, Lt), 0.25, np.float32)
    L = min(cp.shape[1], Lt)
    pp[:, :L] = cp[:, :L]
    rs = []
    for j in range(Lt):
        a, b = target_core[:, j], pp[:, j]
        if a.std() < 1e-8 or b.std() < 1e-8:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.nanmean(rs))


def score_preds(preds: dict, tcore: dict, shared: list) -> np.ndarray:
    return np.array([canon_fixed_r(tcore[fn], preds[fn]) for fn in shared if fn in preds])


def build_ensemble(fold_preds: dict, shared: list) -> dict:
    available = sorted(fold_preds.keys())
    out = {}
    for fn in shared:
        arrays = [fold_preds[f][fn] for f in available if fn in fold_preds[f]]
        if not arrays:
            continue
        max_L = max(a.shape[1] for a in arrays)
        padded = []
        for a in arrays:
            p = np.full((4, max_L), 0.25, dtype=np.float32)
            p[:, :a.shape[1]] = a
            padded.append(p)
        avg = np.mean(padded, axis=0)
        out[fn] = avg / avg.sum(0, keepdims=True).clip(1e-8)
    return out


def eval_one(cfg_key: str, tcore: dict, dpbs_scores: np.ndarray, shared: list) -> dict:
    cfg = CONFIGS[cfg_key]
    print(f"\n─── {cfg['label']} ({cfg_key}) ───")
    fold_preds = {}
    tgts_ref   = None
    for fold in range(N_FOLDS):
        print(f"  Loading fold {fold} ...", flush=True)
        p, t = infer_fold(fold, cfg["ckpt_root"], cfg["data"])
        if p is None:
            continue
        fold_preds[fold] = p
        if tgts_ref is None:
            tgts_ref = t

    if not fold_preds:
        print("  No checkpoints found — skipping.")
        return {}

    available = sorted(fold_preds.keys())
    n_ens = len(available)

    fold_scores = {f: score_preds(fold_preds[f], tcore, shared) for f in available}
    ensemble_preds = build_ensemble(fold_preds, shared)
    ens_scores = score_preds(ensemble_preds, tcore, shared)

    print(f"\n  {cfg['label']}  ({n_ens}/{N_FOLDS} folds), {len(shared)} test TFs")
    print(f"  {'Model':<22} {'mean r':>9} {'median r':>10} {'n':>5}")
    print("  " + "-" * 50)
    for f in available:
        rs = fold_scores[f]
        print(f"    Fold {f}              {np.nanmean(rs):>9.4f} {np.nanmedian(rs):>10.4f} {len(rs):>5}")
    print(f"  {'Ensemble':22} {np.nanmean(ens_scores):>9.4f} {np.nanmedian(ens_scores):>10.4f} {len(ens_scores):>5}")
    print(f"  {'DeepPBS':22} {np.nanmean(dpbs_scores):>9.4f} {np.nanmedian(dpbs_scores):>10.4f} {len(dpbs_scores):>5}")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = {"ensemble": ens_scores.tolist(), "deeppbs": dpbs_scores.tolist()}
    for f in available:
        out[f"fold_{f}"] = fold_scores[f].tolist()
    json.dump(out, open(os.path.join(cfg["out_dir"], "canon_scores.json"), "w"), indent=2)
    print(f"  Saved {cfg['out_dir']}/canon_scores.json")
    return {"mean": float(np.nanmean(ens_scores)), "median": float(np.nanmedian(ens_scores)),
            "n_folds": n_ens, "n_tfs": len(ens_scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["deeppbs", "fulldata", "deeppbs_v18a", "fulldata_v18a", "both", "both_v18a"],
                    default="both_v18a", help="Which experiment to evaluate (default: both_v18a)")
    args = ap.parse_args()

    if args.mode == "both":
        modes = ["deeppbs", "fulldata"]
    elif args.mode == "both_v18a":
        modes = ["deeppbs_v18a", "fulldata_v18a"]
    else:
        modes = [args.mode]

    # Build canonical targets once (same test set for both)
    print("Building canonical targets from test set ...")
    tmp_cfg = TFScopeConfig()
    ds_ref = TFDataset(tmp_cfg, CONFIGS["deeppbs"]["data"], SPLIT,
                       split="test", max_seq_len=1024)
    ld_ref = DataLoader(ds_ref, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_variable_length)
    T_all, M_all = [], []
    for b in ld_ref:
        T_all.append(b["target_pwm"].numpy())
        M_all.append(b["pwm_mask"].numpy())
    T_all = np.concatenate(T_all)
    M_all = np.concatenate(M_all)

    tcore = {}
    for i, fn in enumerate(ds_ref.filenames):
        idx = M_all[i].astype(bool)
        if not idx.any(): continue
        c = canonicalize(np.clip(T_all[i][:, idx], 1e-8, 1.0).astype(np.float32))
        tcore[fn] = c / c.sum(0, keepdims=True).clip(1e-8)

    dpp = deeppbs_preds(list(tcore.keys()))
    shared = [fn for fn in tcore if fn in dpp]
    dpbs_scores = np.array([canon_fixed_r(tcore[fn], dpp[fn]) for fn in shared])
    print(f"Test set: {len(shared)} TFs with DeepPBS predictions\n")

    results = {}
    for mode in modes:
        results[mode] = eval_one(mode, tcore, dpbs_scores, shared)

    if args.mode == "both" and all(results.values()):
        print("\n═══ Summary ══════════════════════════════════════════")
        print(f"  {'Model':<28} {'mean r':>9} {'median r':>10} {'folds':>7}")
        print("  " + "─" * 58)
        for mode in modes:
            r = results[mode]
            lbl = CONFIGS[mode]["label"]
            print(f"  {lbl:<28} {r['mean']:>9.4f} {r['median']:>10.4f}  {r['n_folds']}/{N_FOLDS}")
        print(f"  {'DeepPBS':<28} {np.nanmean(dpbs_scores):>9.4f} {np.nanmedian(dpbs_scores):>10.4f}  {'ref':>7}")
        print("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
