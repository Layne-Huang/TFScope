#!/usr/bin/env python
"""Oracle-aligned (trimmed-core, offset+RC) ensemble evaluation.

For each test TF: ensemble-average the 5 fold predictions, then align vs
the trimmed informative core with full offset+RC search.

Usage:
  python scripts/eval_ensemble_oracle.py --mode fulldata_v18a
  python scripts/eval_ensemble_oracle.py --mode both_v18a
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")

from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from torch.utils.data import DataLoader

SPLIT    = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
DATA_RAW = "data/processed/tf_pwm_deeppbs_only.parquet"
IC_THRESH = 0.25  # bits
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


def last_ckpt(fold_dir):
    """Return the highest-epoch periodic checkpoint in fold_dir, or None."""
    import glob
    files = sorted(glob.glob(os.path.join(fold_dir, "ckpt_epoch*.pt")))
    return files[-1] if files else None


def infer_fold(fold, ckpt_root, data, use_last=False):
    fold_dir = os.path.join(ckpt_root, f"fold_{fold}")
    if use_last:
        ckpt = last_ckpt(fold_dir)
        if ckpt is None:
            print(f"  [skip] fold {fold}: no epoch ckpt found"); return None, None, None
        print(f"  fold {fold}: using {os.path.basename(ckpt)}")
    else:
        ckpt = os.path.join(fold_dir, "ckpt_best.pt")
    if not os.path.exists(ckpt):
        print(f"  [skip] fold {fold}: not found"); return None, None, None

    cfg = TFScopeConfig()
    cfg_path = os.path.join(os.path.dirname(ckpt), "config.json")
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
            P.append(F.softmax(pw, 1).cpu().numpy())   # (B, 4, L_max)
            T.append(b["target_pwm"].cpu().numpy())
            M.append(b["pwm_mask"].cpu().numpy())
            G.append(gate.sigmoid().cpu().numpy())      # soft gate prob (B, L_max)

    P = np.concatenate(P); T = np.concatenate(T)
    M = np.concatenate(M); G = np.concatenate(G)
    # Return full L_max outputs (gate prob + PWM) so the ensemble can do
    # gate-weighted averaging across folds before thresholding.
    preds = {fn: (P[i], G[i]) for i, fn in enumerate(ds.filenames)}   # (4,L), (L,)
    tgts  = {fn: (T[i], M[i]) for i, fn in enumerate(ds.filenames)}
    avg_len = np.mean([G[i] > 0.5 for i in range(len(ds.filenames))], axis=0).sum() \
              if len(ds.filenames) else 0
    print(f"  fold {fold}: {len(preds)} test TFs  "
          f"(pred_len mean={(G > 0.5).sum(1).mean():.1f})")
    return preds, tgts, ds.filenames


def ic_bits(pwm):
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)


def trimmed_core(target, mask):
    idx = mask.astype(bool)
    if not idx.any(): return None
    t = target[:, idx]
    ic = ic_bits(t)
    inf = np.where(ic >= IC_THRESH)[0]
    if len(inf) == 0: return None
    return t[:, inf[0]:inf[-1] + 1]


def build_ensemble(fold_preds, fns, gate_thresh=0.5):
    """IC-weighted ensemble across folds.

    Each fold returns (pwm: 4×L_max, gate_prob: L_max).
    Active columns: majority (≥ F/2) of folds must vote gate > gate_thresh.
    Fold contribution weighted by total information content (IC) over its active
    columns: folds predicting specific motifs dominate over near-uniform folds.
    """
    available = sorted(fold_preds.keys())
    F_total = len(available)
    out = {}
    for fn in fns:
        items = [fold_preds[f][fn] for f in available if fn in fold_preds.get(f, {})]
        if not items: continue
        F = len(items)
        pwms  = np.stack([pwm  for pwm, _   in items], axis=0)  # (F, 4, L)
        gates = np.stack([gate for _,   gate in items], axis=0)  # (F, L)
        pwms  = np.clip(pwms, 1e-8, 1.0)
        # majority-vote active set
        votes = (gates > gate_thresh).astype(float)              # (F, L)
        n_votes = votes.sum(0)
        active = n_votes >= (F_total / 2.0)
        if not active.any():
            active = n_votes >= n_votes.max() * 0.5
        # per-fold IC over its own active columns (bits, non-negative)
        ic_per_col = (2.0 + (pwms * np.log2(pwms)).sum(1)).clip(0)  # (F, L)
        fold_ic = (ic_per_col * votes).sum(1).clip(1e-8)            # (F,) total IC
        # IC-weighted fold contributions, restricted to majority-voted columns
        w = (fold_ic[:, None, None] * votes[:, None, :])            # (F, 1, L)
        w_sum = w.sum(0).clip(1e-8)                                  # (1, L)
        avg_pwm = (pwms * w).sum(0) / w_sum                         # (4, L)
        avg_pwm = avg_pwm / avg_pwm.sum(0, keepdims=True).clip(1e-8)
        out[fn] = avg_pwm[:, active]
    return out


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


def oracle_score(pred, core):
    if pred is None or pred.shape[1] == 0: return None
    _, _, _, s = align_pwm(pred, core, max_shift=MAX_SHIFT, consider_revcomp=True)
    return s


def eval_one(cfg_key, T_all, M_all, fns_ref, dpp, cores, shared, use_last=False):
    cfg = CONFIGS[cfg_key]
    tag = " [last ckpt]" if use_last else " [best ckpt]"
    print(f"\n─── {cfg['label']}{tag} ───")
    fold_preds = {}
    for fold in range(N_FOLDS):
        p, _, fns = infer_fold(fold, cfg["ckpt_root"], cfg["data"], use_last=use_last)
        if p is None: continue
        fold_preds[fold] = p

    if not fold_preds:
        print("  No checkpoints — skipping."); return

    available = sorted(fold_preds.keys())
    ensemble  = build_ensemble(fold_preds, fns_ref)

    def score_preds(preds_dict, name, is_fold=False):
        rs = []
        for fn in shared:
            entry = preds_dict.get(fn)
            if entry is None: continue
            # individual fold returns (pwm, gate_prob); ensemble returns plain pwm array
            if is_fold:
                pwm, gate = entry
                active = gate > 0.5
                if not active.any(): active = gate > gate.max() * 0.5
                pred = pwm[:, active]
            else:
                pred = entry
            s = oracle_score(pred, cores[fn])
            if s is not None: rs.append(s)
        rs = np.array(rs)
        print(f"  {name:<26} mean={np.nanmean(rs):.4f}  median={np.nanmedian(rs):.4f}  n={len(rs)}")
        return rs

    fold_rs = {}
    for f in available:
        fold_rs[f] = score_preds(fold_preds[f], f"Fold {f}", is_fold=True)
    ens_rs = score_preds(ensemble, "Ensemble (5-fold)", is_fold=False)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    fname = "oracle_scores_last.json" if use_last else "oracle_scores.json"
    out = {"ensemble_oracle": ens_rs.tolist()}
    for f in available:
        out[f"fold_{f}_oracle"] = fold_rs[f].tolist()
    json.dump(out, open(os.path.join(cfg["out_dir"], fname), "w"), indent=2)
    print(f"  Saved {cfg['out_dir']}/{fname}")
    return np.nanmean(ens_rs), np.nanmedian(ens_rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["deeppbs_v18a", "fulldata_v18a", "both_v18a"],
                    default="fulldata_v18a")
    ap.add_argument("--use-last", action="store_true",
                    help="Use last epoch checkpoint instead of best-val checkpoint")
    args = ap.parse_args()

    modes = ["deeppbs_v18a", "fulldata_v18a"] if args.mode == "both_v18a" else [args.mode]

    # Load targets once using the DeepPBS-canon dataset
    print("Loading test targets ...")
    tmp_cfg = TFScopeConfig()
    ds_ref  = TFDataset(tmp_cfg, CONFIGS["fulldata_v18a"]["data"],
                        SPLIT, split="test", max_seq_len=1024)
    ld_ref  = DataLoader(ds_ref, batch_size=8, shuffle=False, num_workers=2,
                         collate_fn=collate_variable_length)
    T_all, M_all = [], []
    for b in ld_ref:
        T_all.append(b["target_pwm"].numpy()); M_all.append(b["pwm_mask"].numpy())
    T_all = np.concatenate(T_all); M_all = np.concatenate(M_all)
    fns_ref = ds_ref.filenames

    cores = {}
    for i, fn in enumerate(fns_ref):
        c = trimmed_core(T_all[i], M_all[i])
        if c is not None: cores[fn] = c

    dpp = deeppbs_preds(fns_ref)
    shared = [fn for fn in cores if fn in dpp]
    print(f"Shared test TFs with DeepPBS predictions: {len(shared)}")

    # DeepPBS oracle score
    dpbs_rs = []
    for fn in shared:
        s = oracle_score(dpp[fn], cores[fn])
        if s is not None: dpbs_rs.append(s)
    dpbs_rs = np.array(dpbs_rs)
    print(f"\n  {'DeepPBS (oracle)':<26} mean={np.nanmean(dpbs_rs):.4f}  median={np.nanmedian(dpbs_rs):.4f}  n={len(dpbs_rs)}")

    summary = {}
    for mode in modes:
        r = eval_one(mode, T_all, M_all, fns_ref, dpp, cores, shared, use_last=args.use_last)
        if r: summary[mode] = r

    if len(summary) > 1:
        print("\n═══ Oracle-Aligned Summary ═══════════════════════════")
        print(f"  {'Model':<30} {'mean r':>9} {'median r':>10}")
        print("  " + "─" * 52)
        for mode, (mn, med) in summary.items():
            print(f"  {CONFIGS[mode]['label']:<30} {mn:>9.4f} {med:>10.4f}")
        print(f"  {'DeepPBS':<30} {np.nanmean(dpbs_rs):>9.4f} {np.nanmedian(dpbs_rs):>10.4f}")
        print("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
