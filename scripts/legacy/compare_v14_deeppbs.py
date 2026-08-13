#!/usr/bin/env python
"""All metrics for v10 vs v14 vs DeepPBS, computed identically from predicted PWMs.

Fixes the MAE-scale issue: every metric is recomputed from each model's actual
predicted PWM arrays with one code path, so MAE/RMSE/CE/AUC/F1 are directly
comparable (no 4x DeepPBS convention).
"""
import os, sys, json
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel

SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
MAX_L = 20
device = "cuda" if torch.cuda.is_available() else "cpu"


def infer(ckpt, data):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False)
    m.eval()
    ds = TFDataset(cfg, data, SPLIT, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    P, T, M = [], [], []
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long) for k, v in b.items()}
            _, pw, _ = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                         retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
                         retrieved_sims=b.get("retrieved_sims"))
            P.append(F.softmax(pw, 1).cpu().numpy()); T.append(b["target_pwm"].cpu().numpy()); M.append(b["pwm_mask"].cpu().numpy())
    return np.concatenate(P), np.concatenate(T), np.concatenate(M), ds.filenames


def metrics(preds, targs, masks):
    R = dict(r=[], med=[], mae=[], rmse=[], ce=[], kl=[], icr=[], top1=[], auc=[], f1=[], mcc=[])
    for i in range(len(preds)):
        idx = masks[i].astype(bool)
        if not idx.any(): continue
        p = preds[i][:, idx]; t = targs[i][:, idx]
        R["r"].append(np.nanmean([pearsonr(t[:, j], p[:, j])[0] for j in range(p.shape[1])]))
        R["mae"].append(np.abs(p - t).mean())
        R["rmse"].append(np.sqrt(((p - t) ** 2).mean()))
        pc = np.clip(p, 1e-8, 1); tc = np.clip(t, 1e-8, 1)
        R["ce"].append(-(tc * np.log(pc)).sum(0).mean())
        R["kl"].append((tc * (np.log(tc) - np.log(pc))).sum(0).mean())
        ic = 2.0 - (-(tc * np.log2(tc)).sum(0)); w = ic / (ic.sum() + 1e-8)
        mp = (w * p).sum(); mt = (w * t).sum()
        cov = (w * (p - mp) * (t - mt)).sum(); sp = np.sqrt((w * (p - mp) ** 2).sum()); st = np.sqrt((w * (t - mt) ** 2).sum())
        R["icr"].append(cov / (sp * st + 1e-8))
        pc2 = p.argmax(0); tc2 = t.argmax(0)
        R["top1"].append((pc2 == tc2).mean())
        tb = t.argmax(0); a = []
        for b in range(4):
            isb = (tb == b).astype(int)
            if isb.sum() >= 1 and (1 - isb).sum() >= 1:
                try: a.append(roc_auc_score(isb, p[b]))
                except: pass
        R["auc"].append(np.mean(a) if a else np.nan)
        R["f1"].append(f1_score(tc2, pc2, average="macro", zero_division=0))
        try: R["mcc"].append(matthews_corrcoef(tc2, pc2))
        except: R["mcc"].append(np.nan)
    R["med"] = [np.nanmedian(R["r"])]
    return {k: (np.nanmean(v) if k != "med" else v[0]) for k, v in R.items()}


def main():
    v10p, T, Mk, fns = infer("/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v10_single/ckpt_epoch100.pt",
                             "data/processed/tf_pwm_deeppbs_only.parquet")
    v14p, _, _, _ = infer("/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc/ckpt_best.pt",
                          "data/processed/tf_pwm_deeppbs_only.parquet")

    # DeepPBS predictions mapped to the same test order
    df = pd.read_parquet("data/processed/tf_pwm_deeppbs_only.parquet")
    fn2gene = dict(zip(df["filename"], df["gene_symbol"]))
    dpbs = np.load("results/deeppbs_blind_benchmark/gene_preds.npz", allow_pickle=True)
    dkeys_ci = {k.upper(): k for k in dpbs.files}   # case-insensitive gene lookup
    dpp = np.full_like(v14p, 0.25)
    dmask = Mk.copy()
    for i, fn in enumerate(fns):
        g = fn2gene.get(fn, "")
        gk = dkeys_ci.get(g.upper())
        if gk is not None:
            p = dpbs[gk]; L = min(p.shape[0], MAX_L)
            arr = np.full((4, MAX_L), 0.25, np.float32); arr[:, :L] = np.clip(p[:L].T, 1e-8, 1)
            dpp[i] = arr / arr.sum(0, keepdims=True)
        else:
            dmask[i] = 0  # exclude TFs with no DeepPBS prediction

    # shared-116 mask: only TFs where DeepPBS has a prediction
    shared = dmask.copy()                      # already zeroed for the 14 missing
    has_dp = shared.any(1)                     # (130,) bool
    n_dp = int(has_dp.sum())

    order = [("Mean Pearson r","r"),("Median Pearson r","med"),("IC-weighted r","icr"),
             ("MAE (same scale)","mae"),("RMSE","rmse"),("Cross-entropy","ce"),
             ("KL divergence","kl"),("Top-1 accuracy","top1"),("AUC (macro OvR)","auc"),
             ("F1 (macro)","f1"),("MCC","mcc")]

    # ── FAIR head-to-head: all three on the shared 116 TFs ───────────────────
    m10s = metrics(v10p, T, shared)
    m14s = metrics(v14p, T, shared)
    mdps = metrics(dpp,  T, shared)
    print(f"\n=== FAIR head-to-head: shared {n_dp} TFs (DeepPBS-covered) ===")
    print(f"{'Metric':<22} {'v10':>9} {'v14':>9} {'DeepPBS':>9}   best")
    print("-"*62)
    for name, k in order:
        lo = name.split()[0] in ("MAE","RMSE","Cross-entropy","KL")
        vals = {"v10": m10s[k], "v14": m14s[k], "DeepPBS": mdps[k]}
        best = min(vals, key=vals.get) if lo else max(vals, key=vals.get)
        print(f"{name:<22} {m10s[k]:>9.4f} {m14s[k]:>9.4f} {mdps[k]:>9.4f}   {best}")

    # ── Coverage view: v10/v14 on all 130 (DeepPBS cannot predict 14) ────────
    m10a = metrics(v10p, T, Mk); m14a = metrics(v14p, T, Mk)
    print(f"\n=== Coverage: v10/v14 on ALL 130 (DeepPBS covers only {n_dp}) ===")
    print(f"{'Metric':<22} {'v10(130)':>9} {'v14(130)':>9}")
    print("-"*44)
    for name, k in order:
        print(f"{name:<22} {m10a[k]:>9.4f} {m14a[k]:>9.4f}")


if __name__ == "__main__":
    main()
