#!/usr/bin/env python
"""Definitive v10 / v14 / DeepPBS comparison — identical metrics, strand-symmetric.

v10/v14: inference + per-TF reverse-complement to best-matching strand.
DeepPBS:  per-structure predictions (struct_preds.npz), already strand-aligned.
All metrics computed with one code path on the per-element scale.
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
device = "cuda" if torch.cuda.is_available() else "cpu"
RC = [3, 2, 1, 0]


def infer(ckpt, data):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False); m.eval()
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
    return np.concatenate(P), np.concatenate(T), np.concatenate(M)


def colr(p, t, L):
    return np.nanmean([pearsonr(t[:, j], p[:, j])[0] for j in range(L)])


def strand_orient(P, T, Mk):
    """Reverse-complement each prediction's valid window to best-match the target."""
    out = P.copy(); nf = 0
    for i in range(len(P)):
        L = int(Mk[i].sum())
        if L < 2: continue
        rf = colr(P[i][:, :L], T[i][:, :L], L)
        rc = P[i].copy(); rc[:, :L] = P[i][:, :L][RC][:, ::-1]
        rr = colr(rc[:, :L], T[i][:, :L], L)
        if not np.isnan(rr) and (np.isnan(rf) or rr > rf):
            out[i] = rc; nf += 1
    return out, nf


def metrics_from_cols(plist, tlist):
    R = dict(r=[], mae=[], rmse=[], ce=[], kl=[], icr=[], top1=[], auc=[], f1=[], mcc=[])
    for p, t in zip(plist, tlist):
        L = p.shape[1]
        if L < 2: continue
        p = np.clip(p, 1e-8, 1); p = p / p.sum(0, keepdims=True)
        t = np.clip(t, 1e-8, 1); t = t / t.sum(0, keepdims=True)
        R["r"].append(np.nanmean([pearsonr(t[:, j], p[:, j])[0] for j in range(L)]))
        R["mae"].append(np.abs(p - t).mean()); R["rmse"].append(np.sqrt(((p - t) ** 2).mean()))
        R["ce"].append(-(t * np.log(p)).sum(0).mean()); R["kl"].append((t * (np.log(t) - np.log(p))).sum(0).mean())
        ic = 2.0 - (-(t * np.log2(t)).sum(0)); w = ic / (ic.sum() + 1e-8)
        mp = (w * p).sum(); mt = (w * t).sum()
        cov = (w * (p - mp) * (t - mt)).sum(); sp = np.sqrt((w * (p - mp) ** 2).sum()); st = np.sqrt((w * (t - mt) ** 2).sum())
        R["icr"].append(cov / (sp * st + 1e-8))
        pc = p.argmax(0); tc = t.argmax(0); R["top1"].append((pc == tc).mean())
        a = []
        for b in range(4):
            isb = (tc == b).astype(int)
            if isb.sum() >= 1 and (1 - isb).sum() >= 1:
                try: a.append(roc_auc_score(isb, p[b]))
                except: pass
        R["auc"].append(np.mean(a) if a else np.nan)
        R["f1"].append(f1_score(tc, pc, average="macro", zero_division=0))
        try: R["mcc"].append(matthews_corrcoef(tc, pc))
        except: R["mcc"].append(np.nan)
    med = np.nanmedian(R["r"])
    return {k: np.nanmean(v) for k, v in R.items()}, med


def to_cols(P, T, Mk):
    pl, tl = [], []
    for i in range(len(P)):
        L = int(Mk[i].sum())
        if L < 2: continue
        pl.append(P[i][:, :L]); tl.append(T[i][:, :L])
    return pl, tl


def main():
    out = {}
    for tag, ckpt, data in [("v10", "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v10_single/ckpt_epoch100.pt", "data/processed/tf_pwm_deeppbs_only.parquet"),
                            ("v14", "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc/ckpt_best.pt", "data/processed/tf_pwm_deeppbs_only.parquet")]:
        P, T, Mk = infer(ckpt, data)
        Ps, nf = strand_orient(P, T, Mk)
        pl, tl = to_cols(Ps, T, Mk)
        M, med = metrics_from_cols(pl, tl)
        M["med"] = med; out[tag] = M
        print(f"{tag}: strand-flipped {nf}/{len(P)}")

    # DeepPBS per-structure (already strand-aligned)
    d = np.load("results/deeppbs_blind_benchmark/struct_preds.npz", allow_pickle=True)
    names = sorted(set(k.rsplit("::", 1)[0] for k in d.files))
    pl = [d[n + "::pred"].T for n in names]; tl = [d[n + "::true"].T for n in names]
    M, med = metrics_from_cols(pl, tl); M["med"] = med; out["DeepPBS"] = M

    order = [("Mean Pearson r","r"),("Median Pearson r","med"),("IC-weighted r","icr"),
             ("MAE","mae"),("RMSE","rmse"),("Cross-entropy","ce"),("KL divergence","kl"),
             ("Top-1 accuracy","top1"),("AUC (macro)","auc"),("F1 (macro)","f1"),("MCC","mcc")]
    print(f"\n{'Metric':<20} {'v10':>9} {'v14':>9} {'DeepPBS':>9}   best")
    print("-"*60)
    for name, k in order:
        lo = name.split()[0] in ("MAE","RMSE","Cross-entropy","KL")
        vals = {t: out[t][k] for t in ("v10","v14","DeepPBS")}
        best = min(vals, key=vals.get) if lo else max(vals, key=vals.get)
        print(f"{name:<20} {out['v10'][k]:>9.4f} {out['v14'][k]:>9.4f} {out['DeepPBS'][k]:>9.4f}   {best}")
    print("\n(All strand-symmetric, identical per-element metric code, 130 TFs)")


if __name__ == "__main__":
    main()
