#!/usr/bin/env python
"""Cross-family phenotype eval: AUROC(Δpred, spec.change) over the 101-pair,
7-family Barrera set (results/mutation_benchmark/crossfamily_pairs.json).
Δpred = 1-corr over the model's WT gate core (same 42-frame). Reports overall,
in-DBD-only, and per-family AUROC.  Pure eval, no training.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
dev = "cuda"; CK = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42"
pairs = json.load(open("results/mutation_benchmark/crossfamily_pairs.json"))["pairs"]

def ic_core(p):
    ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= 0.2)[0]; return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])

def auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

cfg = TFScopeConfig()
for k, v in json.load(open(CK + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval(); m.use_contact_pred_head = False
m.load_state_dict(torch.load(CK + "/ckpt_best.pt", map_location=dev, weights_only=False)["model"], strict=False)

@torch.no_grad()
def pwm(seq):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([4], device=dev)
    _, pl, _ = m(t, dm, fi); return F.softmax(pl[0], 0).cpu().numpy()

for p in pairs:
    if p["wt_seq"] == p["mut_seq"]:                 # out-of-DBD -> identical crop -> Δpred 0
        p["dpred"] = 0.0; continue
    pw = pwm(p["wt_seq"]); pm = pwm(p["mut_seq"]); a, b = ic_core(pw)
    p["dpred"] = float(1 - np.corrcoef(pw[:, a:b].ravel(), pm[:, a:b].ravel())[0, 1])

df = pd.DataFrame(pairs)
print(f"pairs {len(df)} | spec.change Yes {df.spec_change.sum()} No {(~df.spec_change).sum()}\n")
print(f"AUROC overall           = {auroc(df.dpred, df.spec_change):.3f}  (n={len(df)})")
d = df[df.in_dbd]
print(f"AUROC in-DBD only       = {auroc(d.dpred, d.spec_change):.3f}  (n={len(d)})")
print(f"\nmean Δpred: Yes={df[df.spec_change].dpred.mean():.4f}  No={df[~df.spec_change].dpred.mean():.4f}")
print("\nper-family (in-DBD):")
for fam, g in df[df.in_dbd].groupby("family"):
    ny = int(g.spec_change.sum())
    au = auroc(g.dpred, g.spec_change) if 0 < ny < len(g) else float("nan")
    print(f"  {fam:<18} n={len(g):>2} nYes={ny:>2}  AUROC={au if au==au else float('nan'):.3f}"
          f"  meanΔ Yes={g[g.spec_change].dpred.mean() if ny else float('nan'):.4f}"
          f" No={g[~g.spec_change].dpred.mean() if ny<len(g) else float('nan'):.4f}")
df.to_csv("results/mutation_benchmark/crossfamily_eval.csv", index=False)
print("\nsaved results/mutation_benchmark/crossfamily_eval.csv")
