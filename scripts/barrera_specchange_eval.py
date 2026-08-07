#!/usr/bin/env python
"""E3-eval: does TFScope's predicted WT->MUT specificity change (Δpred) discriminate
Barrera-2016 spec.change=Yes vs No?  Phenotype-level mutation sensitivity.

spec.change (motif-shape change) is the RIGHT target for a normalized-PWM model;
aff.change (8-mer count) is affinity, which PWMs cannot see. Δpred = 1-corr over the
model's WT gate core (same 42-frame, no alignment needed). Reports AUROC(Δpred, Yes)
plus the measured-PWM sanity AUROC(Δtrue, Yes). Pure eval, no training.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
dev = "cuda"; CK = os.environ.get("BENCH_CK", "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42")
S6 = "/data1/leihuang/rCLAMPS/barrera2016_SuppTable_S6_combined.csv"
pairs = json.load(open("results/mutation_benchmark/barrera_pairs.json"))["pairs"]

# S6 spec.change / aff.change by (gene, sub); rep R1/R2 -> take "Yes if any rep Yes"
s6 = pd.read_csv(S6)
spec = {}; aff = {}
for _, r in s6.iterrows():
    k = (r["prot"], r["sub"])
    spec[k] = spec.get(k, False) or (str(r["spec.change"]).strip() == "Yes")
    aff.setdefault(k, str(r["aff.change"]).strip())

def ic_core(p):
    ic = 2 + (np.clip(p, 1e-8, 1) * np.log2(np.clip(p, 1e-8, 1))).sum(0)
    inf = np.where(ic >= 0.2)[0]; return (inf[0], inf[-1] + 1) if len(inf) else (0, p.shape[1])

def auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, bool)
    pos = score[label]; neg = score[~label]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    # Mann-Whitney U / (n_pos*n_neg)
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

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
    _, pl, _ = m(t, dm, fi)
    return F.softmax(pl[0], 0).cpu().numpy()          # (4,42)

rows = []
for p in pairs:
    k = (p["gene"], p["mut"])
    if k not in spec:                                  # not in S6
        continue
    pw = pwm(p["wt_seq"]); pm = pwm(p["mut_seq"])
    a, b = ic_core(pw)
    dpred = 1 - np.corrcoef(pw[:, a:b].ravel(), pm[:, a:b].ravel())[0, 1]
    wt = np.array(p["wt_pwm"], np.float32); mt = np.array(p["mut_pwm"], np.float32)
    L = min(wt.shape[1], mt.shape[1])
    dtrue = 1 - np.corrcoef(wt[:, :L].ravel(), mt[:, :L].ravel())[0, 1]
    rows.append(dict(gene=p["gene"], mut=p["mut"], spec=spec[k], aff=aff.get(k, "?"),
                     dpred=float(dpred), dtrue=float(dtrue)))

df = pd.DataFrame(rows)
print(f"matched {len(df)}/{len(pairs)} HD pairs to S6 | spec.change Yes={df.spec.sum()} No={(~df.spec).sum()}")
print(f"\nAUROC(Δpred, spec.change=Yes)  = {auroc(df.dpred, df.spec):.3f}   <- model, phenotype-level")
print(f"AUROC(Δtrue, spec.change=Yes)  = {auroc(df.dtrue, df.spec):.3f}   <- measured-PWM sanity")
print(f"\nmean Δpred: Yes={df[df.spec].dpred.mean():.3f}  No={df[~df.spec].dpred.mean():.3f}")
print(f"mean Δtrue: Yes={df[df.spec].dtrue.mean():.3f}  No={df[~df.spec].dtrue.mean():.3f}")
os.makedirs("results/mutation_benchmark", exist_ok=True)
df.to_csv("results/mutation_benchmark/specchange_hd_eval.csv", index=False)
print("\nsaved results/mutation_benchmark/specchange_hd_eval.csv")
