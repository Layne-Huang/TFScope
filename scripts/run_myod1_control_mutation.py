#!/usr/bin/env python
"""Negative control for the MyoD1 L122R switch: mutating an UNNECESSARY (non-recognition)
residue should leave the predicted PWM essentially unchanged, whereas L122R (a DNA-reading
basic-region residue) drives the CACCTG->CACGTG switch.

Ranks every DBD residue by in-silico importance (Ala-scan Delta-motif), then mutates the
lowest-importance residue(s) with the SAME kind of change (-> R, charge-matched to L122R)
and compares Delta_switch and total motif change to L122R.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["HF_HOME"] = "/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import logomaker, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CK = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
WT = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"   # MyoD1 bHLH DBD
L122_POS = 11   # WT[11]=='L' -> R is the recognition switch
FID = 3
FIGD = "figures/figure4a_control"; os.makedirs(FIGD, exist_ok=True); os.makedirs("results/myod1_mut", exist_ok=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
B = {"A": 0, "C": 1, "G": 2, "T": 3}
def rc(s): return s[::-1].translate(str.maketrans("ACGT", "TGCA"))

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.dirname(CK) + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CK, map_location=dev, weights_only=False)["model"], strict=False)

@torch.no_grad()
def predict(seq):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([FID], device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return torch.sigmoid(gl)[0].cpu().numpy(), F.softmax(pl, 1)[0].cpu().numpy()

def core_idx(p, g):
    c = np.where(g > 0.5)[0]
    if len(c) < 4:
        ic = (p * np.log2(p + 1e-9)).sum(0) + 2; a = ic.argmax(); c = np.arange(max(0, a - 4), min(p.shape[1], a + 5))
    return c.min(), c.max() + 1

def score(P, seq):
    lo = np.log2(np.clip(P, 1e-6, 1) / 0.25); W = P.shape[1]; L = len(seq); best = -1e9
    for s in (seq, rc(seq)):
        idx = [B[c] for c in s]
        for off in range(0, W - L + 1):
            best = max(best, float(sum(lo[idx[j], off + j] for j in range(L))))
    return best

def d_switch(pw, pm):
    dWT = score(pw, "CACGTG") - score(pw, "CACCTG")
    dMT = score(pm, "CACGTG") - score(pm, "CACCTG")
    return dMT - dWT

gw, pw = predict(WT); lo, hi = core_idx(pw, gw); Wcore = pw[:, lo:hi]
def dmotif(pm): return float(np.abs(pm[:, lo:hi] - Wcore).sum())   # total L1 change over WT core

# ---- Ala-scan importance to find UNNECESSARY residues ----
imp = []
for i, aa in enumerate(WT):
    if aa == "A":
        imp.append((i, aa, 0.0)); continue
    mut = WT[:i] + "A" + WT[i+1:]
    _, pmi = predict(mut)
    imp.append((i, aa, dmotif(pmi)))
imp_sorted = sorted(imp, key=lambda z: z[2])
rank = {i: r for r, (i, _, _) in enumerate(sorted(imp, key=lambda z: -z[2]))}
print(f"L122 (pos {L122_POS}, {WT[L122_POS]}) importance rank = {rank[L122_POS]+1}/{len(WT)} (1=most important)", flush=True)
# pick lowest-importance non-basic-region residues (avoid the N-terminal basic region 0-12)
controls = [i for i, aa, _ in imp_sorted if i > 12 and aa != "R"][:2]

def mutate(seq, pos, to): return seq[:pos] + to + seq[pos+1:]

cases = [("L122R (recognition)", mutate(WT, L122_POS, "R"), "#d9544d")]
for c in controls:
    cases.append((f"{WT[c]}{c}R (unnecessary, imp-rank {rank[c]+1})", mutate(WT, c, "R"), "#7f8c9b"))

rows = [("MyoD1 WT", pw, "#333")]
print("\ncase                                Δ_switch   Δmotif(L1)", flush=True)
print(f"{'WT (reference)':34s}   {0.0:+7.2f}   {0.0:7.2f}", flush=True)
res = {}
for name, seq, col in cases:
    gm, pm = predict(seq); ds = d_switch(pw, pm); dm = dmotif(pm)
    res[name] = dict(delta_switch=round(ds, 2), delta_motif_L1=round(dm, 3))
    rows.append((f"{name}   Δswitch={ds:+.2f}  Δmotif={dm:.2f}", pm, col))
    print(f"{name:34s}   {ds:+7.2f}   {dm:7.2f}", flush=True)
json.dump(res, open("results/myod1_mut/control_mutation.json", "w"), indent=1)

def logo(ax, pwm, title, color):
    p = np.clip(pwm[:, lo:hi], 1e-8, 1.0); ic = np.maximum(2 + (p * np.log2(p)).sum(0), 0)
    logomaker.Logo(pd.DataFrame((p * ic).T, columns=list("ACGT")), ax=ax, color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2); ax.set_title(title, fontsize=8.5, color=color, loc="left", pad=2)

n = len(rows)
fig, axes = plt.subplots(n, 1, figsize=(5.6, 1.15*n + 0.4))
for ax, (t, p, c) in zip(axes, rows): logo(ax, p, t, c)
fig.suptitle("Control: only the recognition residue (L122R) shifts the motif;\nmutating unnecessary residues does not",
             fontsize=9.5, fontweight="bold", y=1.0)
fig.tight_layout()
for e in ["png", "pdf"]: fig.savefig(f"{FIGD}/myod1_control.{e}", dpi=200, bbox_inches="tight")
print("\nsaved", f"{FIGD}/myod1_control.png", flush=True)
