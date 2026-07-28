#!/usr/bin/env python
"""Run the residue-MoE (deeptune) model on ADNP under three input crops, since ADNP is
multi-domain (9 C2H2 zinc fingers + an atypical homeobox), and compare each predicted
motif to the independent JASPAR UN0305.1 (UNVALIDATED, ChIP-seq) reference.

Crops (UniProt Q9H2P0, 1102 aa):
  1) Homeobox-only        752-822  (homeobox 754-814)        FID=Homeodomain(4)
  2) C2H2 array (ZF5-7)   447-535                            FID=C2H2_medium(1)
  3) Multi-domain         447-822, mask = C2H2(447-535) + homeobox(752-814)  FID=Homeodomain(4)
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["HF_HOME"] = "/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/case_study")
import numpy as np, torch
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import logomaker, pandas as pd
from cs_utils import load_model, active_cols, infer, device
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42/ckpt_best.pt"
SEQ = open("/tmp/adnp_seq.txt").read().strip()   # full 1102-aa ADNP (UniProt Q9H2P0)
OUTD = "results/adnp_case/multidomain"; os.makedirs(OUTD, exist_ok=True)
FIGD = "figures/figure_adnp_multidomain"; os.makedirs(FIGD, exist_ok=True)
GTH = 0.5

def ic_trim(p, thr=0.25):
    p = np.clip(p, 1e-8, 1.0); ic = 2.0 + (p * np.log2(p)).sum(0)
    inf = np.where(ic >= thr)[0]
    return p if len(inf) == 0 else p[:, inf[0]:inf[-1] + 1]
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))
def logo(ax, pwm, title, color="black"):
    p = np.clip(pwm, 1e-8, 1.0); ic = np.maximum(2 + (p * np.log2(p)).sum(0), 0)
    logomaker.Logo(pd.DataFrame((pwm * ic).T, columns=list("ACGT")), ax=ax, color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2); ax.set_title(title, fontsize=8.5, color=color, loc="left", pad=3)

def run(seq_crop, mask_bool, fid):
    tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq_crop]], dtype=torch.long, device=device)
    mask = torch.tensor([mask_bool], dtype=torch.bool, device=device)
    g, p, _ = infer(model, tok, mask, fid, ret=None)
    return ic_trim(p[:, active_cols(g, GTH)])

# JASPAR UN0305.1 reference (fetched to /tmp/un0305_pwm.npy)
ref = ic_trim(np.load("/tmp/un0305_pwm.npy")) if os.path.exists("/tmp/un0305_pwm.npy") else None
def r_to_ref(pred):
    if ref is None: return float("nan")
    _, _, _, r = align_pwm(pred, ref, max_shift=12, consider_revcomp=True)
    return float(r)

print("loading residue-MoE (deeptune) ...", flush=True)
model, _ = load_model(CKPT, force_retrieval=False)

# ---- crop definitions (1-based inclusive -> 0-based slicing) ----
def sl(a, b): return SEQ[a-1:b]            # 1-based inclusive
crops = {}
# 1) homeobox-only
crops["HD (752-822)\nHomeodomain"] = (run(sl(752, 822), [True]*(822-752+1), 4), 4)
# 2) C2H2 array ZF5-7
c2 = sl(447, 535); crops["C2H2 array (447-535)\nZF5-7"] = (run(c2, [True]*len(c2), 1), 1)
# 3) multi-domain crop 447-822 with mask over C2H2(447-535)+homeobox(752-814)
mdseq = sl(447, 822); L = len(mdseq); m = [False]*L
for a, b in [(447, 535), (752, 814)]:
    for i in range(a-447, b-447+1): m[i] = True
crops["Multi-domain (447-822)\nmask C2H2+HD"] = (run(mdseq, m, 4), 4)

rows = [("JASPAR UN0305.1 (UNVALIDATED, ChIP-seq)", ref, "#888", None)] if ref is not None else []
for name, (pwm, fid) in crops.items():
    r = r_to_ref(pwm)
    np.save(f"{OUTD}/ADNP_{name.split(chr(10))[0].split()[0].replace('(','').replace(')','')}.npy", pwm)
    rows.append((f"{name}   consensus {cons(pwm)}   (r vs JASPAR {r:.2f})", pwm, "#0072B2", r))
    print(f"{name.splitlines()[0]:28s} consensus {cons(pwm):14s} width {pwm.shape[1]:2d}  r_vs_JASPAR {r:.3f}", flush=True)

n = len(rows)
fig, axes = plt.subplots(n, 1, figsize=(6.4, 1.35*n + 0.5))
if n == 1: axes = [axes]
for ax, (title, pwm, color, r) in zip(axes, rows):
    logo(ax, pwm, title, color)
fig.suptitle("ADNP is multi-domain: motif depends on which DNA-binding module is the input",
             fontsize=10, fontweight="bold", y=1.0)
fig.tight_layout()
for e in ["png", "pdf"]: fig.savefig(f"{FIGD}/adnp_multidomain.{e}", dpi=200, bbox_inches="tight")
print("saved", f"{FIGD}/adnp_multidomain.png", flush=True)
