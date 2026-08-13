#!/usr/bin/env python
"""ADNP three-crop analysis, COMBINED vs residue-MoE (deeptune), vs JASPAR UN0305.1.

Crops (UniProt Q9H2P0): HD 752-822 (FID Homeodomain 4); C2H2 array 447-535 (FID C2H2_medium 1);
multi-domain 447-822 with mask over C2H2(447-535)+homeobox(752-814) (FID 4).
"""
import os, sys
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

MODELS = {
    "combined":     "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
    "residue-MoE":  "/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe/residue_moe_seed42/ckpt_best.pt",
}
SEQ = open("/tmp/adnp_seq.txt").read().strip()
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
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2); ax.set_title(title, fontsize=8, color=color, loc="left", pad=2)
def sl(a, b): return SEQ[a-1:b]

ref = ic_trim(np.load("/tmp/un0305_pwm.npy"))
def r_ref(p):
    _, _, _, r = align_pwm(p, ref, max_shift=12, consider_revcomp=True); return float(r)

# crop -> (seq, mask, fid)
CROPS = {}
CROPS["HD (752-822)"]        = (sl(752, 822), [True]*(822-752+1), 4)
c2 = sl(447, 535);           CROPS["C2H2 array (447-535)"] = (c2, [True]*len(c2), 1)
md = sl(447, 822); m = [False]*len(md)
for a, b in [(447, 535), (752, 814)]:
    for i in range(a-447, b-447+1): m[i] = True
CROPS["Multi-domain (447-822)"] = (md, m, 4)

def run(model, seqc, maskb, fid):
    tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seqc]], dtype=torch.long, device=device)
    mask = torch.tensor([maskb], dtype=torch.bool, device=device)
    g, p, _ = infer(model, tok, mask, fid, ret=None)
    return ic_trim(p[:, active_cols(g, GTH)])

results = {}   # (crop, model) -> (pwm, r)
for mname, ckpt in MODELS.items():
    print(f"loading {mname} ...", flush=True)
    model, _ = load_model(ckpt, force_retrieval=False)
    for cname, (s, mk, fid) in CROPS.items():
        p = run(model, s, mk, fid); results[(cname, mname)] = (p, r_ref(p))
        print(f"  {mname:11s} {cname:24s} {cons(p):14s} r_vs_JASPAR {results[(cname,mname)][1]:.3f}", flush=True)
    del model; torch.cuda.empty_cache()

# figure: JASPAR ref on top, then per crop: combined + residue-MoE
order = list(CROPS.keys()); mnames = list(MODELS.keys())
rows = [("JASPAR UN0305.1 (UNVALIDATED, ChIP-seq)", ref, "#888")]
for c in order:
    for mn in mnames:
        p, r = results[(c, mn)]
        rows.append((f"{c}  |  {mn}   {cons(p)}   (r vs JASPAR {r:.2f})", p, "#0072B2" if mn=="residue-MoE" else "#009E73"))
n = len(rows)
fig, axes = plt.subplots(n, 1, figsize=(6.6, 1.05*n + 0.4))
for ax, (t, p, col) in zip(axes, rows): logo(ax, p, t, col)
fig.suptitle("ADNP three crops × two models, vs JASPAR UN0305.1", fontsize=10, fontweight="bold", y=1.0)
fig.tight_layout()
for e in ["png", "pdf"]: fig.savefig(f"{FIGD}/adnp_compare_models.{e}", dpi=200, bbox_inches="tight")
print("saved", f"{FIGD}/adnp_compare_models.png", flush=True)
