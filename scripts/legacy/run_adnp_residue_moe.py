#!/usr/bin/env python
"""Run the new residue-MoE (deeptune) checkpoint on the orphan TF ADNP, sequence-only.

Reuses the ADNP case-study input (DBD sequence + family_id) but swaps in the new MoE
checkpoint. Saves the predicted PWM + a logo, and overlays the previous combined-model
ADNP prediction for comparison.
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"
os.environ["HF_HOME"] = "/data1/leihuang/.cache"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/case_study")
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker, pandas as pd
from cs_utils import load_model, tokens_from_seq, active_cols, infer

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_deeptune/deeptune_ddp_seed42/ckpt_best.pt"
CFG  = json.load(open("configs/case_study_adnp.yaml".replace(".yaml", ".yaml"))) if False else None
# ADNP case-study facts (from configs/case_study_adnp.yaml)
ADNP_SEQ = "LALDPKGHEDDSYEARKSFLTKYFNKQPYPTRREIEKLAASLWLWKSDIASHFSNKRKKCVRDCEKYKPGV"
FID = 4          # Homeodomain
GTH = 0.5
OUTD = "results/adnp_case/residue_moe"; os.makedirs(OUTD, exist_ok=True)
FIGD = "figures/figure_adnp_residue_moe"; os.makedirs(FIGD, exist_ok=True)


def ic_trim(pwm, thr=0.25):
    p = np.clip(pwm, 1e-8, 1.0); ic = 2.0 + (p * np.log2(p)).sum(0)
    inf = np.where(ic >= thr)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]

def consensus(p): return "".join("ACGT"[i] for i in p.argmax(0))

def logo(ax, pwm, title, color="black"):
    p = np.clip(pwm, 1e-8, 1.0); ic = np.maximum(2 + (p * np.log2(p)).sum(0), 0)
    logomaker.Logo(pd.DataFrame((pwm * ic).T, columns=list("ACGT")), ax=ax,
                   color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2)
    ax.set_title(title, fontsize=9, color=color, loc="left", pad=3)


print("loading residue-MoE (deeptune) ...", flush=True)
model, _ = load_model(CKPT, force_retrieval=False)
tok, mask = tokens_from_seq(ADNP_SEQ)
g_nr, p_nr, _ = infer(model, tok, mask, FID, ret=None)
core = p_nr[:, active_cols(g_nr, GTH)]
core = ic_trim(core)
print("ADNP residue-MoE predicted consensus:", consensus(core), "| width", core.shape[1], flush=True)
np.save(f"{OUTD}/ADNP_residue_moe.npy", core)
pd.DataFrame(core.T, columns=list("ACGT")).to_csv(f"{OUTD}/ADNP_residue_moe.pwm.tsv", sep="\t", index=False)

# comparison: previous combined-model ADNP prediction (used in Fig 3b-c)
prev_path = "results/genome_cre_scan/pwms/ADNP.npy"
prev = ic_trim(np.load(prev_path)) if os.path.exists(prev_path) else None

n = 2 if prev is not None else 1
fig, axes = plt.subplots(n, 1, figsize=(5.2, 1.5 * n + 0.4))
if n == 1: axes = [axes]
logo(axes[0], core, f"ADNP — residue-MoE (deeptune)  consensus {consensus(core)}", "#0072B2")
if prev is not None:
    logo(axes[1], prev, f"ADNP — combined (Fig 3b-c)  consensus {consensus(prev)}", "#555")
fig.suptitle("ADNP orphan-TF motif: new residue-MoE vs combined", fontsize=10, fontweight="bold", y=1.0)
fig.tight_layout()
for e in ["png", "pdf"]: fig.savefig(f"{FIGD}/adnp_residue_moe.{e}", dpi=200, bbox_inches="tight")
print("saved", f"{FIGD}/adnp_residue_moe.png", flush=True)
