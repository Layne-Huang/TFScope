#!/usr/bin/env python
"""All 6 Fig 3b-c orphan TFs: predict motif from the WHOLE DBD-spanning sequence (all DBDs
masked together), with BOTH the combined and residue-MoE (deeptune) models.

Input per TF = contiguous window [first_DBD .. last_DBD] (capped to 1000 aa, ending at the
last DBD so the canonical C-terminal DBD is always kept), mask = union of all DBD regions.
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

MODELS = {
    "combined":    "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt",
    "residue-MoE": "/data1/leihuang/project/TFScope/checkpoints/v19_residue_moe_deeptune/deeptune_ddp_seed42/ckpt_best.pt",
}
FID = {"SOHLH1": 3, "ADNP": 4, "ADNP2": 4, "ZHX2": 4, "ZHX3": 4, "ZGLP1": 9}
ORDER = ["SOHLH1", "ADNP", "ADNP2", "ZHX2", "ZHX3", "ZGLP1"]
FEAT = json.load(open("results/orphan_multidomain/dbd_features.json"))
FIGD = "figures/figure_orphans_multidomain"; os.makedirs(FIGD, exist_ok=True)
OUTD = "results/orphan_multidomain"; GTH = 0.5; MAXW = 1000

def ic_trim(p, thr=0.25):
    p = np.clip(p, 1e-8, 1.0); ic = 2.0 + (p * np.log2(p)).sum(0)
    inf = np.where(ic >= thr)[0]
    return p if len(inf) == 0 else p[:, inf[0]:inf[-1] + 1]
def cons(p): return "".join("ACGT"[i] for i in p.argmax(0))

def build_input(g):
    seq = open(f"{OUTD}/{g}.seq.txt").read().strip()
    dbds = [(a, b) for a, b, _, _ in FEAT[g]["dbds"]]
    lastend = max(b for _, b in dbds); firststart = min(a for a, _ in dbds)
    end = min(len(seq), lastend + 3)
    start = max(1, firststart - 3)
    if end - start + 1 > MAXW: start = end - MAXW + 1          # keep C-terminal-most DBDs
    win = seq[start-1:end]
    mask = [False] * len(win)
    kept = 0
    for a, b in dbds:
        for i in range(a, b + 1):
            j = i - start
            if 0 <= j < len(win): mask[j] = True
        if a >= start: kept += 1
    return win, mask, (start, end), len(dbds), kept

def run(model, seqc, maskb, fid):
    tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seqc]], dtype=torch.long, device=device)
    mask = torch.tensor([maskb], dtype=torch.bool, device=device)
    g, p, _ = infer(model, tok, mask, fid, ret=None)
    return ic_trim(p[:, active_cols(g, GTH)])

# precompute windows
WIN = {g: build_input(g) for g in ORDER}
for g in ORDER:
    _, _, (s, e), nd, kept = WIN[g]
    print(f"{g:7s} window {s}-{e} ({e-s+1} aa), DBDs {kept}/{nd} kept", flush=True)

results = {}
for mname, ckpt in MODELS.items():
    print(f"\nloading {mname} ...", flush=True)
    model, _ = load_model(ckpt, force_retrieval=False)
    for g in ORDER:
        win, mask, _, _, _ = WIN[g]
        p = run(model, win, mask, FID[g]); results[(g, mname)] = p
        np.save(f"{OUTD}/{g}_{mname}.npy", p)
        print(f"  {mname:11s} {g:7s} consensus {cons(p):16s} width {p.shape[1]}", flush=True)
    del model; torch.cuda.empty_cache()

def logo(ax, pwm, title, color):
    p = np.clip(pwm, 1e-8, 1.0); ic = np.maximum(2 + (p * np.log2(p)).sum(0), 0)
    logomaker.Logo(pd.DataFrame((p * ic).T, columns=list("ACGT")), ax=ax, color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2); ax.set_title(title, fontsize=8, color=color, loc="left", pad=2)

mnames = list(MODELS.keys())
fig, axes = plt.subplots(len(ORDER), 2, figsize=(9.5, 1.35 * len(ORDER)))
for r, g in enumerate(ORDER):
    fam = {3: "bHLH", 4: "Homeo", 9: "GATA/Other"}[FID[g]]
    for c, mn in enumerate(mnames):
        p = results[(g, mn)]
        col = "#0072B2" if mn == "residue-MoE" else "#009E73"
        ttl = f"{g} ({fam}) | {mn}   {cons(p)}"
        logo(axes[r, c], p, ttl, col)
fig.suptitle("Orphan TFs: motif from whole DBD-spanning sequence (all DBDs) — combined vs residue-MoE",
             fontsize=11, fontweight="bold", y=1.0)
fig.tight_layout()
for e in ["png", "pdf"]: fig.savefig(f"{FIGD}/orphans_multidomain.{e}", dpi=200, bbox_inches="tight")
print("\nsaved", f"{FIGD}/orphans_multidomain.png", flush=True)
