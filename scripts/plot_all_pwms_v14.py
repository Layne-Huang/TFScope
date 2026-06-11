#!/usr/bin/env python
"""Multi-page PWM-logo comparison for all 130 test TFs: truth | v14 | DeepPBS.

Runs v14 inference, loads targets + DeepPBS predictions, renders sequence logos
in a paged grid (8 TFs per page, 3 columns), annotated with per-column Pearson r.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
sys.path.insert(0, "pwm_rosetta")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from pwm_hybrid.pwm.viz import makeLogo

CKPT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc/ckpt_best.pt"
DATA = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
MAX_L = 20
device = "cuda" if torch.cuda.is_available() else "cpu"


def percol_r(pred, targ, L):
    return np.nanmean([pearsonr(targ[:, j], pred[:, j])[0] for j in range(L)])


def draw_logo(ax, pwm, L, title, color):
    if L < 1:
        ax.axis("off"); return
    ppm = np.clip(pwm[:, :L].T, 1e-8, 1.0)
    ppm = ppm / ppm.sum(1, keepdims=True)
    makeLogo(ppm, ax)
    ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7, color=color, fontweight="bold")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-index-path", default=None,
                    help="Override retrieval index (e.g. LGO for clean eval)")
    ap.add_argument("--out", default="results/tfscope_v14_best/all_pwms_truth_v14_deeppbs.pdf")
    ap.add_argument("--v14-label", default="v14")
    a = ap.parse_args()

    # ── v14 inference ────────────────────────────────────────────────────────
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    if a.retrieval_index_path:
        cfg.retrieval_index_path = a.retrieval_index_path
        print(f"Overriding retrieval index -> {a.retrieval_index_path}")
    model = TFScopeModel(cfg).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"], strict=False)
    model.eval()

    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    fns = ds.filenames
    preds, targs, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                     for k, v in batch.items()}
            _, pwm, _ = model(batch["sequence_tokens"], batch["dbd_mask"], batch["family_id"],
                              retrieved_pwms=batch.get("retrieved_pwms"),
                              retrieved_masks=batch.get("retrieved_masks"),
                              retrieved_sims=batch.get("retrieved_sims"))
            preds.append(F.softmax(pwm, dim=1).cpu().numpy())
            targs.append(batch["target_pwm"].cpu().numpy())
            masks.append(batch["pwm_mask"].cpu().numpy())
    preds = np.concatenate(preds); targs = np.concatenate(targs); masks = np.concatenate(masks)

    # strand-symmetric: flip each v14 prediction to best-matching strand
    RC = [3, 2, 1, 0]
    def colr(p, t, L):
        return np.nanmean([pearsonr(t[:, j], p[:, j])[0] for j in range(L)])
    for i in range(len(preds)):
        L = int(masks[i].sum())
        if L < 2: continue
        rf = colr(preds[i][:, :L], targs[i][:, :L], L)
        rc = preds[i].copy(); rc[:, :L] = preds[i][:, :L][RC][:, ::-1]
        rr = colr(rc[:, :L], targs[i][:, :L], L)
        if not np.isnan(rr) and (np.isnan(rf) or rr > rf):
            preds[i] = rc

    # ── metadata + DeepPBS (per-structure, strand-aligned, covers all 130) ────
    df = pd.read_parquet(DATA)
    fn2gene = dict(zip(df["filename"], df["gene_symbol"]))
    fn2name = dict(zip(df["filename"], df["tf_name"]))
    dstruct = np.load("results/deeppbs_blind_benchmark/struct_preds.npz", allow_pickle=True)
    # map our filename -> DeepPBS struct name by PDB_chain prefix
    dnames = sorted(set(k.rsplit("::", 1)[0] for k in dstruct.files))
    pdbc2name = {}
    for nm in dnames:
        pc = "_".join(nm.split("_")[:2])
        pdbc2name.setdefault(pc, nm)

    def dpbs_pwm_for(fn):
        pc = "_".join(fn.split("_")[:2])
        nm = pdbc2name.get(pc)
        if nm is None:
            return None, 0
        p = dstruct[nm + "::pred"]              # (L,4)
        arr = np.full((4, MAX_L), 0.25, np.float32)
        L = min(p.shape[0], MAX_L); arr[:, :L] = np.clip(p[:L].T, 1e-8, 1)
        arr = arr / arr.sum(0, keepdims=True)
        return arr, L

    # ── paged plot ───────────────────────────────────────────────────────────
    per_page = 8
    n = len(fns)
    out = a.out
    with PdfPages(out) as pdf:
        for start in range(0, n, per_page):
            chunk = list(range(start, min(start + per_page, n)))
            fig, axes = plt.subplots(len(chunk), 3, figsize=(11, 1.5 * len(chunk)),
                                     squeeze=False)
            for row, i in enumerate(chunk):
                fn = fns[i]; gene = fn2gene.get(fn, ""); name = fn2name.get(fn, fn)
                L = int(masks[i].sum())
                # ground truth
                draw_logo(axes[row][0], targs[i], L, f"{name}  (truth)", "#2E7D32")
                # v14 (strand-oriented)
                rv = percol_r(preds[i], targs[i], L)
                draw_logo(axes[row][1], preds[i], L, f"{a.v14_label}  r={rv:.2f}", "#1565C0")
                # DeepPBS (per-structure)
                dp, dL = dpbs_pwm_for(fn)
                if dp is not None and dL > 0:
                    Lc = min(L, dL)
                    rd = percol_r(dp, targs[i], Lc)
                    draw_logo(axes[row][2], dp, dL, f"DeepPBS  r={rd:.2f}", "#C62828")
                else:
                    axes[row][2].text(0.5, 0.5, "no DeepPBS", ha="center", va="center",
                                      fontsize=7, color="#999"); axes[row][2].axis("off")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}  ({n} TFs, {(n + per_page - 1)//per_page} pages)")


if __name__ == "__main__":
    main()
