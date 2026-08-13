#!/usr/bin/env python
"""v23 N-chain case gallery: predicted PWM logo (left) vs ground-truth (right)
for ~10 representative test TFs, spanning the multimer families v23 targets
(p53 tetramer, POU) plus the standard families. Predicted columns are the
gate-active columns aligned into the GT core frame (shift+RC).
"""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd, logomaker

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from eval_full_metrics import trimmed_core
from torch.utils.data import DataLoader

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v23_nchain/nchain_v23_seed42"
DATA = "data/processed/tf_pwm_training_v23.parquet"
SPLIT = "data/processed/splits/train_v22/split.json"
OUT = "figures/case_logos/v23_case_logos.pdf"
BASES = ["A", "C", "G", "T"]
IC, MAXSHIFT, MINPOS = 0.25, 10, 4
dev = "cuda" if torch.cuda.is_available() else "cpu"

# representative genes (family shown for the banner); multimers first
WANT = [
    ("TP53", "p53 (tetramer)"), ("P53", "p53 (tetramer)"),
    ("POU5F1", "POU"), ("POU2F1", "POU"),
    ("NFE2L2", "bZIP (het/MAF)"), ("GCN4", "bZIP (homo)"),
    ("THRB", "Nuclear receptor"), ("CLOCK", "bHLH-PAS"),
    ("FOXO1", "Forkhead"), ("ETS1", "ETS"),
]


def pwm_to_ic_df(pwm):
    p = np.clip(pwm, 1e-8, 1.0); ic = 2.0 + (p * np.log2(p)).sum(0)
    return pd.DataFrame((p * ic[None, :]).T, columns=BASES)


def draw_logo(ax, pwm, title):
    logomaker.Logo(pwm_to_ic_df(pwm), ax=ax, color_scheme="classic", show_spines=False)
    ax.set_ylim(0, 2); ax.set_yticks([0, 1, 2]); ax.set_ylabel("bits", fontsize=8)
    ax.set_xticks([]); ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)


def _trimg(pwm, th=IC):
    p = np.clip(pwm, 1e-8, 1); ic = 2 + (p * np.log2(p)).sum(0); inf = np.where(ic >= th)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]


def main():
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(CKPT, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(CKPT, "ckpt_best.pt"), map_location=dev,
                                 weights_only=False)["model"], strict=False)
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    genes = [str(g).upper() for g in ds.df["gene_symbol"].values]

    # best-covR representative row per requested gene
    per_gene = {}
    with torch.no_grad():
        ld = DataLoader(ds, batch_size=12, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
        idx = 0
        for b in ld:
            bb = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long) for k, v in b.items()}
            gl, pl, _ = m(bb['sequence_tokens'], bb['dbd_mask'], bb['family_id'], recog_prior=bb.get('recog_prior'))
            pp = F.softmax(pl, 1).cpu().numpy(); gp = torch.sigmoid(gl).cpu().numpy()
            tg = bb['target_pwm'].cpu().numpy(); mk = bb['pwm_mask'].cpu().numpy()
            for j in range(pp.shape[0]):
                g = genes[idx]; idx += 1
                if g not in [w[0] for w in WANT]: continue
                core = trimmed_core(tg[j], mk[j], IC)
                if core is None or core.shape[1] < MINPOS: continue
                gate = gp[j]; act = gate > 0.5
                if not act.any(): act = gate > gate.max() * 0.5
                pc = pp[j][:, act]; gc = _trimg(tg[j][:, mk[j].astype(bool)])
                if pc.shape[1] == 0 or gc.shape[1] == 0: continue
                al, sh, ori, r = align_pwm(pc, gc, max_shift=MAXSHIFT, consider_revcomp=True)
                nov = sum(1 for i in range(pc.shape[1]) if 0 <= i + sh < gc.shape[1]); cov = nov / gc.shape[1]
                rec = dict(gene=g, pred=al, gt=gc, r=r, cov=cov, covr=r * cov,
                           lp=pc.shape[1], lg=gc.shape[1], orient=ori)
                if g not in per_gene or rec["covr"] > per_gene[g]["covr"]:
                    per_gene[g] = rec

    # assemble in WANT order (dedup p53: TP53 or P53 whichever present)
    cases, seen_fam = [], set()
    for g, fam in WANT:
        if g in per_gene and (fam, g) not in seen_fam:
            c = per_gene[g]; c["fam"] = fam; cases.append(c)
            if fam.startswith("p53"): seen_fam.add((fam, "TP53")); seen_fam.add((fam, "P53"))
    cases = cases[:10]

    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(11, 1.55 * n))
    fig.suptitle("v23 N-chain: predicted vs ground-truth PWM logos (held-out test)",
                 fontsize=13, fontweight="bold", y=0.998)
    for i, c in enumerate(cases):
        rc = "  (RC)" if c["orient"] == "rc" else ""
        banner = (f"{c['gene']}  ·  {c['fam']}   |   r={c['r']:.2f}   cov={c['cov']:.2f}   "
                  f"covR={c['covr']:.2f}   len {c['lp']}/{c['lg']}{rc}")
        draw_logo(axes[i, 0], c["pred"], "predicted")
        draw_logo(axes[i, 1], c["gt"], "ground truth")
        axes[i, 0].annotate(banner, xy=(0, 1.32), xycoords="axes fraction", fontsize=9.5,
                            fontweight="bold", ha="left", va="bottom", annotation_clip=False)
    fig.tight_layout(rect=[0, 0.01, 1, 0.975], h_pad=2.4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with PdfPages(OUT) as pdf:
        pdf.savefig(fig, dpi=200)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=170)
    print("cases:", [(c["gene"], round(c["covr"], 2), f"{c['lp']}/{c['lg']}") for c in cases])
    print(f"saved {OUT} (+ .png)")


if __name__ == "__main__":
    main()
