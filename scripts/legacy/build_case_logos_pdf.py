#!/usr/bin/env python
"""Two-column case gallery: predicted PWM logo (left) vs ground-truth PWM logo
(right) for 10 classical TFs from the held-out test set, with per-case metrics.

Predicted columns are the gate-active columns, aligned into the GT informative
core frame (shift + RC) via models.alignment.align_pwm, so the two logos line
up column-for-column. Logos are information-content (bits) scaled.

Usage:
  python scripts/build_case_logos_pdf.py <ckpt_dir> <split.json> <data.parquet> [out.pdf]
"""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
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

CKPT_DIR, SPLIT, DATA = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else "figures/case_logos/test_case_logos.pdf"
IC_THRESH, MAX_SHIFT, MIN_POS = 0.25, 10, 4
BASES = ["A", "C", "G", "T"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

# classical, widely-recognised TFs spanning ETS / Forkhead / POU / p53 / bZIP / NR.
# label = display family (test rows are all tagged "Other"; these are the real ones)
WANT = [
    ("ETS1",   "ETS"),
    ("FLI1",   "ETS"),
    ("GABPA",  "ETS"),
    ("FOXA1",  "Forkhead"),
    ("FOXO1",  "Forkhead"),
    ("FOXP3",  "Forkhead"),
    ("POU5F1", "POU/OCT4"),
    ("TP53",   "p53"),
    ("GCN4",   "bZIP"),
    ("THRB",   "Nuclear receptor"),
]


def _ic(p):
    p = np.clip(p, 1e-8, 1.0); return 2.0 + (p * np.log2(p)).sum(0)


def _trim_core_gate(pwm, thresh=IC_THRESH):
    ic = _ic(pwm); inf = np.where(ic >= thresh)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]


def pwm_to_ic_df(pwm):
    """(4, L) probabilities -> logomaker df (L x ACGT) scaled to bits."""
    p = np.clip(pwm, 1e-8, 1.0)
    ic = 2.0 + (p * np.log2(p)).sum(0)          # bits per column
    heights = (p * ic[None, :]).T                # (L, 4)
    return pd.DataFrame(heights, columns=BASES)


def draw_logo(ax, pwm, title):
    df = pwm_to_ic_df(pwm)
    logomaker.Logo(df, ax=ax, color_scheme="classic", show_spines=False)
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel("bits", fontsize=8)
    ax.set_xticks([])
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def build_model():
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(CKPT_DIR, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(dev).eval()
    sd = torch.load(os.path.join(CKPT_DIR, "ckpt_best.pt"), map_location=dev,
                    weights_only=False)["model"]
    m.load_state_dict(sd, strict=False)
    return m, cfg


def main():
    m, cfg = build_model()
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    # map dataset index -> gene, using the underlying dataframe rows for this split
    genes = [str(g).upper() for g in ds.df["gene_symbol"].values]
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                    collate_fn=collate_variable_length)

    # collect one representative record per gene: the row with the LONGEST GT
    # core (most complete motif to display) -- deterministic, model-independent.
    per_gene = {}   # gene -> best record dict
    idx = 0
    with torch.no_grad():
        for b in ld:
            bb = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long)
                  for k, v in b.items()}
            gl, pl, _ = m(bb['sequence_tokens'], bb['dbd_mask'], bb['family_id'],
                          retrieved_pwms=bb.get('retrieved_pwms'),
                          retrieved_masks=bb.get('retrieved_masks'),
                          retrieved_sims=bb.get('retrieved_sims'),
                          recog_prior=bb.get('recog_prior'))
            pwm_prob = F.softmax(pl, dim=1).cpu().numpy()
            gate_prob = torch.sigmoid(gl).cpu().numpy()
            target = bb['target_pwm'].cpu().numpy(); mask = bb['pwm_mask'].cpu().numpy()
            for j in range(pwm_prob.shape[0]):
                gene = genes[idx]; idx += 1
                core = trimmed_core(target[j], mask[j], IC_THRESH)
                if core is None or core.shape[1] < MIN_POS:
                    continue
                rec = dict(gene=gene, pred=pwm_prob[j], gate=gate_prob[j],
                           tgt=target[j], msk=mask[j], core_len=core.shape[1])
                if gene not in per_gene or rec["core_len"] > per_gene[gene]["core_len"]:
                    per_gene[gene] = rec

    # assemble the 10 classical cases (skip any missing, warn)
    cases = []
    for gene, fam in WANT:
        if gene not in per_gene:
            print(f"  WARNING: {gene} not found / no valid core in test -- skipped")
            continue
        r = per_gene[gene]
        gate = r["gate"]; pred = r["pred"]; tgt = r["tgt"]; msk = r["msk"]
        active = gate > 0.5
        if not active.any(): active = gate > gate.max() * 0.5
        pred_core = pred[:, active]
        gt_core = _trim_core_gate(tgt[:, msk.astype(bool)])
        aligned, shift, orient, rr = align_pwm(pred_core, gt_core, max_shift=MAX_SHIFT,
                                               consider_revcomp=True)
        wp, lr = pred_core.shape[1], gt_core.shape[1]
        n_ov = sum(1 for i in range(wp) if 0 <= i + shift < lr)
        cov = n_ov / lr if lr > 0 else 0.0
        cases.append(dict(gene=gene, fam=fam, pred_aligned=aligned, gt=gt_core,
                          legacy_r=rr, cov=cov, covr=rr * cov,
                          len_pred=wp, len_gt=lr, orient=orient))
        print(f"  {gene:8s} [{fam:16s}] r={rr:.3f} cov={cov:.3f} covR={rr*cov:.3f} "
              f"len {wp}/{lr} orient={orient}")

    # ---- render: one row per case, two logo columns ----
    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(11, 1.55 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle("TFScope predicted vs. ground-truth PWM logos "
                 "(held-out test set, 10 classical TFs)",
                 fontsize=13, fontweight="bold", y=0.997)
    for i, c in enumerate(cases):
        rc = "  (RC)" if c["orient"] == "rc" else ""
        left_title = (f"{c['gene']}  ·  {c['fam']}   |   r={c['legacy_r']:.2f}   "
                      f"cov={c['cov']:.2f}   covR={c['covr']:.2f}   "
                      f"len {c['len_pred']}/{c['len_gt']}{rc}")
        draw_logo(axes[i, 0], c["pred_aligned"], "predicted")
        draw_logo(axes[i, 1], c["gt"], "ground truth")
        # gene + metrics banner spanning both columns, above the pair
        axes[i, 0].annotate(left_title, xy=(0, 1.32), xycoords="axes fraction",
                            fontsize=9.5, fontweight="bold", ha="left", va="bottom",
                            annotation_clip=False)
    fig.tight_layout(rect=[0, 0.01, 1, 0.975], h_pad=2.4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with PdfPages(OUT) as pdf:
        pdf.savefig(fig, dpi=200)
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=170)
    plt.close(fig)
    print(f"\nsaved {OUT}  (+ .png)  |  {n} cases")


if __name__ == "__main__":
    main()
