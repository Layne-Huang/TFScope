#!/usr/bin/env python
"""TFScope-RAG logo plots on the cluster40 test set, FIXED frame (no offset
override) but reverse-complement allowed.

Per spec:
  - NO registration override: prediction is left-anchored in its own frame,
    we do NOT slide it +/- to best-match the target.
  - RC allowed: for each TF we try forward and reverse-complement (no shift)
    and keep whichever orientation has the higher per-column Pearson r to the
    target core. (Orientation is a symmetry, not a registration cheat.)

Each row: [ Ground truth core ]  [ TFScope-RAG pred (chosen orientation) ].
Sorted best-r first. Multi-page PDF + a top-12 PNG preview.

Usage:
  python scripts/plot_rag_logos_norc_override.py [ckpt_dir] [split.json] [--n-png 12]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logomaker
from scipy.stats import pearsonr
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from torch.utils.data import DataLoader

CB = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
ap = argparse.ArgumentParser()
ap.add_argument("ckpt_dir", nargs="?", default=f"{CB}/cluster40_v18a_rag")
ap.add_argument("split", nargs="?", default="data/processed/splits/cluster40/split.json")
ap.add_argument("--data", default="data/processed/tf_pwm_aug_dbd_canon_trim.parquet")
ap.add_argument("--ckpt-name", default="ckpt_best.pt")
ap.add_argument("--out", default="figures/rag_logos_cluster40_norc_override.pdf")
ap.add_argument("--png", default="figures/rag_logos_cluster40_top12.png")
ap.add_argument("--n-png", type=int, default=12)
ap.add_argument("--per-page", type=int, default=6)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
RC_ROWS = [3, 2, 1, 0]
IC_THRESH = 0.25
MIN_POS = 4          # rule: motifs shorter than 4 informative positions are not allowed


def ic_cols(pwm):                              # per-col IC bits, pwm (4,L)
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)

def trim_core(pwm):                            # left-anchor informative core
    ic = ic_cols(pwm); inf = np.where(ic >= IC_THRESH)[0]
    if len(inf) == 0: return None
    c = pwm[:, inf[0]:inf[-1] + 1]
    return c / c.sum(0, keepdims=True)

def revcomp(pwm):                              # (4,L)
    return pwm[RC_ROWS][:, ::-1]

def fixed_r(pred, core):                       # per-col r, NO shift, min overlap
    L = min(pred.shape[1], core.shape[1])
    rs = []
    for j in range(L):
        a, b = core[:, j], pred[:, j]
        if a.std() < 1e-8 or b.std() < 1e-8: rs.append(0.0)
        else: rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)) if rs else float("nan")

def ppm_to_ic_df(ppm_L4):                      # (L,4) -> IC-scaled df
    p = np.clip(ppm_L4, 1e-10, 1.0)
    icp = (p * np.log2(p / 0.25)).sum(1)
    return pd.DataFrame(p * icp[:, None], columns=["A", "C", "G", "T"])

def draw(ax, ppm_4L, title, r=None):
    if ppm_4L is None:
        ax.text(.5, .5, "N/A", ha="center", va="center", transform=ax.transAxes); ax.axis("off"); return
    df = ppm_to_ic_df(ppm_4L.T)
    try: logomaker.Logo(df, ax=ax, color_scheme="classic", show_spines=False, baseline_width=0.5)
    except Exception: ax.text(.5, .5, "err", color="red", ha="center", transform=ax.transAxes)
    ax.set_ylim(0, 2); ax.set_xticks(range(df.shape[0]))
    ax.set_xticklabels(range(1, df.shape[0] + 1), fontsize=5); ax.yaxis.set_tick_params(labelsize=6)
    if r is not None and not np.isnan(r): title += f"   r={r:.2f}"
    ax.set_title(title, fontsize=8, pad=3)


# ── infer ──────────────────────────────────────────────────────────────────
cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(args.ckpt_dir, "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: setattr(cfg, k, v)
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(os.path.join(args.ckpt_dir, args.ckpt_name), map_location=dev,
                             weights_only=False)["model"], strict=False)
ds = TFDataset(cfg, args.data, args.split, split="test", max_seq_len=1024)
# recover filenames in dataset order
fns = ds.filenames if hasattr(ds, "filenames") else [None] * len(ds)
ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_variable_length)

entries = []   # (name, core(4,L), pred_oriented(4,L), r, orient)
idx = 0; n_skip = 0
with torch.no_grad():
    for b in ld:
        b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long) for k, v in b.items()}
        _, pwm_logits, _ = m(b['sequence_tokens'], b['dbd_mask'], b['family_id'],
                             retrieved_pwms=b.get('retrieved_pwms'), retrieved_masks=b.get('retrieved_masks'),
                             retrieved_sims=b.get('retrieved_sims'), recog_prior=b.get('recog_prior'))
        prob = F.softmax(pwm_logits, dim=1).cpu().numpy()
        tgt = b['target_pwm'].cpu().numpy(); msk = b['pwm_mask'].cpu().numpy()
        for pred, t, mk in zip(prob, tgt, msk):
            name = fns[idx] if idx < len(fns) else f"tf{idx}"; idx += 1
            core = trim_core(t[:, mk.astype(bool)])
            if core is None or core.shape[1] < MIN_POS: n_skip += 1; continue   # rule: >= 4 positions
            pm = np.clip(pred[:, mk.astype(bool)], 1e-8, 1.0); pm = pm / pm.sum(0, keepdims=True)
            pcore = trim_core(pm)                       # left-anchor pred's own informative core
            if pcore is None or pcore.shape[1] < MIN_POS: pcore = pm
            r_f = fixed_r(pcore, core)
            r_r = fixed_r(revcomp(pcore), core)
            if r_r > r_f: pred_or, r, orient = revcomp(pcore), r_r, "RC"
            else:          pred_or, r, orient = pcore, r_f, "fwd"
            entries.append((str(name), core, pred_or, r, orient))

entries.sort(key=lambda e: (-1 if np.isnan(e[3]) else 0, -(e[3] if not np.isnan(e[3]) else -9)))
rs = np.array([e[3] for e in entries], float)
print(f"n={len(entries)}  skipped(<{MIN_POS}pos)={n_skip}  fixed+RC r: mean={np.nanmean(rs):.3f} median={np.nanmedian(rs):.3f}  RC-chosen={sum(e[4]=='RC' for e in entries)}/{len(entries)}", flush=True)

# ── top-N PNG preview ────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
N = min(args.n_png, len(entries))
fig, axes = plt.subplots(N, 2, figsize=(9, 1.5 * N))
if N == 1: axes = axes[None, :]
for i in range(N):
    name, core, pred_or, r, orient = entries[i]
    short = name.replace(".txt", "")[:34]
    draw(axes[i, 0], core, f"GT  {short}")
    draw(axes[i, 1], pred_or, f"RAG ({orient})", r)
axes[0, 0].annotate("Ground truth", (0.5, 1.35), xycoords="axes fraction", ha="center", fontsize=10, weight="bold")
axes[0, 1].annotate("TFScope-RAG (no override, RC allowed)", (0.5, 1.35), xycoords="axes fraction", ha="center", fontsize=10, weight="bold")
fig.tight_layout(); fig.savefig(args.png, dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote", args.png)

# ── full multi-page PDF ──────────────────────────────────────────────────────
PP = args.per_page
with PdfPages(args.out) as pdf:
    for p0 in range(0, len(entries), PP):
        chunk = entries[p0:p0 + PP]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(9, 1.5 * len(chunk)))
        if len(chunk) == 1: axes = axes[None, :]
        for i, (name, core, pred_or, r, orient) in enumerate(chunk):
            short = name.replace(".txt", "")[:34]
            draw(axes[i, 0], core, f"GT  {short}")
            draw(axes[i, 1], pred_or, f"RAG ({orient})", r)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
print("wrote", args.out)
