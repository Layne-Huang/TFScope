"""AF3+Rosetta calibration case study — logo comparison (generic, RC-aware),
styled to match figures/logo_comparison.pdf: vertical stack of 3 plain logos
(GT / TFScope / AF3+Rosetta calibrated), open axes (no top/right spines).

Usage: python scripts/plot_calibration_logo_generic.py --gene E2F4
"""
import json, argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

BASES = list("ACGT")

ap = argparse.ArgumentParser()
ap.add_argument("--gene", required=True)
args = ap.parse_args()
gene = args.gene
ROOT = f"results/calibration_case_study/{gene}"

seed = json.load(open(f"{ROOT}/seed_pwm.json"))
cal = json.load(open(f"{ROOT}/rosetta_scan/calibrated_pwm.json"))

pwm = np.array(seed["pwm"])
lo, hi, flank = seed["core_lo"], seed["core_hi"], seed["flank"]
lo_f, hi_f = max(0, lo - flank), min(pwm.shape[1] - 1, hi + flank)
window_fwd = pwm[:, lo_f:hi_f + 1]
core_lo_local_fwd, core_hi_local_fwd = lo - lo_f, hi - lo_f

rc, shift = seed["align_rc"], seed["align_shift"]
if rc:
    window = window_fwd[::-1, ::-1]
    W = window_fwd.shape[1]
    core_lo_local = W - 1 - core_hi_local_fwd
else:
    window = window_fwd
    core_lo_local = core_lo_local_fwd

gt_pwm = np.array(seed["gt_pwm"])
K = gt_pwm.shape[1]

# crop the aligned window to the gt-comparable frame: window index i -> gt index (i - core_lo_local) - shift
crop_lo = core_lo_local + shift
tfscope_crop = np.zeros((4, K))
for j in range(K):
    i = crop_lo + j
    if 0 <= i < window.shape[1]:
        tfscope_crop[:, j] = window[:, i]

calibrated_crop = tfscope_crop.copy()
for pos_str, dist in cal["calibrated_pwm_columns"].items():
    i = int(pos_str)
    j = i - crop_lo
    if 0 <= j < K:
        calibrated_crop[:, j] = [dist[b] for b in BASES]


def pearson_r(pred, gt):
    return np.corrcoef(pred.ravel(), gt.ravel())[0, 1]

r_tfscope = pearson_r(tfscope_crop, gt_pwm)
r_calibrated = pearson_r(calibrated_crop, gt_pwm)

fig, axes = plt.subplots(3, 1, figsize=(3.2, 6.5))
panels = [
    (gt_pwm, f"GT {gene} (K={K})"),
    (tfscope_crop, f"TFScope  r={r_tfscope:.2f}"),
    (calibrated_crop, f"AF3+Rosetta  r={r_calibrated:.2f}"),
]
for ax, (mat, title) in zip(axes, panels):
    df = pd.DataFrame(mat.T, columns=BASES)
    df.index = range(1, K + 1)
    logo = logomaker.Logo(df, ax=ax, color_scheme="classic")
    logo.style_spines(spines=["top", "right"], visible=False)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(1, K + 1))
    ax.set_ylim(0, 1.5)

fig.tight_layout()
out_dir = f"figures/calibration_case_study"
os.makedirs(out_dir, exist_ok=True)
out = f"{out_dir}/{gene.lower()}_calibration_logo"
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
print(f"{gene}: r_tfscope={r_tfscope:.3f} -> r_calibrated={r_calibrated:.3f}  ({out}.png)")
