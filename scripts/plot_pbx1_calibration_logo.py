"""PBX1 AF3+Rosetta calibration case study — logo comparison (GT / TFScope / AF3+Rosetta
calibrated), styled to match figures/logo_comparison.pdf: one row, 3 plain logos,
"GT <name> (K=..)" / "<method> r=X.XX" titles, no shading or subtitle blocks.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

BASES = list("ACGT")
seed = json.load(open("results/pbx1_case_study/seed_pwm.json"))
cal = json.load(open("results/pbx1_case_study/rosetta_scan/calibrated_pwm.json"))

pwm_full = np.array(seed["pwm"])  # (4, 20) full predicted canvas, ACGT rows
gt_pwm = np.array(seed["gt_pwm"])  # (4, 7)
K = gt_pwm.shape[1]

window = pwm_full[:, :K]  # crop TFScope's window to GT's length K, same alignment (shift=0)
calibrated = window.copy()
for pos_str, dist in cal["calibrated_pwm_columns"].items():
    pos = int(pos_str)
    if pos < K:
        calibrated[:, pos] = [dist[b] for b in BASES]


def pearson_r(pred, gt):
    return np.corrcoef(pred.ravel(), gt.ravel())[0, 1]

r_tfscope = pearson_r(window, gt_pwm)
r_calibrated = pearson_r(calibrated, gt_pwm)

fig, axes = plt.subplots(3, 1, figsize=(3.2, 6.5))
panels = [
    (gt_pwm, f"GT PBX1 (K={K})"),
    (window, f"TFScope  r={r_tfscope:.2f}"),
    (calibrated, f"AF3+Rosetta  r={r_calibrated:.2f}"),
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
out = "figures/pbx1_case_study/pbx1_calibration_logo"
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
print(f"Wrote {out}.png / .pdf  (r_tfscope={r_tfscope:.3f}, r_calibrated={r_calibrated:.3f})")
