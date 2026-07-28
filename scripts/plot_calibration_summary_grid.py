"""AF3+Rosetta calibration case study — combined summary grid across all tested
genes: one row per gene (GT / TFScope / AF3+Rosetta calibrated), matching the
style of figures/logo_comparison.pdf (rows=TFs, cols=methods).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

BASES = list("ACGT")
GENES = ["PBX1", "E2F4", "SPI1", "MYB", "TBX1", "TBX3", "TBXT", "TBX21", "TBX5",
         "exd", "Nkx2-5", "GATA1", "ZBTB7A"]


def load_gene(gene):
    root = "results/pbx1_case_study" if gene == "PBX1" else f"results/calibration_case_study/{gene}"
    seed = json.load(open(f"{root}/seed_pwm.json"))
    cal = json.load(open(f"{root}/rosetta_scan/calibrated_pwm.json"))

    pwm = np.array(seed["pwm"])
    lo, hi, flank = seed["core_lo"], seed["core_hi"], seed.get("flank", 3)
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

    return dict(gt=gt_pwm, tfscope=tfscope_crop, calibrated=calibrated_crop, K=K,
                r_tfscope=pearson_r(tfscope_crop, gt_pwm),
                r_calibrated=pearson_r(calibrated_crop, gt_pwm))


data = {g: load_gene(g) for g in GENES}

fig, axes = plt.subplots(len(GENES), 3, figsize=(9.5, 2.2 * len(GENES)))
for row, gene in enumerate(GENES):
    d = data[gene]
    K = d["K"]
    panels = [
        (d["gt"], f"GT {gene} (K={K})"),
        (d["tfscope"], f"TFScope  r={d['r_tfscope']:.2f}"),
        (d["calibrated"], f"AF3+Rosetta  r={d['r_calibrated']:.2f}"),
    ]
    for col, (mat, title) in enumerate(panels):
        ax = axes[row, col]
        df = pd.DataFrame(mat.T, columns=BASES)
        df.index = range(1, K + 1)
        logo = logomaker.Logo(df, ax=ax, color_scheme="classic")
        logo.style_spines(spines=["top", "right"], visible=False)
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks(range(1, K + 1))
        ax.set_ylim(0, 1.5)

fig.tight_layout()
out = "figures/calibration_case_study/summary_grid"
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{out}.pdf", bbox_inches="tight")
print(f"Wrote {out}.png / .pdf")
print("\nSummary (r: TFScope -> AF3+Rosetta calibrated):")
for g in GENES:
    d = data[g]
    delta = d["r_calibrated"] - d["r_tfscope"]
    verdict = "WIN" if delta > 0.03 else ("LOSS" if delta < -0.03 else "flat")
    print(f"  {g:6s}  {d['r_tfscope']:.3f} -> {d['r_calibrated']:.3f}  (Δ={delta:+.3f})  [{verdict}]")
