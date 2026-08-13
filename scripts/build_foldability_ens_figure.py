"""Fig — an independent structure predictor (Boltz-2) judges the v24-ENSEMBLE's
sequence-only predicted consensus motifs as forming protein-DNA complexes at least
as confident as the structure-based DeepPBS predictions.

Reads results/af3_v24_foldability/boltz_foldability_ens.json (produced by
iclr.collect_boltz_foldability_ens) and renders a paired ipTM comparison
(per-TF connected dots + means + Δ histogram).

Out: figures_v24_ensemble/figure_foldability/figure_foldability.{png,pdf}
     figures_v24_ensemble/figure_foldability/foldability_ens.json (raw data copy)
"""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "results/af3_v24_foldability/boltz_foldability_ens.json"
OUTD = "figures_v24_ensemble/figure_foldability"
os.makedirs(OUTD, exist_ok=True)
TEAL, ORANGE = "#2a9d8f", "#e76f51"


def main():
    d = json.load(open(SRC))
    rows = d["per_tf"]
    ens = np.array([r["iptm_ens"] for r in rows])
    dpp = np.array([r["iptm_deeppbs"] for r in rows])
    delta = ens - dpp
    n = len(rows); wins = int((delta > 0).sum())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 4.2),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # --- panel A: paired dots (DeepPBS -> TFScope-ensemble), colored by winner ---
    for i in range(n):
        c = TEAL if delta[i] > 0 else ORANGE
        axA.plot([0, 1], [dpp[i], ens[i]], "-", color=c, alpha=0.35, lw=1.0, zorder=1)
    axA.scatter(np.zeros(n), dpp, s=26, color="#555", zorder=3, label="DeepPBS")
    axA.scatter(np.ones(n), ens, s=26, color=TEAL, zorder=3, label="TFScope (ens)")
    for x, v, col, dx, ha in [(0, dpp, "#555", -0.20, "right"), (1, ens, TEAL, 0.20, "left")]:
        axA.plot([x - 0.12, x + 0.12], [v.mean()] * 2, color=col, lw=3, zorder=4)
        axA.text(x + dx, v.mean(), f"{v.mean():.3f}", ha=ha, va="center",
                 fontsize=11, fontweight="bold", color=col, zorder=5)
    axA.set_xticks([0, 1]); axA.set_xticklabels(["DeepPBS\nconsensus", "TFScope-ens\nconsensus"])
    axA.set_ylabel("Boltz-2 complex ipTM"); axA.set_xlim(-0.35, 1.35)
    p = d.get("wilcoxon_p_iptm")
    ptxt = f"Wilcoxon p = {p:.1e}" if p is not None else ""
    axA.set_title(f"Sequence-only motifs fold as confidently\n"
                  f"ens {ens.mean():.3f} vs DeepPBS {dpp.mean():.3f}  "
                  f"(wins {wins}/{n}) {ptxt}", fontsize=9.5)

    # --- panel B: Δ-ipTM histogram ---
    axB.axvline(0, color="k", lw=1)
    axB.hist(delta, bins=16, color=TEAL, alpha=0.85, edgecolor="white")
    axB.axvline(delta.mean(), color=ORANGE, lw=2, label=f"mean Δ = {delta.mean():+.3f}")
    axB.set_xlabel("Δ ipTM  (TFScope-ens − DeepPBS)"); axB.set_ylabel("# TFs")
    axB.legend(fontsize=9, frameon=False)
    axB.set_title(f"{wins}/{n} TFs favor TFScope-ensemble", fontsize=9.5)

    fig.suptitle("v24 ensemble: sequence-only consensus vs structure-based DeepPBS, "
                 "judged by Boltz-2 foldability", fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUTD}/figure_foldability.{ext}", dpi=200, bbox_inches="tight")
    shutil.copy(SRC, f"{OUTD}/foldability_ens.json")
    print(f"saved {OUTD}/figure_foldability.png  (n={n}, ens {ens.mean():.3f} vs "
          f"DeepPBS {dpp.mean():.3f}, wins {wins}/{n})")


if __name__ == "__main__":
    main()
