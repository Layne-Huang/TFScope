"""Fig 4a — directional specificity-switch score for MyoD1 L122R (combined no-RAG model).

Instead of comparing consensus strings (misleading: argmax can flip to a spurious base), we score
the two competing E-boxes under the WT and mutant PREDICTED PWMs and form a difference-in-differences:

    S(seq | PWM)  = best PWM log-odds (background 0.25), max over offsets and both strands
    Δ_switch      = [S_mut(CACGTG) - S_mut(CACCTG)] - [S_WT(CACGTG) - S_WT(CACCTG)]

    Δ_switch > 0  : L122R pushes preference toward the MYC-like CACGTG (switch reproduced)
    Δ_switch <= 0 : expected switch NOT reproduced

Panel: (left) WT and L122R predicted E-box logos; (right) S(CACGTG)/S(CACCTG) bars for WT vs mutant,
annotated with Δ_switch. Out: figures/figure4a_switch/figure4a_switch.{png,pdf,svg};
results/myod1_mut/switch_score_tfscope.json
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CK = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
OUTD = "figures/figure4a_switch"; os.makedirs(OUTD, exist_ok=True); os.makedirs("results/myod1_mut", exist_ok=True)
WT_DBD = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"   # MyoD1 bHLH DBD (...RERRRL...)
MUT_DBD = WT_DBD[:11] + "R" + WT_DBD[12:]                        # L122R / L112R(DBD): ...RERRRR...
assert WT_DBD[11] == "L" and MUT_DBD[11] == "R"
FID = 3  # bHLH

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.dirname(CK) + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to("cuda:0").eval()
m.load_state_dict(torch.load(CK, map_location="cuda:0", weights_only=False)["model"], strict=False)

@torch.no_grad()
def predict(seq, fid=FID):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device="cuda:0")
    dm = torch.ones(1, len(seq), dtype=torch.bool, device="cuda:0"); fi = torch.tensor([fid], device="cuda:0")
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return torch.sigmoid(gl)[0].cpu().numpy(), F.softmax(pl, 1)[0].cpu().numpy()

B = {"A": 0, "C": 1, "G": 2, "T": 3}
def rc(s): return s[::-1].translate(str.maketrans("ACGT", "TGCA"))
def score(P, seq):
    lo = np.log2(np.clip(P, 1e-6, 1) / 0.25); W = P.shape[1]; L = len(seq); best = -1e9
    for s in (seq, rc(seq)):
        idx = [B[c] for c in s]
        for off in range(0, W - L + 1):
            best = max(best, float(sum(lo[idx[j], off + j] for j in range(L))))
    return best
def core(p, g):
    c = np.where(g > 0.5)[0]
    if len(c) < 4:
        ic = (p * np.log2(p + 1e-9)).sum(0) + 2; a = ic.argmax(); c = np.arange(max(0, a - 4), min(p.shape[1], a + 5))
    return c.min(), c.max() + 1

gw, pw = predict(WT_DBD); gm, pm = predict(MUT_DBD)
lo, hi = core(pw, gw); Wwt, Wmt = pw[:, lo:hi], pm[:, lo:hi]
S = {"WT": {"CACGTG": score(pw, "CACGTG"), "CACCTG": score(pw, "CACCTG")},
     "mut": {"CACGTG": score(pm, "CACGTG"), "CACCTG": score(pm, "CACCTG")}}
dWT = S["WT"]["CACGTG"] - S["WT"]["CACCTG"]
dMUT = S["mut"]["CACGTG"] - S["mut"]["CACCTG"]
d_switch = dMUT - dWT
print(f"WT : S(CACGTG)={S['WT']['CACGTG']:.2f} S(CACCTG)={S['WT']['CACCTG']:.2f}  d={dWT:+.2f}")
print(f"mut: S(CACGTG)={S['mut']['CACGTG']:.2f} S(CACCTG)={S['mut']['CACCTG']:.2f}  d={dMUT:+.2f}")
print(f"Δ_switch = {d_switch:+.2f}  -> {'switch reproduced (>0)' if d_switch > 0 else 'NOT reproduced'}")
json.dump({"WT": {k: round(float(v), 3) for k, v in S["WT"].items()},
           "mut": {k: round(float(v), 3) for k, v in S["mut"].items()},
           "delta_switch": round(float(d_switch), 3)},
          open("results/myod1_mut/switch_score_tfscope.json", "w"), indent=1)

# ── figure ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker, pandas as pd
plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.7})
cons = lambda P: "".join("ACGT"[i] for i in P.argmax(0))
def logo(ax, P, title, ct="black"):
    Pn = np.clip(P, 1e-9, 1); Pn = Pn / Pn.sum(0, keepdims=True)
    ic = np.maximum(2 + (Pn * np.log2(Pn)).sum(0), 0)
    logomaker.Logo(pd.DataFrame((Pn * ic).T, columns=list("ACGT")), ax=ax, color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([0, 2]); ax.set_ylim(0, 2); ax.tick_params(length=2)
    ax.set_title(title, fontsize=8, color=ct, loc="left", pad=2)

fig = plt.figure(figsize=(8.2, 4.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1, 1], hspace=0.65, wspace=0.32)
logo(fig.add_subplot(gs[0, 0]), Wwt, f"MyoD1 WT — predicted ({cons(Wwt)})", ct="#333")
logo(fig.add_subplot(gs[1, 0]), Wmt, f"MyoD1 L122R — predicted ({cons(Wmt)})", ct="#c0392b")

axb = fig.add_subplot(gs[:, 1])
groups = ["CACGTG\n(MYC-like)", "CACCTG\n(WT E-box)"]
x = np.arange(2); w = 0.36
wt_vals = [S["WT"]["CACGTG"], S["WT"]["CACCTG"]]
mt_vals = [S["mut"]["CACGTG"], S["mut"]["CACCTG"]]
axb.bar(x - w/2, wt_vals, w, label="WT", color="#7f8c9b", edgecolor="k", lw=0.5)
axb.bar(x + w/2, mt_vals, w, label="L122R", color="#d9544d", edgecolor="k", lw=0.5)
axb.axhline(0, color="#888", lw=0.6)
axb.set_xticks(x); axb.set_xticklabels(groups, fontsize=8)
axb.set_ylabel("PWM log-odds score  S(E-box | predicted PWM)", fontsize=8)
axb.set_title("Directional specificity-switch score", fontsize=9, fontweight="bold", loc="left", pad=8)
for s in ["top", "right"]: axb.spines[s].set_visible(False)
axb.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.80))
# annotate the CACGTG gain (WT -> mut) — the switch
ya, yb = wt_vals[0], mt_vals[0]
axb.annotate("", xy=(x[0] + w/2, yb), xytext=(x[0] - w/2, ya),
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
axb.text(x[0], max(ya, yb) + 1.0, f"+{mt_vals[0]-wt_vals[0]:.1f}\non CACGTG", fontsize=6.8, color="#c0392b", ha="center")
top = max(wt_vals + mt_vals)
axb.text(0.5, top + 3.0,
         r"$\Delta_{\mathrm{switch}}=[S_{mut}^{CACGTG}\!-\!S_{mut}^{CACCTG}]-[S_{WT}^{CACGTG}\!-\!S_{WT}^{CACCTG}]$",
         fontsize=7.2, ha="center", transform=axb.transData)
axb.set_ylim(min(0, min(wt_vals + mt_vals) - 1), top + 4.5)

fig.suptitle("L122R shifts MyoD1's predicted preference toward the MYC-like E-box (CACGTG)",
             fontsize=10, fontweight="bold", y=1.0)
out = f"{OUTD}/figure4a_switch"
for e in ["pdf", "svg"]: fig.savefig(f"{out}.{e}", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
print(f"saved {out}.{{png,pdf,svg}}")
