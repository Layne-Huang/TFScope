"""Fig 4a — sequence localizes but does not resolve mutation-induced specificity switches.
For MyoD1 L112R and KLF4 K409Q (combined no-RAG model, same as Figs 1-3): predict WT and mutant
PWMs, show the per-position prediction change (L1) that LOCALIZES the affected motif position, and
annotate that the predicted new base differs from the experimentally-known switch base.
Out: figures/figure4a_localize/figure4a_localize.{png,pdf,svg}; results/myod1_mut/fig4a_data.json
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
OUTD = "figures/figure4a_localize"; os.makedirs(OUTD, exist_ok=True); os.makedirs("results/myod1_mut", exist_ok=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
# cases: MyoD1 L112R (bHLH basic-region point mutation) + ER->GR P-box swap (NR, textbook 3-residue
# specificity determinant). Both are well-documented switches where TFScope RESPONDS but does not RESOLVE.
import pandas as _pd
_d = _pd.read_parquet("data/processed/tf_pwm_aug_dbd.parquet")
def _dbd(g): r = _d[_d.gene_symbol == g].iloc[0]; return str(r.sequence)[int(r.dbd_start):int(r.dbd_end)], int(r.family_id)
MYOD1_WT = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
ESR1_WT, ESR1_FID = _dbd("ESR1")                          # ER DBD (binds ERE AGGTCA)
CASES = [
    dict(tf="MyoD1", mut="L112R", wt=MYOD1_WT, mt=MYOD1_WT[:11] + "R" + MYOD1_WT[12:], fid=3,
         known_wt="CACCTG", known="CACGTG", note="true switch C→G"),
    dict(tf="ER (ESR1)", mut="P-box→GR", wt=ESR1_WT, mt=ESR1_WT.replace("EGCKA", "GSCKV", 1), fid=ESR1_FID,
         known_wt="AGGTCA", known=None, note="responds but stays ERE, does not reach GRE"),
]

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.dirname(CKPT) + "/config.json")).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)

@torch.no_grad()
def predict(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev); fi = torch.tensor([fid], device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return torch.sigmoid(gl)[0].cpu().numpy(), F.softmax(pl, 1)[0].cpu().numpy()   # gate(W), pwm(4,W)

def core(pwm, gate):
    cols = np.where(gate > 0.5)[0]
    if len(cols) < 4:
        ic = (pwm * np.log2(pwm + 1e-9)).sum(0) + 2; a = ic.argmax(); cols = np.arange(max(0, a - 4), min(pwm.shape[1], a + 5))
    return cols.min(), cols.max() + 1

res = []
for c in CASES:
    gw, pw = predict(c["wt"], c["fid"]); gm, pm = predict(c["mt"], c["fid"])
    lo, hi = core(pw, gw)                                   # window from WT
    W_wt, W_mt = pw[:, lo:hi], pm[:, lo:hi]
    l1 = np.abs(W_wt - W_mt).sum(0)                         # per-position change
    cons = lambda P: "".join("ACGT"[i] for i in P.argmax(0))
    res.append(dict(c, wt_pwm=W_wt, mt_pwm=W_mt, l1=l1, wt_cons=cons(W_wt), mt_cons=cons(W_mt),
                    argmax_change=int(l1.argmax())))
    print(f"{c['tf']} {c['mut']}: WT={cons(W_wt)} MUT={cons(W_mt)}  L1-peak@pos{int(l1.argmax())} (L1={l1.max():.2f})")

# ── figure ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker, pandas as pd
plt.rcParams.update({"font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.7})
def logo(ax, P, title, mark=None, ct="black"):
    Pn = np.clip(P, 1e-9, 1); Pn = Pn / Pn.sum(0, keepdims=True)
    ic = np.maximum(2 + (Pn * np.log2(Pn)).sum(0), 0)
    logomaker.Logo(pd.DataFrame((Pn * ic).T, columns=list("ACGT")), ax=ax, color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([0, 2]); ax.set_ylim(0, 2); ax.tick_params(length=2)
    ax.set_title(title, fontsize=8, color=ct, loc="left", pad=2)
    if mark is not None: ax.axvspan(mark, mark + 1, color="#f1c40f", alpha=0.35, zorder=0)

fig = plt.figure(figsize=(9, 4.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.7, wspace=0.22)
for j, r in enumerate(res):
    mk = r["argmax_change"]
    logo(fig.add_subplot(gs[0, j]), r["wt_pwm"], f"{r['tf']} WT  ({r['wt_cons']})", mark=mk, ct="#333")
    logo(fig.add_subplot(gs[1, j]), r["mt_pwm"], f"{r['tf']} {r['mut']}  ({r['mt_cons']})", mark=mk, ct="#c0392b")
    axl = fig.add_subplot(gs[2, j])
    axl.bar(np.arange(len(r["l1"])), r["l1"], color="#f1c40f", edgecolor="k", lw=0.4)
    axl.bar([mk], [r["l1"][mk]], color="#e67e22", edgecolor="k", lw=0.5)
    axl.set_xlim(-0.5, len(r["l1"]) - 0.5); axl.set_ylabel("Δ pred (L1)", fontsize=7)
    axl.set_xlabel("motif position", fontsize=7); axl.tick_params(labelsize=6, length=2)
    for s in ["top", "right"]: axl.spines[s].set_visible(False)
    if r["known"]:
        kb = r["known"][mk] if mk < len(r["known"]) else "?"; pb = r["mt_cons"][mk]
        axl.set_title(f"localizes pos {mk}: predicts {pb}, {r['note']}",
                      fontsize=6.8, loc="left", color="#c0392b", pad=2)
    else:
        axl.set_title(f"responds (L1 peak {r['l1'].max():.1f}) — {r['note']}",
                      fontsize=6.5, loc="left", color="#c0392b", pad=2)

fig.suptitle("Sequence localizes the affected position but does not resolve the new base (de-novo variants)",
             fontsize=10, fontweight="bold", y=1.0)
out = f"{OUTD}/figure4a_localize"
for e in ["pdf", "svg"]: fig.savefig(f"{out}.{e}", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()
            if k not in ("wt", "mt")} for r in res], open("results/myod1_mut/fig4a_data.json", "w"), indent=1, default=str)
print(f"saved {out}.{{png,pdf,svg}}")
