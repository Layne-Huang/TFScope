#!/usr/bin/env python
"""GCN4 (bZIP homodimer) visualization:
  (top)    GT motif logo vs v20 single-chain vs v21 two-chain predicted logos
  (bottom) 1YSA co-crystal: which protomer (chain C vs D) contacts which DNA
           base -> maps each half-site of the pseudo-palindromic TGACTCA motif
           to a protomer.

Shows why GCN4 needs BOTH protomers: each bZIP monomer reads one TGAC half-site.
"""
import os, sys, json, importlib.util
import numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd, logomaker
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from eval_full_metrics import trimmed_core
from torch.utils.data import DataLoader

spec = importlib.util.spec_from_file_location("bd", "scripts/build_deeppbs_structural_v2.py")
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

DATA = "data/processed/tf_pwm_training_v2p.parquet"
SPLIT = "data/processed/splits/train_v2/split.json"
CIF = "data/raw/pdb_cif_cache/1ysa.cif"
BASES = ["A", "C", "G", "T"]
OUT = "figures/case_logos/gcn4_partner.pdf"
dev = "cuda" if torch.cuda.is_available() else "cpu"
IC, MAXSHIFT = 0.25, 10
MODELS = {
    "v20_single": "/data1/leihuang/project/TFScope/checkpoints/v20_residue_moe_newdata/residue_moe_v2_seed42",
    "v21_twochain": "/data1/leihuang/project/TFScope/checkpoints/v21_twochain_heterodimer/twochain_v2p_ddp6_seed42",
}


def pwm_to_ic_df(pwm):
    p = np.clip(pwm, 1e-8, 1.0); ic = 2.0 + (p * np.log2(p)).sum(0)
    return pd.DataFrame((p * ic[None, :]).T, columns=BASES)


def draw_logo(ax, pwm, title):
    logomaker.Logo(pwm_to_ic_df(pwm), ax=ax, color_scheme="classic", show_spines=False)
    ax.set_ylim(0, 2); ax.set_yticks([0, 1, 2]); ax.set_ylabel("bits", fontsize=8)
    ax.set_xticks([]); ax.set_title(title, fontsize=10); ax.tick_params(labelsize=7)


def load(ckpt_dir):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(ckpt_dir, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(ckpt_dir, "ckpt_best.pt"), map_location=dev,
                                 weights_only=False)["model"], strict=False)
    return m, cfg


def _trimg(pwm, th=IC):
    p = np.clip(pwm, 1e-8, 1); ic = 2 + (p * np.log2(p)).sum(0); inf = np.where(ic >= th)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]


def gcn4_pred(ckpt_dir):
    """Return (gt_core, aligned_pred, covr) for the best GCN4 test row."""
    m, cfg = load(ckpt_dir)
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    genes = [str(g).upper() for g in ds.df["gene_symbol"].values]
    idxs = [i for i, g in enumerate(genes) if g == "GCN4"]
    best = None
    with torch.no_grad():
        for i in idxs:
            b = collate_variable_length([ds[i]])
            b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long) for k, v in b.items()}
            gl, pl, _ = m(b['sequence_tokens'], b['dbd_mask'], b['family_id'], recog_prior=b.get('recog_prior'))
            pp = F.softmax(pl, 1)[0].cpu().numpy(); gate = torch.sigmoid(gl)[0].cpu().numpy()
            tg = b['target_pwm'][0].cpu().numpy(); mk = b['pwm_mask'][0].cpu().numpy()
            active = gate > 0.5
            if not active.any(): active = gate > gate.max() * 0.5
            pc = pp[:, active]; gc = _trimg(tg[:, mk.astype(bool)])
            if pc.shape[1] == 0 or gc.shape[1] == 0: continue
            al, sh, _, r = align_pwm(pc, gc, max_shift=MAXSHIFT, consider_revcomp=True)
            nov = sum(1 for k in range(pc.shape[1]) if 0 <= k + sh < gc.shape[1]); cov = nov / gc.shape[1]
            if best is None or r * cov > best[2]:
                best = (gc, al, r * cov)
    del m; torch.cuda.empty_cache()
    return best


def structural_halfsites():
    """From 1YSA, assign each core DNA BASE PAIR to the nearer protomer (C/D).

    A bZIP homodimer is dyad-symmetric: each protomer reads one half-site, and
    the two half-sites sit on OPPOSITE strands. So per motif column we must take
    the minimum protomer distance over BOTH the strand-A base and its
    Watson-Crick partner on strand B (mapped by antiparallel index)."""
    p = MMCIFParser(QUIET=True); s = p.get_structure("s", CIF); model = next(iter(s))
    chain_atoms = {}; dna = {}
    for ch in model:
        res = list(ch)
        if any(r.get_resname().strip() in bd.DNA_RESNAMES for r in res):
            dna[ch.id] = [(r, r.get_resname().strip()[-1]) for r in res
                          if r.get_resname().strip() in bd.DNA_RESNAMES]
        elif any(is_aa(r, standard=True) for r in res):
            chain_atoms[ch.id] = np.array([a.coord for r in res if is_aa(r, standard=True) for a in r])
    strandA, strandB = dna["A"], dna["B"]
    LA, LB = len(strandA), len(strandB)
    seqA = "".join(b for _, b in strandA)
    core = "TGACTCA"; start = seqA.find(core)
    C, D = chain_atoms["C"], chain_atoms["D"]

    def mind(res, X):
        bc = np.array([a.coord for a in res])
        return np.sqrt(((bc[:, None] - X[None]) ** 2).sum(-1)).min()

    assign, dists = [], []
    for k in range(start, start + len(core)):
        resA = strandA[k][0]
        j = LB - 1 - k                       # antiparallel WC partner on strand B
        pair = [resA] + ([strandB[j][0]] if 0 <= j < LB else [])
        dC = min(mind(r, C) for r in pair)
        dD = min(mind(r, D) for r in pair)
        assign.append("C" if dC < dD else "D"); dists.append((dC, dD))
    return core, assign, dists


def main():
    gt, al20, cv20 = gcn4_pred(MODELS["v20_single"])
    _, al21, cv21 = gcn4_pred(MODELS["v21_twochain"])
    core, assign, dists = structural_halfsites()
    print("GCN4 core:", core, "| per-base nearer protomer:", assign)
    for i, (b, a, (dC, dD)) in enumerate(zip(core, assign, dists)):
        print(f"  pos{i} {b}: chainC={dC:.1f}A chainD={dD:.1f}A -> {a}")

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.3], hspace=0.55)
    draw_logo(fig.add_subplot(gs[0]), gt, f"GCN4 ground truth  (TGACTCA, bZIP homodimer)")
    draw_logo(fig.add_subplot(gs[1]), al20, f"v20 single-chain prediction   (covR={cv20:.3f})")
    draw_logo(fig.add_subplot(gs[2]), al21, f"v21 two-chain prediction   (covR={cv21:.3f})")

    # protomer -> half-site map
    axm = fig.add_subplot(gs[3]); axm.axis("off")
    n = len(core)
    colC, colD = "#2c7fb8", "#d95f0e"
    axm.set_xlim(-0.5, n - 0.5); axm.set_ylim(0, 3)
    for i, (b, a) in enumerate(zip(core, assign)):
        c = colC if a == "C" else colD
        axm.add_patch(plt.Rectangle((i - 0.5, 1.2), 1, 0.9, color=c, alpha=0.35, ec="k", lw=0.5))
        axm.text(i, 1.65, b, ha="center", va="center", fontsize=15, fontweight="bold")
        axm.text(i, 0.9, a, ha="center", va="center", fontsize=9, color=c, fontweight="bold")
    # half-site brackets
    idxC = [i for i, a in enumerate(assign) if a == "C"]
    idxD = [i for i, a in enumerate(assign) if a == "D"]
    # label 5'/3' by actual position (lower index = 5' end of the motif)
    groups = [("C", idxC, colC), ("D", idxD, colD)]
    groups = [g for g in groups if g[1]]
    for name, idx, col in groups:
        half = "5′ half-site" if np.mean(idx) < (n - 1) / 2 else "3′ half-site"
        y = 2.55 if np.mean(idx) < (n - 1) / 2 else 2.05
        axm.annotate("", xy=(min(idx) - 0.4, 2.35), xytext=(max(idx) + 0.4, 2.35),
                     arrowprops=dict(arrowstyle="-", color=col, lw=2))
        axm.text(np.mean(idx), y, f"protomer {name}  ({half})", ha="center",
                 color=col, fontsize=10, fontweight="bold")
    axm.text(n / 2 - 0.5, 0.2, "1YSA: each base colored by the nearer GCN4 protomer (min heavy-atom distance)",
             ha="center", fontsize=8, style="italic")
    fig.suptitle("GCN4 homodimer: two protomers read the two halves of TGACTCA",
                 fontsize=12, fontweight="bold", y=0.98)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight"); fig.savefig(OUT.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print(f"\nsaved {OUT} (+ .png)")


if __name__ == "__main__":
    main()
