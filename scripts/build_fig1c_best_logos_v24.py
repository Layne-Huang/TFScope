#!/usr/bin/env python
"""Fig 1c - cherry-picked best TFScope predictions: GT vs TFScope sequence logos.

Runs the combined TFScope model on the 84-TF cluster40 test, ranks TFs by oracle-r
(same unified metric as Fig 1d), and draws GT (top) vs TFScope-predicted (bottom)
logos for the top predictions (deduped by gene, family-diverse). TFScope's predicted
core is rendered in the GT frame via align_pwm.
"""
import os, sys, json
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
os.environ.setdefault("HF_HOME", "/data1/leihuang/.cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from torch.utils.data import DataLoader

CKPT_DIR = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
DATA  = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
DEEPPBS_NPZ = "/data1/leihuang/data/DeepPBS/output_cluster40/cluster40_deeppbs_preds.npz"
IC_THRESH, MAXSHIFT, MINPOS = 0.25, 10, 4
# Curated family-diverse showcase (one clean recovery per family). ttk dropped — its
# TFScope logo degenerates to CCCT and misses the TTATCCT core; ZFP57 (C2H2, GCGG) recovers cleanly.
PICKS = ["GATA3", "CLOCK", "MATALPHA2", "ZBTB7A", "PPARG", "ZFP57"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def ic_trim(pwm):
    p = np.clip(pwm, 1e-8, 1.0); ic = 2.0 + (p * np.log2(p)).sum(0)
    inf = np.where(ic >= IC_THRESH)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]

def load_deeppbs():
    """pdb_chain -> (4,L) DeepPBS predicted PWM."""
    z = np.load(DEEPPBS_NPZ, allow_pickle=True)
    n = sum(1 for k in z.keys() if k.startswith("pred_"))
    out = {}
    for i in range(n):
        nm = str(z[f"name_{i}"]); p = np.asarray(z[f"pred_{i}"], dtype=np.float32)
        if p.shape[1] != 4 and p.shape[0] == 4: p = p.T
        out["_".join(nm.split("_")[:2])] = p.T  # (4,L), keyed by pdb_chain
    return out

def logo(ax, pwm, title, color):
    p = np.clip(pwm, 1e-8, 1.0)
    ic = np.maximum(2 + (p * np.log2(p)).sum(0), 0)
    df = pd.DataFrame((pwm * ic).T, columns=list("ACGT"))
    logomaker.Logo(df, ax=ax, color_scheme="classic", show_spines=False, vpad=0.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 2)
    if title:
        ax.set_title(title, fontsize=8.5, color=color, pad=3)


def main():
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(CKPT_DIR, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ckpt_best.pt"),
                                 map_location=dev, weights_only=False)["model"], strict=False)
    df = pd.read_parquet(DATA)
    fn_gene = {r["filename"]: str(r["gene_symbol"]) for _, r in df.iterrows()}
    fn_fam  = {r["filename"]: str(r["family_name"]) for _, r in df.iterrows()}
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    names = list(ds.filenames)
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_variable_length)

    recs, gi = [], 0
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            gl, pl, _ = m(b['sequence_tokens'], b['dbd_mask'], b['family_id'],
                          retrieved_pwms=b.get('retrieved_pwms'), retrieved_masks=b.get('retrieved_masks'),
                          retrieved_sims=b.get('retrieved_sims'), recog_prior=b.get('recog_prior'))
            pwm = F.softmax(pl, 1).cpu().numpy(); gate = torch.sigmoid(gl).cpu().numpy()
            tgt = b['target_pwm'].cpu().numpy(); msk = b['pwm_mask'].cpu().numpy()
            for pred, tg_full, ms, ga in zip(pwm, tgt, msk, gate):
                name = names[gi]; gi += 1
                tg = ic_trim(tg_full[:, ms.astype(bool)])
                if tg.shape[1] < MINPOS: continue
                active = ga > 0.5
                if not active.any(): active = ga > ga.max() * 0.5
                pcore = pred[:, active]
                if pcore.shape[1] == 0: continue
                aligned, _, _, r = align_pwm(pcore, tg, max_shift=MAXSHIFT, consider_revcomp=True)
                recs.append(dict(name=name, gene=fn_gene.get(name, name), fam=fn_fam.get(name, "?"),
                                 r=float(r), gt=tg, pred=aligned))

    # attach DeepPBS predictions (aligned to the same GT core)
    dpp = load_deeppbs()
    for x in recs:
        pc = "_".join(x["name"].split("_")[:2])
        if pc in dpp:
            da, _, _, dr = align_pwm(ic_trim(dpp[pc]), x["gt"], max_shift=MAXSHIFT, consider_revcomp=True)
            x["dpp"], x["dpp_r"] = da, float(dr)
        else:
            x["dpp"], x["dpp_r"] = None, float("nan")

    # best per gene, then family-diverse top-N
    best_by_gene = {}
    for x in sorted(recs, key=lambda z: -z["r"]):
        best_by_gene.setdefault(x["gene"], x)
    ranked = sorted(best_by_gene.values(), key=lambda z: -z["r"])

    import pickle
    os.makedirs("results/figure1c", exist_ok=True)
    pickle.dump(ranked, open("results/figure1c/recs.pkl", "wb"))
    print("\n=== per-gene ranking (gene | family | TFScope r | DeepPBS r) ===")
    for x in ranked:
        print(f"  {x['gene']:12s} {x['fam']:18s}  TF {x['r']:.3f}   DPP {x.get('dpp_r', float('nan')):.3f}")
    picks = [best_by_gene[g] for g in PICKS if g in best_by_gene]
    print("picks:", [(p["gene"], p["fam"], round(p["r"], 3)) for p in picks])

    n = len(picks)
    fig, axes = plt.subplots(3, n, figsize=(1.7 * n, 4.4),
                             gridspec_kw=dict(hspace=0.55, wspace=0.25))
    uni = np.full((4, 1), 0.25, np.float32)
    for j, p in enumerate(picks):
        logo(axes[0, j], p["gt"], f"{p['gene']}\n{p['fam']}", "#333333")
        if p["dpp"] is not None:
            logo(axes[1, j], p["dpp"], f"r = {p['dpp_r']:.2f}", "#D55E00")
        else:
            logo(axes[1, j], uni, "n/a", "#D55E00")
        logo(axes[2, j], p["pred"], f"r = {p['r']:.2f}", "#0072B2")
    axes[0, 0].set_ylabel("Ground truth", fontsize=9)
    axes[1, 0].set_ylabel("DeepPBS", fontsize=9, color="#D55E00")
    axes[2, 0].set_ylabel("TFScope", fontsize=9, color="#0072B2")
    fig.suptitle("TFScope recovers experimental motifs from sequence alone",
                 fontsize=11, fontweight="bold", y=1.01)
    out = "figures_v24/figure1c_best_logos/figure1c_best_logos"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("saved", out)

    # second version: Ground truth + TFScope only (no DeepPBS row)
    fig2, ax2 = plt.subplots(2, n, figsize=(1.7 * n, 3.1),
                             gridspec_kw=dict(hspace=0.55, wspace=0.25))
    for j, p in enumerate(picks):
        logo(ax2[0, j], p["gt"], f"{p['gene']}\n{p['fam']}", "#333333")
        logo(ax2[1, j], p["pred"], f"r = {p['r']:.2f}", "#0072B2")
    ax2[0, 0].set_ylabel("Ground truth", fontsize=9)
    ax2[1, 0].set_ylabel("TFScope", fontsize=9, color="#0072B2")
    fig2.suptitle("TFScope recovers experimental motifs from sequence alone",
                  fontsize=11, fontweight="bold", y=1.02)
    out2 = "figures_v24/figure1c_best_logos/figure1c_best_logos_no_deeppbs"
    fig2.savefig(out2 + ".png", dpi=300, bbox_inches="tight")
    fig2.savefig(out2 + ".pdf", bbox_inches="tight")
    print("saved", out2)


if __name__ == "__main__":
    main()
