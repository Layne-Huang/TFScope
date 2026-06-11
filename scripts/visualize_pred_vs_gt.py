#!/usr/bin/env python
"""Visualize predicted vs ground-truth PWMs as sequence logos.

For each selected test TF: show GT logo (top) and model prediction logo (bottom),
oracle-aligned. Groups TFs by family, picks best/worst/median examples per family.

Usage:
  python scripts/visualize_pred_vs_gt.py --mode deeppbs  # 130-TF DeepPBS test set
  python scripts/visualize_pred_vs_gt.py --mode cluster40  # 639-TF strict holdout
  python scripts/visualize_pred_vs_gt.py --mode both
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logomaker
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from torch.utils.data import DataLoader

CKPT_ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
IC_THRESH = 0.25
MAX_SHIFT = 10
BASES = list("ACGT")
device = "cuda" if torch.cuda.is_available() else "cpu"

MODES = {
    "deeppbs": {
        "data":  "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet",
        "split": "data/processed/splits/deeppbs_only/benchmark_no_val.json",
        "ckpt":  f"{CKPT_ROOT}/fulldata_clval_v18a_trim/ckpt_epoch200.pt",
        "label": "TFScope (trim_ep200)",
        "out":   "figures/pred_vs_gt_deeppbs.pdf",
        "n_per_family": 3,
    },
    "cluster40": {
        "data":  "data/processed/tf_pwm_aug_dbd_canon_trim.parquet",
        "split": "data/processed/splits/cluster40/split.json",
        "ckpt":  f"{CKPT_ROOT}/fulldata_cluster40_v18a/ckpt_best.pt",
        "label": "TFScope (c40_best)",
        "out":   "figures/pred_vs_gt_cluster40.pdf",
        "n_per_family": 3,
    },
}


def ic_bits(pwm):
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)


def trim_core(pwm, thresh=IC_THRESH):
    ic = ic_bits(pwm)
    inf = np.where(ic >= thresh)[0]
    if len(inf) == 0: return pwm
    return pwm[:, inf[0]:inf[-1]+1]


def pwm_to_logo_df(pwm):
    """Convert (4, L) PWM to IC-scaled logomaker DataFrame."""
    pwm = np.clip(pwm, 1e-8, 1.0)
    ic = ic_bits(pwm)                      # (L,)
    ic_mat = pwm * ic[None, :]             # (4, L) IC-weighted
    return pd.DataFrame(ic_mat.T, columns=BASES)


def infer_all(cfg_dict):
    """Run inference, return dict fn→(pred_pwm, gate, tgt_pwm, tgt_mask, gene, family)."""
    data, split_path, ckpt_path = cfg_dict["data"], cfg_dict["split"], cfg_dict["ckpt"]
    cfg = TFScopeConfig()
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    if os.path.exists(cfg_path):
        for k, v in json.load(open(cfg_path)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"], strict=False)
    m.eval()

    ds = TFDataset(cfg, data, split_path, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                    collate_fn=collate_variable_length)

    df_meta = pd.read_parquet(data)
    fn2gene = dict(zip(df_meta["filename"], df_meta["gene_symbol"]))
    fn2fam  = dict(zip(df_meta["filename"], df_meta["family_name"]))

    results = {}
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            gate, pw, _ = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                            retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None,
                            recog_prior=b.get("recog_prior"))
            preds = F.softmax(pw, 1).cpu().numpy()
            tgts  = b["target_pwm"].cpu().numpy()
            masks = b["pwm_mask"].cpu().numpy()
            gates = gate.sigmoid().cpu().numpy()
            for i, fn in enumerate(ds.filenames[len(results):len(results)+len(gates)]):
                results[fn] = dict(pred=preds[i], gate=gates[i],
                                   tgt=tgts[i], mask=masks[i],
                                   gene=fn2gene.get(fn,"?"), family=fn2fam.get(fn,"?"))
    return results


def score_one(results_entry):
    pred, gate, tgt, mask = (results_entry[k] for k in ("pred","gate","tgt","mask"))
    active = gate > 0.5
    if not active.any(): active = gate > gate.max()*0.5
    pred_core = pred[:, active]
    tidx = mask.astype(bool)
    if not tidx.any(): return None, None, None
    t = tgt[:, tidx]
    tgt_core = trim_core(t)
    if tgt_core.shape[1] == 0: return None, None, None
    # align_pwm returns (aligned_pwm_in_ref_frame, offset, orient, score)
    pred_al, _, _, r = align_pwm(pred_core, tgt_core, max_shift=MAX_SHIFT, consider_revcomp=True)
    return tgt_core, pred_al, r


def draw_logo_pair(ax_gt, ax_pred, tgt_core, pred_aligned, gene, family, r, label):
    """Draw GT logo on ax_gt, predicted logo on ax_pred."""
    for ax, pwm, title, color in [
        (ax_gt,   tgt_core,     f"{gene} ({family})\nGT", "#2166ac"),
        (ax_pred, pred_aligned, f"r={r:.3f}", "#d6604d"),
    ]:
        df = pwm_to_logo_df(pwm)
        try:
            logo = logomaker.Logo(df, ax=ax, color_scheme="classic", show_spines=False)
        except Exception:
            ax.bar(range(pwm.shape[1]), ic_bits(pwm), color=color, alpha=0.7)
        ax.set_ylim(0, 2.05)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["0","1","2"], fontsize=6)
        ax.set_xticks([])
        ax.set_title(title, fontsize=7, pad=2)
        ax.spines[["top","right","bottom","left"]].set_visible(False)
        ax.set_ylabel("bits", fontsize=6, labelpad=2)


def make_figure(mode_key):
    cfg = MODES[mode_key]
    print(f"\n[{mode_key}] Loading model & running inference ...")
    results = infer_all(cfg)

    # Score all
    scored = {}
    for fn, entry in results.items():
        tgt_core, pred_al, r = score_one(entry)
        if r is None: continue
        scored[fn] = dict(**entry, tgt_core=tgt_core, pred_al=pred_al, r=r)

    print(f"  Scored {len(scored)} TFs  mean_r={np.mean([v['r'] for v in scored.values()]):.4f}")

    # Pick n_per_family: best / median / worst per family
    by_fam = {}
    for fn, v in scored.items():
        by_fam.setdefault(v["family"], []).append((v["r"], fn))
    for fam in by_fam:
        by_fam[fam].sort(key=lambda x: x[0])

    selected = []  # list of (family, fn, r, label)
    n = cfg["n_per_family"]
    for fam, items in sorted(by_fam.items()):
        if len(items) == 0: continue
        indices = np.round(np.linspace(0, len(items)-1, min(n, len(items)))).astype(int)
        labels = ["worst", "median", "best"] if n == 3 else [f"p{int(100*i/(n-1))}" for i in range(n)]
        for rank_i, idx in enumerate(indices):
            r_val, fn = items[idx]
            selected.append((fam, fn, r_val, labels[rank_i] if rank_i < len(labels) else ""))

    n_rows = len(selected)
    fig = plt.figure(figsize=(12, n_rows * 1.4 + 1.0))
    fig.suptitle(f"{cfg['label']}  —  Predicted vs Ground Truth PWMs\n"
                 f"({mode_key} test set, oracle-aligned, IC≥{IC_THRESH} bits)",
                 fontsize=10, y=0.995)

    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.9, wspace=0.3)

    for row_i, (fam, fn, r_val, rank_label) in enumerate(selected):
        v = scored[fn]
        ax_gt   = fig.add_subplot(gs[row_i, 0])
        ax_pred = fig.add_subplot(gs[row_i, 1])
        title_gt = f"{v['gene']}  [{fam}]  ({rank_label})"
        draw_logo_pair(ax_gt, ax_pred, v["tgt_core"], v["pred_al"],
                       v["gene"], fam, r_val, cfg["label"])
        ax_gt.set_title(f"{title_gt}\nGround Truth", fontsize=7, pad=2)
        ax_pred.set_title(f"Predicted   r={r_val:.3f}", fontsize=7, pad=2)

    # Column headers
    fig.text(0.28, 0.998, "Ground Truth", ha="center", fontsize=9, fontweight="bold")
    fig.text(0.75, 0.998, "Prediction",   ha="center", fontsize=9, fontweight="bold")

    os.makedirs("figures", exist_ok=True)
    fig.savefig(cfg["out"], bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {cfg['out']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["deeppbs","cluster40","both"], default="both")
    args = ap.parse_args()
    modes = ["deeppbs","cluster40"] if args.mode == "both" else [args.mode]
    for m in modes:
        make_figure(m)
    print("\nDone.")


if __name__ == "__main__":
    main()
