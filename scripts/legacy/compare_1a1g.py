#!/usr/bin/env python
"""3-panel logo for 1a1g_A (Egr1): truth | v18a prediction | LGO retrieval prior.

The LGO prior is the similarity-weighted average of the K leave-gene-out
neighbours the index supplies for this TF (WT1 / ZBTB7A C2H2 fingers).
Each prediction/prior is strand-oriented to the target; titles show per-column
Pearson r vs truth.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "pwm_rosetta")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from pwm_hybrid.pwm.viz import makeLogo

CKPT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18a_attnrepair/ckpt_best.pt"
DATA = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
TARGET_FN = "1a1g_A_Egr1.MA0162.1.txt"
RC = [3, 2, 1, 0]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def colr(p, t, L):
    return np.nanmean([pearsonr(t[:, j], p[:, j])[0] for j in range(L)])


def orient(p, t, L):
    """Strand-flip p to best match t over L columns."""
    rf = colr(p[:, :L], t[:, :L], L)
    pc = p.copy(); pc[:, :L] = p[:, :L][RC][:, ::-1]
    rr = colr(pc[:, :L], t[:, :L], L)
    return (pc if (not np.isnan(rr) and (np.isnan(rf) or rr > rf)) else p)


def draw(ax, pwm, L, title, color):
    ppm = np.clip(pwm[:, :L].T, 1e-8, 1.0); ppm = ppm / ppm.sum(1, keepdims=True)
    makeLogo(ppm, ax)
    ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, color=color, fontweight="bold")


def main():
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)

    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    i = ds.filenames.index(TARGET_FN)
    s = ds[i]
    b = collate_variable_length([s])
    b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long) for k, v in b.items()}
    with torch.no_grad():
        _, pw, _ = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                     retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
                     retrieved_sims=b.get("retrieved_sims"), recog_prior=b.get("recog_prior"))
    pred = F.softmax(pw, 1)[0].cpu().numpy()
    truth = s["target_pwm"].numpy()
    L = int(s["pwm_mask"].sum())

    # identify the K LGO neighbours (filename + cos) from the index
    import pandas as pd
    idx = json.load(open(cfg.retrieval_index_path))
    nbrs = idx.get(TARGET_FN, [])
    df = pd.read_parquet(DATA)
    fn2gene = dict(zip(df["filename"], df["gene_symbol"].astype(str)))

    ret_pwms = b["retrieved_pwms"][0].cpu().numpy()      # (K,4,20)
    ret_mask = b["retrieved_masks"][0].cpu().numpy()     # (K,20)
    # aggregated (sim-weighted) prior — what the model actually consumes
    prior = TFScopeModel._aggregate_prior(b["retrieved_pwms"], b["retrieved_masks"],
                                          b["retrieved_sims"])[0].cpu().numpy()

    pred = orient(pred, truth, L)
    r_pred = colr(pred[:, :L], truth[:, :L], L)
    prior_o = orient(prior, truth, L); r_prior = colr(prior_o[:, :L], truth[:, :L], L)

    K = ret_pwms.shape[0]
    ncol = 3 + K                                          # truth | pred | agg | each neighbour
    fig, ax = plt.subplots(1, ncol, figsize=(2.6 * ncol, 2.4))
    draw(ax[0], truth, L, "Egr1 (truth, MA0162.1)", "#2E7D32")
    draw(ax[1], pred, L, f"v18a prediction  r={r_pred:.2f}", "#1565C0")
    draw(ax[2], prior_o, L, f"LGO prior (sim-avg of K={K})  r={r_prior:.2f}", "#C62828")
    for k in range(K):
        Lk = int(ret_mask[k].sum())
        nbfn = nbrs[k]["nn_filename"] if k < len(nbrs) else "?"
        gene = fn2gene.get(nbfn, "?"); cos = nbrs[k]["cos_sim"] if k < len(nbrs) else 0
        if Lk < 1:
            ax[3 + k].axis("off"); continue
        nb_o = orient(ret_pwms[k], truth, min(L, Lk))
        rk = colr(nb_o[:, :min(L, Lk)], truth[:, :min(L, Lk)], min(L, Lk))
        draw(ax[3 + k], nb_o, Lk, f"nbr{k+1}: {gene}\ncos={cos:.2f}  r={rk:.2f}", "#EF6C00")
    fig.suptitle("1a1g_A Egr1 — truth | v18a | LGO sim-avg prior | individual LGO neighbours",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    os.makedirs("results/v18_compare", exist_ok=True)
    out = "results/v18_compare/compare_1a1g_Egr1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}  (L={L}, r_pred={r_pred:.3f}, r_prior={r_prior:.3f}, K={K})")


if __name__ == "__main__":
    main()
