#!/usr/bin/env python
"""Multi-page PWM-logo comparison for all 130 test TFs.

Columns: truth | <model_1> | <model_2> | ... | DeepPBS  (default: v17, v18a).
Same format as results/tfscope_v14_LGO/all_pwms_truth_v14LGO_deeppbs.pdf:
sequence logos in a paged grid, each prediction strand-oriented to the target
and annotated with per-column Pearson r.

Usage:
  python scripts/plot_all_pwms_compare.py --models v17 v18a \
      --out results/v18_compare/all_pwms_truth_v17_v18a_deeppbs.pdf
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src"); sys.path.insert(0, "pwm_rosetta")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from pwm_hybrid.pwm.viz import makeLogo

CKPT_ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
CKPTS = {"v14": f"{CKPT_ROOT}/deeppbs_v14_icpcc/ckpt_best.pt",
         "v16": f"{CKPT_ROOT}/deeppbs_v16/ckpt_best.pt",
         "v17": f"{CKPT_ROOT}/deeppbs_v17_200ep/ckpt_best.pt",
         "v18a": f"{CKPT_ROOT}/deeppbs_v18a_attnrepair/ckpt_best.pt",
         "v18b": f"{CKPT_ROOT}/deeppbs_v18b_contact/ckpt_best.pt",
         "v18a_noRAG": f"{CKPT_ROOT}/deeppbs_v18a_noRAG/ckpt_best.pt"}
DATA = "data/processed/tf_pwm_deeppbs_only.parquet"
CANON = "data/processed/tf_pwm_deeppbs_only_canon.parquet"
DATA_OF = {"v16": CANON}
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
MAX_L = 20
RC = [3, 2, 1, 0]
COLORS = {"v14": "#6A1B9A", "v16": "#00838F", "v17": "#1565C0", "v18a": "#1565C0",
          "v18b": "#283593", "v18a_noRAG": "#00695C"}
device = "cuda" if torch.cuda.is_available() else "cpu"


def percol_r(pred, targ, L):
    return np.nanmean([pearsonr(targ[:, j], pred[:, j])[0] for j in range(L)])


def strand_orient(preds, targs, masks):
    """Flip each prediction to the strand best matching the target (strand-symmetric)."""
    for i in range(len(preds)):
        L = int(masks[i].sum())
        if L < 2: continue
        rf = percol_r(preds[i][:, :L], targs[i][:, :L], L)
        rc = preds[i].copy(); rc[:, :L] = preds[i][:, :L][RC][:, ::-1]
        rr = percol_r(rc[:, :L], targs[i][:, :L], L)
        if not np.isnan(rr) and (np.isnan(rf) or rr > rf):
            preds[i] = rc
    return preds


def infer(name):
    ckpt = CKPTS[name]; data = DATA_OF.get(name, DATA)
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False)
    m.eval()
    ds = TFDataset(cfg, data, SPLIT, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    P, T, M, BE, TR = [], [], [], [], []
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            _, pw, aux = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                           retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
                           retrieved_sims=b.get("retrieved_sims"), recog_prior=b.get("recog_prior"))
            P.append(F.softmax(pw, 1).cpu().numpy()); T.append(b["target_pwm"].cpu().numpy())
            M.append(b["pwm_mask"].cpu().numpy())
            n = P[-1].shape[0]
            # β gate (v18 stores it on prior_head); max per-neighbour trust
            be = aux.get("beta_gated")
            if be is None:
                ph = getattr(m.pwm_head, "prior_head", m.pwm_head)
                be = getattr(ph, "_last_beta_gated", None)
            BE.append(be.cpu().numpy() if be is not None else np.full(n, np.nan))
            tl = aux.get("trust_logits")
            TR.append(torch.sigmoid(tl).max(-1).values.cpu().numpy() if tl is not None else np.full(n, np.nan))
    P, T, M = np.concatenate(P), np.concatenate(T), np.concatenate(M)
    BE, TR = np.concatenate(BE), np.concatenate(TR)
    # NOTE: store RAW predictions; offset+RC alignment to truth is done at draw time.
    return ({fn: P[i] for i, fn in enumerate(ds.filenames)},
            {fn: (T[i], int(M[i].sum())) for i, fn in enumerate(ds.filenames)},
            ds.filenames,
            {fn: (float(BE[i]), float(TR[i])) for i, fn in enumerate(ds.filenames)})


def align_to_truth(pred_valid, truth_valid):
    """Offset+RC align a prediction to the truth (matches the benchmark metric).
    Returns (aligned-in-truth-frame, per-col r, shift, orient)."""
    aligned, shift, orient, score = align_pwm(pred_valid, truth_valid,
                                              max_shift=10, consider_revcomp=True)
    return aligned, score, shift, orient


def _tag(label, r, shift, orient):
    extra = (" rc" if orient == "rc" else "") + (f" Δ{shift:+d}" if shift else "")
    return f"{label}  r={r:.2f}{extra}"


def load_lgo_neighbors(index_path, K):
    """Return get(fn) -> list of (gene, cos, padded_pwm(4,20), valid_len) for the K LGO neighbours."""
    idx = json.load(open(index_path))
    df = pd.read_parquet(DATA)
    fn2gene = dict(zip(df["filename"], df["gene_symbol"].astype(str)))
    fn2pwm = {}
    for _, r in df.iterrows():
        b = r["pwm"]
        p = (np.frombuffer(b, dtype=np.float32).reshape(4, -1) if isinstance(b, bytes)
             else np.full((4, 4), 0.25, np.float32))
        L = min(p.shape[1], MAX_L)
        arr = np.full((4, MAX_L), 0.25, np.float32); arr[:, :L] = p[:, :L]
        fn2pwm[r["filename"]] = (arr / arr.sum(0, keepdims=True), L)

    def get(fn):
        out = []
        for nb in idx.get(fn, [])[:K]:
            nn = nb["nn_filename"]; pw = fn2pwm.get(nn)
            if pw is not None:
                full = nn.replace(".txt", "")            # pdb_chain_GENE.SOURCE
                out.append((full, nb.get("cos_sim", 0.0), pw[0], pw[1]))
        return out
    return get


def _rtag(r, sh, ori):
    """Compact r annotation carrying the offset (Δ) and reverse-complement (rc)."""
    s = f"{r:.2f}"
    if ori == "rc":
        s += " rc"
    if sh:
        s += f" Δ{sh:+d}"
    return s


def draw_logo(ax, pwm, L, title, color, fontsize=6):
    if L < 1:
        ax.axis("off"); return
    ppm = np.clip(pwm[:, :L].T, 1e-8, 1.0); ppm = ppm / ppm.sum(1, keepdims=True)
    makeLogo(ppm, ax)
    ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize, color=color, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v17", "v18a"])
    ap.add_argument("--out", default="results/v18_compare/all_pwms_truth_v17_v18a_deeppbs.pdf")
    ap.add_argument("--show-lgo-neighbors", type=int, default=0,
                    help="append N LGO retrieval-neighbour logos per TF")
    ap.add_argument("--lgo-index", default="data/processed/tf_nn_index_lgo_deeppbs.json")
    a = ap.parse_args()

    model_preds, model_meta, ref_tgt, fns = {}, {}, None, None
    for name in a.models:
        if not os.path.exists(CKPTS[name]): print(f"[skip] {name}"); continue
        print(f"infer {name} ...", flush=True)
        p, t, f, meta = infer(name)
        model_preds[name] = p; model_meta[name] = meta
        if ref_tgt is None: ref_tgt, fns = t, f

    # DeepPBS per-structure (covers all 130), mapped by PDB_chain prefix
    df = pd.read_parquet(DATA)
    fn2name = dict(zip(df["filename"], df["tf_name"]))
    dstruct = np.load("results/deeppbs_blind_benchmark/struct_preds.npz", allow_pickle=True)
    dnames = sorted(set(k.rsplit("::", 1)[0] for k in dstruct.files))
    pdbc2name = {}
    for nm in dnames:
        pdbc2name.setdefault("_".join(nm.split("_")[:2]), nm)

    def dpbs_for(fn):
        nm = pdbc2name.get("_".join(fn.split("_")[:2]))
        if nm is None or (nm + "::pred") not in dstruct.files: return None, 0
        p = dstruct[nm + "::pred"]
        arr = np.full((4, MAX_L), 0.25, np.float32)
        L = min(p.shape[0], MAX_L); arr[:, :L] = np.clip(p[:L].T, 1e-8, 1)
        return arr / arr.sum(0, keepdims=True), L

    K = a.show_lgo_neighbors
    neigh_get = load_lgo_neighbors(a.lgo_index, K) if K > 0 else None
    cols = ["truth"] + list(model_preds.keys()) + ["DeepPBS"] + [f"LGO nbr{k+1}" for k in range(K)]
    ncol = len(cols)
    n_models = len(model_preds)
    per_page = 8
    n = len(fns)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with PdfPages(a.out) as pdf:
        for start in range(0, n, per_page):
            chunk = list(range(start, min(start + per_page, n)))
            fig, axes = plt.subplots(len(chunk), ncol, figsize=(2.6 * ncol, 1.5 * len(chunk)),
                                     squeeze=False)
            for row, i in enumerate(chunk):
                fn = fns[i]
                full = fn.replace(".txt", "")            # pdb_chain_GENE.SOURCE
                T_i, L = ref_tgt[fn]
                tv = T_i[:, :L]
                # truth — full name (PDB id + source); logo in own frame
                draw_logo(axes[row][0], T_i, L, f"{full}\n(truth)", "#2E7D32")
                # model predictions — own frame; r is offset+RC aligned (Δ/rc in the tag)
                for c, mname in enumerate(model_preds.keys(), start=1):
                    p = model_preds[mname][fn]
                    _, r, sh, ori = align_to_truth(p[:, :L], tv)
                    be, tr = model_meta.get(mname, {}).get(fn, (np.nan, np.nan))
                    gate = "" if np.isnan(be) else f"\nβ={be:.2f} maxT={tr:.2f}"
                    draw_logo(axes[row][c], p, L, f"{mname}  r={_rtag(r, sh, ori)}{gate}",
                              COLORS.get(mname, "#1565C0"))
                # DeepPBS — own frame; aligned r
                dp, dL = dpbs_for(fn)
                dax = axes[row][1 + n_models]
                if dp is not None and dL > 0:
                    _, r, sh, ori = align_to_truth(dp[:, :dL], tv)
                    draw_logo(dax, dp, dL, f"DeepPBS  r={_rtag(r, sh, ori)}", "#C62828")
                else:
                    dax.text(0.5, 0.5, "no DeepPBS", ha="center", va="center", fontsize=6, color="#999")
                    dax.axis("off")
                # LGO neighbours — own frame; full name; TWO offset+RC-aligned r's
                # (r_pred vs prediction, r_gt vs truth), each carrying its Δ/rc.
                if neigh_get is not None:
                    nbrs = neigh_get(fn)
                    pred_raw = model_preds[next(iter(model_preds))][fn]
                    for k in range(K):
                        ax = axes[row][2 + n_models + k]
                        if k < len(nbrs):
                            full_nb, cos, npw, nL = nbrs[k]
                            _, rg, shg, og = align_to_truth(npw[:, :nL], tv)
                            _, rp, shp, op = align_to_truth(npw[:, :nL], pred_raw[:, :L])
                            ttl = (f"{full_nb}\ncos={cos:.2f}\n"
                                   f"r_pred={_rtag(rp, shp, op)}\nr_gt={_rtag(rg, shg, og)}")
                            draw_logo(ax, npw, nL, ttl, "#EF6C00")
                        else:
                            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {a.out}  ({n} TFs, {(n + per_page - 1)//per_page} pages, cols: {cols})")


if __name__ == "__main__":
    main()
