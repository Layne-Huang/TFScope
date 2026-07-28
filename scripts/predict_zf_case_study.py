#!/usr/bin/env python
"""Run TFScope deploy model on designed ZF arrays from PDB files.

Extracts:
  - Protein sequence (chain A) → TFScope input
  - DNA sequence (chain B)     → ground truth consensus

Pipeline:
  1. Parse PDB → protein seq + DNA ground truth
  2. Compute ESM2 DBD-mean embedding for retrieval
  3. Find K=16 nearest neighbours from deploy index
  4. TFScope forward → predicted PWM
  5. Align predicted PWM to DNA ground truth, compute metrics
  6. Save logo plots

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/predict_zf_case_study.py
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")

from Bio import PDB
from Bio.PDB.Polypeptide import protein_letters_3to1
from scipy.stats import pearsonr

from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from tfscope.data.dataset import AA_TO_TOKEN

# ── Paths ────────────────────────────────────────────────────────────────────
PDB_DIR  = "/afs/csail.mit.edu/u/l/leihuang/project/TFScope/case_study/pdb/zf"
EMB_FILE = "data/processed/tf_dbd_embeddings_aug.npz"
PARQUET  = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
OUT_DIR  = "results/zf_case_study"

MODELS = {
    "E2": {
        "ckpt_dir":  "/data1/leihuang/project/TFScope/checkpoints/v19_e2_gene_balanced_bf16_ddp3/rag_seed42",
        "ckpt_name": "ckpt_best.pt",
        "index":     "data/processed/tf_nn_index_cluster40_clean.json",
        "color":     "#2166ac",
        "label":     "E2 (clean split)",
    },
    "Deploy": {
        "ckpt_dir":  "/data1/leihuang/project/TFScope/checkpoints/v19_deploy_e2_full/deploy_rag_seed42",
        "ckpt_name": "ckpt_best.pt",
        "index":     "data/processed/tf_nn_index_cluster40_full_deploy.json",
        "color":     "#b2182b",
        "label":     "Deploy (all data)",
    },
}

FAMILY_ID = 1   # C2H2_medium (3-finger arrays)
K         = 16
MAX_SHIFT = 10
BASES     = list("ACGT")
DNA_3to1  = {"DA": "A", "DT": "T", "DC": "C", "DG": "G",
             "A":  "A", "T":  "T", "C":  "C", "G":  "G"}
COLORS    = {"A": "#1a9641", "C": "#2b83ba", "G": "#fdae61", "T": "#d7191c"}

# ── PDB parsing ──────────────────────────────────────────────────────────────

def parse_pdb(path):
    parser = PDB.PDBParser(QUIET=True)
    struct  = parser.get_structure("s", path)
    model   = list(struct.get_models())[0]
    prot_seq, dna_seq = "", ""
    for chain in model.get_chains():
        residues = list(chain.get_residues())
        if not residues: continue
        first = residues[0].get_resname().strip()
        if first in DNA_3to1 and chain.get_id() == "B":
            dna_seq = "".join(DNA_3to1.get(r.get_resname().strip(), "N") for r in residues)
        elif first not in DNA_3to1 and chain.get_id() == "A":
            prot_seq = "".join(protein_letters_3to1.get(r.get_resname().strip(), "X") for r in residues)
    return prot_seq, dna_seq


def dna_to_pwm(dna_seq):
    """Convert a consensus DNA string to a one-hot PWM (4 × L)."""
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    L   = len(dna_seq)
    pwm = np.zeros((4, L), dtype=np.float32)
    for i, nt in enumerate(dna_seq):
        if nt.upper() in idx:
            pwm[idx[nt.upper()], i] = 1.0
        else:
            pwm[:, i] = 0.25
    return pwm


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(ckpt_dir, ckpt_name, device):
    cfg = TFScopeConfig()
    with open(os.path.join(ckpt_dir, "config.json")) as fh:
        saved = json.load(fh)
    for k, v in saved.items():
        if hasattr(cfg, k):
            try:    setattr(cfg, k, type(getattr(cfg, k))(v))
            except: setattr(cfg, k, v)
    model = TFScopeModel(cfg).to(device).eval()
    ckpt  = torch.load(os.path.join(ckpt_dir, ckpt_name),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"  Loaded {ckpt_dir.split('/')[-2]}/{ckpt_dir.split('/')[-1]} (epoch {ckpt.get('epoch','?')})")
    return model, cfg


# ── Retrieval: compute ESM2 embedding on-the-fly ────────────────────────────

def compute_esm2_embedding(seq, model_tfscope, device):
    """Use TFScope's frozen ESM2 backbone to embed the query sequence."""
    tokens = torch.tensor(
        [[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=device
    )
    with torch.no_grad():
        # backbone.forward() returns (B, L, D) weighted-layer embeddings
        emb_3d = model_tfscope.backbone(tokens)   # (1, L, D)
        emb    = emb_3d[0].mean(0).cpu().numpy().astype(np.float32)
    return emb


def find_neighbors(query_emb, index, emb_npz, parquet_df, k=16):
    """Cosine-nearest donors from the pre-built deploy index."""
    # The index already stores top-K per query; for new sequences not in the
    # index we do a direct cosine scan against all donor embeddings.
    with np.load(emb_npz) as ef:
        donor_fns  = list(ef.files)
        donor_mat  = np.stack([ef[fn] for fn in donor_fns]).astype(np.float32)

    donor_mat /= np.linalg.norm(donor_mat, axis=1, keepdims=True) + 1e-8
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    sims = donor_mat @ q
    top_idx = np.argsort(-sims)[:k]

    fn2pwm = {
        row["filename"]: np.frombuffer(row["pwm"], dtype=np.float32).reshape(4, -1)
        for _, row in parquet_df.iterrows()
    }
    fn2gene = dict(zip(parquet_df["filename"], parquet_df["gene_symbol"]))

    neighbors = []
    for idx in top_idx:
        fn   = donor_fns[idx]
        pwm  = fn2pwm.get(fn)
        gene = fn2gene.get(fn, "?")
        neighbors.append((fn, float(sims[idx]), pwm, gene))
    return neighbors


# ── TFScope inference ─────────────────────────────────────────────────────────

def run_inference(seq, neighbors, model, cfg, device, family_id=FAMILY_ID):
    K_ret = len(neighbors)
    ML    = cfg.max_motif_length

    tokens   = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]],
                             dtype=torch.long, device=device)
    dbd_mask = torch.ones(1, len(seq), dtype=torch.bool, device=device)
    fam_id   = torch.tensor([family_id], dtype=torch.long, device=device)

    ret_pwms  = torch.full((1, K_ret, 4, ML), 0.25, dtype=torch.float32, device=device)
    ret_masks = torch.zeros((1, K_ret, ML),        dtype=torch.float32, device=device)
    ret_sims  = torch.zeros((1, K_ret),             dtype=torch.float32, device=device)

    for i, (fn, sim, pwm, gene) in enumerate(neighbors):
        if pwm is not None:
            L = min(pwm.shape[1], ML)
            ret_pwms[0, i, :, :L]  = torch.from_numpy(pwm[:, :L]).to(device)
            ret_masks[0, i, :L]    = 1.0
        ret_sims[0, i] = sim

    with torch.no_grad():
        gate_logits, pwm_logits, _ = model(
            tokens, dbd_mask, fam_id,
            retrieved_pwms=ret_pwms, retrieved_masks=ret_masks, retrieved_sims=ret_sims
        )
    pred_pwm  = F.softmax(pwm_logits, dim=1)[0].cpu().numpy()       # (4, L)
    gate_prob = torch.sigmoid(gate_logits)[0].cpu().numpy()          # (L,)
    # active columns (gate > 0.5)
    active    = gate_prob > 0.5
    if not active.any():
        active = gate_prob > gate_prob.max() * 0.5
    return pred_pwm, gate_prob, active


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pred_4L, gt_4L, max_shift=MAX_SHIFT):
    """Oracle-aligned and fixed-frame metrics vs one-hot GT."""
    # Oracle-aligned panel-r
    aligned, shift, orient, _ = align_pwm(
        pred_4L, gt_4L, max_shift=max_shift, consider_revcomp=True
    )
    L = min(aligned.shape[1], gt_4L.shape[1])
    a = aligned[:, :L].ravel(); b = gt_4L[:, :L].ravel()
    denom = np.sqrt(((a-a.mean())**2).sum() * ((b-b.mean())**2).sum())
    panel_r = float(np.dot(a-a.mean(), b-b.mean()) / denom) if denom > 0 else 0.0

    # DeepPBS-style aligned per-pos r (4-element vectors)
    al_L4 = aligned.T; gt_L4 = gt_4L.T
    Lmin  = min(al_L4.shape[0], gt_L4.shape[0])
    per_pos_r = float(np.nanmean([pearsonr(gt_L4[i], al_L4[i])[0] for i in range(Lmin)]))

    # Fixed MAE
    Lf = min(pred_4L.shape[1], gt_4L.shape[1])
    mae = float(np.mean(np.sum(np.abs(pred_4L[:, :Lf].T - gt_4L[:, :Lf].T), axis=1)))

    return {
        "panel_r":     panel_r,
        "aligned_r":   per_pos_r,
        "fixed_mae":   mae,
        "shift":       int(shift),
        "orient":      orient,
    }


# ── Logo plotting ─────────────────────────────────────────────────────────────

def ic_bits(pwm_4L):
    p = np.clip(pwm_4L, 1e-8, 1.0)
    p = p / p.sum(0, keepdims=True)
    return 2.0 + (p * np.log2(p)).sum(0)


def logo_df(pwm_4L):
    p = np.clip(pwm_4L, 1e-8, 1.0)
    p = p / p.sum(0, keepdims=True)
    ic = ic_bits(p)
    return pandas_df(p * ic[None, :])


def pandas_df(mat_4L):
    import pandas as pd
    return pd.DataFrame(mat_4L.T, columns=BASES)


def draw_logo(ax, pwm_4L, title, color="black"):
    import pandas as pd
    p = np.clip(pwm_4L, 1e-8, 1.0)
    p = p / p.sum(0, keepdims=True)
    ic = 2.0 + (p * np.log2(p)).sum(0)
    df = pd.DataFrame((p * ic[None, :]).T, columns=BASES)
    logomaker.Logo(df, ax=ax, color_scheme=COLORS, show_spines=False,
                   fade_probabilities=False, font_name="DejaVu Sans")
    ax.set_ylim(0, 2.1); ax.set_yticks([0, 1, 2])
    ax.tick_params(axis="both", labelsize=7, length=0)
    ax.set_title(title, fontsize=8, color=color, pad=2)
    ax.set_ylabel("bits", fontsize=7)
    ax.axhline(0, color="black", lw=0.5)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    import pandas as pd
    parquet_df = pd.read_parquet(PARQUET)

    # Parse PDB files
    pdb_files = sorted(f for f in os.listdir(PDB_DIR) if f.endswith(".pdb"))
    samples   = []
    for fname in pdb_files:
        path = os.path.join(PDB_DIR, fname)
        prot_seq, dna_seq = parse_pdb(path)
        name = fname.replace("_model.pdb", "")
        print(f"{name}: protein {len(prot_seq)} aa, DNA {len(dna_seq)} nt: {dna_seq}")
        samples.append({"name": name, "prot_seq": prot_seq,
                        "dna_seq": dna_seq, "gt_pwm": dna_to_pwm(dna_seq)})
    print()

    # ── Run both models ────────────────────────────────────────────────────────
    # model_preds[model_key][sample_idx] = pred_pwm (4, L)
    model_preds   = {}
    model_results = {}   # model_key → list of per-sample metric dicts

    for mkey, mcfg in MODELS.items():
        print(f"── {mkey}: {mcfg['label']} ──")
        model, cfg = load_model(mcfg["ckpt_dir"], mcfg["ckpt_name"], device)

        preds   = []
        results = []
        for s in samples:
            qemb      = compute_esm2_embedding(s["prot_seq"], model, device)
            neighbors = find_neighbors(qemb, mcfg["index"], EMB_FILE, parquet_df, k=K)
            pred_pwm, gate_prob, active = run_inference(
                s["prot_seq"], neighbors, model, cfg, device
            )
            metrics = compute_metrics(pred_pwm, s["gt_pwm"])
            print(f"  {s['name']}: panel-r={metrics['panel_r']:.3f}  "
                  f"aligned-r={metrics['aligned_r']:.3f}  "
                  f"shift={metrics['shift']} {metrics['orient']}  "
                  f"top-donor={neighbors[0][3]} ({neighbors[0][1]:.3f})")
            preds.append(pred_pwm)
            results.append({"name": s["name"], "dna_gt": s["dna_seq"], **metrics,
                            "top_donor": neighbors[0][3], "top_cos": neighbors[0][1]})

        model_preds[mkey]   = preds
        model_results[mkey] = results

        # free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()
        print()

    # ── Figure: GT | E2 | Deploy  (n_samples rows × 3 cols) ──────────────────
    n_samples = len(samples)
    n_cols    = 1 + len(MODELS)   # GT + one col per model
    fig, axes = plt.subplots(n_samples, n_cols,
                             figsize=(5 * n_cols, 2.8 * n_samples), squeeze=False)

    col_titles = ["Ground truth"] + [MODELS[k]["label"] for k in MODELS]
    col_colors = ["#222222"]      + [MODELS[k]["color"]  for k in MODELS]

    for i, s in enumerate(samples):
        gt_pwm = s["gt_pwm"]
        draw_logo(axes[i, 0], gt_pwm,
                  f"{s['name']}  |  GT: {s['dna_seq']}", color="#222222")

        for j, (mkey, mcfg) in enumerate(MODELS.items(), start=1):
            m    = model_results[mkey][i]
            pred = model_preds[mkey][i]
            draw_logo(axes[i, j], pred,
                      f"{mcfg['label']}\n"
                      f"panel-r={m['panel_r']:.3f}  aligned-r={m['aligned_r']:.3f}"
                      f"  shift={m['shift']} {m['orient']}",
                      color=mcfg["color"])

    # Column headers on first row
    for j, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, j].set_title(
            title + "\n" + axes[0, j].get_title(),
            fontsize=9, color=color, pad=3, fontweight="bold"
        )

    fig.suptitle("TFScope: designed zinc finger predictions vs DNA ground truth",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_fig = os.path.join(OUT_DIR, "zf_e2_vs_deploy.png")
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    print(f"Saved figure: {out_fig}")
    plt.close(fig)

    # ── Combined comparison table ──────────────────────────────────────────────
    print("\n=== Comparison Table ===")
    header = f"{'Name':<8} {'DNA GT':<12}"
    for mkey in MODELS:
        header += f"  {mkey+' panel-r':>14} {mkey+' aligned-r':>15} {mkey+' MAE':>10}"
    print(header)
    print("-" * len(header))

    all_results = {}
    for mkey in MODELS:
        all_results[mkey] = {r["name"]: r for r in model_results[mkey]}

    for s in samples:
        name = s["name"]
        row  = f"{name:<8} {s['dna_seq']:<12}"
        for mkey in MODELS:
            r = all_results[mkey][name]
            row += f"  {r['panel_r']:>14.3f} {r['aligned_r']:>15.3f} {r['fixed_mae']:>10.3f}"
        print(row)

    # Mean row
    print("-" * len(header))
    row = f"{'Mean':<8} {'':<12}"
    for mkey in MODELS:
        vals = model_results[mkey]
        row += (f"  {np.mean([v['panel_r'] for v in vals]):>14.3f}"
                f" {np.mean([v['aligned_r'] for v in vals]):>15.3f}"
                f" {np.mean([v['fixed_mae'] for v in vals]):>10.3f}")
    print(row)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "zf_e2_vs_deploy_metrics.json")
    with open(out_json, "w") as fh:
        json.dump({k: v for k, v in model_results.items()}, fh, indent=2)
    print(f"\nSaved metrics: {out_json}")


if __name__ == "__main__":
    main()
