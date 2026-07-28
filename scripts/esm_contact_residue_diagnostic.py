#!/usr/bin/env python
"""Frozen ESM-2 residue-level DNA-contact diagnostic.

Tests whether frozen esm2_t33_650M_UR50D residue embeddings contain signal for
which DBD residues contact DNA. Contacts are computed directly from the
protein-DNA co-crystal structures (heavy-atom distance <= 4.5 A); no TFScope
model output is used.

Pipeline
--------
1. Parse each complex with Biopython; heavy-atom contact = any protein residue
   heavy atom within 4.5 A of any DNA heavy atom.
2. The i-th modeled protein residue maps to sequence position i+1; keep residues
   whose position lies in an annotated DBD range.
3. Frozen ESM-2 residue embeddings (layer 33) as features; label = contact.
4. GroupKFold (5-fold) grouped by protein sequence (identical DBD -> same group)
   so no TF leaks across folds. Logistic-regression probe.
5. Report AUROC / AUPRC (pooled out-of-fold) and per-TF enrichment@5 / @10.
6. Figures: ROC+PR curves; one case-study residue-score track.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd

NUC = {"DA", "DC", "DG", "DT", "DU", "DI", "DN"}
CONTACT_CUTOFF = 4.5
TEAL, ORANGE = "#2A9D8F", "#E76F51"


# --------------------------------------------------------------------------- #
# Structure parsing + contacts
# --------------------------------------------------------------------------- #
def parse_complex(pdb_file, protein_chain, dna_chains, cutoff=CONTACT_CUTOFF):
    """Return (sequence, contact_mask) over ordered modeled protein residues."""
    from Bio.PDB import PDBParser, is_aa, NeighborSearch
    from Bio.Data.PDBData import protein_letters_3to1_extended as T2O

    st = PDBParser(QUIET=True).get_structure("x", pdb_file)
    model = next(st.get_models())
    chain_ids = {c.id for c in model.get_chains()}
    if protein_chain not in chain_ids:
        return None, None

    prot_res, seq = [], []
    for res in model[protein_chain].get_residues():
        if not is_aa(res, standard=False) or res.id[0].strip():
            continue
        one = T2O.get(res.get_resname().upper(), "X") or "X"
        seq.append(one)
        prot_res.append(res)
    if not prot_res:
        return None, None

    # DNA heavy atoms: prefer the annotated DNA chains, else any nucleotide.
    want = set(dna_chains)
    dna_atoms = []
    for ch in model.get_chains():
        use = ch.id in want
        for res in ch.get_residues():
            if res.get_resname().strip().upper() in NUC:
                if use or not want:
                    dna_atoms += [a for a in res.get_atoms()
                                  if a.element not in ("H", "D")]
    if not dna_atoms:  # fallback: every nucleotide in the structure
        for res in model.get_residues():
            if res.get_resname().strip().upper() in NUC:
                dna_atoms += [a for a in res.get_atoms()
                              if a.element not in ("H", "D")]
    if not dna_atoms:
        return "".join(seq), None

    ns = NeighborSearch(dna_atoms)
    mask = np.zeros(len(prot_res), dtype=bool)
    for i, res in enumerate(prot_res):
        for atom in res.get_atoms():
            if atom.element in ("H", "D"):
                continue
            if ns.search(atom.coord, cutoff):
                mask[i] = True
                break
    return "".join(seq), mask


def parse_ranges(s):
    out = []
    for part in str(s).split(";"):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


# --------------------------------------------------------------------------- #
# ESM
# --------------------------------------------------------------------------- #
ESM_MAX = 1022


def esm_residue_embeddings(seqs, device, batch_tokens, use_fp16):
    """seqs: list[str] -> list[np.ndarray (Li, D)] aligned to input order."""
    import torch, esm
    print("[esm] loading esm2_t33_650M_UR50D ...", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    model = model.to(device).eval()
    if use_fp16 and device.startswith("cuda"):
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)

    # Truncate over-long sequences (rare); embeddings only needed for DBD, which
    # sits well within the first ESM_MAX residues for these DBD constructs.
    trunc = [s[:ESM_MAX] for s in seqs]
    order = sorted(range(len(trunc)), key=lambda k: len(trunc[k]))
    embs = [None] * len(trunc)

    batch, bmax, done = [], 0, 0

    def flush(batch):
        nonlocal done
        if not batch:
            return
        data = [(str(k), trunc[k]) for k in batch]
        _, _, toks = bc(data)
        toks = toks.to(device)
        with torch.no_grad():
            rep = model(toks, repr_layers=[33])["representations"][33].float()
        for j, k in enumerate(batch):
            L = len(trunc[k])
            embs[k] = rep[j, 1:L + 1].cpu().numpy()  # drop BOS
        done += len(batch)
        print(f"[esm] embedded {done}/{len(trunc)}", flush=True)

    for k in order:
        tl = len(trunc[k]) + 2
        if batch and max(bmax, tl) * (len(batch) + 1) > batch_tokens:
            flush(batch); batch, bmax = [], 0
        batch.append(k); bmax = max(bmax, tl)
    flush(batch)
    return embs


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 300,
    })
    return plt


def plot_curves(y, s, auroc, auprc, base_rate, out_png):
    from sklearn.metrics import roc_curve, precision_recall_curve
    plt = _style()
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.1))
    fpr, tpr, _ = roc_curve(y, s)
    ax[0].plot(fpr, tpr, color=TEAL, lw=1.8)
    ax[0].plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls="--")
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title(f"ROC (AUROC = {auroc:.3f})", fontsize=9)
    prec, rec, _ = precision_recall_curve(y, s)
    ax[1].plot(rec, prec, color=ORANGE, lw=1.8)
    ax[1].axhline(base_rate, color="#BBBBBB", lw=0.8, ls="--")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title(f"PR (AUPRC = {auprc:.3f}, base {base_rate:.2f})", fontsize=9)
    for a in ax:
        a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
        a.grid(False); a.set_facecolor("white")
    fig.patch.set_facecolor("white"); fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(Path(out_png).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_case_study(df_case, title, out_png):
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    pos = df_case["position"].to_numpy()
    score = df_case["score"].to_numpy()
    ax.plot(pos, score, color=TEAL, lw=1.3, zorder=2)
    ax.fill_between(pos, 0, score, color=TEAL, alpha=0.12, zorder=1)
    tc = df_case[df_case["label"] == 1]
    ax.scatter(tc["position"], tc["score"], marker="^", s=42, color=ORANGE,
               edgecolors="white", linewidths=0.5, zorder=3,
               label="true DNA-contact residue")
    ax.set_xlabel("DBD residue position"); ax.set_ylabel("Predicted contact score")
    ax.set_title(title, fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(False); ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(Path(out_png).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-tokens", type=int, default=16000)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-emb", action="store_true")
    ap.add_argument("--emb-cache", default=None,
                    help="path for the feature cache (defaults to outdir); "
                         "point at big storage to avoid home-quota limits")
    args = ap.parse_args()

    import torch
    device = args.device if (args.device.startswith("cuda")
                             and torch.cuda.is_available()) else "cpu"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.input)
    print(f"[data] {len(meta)} complexes", flush=True)

    # --- structures -> residue table + per-complex sequence ---
    rows, seqs, seq_index = [], [], []
    n_no_dna = 0
    for _, r in meta.iterrows():
        seq, mask = parse_complex(r["pdb_file"], str(r["protein_chain"]),
                                  str(r["dna_chains"]))
        if seq is None or mask is None:
            n_no_dna += 1
            continue
        ranges = parse_ranges(r["dbd_ranges"])
        in_dbd = np.zeros(len(seq), dtype=bool)
        for a, b in ranges:
            in_dbd[a - 1:b] = True  # 1-based inclusive
        ci = len(seqs)
        seqs.append(seq)
        for i in range(len(seq)):
            if not in_dbd[i]:
                continue
            rows.append({
                "complex_id": r["complex_id"], "family": r["family"],
                "pdb_id": r.get("pdb_id", r["complex_id"][:4]),
                "seq_group": hashlib.md5(seq.encode()).hexdigest()[:12],
                "position": i + 1, "amino_acid": seq[i],
                "label": int(mask[i]), "_ci": ci, "_i": i,
            })
        seq_index.append(ci)
    res = pd.DataFrame(rows)
    print(f"[data] {len(seqs)} usable complexes ({n_no_dna} lacked DNA); "
          f"{len(res)} DBD residues, {res['label'].mean():.3f} contact rate",
          flush=True)

    # --- ESM residue embeddings ---
    emb_cache = Path(args.emb_cache) if args.emb_cache else out / "residue_embeddings.npz"
    if args.cache_emb and emb_cache.exists():
        d = np.load(emb_cache, allow_pickle=True)
        feats = d["feats"].astype(np.float32)
        print(f"[esm] loaded cached features {feats.shape}", flush=True)
    else:
        embs = esm_residue_embeddings(seqs, device, args.batch_tokens, args.fp16)
        feats = np.stack([embs[row["_ci"]][row["_i"]] for row in rows]).astype(np.float32)
        print(f"[esm] built feature matrix {feats.shape}", flush=True)
        try:  # cache is optional — a save failure must not lose the run
            emb_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(emb_cache, feats=feats.astype(np.float16))
            print(f"[esm] cached features -> {emb_cache}", flush=True)
        except OSError as e:
            print(f"[esm] WARNING: could not cache features ({e}); continuing",
                  flush=True)

    y = res["label"].to_numpy()
    groups = res["seq_group"].to_numpy()

    # --- GroupKFold probe (pooled out-of-fold predictions) ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score

    oof = np.full(len(y), np.nan)
    fold_metrics = []
    gkf = GroupKFold(n_splits=args.folds)
    for f, (tr, te) in enumerate(gkf.split(feats, y, groups)):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"))
        clf.fit(feats[tr], y[tr])
        p = clf.predict_proba(feats[te])[:, 1]
        oof[te] = p
        fold_metrics.append({
            "fold": f, "n_test": int(len(te)),
            "auroc": float(roc_auc_score(y[te], p)),
            "auprc": float(average_precision_score(y[te], p))})
        print(f"[cv] fold {f}: AUROC={fold_metrics[-1]['auroc']:.3f} "
              f"AUPRC={fold_metrics[-1]['auprc']:.3f}", flush=True)

    res["score"] = oof
    auroc = float(roc_auc_score(y, oof))
    auprc = float(average_precision_score(y, oof))
    base_rate = float(y.mean())

    # --- enrichment@k (per complex, ranked by out-of-fold score) ---
    def enrichment(group, k):
        g = group.sort_values("score", ascending=False)
        topk = g.head(k)
        n_true = int(group["label"].sum())
        hit = int(topk["label"].sum())
        return pd.Series({"n_true": n_true, f"hit_top{k}": hit,
                          f"recall_top{k}": hit / n_true if n_true else np.nan})
    per_complex = []
    for cid, g in res.groupby("complex_id"):
        e5 = enrichment(g, 5); e10 = enrichment(g, 10)
        per_complex.append({
            "complex_id": cid, "family": g["family"].iloc[0],
            "n_residues": len(g), "n_contacts": int(g["label"].sum()),
            "hit_top5": int(e5["hit_top5"]), "recall_top5": float(e5["recall_top5"]),
            "hit_top10": int(e10["hit_top10"]), "recall_top10": float(e10["recall_top10"]),
        })
    pc = pd.DataFrame(per_complex)

    # --- save outputs ---
    res_out = res.drop(columns=["_ci", "_i"])
    res_out.to_csv(out / "residue_probe_predictions.csv", index=False)
    pc.to_csv(out / "per_complex_enrichment.csv", index=False)

    summary = {
        "n_complexes": int(len(seqs)),
        "n_dbd_residues": int(len(res)),
        "contact_base_rate": base_rate,
        "esm_model": "esm2_t33_650M_UR50D", "esm_layer": 33,
        "contact_cutoff_angstrom": CONTACT_CUTOFF,
        "grouping": "protein sequence (GroupKFold, no TF across folds)",
        "n_folds": args.folds,
        "pooled_auroc": auroc, "pooled_auprc": auprc,
        "fold_metrics": fold_metrics,
        "mean_recall_top5": float(pc["recall_top5"].mean()),
        "mean_recall_top10": float(pc["recall_top10"].mean()),
        "mean_hits_top5": float(pc["hit_top5"].mean()),
        "mean_hits_top10": float(pc["hit_top10"].mean()),
        "median_contacts_per_complex": float(pc["n_contacts"].median()),
    }
    with open(out / "summary_metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    plot_curves(y, oof, auroc, auprc, base_rate,
                out / "contact_probe_auroc_auprc.png")

    # --- case study: a well-recovered complex of decent size ---
    cand = pc[(pc["n_contacts"] >= 5) & (pc["n_residues"] >= 20)].copy()
    cand = cand.sort_values(["recall_top10", "n_contacts"], ascending=False)
    case_id = cand.iloc[0]["complex_id"]
    case = res[res["complex_id"] == case_id].sort_values("position")
    fam = case["family"].iloc[0]
    plot_case_study(case, f"Case study: {case_id}  ({fam})",
                    out / "case_study_contact_track.png")
    case.drop(columns=["_ci", "_i"]).to_csv(out / "case_study_residues.csv", index=False)
    summary["case_study"] = {
        "complex_id": case_id, "family": str(fam),
        "recall_top10": float(cand.iloc[0]["recall_top10"]),
        "n_contacts": int(cand.iloc[0]["n_contacts"])}
    with open(out / "summary_metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== SUMMARY ===")
    print(f"pooled AUROC={auroc:.3f}  AUPRC={auprc:.3f} (base {base_rate:.3f})")
    print(f"mean recall@5={summary['mean_recall_top5']:.3f}  "
          f"recall@10={summary['mean_recall_top10']:.3f}")
    print(f"case study: {case_id} ({fam})")


if __name__ == "__main__":
    main()
