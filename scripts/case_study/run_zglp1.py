#!/usr/bin/env python
"""ZGLP1 (GATA-type) sequence-only motif nomination — honest dual-checkpoint run.

Three inference modes for the ZGLP1 GATA DBD (UniProt P0C6A0, ZF window 200-273):
  (1) CLEAN de-novo  : lofo/Other checkpoint (NEVER saw any GATA factor)  -> generalization test
  (2) PRODUCTION noRAG: cluster40_v18a_rag, retrieval off (ZGLP1 leaked in encoder)
  (3) PRODUCTION RAG  : cluster40_v18a_rag, leave-gene-out retrieval (ZGLP1 excluded)

All predictions are scored against (a) ZGLP1's own divergent HOCOMOCO H13CORE motif
(leaky ground truth, consensus ATGATCGAT) and (b) the canonical GATA family motif.
The honest finding: the family-masked model does NOT recover GATA de-novo; retrieval
of GATA paralogs recovers the GATA *core* but not ZGLP1's divergent flanking.
"""
import os, sys, json
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd
from cs_utils import (load_cfg, load_model, column_ic, active_cols, aligned_r, tokens_from_seq,
                      infer, confidence_score, build_retrieval, write_meme, write_pwm_tsv,
                      ebox_pwm, BASES)


def ungapped_identity(a, b):
    """Best ungapped overlap % identity between two AA strings (>=8 aligned)."""
    best = 0.0
    for off in range(-(len(b) - 4), len(a) - 4 + 1):
        m = n = 0
        for i in range(len(a)):
            j = i - off
            if 0 <= j < len(b):
                n += 1; m += (a[i] == b[j])
        if n >= 8:
            best = max(best, m / n)
    return best

cfg = load_cfg("configs/case_study_zglp1.yaml")
OUT = cfg["output_dir"]; ML = cfg["max_motif_length"]; K = cfg["retrieval_top_k"]
GTH = cfg["active_gate_threshold"]; FID = cfg["case_family_id"]
for s in ["predictions", "leakage"]:
    os.makedirs(f"{OUT}/{s}", exist_ok=True)


def gata_pwm(consensus="AGATAA"):
    """Canonical GATA reference PWM with degenerate W(AT)/R(AG)."""
    iupac = {"A": "A", "C": "C", "G": "G", "T": "T", "W": "AT", "R": "AG", "N": "ACGT"}
    L = len(consensus); m = np.zeros((4, L))
    for j, ch in enumerate(consensus):
        for b in iupac[ch]:
            m["ACGT".index(b), j] = 1.0
    return m / m.sum(0, keepdims=True)


# ── data ─────────────────────────────────────────────────────────────────────
df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
fn2gene = dict(zip(df["filename"], df["g"]))
fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
fn2seq  = dict(zip(df["filename"], df["sequence"]))


def trim(pwm):
    L = int((pwm.sum(0) > 1e-6).sum()); return pwm[:, :L] if L >= 2 else pwm


sp = json.load(open(cfg["cluster40_split"]))
train_val = set(sp["train"]) | set(sp.get("val", []))
embs = np.load(cfg["embeddings"])
donors = [fn for fn in embs.files if fn in train_val]

seq = cfg["case_dbd_sequence"]
tok, mask = tokens_from_seq(seq)
gt = trim(fn2pwm[cfg["ground_truth_filename"]])          # ZGLP1 H13CORE (divergent)
gata_exemplar = trim(fn2pwm[cfg["gata_exemplar_filename"]])
gata_canon = gata_pwm(cfg["canonical_gata"])

# ── (1) CLEAN de-novo: lofo/Other, family-masked (use_retrieval=false) ─────────
m_clean, _ = load_model(cfg["checkpoint_clean"], force_retrieval=False)
g_cl, p_cl, _ = infer(m_clean, tok, mask, FID, ret=None)
core_cl = p_cl[:, active_cols(g_cl, GTH)]

# ── (2,3) PRODUCTION cluster40 (leaky encoder) ────────────────────────────────
m_prod, _ = load_model(cfg["checkpoint_production"], force_retrieval=True)
rp, rm, rs, nbrs, sims, order = build_retrieval(seq, embs, donors, fn2gene, fn2pwm, K, ML,
                                                exclude_genes=cfg["retrieval_exclude_genes"])
g_nr, p_nr, attn_nr = infer(m_prod, tok, mask, FID, ret=None)
g_rag, p_rag, attn_rag = infer(m_prod, tok, mask, FID, ret=(rp, rm, rs))
core_nr = p_nr[:, active_cols(g_nr, GTH)]
m_rag = active_cols(g_rag, GTH); core_rag = p_rag[:, m_rag]

pd.DataFrame([dict(rank=i + 1, gene=g, cos_sim=round(s, 4),
                   family=df.loc[df.filename == fn, "family_name"].iloc[0], filename=fn)
              for i, (g, s, fn) in enumerate(nbrs)]).to_csv(
    f"{OUT}/predictions/ZGLP1_retrieved_neighbors.tsv", sep="\t", index=False)

for tag, core in [("clean_deNovo", core_cl), ("prod_noRAG", core_nr), ("prod_RAG_LGO", core_rag)]:
    write_pwm_tsv(f"{OUT}/predictions/ZGLP1_{tag}.pwm.tsv", core)
    write_meme(f"{OUT}/predictions/ZGLP1_{tag}.meme", f"ZGLP1_{tag}", core)
np.save(f"{OUT}/predictions/ZGLP1_attention.npy", attn_rag if attn_rag is not None else np.array([]))

# ── confidence (RAG prediction, same calibrated score as SOHLH1) ──────────────
mean_ic_rag = float(column_ic(core_rag).mean())
gate_conf = float(g_rag[m_rag].mean())
conf, cls, comps = confidence_score(cfg, mean_ic_rag, gate_conf)

# ── scoring vs ground truth + GATA references ─────────────────────────────────
def score(core, ref): return aligned_r(core, ref)[0]
rows = []
for tag, core in [("clean de-novo (family-masked)", core_cl),
                  ("production noRAG", core_nr),
                  ("production RAG (LGO)", core_rag)]:
    rows.append(dict(prediction=tag,
                     consensus="".join(BASES[core.argmax(0)]),
                     mean_IC_bits=round(float(column_ic(core).mean()), 3),
                     r_vs_ZGLP1_H13CORE=round(score(core, gt), 3),
                     r_vs_GATA_exemplar=round(score(core, gata_exemplar), 3),
                     r_vs_canonical_GATA=round(score(core, gata_canon), 3)))
metrics = pd.DataFrame(rows)
metrics.to_csv(f"{OUT}/predictions/ZGLP1_prediction_metrics.tsv", sep="\t", index=False)

# GATA-family panel: RAG prediction vs each GATA1-6 representative motif
gata_rows = []
for g in cfg["gata_reference_genes"]:
    cand = [fn for fn in df[df.g == g].filename if (".MA" in fn or "H13CORE.0" in fn)]
    if not cand: continue
    ref = trim(fn2pwm[cand[0]])
    gata_rows.append(dict(gata_gene=g, motif_file=cand[0],
                          gata_consensus="".join(BASES[ref.argmax(0)]),
                          r_RAG_vs_this_GATA=round(score(core_rag, ref), 3)))
gpanel = pd.DataFrame(gata_rows)
gpanel.to_csv(f"{OUT}/predictions/ZGLP1_RAG_vs_GATA_family.tsv", sep="\t", index=False)

# ── divergence analysis: align RAG pred to ZGLP1 H13CORE, per-column r ─────────
sys.path.insert(0, "src")
from tfscope.models.alignment import align_pwm
al, _, _, _ = align_pwm(core_rag, gt, max_shift=cfg["max_alignment_offset"], consider_revcomp=True)
Lc = min(al.shape[1], gt.shape[1])
from scipy.stats import pearsonr
percol = []
for j in range(Lc):
    a, b = gt[:, j], al[:, j]
    r = 0.0 if (np.std(a) < 1e-8 or np.std(b) < 1e-8) else pearsonr(a, b)[0]
    ic_gt = 2.0 + float((np.clip(a, 1e-9, 1) * np.log2(np.clip(a, 1e-9, 1))).sum())
    percol.append(dict(pos=j, gt_base=BASES[a.argmax()], pred_base=BASES[b.argmax()],
                       gt_IC_bits=round(ic_gt, 2), column_r=round(float(r), 3)))
pd.DataFrame(percol).to_csv(f"{OUT}/predictions/ZGLP1_divergence_percolumn.tsv", sep="\t", index=False)

# ── leakage audit (HONEST: ZGLP1 IS in cluster40 training) ────────────────────
train_dbds = [(fn2gene.get(fn), fn2seq.get(fn)) for fn in donors if isinstance(fn2seq.get(fn), str)]
def max_identity(qseq, exclude=None):
    best, who = 0.0, None
    for g, s in train_dbds:
        if exclude and g == exclude: continue
        idn = ungapped_identity(qseq, s)
        if idn > best: best, who = idn, g
    return best, who
id_excl, who_excl = max_identity(seq, exclude="ZGLP1")
ret_genes = [g for g, _, _ in nbrs]
audit = [dict(
    gene_symbol="ZGLP1", uniprot_id=cfg["case_uniprot"],
    in_curated_DB_motif_JASPAR=False,                  # no JASPAR/PBM/SELEX motif
    in_TFScope_training_motif=True,                    # HOCOMOCO H13CORE -> in training
    in_cluster40_train=True, clean_checkpoint_holds_out=True,  # lofo/Other masks it
    in_retrieval_index_LGO=False,                      # excluded from its own retrieval
    max_train_dbd_identity_excl_self=round(id_excl, 3), nearest_train_gene=who_excl,
    retrieved_neighbors=";".join(ret_genes),
    notes=("ZGLP1 lacks an experimentally-curated (JASPAR/PBM/SELEX) motif but HAS a "
           "HOCOMOCO H13CORE motif in TFScope training -> production run is encoder-leaky; "
           "lofo/Other checkpoint never saw any GATA (clean de-novo)."))]
pd.DataFrame(audit).to_csv(f"{OUT}/leakage/ZGLP1_leakage_audit.tsv", sep="\t", index=False)

# ── summary ───────────────────────────────────────────────────────────────────
core_match = float(np.mean([c["column_r"] for c in percol if c["gt_IC_bits"] >= 1.0]))
summ = dict(
    gene="ZGLP1", uniprot=cfg["case_uniprot"], dbd_window=cfg["case_dbd_uniprot"],
    clean_checkpoint=os.path.basename(os.path.dirname(cfg["checkpoint_clean"])),
    prod_checkpoint=os.path.basename(os.path.dirname(cfg["checkpoint_production"])),
    clean_deNovo_consensus="".join(BASES[core_cl.argmax(0)]),
    prod_noRAG_consensus="".join(BASES[core_nr.argmax(0)]),
    prod_RAG_consensus="".join(BASES[core_rag.argmax(0)]),
    RAG_mean_IC=mean_ic_rag, gate_prob=gate_conf, confidence=conf, confidence_class=cls,
    retrieved=";".join(ret_genes),
    r_RAG_vs_ZGLP1_H13CORE=score(core_rag, gt),
    r_RAG_vs_GATA_exemplar=score(core_rag, gata_exemplar),
    r_cleanDeNovo_vs_GATA_exemplar=score(core_cl, gata_exemplar),
    gata_family_r_mean=float(gpanel["r_RAG_vs_this_GATA"].mean()) if len(gpanel) else None,
    core_column_r_highIC=core_match,
    ground_truth_consensus="".join(BASES[gt.argmax(0)]))
json.dump(summ, open(f"{OUT}/predictions/ZGLP1_prediction_summary.json", "w"), indent=2, default=float)

print("\n=== ZGLP1 prediction metrics ===")
print(metrics.to_string(index=False))
print("\n=== RAG vs GATA family ===")
print(gpanel.to_string(index=False))
print(f"\nretrieved neighbours (ZGLP1 LGO): {ret_genes}")
print(f"confidence {conf:.2f} ({cls}); RAG core (high-IC col) r vs ZGLP1 H13CORE = {core_match:.2f}")
print(json.dumps(summ, indent=2, default=float))
