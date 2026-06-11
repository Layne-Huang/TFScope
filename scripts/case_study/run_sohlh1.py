#!/usr/bin/env python
"""SOHLH1 inference (noRAG + LGO-RAG), leakage audit, and metrics.
Writes into results/sohlh1_case/ per the revised plan. SOHLH1 AND SOHLH2 are
excluded from retrieved neighbours (paralog kept only as a post-hoc reference)."""
import os, sys, json
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd
from cs_utils import (load_cfg, load_model, column_ic, active_cols, aligned_r, tokens_from_seq,
                      infer, confidence_score, build_retrieval, write_meme, write_pwm_tsv,
                      ebox_pwm, BASES)

cfg = load_cfg()
OUT = cfg["output_dir"]; ML = cfg["max_motif_length"]; K = cfg["retrieval_top_k"]
GTH = cfg["active_gate_threshold"]
for s in ["predictions", "leakage"]:
    os.makedirs(f"{OUT}/{s}", exist_ok=True)


def ungapped_identity(a, b):
    """Best ungapped overlap % identity between two AA strings."""
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


# ── data ─────────────────────────────────────────────────────────────────────
df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
fn2gene = dict(zip(df["filename"], df["g"]))
fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
fn2seq  = dict(zip(df["filename"], df["sequence"]))
sp = json.load(open(cfg["cluster40_split"]))
train_val = set(sp["train"]) | set(sp.get("val", []))
embs = np.load(cfg["embeddings"])
donors = [fn for fn in embs.files if fn in train_val]

seq1 = cfg["case_dbd_sequence"]
model, _ = load_model(cfg["checkpoint_production"], force_retrieval=True)
tok, mask = tokens_from_seq(seq1)

# retrieval (exclude SOHLH1 + SOHLH2)
rp, rm, rs, nbrs, sims, order = build_retrieval(seq1, embs, donors, fn2gene, fn2pwm, K, ML,
                                                exclude_genes=cfg["retrieval_exclude_genes"])
ret = (rp, rm, rs)
pd.DataFrame([dict(rank=i + 1, gene=g, cos_sim=s,
                   family=df.loc[df.filename == fn, "family_name"].iloc[0], filename=fn)
             for i, (g, s, fn) in enumerate(nbrs)]).to_csv(
    f"{OUT}/predictions/SOHLH1_retrieved_neighbors.tsv", sep="\t", index=False)

# inference
g_nr, p_nr, attn_nr = infer(model, tok, mask, cfg["case_family_id"], ret=None)
g_rag, p_rag, attn_rag = infer(model, tok, mask, cfg["case_family_id"], ret=ret)
m_nr = active_cols(g_nr, GTH); core_nr = p_nr[:, m_nr]
m_rag = active_cols(g_rag, GTH); core_rag = p_rag[:, m_rag]
write_pwm_tsv(f"{OUT}/predictions/SOHLH1_noRAG.pwm.tsv", core_nr)
write_pwm_tsv(f"{OUT}/predictions/SOHLH1_RAG_LGO.pwm.tsv", core_rag)
write_meme(f"{OUT}/predictions/SOHLH1_noRAG.meme", "SOHLH1_noRAG", core_nr)
write_meme(f"{OUT}/predictions/SOHLH1_RAG_LGO.meme", "SOHLH1_RAG_LGO", core_rag)
np.save(f"{OUT}/predictions/SOHLH1_attention.npy", attn_rag if attn_rag is not None else np.array([]))
pd.DataFrame({"pos": range(ML), "gate_prob": g_rag}).to_csv(
    f"{OUT}/predictions/SOHLH1_gate_probs.tsv", sep="\t", index=False)

# confidence (corrected, calibrated score)
mean_ic = float(column_ic(core_rag).mean())
gate_conf = float(g_rag[m_rag].mean())
conf, cls, comps = confidence_score(cfg, mean_ic, gate_conf)

# reference comparison
s2_pwm = fn2pwm[cfg["paralog_reference_filename"]]
ebox = ebox_pwm(cfg["canonical_ebox"])
r_rn, _, _    = aligned_r(core_nr, core_rag)
r_nr_s2, _, _ = aligned_r(core_nr, s2_pwm)
r_rag_s2, icw_s2, _ = aligned_r(core_rag, s2_pwm)
r_nr_eb, _, _ = aligned_r(core_nr, ebox)
r_rag_eb, _, _ = aligned_r(core_rag, ebox)

metrics = [
    dict(comparison="SOHLH1 noRAG vs SOHLH1 RAG", r=r_rn),
    dict(comparison="SOHLH1 noRAG vs SOHLH2",     r=r_nr_s2),
    dict(comparison="SOHLH1 RAG vs SOHLH2",       r=r_rag_s2),
    dict(comparison="SOHLH1 RAG vs SOHLH2 (IC-weighted)", r=icw_s2),
    dict(comparison="SOHLH1 noRAG vs canonical E-box", r=r_nr_eb),
    dict(comparison="SOHLH1 RAG vs canonical E-box",   r=r_rag_eb),
    dict(comparison="SOHLH1 RAG mean IC (bits)",   r=mean_ic),
    dict(comparison="SOHLH1 noRAG mean IC (bits)", r=float(column_ic(core_nr).mean())),
    dict(comparison="SOHLH1 calibrated confidence", r=conf),
]
pd.DataFrame(metrics).to_csv(f"{OUT}/predictions/SOHLH1_prediction_metrics.tsv", sep="\t",
                             index=False, float_format="%.4f")

# ── leakage audit (expanded) ─────────────────────────────────────────────────
all_motif_genes = set(df["g"])
def gene_present(g): return g.upper() in all_motif_genes
# DBD identities
train_dbds = [(fn2gene.get(fn), fn2seq.get(fn)) for fn in donors if isinstance(fn2seq.get(fn), str)]
def max_identity(qseq, exclude_gene=None):
    best, who = 0.0, None
    for g, s in train_dbds:
        if exclude_gene and g == exclude_gene: continue
        idn = ungapped_identity(qseq, s)
        if idn > best: best, who = idn, g
    return best, who
s1_id, s1_who = max_identity(seq1, exclude_gene="SOHLH1")
s2seq = cfg["paralog_dbd_sequence"]
s2_id, s2_who = max_identity(s2seq, exclude_gene="SOHLH2")
ret_genes = [g for g, _, _ in nbrs]
audit = [
    dict(gene_symbol="SOHLH1", uniprot_id=cfg["case_uniprot"],
         in_training_gene=False, in_training_motif=False,
         in_retrieval_index=False, in_benchmark_tables=False,
         max_train_dbd_identity=round(s1_id, 3), nearest_train_gene=s1_who,
         max_retrieval_dbd_identity=round(float(sims[order[0]]), 3),
         same_gene_retrieval_allowed=False, same_gene_retrieval_found=False,
         same_paralog_retrieval_found=("SOHLH2" in ret_genes),
         notes="orphan; SOHLH1 absent from all tables; SOHLH2 excluded from retrieval"),
    dict(gene_symbol="SOHLH2", uniprot_id=cfg["paralog_reference_uniprot"],
         in_training_gene=gene_present("SOHLH2"), in_training_motif=gene_present("SOHLH2"),
         in_retrieval_index=True, in_benchmark_tables=True,
         max_train_dbd_identity=round(s2_id, 3), nearest_train_gene=s2_who,
         max_retrieval_dbd_identity=np.nan,
         same_gene_retrieval_allowed=False, same_gene_retrieval_found=False,
         same_paralog_retrieval_found=False,
         notes="paralog reference only; in training, so NOT independent validation"),
]
pd.DataFrame(audit).to_csv(f"{OUT}/leakage/SOHLH1_leakage_audit.tsv", sep="\t", index=False)

summ = dict(gene="SOHLH1", checkpoint=os.path.basename(os.path.dirname(cfg["checkpoint_production"])),
            noRAG_consensus="".join(BASES[core_nr.argmax(0)]),
            RAG_consensus="".join(BASES[core_rag.argmax(0)]),
            mean_IC_RAG=mean_ic, gate_prob=gate_conf, confidence=conf, confidence_class=cls,
            r_RAG_vs_SOHLH2=r_rag_s2, r_RAG_vs_Ebox=r_rag_eb,
            retrieved=";".join(ret_genes), max_train_dbd_identity=s1_id, nearest_train_gene=s1_who)
json.dump(summ, open(f"{OUT}/predictions/SOHLH1_prediction_summary.json", "w"), indent=2, default=float)
print(json.dumps(summ, indent=2, default=float))
print(f"\nleakage: SOHLH1 max train DBD identity = {s1_id:.2f} (nearest {s1_who}); "
      f"retrieved = {ret_genes}")
