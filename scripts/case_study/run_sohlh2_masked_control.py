#!/usr/bin/env python
"""Retrieval-masked SOHLH2 positive control.

Treat the paralog SOHLH2 as if it were orphan: run the SAME production checkpoint
(cluster40_v18a_rag) and workflow used for SOHLH1, with SOHLH2 (and SOHLH1)
excluded from retrieved neighbours, then compare the prediction to the curated
JASPAR MA1560.1 motif. SOHLH2 IS in this checkpoint's training set, so this is a
RETRIEVAL-MASKED control (tests retrieval-time recovery), NOT a fully
train-and-retrieval-masked independent validation. Labelled accordingly.
"""
import os, sys, json
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd
from cs_utils import (load_cfg, load_model, column_ic, active_cols, aligned_r, tokens_from_seq,
                      infer, confidence_score, build_retrieval, write_pwm_tsv, ebox_pwm, BASES)

cfg = load_cfg(); OUT = f"{cfg['output_dir']}/validation"; os.makedirs(OUT, exist_ok=True)
ML = cfg["max_motif_length"]; K = cfg["retrieval_top_k"]; GTH = cfg["active_gate_threshold"]

df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
fn2gene = dict(zip(df["filename"], df["g"]))
fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
sp = json.load(open(cfg["cluster40_split"])); train_val = set(sp["train"]) | set(sp.get("val", []))
embs = np.load(cfg["embeddings"]); donors = [fn for fn in embs.files if fn in train_val]

seq2 = cfg["paralog_dbd_sequence"]
s2_pwm = fn2pwm[cfg["paralog_reference_filename"]]          # JASPAR MA1560.1
ebox = ebox_pwm(cfg["canonical_ebox"])

model, _ = load_model(cfg["checkpoint_production"], force_retrieval=True)
tok, mask = tokens_from_seq(seq2)
rp, rm, rs, nbrs, sims, order = build_retrieval(seq2, embs, donors, fn2gene, fn2pwm, K, ML,
                                                exclude_genes=cfg["retrieval_exclude_genes"])

g_nr, p_nr, _ = infer(model, tok, mask, cfg["case_family_id"], ret=None)
g_rag, p_rag, _ = infer(model, tok, mask, cfg["case_family_id"], ret=(rp, rm, rs))
core_nr = p_nr[:, active_cols(g_nr, GTH)]; core_rag = p_rag[:, active_cols(g_rag, GTH)]
write_pwm_tsv(f"{OUT}/SOHLH2_masked_noRAG.pwm.tsv", core_nr)
write_pwm_tsv(f"{OUT}/SOHLH2_masked_RAG_LGO.pwm.tsv", core_rag)

r_nr, _, _   = aligned_r(core_nr, s2_pwm)
r_rag, icw, _ = aligned_r(core_rag, s2_pwm)
r_nr_eb, _, _ = aligned_r(core_nr, ebox)
r_rag_eb, _, _ = aligned_r(core_rag, ebox)
mean_ic = float(column_ic(core_rag).mean()); gate = float(g_rag[active_cols(g_rag, GTH)].mean())
conf, cls, _ = confidence_score(cfg, mean_ic, gate)

rows = [
    dict(metric="masking_level", value="retrieval-masked (SOHLH2 in training; excluded from retrieval)"),
    dict(metric="retrieved_neighbors", value=";".join(g for g, _, _ in nbrs)),
    dict(metric="noRAG_consensus", value="".join(BASES[core_nr.argmax(0)])),
    dict(metric="RAG_consensus", value="".join(BASES[core_rag.argmax(0)])),
    dict(metric="r_noRAG_vs_JASPAR_MA1560.1", value=round(r_nr, 4)),
    dict(metric="r_RAG_vs_JASPAR_MA1560.1", value=round(r_rag, 4)),
    dict(metric="r_RAG_vs_JASPAR_ICweighted", value=round(icw, 4)),
    dict(metric="r_noRAG_vs_canonical_Ebox", value=round(r_nr_eb, 4)),
    dict(metric="r_RAG_vs_canonical_Ebox", value=round(r_rag_eb, 4)),
    dict(metric="RAG_mean_IC_bits", value=round(mean_ic, 3)),
    dict(metric="RAG_confidence", value=round(conf, 3)),
    dict(metric="recovered_known_motif (r>=0.6)", value=bool(r_rag >= cfg["success_threshold_r"])),
]
pd.DataFrame(rows).to_csv(f"{OUT}/SOHLH2_masked_control_metrics.tsv", sep="\t", index=False)
print(pd.DataFrame(rows).to_string(index=False))
