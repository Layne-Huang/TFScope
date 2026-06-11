#!/usr/bin/env python
"""Masked homeodomain positive controls for the ADNP case study.

For each in-training homeodomain TF (EN1, PITX1, ISL1, PBX1), run the production
workflow with that gene (and ADNP/ADNP2) excluded from retrieval, and check whether
noRAG->RAG recovers its curated motif. Validates the homeodomain pipeline that the
ADNP nomination relies on. These genes ARE in training (retrieval-masked, not
train-masked), analogous to the SOHLH2 control.
"""
import os, sys, json
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd
from cs_utils import (load_cfg, load_model, active_cols, aligned_r, tokens_from_seq,
                      infer, build_retrieval, BASES)

cfg = load_cfg("configs/case_study_adnp.yaml")
OUT = f"{cfg['output_dir']}/validation"; os.makedirs(OUT, exist_ok=True)
ML = cfg["max_motif_length"]; K = cfg["retrieval_top_k"]; GTH = cfg["active_gate_threshold"]
FID = cfg["case_family_id"]

df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
fn2gene = dict(zip(df["filename"], df["g"]))
fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
fn2seq  = dict(zip(df["filename"], df["sequence"]))
def trim(p):
    L = int((p.sum(0) > 1e-6).sum()); return p[:, :L] if L >= 2 else p
sp = json.load(open(cfg["cluster40_split"])); train_val = set(sp["train"]) | set(sp.get("val", []))
embs = np.load(cfg["embeddings"]); donors = [fn for fn in embs.files if fn in train_val]
model, _ = load_model(cfg["checkpoint_production"], force_retrieval=True)

rows = []
for gene in cfg["homeodomain_control_genes"]:
    sub = df[(df.g == gene) & (df.family_name == "Homeodomain")]
    cand = [fn for fn in sub.filename if (".MA" in fn or "H13CORE.0" in fn)]
    fn0 = cand[0] if cand else sub.filename.iloc[0]
    ref = trim(fn2pwm[fn0]); seq = fn2seq[fn0]
    tok, mask = tokens_from_seq(seq)
    rp, rm, rs, nbrs, sims, order = build_retrieval(seq, embs, donors, fn2gene, fn2pwm, K, ML,
                                                    exclude_genes=[gene, "ADNP", "ADNP2"])
    g_nr, p_nr, _ = infer(model, tok, mask, FID, ret=None)
    g_rag, p_rag, _ = infer(model, tok, mask, FID, ret=(rp, rm, rs))
    core_nr = p_nr[:, active_cols(g_nr, GTH)]; core_rag = p_rag[:, active_cols(g_rag, GTH)]
    r_nr, _, _ = aligned_r(core_nr, ref); r_rag, icw, _ = aligned_r(core_rag, ref)
    rows.append(dict(gene=gene, motif_file=fn0, curated_consensus="".join(BASES[ref.argmax(0)]),
                     retrieved=";".join(g for g, _, _ in nbrs),
                     noRAG_consensus="".join(BASES[core_nr.argmax(0)]), r_noRAG=round(r_nr, 3),
                     RAG_consensus="".join(BASES[core_rag.argmax(0)]), r_RAG=round(r_rag, 3),
                     r_RAG_ICweighted=round(icw, 3),
                     recovered=bool(r_rag >= cfg["success_threshold_r"])))
ctl = pd.DataFrame(rows)
ctl.to_csv(f"{OUT}/homeodomain_masked_control_metrics.tsv", sep="\t", index=False)
print(ctl.to_string(index=False))
print(f"\nretrieval-masked homeodomain recovery (RAG r>=0.6): {ctl.recovered.mean()*100:.0f}% "
      f"({ctl.recovered.sum()}/{len(ctl)})")
