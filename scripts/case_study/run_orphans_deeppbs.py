#!/usr/bin/env python
"""Sequence-only homeodomain motif nominations for the remaining orphans
(ADNP2, ZHX2, ZHX3) on the deeppbs v18a RAG checkpoint.

Per orphan: noRAG + leave-gene-out RAG, calibrated confidence, comparison to a
canonical TAAT site and the top retrieved neighbour, leakage audit. Plus paralog
consistency (ZHX2 vs ZHX3, ADNP2 vs ADNP) and masked homeodomain positive controls.
"""
import os, sys, json
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd, yaml
from cs_utils import (load_model, column_ic, active_cols, aligned_r, tokens_from_seq,
                      infer, confidence_score, build_retrieval, write_pwm_tsv, write_meme, BASES)

cfg = yaml.safe_load(open("configs/case_study_orphans_deeppbs.yaml"))
OUT = cfg["output_dir"]; ML = cfg["max_motif_length"]; K = cfg["retrieval_top_k"]
GTH = cfg["active_gate_threshold"]; FID = cfg["family_id"]
for s in ["predictions", "leakage", "validation"]:
    os.makedirs(f"{OUT}/{s}", exist_ok=True)


def hd_pwm(consensus="TAATTA"):
    iupac = {"A": "A", "C": "C", "G": "G", "T": "T", "W": "AT", "R": "AG", "Y": "CT", "N": "ACGT"}
    m = np.zeros((4, len(consensus)))
    for j, ch in enumerate(consensus):
        for b in iupac[ch]: m["ACGT".index(b), j] = 1.0
    return m / m.sum(0, keepdims=True)

def trim(p):
    L = int((p.sum(0) > 1e-6).sum()); return p[:, :L] if L >= 2 else p

def ungapped_identity(a, b):
    best = 0.0
    for off in range(-(len(b) - 4), len(a) - 4 + 1):
        m = n = 0
        for i in range(len(a)):
            j = i - off
            if 0 <= j < len(b): n += 1; m += (a[i] == b[j])
        if n >= 8: best = max(best, m / n)
    return best


df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
fn2gene = dict(zip(df["filename"], df["g"]))
fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
fn2seq  = dict(zip(df["filename"], df["sequence"]))
sp = json.load(open(cfg["deeppbs_split"])); train_val = set(sp["train"]) | set(sp.get("val", []))
embs = np.load(cfg["embeddings"]); donors = [fn for fn in embs.files if fn in train_val]
model, _ = load_model(cfg["checkpoint_production"])
taat = hd_pwm(cfg["canonical_homeodomain"])
EXCL = cfg["retrieval_exclude_genes"]
train_dbds = [(fn2gene.get(fn), fn2seq.get(fn)) for fn in donors if isinstance(fn2seq.get(fn), str)]


def predict(seq, gene):
    rp, rm, rs, nbrs, sims, order = build_retrieval(seq, embs, donors, fn2gene, fn2pwm, K, ML,
                                                    exclude_genes=EXCL)
    tok, mask = tokens_from_seq(seq)
    g_nr, p_nr, _ = infer(model, tok, mask, FID, ret=None)
    g_rag, p_rag, attn = infer(model, tok, mask, FID, ret=(rp, rm, rs))
    return (p_nr[:, active_cols(g_nr, GTH)], p_rag[:, active_cols(g_rag, GTH)],
            g_rag[active_cols(g_rag, GTH)], nbrs, attn)


# ── ADNP (auxiliary, for ADNP2 consistency) — its homeobox window ─────────────
ADNP_SEQ = "LALDPKGHEDDSYEARKSFLTKYFNKQPYPTRREIEKLAASLWLWKSDIASHFSNKRKKCVRDCEKYKPGV"
cores = {}
_, adnp_rag, _, _, _ = predict(ADNP_SEQ, "ADNP"); cores["ADNP"] = adnp_rag

# ── orphans ───────────────────────────────────────────────────────────────────
metrics, summaries, audit = [], {}, []
for o in cfg["orphans"]:
    gene = o["gene"]; seq = o["dbd_sequence"]
    core_nr, core_rag, gate, nbrs, attn = predict(seq, gene)
    cores[gene] = core_rag
    write_pwm_tsv(f"{OUT}/predictions/{gene}_noRAG.pwm.tsv", core_nr)
    write_pwm_tsv(f"{OUT}/predictions/{gene}_RAG_LGO.pwm.tsv", core_rag)
    write_meme(f"{OUT}/predictions/{gene}_RAG_LGO.meme", f"{gene}_RAG_LGO", core_rag)
    np.save(f"{OUT}/predictions/{gene}_attention.npy", attn if attn is not None else np.array([]))
    pd.DataFrame([dict(rank=i+1, gene=g, cos_sim=round(s, 4),
                       family=df.loc[df.filename == fn, "family_name"].iloc[0])
                  for i, (g, s, fn) in enumerate(nbrs)]).to_csv(
        f"{OUT}/predictions/{gene}_retrieved_neighbors.tsv", sep="\t", index=False)
    mean_ic = float(column_ic(core_rag).mean()); gate_conf = float(gate.mean())
    conf, cls, _ = confidence_score(cfg, mean_ic, gate_conf)
    top_nbr = nbrs[0][0]; top_fn = nbrs[0][2]; top_ref = trim(fn2pwm[top_fn])
    metrics.append(dict(gene=gene, dbd=o["dbd_uniprot"],
                        noRAG_consensus="".join(BASES[core_nr.argmax(0)]),
                        RAG_consensus="".join(BASES[core_rag.argmax(0)]),
                        mean_IC=round(mean_ic, 3), confidence=round(conf, 3), confidence_class=cls,
                        r_vs_TAAT=round(aligned_r(core_rag, taat)[0], 3),
                        top_neighbor=top_nbr, r_vs_top_neighbor=round(aligned_r(core_rag, top_ref)[0], 3),
                        retrieved=";".join(g for g, _, _ in nbrs)))
    idn, who = max(((ungapped_identity(seq, s), g) for g, s in train_dbds if g != gene), default=(0, None))
    audit.append(dict(gene_symbol=gene, uniprot_id=o["uniprot"],
                      in_training_motif=False, in_retrieval_index=False, in_benchmark=False,
                      max_train_dbd_identity=round(idn, 3), nearest_train_gene=who,
                      retrieved=";".join(g for g, _, _ in nbrs), note=o["note"]))
    summaries[gene] = dict(gene=gene, uniprot=o["uniprot"], dbd=o["dbd_uniprot"],
                           RAG_consensus="".join(BASES[core_rag.argmax(0)]), mean_IC=mean_ic,
                           confidence=conf, confidence_class=cls,
                           retrieved=";".join(g for g, _, _ in nbrs),
                           r_vs_TAAT=aligned_r(core_rag, taat)[0])

md = pd.DataFrame(metrics)
md.to_csv(f"{OUT}/predictions/orphan_prediction_metrics.tsv", sep="\t", index=False)
pd.DataFrame(audit).to_csv(f"{OUT}/leakage/orphan_leakage_audit.tsv", sep="\t", index=False)

# ── paralog consistency ───────────────────────────────────────────────────────
cons = []
for a, b in cfg["consistency_pairs"]:
    if a in cores and b in cores:
        cons.append(dict(pair=f"{a} vs {b}", r=round(aligned_r(cores[a], cores[b])[0], 3)))
pd.DataFrame(cons).to_csv(f"{OUT}/predictions/paralog_consistency.tsv", sep="\t", index=False)

# ── masked homeodomain positive controls ─────────────────────────────────────
ctl = []
for gene in cfg["homeodomain_control_genes"]:
    sub = df[(df.g == gene) & (df.family_name == "Homeodomain")]
    cand = [fn for fn in sub.filename if (".MA" in fn or "H13CORE.0" in fn)]
    fn0 = cand[0] if cand else sub.filename.iloc[0]
    ref = trim(fn2pwm[fn0]); seq = fn2seq[fn0]
    rp, rm, rs, nbrs, sims, order = build_retrieval(seq, embs, donors, fn2gene, fn2pwm, K, ML,
                                                    exclude_genes=[gene] + EXCL)
    g_nr, p_nr, _ = infer(model, *tokens_from_seq(seq), FID, ret=None)
    g_rag, p_rag, _ = infer(model, *tokens_from_seq(seq), FID, ret=(rp, rm, rs))
    core_rag = p_rag[:, active_cols(g_rag, GTH)]; r_rag, icw, _ = aligned_r(core_rag, ref)
    ctl.append(dict(gene=gene, curated_consensus="".join(BASES[ref.argmax(0)]),
                    RAG_consensus="".join(BASES[core_rag.argmax(0)]), r_RAG=round(r_rag, 3),
                    recovered=bool(r_rag >= cfg["success_threshold_r"])))
cdf = pd.DataFrame(ctl); cdf.to_csv(f"{OUT}/validation/homeodomain_masked_control_metrics.tsv", sep="\t", index=False)
json.dump(summaries, open(f"{OUT}/predictions/orphan_summaries.json", "w"), indent=2, default=float)

print("=== orphan nominations (deeppbs v18a RAG) ==="); print(md.to_string(index=False))
print("\n=== paralog consistency ==="); print(pd.DataFrame(cons).to_string(index=False))
print(f"\n=== masked homeodomain control: {cdf.recovered.mean()*100:.0f}% recover ===")
print(cdf.to_string(index=False))
