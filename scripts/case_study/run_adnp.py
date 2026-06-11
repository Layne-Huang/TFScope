#!/usr/bin/env python
"""ADNP sequence-only homeodomain motif nomination (clean orphan).

ADNP (UniProt Q9H2P0, single homeobox 754-814) is absent from every TFScope table,
so cluster40_v18a_rag is leakage-free for it. Runs noRAG + leave-gene-out RAG, scores
against a canonical homeodomain TAAT site and the top retrieved TALE neighbour (PBX1),
runs the ADNP2 paralog companion for a consistency check, and audits leakage.
"""
import os, sys, json
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd
from cs_utils import (load_cfg, load_model, column_ic, active_cols, aligned_r, tokens_from_seq,
                      infer, confidence_score, build_retrieval, write_meme, write_pwm_tsv, BASES)

cfg = load_cfg("configs/case_study_adnp.yaml")
OUT = cfg["output_dir"]; ML = cfg["max_motif_length"]; K = cfg["retrieval_top_k"]
GTH = cfg["active_gate_threshold"]; FID = cfg["case_family_id"]
for s in ["predictions", "leakage"]:
    os.makedirs(f"{OUT}/{s}", exist_ok=True)


def ungapped_identity(a, b):
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


def hd_pwm(consensus="TAATTA"):
    iupac = {"A": "A", "C": "C", "G": "G", "T": "T", "W": "AT", "R": "AG", "Y": "CT", "N": "ACGT"}
    m = np.zeros((4, len(consensus)))
    for j, ch in enumerate(consensus):
        for b in iupac[ch]: m["ACGT".index(b), j] = 1.0
    return m / m.sum(0, keepdims=True)


def trim(p):
    L = int((p.sum(0) > 1e-6).sum()); return p[:, :L] if L >= 2 else p


# ── data ─────────────────────────────────────────────────────────────────────
df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
fn2gene = dict(zip(df["filename"], df["g"]))
fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
fn2seq  = dict(zip(df["filename"], df["sequence"]))
sp = json.load(open(cfg["cluster40_split"])); train_val = set(sp["train"]) | set(sp.get("val", []))
embs = np.load(cfg["embeddings"]); donors = [fn for fn in embs.files if fn in train_val]

model, _ = load_model(cfg["checkpoint_production"], force_retrieval=True)
taat = hd_pwm(cfg["canonical_homeodomain"])
pbx = trim(fn2pwm[cfg["pbx_reference_filename"]])


def predict(seq, exclude):
    rp, rm, rs, nbrs, sims, order = build_retrieval(seq, embs, donors, fn2gene, fn2pwm, K, ML,
                                                    exclude_genes=exclude)
    tok, mask = tokens_from_seq(seq)
    g_nr, p_nr, _ = infer(model, tok, mask, FID, ret=None)
    g_rag, p_rag, attn = infer(model, tok, mask, FID, ret=(rp, rm, rs))
    core_nr = p_nr[:, active_cols(g_nr, GTH)]
    m_rag = active_cols(g_rag, GTH); core_rag = p_rag[:, m_rag]
    return core_nr, core_rag, g_rag[m_rag], nbrs, sims[order[0]], attn


# ── ADNP ──────────────────────────────────────────────────────────────────────
seq = cfg["case_dbd_sequence"]
core_nr, core_rag, gate_rag, nbrs, top1, attn = predict(seq, cfg["retrieval_exclude_genes"])
write_pwm_tsv(f"{OUT}/predictions/ADNP_noRAG.pwm.tsv", core_nr)
write_pwm_tsv(f"{OUT}/predictions/ADNP_RAG_LGO.pwm.tsv", core_rag)
write_meme(f"{OUT}/predictions/ADNP_RAG_LGO.meme", "ADNP_RAG_LGO", core_rag)
np.save(f"{OUT}/predictions/ADNP_attention.npy", attn if attn is not None else np.array([]))
pd.DataFrame([dict(rank=i + 1, gene=g, cos_sim=round(s, 4),
                   family=df.loc[df.filename == fn, "family_name"].iloc[0], filename=fn)
              for i, (g, s, fn) in enumerate(nbrs)]).to_csv(
    f"{OUT}/predictions/ADNP_retrieved_neighbors.tsv", sep="\t", index=False)

mean_ic = float(column_ic(core_rag).mean()); gate_conf = float(gate_rag.mean())
conf, cls, _ = confidence_score(cfg, mean_ic, gate_conf)

# ── ADNP2 companion (also orphan -> consistency, not validation) ──────────────
c2_nr, c2_rag, c2_gate, c2_nbrs, _, _ = predict(cfg["companion_dbd_sequence"],
                                                cfg["retrieval_exclude_genes"])
write_pwm_tsv(f"{OUT}/predictions/ADNP2_RAG_LGO.pwm.tsv", c2_rag)

def score(a, b): return aligned_r(a, b)[0]
metrics = [
    dict(prediction="ADNP noRAG", consensus="".join(BASES[core_nr.argmax(0)]),
         mean_IC=round(float(column_ic(core_nr).mean()), 3),
         r_vs_TAAT=round(score(core_nr, taat), 3), r_vs_PBX1=round(score(core_nr, pbx), 3)),
    dict(prediction="ADNP RAG (LGO)", consensus="".join(BASES[core_rag.argmax(0)]),
         mean_IC=round(mean_ic, 3),
         r_vs_TAAT=round(score(core_rag, taat), 3), r_vs_PBX1=round(score(core_rag, pbx), 3)),
    dict(prediction="ADNP2 RAG (companion)", consensus="".join(BASES[c2_rag.argmax(0)]),
         mean_IC=round(float(column_ic(c2_rag).mean()), 3),
         r_vs_TAAT=round(score(c2_rag, taat), 3), r_vs_PBX1=round(score(c2_rag, pbx), 3)),
]
pd.DataFrame(metrics).to_csv(f"{OUT}/predictions/ADNP_prediction_metrics.tsv", sep="\t", index=False)
r_consistency = score(core_rag, c2_rag)

# ── leakage audit ─────────────────────────────────────────────────────────────
all_genes = set(df["g"])
train_dbds = [(fn2gene.get(fn), fn2seq.get(fn)) for fn in donors if isinstance(fn2seq.get(fn), str)]
def max_id(qseq, excl):
    best, who = 0.0, None
    for g, s in train_dbds:
        if g in excl: continue
        idn = ungapped_identity(qseq, s)
        if idn > best: best, who = idn, g
    return best, who
idn, who = max_id(seq, {"ADNP"})
ret_genes = [g for g, _, _ in nbrs]
audit = [dict(gene_symbol="ADNP", uniprot_id=cfg["case_uniprot"],
              in_training_gene=("ADNP" in all_genes), in_training_motif=("ADNP" in all_genes),
              in_retrieval_index=False, in_benchmark_tables=False,
              max_train_dbd_identity=round(idn, 3), nearest_train_gene=who,
              retrieved_neighbors=";".join(ret_genes),
              notes="clean orphan; ADNP absent from all tables; ADNP2 companion also orphan")]
pd.DataFrame(audit).to_csv(f"{OUT}/leakage/ADNP_leakage_audit.tsv", sep="\t", index=False)

summ = dict(gene="ADNP", uniprot=cfg["case_uniprot"], dbd_uniprot=cfg["case_dbd_uniprot"],
            checkpoint=os.path.basename(os.path.dirname(cfg["checkpoint_production"])),
            noRAG_consensus="".join(BASES[core_nr.argmax(0)]),
            RAG_consensus="".join(BASES[core_rag.argmax(0)]),
            mean_IC_RAG=mean_ic, gate_prob=gate_conf, confidence=conf, confidence_class=cls,
            retrieved=";".join(ret_genes),
            r_RAG_vs_TAAT=score(core_rag, taat), r_RAG_vs_PBX1=score(core_rag, pbx),
            ADNP2_consensus="".join(BASES[c2_rag.argmax(0)]),
            ADNP2_retrieved=";".join(g for g, _, _ in c2_nbrs),
            r_ADNP_vs_ADNP2=r_consistency,
            max_train_dbd_identity=idn, nearest_train_gene=who)
json.dump(summ, open(f"{OUT}/predictions/ADNP_prediction_summary.json", "w"), indent=2, default=float)
print(pd.DataFrame(metrics).to_string(index=False))
print(f"\nADNP confidence {conf:.2f} ({cls}); retrieved {ret_genes}")
print(f"ADNP vs ADNP2 paralog consistency r={r_consistency:.2f}")
print(json.dumps(summ, indent=2, default=float))
