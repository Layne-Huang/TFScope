#!/usr/bin/env python
"""GATA-family positive controls for the ZGLP1 case study.

Two controls per GATA factor (GATA4, GATA6), using each factor's stored training
DBD window and its curated motif as ground truth:

  A) retrieval-masked recovery: cluster40_v18a_rag with this gene (and ZGLP1)
     excluded from retrieval -> does the noRAG->RAG workflow recover the known GATA
     motif?  (gene IS in cluster40 training, so retrieval-masked, not train-masked;
     analogous to the SOHLH2 control.)

  B) family-masked de-novo: lofo/Other checkpoint (never saw ANY GATA) -> confirms
     the clean checkpoint's failure to produce GATA is a fully-unseen-family effect,
     not specific to ZGLP1.
"""
import os, sys, json, re, urllib.request, urllib.parse
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd
from cs_utils import (load_cfg, load_model, column_ic, active_cols, aligned_r, tokens_from_seq,
                      infer, build_retrieval, BASES)


def gata_dbd_window(gene):
    """Fetch the C-terminal (DNA-binding) GATA-type zinc finger + basic tail from
    UniProt, mirroring the ZGLP1 window convention (ZF start-8 .. ZF end+25)."""
    fields = "accession,ft_zn_fing,sequence"
    url = ("https://rest.uniprot.org/uniprotkb/search?query=" +
           urllib.parse.quote(f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true") +
           f"&fields={fields}&format=tsv&size=1")
    p = urllib.request.urlopen(url, timeout=60).read().decode().strip().split("\n")
    if len(p) < 2: return None, None
    acc, znf, seq = p[1].split("\t")
    spans = [(int(a), int(b)) for a, b in re.findall(r"ZN_FING (\d+)\.\.(\d+)", znf)]
    if not spans: return None, None
    zs, ze = spans[-1]                     # C-terminal finger = primary DNA-contacting
    s = max(1, zs - 8); e = min(len(seq), ze + 25)
    return seq[s - 1:e], acc

cfg = load_cfg("configs/case_study_zglp1.yaml")
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

m_prod, _ = load_model(cfg["checkpoint_production"], force_retrieval=True)
m_clean, _ = load_model(cfg["checkpoint_clean"], force_retrieval=False)

rows = []
for gene in cfg["gata_control_genes"]:
    sub = df[df.g == gene]
    # representative curated motif + its DBD window
    cand = [fn for fn in sub.filename if (".MA" in fn or "H13CORE.0" in fn)]
    fn0 = cand[0] if cand else sub.filename.iloc[0]
    ref = trim(fn2pwm[fn0])
    # proper C-terminal GATA zinc-finger DBD window (UniProt), ZGLP1-matched convention
    seq, _acc = gata_dbd_window(gene)
    if seq is None: seq = fn2seq[fn0]
    tok, mask = tokens_from_seq(seq)
    # (A) retrieval-masked recovery (cluster40 RAG, exclude this gene + ZGLP1)
    rp, rm, rs, nbrs, sims, order = build_retrieval(seq, embs, donors, fn2gene, fn2pwm, K, ML,
                                                    exclude_genes=[gene, "ZGLP1"])
    g_nr, p_nr, _ = infer(m_prod, tok, mask, FID, ret=None)
    g_rag, p_rag, _ = infer(m_prod, tok, mask, FID, ret=(rp, rm, rs))
    core_nr = p_nr[:, active_cols(g_nr, GTH)]; core_rag = p_rag[:, active_cols(g_rag, GTH)]
    r_nr, _, _ = aligned_r(core_nr, ref); r_rag, icw, _ = aligned_r(core_rag, ref)
    # (B) family-masked de-novo (lofo/Other)
    g_cl, p_cl, _ = infer(m_clean, tok, mask, FID, ret=None)
    core_cl = p_cl[:, active_cols(g_cl, GTH)]; r_cl, _, _ = aligned_r(core_cl, ref)
    rows.append(dict(
        gene=gene, motif_file=fn0, gata_consensus="".join(BASES[ref.argmax(0)]),
        retrieved=";".join(g for g, _, _ in nbrs),
        prod_noRAG_consensus="".join(BASES[core_nr.argmax(0)]), r_prod_noRAG=round(r_nr, 3),
        prod_RAG_consensus="".join(BASES[core_rag.argmax(0)]), r_prod_RAG=round(r_rag, 3),
        r_prod_RAG_ICweighted=round(icw, 3),
        clean_deNovo_consensus="".join(BASES[core_cl.argmax(0)]), r_clean_deNovo=round(r_cl, 3),
        retrieval_recovered=bool(r_rag >= cfg["success_threshold_r"]),
        deNovo_recovered=bool(r_cl >= cfg["success_threshold_r"])))

ctl = pd.DataFrame(rows)
ctl.to_csv(f"{OUT}/GATA_masked_control_metrics.tsv", sep="\t", index=False)
print(ctl.to_string(index=False))
print(f"\nretrieval-masked recovery (RAG r>=0.6): {ctl.retrieval_recovered.mean()*100:.0f}% of GATA controls")
print(f"family-masked de-novo recovery (r>=0.6): {ctl.deNovo_recovered.mean()*100:.0f}% "
      f"-> confirms unseen-family de-novo failure")
