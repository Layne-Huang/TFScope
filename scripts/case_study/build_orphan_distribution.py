#!/usr/bin/env python
"""Orphan bHLH TF confidence distribution (Fig 5a overlay + Extended 5b table).

Pulls human reviewed bHLH-domain TFs from UniProt, keeps those ABSENT from the
TFScope training motif tables (true orphans w.r.t. our data), extracts the bHLH
DBD window (UniProt domain, SOHLH2-matched +6 C-term), and runs the calibrated
confidence pipeline (no ground truth needed). Shows SOHLH1 is not cherry-picked.
"""
import os, sys, json, re, urllib.request, urllib.parse
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd
from cs_utils import (load_cfg, load_model, column_ic, active_cols, tokens_from_seq,
                      infer, confidence_score, build_retrieval, BASES)

cfg = load_cfg(); OUT = f"{cfg['output_dir']}/orphans"; os.makedirs(OUT, exist_ok=True)
ML = cfg["max_motif_length"]; K = cfg["retrieval_top_k"]; GTH = cfg["active_gate_threshold"]
N_MAX = 60

# ── fetch human bHLH TFs ─────────────────────────────────────────────────────
fields = "accession,gene_primary,length,ft_domain,sequence"
url = ("https://rest.uniprot.org/uniprotkb/search?query=" +
       urllib.parse.quote("organism_id:9606 AND reviewed:true AND ft_domain:bHLH") +
       f"&fields={fields}&format=tsv&size=200")
txt = urllib.request.urlopen(url, timeout=90).read().decode()
lines = txt.strip().split("\n")[1:]
cand = []
for ln in lines:
    p = ln.split("\t")
    if len(p) < 5: continue
    acc, gene, length, dom, seq = p[0], p[1].strip().upper(), p[2], p[3], p[4]
    m = re.search(r"DOMAIN (\d+)\.\.(\d+);[^\n]*bHLH", dom)
    if not m or not gene or not seq: continue
    ds, de = int(m.group(1)), int(m.group(2))
    cand.append(dict(gene=gene, uniprot=acc, length=int(length), dom_start=ds, dom_end=de, seq=seq))
print(f"fetched {len(cand)} human bHLH TFs from UniProt")

# ── filter to orphans (gene absent from training motif tables) ───────────────
df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
train_genes = set(df["g"])
fn2gene = dict(zip(df["filename"], df["g"]))
fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
sp = json.load(open(cfg["cluster40_split"])); train_val = set(sp["train"]) | set(sp.get("val", []))
embs = np.load(cfg["embeddings"]); donors = [fn for fn in embs.files if fn in train_val]

orphans = [c for c in cand if c["gene"] not in train_genes]
print(f"orphan bHLH (absent from training): {len(orphans)}")
orphans = orphans[:N_MAX]

model, _ = load_model(cfg["checkpoint_production"], force_retrieval=True)
rows = []
for c in orphans:
    # DBD window: domain start .. domain end +6 (SOHLH2-matched), capped
    ds = c["dom_start"]; de = min(c["dom_end"] + 6, c["length"])
    dbd = c["seq"][ds - 1:de]
    if len(dbd) < 12: continue
    tok, mask = tokens_from_seq(dbd)
    rp, rm, rs, nbrs, sims, order = build_retrieval(dbd, embs, donors, fn2gene, fn2pwm, K, ML,
                                                    exclude_genes=[c["gene"]])
    g_nr, p_nr, _ = infer(model, tok, mask, cfg["case_family_id"], ret=None)
    g_rag, p_rag, _ = infer(model, tok, mask, cfg["case_family_id"], ret=(rp, rm, rs))
    m_nr = active_cols(g_nr, GTH); m_rag = active_cols(g_rag, GTH)
    core_rag = p_rag[:, m_rag]
    mean_ic = float(column_ic(core_rag).mean()); gate = float(g_rag[m_rag].mean())
    conf, cls, _ = confidence_score(cfg, mean_ic, gate)
    rows.append(dict(gene_symbol=c["gene"], uniprot_id=c["uniprot"], family="bHLH",
                     dbd_start=ds, dbd_end=de, known_motif_status="orphan (no motif in TFScope)",
                     in_training_gene=False, in_retrieval_index=False,
                     confidence=conf, confidence_class=cls,
                     mean_IC_RAG=mean_ic, gate_confidence=gate,
                     RAG_consensus="".join(BASES[core_rag.argmax(0)]),
                     retrieval_top1_gene=nbrs[0][0], retrieval_top1_similarity=round(nbrs[0][1], 3),
                     retrieval_top3_genes=";".join(g for g, _, _ in nbrs)))
od = pd.DataFrame(rows).sort_values("confidence", ascending=False)
od.to_csv(f"{OUT}/orphan_tf_confidence_table.tsv", sep="\t", index=False)
od.to_csv(f"{cfg['output_dir']}/orphan_tf_confidence_table.tsv", sep="\t", index=False)
print(f"\nscored {len(od)} orphan bHLH TFs")
print(f"confidence: mean={od.confidence.mean():.3f} median={od.confidence.median():.3f} "
      f"range [{od.confidence.min():.3f},{od.confidence.max():.3f}]")
so = od[od.gene_symbol == "SOHLH1"]
if len(so):
    pct = (od.confidence < so.confidence.iloc[0]).mean() * 100
    print(f"SOHLH1 conf={so.confidence.iloc[0]:.3f} -> {pct:.0f}th percentile among orphan bHLH")
print("top 10:\n", od[["gene_symbol", "confidence", "confidence_class", "RAG_consensus",
                        "retrieval_top1_gene"]].head(10).to_string(index=False))
