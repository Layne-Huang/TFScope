#!/usr/bin/env python
"""Held-out KNOWN-TF confidence calibration (the core upgrade).

Runs the production RAG checkpoint over the cluster40 TEST partition (held out at
<=40% identity -> leakage-clean known TFs). For each TF computes the SAME
confidence features used for SOHLH1 AND the actual oracle_r against its curated
motif, so confidence can be calibrated against real motif-recovery accuracy.

Retrieval uses precomputed donor embeddings (gene-deduplicated, leave-gene-out),
so no ESM re-embedding is needed for the query.
"""
import os, sys, json
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd, torch
from cs_utils import (load_cfg, load_model, column_ic, active_cols, aligned_r,
                      tokens_from_seq, infer, confidence_score)

cfg = load_cfg()
OUT = f"{cfg['output_dir']}/confidence"; os.makedirs(OUT, exist_ok=True)
ML = cfg["max_motif_length"]; K = cfg["retrieval_top_k"]; GTH = cfg["active_gate_threshold"]

# ── data ─────────────────────────────────────────────────────────────────────
df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
fn2gene = dict(zip(df["filename"], df["g"]))
fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
fn2seq  = dict(zip(df["filename"], df["sequence"]))
fn2fam  = dict(zip(df["filename"], df["family_id"]))
fn2famn = dict(zip(df["filename"], df["family_name"]))
sp = json.load(open(cfg["cluster40_split"]))
train_val = set(sp["train"]) | set(sp.get("val", []))
test_fns = sp["test"]
embs = np.load(cfg["embeddings"])

# donor pool (train+val); query is each test TF's precomputed embedding
donors = [fn for fn in embs.files if fn in train_val]
Dmat = np.stack([embs[fn] / (np.linalg.norm(embs[fn]) + 1e-8) for fn in donors])
donor_genes = np.array([fn2gene.get(fn, "?") for fn in donors])

model, _ = load_model(cfg["checkpoint_production"], force_retrieval=True)

def retrieval_for(query_fn):
    q = embs[query_fn]; q = q / (np.linalg.norm(q) + 1e-8)
    sims = Dmat @ q
    order = np.argsort(-sims)
    qg = fn2gene.get(query_fn, "?")
    top, seen = [], set()
    for di in order:
        g = donor_genes[di]
        if g == qg or g in seen:    # leave-gene-out
            continue
        seen.add(g); top.append(di)
        if len(top) == K: break
    rp = torch.full((1, K, 4, ML), 0.25); rm = torch.zeros((1, K, ML)); rs = torch.zeros((1, K))
    for ki, di in enumerate(top):
        pwm = fn2pwm[donors[di]]; L = min(pwm.shape[1], ML)
        rp[0, ki, :, :L] = torch.from_numpy(pwm[:, :L].copy()); rm[0, ki, :L] = 1.0
        rs[0, ki] = float(sims[di])
    return (rp, rm, rs), float(sims[order[0]] if len(order) else 0.0), \
           [donor_genes[di] for di in top]

rows = []
seen_gene = set()
for i, fn in enumerate(test_fns):
    g = fn2gene.get(fn, "?")
    if g in seen_gene:               # one row per gene (best motif source already canon)
        continue
    seen_gene.add(g)
    seq = fn2seq.get(fn);
    if not isinstance(seq, str) or len(seq) < 4: continue
    tok, mask = tokens_from_seq(seq)
    fam_id = int(fn2fam.get(fn, 9)); famn = fn2famn.get(fn, "Other")
    ret, top1, topg = retrieval_for(fn)
    g_nr, p_nr, _ = infer(model, tok, mask, fam_id, ret=None)
    g_rag, p_rag, _ = infer(model, tok, mask, fam_id, ret=ret)
    m_rag = active_cols(g_rag, GTH); core_rag = p_rag[:, m_rag]
    m_nr = active_cols(g_nr, GTH); core_nr = p_nr[:, m_nr]
    sim_rn, _, _ = aligned_r(core_nr, core_rag)
    gate_conf = float(g_rag[m_rag].mean())
    mean_ic = float(column_ic(core_rag).mean())
    conf, cls, comps = confidence_score(cfg, mean_ic, gate_conf)
    # ground-truth oracle_r (RAG prediction vs curated motif)
    tgt = fn2pwm[fn]; Lt = int((tgt.sum(0) > 1e-6).sum()); tgt = tgt[:, :Lt] if Lt >= 2 else tgt
    orc, icw, _ = aligned_r(core_rag, tgt)
    rows.append(dict(gene=g, family=famn, confidence=conf, confidence_class=cls,
                     oracle_r=orc, oracle_icw_r=icw, success=int(orc >= cfg["success_threshold_r"]),
                     rag_noRAG_similarity=sim_rn, gate_confidence=gate_conf,
                     mean_IC_RAG=mean_ic, retrieval_top1=top1, retrieval_top3=";".join(topg)))
    if len(rows) % 50 == 0:
        print(f"  {len(rows)} TFs ...")

cal = pd.DataFrame(rows)
cal.to_csv(f"{OUT}/heldout_known_confidence.tsv", sep="\t", index=False)

# calibration bins
bins = cfg["confidence_bins"]
brows = []
cal["bin"] = pd.cut(cal["confidence"], bins=bins, include_lowest=True)
for b, gdf in cal.groupby("bin", observed=True):
    brows.append(dict(confidence_bin=str(b), n=len(gdf),
                      success_fraction=float(gdf["success"].mean()),
                      median_oracle_r=float(gdf["oracle_r"].median())))
pd.DataFrame(brows).to_csv(f"{OUT}/confidence_calibration_bins.tsv", sep="\t", index=False)

print(f"\nCalibration on {len(cal)} held-out known TFs:")
print(pd.DataFrame(brows).to_string(index=False))
# context for SOHLH1 (~0.61): the 0.6-0.8 bin
mid = cal[(cal.confidence >= 0.6) & (cal.confidence < 0.8)]
print(f"\nConfidence 0.6-0.8 bin: n={len(mid)}, median oracle_r={mid['oracle_r'].median():.3f}, "
      f"success_frac={mid['success'].mean():.2f}")
bh = cal[cal.family == "bHLH"]
print(f"bHLH held-out TFs: n={len(bh)}, median oracle_r={bh['oracle_r'].median():.3f}, "
      f"median conf={bh['confidence'].median():.3f}")
print(f"saved -> {OUT}/heldout_known_confidence.tsv")
