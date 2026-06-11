#!/usr/bin/env python
"""Held-out KNOWN-TF calibration for the deeppbs v18a RAG checkpoint.

Runs deeppbs_v18a_attnrepair over the deeppbs TEST partition (held out from training),
recording the calibrated confidence and the oracle_r vs the curated motif per gene, so
the orphan (ADNP2/ZHX2/ZHX3) confidences map to real motif-recovery accuracy on THIS
checkpoint/split. Mirrors build_calibration.py but for the deeppbs setup.
"""
import os, sys, json
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd, torch, yaml
from cs_utils import (load_model, column_ic, active_cols, aligned_r, tokens_from_seq, infer, confidence_score)

cfg = yaml.safe_load(open("configs/case_study_orphans_deeppbs.yaml"))
OUT = f"{cfg['output_dir']}/confidence"; os.makedirs(OUT, exist_ok=True)
ML = cfg["max_motif_length"]; K = cfg["retrieval_top_k"]; GTH = cfg["active_gate_threshold"]

df = pd.read_parquet(cfg["donor_parquet"]); df["g"] = df["gene_symbol"].astype(str).str.upper()
fn2gene = dict(zip(df["filename"], df["g"]))
fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
fn2seq  = dict(zip(df["filename"], df["sequence"]))
fn2fam  = dict(zip(df["filename"], df["family_id"]))
fn2famn = dict(zip(df["filename"], df["family_name"]))
sp = json.load(open(cfg["deeppbs_split"]))
train_val = set(sp["train"]) | set(sp.get("val", [])); test_fns = sp["test"]
embs = np.load(cfg["embeddings"])
donors = [fn for fn in embs.files if fn in train_val]
Dmat = np.stack([embs[fn] / (np.linalg.norm(embs[fn]) + 1e-8) for fn in donors])
donor_genes = np.array([fn2gene.get(fn, "?") for fn in donors])
model, _ = load_model(cfg["checkpoint_production"])

def retrieval_for(query_fn):
    q = embs[query_fn]; q = q / (np.linalg.norm(q) + 1e-8); sims = Dmat @ q
    order = np.argsort(-sims); qg = fn2gene.get(query_fn, "?")
    top, seen = [], set()
    for di in order:
        g = donor_genes[di]
        if g == qg or g in seen: continue
        seen.add(g); top.append(di)
        if len(top) == K: break
    rp = torch.full((1, K, 4, ML), 0.25); rm = torch.zeros((1, K, ML)); rs = torch.zeros((1, K))
    for ki, di in enumerate(top):
        pwm = fn2pwm[donors[di]]; L = min(pwm.shape[1], ML)
        rp[0, ki, :, :L] = torch.from_numpy(pwm[:, :L].copy()); rm[0, ki, :L] = 1.0; rs[0, ki] = float(sims[di])
    return rp, rm, rs

rows = []; seen = set()
for fn in test_fns:
    g = fn2gene.get(fn, "?")
    if g in seen or fn not in embs.files: continue
    seen.add(g)
    seq = fn2seq.get(fn)
    if not isinstance(seq, str) or len(seq) < 4: continue
    tok, mask = tokens_from_seq(seq); fam_id = int(fn2fam.get(fn, 9)); famn = fn2famn.get(fn, "Other")
    ret = retrieval_for(fn)
    g_rag, p_rag, _ = infer(model, tok, mask, fam_id, ret=ret)
    m_rag = active_cols(g_rag, GTH); core_rag = p_rag[:, m_rag]
    gate_conf = float(g_rag[m_rag].mean()); mean_ic = float(column_ic(core_rag).mean())
    conf, cls, _ = confidence_score(cfg, mean_ic, gate_conf)
    tgt = fn2pwm[fn]; Lt = int((tgt.sum(0) > 1e-6).sum()); tgt = tgt[:, :Lt] if Lt >= 2 else tgt
    orc, icw, _ = aligned_r(core_rag, tgt)
    rows.append(dict(gene=g, family=famn, confidence=conf, confidence_class=cls,
                     oracle_r=orc, success=int(orc >= cfg["success_threshold_r"]),
                     gate_confidence=gate_conf, mean_IC_RAG=mean_ic))
    if len(rows) % 50 == 0: print(f"  {len(rows)} TFs ...")

cal = pd.DataFrame(rows); cal.to_csv(f"{OUT}/heldout_known_confidence.tsv", sep="\t", index=False)
bins = cfg["confidence_bins"]; cal["bin"] = pd.cut(cal["confidence"], bins=bins, include_lowest=True)
brows = [dict(confidence_bin=str(b), n=len(gd), success_fraction=float(gd.success.mean()),
              median_oracle_r=float(gd.oracle_r.median()))
         for b, gd in cal.groupby("bin", observed=True)]
pd.DataFrame(brows).to_csv(f"{OUT}/confidence_calibration_bins.tsv", sep="\t", index=False)
from scipy.stats import spearmanr
hd = cal[cal.family == "Homeodomain"]
print(f"\ndeeppbs calibration: n={len(cal)} | Spearman rho={spearmanr(cal.confidence, cal.oracle_r).correlation:.2f}")
print(f"Homeodomain held-out: n={len(hd)}, median oracle_r={hd.oracle_r.median():.3f}, median conf={hd.confidence.median():.3f}")
print(pd.DataFrame(brows).to_string(index=False))
