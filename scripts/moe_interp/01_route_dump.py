#!/usr/bin/env python
"""Dump the v24 per-residue MoE routing decisions to a tidy per-token table.

The existing analyze_v24_ens_expert_residues.py only keeps aggregate statistics and
only the top-1 expert.  The biochemical-interpretability figure needs the raw
per-residue records so we can (i) join routing against per-residue structural
annotations, and (ii) look at the *pair* of experts top-2 routing selects.

Output: one row per DBD token of the (deduplicated) v23 table.

  prot      row index into the deduplicated protein table
  pos       0-based position inside that row's DBD crop
  aa        amino acid
  e1, e2    top-1 / top-2 expert index
  w1, w2    softmax gate weight over the top-k logits (what the MoE actually mixes with)
  p1        softmax-over-all-experts probability of the top-1 expert
  ent       routing entropy over all experts, normalised to [0, 1]
  e1_const  top-1 expert when family_id is forced to a constant (family ablation)

Usage:
  python scripts/moe_interp/01_route_dump.py --device cuda:6 [--seed seed42]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN, ESM2_PAD_TOKEN

DATA = "data/processed/tf_pwm_training_v23.parquet"
CONTACTS = "data/contact_maps/contact_targets_v23.json"
CKPTS = {
    "seed42": "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt",
    "seed1": "/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/seed1/ckpt_best.pt",
    "seed7": "/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/seed7/ckpt_best.pt",
    "seed13": "/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/seed13/ckpt_best.pt",
    "seed23": "/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/seed23/ckpt_best.pt",
}
OUTDIR = "results/moe_biochem"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_model(ckpt, device):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(ckpt), "config.json"))).items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception:
                setattr(cfg, k, v)
    cfg.use_retrieval = False
    assert getattr(cfg, "moe_granularity", "protein") == "residue", \
        f"{ckpt}: expected residue-granularity MoE, got {cfg.moe_granularity}"
    m = TFScopeModel(cfg).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model", sd), strict=False)
    return m, cfg


def load_proteins(max_len=1024):
    """Same protein table as analyze_v24_ens_expert_residues.load_proteins()."""
    df = pd.read_parquet(DATA).drop_duplicates("sequence").reset_index(drop=True)
    df["seq"] = df.sequence.astype(str).str.slice(0, max_len)
    df = df[df.seq.str.len() >= 8].reset_index(drop=True)
    return df


@torch.no_grad()
def route_all(model, cfg, df, device, const_fid, batch=8, tok_budget=4096):
    E, K = cfg.num_experts, cfg.top_k
    order = np.argsort([len(s) for s in df.seq.values])

    batches, cur = [], []
    for i in order:
        cand = cur + [i]
        Lmax = max(len(df.seq.values[j]) for j in cand)
        if cur and (len(cand) * Lmax > tok_budget or len(cand) > batch):
            batches.append(cur); cur = [i]
        else:
            cur = cand
    if cur:
        batches.append(cur)

    cols = {k: [] for k in ("prot", "pos", "aa", "e1", "e2", "w1", "w2", "p1", "ent", "e1_const")}
    t0, done = time.time(), 0
    for bnum, idx in enumerate(batches):
        seqs = [df.seq.values[i] for i in idx]
        L = max(len(s) for s in seqs)
        tok = torch.full((len(idx), L), ESM2_PAD_TOKEN, dtype=torch.long)
        dm = torch.zeros(len(idx), L, dtype=torch.bool)
        for j, s in enumerate(seqs):
            tok[j, :len(s)] = torch.tensor([AA_TO_TOKEN.get(a, 3) for a in s])
            dm[j, :len(s)] = True
        tok, dm = tok.to(device), dm.to(device)
        fid = torch.tensor([int(df.family_id.values[i]) for i in idx], device=device)

        _, _, aux = model(tok, dm, fid)
        gl = aux["gate_logits"].float()                     # (N_dbd, E), row-major over DBD tokens
        topi = aux["top_indices"]                           # (N_dbd, K)
        assert topi.shape[1] == K >= 2, f"expected top_k>=2, got {topi.shape}"
        w = F.softmax(torch.gather(gl, 1, topi), dim=-1)    # weights the MoE actually mixes with
        p = F.softmax(gl, dim=-1)
        ent = (-(p * (p + 1e-9).log()).sum(-1) / np.log(E))

        topi = topi.cpu().numpy(); w = w.cpu().numpy()
        p1 = p.gather(1, aux["top_indices"][:, :1]).squeeze(1).cpu().numpy()
        ent = ent.cpu().numpy()

        _, _, aux_c = model(tok, dm, torch.full_like(fid, const_fid))
        top_c = aux_c["top_indices"][:, 0].cpu().numpy()

        pos = 0
        for j, i in enumerate(idx):
            n = len(seqs[j]); sl = slice(pos, pos + n); pos += n
            cols["prot"].append(np.full(n, int(i), dtype=np.int32))
            cols["pos"].append(np.arange(n, dtype=np.int32))
            cols["aa"].append(np.array(list(seqs[j])))
            cols["e1"].append(topi[sl, 0].astype(np.int8))
            cols["e2"].append(topi[sl, 1].astype(np.int8))
            cols["w1"].append(w[sl, 0].astype(np.float32))
            cols["w2"].append(w[sl, 1].astype(np.float32))
            cols["p1"].append(p1[sl].astype(np.float32))
            cols["ent"].append(ent[sl].astype(np.float32))
            cols["e1_const"].append(top_c[sl].astype(np.int8))
        assert pos == len(topi), f"token bookkeeping mismatch {pos} vs {len(topi)}"

        done += len(idx)
        if bnum % 50 == 0 or done == len(order):
            log(f"    routed {done}/{len(order)} proteins "
                f"(batch {bnum + 1}/{len(batches)}, {time.time() - t0:.0f}s)")
    return pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", default="seed42", choices=sorted(CKPTS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = load_proteins()
    const_fid = int(df[df.family_name == "Other"].family_id.mode().iloc[0]) \
        if (df.family_name == "Other").any() else int(df.family_id.mode().iloc[0])
    log(f"proteins={len(df)}  family-ablation constant family_id={const_fid}")

    ptab = df[["filename", "gene_symbol", "family_id", "family_name", "seq"]].copy()
    ptab["prot"] = np.arange(len(df), dtype=np.int32)
    ptab["seq_len"] = ptab.seq.str.len().astype(np.int32)
    ppath = os.path.join(a.outdir, "protein_table.parquet")
    ptab.to_parquet(ppath, index=False)
    log(f"saved {ppath}  ({len(ptab)} proteins)")

    model, cfg = build_model(CKPTS[a.seed], a.device)
    log(f"{a.seed}: num_experts={cfg.num_experts} top_k={cfg.top_k}")
    tok = route_all(model, cfg, df, a.device, const_fid, a.batch)

    # co-crystal base-contact supervision set, same clipping rule as the v24 analysis
    ct = json.load(open(CONTACTS))
    fn2prot = dict(zip(ptab.filename.astype(str), ptab.prot))
    ct_pairs = set()
    have = set()
    for fn, e in ct.items():
        pr = fn2prot.get(str(fn))
        if pr is None:
            continue
        have.add(int(pr))
        n = int(ptab.seq_len.values[pr])
        for _col, rows in e["cols"].items():
            for ridx, w in rows:
                if w > 0 and int(ridx) < n:
                    ct_pairs.add((int(pr), int(ridx)))
    key = list(zip(tok["prot"].tolist(), tok["pos"].tolist()))
    tok["has_contact_label"] = np.fromiter((k[0] in have for k in key), bool, len(key))
    tok["base_contact_v23"] = np.fromiter((k in ct_pairs for k in key), bool, len(key))
    log(f"co-crystal contact labels on {len(have)} proteins, "
        f"{int(tok.base_contact_v23.sum())} contact residues")

    out = os.path.join(a.outdir, f"routing_{a.seed}.parquet")
    tok.to_parquet(out, index=False)
    log(f"saved {out}  ({len(tok)} tokens)")


if __name__ == "__main__":
    main()
