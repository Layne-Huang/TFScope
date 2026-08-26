#!/usr/bin/env python
"""Phase-4 GPU smoke test: build the REAL v26 model (actual ESM-2 + LoRA) and run a
forward/backward on REAL v26 data with the frozen split. Verifies the pieces the CPU stub
tests cannot: ESM loads, LoRA params exist and receive gradient, memory fits, chains pack
correctly from the manifest.

  python scripts/v26/smoke_test_gpu.py --config configs/v26/core.yaml --steps 3
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src")
from tfscope.v26.config import V26Config
from tfscope.v26.model import TFScopeV26

AA = {"L":4,"A":5,"G":6,"V":7,"S":8,"E":9,"R":10,"T":11,"I":12,"D":13,"P":14,"K":15,
      "Q":16,"N":17,"F":18,"Y":19,"M":20,"H":21,"W":22,"C":23}
PAD = 1

def load_cfg(p):
    d = {}
    for line in open(p):
        line = line.split("#")[0].strip()
        if not line or ":" not in line: continue
        k, v = line.split(":", 1)
        try: d[k.strip()] = json.loads(v.strip())
        except Exception: d[k.strip()] = v.strip()
    cfg = V26Config()
    for k, v in d.items():
        if hasattr(cfg, k): setattr(cfg, k, v)
    return cfg, d

def pack(rows, max_partners):
    """Build per-CHAIN tensors: one row per chain across the batch."""
    toks, dms, cidx, prim = [], [], [], []
    for b, r in enumerate(rows):
        seq = str(r.sequence)
        toks.append([AA.get(c, 4) for c in seq])
        dm = [0]*len(seq)
        for i in range(int(r.dbd_start), min(int(r.dbd_end), len(seq))): dm[i] = 1
        dms.append(dm); cidx.append(b); prim.append(True)
        for p in json.loads(r.partner_entities or "[]")[:max_partners]:
            ps = str(p["sequence"])
            toks.append([AA.get(c, 4) for c in ps]); dms.append([1]*len(ps))
            cidx.append(b); prim.append(False)
    L = max(len(t) for t in toks)
    T = torch.full((len(toks), L), PAD, dtype=torch.long)
    D = torch.zeros((len(toks), L), dtype=torch.bool)
    for i,(t,d) in enumerate(zip(toks,dms)):
        T[i,:len(t)] = torch.tensor(t); D[i,:len(d)] = torch.tensor(d, dtype=torch.bool)
    return T, D, torch.tensor(cidx), torch.tensor(prim)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v26/core.yaml")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    cfg, raw = load_cfg(a.config)
    ds = raw.get("dataset", "core")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config={a.config} dataset={ds} device={dev}", flush=True)

    ex = pd.read_parquet(f"data/processed/v26/v26_{ds}.parquet")
    # join on target_unit_id: example_id encodes the crop bounds and so differs per dataset
    man = pd.read_parquet("data/processed/splits/v26/manifest.parquet")
    man = man[["target_unit_id", "split", "application_holdout"]].drop_duplicates("target_unit_id")
    ex = ex.merge(man, on="target_unit_id", how="inner")
    ex = ex[~ex.application_holdout]
    tr = ex[ex.split == "train"].reset_index(drop=True)
    print(f"train examples for this dataset: {len(tr)} (of {len(ex)} with a split label)", flush=True)
    assert len(tr) > 0, "no training examples -- split/dataset mismatch"

    model = TFScopeV26(cfg).to(dev); model.build(dev)
    pc = model.param_counts(); print("param counts:", pc, flush=True)
    lora = [n for n,p in model.named_parameters() if "lora_" in n and p.requires_grad]
    print(f"trainable LoRA tensors: {len(lora)}", flush=True)
    assert len(lora) > 0 or cfg.lora_rank == 0, "LoRA requested but no trainable LoRA params"

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    t0 = time.time()
    for step in range(a.steps):
        rows = [tr.iloc[i] for i in np.random.default_rng(step).integers(0, len(tr), a.batch)]
        T, D, C, P = [x.to(dev) for x in pack(rows, cfg.max_partners)]
        pwm, gate, aux = model(T, D, C, P)
        # crude target just to exercise the graph: uniform PWM
        tgt = torch.full_like(pwm, 0.25)
        loss = ((pwm - tgt) ** 2).mean() + cfg.balance_loss_weight * aux["balance_loss"]
        opt.zero_grad(); loss.backward(); opt.step()
        g = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None
                and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        lg = sum(1 for n,p in model.named_parameters() if "lora_" in n and p.grad is not None
                 and p.grad.abs().sum() > 0)
        print(f"  step {step}: chains={T.shape[0]} L={T.shape[1]} loss={loss.item():.5f} "
              f"grads={g} lora_grads={lg} alpha={float(aux['alpha_flank'].max()):.3f} "
              f"beta={float(aux['beta_partner'].max()):.3f} "
              f"lambda={float(aux['lambda_contact']):.3f} "
              f"exp_len={float(aux['expected_length'].mean()):.1f}", flush=True)
        assert torch.isfinite(loss), "non-finite loss"
    if dev.type == "cuda":
        print(f"peak GPU mem: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    print(f"SMOKE OK in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
