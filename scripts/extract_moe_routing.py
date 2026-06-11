#!/usr/bin/env python
"""CS-MoE diagnostic: does the family-conditioned MoE specialize by TF family?

Runs the cluster40 TFScope checkpoint over its held-out test set, captures the
top-2 routed expert indices per TF from model.moe.aux_dict, and aggregates an
(expert x family) routing-frequency matrix. Tests for family-structured
specialization vs. diffuse routing.

Outputs:
  results/moe_routing/routing.npz        — raw per-TF top_indices + family ids
  results/moe_routing/expert_family.csv  — normalized expert x family matrix
  results/moe_routing/summary.json       — specialization statistics
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")

from torch.utils.data import DataLoader
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import TFDataset, collate_variable_length

CKPT_ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
CKPT  = f"{CKPT_ROOT}/fulldata_cluster40_v18a/ckpt_best.pt"
DATA  = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
SPLIT = "data/processed/splits/cluster40/split.json"
OUTDIR = "results/moe_routing"
os.makedirs(OUTDIR, exist_ok=True)

FAM = {0: "C2H2_short", 1: "C2H2_medium", 2: "C2H2_long", 3: "bHLH",
       4: "Homeodomain", 5: "bZIP", 6: "Nuclear_Receptor", 7: "Forkhead",
       8: "ETS", 9: "Other"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}  ckpt: {CKPT}")


def load_model(ckpt_path):
    cfg = TFScopeConfig()
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    if os.path.isfile(cfg_path):
        for k, v in json.load(open(cfg_path)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except: pass
    m = TFScopeModel(cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model"]
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, cfg


m, cfg = load_model(CKPT)
E, K = cfg.num_experts, cfg.top_k
print(f"num_experts={E}  top_k={K}")

ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                collate_fn=collate_variable_length)

all_top, all_fam = [], []
with torch.no_grad():
    for b in ld:
        b = {k: v.to(device, dtype=(torch.float32 if v.is_floating_point() else torch.long))
             for k, v in b.items()}
        m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
          retrieved_pwms=b.get("retrieved_pwms"), retrieved_masks=b.get("retrieved_masks"),
          retrieved_sims=b.get("retrieved_sims"), recog_prior=b.get("recog_prior"))
        aux = m.moe.aux_dict
        all_top.append(aux["top_indices"].cpu().numpy())     # (B, K)
        all_fam.append(aux["family_id"].cpu().numpy())        # (B,)

top = np.concatenate(all_top)      # (N, K)
fam = np.concatenate(all_fam)      # (N,)
N = len(fam)
print(f"collected routing for {N} TFs")

# ── expert x family routing-frequency matrix ─────────────────────────────────
fams_present = sorted(set(fam.tolist()))
M = np.zeros((E, len(fams_present)), dtype=float)   # expert x family counts (top-2 pooled)
for fi, f in enumerate(fams_present):
    rows = top[fam == f]                            # (nf, K)
    for e in range(E):
        M[e, fi] = (rows == e).sum()
# normalize each family column to a routing distribution over experts
Mn = M / np.clip(M.sum(0, keepdims=True), 1, None)

# ── specialization statistics ────────────────────────────────────────────────
def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

# per-family routing entropy (low = specialized to few experts)
fam_entropy = {FAM[f]: entropy(Mn[:, fi]) for fi, f in enumerate(fams_present)}
max_entropy = np.log2(E)

# dominant expert per family + its share
fam_dom = {}
for fi, f in enumerate(fams_present):
    e = int(np.argmax(Mn[:, fi]))
    fam_dom[FAM[f]] = {"top_expert": e, "share": float(Mn[e, fi])}

# global expert usage balance
global_use = M.sum(1) / M.sum()
use_entropy = entropy(global_use)

# how concentrated: mean per-family entropy vs uniform
mean_fam_entropy = float(np.mean(list(fam_entropy.values())))

# row-normalized: for each expert, which family dominates it (specialization)
Mr = M / np.clip(M.sum(1, keepdims=True), 1, None)   # expert x family, row-normalized
expert_dom = {}
for e in range(E):
    fi = int(np.argmax(Mr[e]))
    expert_dom[e] = {"top_family": FAM[fams_present[fi]], "share": float(Mr[e, fi]),
                     "usage": float(global_use[e])}

summary = {
    "n_tfs": N, "num_experts": E, "top_k": K,
    "max_entropy_bits": float(max_entropy),
    "mean_per_family_routing_entropy_bits": mean_fam_entropy,
    "global_expert_usage_entropy_bits": use_entropy,
    "per_family_entropy": fam_entropy,
    "per_family_dominant_expert": fam_dom,
    "per_expert_dominant_family": expert_dom,
    "interpretation": (
        "Lower per-family entropy than max (%.2f bits) indicates families route "
        "to a specialized subset of experts." % max_entropy),
}

# ── save ─────────────────────────────────────────────────────────────────────
np.savez(f"{OUTDIR}/routing.npz", top_indices=top, family_id=fam,
         expert_family_counts=M, fams_present=np.array(fams_present))
import csv
with open(f"{OUTDIR}/expert_family.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["expert"] + [FAM[f] for f in fams_present])
    for e in range(E):
        w.writerow([e] + [f"{Mn[e, fi]:.4f}" for fi in range(len(fams_present))])
json.dump(summary, open(f"{OUTDIR}/summary.json", "w"), indent=2)

# ── console report ───────────────────────────────────────────────────────────
print("\n=== MoE routing specialization ===")
print(f"max entropy (uniform over {E} experts): {max_entropy:.2f} bits")
print(f"mean per-family routing entropy:         {mean_fam_entropy:.2f} bits")
print(f"global expert-usage entropy:             {use_entropy:.2f} bits")
print("\nper-family dominant expert (share):")
for f, d in fam_dom.items():
    print(f"  {f:<18} expert {d['top_expert']:>2}  ({d['share']*100:4.1f}% of routes)")
print("\nper-expert dominant family (row-normalized share, usage):")
for e, d in expert_dom.items():
    print(f"  expert {e:>2}: {d['top_family']:<18} {d['share']*100:4.1f}%  usage={d['usage']*100:4.1f}%")
print(f"\nsaved -> {OUTDIR}/{{routing.npz, expert_family.csv, summary.json}}")
