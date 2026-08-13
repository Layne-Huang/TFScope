#!/usr/bin/env python
"""One-shot collection of all GPU-dependent case-study arrays.

CS1: Egr1 (1a1g_A) — v18a prediction + DeepPBS prediction + truth (head-to-head logo)
CS2: KLF4 + MyoD — v17 (degenerate) + v18a (repaired) attention matrices
CS3: ZNF76 + ZNF649 — LFO C2H2_long model predictions + truth

Saves numpy arrays to results/case_study_arrays.npz so figure scripts run without GPU.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")

import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import TFDataset, collate_variable_length, AA_TO_TOKEN
from tfscope.models.alignment import align_pwm

CKPT_ROOT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

DATA       = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
DATA_DPBS  = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT_DPBS = "data/processed/splits/deeppbs_only/benchmark_no_val.json"

# ── helpers ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device=device):
    cfg = TFScopeConfig()
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    for k, v in json.load(open(config_path)).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"], strict=False)
    m.eval()
    return m, cfg

def run_seq(m, seq, fam_id, device=device):
    """Run a raw sequence through a model, no retrieval (de-novo). Returns (attn, pwm_prob)."""
    tok = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=device)
    dm  = torch.ones(1, len(seq), dtype=torch.bool, device=device)
    fi  = torch.tensor([fam_id], device=device)
    with torch.no_grad():
        _, pl, _ = m(tok, dm, fi, retrieved_pwms=None, retrieved_masks=None,
                     retrieved_sims=None, recog_prior=None)
    is_v18 = hasattr(m.pwm_head, "_last_attn")
    attn = m.pwm_head._last_attn[0].cpu().numpy() if is_v18 else None
    pwm  = F.softmax(pl, 1)[0].cpu().numpy()
    return attn, pwm

def infer_dataset(m, data_parquet, split_json, split="test"):
    """Run full dataset inference. Returns (preds dict, targets dict, fns list)."""
    cfg_dummy = TFScopeConfig()
    # load config from model's module attribute if available
    ds  = TFDataset(cfg_dummy, data_parquet, split_json, split=split, max_seq_len=1024)
    ld  = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                     collate_fn=collate_variable_length)
    P, T, M, fns = [], [], [], list(ds.filenames)
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(device, dtype=(torch.float32 if v.is_floating_point() else torch.long))
                 for k, v in b.items()}
            _, pw, _ = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                         retrieved_pwms=b.get("retrieved_pwms"),
                         retrieved_masks=b.get("retrieved_masks"),
                         retrieved_sims=b.get("retrieved_sims"),
                         recog_prior=b.get("recog_prior"))
            P.append(F.softmax(pw, 1).cpu().numpy())
            T.append(b["target_pwm"].cpu().numpy())
            M.append(b["pwm_mask"].cpu().numpy())
    P, T, M = np.concatenate(P), np.concatenate(T), np.concatenate(M)
    preds   = {fn: P[i] for i, fn in enumerate(fns)}
    targets = {fn: (T[i], int(M[i].sum())) for i, fn in enumerate(fns)}
    return preds, targets, fns

# ── CS1: Egr1 head-to-head ───────────────────────────────────────────────────

print("\n=== CS1: Egr1 (1a1g_A) head-to-head ===")
CKPT_V18A = f"{CKPT_ROOT}/deeppbs_v18a_attnrepair/ckpt_best.pt"
m_v18a, cfg_v18a = load_model(CKPT_V18A)

# We need to run on the deeppbs_only split to get v18a predictions aligned with DeepPBS
# Load the deeppbs_only dataset; find Egr1
ds_dpbs = TFDataset(cfg_v18a, DATA_DPBS, SPLIT_DPBS, split="test", max_seq_len=1024)
fns_dpbs = list(ds_dpbs.filenames)
egr1_fn = next((f for f in fns_dpbs if "1a1g" in f.lower()), None)
print(f"  Egr1 filename: {egr1_fn}")

# Run full test set (needed for CS1 + to get per-TF context)
print("  Running v18a inference on DeepPBS test split …")
preds_v18a, targets_v18a, _ = infer_dataset(m_v18a, DATA_DPBS, SPLIT_DPBS)
print(f"  done. {len(preds_v18a)} TFs")

# DeepPBS struct predictions
dpbs_npz = np.load("results/deeppbs_blind_benchmark/struct_preds.npz", allow_pickle=True)
MAX_L = 20

def get_dpbs_pred(fn):
    prefix = "_".join(fn.split("_")[:2]).replace(".txt", "")
    key = next((k for k in dpbs_npz.files if prefix in k and "::pred" in k), None)
    if key is None: return None, 0
    p = dpbs_npz[key]
    arr = np.full((4, MAX_L), 0.25, np.float32)
    L = min(p.shape[0], MAX_L); arr[:, :L] = np.clip(p[:L].T, 1e-8, 1)
    return arr / arr.sum(0, keepdims=True), L

# Extract CS1 arrays for Egr1
if egr1_fn:
    tgt_e, L_e = targets_v18a[egr1_fn]
    pred_e      = preds_v18a[egr1_fn]
    dpbs_e, _   = get_dpbs_pred(egr1_fn)

    tv = tgt_e[:, :L_e]
    aligned_v18a, _, _, _ = align_pwm(pred_e[:, :L_e], tv, max_shift=10, consider_revcomp=True)
    r_v18a = np.nanmean([pearsonr(tv[:, j], aligned_v18a[:, j])[0] for j in range(L_e)])

    if dpbs_e is not None:
        dL = min(dpbs_e.shape[1], MAX_L)
        aligned_dpbs, _, _, _ = align_pwm(dpbs_e[:, :dL], tv, max_shift=10, consider_revcomp=True)
        r_dpbs = np.nanmean([pearsonr(tv[:, j], aligned_dpbs[:, j])[0] for j in range(L_e)])
    else:
        aligned_dpbs, r_dpbs = np.full_like(tgt_e, 0.25), 0.0

    print(f"  Egr1: TFScope r={r_v18a:.3f}  DeepPBS r={r_dpbs:.3f}  L={L_e}")

    cs1_arrays = dict(
        egr1_truth=tgt_e, egr1_tfscope=aligned_v18a, egr1_deeppbs=aligned_dpbs,
        egr1_L=np.array([L_e]), egr1_r_tfscope=np.array([r_v18a]), egr1_r_dpbs=np.array([r_dpbs])
    )
else:
    print("  WARNING: Egr1 not found, skipping CS1 arrays")
    cs1_arrays = {}

# ── CS2: KLF4 + MyoD attention ───────────────────────────────────────────────

print("\n=== CS2: KLF4 / MyoD attention (v17 degenerate → v18a repaired) ===")
CKPT_V17 = f"{CKPT_ROOT}/deeppbs_v17_200ep/ckpt_best.pt"

CASES = {
    "KLF4": dict(fam=0, mutsite=18,
        wt="HTCDYAGCGKTYTKSSHLKAHLRTHTGEKPYHCDWDGCGWKFARSDELTRHYRKHTGHRPFQCQKCDRAFSRSDHLALHMKRH",
        mut_aa="Q"),
    "MyoD": dict(fam=3, mutsite=11,
        wt="RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA",
        mut_aa="R"),
}

cs2_arrays = {}
for ckpt_name, ckpt_path in [("v17", CKPT_V17), ("v18a", CKPT_V18A)]:
    if not os.path.exists(ckpt_path):
        print(f"  [skip] {ckpt_name}"); continue
    m, _ = load_model(ckpt_path)
    for tf, c in CASES.items():
        attn, pwm = run_seq(m, c["wt"], c["fam"])
        if attn is None:
            # legacy: try to hook cross_attn
            cap = {}
            m.pwm_head.cross_attn.register_forward_hook(
                lambda mo, i, o, cap=cap: cap.__setitem__("a", o[1].detach().cpu().numpy()))
            _, _ = run_seq(m, c["wt"], c["fam"])
            attn = cap.get("a", np.zeros((1, 1, 1)))[0]
        key = f"cs2_{ckpt_name}_{tf}_attn"
        cs2_arrays[key] = attn.astype(np.float32)
        mass = attn[:, c["mutsite"]].sum() if attn is not None else 0
        rc = np.nanmean(np.corrcoef(attn)[np.triu_indices(attn.shape[0], 1)]) if attn is not None else 0
        print(f"  {ckpt_name} {tf}: shape={attn.shape} mass@{c['mutsite']}={mass:.3f} rowconst={rc:.3f}")
    del m

# ── CS3: ZNF76 + ZNF649 from LFO C2H2_long model ────────────────────────────

print("\n=== CS3: ZNF76 / ZNF649 from LFO C2H2_long model ===")
CKPT_LFO = f"{CKPT_ROOT}/lofo_v18a/C2H2_long/ckpt_best.pt"
SPLIT_LFO = "data/processed/splits/lofo_v2/C2H2_long.json"

m_lfo, cfg_lfo = load_model(CKPT_LFO)
preds_lfo, targets_lfo, fns_lfo = infer_dataset(m_lfo, DATA, SPLIT_LFO)

cs3_arrays = {}
for label, fn in [("ZNF76", "ORIG__ZNF76.M09397_1.94d.txt"),
                  ("ZNF649", "ORIG__ZNF649.RCADE.txt")]:
    if fn not in preds_lfo:
        print(f"  WARNING: {fn} not in LFO test split"); continue
    tgt, L = targets_lfo[fn]; pred = preds_lfo[fn]
    tv = tgt[:, :L]
    aligned, _, _, _ = align_pwm(pred[:, :L], tv, max_shift=10, consider_revcomp=True)
    r = np.nanmean([pearsonr(tv[:, j], aligned[:, j])[0] for j in range(L)])
    print(f"  {label}: oracle r={r:.3f}  L={L}")
    cs3_arrays[f"cs3_{label}_truth"] = tgt
    cs3_arrays[f"cs3_{label}_pred"]  = aligned
    cs3_arrays[f"cs3_{label}_L"]     = np.array([L])
    cs3_arrays[f"cs3_{label}_r"]     = np.array([r])
del m_lfo

# ── Save all arrays ───────────────────────────────────────────────────────────

out = "results/case_study_arrays.npz"
all_arrays = {**cs1_arrays, **cs2_arrays, **cs3_arrays}
np.savez(out, **all_arrays)
print(f"\nSaved {len(all_arrays)} arrays to {out}")
print("Keys:", list(all_arrays.keys()))
