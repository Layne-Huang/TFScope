#!/usr/bin/env python
"""v18a WITH retrieval vs v18a with RAG DISABLED at inference (retrieved=None).

Same trained v18a checkpoint; full metric panel (trimmed core, offset+RC aligned)
for both modes, plus deployable canonical-fixed r. If the two are ~identical,
v18a's accuracy is de-novo, not retrieval-driven.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from eval_full_metrics import trimmed_core, aligned_cols, panel, canon_fixed_r
from torch.utils.data import DataLoader

CKPT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18a_attnrepair/ckpt_best.pt"
DATA = "data/processed/tf_pwm_deeppbs_only.parquet"
SPLIT = "data/processed/splits/deeppbs_only/benchmark_no_val.json"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def run(disable_rag):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except: pass
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    rows, rmed, cfix = [], [], []
    with torch.no_grad():
        for b in ld:
            b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long) for k, v in b.items()}
            rp = None if disable_rag else b.get("retrieved_pwms")
            rm = None if disable_rag else b.get("retrieved_masks")
            rsim = None if disable_rag else b.get("retrieved_sims")
            _, pw, _ = m(b["sequence_tokens"], b["dbd_mask"], b["family_id"],
                         retrieved_pwms=rp, retrieved_masks=rm, retrieved_sims=rsim,
                         recog_prior=b.get("recog_prior"))
            P = F.softmax(pw, 1).cpu().numpy(); T = b["target_pwm"].cpu().numpy(); M = b["pwm_mask"].cpu().numpy()
            for i in range(len(P)):
                tc = trimmed_core(T[i], M[i], 0.25)
                if tc is None: continue
                L = int(M[i].sum()); pv = P[i][:, :L]
                aligned, cols, _ = aligned_cols(pv, tc)
                d = panel(tc, aligned, cols)
                if d: rows.append(d); rmed.append(d["r"])
                cfix.append(canon_fixed_r(tc, pv))
    agg = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
    agg["med"] = np.nanmedian(rmed); agg["cfix"] = np.nanmean(cfix)
    return agg


print("v18a WITH rag ...", flush=True);  W = run(False)
print("v18a NO  rag ...", flush=True);    N = run(True)
keys = [("Mean Pearson r","r"),("Median Pearson r","med"),("IC-weighted r","icr"),
        ("MAE","mae"),("RMSE","rmse"),("Cross-entropy","ce"),("KL divergence","kl"),
        ("Top-1 accuracy","top1"),("AUC (macro OvR)","auc"),("F1 (macro)","f1"),("MCC","mcc"),
        ("Canonical-fixed r (deploy)","cfix")]
print(f"\n=== v18a: full panel (trimmed core, offset+RC aligned), 130 TFs ===")
print(f"{'Metric':<28}{'with RAG':>12}{'RAG off':>12}{'Δ(off-with)':>14}")
print("-" * 66)
for label, k in keys:
    print(f"{label:<28}{W[k]:>12.4f}{N[k]:>12.4f}{N[k]-W[k]:>+14.4f}")
