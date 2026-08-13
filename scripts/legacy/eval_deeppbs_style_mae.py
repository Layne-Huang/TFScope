#!/usr/bin/env python
"""Compute MAE the DeepPBS way for a TFScope checkpoint.

DeepPBS convention (scripts/eval_deeppbs.py:compute_metrics):
  - pred, true are (L, 4) over the valid motif window (pwm_mask), NO IC trim
  - MAE = mean_positions( sum_{4 bases} |pred - true| )   <-- sum over bases (≈4x our panel mean)
  - per-position Pearson r over the 4 bases, then averaged

We report it in two frames:
  - FIXED   : pred = canonicalized, left-anchored to the pwm_mask window, no alignment
              (this is the DeepPBS-comparable regime — DeepPBS predicts in the experimental frame)
  - ALIGNED : pred aligned to target over +/-10 + RC first (best-case, like panel)

Usage: python scripts/eval_deeppbs_style_mae.py <ckpt_dir> <split.json> [data.parquet] [ckpt_name]
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, entropy
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm, revcomp_pwm_np
from eval_canonical_registration import canonicalize
from torch.utils.data import DataLoader

args = [a for a in sys.argv[1:] if not a.startswith("--")]
CKPT_DIR = args[0]; SPLIT = args[1]
DATA = args[2] if len(args) > 2 else "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
CKPT_NAME = args[3] if len(args) > 3 else "ckpt_best.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"
BKG = np.array([0.25, 0.25, 0.25, 0.25])


def deeppbs_metrics(pred_L4, true_L4):
    """pred_L4, true_L4: (L,4) probability rows. DeepPBS formulas exactly."""
    mae = float(np.mean(np.sum(np.abs(pred_L4 - true_L4), axis=1)))
    rs = [pearsonr(true_L4[i], pred_L4[i])[0] for i in range(len(true_L4))]
    r = float(np.nanmean(rs))
    ic_t = entropy(true_L4, BKG, base=2, axis=1)
    ic_p = entropy(pred_L4, BKG, base=2, axis=1)
    with np.errstate(invalid="ignore"):
        ic_r = float(pearsonr(ic_t, ic_p)[0]) if len(ic_t) > 1 else float("nan")
    return mae, r, ic_r


cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(CKPT_DIR, "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: setattr(cfg, k, v)

m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(os.path.join(CKPT_DIR, CKPT_NAME), map_location=dev, weights_only=False)["model"], strict=False)
ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_variable_length)

fix, ali = [], []
with torch.no_grad():
    for b in ld:
        b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long) for k, v in b.items()}
        _, pwm_logits, _ = m(b['sequence_tokens'], b['dbd_mask'], b['family_id'],
                             retrieved_pwms=b.get('retrieved_pwms'), retrieved_masks=b.get('retrieved_masks'),
                             retrieved_sims=b.get('retrieved_sims'), recog_prior=b.get('recog_prior'))
        pwm_prob = F.softmax(pwm_logits, dim=1).cpu().numpy()
        target = b['target_pwm'].cpu().numpy(); mask = b['pwm_mask'].cpu().numpy()
        for pred, tgt, msk in zip(pwm_prob, target, mask):
            idx = msk.astype(bool)
            if idx.sum() < 4: continue
            tw = np.clip(tgt[:, idx], 1e-8, 1.0); tw = tw / tw.sum(0, keepdims=True)  # IC-core span rule
            icv = 2.0 + (tw * np.log2(tw)).sum(0); inf = np.where(icv >= 0.25)[0]
            if len(inf) == 0 or (inf[-1] - inf[0] + 1) < 4: continue   # rule: core span >= 4
            true = np.clip(tgt[:, idx], 1e-8, 1.0); true = true / true.sum(0, keepdims=True)   # (4,L)
            pm = np.clip(pred[:, idx], 1e-8, 1.0); pm = pm / pm.sum(0, keepdims=True)           # (4,L)
            # --- FIXED frame (canonicalize + left-anchor, no alignment) ---
            cp = canonicalize(pm.astype(np.float32)); cp = cp / cp.sum(0, keepdims=True).clip(1e-8)
            Lt = true.shape[1]; pp = np.full((4, Lt), 0.25, np.float32)
            L = min(cp.shape[1], Lt); pp[:, :L] = cp[:, :L]
            fix.append(deeppbs_metrics(pp.T, true.T))
            # --- ALIGNED frame (+/-10 + RC) ---
            al, shift, orient, _ = align_pwm(pm, true, max_shift=10, consider_revcomp=True)
            al = np.clip(al, 1e-8, 1.0); al = al / al.sum(0, keepdims=True)
            ali.append(deeppbs_metrics(al.T, true.T))

def agg(rows, name):
    a = np.array(rows)
    return {f"{name}_mae_mean": float(a[:,0].mean()), f"{name}_mae_median": float(np.median(a[:,0])),
            f"{name}_r_mean": float(np.nanmean(a[:,1])), f"{name}_r_median": float(np.nanmedian(a[:,1])),
            f"{name}_ic_r_mean": float(np.nanmean(a[:,2]))}

res = {"ckpt": os.path.join(CKPT_DIR, CKPT_NAME), "n": len(fix), **agg(fix, "fixed"), **agg(ali, "aligned")}
print(json.dumps(res, indent=2))
