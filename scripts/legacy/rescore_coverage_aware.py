#!/usr/bin/env python
"""Re-score a checkpoint under BOTH the legacy and the coverage-aware metric.

The legacy gate-based oracle-r (train.py / eval_oracle_r_testset.py) takes
`active = gate > 0.5`, slices the prediction to those columns, and reports the
per-column Pearson over the ALIGNED OVERLAP ONLY. Uncovered ground-truth
columns are simply not scored, so a narrower gate is graded on fewer, easier
columns -- a perfect 3-of-14-column prediction scores r=1.000, and even a
RANDOM 3-column prediction beats a random full-length one (0.57 vs 0.27).

This script reports the legacy numbers unchanged next to the coverage-aware
ones (`panel_full`), so the size of the correction is explicit and the two are
directly comparable on identical predictions.

Usage:
  python scripts/rescore_coverage_aware.py <ckpt_dir> <split.json> <data.parquet> [name]
"""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from eval_full_metrics import (trimmed_core, aligned_cols, panel, panel_full,
                                uniform_floor)
from torch.utils.data import DataLoader

CKPT_DIR, SPLIT = sys.argv[1], sys.argv[2]
DATA = sys.argv[3]
NAME = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(CKPT_DIR.rstrip("/"))
IC_THRESH, MAX_SHIFT, MIN_POS = 0.25, 10, 4
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _ic(p):
    p = np.clip(p, 1e-8, 1.0); return 2.0 + (p * np.log2(p)).sum(0)


def _trim_core_gate(pwm, thresh=IC_THRESH):
    ic = _ic(pwm); inf = np.where(ic >= thresh)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]


cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(CKPT_DIR, "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: setattr(cfg, k, v)

m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ckpt_best.pt"),
                              map_location=dev, weights_only=False)["model"], strict=False)
ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                collate_fn=collate_variable_length)

gate_old, gate_new, panel_old, panel_new, floors = [], [], [], [], []
with torch.no_grad():
    for b in ld:
        b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long)
             for k, v in b.items()}
        gl, pl, _ = m(b['sequence_tokens'], b['dbd_mask'], b['family_id'],
                      retrieved_pwms=b.get('retrieved_pwms'),
                      retrieved_masks=b.get('retrieved_masks'),
                      retrieved_sims=b.get('retrieved_sims'),
                      recog_prior=b.get('recog_prior'))
        pwm_prob = F.softmax(pl, dim=1).cpu().numpy()
        gate_prob = torch.sigmoid(gl).cpu().numpy()
        target = b['target_pwm'].cpu().numpy(); mask = b['pwm_mask'].cpu().numpy()

        for pred, tgt, msk, gate in zip(pwm_prob, target, mask, gate_prob):
            core = trimmed_core(tgt, msk, IC_THRESH)
            if core is None or core.shape[1] < MIN_POS:
                continue
            # ---- gate-based (the metric with the blind spot) ----
            active = gate > 0.5
            if not active.any(): active = gate > gate.max() * 0.5
            pcore = pred[:, active]
            if pcore.shape[1] > 0:
                tg = _trim_core_gate(tgt[:, msk.astype(bool)])
                if tg.shape[1] > 0:
                    al, cols, r = aligned_cols(pcore, tg, MAX_SHIFT)
                    gate_old.append(r)
                    pf = panel_full(tg, al, cols, pred_ncols=pcore.shape[1])
                    if pf: gate_new.append(pf)
            # ---- panel (prediction restricted to the GT mask window) ----
            pm = pred[:, msk.astype(bool)]
            al, cols, _ = aligned_cols(pm, core, MAX_SHIFT)
            d = panel(core, al, cols)
            if d is not None:
                panel_old.append(d)
                pf = panel_full(core, al, cols, pred_ncols=pm.shape[1])
                if pf: panel_new.append(pf)
                floors.append(uniform_floor(core))

def M(rows, k): return float(np.nanmean([r[k] for r in rows])) if rows else float("nan")

res = {
    "name": NAME, "ckpt": CKPT_DIR, "n": len(panel_old),
    "LEGACY_gate_oracle_r": float(np.mean(gate_old)) if gate_old else float("nan"),
    "LEGACY_panel_r": M(panel_old, "r"),
    "LEGACY_top1": M(panel_old, "top1"),
    "gate_r_full": M(gate_new, "r_full"),
    "gate_coverage": M(gate_new, "coverage"),
    "gate_len_pred": M(gate_new, "len_pred"),
    "gate_len_gt": M(gate_new, "len_gt"),
    "gate_len_mae": M(gate_new, "len_mae"),
    "panel_r_full": M(panel_new, "r_full"),
    "panel_top1_full": M(panel_new, "top1_full"),
    "panel_coverage": M(panel_new, "coverage"),
    "FLOOR_r_full": M(floors, "r_full"),
    "FLOOR_top1_full": M(floors, "top1_full"),
}
print(json.dumps(res, indent=2))
out = f"results/rescore_coverage/{NAME}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=2)
print(f"saved {out}")
