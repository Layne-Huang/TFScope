#!/usr/bin/env python
"""Full metric panel for ONE TFScope checkpoint on a test split, reporting:
  - gate-based oracle-r  (pred = gate>0.5 cols; training run_oracle_r_eval protocol)
  - panel/manuscript metrics (pred = target pwm_mask window, aligned +/-10+RC to
    IC>=0.25 target core): r, median r, IC-weighted r, MAE, RMSE, CE, KL,
    top-1 acc, AUC, F1, MCC  -- identical to eval_full_metrics.panel
  - canon_fixed_r (deployable, no alignment freedom)

Usage:
  python scripts/eval_oracle_r_testset.py <ckpt_dir> <split.json> [data.parquet] [ckpt_name] [--json out.json]
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from eval_full_metrics import trimmed_core, aligned_cols, panel, canon_fixed_r
from torch.utils.data import DataLoader

args = [a for a in sys.argv[1:] if not a.startswith("--")]
CKPT_DIR  = args[0]
SPLIT     = args[1]
DATA      = args[2] if len(args) > 2 else "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
CKPT_NAME = args[3] if len(args) > 3 else "ckpt_best.pt"
JSON_OUT  = None
if "--json" in sys.argv:
    JSON_OUT = sys.argv[sys.argv.index("--json") + 1]
CKPT = os.path.join(CKPT_DIR, CKPT_NAME)
dev = "cuda" if torch.cuda.is_available() else "cpu"
IC_THRESH, MAX_SHIFT = 0.25, 10
MIN_POS = 4   # rule: motifs with fewer than 4 informative positions are excluded


def _ic(pwm):
    p = np.clip(pwm, 1e-8, 1.0)
    return 2.0 + (p * np.log2(p)).sum(0)

def _trim_core_gate(pwm, thresh=IC_THRESH):
    ic = _ic(pwm)
    inf = np.where(ic >= thresh)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]


cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(CKPT_DIR, "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except: setattr(cfg, k, v)

m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)
ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_variable_length)

gate_rs = []          # gate-based oracle-r
rows, cfix = [], []   # panel dicts, canon-fixed r
with torch.no_grad():
    for b in ld:
        b = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long)
             for k, v in b.items()}
        gate_logits, pwm_logits, _ = m(
            b['sequence_tokens'], b['dbd_mask'], b['family_id'],
            retrieved_pwms=b.get('retrieved_pwms'), retrieved_masks=b.get('retrieved_masks'),
            retrieved_sims=b.get('retrieved_sims'), recog_prior=b.get('recog_prior'))
        pwm_prob  = F.softmax(pwm_logits, dim=1).cpu().numpy()
        gate_prob = torch.sigmoid(gate_logits).cpu().numpy()
        target    = b['target_pwm'].cpu().numpy()
        mask      = b['pwm_mask'].cpu().numpy()
        for pred, tgt, msk, gate in zip(pwm_prob, target, mask, gate_prob):
            core = trimmed_core(tgt, msk, IC_THRESH)   # renormalized IC core (manuscript)
            if core is None or core.shape[1] < MIN_POS:   # rule: >= 4 informative positions
                continue
            # --- gate-based oracle-r ---
            active = gate > 0.5
            if not active.any():
                active = gate > gate.max() * 0.5
            pcore = pred[:, active]
            if pcore.shape[1] > 0:
                tg = _trim_core_gate(tgt[:, msk.astype(bool)])
                if tg.shape[1] > 0:
                    _, _, _, gr = align_pwm(pcore, tg, max_shift=MAX_SHIFT, consider_revcomp=True)
                    gate_rs.append(gr)
            # --- panel (pred restricted to pwm_mask window) ---
            pred_mask = pred[:, msk.astype(bool)]
            aligned, cols, _ = aligned_cols(pred_mask, core, MAX_SHIFT)
            d = panel(core, aligned, cols)
            if d is not None:
                rows.append(d)
                cfix.append(canon_fixed_r(core, pred_mask))

def mean(key): return float(np.nanmean([r[key] for r in rows]))
def med(key):  return float(np.nanmedian([r[key] for r in rows]))

res = {
    "ckpt": CKPT, "use_retrieval": bool(cfg.use_retrieval), "n_panel": len(rows),
    "gate_oracle_r_mean": float(np.mean(gate_rs)), "gate_oracle_r_median": float(np.median(gate_rs)),
    "panel_r_mean": mean("r"), "panel_r_median": med("r"),
    "icw_r": mean("icr"), "mae": mean("mae"), "rmse": mean("rmse"),
    "ce": mean("ce"), "kl": mean("kl"), "top1": mean("top1"),
    "auc": mean("auc"), "f1": mean("f1"), "mcc": mean("mcc"),
    "canon_fixed_r": float(np.nanmean(cfix)),
}
print(json.dumps(res))
if JSON_OUT:
    json.dump(res, open(JSON_OUT, "w"), indent=2)
