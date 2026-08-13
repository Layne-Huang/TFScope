#!/usr/bin/env python
"""Compare two checkpoints under BOTH the legacy gate-oracle-r and the
coverage-aware gate_r_full, on a chosen split.

Motivation: v20's ckpt_best was selected on the LEGACY length-blind
gate-oracle-r (overlap only), which rewards a short/collapsed gate scored on
fewer, easier columns. This script rescores any .pt under the honest
coverage-aware metric (r x coverage == eval_full_metrics.panel_full r_cov) so
we can see whether the legacy selector actually picked a worse checkpoint than
a later one -- i.e. whether switching train.py to the coverage-aware selector
changes which epoch wins.

Usage:
  python scripts/compare_ckpt_covsel.py <ckpt_dir> <split.json> <data.parquet> \
      --split-name val --ckpts ckpt_best.pt ckpt_epoch175.pt
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from eval_full_metrics import (trimmed_core, aligned_cols, panel_full, uniform_floor)
from torch.utils.data import DataLoader

IC_THRESH, MAX_SHIFT, MIN_POS = 0.25, 10, 4
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _ic(p):
    p = np.clip(p, 1e-8, 1.0); return 2.0 + (p * np.log2(p)).sum(0)


def _trim_core_gate(pwm, thresh=IC_THRESH):
    ic = _ic(pwm); inf = np.where(ic >= thresh)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]


def build_model(ckpt_dir, ckpt_file):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(ckpt_dir, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(dev).eval()
    sd = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location=dev,
                    weights_only=False)["model"]
    m.load_state_dict(sd, strict=False)
    return m, cfg


def score(m, loader):
    """Return per-TF lists: legacy gate-oracle-r, coverage-aware gate_r_full,
    gate coverage, gate len_pred/len_gt, and the uniform floor r_full."""
    gate_old, gate_cov, cov, lp, lg, floors = [], [], [], [], [], []
    with torch.no_grad():
        for b in loader:
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
                active = gate > 0.5
                if not active.any(): active = gate > gate.max() * 0.5
                pcore = pred[:, active]
                if pcore.shape[1] == 0:
                    continue
                tg = _trim_core_gate(tgt[:, msk.astype(bool)])
                if tg.shape[1] == 0:
                    continue
                al, cols, r = aligned_cols(pcore, tg, MAX_SHIFT)
                gate_old.append(r)
                pf = panel_full(tg, al, cols, pred_ncols=pcore.shape[1])
                if pf:
                    gate_cov.append(pf["r_cov"] if "r_cov" in pf else pf["r_full"])
                    cov.append(pf["coverage"]); lp.append(pf["len_pred"]); lg.append(pf["len_gt"])
                floors.append(uniform_floor(core)["r_full"])
    return dict(n=len(gate_old),
                legacy_gate_oracle_r=float(np.mean(gate_old)) if gate_old else float("nan"),
                cov_aware_gate_r=float(np.nanmean(gate_cov)) if gate_cov else float("nan"),
                mean_coverage=float(np.nanmean(cov)) if cov else float("nan"),
                mean_len_pred=float(np.nanmean(lp)) if lp else float("nan"),
                mean_len_gt=float(np.nanmean(lg)) if lg else float("nan"),
                floor_r_full=float(np.nanmean(floors)) if floors else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir"); ap.add_argument("split"); ap.add_argument("data")
    ap.add_argument("--split-name", default="val", choices=["train", "val", "test"])
    ap.add_argument("--ckpts", nargs="+", default=["ckpt_best.pt"])
    a = ap.parse_args()

    ds = TFDataset(TFScopeConfig(), a.data, a.split, split=a.split_name, max_seq_len=1024)
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                    collate_fn=collate_variable_length)
    print(f"split={a.split_name}  n_rows={len(ds)}")
    rows = []
    for ck in a.ckpts:
        m, _ = build_model(a.ckpt_dir, ck)
        r = score(m, ld); r["ckpt"] = ck; rows.append(r)
        print(f"\n== {ck} ==")
        print(f"  legacy gate-oracle-r (overlap only) : {r['legacy_gate_oracle_r']:.4f}")
        print(f"  coverage-aware gate-r (r x cov)      : {r['cov_aware_gate_r']:.4f}")
        print(f"  mean coverage                        : {r['mean_coverage']:.3f}")
        print(f"  mean len pred / gt                   : {r['mean_len_pred']:.1f} / {r['mean_len_gt']:.1f}")
        print(f"  uniform floor r_full                 : {r['floor_r_full']:.4f}   (n={r['n']})")
    out = f"results/rescore_coverage/compare_{a.split_name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
