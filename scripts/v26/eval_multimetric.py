#!/usr/bin/env python
"""Unified multi-metric evaluator for v24 and v26 checkpoints.

Backfills the metrics that v26 training never computed. `validate()` in train_v26.py only ever
returned content_r and cov_r, so v26 has no MAE / RMSE / JSD / top-base / AUROC / length-error --
which matters because v26's loss CONTAINS an L1 term while v24's does not, so v26 may well be
better on calibration metrics and worse on shape metrics. Reporting only cov_r hides that.

Metrics (all after the SAME oracle registration both models get: best offset + orientation via
align_pwm, so neither is penalised for placement):

  shape / ranking   pearson_r, cosine, topbase_acc, macro_f1, auroc
  calibration       mae, rmse, jsd_bits, ic_mae
  length            gate_len_mae, gate_len_bias, coverage

Statistical unit is the TARGET UNIT, not the row: a TF with 30 motif records counts once.

Handles both model families:
  v24  TFScopeModel(tokens, dbd_mask, family_id) -> (gate_LOGITS, pwm_LOGITS, aux)
  v26  TFScopeV26(tokens, dbd_mask, chain_index, is_primary) -> (pwm_PROBS, gate_PROBS, aux)

  python scripts/v26/eval_multimetric.py --all
  python scripts/v26/eval_multimetric.py --ckpt <path> --kind v26 --dataset core --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from tfscope.models.alignment import align_pwm                      # noqa: E402

RESD = "results/v26"
CK = "/data1/leihuang/TFScope_store/checkpoints/v26"
V24_CKPT = ("/data1/leihuang/project/TFScope/checkpoints/v24_contact/"
            "contact_v24_seed42/ckpt_best.pt")
AA = {"L": 4, "A": 5, "G": 6, "V": 7, "S": 8, "E": 9, "R": 10, "T": 11, "I": 12, "D": 13,
      "P": 14, "K": 15, "Q": 16, "N": 17, "F": 18, "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23}
PAD = 1
FID = {"C2H2_short": 0, "C2H2_medium": 1, "C2H2_long": 2, "bHLH": 3, "Homeodomain": 4,
       "bZIP": 5, "Nuclear_Receptor": 6, "Forkhead": 7, "ETS": 8, "Other": 9}


def map_family(s):
    for f in str(s).split(";"):
        if f in FID:
            return FID[f]
        if f.startswith("C2H2"):
            return FID["C2H2_long"]
    return FID["Other"]


# ----------------------------------------------------------------------- metrics
def _jsd_bits(p, q):
    """Mean per-column Jensen-Shannon divergence in bits."""
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * np.log2(a / b)).sum(0)
    return float((0.5 * kl(p, m) + 0.5 * kl(q, m)).mean())


def _ic(p):
    p = np.clip(p, 1e-12, 1)
    return 2.0 + (p * np.log2(p)).sum(0)


def pair_metrics(pred, gt):
    """pred, gt: (4, L) column-normalised, already registered to each other."""
    a, b = pred.flatten(), gt.flatten()
    out = {}
    out["pearson_r"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    out["cosine"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    out["mae"] = float(np.abs(pred - gt).mean())
    out["rmse"] = float(np.sqrt(((pred - gt) ** 2).mean()))
    out["jsd_bits"] = _jsd_bits(pred, gt)
    out["ic_mae"] = float(np.abs(_ic(pred) - _ic(gt)).mean())
    tp, tg = pred.argmax(0), gt.argmax(0)
    out["topbase_acc"] = float((tp == tg).mean())
    f1s = []
    for c in range(4):
        tpos = int(((tp == c) & (tg == c)).sum())
        fp = int(((tp == c) & (tg != c)).sum()); fn = int(((tp != c) & (tg == c)).sum())
        if tpos + fp + fn == 0:
            continue
        prec = tpos / max(tpos + fp, 1); rec = tpos / max(tpos + fn, 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    # AUROC: label = target base prob > 0.25, score = predicted prob (pooled base x column)
    lab = (gt.flatten() > 0.25).astype(int); sc = pred.flatten()
    npos, nneg = int(lab.sum()), int((1 - lab).sum())
    if npos and nneg:
        order = np.argsort(sc)
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(sc) + 1)
        out["auroc"] = float((ranks[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    else:
        out["auroc"] = float("nan")
    return out


# ------------------------------------------------------------------------ models
def load_v24(device, ckpt=None):
    from tfscope.config import TFScopeConfig
    from tfscope.models.tfscope import TFScopeModel
    ckpt = ckpt or V24_CKPT
    cfg = TFScopeConfig()
    _cfgp = os.path.join(os.path.dirname(ckpt), "config.json")
    if not os.path.exists(_cfgp):
        _cfgp = os.path.join(os.path.dirname(V24_CKPT), "config.json")
    for k, v in json.load(open(_cfgp)).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model", sd), strict=False)
    return m.eval(), cfg.max_motif_length


def load_v26(path, device):
    from tfscope.v26.config import V26Config
    from tfscope.v26.model import TFScopeV26
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = V26Config()
    for k, v in (sd.get("config") or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    m = TFScopeV26(cfg).to(device); m.build(device)
    m.load_state_dict(sd["model"], strict=False)
    return m.eval(), cfg.max_motif_length


# -------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, kind, maxlen, ex, device, batch=4):
    per_unit = defaultdict(list)
    for s in range(0, len(ex), batch):
        sub = ex.iloc[s:s + batch]
        seqs = [str(r.sequence) for r in sub.itertuples()]
        L = max(len(x) for x in seqs)
        T = torch.full((len(seqs), L), PAD, dtype=torch.long)
        D = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, q in enumerate(seqs):
            T[i, :len(q)] = torch.tensor([AA.get(c, 4) for c in q])
            D[i, :len(q)] = True
        T, D = T.to(device), D.to(device)
        if kind == "v24":
            fid = torch.tensor([map_family(r.dbd_families_for_analysis_only)
                                for r in sub.itertuples()]).to(device)
            gl, pl, _ = model(T, D, fid)
            pwm = pl.softmax(1).float().cpu().numpy()
            gate = gl.sigmoid().float().cpu().numpy()
        else:
            ci = torch.arange(len(seqs)).to(device)
            pr = torch.ones(len(seqs), dtype=torch.bool).to(device)
            p, g, _ = model(T, D, ci, pr)
            pwm = p.float().cpu().numpy(); gate = g.float().cpu().numpy()
        for j, r in enumerate(sub.itertuples()):
            arr = np.frombuffer(r.pwm, dtype=np.float32).reshape(4, -1).astype(np.float64)
            Lg = min(int(r.motif_length), maxlen, arr.shape[1])
            gt = arr[:, :Lg]
            span = max(1, int((gate[j] > 0.5).sum()))
            pred = pwm[j][:, :span].astype(np.float64)
            try:
                al, _, _, _ = align_pwm(pred, gt, max_shift=10, consider_revcomp=True,
                                        min_overlap=3)
            except Exception:
                continue
            if al.shape != gt.shape:
                continue
            al = al / np.clip(al.sum(0, keepdims=True), 1e-9, None)
            m = pair_metrics(al, gt)
            m["gate_len_err"] = span - Lg
            m["coverage"] = min(span, Lg) / max(span, Lg)
            per_unit[r.target_unit_id].append(m)
    if not per_unit:
        raise RuntimeError("scored zero target units -- evaluation is broken, not the model")
    keys = list(next(iter(per_unit.values()))[0])
    unit = {k: [] for k in keys}
    for v in per_unit.values():
        for k in keys:
            vals = [x[k] for x in v if not np.isnan(x[k])]
            if vals:
                unit[k].append(float(np.mean(vals)))
    out = {k: float(np.mean(v)) for k, v in unit.items() if v}
    out["gate_len_mae"] = float(np.mean(np.abs(unit["gate_len_err"])))
    out["gate_len_bias"] = out.pop("gate_len_err")
    out["cov_r"] = float(np.mean([u for u in
                                  [np.mean([x["pearson_r"] * x["coverage"] for x in v])
                                   for v in per_unit.values()]]))
    out["n_target_units"] = len(per_unit)
    return out


def load_split(dataset, split, manifest):
    ex = pd.read_parquet(f"data/processed/v26/v26_{dataset}.parquet")
    man = pd.read_parquet(manifest)
    cols = ["target_unit_id", "split"] + (
        ["application_holdout"] if "application_holdout" in man.columns else [])
    man = man[cols].drop_duplicates("target_unit_id")
    ex = ex.merge(man, on="target_unit_id", how="inner")
    if "application_holdout" in ex.columns:
        ex = ex[~ex.application_holdout]
    return ex[ex.split == split].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ckpt"); ap.add_argument("--kind", choices=["v24", "v26"])
    ap.add_argument("--dataset", default="core"); ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", default="data/processed/splits/v26/manifest.parquet")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    os.makedirs(RESD, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    V26MAN = "data/processed/splits/v26/manifest.parquet"
    V23MAN = "data/processed/splits/v26/manifest_v23compat.parquet"
    if a.all:
        jobs = [
            # same clean v26 test set -- v24 vs the best v26
            ("v24_on_v26clean", "v24", V24_CKPT, "core", "test", V26MAN),
            ("v26_regstrong_on_v26clean", "v26", f"{CK}/reg_strong/seed42/ckpt_best.pt",
             "core", "test", V26MAN),
            ("v26_core_on_v26clean", "v26", f"{CK}/core/seed42/ckpt_best.pt",
             "core", "test", V26MAN),
            # THE DECISIVE ONE: v24 ARCHITECTURE trained on v26 DATA (flanks + partners +
            # clean split). Compared against v26 models on the same clean test set, this
            # separates architecture from data.
            ("v24arch_trained_on_v26data", "v24",
             f"{CK}/v24arch_on_v26data/seed42/ckpt_best.pt", "core", "test", V26MAN),
            ("v26_ic_on_legacy291", "v26",
             f"{CK}/v23compat_ic/seed42/ckpt_best.pt", "v23compat", "test", V23MAN),
            # same LEGACY 291 rows -- the like-for-like ablation
            ("v24_on_legacy291", "v24", V24_CKPT, "v23compat", "test", V23MAN),
            ("v26_regstrong_on_legacy291", "v26",
             f"{CK}/v23compat_regstrong/seed42/ckpt_best.pt", "v23compat", "test", V23MAN),
            ("v26_core_on_legacy291", "v26",
             f"{CK}/v23compat_core/seed42/ckpt_best.pt", "v23compat", "test", V23MAN),
        ]
    else:
        jobs = [(a.tag or os.path.basename(a.ckpt), a.kind, a.ckpt, a.dataset, a.split,
                 a.manifest)]

    res = {}
    for tag, kind, ckpt, ds, sp, man in jobs:
        if not os.path.exists(ckpt):
            print(f"  SKIP {tag}: missing {ckpt}", flush=True); continue
        ex = load_split(ds, sp, man)
        model, maxlen = (load_v24(dev, ckpt) if kind == "v24" else load_v26(ckpt, dev))
        try:
            m = evaluate(model, kind, maxlen, ex, dev)
        except Exception as e:                                       # noqa: BLE001
            print(f"  FAILED {tag}: {str(e)[:160]}", flush=True); continue
        m["dataset"], m["split"], m["n_examples"], m["ckpt"] = ds, sp, int(len(ex)), ckpt
        res[tag] = m
        print(f"  {tag}: units={m['n_target_units']} pearson={m['pearson_r']:.4f} "
              f"mae={m['mae']:.4f} jsd={m['jsd_bits']:.4f} topbase={m['topbase_acc']:.4f}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    json.dump(res, open(f"{RESD}/multimetric.json", "w"), indent=2)
    order = ["pearson_r", "cov_r", "cosine", "topbase_acc", "macro_f1", "auroc",
             "mae", "rmse", "jsd_bits", "ic_mae", "gate_len_mae", "gate_len_bias",
             "coverage", "n_target_units"]
    print(f"\n{'metric':<16}" + "".join(f"{t[:17]:>18}" for t in res))
    for k in order:
        print(f"{k:<16}" + "".join(f"{res[t].get(k, float('nan')):18.4f}" for t in res))
    print(f"\nwrote {RESD}/multimetric.json")


if __name__ == "__main__":
    main()
