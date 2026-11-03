#!/usr/bin/env python
"""D1: is motif length recoverable from the vector the gate head actually reads?

The gate head consumes `moe_out` -- a single pooled 512-d protein vector
(`models/tfscope.py`: global_pool + dbd_pool -> projection -> MoE). Both pooling paths
normalise over positions, so they are *intensive* operators and are close to invariant to
how many residues a protein has. Motif length is a count-like, *extensive* quantity. This
probe decides which of two stories is true:

  ridge(moe_out) >> the gate's own pearson 0.10  -> the information SURVIVED pooling and
                                                   the failure is in the head/loss
  ridge(moe_out) ~ 0.10                          -> pooling destroyed it and no amount of
                                                   loss tuning will fix the current readout

Controls are the point, not the probe. `log(crop_len)` and `family_id` are free features
that together reach pearson ~0.45 on this data, so a pooled-representation probe only
means something if it beats them. Folds are grouped by CD-HIT `cluster`, never random --
random folds would read homology as signal.

  PYTHONPATH=src python -m iclr.probe_length_from_pooled --ckpt <ckpt.pt> --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "."); sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from iclr.unified_eval import trimmed_core  # noqa: E402

UNI = "data/processed/tf_pwm_unified_v25.parquet"
SPLIT = "data/processed/splits/unified_v25/split.json"
OUT = "results/v25_bench84/length_probe.json"


def dump_pooled(ckpt, data, split, device, which):
    """Return {filename: {'moe_out':(512,), 'combined':(512,), 'gate_len':int}}."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tfscope.config import TFScopeConfig
    from tfscope.models.tfscope import TFScopeModel
    from tfscope.data.dataset import TFDataset, collate_variable_length

    cfg = TFScopeConfig()
    cfgp = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.exists(cfgp):
        for k, v in json.load(open(cfgp)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception: setattr(cfg, k, v)
    model = TFScopeModel(cfg).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.eval()

    # tap the two candidate vectors without touching the model definition
    grab = {}
    model.projection.register_forward_hook(
        lambda m, i, o: grab.__setitem__("combined", o.detach().float().cpu().numpy()))
    if getattr(model, "use_moe", False) and getattr(cfg, "moe_granularity", "") == "residue":
        grab_moe_is_combined = True     # residue-MoE path: moe_out IS combined
    else:
        grab_moe_is_combined = False
        model.moe.register_forward_hook(
            lambda m, i, o: grab.__setitem__("moe_out", o.detach().float().cpu().numpy()))

    out = {}
    for sp in which:
        ds = TFDataset(cfg, data, split, split=sp, max_seq_len=1024)
        ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_variable_length)
        i0 = 0
        for b in ld:
            b = {k: v.to(device, dtype=torch.float32 if v.is_floating_point() else torch.long)
                 for k, v in b.items()}
            with torch.no_grad():
                gate_logits, _, aux = model(b["sequence_tokens"], b["dbd_mask"],
                                            b["family_id"], esmc_emb=b.get("esmc_emb"))
            comb = grab["combined"]
            moe = comb if grab_moe_is_combined else grab["moe_out"]
            glen = (torch.sigmoid(gate_logits) > 0.5).sum(1).cpu().numpy()
            for j in range(comb.shape[0]):
                out[ds.filenames[i0 + j]] = {"combined": comb[j], "moe_out": moe[j],
                                             "gate_len": int(glen[j]), "split": sp}
            i0 += comb.shape[0]
        print(f"  [{sp}] {i0} rows")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    from scipy.stats import pearsonr
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import GradientBoostingRegressor

    uni = pd.read_parquet(UNI); uni["filename"] = uni.filename.astype(str)
    uni["core_len"] = uni.pwm.map(
        lambda b: (lambda c: c.shape[1] if c is not None else np.nan)(
            trimmed_core(np.frombuffer(b, np.float32).reshape(4, -1))))
    meta = uni.set_index("filename")

    feats = dump_pooled(a.ckpt, UNI, SPLIT, a.device, ("train", "val", "test"))
    fns = [f for f in feats if f in meta.index and np.isfinite(meta.core_len.get(f, np.nan))]
    sp = np.array([feats[f]["split"] for f in fns])
    y = np.array([meta.core_len[f] for f in fns], float)
    crop = np.log(np.array([len(meta.sequence[f]) for f in fns], float))
    fam = np.array([int(meta.family_id[f]) for f in fns])
    gate = np.array([feats[f]["gate_len"] for f in fns], float)
    tr, te = sp == "train", sp == "test"
    print(f"\n{len(fns)} rows | train {tr.sum()} test {te.sum()} | "
          f"true len sd {y[te].std():.2f}")

    def report(name, pred_te):
        r = pearsonr(pred_te, y[te])
        print(f"  {name:<34}{np.abs(pred_te - y[te]).mean():>8.2f}"
              f"{r[0]:>9.3f}{r[1]:>10.2g}")
        return {"mae": float(np.abs(pred_te - y[te]).mean()),
                "pearson": float(r[0]), "p": float(r[1])}

    res = {}
    print(f"\n{'predictor':<34}{'MAE':>8}{'pearson':>9}{'p':>10}")
    res["constant_train_median"] = report("constant = train median",
                                          np.full(te.sum(), np.median(y[tr])))
    res["gate_head"] = report("the model's gate (what we have)", gate[te])
    fam_med = pd.Series(y[tr]).groupby(fam[tr]).median()
    res["family_median"] = report("family median (free feature)",
                                  np.array([fam_med.get(f, np.median(y[tr])) for f in fam[te]]))
    g = GradientBoostingRegressor(random_state=0).fit(
        np.c_[crop[tr], fam[tr]], y[tr])
    res["gbm_crop_family"] = report("GBM(log crop len, family) <- the bar",
                                    g.predict(np.c_[crop[te], fam[te]]))

    for key in ("combined", "moe_out"):
        X = np.stack([feats[f][key] for f in fns])
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], y[tr])
        res[f"ridge_{key}"] = report(f"ridge on {key} (512-d pooled)", m.predict(X[te]))
        Xa = np.c_[X, crop]
        m2 = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xa[tr], y[tr])
        res[f"ridge_{key}_plus_croplen"] = report(f"ridge on {key} + log crop len",
                                                  m2.predict(Xa[te]))

    verdict = ("head/loss problem — the information survived pooling"
               if res["ridge_moe_out"]["pearson"] > 0.30 else
               "representation problem — pooling destroyed it")
    print(f"\nVERDICT: {verdict}")
    res["_verdict"] = verdict
    res["_ckpt"] = a.ckpt
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
