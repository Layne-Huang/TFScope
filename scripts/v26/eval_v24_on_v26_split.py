#!/usr/bin/env python
"""PRIORITY 1: score the v24 checkpoint on the CLEAN v26 split.

The decisive missing number. Two figures are currently quoted side by side but are NOT
comparable:
    v24 covR 0.5304   on the LEAKY train_v22 split (291/291 test rows have
                      structure-defined input boundaries; all 20 Barrera genes in train)
    v26 covR 0.3507   on the clean v26 split (178 target units, all leakage assertions 0)

If v24 also drops to ~0.35 here, v26 has not regressed and the ~0.18 gap IS the leakage --
which is the audit's central claim. If v24 stays near 0.50, v26 has genuinely regressed and
the cause must be found before any more architecture work.

v24 requires family_id, which v26 deliberately never supplies. We therefore run TWO variants:
  --family mapped : v26's analysis-only family string mapped onto v24's 10 canonical ids
  --family dummy  : every example forced to 'Other' (id 9)
Prior work found family conditioning to be near-inert, so the two should agree -- but that is
verified here rather than assumed.

  python scripts/v26/eval_v24_on_v26_split.py --family mapped
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
from tfscope.models.alignment import align_pwm            # noqa: E402

V24_CKPT = ("/data1/leihuang/project/TFScope/checkpoints/v24_contact/"
            "contact_v24_seed42/ckpt_best.pt")
RESD = "results/v26"
AA = {"L": 4, "A": 5, "G": 6, "V": 7, "S": 8, "E": 9, "R": 10, "T": 11, "I": 12, "D": 13,
      "P": 14, "K": 15, "Q": 16, "N": 17, "F": 18, "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23}

# v24's canonical 10-family scheme (build_training_table.py FID)
FID = {"C2H2_short": 0, "C2H2_medium": 1, "C2H2_long": 2, "bHLH": 3, "Homeodomain": 4,
       "bZIP": 5, "Nuclear_Receptor": 6, "Forkhead": 7, "ETS": 8, "Other": 9}


def map_family(fam_str: str) -> int:
    """v26 analysis-only family string -> v24 family id. Unknown -> Other."""
    for f in str(fam_str).split(";"):
        if f in FID:
            return FID[f]
        if f.startswith("C2H2"):
            return FID["C2H2_long"]
        if f in ("Homeo_prospero",):
            return FID["Homeodomain"]
    return FID["Other"]


def build_v24(device):
    from tfscope.config import TFScopeConfig
    from tfscope.models.tfscope import TFScopeModel
    cfg = TFScopeConfig()
    cfgp = os.path.join(os.path.dirname(V24_CKPT), "config.json")
    for k, v in json.load(open(cfgp)).items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception:
                setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(device)
    sd = torch.load(V24_CKPT, map_location=device, weights_only=False)
    missing = m.load_state_dict(sd.get("model", sd), strict=False)
    print(f"  loaded v24 (missing={len(missing.missing_keys)} "
          f"unexpected={len(missing.unexpected_keys)})", flush=True)
    return m.eval(), cfg


def target_pwm(row, maxlen):
    a = np.frombuffer(row.pwm, dtype=np.float32).reshape(4, -1).astype(np.float32)
    L = min(int(row.motif_length), maxlen, a.shape[1])
    return a[:, :L], L


@torch.no_grad()
def score(model, cfg, ex, device, family_mode, batch=4):
    per_unit = defaultdict(list)
    n_fail = 0
    for s in range(0, len(ex), batch):
        sub = ex.iloc[s:s + batch]
        seqs = [str(r.sequence) for r in sub.itertuples()]
        L = max(len(x) for x in seqs)
        T = torch.full((len(seqs), L), 1, dtype=torch.long)
        D = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, q in enumerate(seqs):
            T[i, :len(q)] = torch.tensor([AA.get(c, 4) for c in q])
            D[i, :len(q)] = True          # v26_core is a tight DBD crop -> mask all True
        fid = torch.tensor([9 if family_mode == "dummy"
                            else map_family(r.dbd_families_for_analysis_only)
                            for r in sub.itertuples()])
        try:
            # v24 returns LOGITS: (gate_logits pre-sigmoid, pwm_logits pre-softmax).
            # Treating them as probabilities made every span nonsense and silently produced
            # zero scorable examples.
            gate_logits, pwm_logits, _ = model(T.to(device), D.to(device), fid.to(device))
        except Exception as e:                                    # noqa: BLE001
            n_fail += len(sub); print(f"    batch failed: {str(e)[:120]}", flush=True); continue
        pwm = pwm_logits.softmax(dim=1).float().cpu().numpy()
        gate = gate_logits.sigmoid().float().cpu().numpy()
        for j, r in enumerate(sub.itertuples()):
            gt, Lg = target_pwm(r, cfg.max_motif_length)
            span = max(1, int((gate[j] > 0.5).sum()))     # committed positions, as v24 scores it
            pred = pwm[j][:, :span]
            try:
                al, _, _, _ = align_pwm(pred, gt, max_shift=10, consider_revcomp=True,
                                        min_overlap=3)
                a, g = al.flatten(), gt.flatten()
                if a.size == g.size and a.std() > 0 and g.std() > 0:
                    rr = float(np.corrcoef(a, g)[0, 1])
                    cov = min(span, Lg) / max(span, Lg)
                    per_unit[r.target_unit_id].append((rr, rr * cov))
            except Exception:
                n_fail += 1
    if not per_unit:
        # Never report 0.0000 as if it were a measurement.
        raise RuntimeError(
            f"scored ZERO target units (n_fail={n_fail}) -- the evaluation is broken, "
            "not the model. Check the model output convention before trusting any number.")
    rs = [np.mean([x[0] for x in v]) for v in per_unit.values()]
    cs = [np.mean([x[1] for x in v]) for v in per_unit.values()]
    return {"content_r": float(np.mean(rs)), "cov_r": float(np.mean(cs)),
            "n_units": len(per_unit), "n_fail": n_fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["mapped", "dummy", "both"], default="both")
    ap.add_argument("--dataset", default="core")
    a = ap.parse_args()
    os.makedirs(RESD, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ex = pd.read_parquet(f"data/processed/v26/v26_{a.dataset}.parquet")
    man = pd.read_parquet("data/processed/splits/v26/manifest.parquet")[
        ["target_unit_id", "split", "application_holdout"]].drop_duplicates("target_unit_id")
    ex = ex.merge(man, on="target_unit_id", how="inner")
    ex = ex[(~ex.application_holdout) & (ex.split == "test")].reset_index(drop=True)
    print(f"v26 CLEAN test: {len(ex)} examples, {ex.target_unit_id.nunique()} target units",
          flush=True)

    model, cfg = build_v24(device)
    out = {"dataset": a.dataset, "n_examples": int(len(ex)),
           "n_target_units": int(ex.target_unit_id.nunique()),
           "v24_ckpt": V24_CKPT}
    modes = ["mapped", "dummy"] if a.family == "both" else [a.family]
    for m in modes:
        r = score(model, cfg, ex, device, m)
        out[f"v24_on_v26test_family_{m}"] = r
        print(f"  v24 on v26 clean test [family={m}]: content_r={r['content_r']:.4f} "
              f"cov_r={r['cov_r']:.4f} units={r['n_units']} failed={r['n_fail']}", flush=True)

    # reference points for the comparison
    out["reference"] = {
        "v24_on_LEAKY_train_v22_test": {"content_r": 0.6294, "cov_r": 0.5304,
                                        "note": "291 rows, structure-defined boundaries"},
        "v26_core_on_v26_val": {"cov_r_mean": 0.3507, "n_seeds": 3,
                                "note": "validation metric, not test"},
    }
    json.dump(out, open(f"{RESD}/v24_on_v26_split.json", "w"), indent=2)
    print(f"\nwrote {RESD}/v24_on_v26_split.json", flush=True)
    if len(modes) == 2:
        d = (out["v24_on_v26test_family_mapped"]["cov_r"]
             - out["v24_on_v26test_family_dummy"]["cov_r"])
        print(f"family sensitivity (mapped - dummy) cov_r = {d:+.4f} -> "
              f"{'family head is NOT inert' if abs(d) > 0.02 else 'family head near-inert, as expected'}")


if __name__ == "__main__":
    main()
