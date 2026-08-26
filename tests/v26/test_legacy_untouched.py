#!/usr/bin/env python
"""Acceptance guard: v24 / v25flank / v25xtal artifacts must never be modified by v26 work.

First run records a baseline manifest of (size, mtime, sha256-of-first-8MB) for every
protected artifact. Every later run re-checks it and FAILS on any drift.

  pytest tests/v26/test_legacy_untouched.py
  python tests/v26/test_legacy_untouched.py --rebaseline   # only after an INTENTIONAL change
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# Absolute AFS paths on purpose: this guard must check the REAL artifacts even when the
# suite runs from the /data1 mirror (which carries only a subset of legacy inputs). Using
# relative paths made the mirror run report 4 v25 contact JSONs as DELETED -- a false alarm.
AFS = "/afs/csail.mit.edu/u/l/leihuang/project/TFScope"
BASELINE = f"{AFS}/tests/v26/legacy_baseline.json"
HEAD_BYTES = 8 * 1024 * 1024        # hash the first 8 MB; full hash of 330 MB ckpts is wasteful

PROTECTED_FILES = [f"{AFS}/" + p for p in [
    "data/processed/tf_pwm_training_v23.parquet",
    "data/processed/tf_pwm_training_v25flank.parquet",
    "data/processed/tf_pwm_training_v25xtal.parquet",
    "data/processed/tf_pwm_training_v22.parquet",
    "data/processed/splits/train_v22/split.json",
    "data/processed/splits/train_v22/assignments.parquet",
    "data/contact_maps/contact_targets_v23.json",
    "data/contact_maps/recognition_residues_v23.json",
    "data/contact_maps/contact_targets_v25flank.json",
    "data/contact_maps/recognition_residues_v25flank.json",
    "data/contact_maps/contact_targets_v25xtal.json",
    "data/contact_maps/recognition_residues_v25xtal.json",
]] + [
    "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt",
    "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/config.json",
]

PROTECTED_DIRS = [
    "/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens",
]


def _fingerprint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(HEAD_BYTES))
    return {"size": st.st_size, "mtime": int(st.st_mtime), "head_sha256": h.hexdigest()}


def _collect() -> dict:
    out = {}
    for p in PROTECTED_FILES:
        out[p] = _fingerprint(p)
    for d in PROTECTED_DIRS:
        for root, _dirs, files in os.walk(d):
            for f in sorted(files):
                fp = os.path.join(root, f)
                out[fp] = _fingerprint(fp)
    return out


def test_legacy_untouched():
    current = _collect()
    if not os.path.exists(BASELINE):
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w") as fh:
            json.dump(current, fh, indent=1, sort_keys=True)
        print(f"baseline created: {BASELINE} ({len(current)} artifacts)")
        return

    baseline = json.load(open(BASELINE))
    drift = []
    for path, base in baseline.items():
        cur = current.get(path)
        if base is None and cur is None:
            continue
        if cur is None:
            drift.append(f"DELETED: {path}")
        elif base is None:
            drift.append(f"APPEARED (was absent at baseline): {path}")
        elif cur["size"] != base["size"] or cur["head_sha256"] != base["head_sha256"]:
            drift.append(f"MODIFIED: {path} "
                         f"(size {base['size']}->{cur['size']})")
    assert not drift, "legacy artifacts changed:\n  " + "\n  ".join(drift)
    print(f"OK: {len(baseline)} legacy artifacts unchanged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebaseline", action="store_true")
    a = ap.parse_args()
    if a.rebaseline and os.path.exists(BASELINE):
        os.remove(BASELINE)
    test_legacy_untouched()
    sys.exit(0)
