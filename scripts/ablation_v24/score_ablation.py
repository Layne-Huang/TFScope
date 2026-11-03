#!/usr/bin/env python
"""Score every finished v24-ablation checkpoint through the frozen unified evaluator.

Reuses `iclr.score_checkpoint_unified` verbatim, so ablation numbers land on exactly
the same two panels as the ICLR B0-B8 ladder and the shipped v24 reference:

  Panel A  gene_content_r — GT motif length + oracle shift/RC registration. Pure
                            "is the motif right". No length/coverage credit.
  Panel B  gene_covR      — the model's own predicted gate span. covR = overlap_r x
                            coverage; coverage and gate-length error reported separately.

Inference is sequence-only: the evaluator never feeds recog_prior / contact targets, so
the contact ablations are compared on what the model actually learned, not on what it is
handed at test time.

Runs on CPU by default so it cannot preempt a training GPU. ~2-4 min per checkpoint.

    scripts/ablation_v24/score_ablation.py                 # score everything unscored
    scripts/ablation_v24/score_ablation.py --rescore       # redo even if already scored
    scripts/ablation_v24/score_ablation.py --only A1_structonly A2_nodistill
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

CKROOT = os.environ.get(
    "ABL_CKROOT", "/data1/leihuang/TFScope_store/ablation_v24/checkpoints/ablation_v24")
OUT = "results/ablation_v24/unified_ablation.json"
PYTHON_BIN = os.environ.get(
    "PYTHON_BIN", "/data1/leihuang/miniconda3/envs/tfscope/bin/python")


def discover():
    """[(tag, seed, ckpt_path)] for every run with a finished ckpt_best.pt."""
    found = []
    for p in sorted(glob.glob(os.path.join(CKROOT, "*", "seed*", "ckpt_best.pt"))):
        seed_dir, tag_dir = os.path.basename(os.path.dirname(p)), \
            os.path.basename(os.path.dirname(os.path.dirname(p)))
        found.append((tag_dir, seed_dir.replace("seed", ""), p))
    return found


SHIPPED_SRC = "results/iclr_phase1_apples_to_apples/unified_models.json"
SHIPPED_KEY = "B8_v24"


def _import_shipped(out_path):
    """Copy the already-scored shipped v24 seed-42 run in as the interim reference.

    B8_v24 is the 6-GPU DDP `contact_v24_seed42` checkpoint scored through this same
    evaluator, so it is directly comparable. `A0_full` re-runs that recipe single-GPU
    under this harness; until it finishes, the table pairs against the shipped run.
    """
    if not os.path.exists(SHIPPED_SRC):
        return
    src = json.load(open(SHIPPED_SRC))
    if SHIPPED_KEY not in src:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cur = json.load(open(out_path)) if os.path.exists(out_path) else {}
    if "A0_shipped_s42" in cur:
        return
    cur["A0_shipped_s42"] = src[SHIPPED_KEY]
    json.dump(cur, open(out_path, "w"), indent=2)
    print(f"  imported {SHIPPED_KEY} -> A0_shipped_s42 (interim reference)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="variant tags to restrict to")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--threads", default="4")
    ap.add_argument("--import-shipped", action="store_true", default=True,
                    help="copy the already-scored shipped v24 seed-42 run (B8_v24) in as "
                         "A0_shipped_s42, so the table has a reference before A0_full finishes")
    args = ap.parse_args()

    _import_shipped(args.out) if args.import_shipped else None

    runs = discover()
    if args.only:
        runs = [r for r in runs if r[0] in set(args.only)]
    if not runs:
        sys.exit(f"no finished checkpoints under {CKROOT}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set(json.load(open(args.out))) if os.path.exists(args.out) else set()

    print(f"{len(runs)} finished run(s) under {CKROOT}")
    for tag, seed, ckpt in runs:
        key = f"{tag}_s{seed}"
        if key in done and not args.rescore:
            print(f"  skip {key} (already in {args.out})")
            continue
        print(f"  scoring {key} ...", flush=True)
        env = dict(os.environ, OMP_NUM_THREADS=args.threads, PYTHONPATH="src")
        rc = subprocess.call(
            [PYTHON_BIN, "-m", "iclr.score_checkpoint_unified",
             "--ckpt", ckpt, "--tag", key, "--device", args.device, "--out", args.out],
            env=env)
        if rc != 0:
            print(f"  FAILED ({rc}) {key}", file=sys.stderr)

    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
