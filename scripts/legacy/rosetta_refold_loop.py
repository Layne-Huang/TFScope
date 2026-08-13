"""Structure-guided iterative refold loop — one iteration.

Given an AF3 fold directory for the CURRENT DNA (model_0..4.cif), this:
  1. runs the pwm_hybrid Rosetta mutant scan on EACH of the 5 models (default ddg_filter, FastRelax,
     RM8B charges — i.e. the user's settings; PWM_INTERFACE_MODE unset),
  2. aggregates per-(position,base) ddG by ENSEMBLE MEDIAN across the 5 models (robust to the large
     per-model relax variance),
  3. builds a PWM via softmax(-median_ddG / tau) [original base ddG := 0] and takes the argmax consensus,
  4. compares to the current DNA and to history → convergence / oscillation check,
  5. writes the NEXT first-strand DNA to fold and the running history.

You then AF3-refold the printed sequence (5 models) and re-run this on the new fold dir. Repeat until
'CONVERGED' (consensus unchanged) or an oscillation/iteration cap is reported.

Usage:
  python scripts/rosetta_refold_loop.py --fold_dir <dir with *_model_{0..4}.cif> --iter <N> \
      [--tau 1.0] [--history results/myod1_mut/refold_loop_history.json] [--workroot <scan out dir>]
Run with the multiflow python (PyRosetta), PYTHONPATH=pwm_rosetta.
"""
import os, sys, glob, json, argparse
os.environ.pop("PWM_INTERFACE_MODE", None)            # user settings: default ddg_filter (NOT prot_dna)
import numpy as np
import pyrosetta
from pwm_hybrid.rosetta import init as rinit

ap = argparse.ArgumentParser()
ap.add_argument("--fold_dir", required=True, help="dir with <name>_model_{0..4}.cif for the CURRENT DNA")
ap.add_argument("--iter", type=int, required=True)
ap.add_argument("--tau", type=float, default=1.0, help="softmax temperature on -ddG (REU)")
ap.add_argument("--history", default="results/myod1_mut/refold_loop_history.json")
ap.add_argument("--workroot", default=None, help="where to write per-model scan CSVs (default <fold_dir>/_scan)")
ap.add_argument("--max_iter", type=int, default=6)
args = ap.parse_args()

pyrosetta.init(rinit.get_pyrosetta_init_flags()); rinit.init_pyrosetta(psipred_exe="")
from pwm_hybrid.pipeline import generate_pwm_hybrid

cifs = sorted(glob.glob(os.path.join(args.fold_dir, "*_model_[0-4].cif")))
assert cifs, f"no *_model_[0-4].cif in {args.fold_dir}"
workroot = args.workroot or os.path.join(args.fold_dir, "_scan")
os.makedirs(workroot, exist_ok=True)

import pandas as pd
BASES = "ACGT"
# run scan per model (cache: skip if CSV already present)
csvs = []
for cif in cifs:
    od = os.path.join(workroot, os.path.basename(cif).replace(".cif", ""))
    csv = os.path.join(od, "pwm_results_hybrid.csv")
    if not os.path.isfile(csv):
        os.makedirs(od, exist_ok=True)
        print(f"[scan] {os.path.basename(cif)}", flush=True)
        generate_pwm_hybrid(protein_seq=None, dna_seq=None, output_dir=od, wt_pdb=cif,
                            minimize_local=True, use_relax=True)
    csvs.append(csv)

# aggregate: per position, per base -> list of ddG across models (original base contributes 0)
dfs = [pd.read_csv(c) for c in csvs]
positions = sorted(dfs[0].position.unique())
orig = {int(r.position): r.original for _, r in dfs[0].iterrows()}
ddg = {p: {b: [] for b in BASES} for p in positions}   # ddG = mut - wt (relative to seeded base)
for df in dfs:
    for p in positions:
        ddg[p][orig[p]].append(0.0)                    # seeded/original base is the reference (0)
        for _, r in df[df.position == p].iterrows():
            ddg[p][r.mutant].append(float(r.ddG))
med = {p: {b: (np.median(v) if v else np.nan) for b, v in ddg[p].items()} for p in positions}

# PWM via softmax(-median ddG / tau); consensus = argmax (== min median ddG)
P = np.zeros((4, len(positions)))
cons = []
for j, p in enumerate(positions):
    s = np.array([-med[p][b] / args.tau for b in BASES])   # favorable (low ddG) -> high score
    s = s - np.nanmax(s); w = np.exp(s); w = w / w.sum()
    P[:, j] = w
    cons.append(BASES[int(np.nanargmin([med[p][b] for b in BASES]))])
new_dna = "".join(cons)
cur_dna = "".join(orig[p] for p in positions)

# history + convergence / oscillation
hist = json.load(open(args.history)) if os.path.isfile(args.history) else []
seqs_so_far = [h["new_dna"] for h in hist]
converged = (new_dna == cur_dna)
oscillating = (not converged) and (new_dna in seqs_so_far)
status = "CONVERGED" if converged else ("OSCILLATING" if oscillating else
                                        ("MAX_ITER" if args.iter >= args.max_iter else "CONTINUE"))

# report core E-box (positions 3-8, 1-based) if present
def core_str(seqstr): return seqstr[2:8] if len(seqstr) >= 8 else seqstr
rec = dict(iter=args.iter, fold_dir=args.fold_dir, tau=args.tau,
           cur_dna=cur_dna, new_dna=new_dna, cur_core=core_str(cur_dna), new_core=core_str(new_dna),
           median_ddg={str(p): {b: (None if np.isnan(med[p][b]) else round(med[p][b], 3)) for b in BASES} for p in positions},
           pwm=[[round(float(P[i, j]), 4) for j in range(len(positions))] for i in range(4)],
           n_models=len(csvs), status=status)
hist.append(rec); json.dump(hist, open(args.history, "w"), indent=1)

print("\n" + "=" * 64)
print(f"ITERATION {args.iter}  (n_models={len(csvs)}, tau={args.tau})")
print(f"  current DNA : {cur_dna}   core {core_str(cur_dna)}")
print(f"  new DNA     : {new_dna}   core {core_str(new_dna)}")
print(f"  per-position median ddG (best base in []):")
for p in positions:
    row = "  ".join(f"{b}:{('  nan' if np.isnan(med[p][b]) else f'{med[p][b]:+5.2f}')}" for b in BASES)
    bb = BASES[int(np.nanargmin([med[p][b] for b in BASES]))]
    print(f"    pos {p:>2} (orig {orig[p]}): {row}   -> [{bb}]")
print(f"\n  STATUS: {status}")
if status in ("CONVERGED", "OSCILLATING", "MAX_ITER"):
    print("  -> stop. " + {"CONVERGED": "consensus is self-consistent.",
                            "OSCILLATING": f"revisited a past sequence ({new_dna}); report the cycle.",
                            "MAX_ITER": "hit iteration cap."}[status])
else:
    print(f"  -> NEXT: AF3-refold this first-strand DNA (5 models), then re-run on the new fold dir:")
    print(f"           {new_dna}")
print("=" * 64)
print(f"history: {args.history}")
