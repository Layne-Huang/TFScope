#!/usr/bin/env python
"""Directed in-silico DNA evolution of a fixed protein fold, scored by MM-GBSA or FoldX.

Same loop as scripts/evolve_pwm_rosetta.py (scan -> Boltzmann PWM -> consensus -> thread
consensus onto the structure -> repeat), but the per-base ΔΔG comes from either
  --method mmgbsa : OpenMM AMBER14/DNA.OL15 + GBn2   (scripts/run_mmpbsa_zf.scan_pdb)
  --method foldx  : FoldX AnalyseComplex interaction (scripts/run_foldx_zf.scan_pdb)
Tracks the wild-type interface energy each round (binding quality of the current site).

Usage:
  python scripts/evolve_pwm_struct.py --method mmgbsa --pdb design_pdbs/zf_samples/zf21_model.pdb \
      --name zf21 --out results/zf_struct/mmpbsa_evolve --rounds 4
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, "pwm_rosetta")
sys.path.insert(0, "scripts")
BASES = ["A", "C", "G", "T"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["mmgbsa", "foldx"], required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--min-iter", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.method == "mmgbsa":
        from run_mmpbsa_zf import scan_pdb
        scan = lambda p: scan_pdb(p, tau=a.tau, min_iter=a.min_iter)
    else:
        from run_foldx_zf import scan_pdb
        scan = lambda p: scan_pdb(p, tau=a.tau)

    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    from pwm_hybrid.rosetta.mutations import mutate_dna_base, minimize_around_mutation
    pyrosetta.init(get_pyrosetta_init_flags() + " -out:level 0", silent=True)
    init_pyrosetta(psipred_exe="")

    cur_pdb = a.pdb
    history, seen = [], set()
    for rnd in range(a.rounds):
        ppm, ddg, seq = scan(cur_pdb)
        consensus = ''.join(BASES[i] for i in ppm.argmax(1))
        wt_iface  = float(ddg[ddg != 0].mean()) if np.any(ddg != 0) else 0.0
        np.save(os.path.join(a.out, f"{a.name}_round{rnd}_pwm.npy"), ppm.T)
        history.append(dict(round=rnd, dna=seq, consensus=consensus,
                            mean_abs_ddg=round(float(np.abs(ddg).mean()), 3)))
        print(f"[evolve {a.method} {a.name}] round {rnd}: DNA={seq} -> consensus={consensus}  "
              f"mean|ddG|={np.abs(ddg).mean():.2f}", flush=True)
        if consensus == seq or consensus in seen:
            print(f"[evolve {a.method} {a.name}] converged at round {rnd}.")
            break
        seen.add(seq)
        pose = pyrosetta.pose_from_file(cur_pdb)
        for i, (cb, nb) in enumerate(zip(seq, consensus)):
            if cb != nb:
                pose = mutate_dna_base(pose, i + 1, nb)
                pose = minimize_around_mutation(pose, i + 1)
        nxt = os.path.join(a.out, f"{a.name}_evolved_round{rnd+1}.pdb")
        pose.dump_pdb(nxt)
        cur_pdb = nxt

    json.dump(history, open(os.path.join(a.out, f"{a.name}_evolution.json"), "w"), indent=2)
    print(f"\n[evolve {a.method} {a.name}] trajectory:")
    for h in history:
        print(f"  round {h['round']}: {h['dna']} -> {h['consensus']}  (mean|ddG| {h['mean_abs_ddg']})")
    print(f"saved -> {a.out}/{a.name}_evolution.json")


if __name__ == "__main__":
    main()
