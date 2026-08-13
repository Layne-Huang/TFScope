#!/usr/bin/env python
"""Evolve the DNA binding site of a fixed protein fold using pwm_rosetta.

Directed in-silico evolution: starting from an AF3-folded protein-DNA complex, repeatedly
  (1) run the pwm_rosetta ddG scan -> Boltzmann PWM -> consensus,
  (2) thread that consensus onto the structure (mutate every base + local minimise),
until the consensus stops changing. Tracks the wild-type interface ddG each round, which should
become more favourable as the site evolves to fit the fold.

Usage:
  python scripts/evolve_pwm_rosetta.py --pdb results/zf_struct/zf21_rag_fold.pdb \
      --name zf21 --outdir results/zf_struct/pwm_rosetta_evolve --rounds 4
"""
import os, sys, argparse, shutil
import numpy as np
sys.path.insert(0, "pwm_rosetta")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    pyrosetta.init(get_pyrosetta_init_flags())
    init_pyrosetta(psipred_exe="")          # disable SSpred filters
    from pwm_hybrid.pipeline import generate_pwm_hybrid
    from pwm_hybrid import pwm_from_hybrid_csv
    from pwm_hybrid.rosetta.mutations import mutate_dna_base, minimize_around_mutation
    from pwm_hybrid.rosetta.structure import extract_dna_sequence
    B = "ACGT"

    cur_pdb = args.pdb
    history = []
    seen = set()
    for rnd in range(args.rounds):
        rdir = os.path.join(args.outdir, f"{args.name}_round{rnd}")
        generate_pwm_hybrid(protein_seq=None, dna_seq=None, output_dir=rdir,
                            wt_pdb=cur_pdb, minimize_local=True, use_relax=False)
        csv = os.path.join(rdir, "pwm_results_hybrid.csv")
        seq, en, ppm, pwm = pwm_from_hybrid_csv(csv, tau=args.tau)
        consensus = "".join(B[i] for i in ppm.argmax(1))
        # wild-type interface ddG of the current structure (binding quality of current DNA)
        import pandas as pd
        wt_ddg = float(pd.read_csv(csv)["wt_ddg"].iloc[0])
        np.save(os.path.join(rdir, "pwm.npy"), ppm.T)
        history.append(dict(round=rnd, dna=seq, consensus=consensus, wt_interface_ddg=round(wt_ddg, 2)))
        print(f"[evolve {args.name}] round {rnd}: DNA={seq} -> consensus={consensus}  "
              f"wt_iface_ddG={wt_ddg:.2f}", flush=True)
        if consensus == seq or consensus in seen:
            print(f"[evolve {args.name}] converged at round {rnd} (consensus stable).")
            break
        seen.add(seq)
        # thread the consensus onto the structure -> next-round PDB
        pose = pyrosetta.pose_from_pdb(cur_pdb)
        for i, (cb, nb) in enumerate(zip(seq, consensus)):
            if cb != nb:
                pose = mutate_dna_base(pose, i + 1, nb)
                pose = minimize_around_mutation(pose, i + 1)
        nxt = os.path.join(args.outdir, f"{args.name}_evolved_round{rnd+1}.pdb")
        pose.dump_pdb(nxt)
        cur_pdb = nxt

    import json
    json.dump(history, open(os.path.join(args.outdir, f"{args.name}_evolution.json"), "w"), indent=2)
    print(f"\n[evolve {args.name}] trajectory:")
    for h in history:
        print(f"  round {h['round']}: {h['dna']} -> {h['consensus']}  (iface ddG {h['wt_interface_ddg']})")
    print(f"saved -> {args.outdir}/{args.name}_evolution.json")

if __name__ == "__main__":
    main()
