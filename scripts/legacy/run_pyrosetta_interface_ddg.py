#!/usr/bin/env python3
"""Compute protein-DNA interface energy for independently Boltz-folded mutant complexes.

Uses the exact same energy method as pwm_rosetta:
  cif → pdb → rename_chains → MinMover → calculate_interface_ddg (ddg_filter)

For each mutant complex, computes the absolute interface energy (wt_ddg in
pwm_rosetta terms), then:

  ΔΔG(mut) = interface_energy(mut_complex) − interface_energy(WT_complex)

Positive ΔΔG = mutation hurts binding → Boltzmann-weighted to PWM.

Usage:
    python scripts/run_pyrosetta_interface_ddg.py \
        --manifest  results/boltz_rela_dna_scan/mutation_manifest.json \
        --boltz-out /path/to/boltz_runs/rela_dna_scan \
        --out-dir   /path/to/pwm_rosetta_runs/rela_dna_scan \
        --wt-name   rela_WT
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, "pwm_rosetta")


def compute_complex_interface_energy(cif_path, work_dir):
    """Return the interface energy of a protein-DNA complex using pwm_rosetta's method.

    Mirrors steps 1-2 of generate_pwm_hybrid:
      cif → pdb → rename_chains → MinMover(bb+chi) → ddg_filter
    """
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pwm_hybrid.rosetta.structure import convert_cif_to_pdb, rename_chains
    from pwm_hybrid.rosetta.scoring import calculate_interface_ddg
    from pwm_hybrid.rosetta import init as _rosetta_init

    os.makedirs(work_dir, exist_ok=True)

    # CIF → PDB → rename chains to A (protein) / B,C (DNA)
    pdb_path     = os.path.join(work_dir, "wt_tmp.pdb")
    renamed_path = os.path.join(work_dir, "wt_renamed.pdb")
    convert_cif_to_pdb(cif_path, pdb_path)
    rename_chains(pdb_path, renamed_path)

    pose = pose_from_pdb(renamed_path)

    # MinMover identical to generate_pwm_hybrid (minimize_local=True, use_relax=False)
    movemap = MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    mm = MinMover()
    mm.score_function(_rosetta_init.sfxn)
    mm.movemap(movemap)
    mm.apply(pose)

    return calculate_interface_ddg(pose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",  required=True)
    ap.add_argument("--out-dir",   required=True)
    ap.add_argument("--wt-name",   default="rela_WT")
    ap.add_argument("--boltz-out", default="")
    ap.add_argument("--skip-done", action="store_true",
                    help="Skip entries whose result is already in interface_energies_raw.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "interface_energies_raw.json")

    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    pyrosetta.init(get_pyrosetta_init_flags())
    init_pyrosetta(psipred_exe="")

    manifest = json.load(open(args.manifest))

    # Resolve CIF paths from Boltz output tree
    for m in manifest:
        name = m.get("af3_name") or m.get("name")
        m["af3_name"] = name
        if not os.path.exists(m.get("cif_path", "")):
            pats = glob.glob(
                f"{args.boltz_out}/**/predictions/{name}/{name}_model_0.cif",
                recursive=True)
            m["cif_path"] = pats[0] if pats else None

    # Load existing results for --skip-done
    results = {}
    if args.skip_done and os.path.exists(raw_path):
        results = json.load(open(raw_path))
        print(f"Loaded {len(results)} cached results from {raw_path}")

    # Compute interface energy for each complex
    for m in manifest:
        name = m["af3_name"]
        if name in results and results[name].get("ok"):
            print(f"[CACHED] {name}: {results[name]['interface_energy']:.3f}")
            continue

        cif = m.get("cif_path")
        if not cif or not os.path.exists(cif):
            print(f"[SKIP]   {name}: CIF not found")
            continue

        work_dir = os.path.join(args.out_dir, name)
        print(f"\n=== {name} ===\n  CIF: {cif}")
        try:
            e = compute_complex_interface_energy(cif, work_dir)
            print(f"  interface energy = {e:.3f} kcal/mol")
            results[name] = {
                "interface_energy": e, "ok": True,
                "pos":      m.get("pos", 0),
                "wt_base":  m.get("wt_base", ""),
                "mut_base": m.get("mut_base", ""),
                "fwd_dna":  m.get("fwd_dna", ""),
            }
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            results[name] = {"ok": False, "error": str(exc)}

        # Save after every structure so --skip-done can resume
        with open(raw_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nRaw interface energies → {raw_path}")

    # ── Build ΔΔG matrix ─────────────────────────────────────────────────────
    wt_name = args.wt_name
    if wt_name not in results or not results[wt_name]["ok"]:
        print(f"WT '{wt_name}' not in results — cannot build PWM.")
        return

    wt_e    = results[wt_name]["interface_energy"]
    fwd_seq = results[wt_name]["fwd_dna"]
    print(f"\nWT interface energy = {wt_e:.3f} kcal/mol")
    print(f"Forward DNA: {fwd_seq}\n")

    L        = len(fwd_seq)
    BASE2IDX = {"A":0,"C":1,"G":2,"T":3}
    ddg_mat  = np.zeros((L, 4))

    print("Interface ΔΔG matrix (cols = A, C, G, T):")
    print("pos  WT    A        C        G        T")
    for pos0, wt_base in enumerate(fwd_seq):
        for base, idx in BASE2IDX.items():
            if base == wt_base:
                continue
            mut_name = f"rela_{wt_base}{pos0+1}{base}"
            if results.get(mut_name, {}).get("ok"):
                ddg_mat[pos0, idx] = results[mut_name]["interface_energy"] - wt_e
        print(f"  {pos0+1:2d}  {wt_base}  " +
              "  ".join(f"{ddg_mat[pos0, i]:+7.3f}" for i in range(4)))

    # Boltzmann PPM sweep and save
    ppm_dict = {}
    for tau in [1.0, 1.5, 2.5, 4.0]:
        p = np.exp(-ddg_mat / tau)
        p = p / p.sum(axis=1, keepdims=True)
        ppm_dict[str(tau)] = p.tolist()

    np.savez_compressed(
        os.path.join(args.out_dir, "ddg_matrix.npz"),
        ddg_mat=ddg_mat, fwd_seq=np.array(list(fwd_seq)),
    )
    with open(os.path.join(args.out_dir, "ppm_tau_sweep.json"), "w") as f:
        json.dump(ppm_dict, f)

    print(f"\nΔΔG matrix + PPM saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
