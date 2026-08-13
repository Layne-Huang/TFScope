#!/usr/bin/env python3
"""PBX1 AF3+Rosetta calibration case study — step 4: score the refold-per-candidate
scan with PyRosetta and build the calibrated PWM.

Mirrors scripts/run_pyrosetta_interface_ddg.py (the validated MyoD1 L112R
refold-per-candidate protocol) but upgrades MinMover -> full FastRelax, per
memory pwm-rosetta-relax-default ("always run with -relax"; raw/unrelaxed
poses give clash-inflated, noisy ddG).

For each of the 9 independently AF3-folded PBX1 complexes (1 baseline +
4 bases x 2 mismatched core positions), computes the ABSOLUTE relaxed
protein-DNA interface energy (default ddg_filter, jump=1 -- correct for
this monomer complex, do NOT set PWM_INTERFACE_MODE=prot_dna). Builds a
Boltzmann-weighted PWM column per scanned position from the 4 candidate
energies and compares it to (a) TFScope's raw seed prediction and (b) the
true HOCOMOCO PBX1 motif.

Run in the `multiflow` conda env (has PyRosetta + gemmi):
    PYTHONPATH=pwm_rosetta CUDA_VISIBLE_DEVICES="" \
    python scripts/score_pbx1_refold_scan.py

Usage:
    python scripts/score_pbx1_refold_scan.py \
        --af3-out case_study/pbx1/af3_output \
        --out-dir results/pbx1_case_study/rosetta_scan \
        --manifest results/pbx1_case_study/af3_inputs/manifest.json \
        --seed-pwm results/pbx1_case_study/seed_pwm.json
"""
import argparse, glob, json, os, sys

sys.path.insert(0, "pwm_rosetta")
import numpy as np

BASES = "ACGT"


def compute_interface_energy(cif_path, work_dir):
    """cif -> pdb -> rename_chains -> FastRelax -> calculate_interface_ddg (relaxed)."""
    from pyrosetta import pose_from_pdb
    from pwm_hybrid.rosetta.structure import convert_cif_to_pdb, rename_chains
    from pwm_hybrid.rosetta.scoring import calculate_interface_ddg
    from pwm_hybrid.rosetta import init as _rosetta_init

    os.makedirs(work_dir, exist_ok=True)
    pdb_path = os.path.join(work_dir, "tmp.pdb")
    renamed_path = os.path.join(work_dir, "renamed.pdb")
    convert_cif_to_pdb(cif_path, pdb_path)
    metal_free = rename_chains(pdb_path, renamed_path)

    pose = pose_from_pdb(renamed_path)
    _rosetta_init.fast_relax.apply(pose)  # full FastRelax, not just MinMover
    ddg = calculate_interface_ddg(pose)
    return ddg, metal_free


def find_cif(af3_out, name):
    pats = [
        os.path.join(af3_out, name, f"{name}_model.cif"),
        os.path.join(af3_out, name, f"{name}_model_0.cif"),
    ]
    for p in pats:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(af3_out, "**", f"{name}*model*.cif"), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af3-out", default="case_study/pbx1/af3_output")
    ap.add_argument("--out-dir", default="results/pbx1_case_study/rosetta_scan")
    ap.add_argument("--manifest", default="results/pbx1_case_study/af3_inputs/manifest.json")
    ap.add_argument("--seed-pwm", default="results/pbx1_case_study/seed_pwm.json")
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "interface_energies_raw.json")
    manifest = json.load(open(args.manifest))
    seed = json.load(open(args.seed_pwm))

    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    pyrosetta.init(get_pyrosetta_init_flags())
    init_pyrosetta(psipred_exe="")

    results = {}
    if args.skip_done and os.path.exists(raw_path):
        results = json.load(open(raw_path))
        print(f"Loaded {len(results)} cached results")

    for m in manifest:
        name = m["af3_name"]
        if name in results and results[name].get("ok"):
            print(f"[CACHED] {name}: {results[name]['interface_energy']:.3f}")
            continue
        cif = find_cif(args.af3_out, name)
        if cif is None:
            print(f"[SKIP] {name}: no CIF found under {args.af3_out}")
            results[name] = {"ok": False, "error": "cif not found"}
            continue
        print(f"\n=== {name} ===\n  CIF: {cif}")
        try:
            e, metal_free = compute_interface_energy(cif, os.path.join(args.out_dir, name))
            print(f"  relaxed interface energy = {e:.3f} REU")
            results[name] = dict(interface_energy=e, metal_free=metal_free, ok=True, **m)
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            results[name] = {"ok": False, "error": str(exc), **m}
        json.dump(results, open(raw_path, "w"), indent=2)

    print(f"\nRaw interface energies -> {raw_path}")

    # ── build calibrated PWM columns from the refold-per-candidate scan ──
    scan_positions = sorted(set(m["scan_local_pos"] for m in manifest if m.get("role") == "scan"))
    core = seed["core_consensus"]
    gt = seed["gt_consensus"]
    calibrated = {}
    print("\nRefold-per-candidate calibration (lower REU = tighter binding):")
    for pos in scan_positions:
        e = {}
        for base in BASES:
            nm = f"pbx1_pos{pos}_{base}"
            if results.get(nm, {}).get("ok"):
                e[base] = results[nm]["interface_energy"]
        if len(e) < 2:
            print(f"  pos {pos}: insufficient folds ({len(e)}/4), skipping")
            continue
        bases_present = list(e.keys())
        energies = np.array([e[b] for b in bases_present])
        p = np.exp(-energies / args.tau)
        p = p / p.sum()
        calibrated[pos] = dict(zip(bases_present, p.tolist()))
        best = bases_present[int(np.argmin(energies))]
        tf_base = core[pos]
        gt_base = gt[pos] if pos < len(gt) else "?"
        print(f"  pos {pos} (TFScope predicted {tf_base}, true HOCOMOCO {gt_base}):")
        for b in BASES:
            if b in e:
                mark = " <-Rosetta best" if b == best else (" <-TFScope" if b == tf_base else "")
                mark += " <-GT" if b == gt_base else ""
                print(f"    {b}: E={e[b]:+8.3f} REU  P={calibrated[pos][b]:.3f}{mark}")

    out = dict(tau=args.tau, scan_positions=scan_positions, calibrated_pwm_columns=calibrated,
               tfscope_core=core, gt_consensus=gt)
    json.dump(out, open(os.path.join(args.out_dir, "calibrated_pwm.json"), "w"), indent=2)
    print(f"\nCalibrated PWM columns -> {args.out_dir}/calibrated_pwm.json")


if __name__ == "__main__":
    main()
