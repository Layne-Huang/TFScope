#!/usr/bin/env python3
"""AF3+Rosetta calibration case study — step 4 (generic): score the refold-per-
candidate scan with PyRosetta and build the calibrated PWM, for any gene
processed by predict_seed_generic.py + build_af3_inputs_generic.py.

Full FastRelax (not just MinMover) per memory pwm-rosetta-relax-default. Default
ddg_filter (jump=1) is correct for these monomer complexes -- do NOT set
PWM_INTERFACE_MODE=prot_dna.

Run in the `multiflow` conda env (has PyRosetta + gemmi):
    PYTHONPATH=pwm_rosetta CUDA_VISIBLE_DEVICES="" \
    python scripts/score_refold_scan_generic.py --gene E2F4
"""
import argparse, glob, json, os, sys

sys.path.insert(0, "pwm_rosetta")
import numpy as np

BASES = "ACGT"


def compute_interface_energy(cif_path, work_dir):
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
    _rosetta_init.fast_relax.apply(pose)
    ddg = calculate_interface_ddg(pose)
    return ddg, metal_free


def find_cif(af3_out, name):
    pats = [os.path.join(af3_out, name, f"{name}_model.cif"),
            os.path.join(af3_out, name, f"{name}_model_0.cif")]
    for p in pats:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(af3_out, "**", f"{name}*model*.cif"), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--skip-done", action="store_true", default=True)
    args = ap.parse_args()
    gene = args.gene

    af3_out = f"/data1/leihuang/project/TFScope/case_study/{gene.lower()}/af3_output"
    out_dir = f"results/calibration_case_study/{gene}/rosetta_scan"
    manifest_path = f"results/calibration_case_study/{gene}/af3_inputs/manifest.json"
    seed_path = f"results/calibration_case_study/{gene}/seed_pwm.json"

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "interface_energies_raw.json")
    manifest = json.load(open(manifest_path))
    seed = json.load(open(seed_path))

    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    pyrosetta.init(get_pyrosetta_init_flags())
    init_pyrosetta(psipred_exe="")

    results = {}
    if args.skip_done and os.path.exists(raw_path):
        results = json.load(open(raw_path))
        print(f"Loaded {len(results)} cached results")

    # baseline first: scan entries with reuse_wt_energy (base == TFScope's own prediction,
    # so the sequence is identical to the WT fold) reuse its energy directly, no redundant
    # fold+relax -- also avoids the several-REU run-to-run AF3/Rosetta noise seen when
    # re-folding an identical sequence independently.
    ordered = sorted(manifest, key=lambda m: m.get("role") != "baseline")
    baseline_name = next(m["af3_name"] for m in manifest if m.get("role") == "baseline")

    for m in ordered:
        name = m["af3_name"]
        if name in results and results[name].get("ok"):
            print(f"[CACHED] {name}: {results[name]['interface_energy']:.3f}")
            continue
        if m.get("reuse_wt_energy"):
            wt = results[baseline_name]
            print(f"[REUSE WT] {name}: {wt['interface_energy']:.3f} (identical seq to {baseline_name}, no fold)")
            results[name] = dict(interface_energy=wt["interface_energy"], metal_free=wt["metal_free"],
                                  ok=True, reused_from=baseline_name, **m)
            json.dump(results, open(raw_path, "w"), indent=2)
            continue
        cif = find_cif(af3_out, name)
        if cif is None:
            print(f"[SKIP] {name}: no CIF found under {af3_out}")
            results[name] = {"ok": False, "error": "cif not found"}
            continue
        print(f"\n=== {name} ===\n  CIF: {cif}")
        try:
            e, metal_free = compute_interface_energy(cif, os.path.join(out_dir, name))
            print(f"  relaxed interface energy = {e:.3f} REU")
            results[name] = dict(interface_energy=e, metal_free=metal_free, ok=True, **m)
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            results[name] = {"ok": False, "error": str(exc), **m}
        json.dump(results, open(raw_path, "w"), indent=2)

    print(f"\nRaw interface energies -> {raw_path}")

    scan_positions = sorted(set(m["scan_local_pos"] for m in manifest if m.get("role") == "scan"))
    calibrated = {}
    print("\nRefold-per-candidate calibration (lower REU = tighter binding):")
    for pos in scan_positions:
        e, tf_base, gt_base = {}, None, None
        for m in manifest:
            if m.get("role") == "scan" and m["scan_local_pos"] == pos:
                b = m["scan_base"]
                if m["is_tfscope_pred"]:
                    tf_base = b
                if m["is_gt_base"]:
                    gt_base = b
                nm = f"{gene}_pos{pos}_{b}"
                if results.get(nm, {}).get("ok"):
                    e[b] = results[nm]["interface_energy"]
        if len(e) < 2:
            print(f"  pos {pos}: insufficient folds ({len(e)}/4), skipping")
            continue
        bases_present = list(e.keys())
        energies = np.array([e[b] for b in bases_present])
        p = np.exp(-energies / args.tau)
        p = p / p.sum()
        calibrated[pos] = dict(zip(bases_present, p.tolist()))
        best = bases_present[int(np.argmin(energies))]
        print(f"  pos {pos} (TFScope predicted {tf_base}, true HOCOMOCO {gt_base}):")
        for b in BASES:
            if b in e:
                mark = " <-Rosetta best" if b == best else (" <-TFScope" if b == tf_base else "")
                mark += " <-GT" if b == gt_base else ""
                print(f"    {b}: E={e[b]:+8.3f} REU  P={calibrated[pos][b]:.3f}{mark}")

    out = dict(gene=gene, tau=args.tau, scan_positions=scan_positions,
               calibrated_pwm_columns=calibrated)
    json.dump(out, open(os.path.join(out_dir, "calibrated_pwm.json"), "w"), indent=2)
    print(f"\nCalibrated PWM columns -> {out_dir}/calibrated_pwm.json")


if __name__ == "__main__":
    main()
