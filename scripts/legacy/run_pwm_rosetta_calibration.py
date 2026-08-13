#!/usr/bin/env python
"""Stage 4: PyRosetta SNP-scan calibration of AF3-predicted protein-DNA complexes.

For each AF3 output complex (CIF), run pwm_rosetta's hybrid pipeline:
  AF3 wild-type structure → PyRosetta mutate every base × 3 → interface ΔΔG
  → Boltzmann PPM → calibrated PWM.

Then align the calibrated PWM back to the v7 core-consensus positions (using
core_offset from the AF3 manifest) and save for evaluation against the true PWM.

Usage:
  python scripts/run_pwm_rosetta_calibration.py \
      --af3-out-dir /n/holylabs/.../af3_runs/pilot_out \
      --manifest results/af3_pipeline/af3_inputs/manifest.json \
      --out-dir results/af3_pipeline/calibrated
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, "pwm_rosetta")


def find_af3_cif(af3_out_dir, af3_name):
    """Locate the predicted complex CIF, supporting both AF3 and Boltz path layouts.

    AF3:    {out_dir}/{name_lower}/{name_lower}_model.cif
    Boltz:  {out_dir}/{name}/{name}_model_0.cif        (case preserved, suffix _0)
    """
    cands = []
    for pat in [
        os.path.join(af3_out_dir, af3_name, f"{af3_name}_model_0.cif"),   # Boltz exact
        os.path.join(af3_out_dir, af3_name, "*model*.cif"),                # Boltz / case-preserved
        os.path.join(af3_out_dir, af3_name.lower(), "*model*.cif"),        # AF3 / lowercased
        os.path.join(af3_out_dir, "**", f"{af3_name}*model*.cif"),         # Boltz recursive
        os.path.join(af3_out_dir, "**", f"*{af3_name.lower()}*model*.cif"),# AF3 recursive
    ]:
        cands += glob.glob(pat, recursive=True)
    return sorted(set(cands))[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af3-out-dir", required=True)
    ap.add_argument("--manifest", default="results/af3_pipeline/af3_inputs/manifest.json")
    ap.add_argument("--out-dir",  default="results/af3_pipeline/calibrated")
    ap.add_argument("--tau", type=float, default=1.5, help="Boltzmann temperature for PPM")
    ap.add_argument("--only", nargs="*", default=None, help="Only these af3_names")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import pyrosetta
    from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
    from pwm_hybrid.pipeline import generate_pwm_hybrid
    from pwm_hybrid import pwm_from_hybrid_csv

    pyrosetta.init(get_pyrosetta_init_flags())
    init_pyrosetta(psipred_exe="")   # disable psipred filter (not needed for ddG)

    manifest = json.load(open(args.manifest))
    if args.only:
        manifest = [m for m in manifest if m["af3_name"] in set(args.only)]

    results = []
    for m in manifest:
        af3_name = m["af3_name"]
        cif = find_af3_cif(args.af3_out_dir, af3_name)
        if cif is None:
            print(f"[SKIP] {af3_name}: no AF3 CIF found under {args.af3_out_dir}")
            continue
        print(f"\n=== {af3_name} ===\n  AF3 CIF: {cif}")

        work = os.path.join(args.out_dir, af3_name)
        os.makedirs(work, exist_ok=True)
        try:
            generate_pwm_hybrid(
                protein_seq=None, dna_seq=None,   # extracted from the PDB
                output_dir=work, wt_pdb=cif,
                minimize_local=True, use_relax=False,
            )
            csv = os.path.join(work, "pwm_results_hybrid.csv")
            seq, energies, PPM, PWM = pwm_from_hybrid_csv(csv, tau=args.tau)
            # PPM is (L_dna, 4) over the AF3 DNA (incl. flanks); core starts at core_offset
            ppm = np.asarray(PPM)                          # (L_dna, 4) ACGT
            off = int(m["core_offset"]); clen = len(m["core"])
            core_ppm = ppm[off: off + clen]                # (core_len, 4)
            np.savez_compressed(os.path.join(work, "calibrated_pwm.npz"),
                                full_ppm=ppm, core_ppm=core_ppm,
                                dna_seq=seq, core_offset=off, core_len=clen)
            results.append({"af3_name": af3_name, "filename": m["filename"],
                            "csv": csv, "core_len": clen, "ok": True})
            print(f"  calibrated core PPM shape: {core_ppm.shape}")
        except Exception as e:
            print(f"  [FAIL] {af3_name}: {e}")
            results.append({"af3_name": af3_name, "filename": m["filename"],
                            "ok": False, "error": str(e)})

    with open(os.path.join(args.out_dir, "calibration_manifest.json"), "w") as f:
        json.dump(results, f, indent=2)
    ok = sum(r["ok"] for r in results)
    print(f"\nCalibrated {ok}/{len(results)} complexes → {args.out_dir}/")


if __name__ == "__main__":
    main()
