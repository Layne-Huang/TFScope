"""PBX1 AF3+Rosetta calibration case study — step 2: build local-AF3 input JSONs.

Follows the exact `run_alphafold3()` pipeline used elsewhere in this project
(scripts/run_af3_zinc_wt.sbatch, scripts/build_af3_inputs.py): dialect
"alphafold3", protein id A, dsDNA ids B (fwd) / C (revcomp), neutral-GC-padded
to >=12bp. This dialect is for the LOCAL af3_mmseqs2 `run_alphafold3()` call on
the Harvard FASRC cluster (gpu_h200, /n/holylabs storage) -- NOT the
"alphafoldserver" web-submission dialect used in case_study/pdb/mutation/.

Job set (refold-per-candidate protocol -- see memory pwm-rosetta-relax-default /
tfscope-vs-deeppbs-findings: in-place SNP scanning on a fixed backbone is
clash-dominated and unreliable; refolding each candidate base as its own AF3
complex then comparing Rosetta interface energy is the validated protocol):

  1. pbx1_wt_predicted   - WT complex seeded with TFScope's OWN full consensus
  2. pbx1_pos{P}_{B}      - one fold per candidate base B in {A,C,G,T} at each
                            position P where TFScope's core disagrees with the
                            known HOCOMOCO PBX1 motif (holding all other
                            positions at TFScope's predicted base)

Reads results/pbx1_case_study/seed_pwm.json (from predict_pbx1_seed.py).
Writes results/pbx1_case_study/af3_inputs/*.json + manifest.json.
"""
import json, os

IN = "results/pbx1_case_study/seed_pwm.json"
OUT_DIR = "results/pbx1_case_study/af3_inputs"
MIN_DNA = 12
FLANK_UNIT = "GC"
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def revcomp(s):
    return "".join(COMP[b] for b in reversed(s))


def pad_dna(core):
    if len(core) >= MIN_DNA:
        return core, 0
    need = MIN_DNA - len(core)
    left_n = need // 2
    right_n = need - left_n
    left = (FLANK_UNIT * ((left_n // 2) + 1))[:left_n]
    right = (FLANK_UNIT * ((right_n // 2) + 1))[:right_n]
    return left + core + right, left_n


def af3_json(name, protein_seq, dna_fwd):
    dna_rev = revcomp(dna_fwd)
    return {
        "name": name,
        "sequences": [
            {"protein": {"id": "A", "sequence": protein_seq}},
            {"dna": {"id": "B", "sequence": dna_fwd}},
            {"dna": {"id": "C", "sequence": dna_rev}},
        ],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rec = json.load(open(IN))
    dbd = rec["dbd"]
    full_consensus = rec["full_consensus"]
    core_lo_local = 0  # gated core starts at column 0 of the full canvas -> starts at index 0 of full_consensus too
    core_hi_local = rec["core_hi"] - rec["core_lo"]  # inclusive local end index within full_consensus core
    core = full_consensus[core_lo_local:core_hi_local + 1]
    gt_consensus = rec["gt_consensus"]

    # Best-align TFScope core vs GT (shift=0, no RC, per seed_pwm.json align_shift/align_rc)
    # -> mismatched local core positions become refold-scan targets.
    shift, rc = rec["align_shift"], rec["align_rc"]
    assert not rc and shift == 0, "re-derive alignment logic below if this changes"
    n = min(len(core), len(gt_consensus))
    mismatches = [i for i in range(n) if core[i] != gt_consensus[i]]
    print(f"DBD={len(dbd)}aa  full_consensus={full_consensus} (core={core})")
    print(f"GT consensus={gt_consensus}")
    print(f"mismatched core positions (0-indexed within core): {mismatches} "
          f"-> {[core[i] for i in mismatches]} vs GT {[gt_consensus[i] for i in mismatches]}")

    manifest = []

    # 1. baseline WT fold, TFScope's own full consensus
    fwd, offset = pad_dna(full_consensus)
    j = af3_json("pbx1_wt_predicted", dbd, fwd)
    json.dump(j, open(f"{OUT_DIR}/pbx1_wt_predicted.json", "w"), indent=2)
    manifest.append(dict(af3_name="pbx1_wt_predicted", role="baseline",
                          full_consensus=full_consensus, fwd_dna=fwd, core_offset=offset))

    # 2. refold-per-candidate scan at each mismatched core position
    for local_pos in mismatches:
        global_pos = core_lo_local + local_pos  # index within full_consensus
        for base in "ACGT":
            variant = full_consensus[:global_pos] + base + full_consensus[global_pos + 1:]
            fwd, offset = pad_dna(variant)
            name = f"pbx1_pos{local_pos}_{base}"
            j = af3_json(name, dbd, fwd)
            json.dump(j, open(f"{OUT_DIR}/{name}.json", "w"), indent=2)
            manifest.append(dict(af3_name=name, role="scan", scan_local_pos=local_pos,
                                  scan_base=base, is_tfscope_pred=(base == core[local_pos]),
                                  is_gt_base=(local_pos < len(gt_consensus) and base == gt_consensus[local_pos]),
                                  variant_consensus=variant, fwd_dna=fwd, core_offset=offset))

    json.dump(manifest, open(f"{OUT_DIR}/manifest.json", "w"), indent=2)
    print(f"\nWrote {len(manifest)} AF3 input JSONs to {OUT_DIR}/")
    for m in manifest:
        print(f"  {m['af3_name']:22s} fwd_dna={m['fwd_dna']}")


if __name__ == "__main__":
    main()
