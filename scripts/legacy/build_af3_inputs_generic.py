"""AF3+Rosetta calibration case study — step 2 (generic, RC-aware): build local-AF3
input JSONs for any gene processed by predict_seed_generic.py.

Handles the fact that TFScope's raw prediction may best align to the true motif
only after reverse-complementing (align_rc=True) and/or shifting (align_shift!=0)
-- both are physically meaningless for the actual dsDNA fold (a PWM and its RC
describe the same duplex), so we re-orient the predicted window into the GT-
comparable frame before identifying which positions TFScope got wrong and
building the refold-per-candidate scan.

Usage: python scripts/build_af3_inputs_generic.py --gene E2F4
Writes results/calibration_case_study/<gene>/af3_inputs/*.json + manifest.json
"""
import json, os, argparse
import numpy as np

MIN_DNA = 12
FLANK_UNIT = "GC"
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
BASES = "ACGT"


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


def af3_json(name, protein_seq, dna_fwd, zn_count=0):
    dna_rev = revcomp(dna_fwd)
    sequences = [
        {"protein": {"id": "A", "sequence": protein_seq}},
        {"dna": {"id": "B", "sequence": dna_fwd}},
        {"dna": {"id": "C", "sequence": dna_rev}},
    ]
    if zn_count:
        # structural Zn2+ for zinc-finger DBDs -- rename_chains() appends metal
        # residues onto chain A automatically; without this the fold is unphysical
        # (see memory: TFScope ZF complexes needed Zn2+ to fix unphysical energies).
        ids = [chr(ord("D") + i) for i in range(zn_count)]
        sequences.append({"ligand": {"id": ids, "ccdCodes": ["ZN"]}})
    return {
        "name": name,
        "sequences": sequences,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zn-count", type=int, default=0,
                     help="Number of structural Zn2+ ions to add (zinc-finger DBDs)")
    ap.add_argument("--gene", required=True)
    args = ap.parse_args()
    gene = args.gene
    ROOT = f"results/calibration_case_study/{gene}"
    OUT_DIR = f"{ROOT}/af3_inputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    rec = json.load(open(f"{ROOT}/seed_pwm.json"))
    dbd = rec["dbd"]
    pwm = np.array(rec["pwm"])  # (4, W) forward/model-native orientation
    lo, hi, flank = rec["core_lo"], rec["core_hi"], rec["flank"]
    lo_f, hi_f = max(0, lo - flank), min(pwm.shape[1] - 1, hi + flank)
    window_fwd = pwm[:, lo_f:hi_f + 1]
    core_lo_local_fwd, core_hi_local_fwd = lo - lo_f, hi - lo_f

    rc, shift = rec["align_rc"], rec["align_shift"]
    if rc:
        window = window_fwd[::-1, ::-1]
        W = window_fwd.shape[1]
        core_lo_local = W - 1 - core_hi_local_fwd
        core_hi_local = W - 1 - core_lo_local_fwd
    else:
        window = window_fwd
        core_lo_local, core_hi_local = core_lo_local_fwd, core_hi_local_fwd

    full_consensus = "".join(BASES[window[:, j].argmax()] for j in range(window.shape[1]))
    core = full_consensus[core_lo_local:core_hi_local + 1]
    gt_consensus = rec["gt_consensus"]

    # window index i -> gt index j = (i - core_lo_local) - shift  (see script docstring derivation)
    mismatches = []
    for i in range(window.shape[1]):
        j = (i - core_lo_local) - shift
        if 0 <= j < len(gt_consensus) and full_consensus[i] != gt_consensus[j]:
            mismatches.append(i)

    print(f"{gene}: DBD={len(dbd)}aa  window(aligned)={full_consensus} (core={core}, rc={rc}, shift={shift})")
    print(f"GT consensus={gt_consensus}")
    print(f"mismatched window positions (0-indexed): {mismatches} -> "
          f"{[full_consensus[i] for i in mismatches]}")

    manifest = []
    fwd, offset = pad_dna(full_consensus)
    j = af3_json(f"{gene}_wt_predicted", dbd, fwd, zn_count=args.zn_count)
    json.dump(j, open(f"{OUT_DIR}/{gene}_wt_predicted.json", "w"), indent=2)
    manifest.append(dict(af3_name=f"{gene}_wt_predicted", role="baseline",
                          full_consensus=full_consensus, fwd_dna=fwd, core_offset=offset))

    n_skipped = 0
    for pos in mismatches:
        for base in BASES:
            is_tfscope_pred = (base == full_consensus[pos])
            gt_j = (pos - core_lo_local) - shift
            is_gt_base = (base == gt_consensus[gt_j]) if 0 <= gt_j < len(gt_consensus) else False
            variant = full_consensus[:pos] + base + full_consensus[pos + 1:]
            name = f"{gene}_pos{pos}_{base}"
            if is_tfscope_pred:
                # identical sequence to the WT fold -- skip the redundant AF3+Rosetta run,
                # the scoring script reuses the WT structure's own interface energy instead.
                n_skipped += 1
                manifest.append(dict(af3_name=name, role="scan", scan_local_pos=pos, scan_base=base,
                                      is_tfscope_pred=True, is_gt_base=is_gt_base,
                                      variant_consensus=variant, reuse_wt_energy=True))
                continue
            fwd, offset = pad_dna(variant)
            j = af3_json(name, dbd, fwd, zn_count=args.zn_count)
            json.dump(j, open(f"{OUT_DIR}/{name}.json", "w"), indent=2)
            manifest.append(dict(af3_name=name, role="scan", scan_local_pos=pos, scan_base=base,
                                  is_tfscope_pred=False, is_gt_base=is_gt_base,
                                  variant_consensus=variant, fwd_dna=fwd, core_offset=offset))
    if n_skipped:
        print(f"\nSkipped {n_skipped} redundant fold(s) (base == TFScope's own prediction == WT sequence);"
              f" will reuse the WT structure's interface energy for those instead.")

    json.dump(manifest, open(f"{OUT_DIR}/manifest.json", "w"), indent=2)
    n_jobs = sum(1 for m in manifest if not m.get("reuse_wt_energy"))
    print(f"\nWrote {n_jobs} AF3 input JSONs to {OUT_DIR}/ ({len(manifest)} manifest entries total)")
    for m in manifest:
        if m.get("reuse_wt_energy"):
            print(f"  {m['af3_name']:24s} (reuses WT energy, no fold)")
        else:
            print(f"  {m['af3_name']:24s} fwd_dna={m['fwd_dna']}")


if __name__ == "__main__":
    main()
