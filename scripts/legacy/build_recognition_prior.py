#!/usr/bin/env python
"""Build family-canonical DNA-recognition-residue priors (v18b, Source A).

For every TF we mark the sequence positions of its *recognition region* — the
structural element that makes base-specific DNA contacts in that DBD family.
These are derived from established structural biology (rule-based), NOT from the
test-set crystal structures, so there is no benchmark leakage.

Output: data/contact_maps/recognition_residues.json
        { "<filename_without_.txt>": [seq_idx, seq_idx, ...], ... }

The v18 contact branch attention is supervised to place mass on these residues
(see TFScopeLoss._v18_attn_terms), forcing genuine residue->base reading rather
than the hub/sink collapse. We deliberately mark the whole recognition helix /
basic region (a few residues wider than the 3-4 exact base contacts) so the
prior reliably *covers* the true contacts; the row-diversity and hub penalties
then spread attention within that region.

Validate coverage afterwards with scripts/extract_pdb_contacts.py (test set).
"""
import json, os, re
import pandas as pd

OUT = "data/contact_maps/recognition_residues.json"
PARQUET = "data/processed/tf_pwm_deeppbs_only.parquet"

# C2H2 zinc finger: ...C-x(2-4)-C-x(3)-[FY]-x(5)-[hydrophobic]-x(2)-H-x(3-5)-H...
# The alpha-helix (recognition helix) runs from ~2 after the 2nd Cys to the 2nd His.
C2H2 = re.compile(r"C.{2,4}C.{8,18}H.{3,6}H")


def recognition_indices(seq, dbd_start, dbd_end, family_name):
    """Return sorted list of 0-based *sequence* indices in the recognition region."""
    dbd = seq[dbd_start:dbd_end]
    n = len(dbd)
    if n < 8:
        return []
    fam = family_name or ""
    rel = set()

    if "C2H2" in fam:
        for m in C2H2.finditer(dbd):
            seg = dbd[m.start():m.end()]
            cpos = [i for i, a in enumerate(seg) if a == "C"]
            hpos = [i for i, a in enumerate(seg) if a == "H"]
            if len(cpos) >= 2 and len(hpos) >= 2:
                helix_start = m.start() + cpos[1] + 2      # ~2 after 2nd Cys
                helix_end = m.start() + hpos[-1]           # up to 2nd His
                for p in range(helix_start, helix_end + 1):
                    if 0 <= p < n:
                        rel.add(p)
        if not rel:  # fallback: no clean finger match → mark middle 60%
            rel = set(range(int(0.2 * n), int(0.85 * n)))

    elif fam == "bHLH":
        # basic region (DNA-reading) is the N-terminal ~15 residues of the domain
        rel = set(range(0, min(15, n)))

    elif fam == "Homeodomain":
        # N-terminal arm (minor groove) + recognition helix-3 (major groove)
        for p in list(range(1, 6)) + list(range(41, 58)):
            if p < n:
                rel.add(p)
        if not rel:
            rel = set(range(int(0.6 * n), n))

    elif fam == "bZIP":
        # basic region precedes the leucine zipper
        rel = set(range(0, min(22, n)))

    elif fam == "Nuclear_Receptor":
        # P-box / first-ZF recognition helix
        rel = set(range(18, min(40, n)))

    elif fam == "Forkhead":
        # winged-helix recognition helix H3
        rel = set(range(30, min(62, n)))

    elif fam == "ETS":
        # recognition helix H3 contacting the GGAA core
        rel = set(range(28, min(52, n)))

    # "Other" / unmapped families → no prior (empty)
    return sorted(dbd_start + p for p in rel)


def main():
    df = pd.read_parquet(PARQUET)
    out, n_empty, by_fam = {}, 0, {}
    for _, r in df.iterrows():
        idxs = recognition_indices(r["sequence"], int(r["dbd_start"]),
                                   int(r["dbd_end"]), r["family_name"])
        key = r["filename"].replace(".txt", "")
        out[key] = idxs
        if not idxs:
            n_empty += 1
        by_fam.setdefault(r["family_name"], []).append(len(idxs))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"Wrote {len(out)} entries to {OUT}  ({n_empty} empty)")
    print(f"{'family':<18}{'n_TF':>6}{'mean_recog_residues':>22}")
    for fam, lens in sorted(by_fam.items()):
        import numpy as np
        print(f"{fam:<18}{len(lens):>6}{np.mean(lens):>22.1f}")


if __name__ == "__main__":
    main()
