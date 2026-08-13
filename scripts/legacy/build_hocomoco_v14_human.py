#!/usr/bin/env python
"""Parse HOCOMOCO v14 CORE PFMs into a human PWM table (replaces our v13).

Motif ID format: <GENE>.H14CORE.<idx>.<evidence>.<quality>
e.g. AHR.H14CORE.0.P.B  ->  gene=AHR, quality=B
Human entries are identified via tf_masterlist.tsv.
"""
import os
import numpy as np
import pandas as pd

HDIR = "data/raw/hocomoco_v14"
PFM = os.path.join(HDIR, "H14CORE_pfms.txt")
OUT = "data/processed/hocomoco_v14_human_pwms.parquet"
MIN_LEN = 5


def parse_pfms(path):
    motifs, name, rows = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name and rows:
                    motifs.append((name, np.array(rows, dtype=np.float32)))
                name, rows = line[1:].strip(), []
            elif line.strip():
                parts = line.split()
                if len(parts) == 4:
                    rows.append([float(x) for x in parts])
    if name and rows:
        motifs.append((name, np.array(rows, dtype=np.float32)))
    return motifs


def main():
    motifs = parse_pfms(PFM)
    print(f"parsed {len(motifs)} motifs from H14CORE", flush=True)

    master = pd.read_csv(os.path.join(HDIR, "tf_masterlist.tsv"), sep="\t", low_memory=False)
    # The masterlist has NO species column -- species is encoded in the
    # curated:uniprot_id suffix (e.g. MBD2_HUMAN / MBD2_MOUSE). An earlier
    # version of this script looked for a "species" column, found none, and
    # silently disabled the filter (kept all 1595 motifs incl. mouse).
    hs = master[master["curated:uniprot_id"].astype(str).str.endswith("_HUMAN")]
    human_genes = set(hs["auto:gene_symbol"].astype(str).str.upper())
    print(f"human genes in masterlist: {len(human_genes)}", flush=True)

    rows, n_short, n_nonhuman = [], 0, 0
    for name, mat in motifs:
        gene = name.split(".")[0]
        # HOCOMOCO human motif ids are uppercase gene symbols; mouse are Titlecase
        if human_genes is not None:
            if gene.upper() not in human_genes:
                n_nonhuman += 1
                continue
        elif not gene.isupper():
            n_nonhuman += 1
            continue
        if mat.shape[0] < MIN_LEN:
            n_short += 1
            continue
        pwm = mat.T.astype(np.float32)            # (4, L) ACGT
        pwm = pwm / pwm.sum(axis=0, keepdims=True)
        quality = name.split(".")[-1] if "." in name else ""
        rows.append({
            "gene_symbol": gene,
            "motif_id": name,
            "quality_grade": quality,
            "motif_length": pwm.shape[1],
            "pwm": pwm.tobytes(),
            "source": "HOCOMOCO_v14",
        })

    out = pd.DataFrame(rows)
    print(f"\nkept {len(out)} human motifs (skipped non-human={n_nonhuman}, too short={n_short})", flush=True)
    print(f"distinct genes: {out['gene_symbol'].str.upper().nunique()}", flush=True)
    print(out["quality_grade"].value_counts().to_dict(), flush=True)
    out.to_parquet(OUT)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
