#!/usr/bin/env python
"""Write the reviewed DBD-crop repairs into a new parquet.

Consumes both repair passes and emits `tf_pwm_training_v25.parquet`: the v23 table with
153 of its 159 truncated protein crops replaced by the correct domain span.

  pass 1  results/data_audit/short_dbd_repair_proposals.csv   140 rows
  pass 2  results/data_audit/short_dbd_repair_pass2.csv        13 rows

The rows the passes could not resolve are LEFT AS THEY WERE, not dropped: five where the
crop matches no UniProt sequence above 60% identity (POU6F1 x3, RXRA x2 -- a different
sequence, not a mutant) and one Drosophila gene absent from the local snapshot. They stay
flagged so a later pass can pick them up.

Every repaired row keeps an audit trail: `repair_status`, `repair_evidence`,
`repair_substitutions` (mutant residues written back over the UniProt span, e.g. C269S)
and the original crop in `repair_old_seq`. `dbd_start`/`dbd_end`/`seq_length` are kept
consistent with the new sequence -- every row in this table stores the crop itself, so
dbd_start stays 0 and dbd_end becomes the new length.

  python scripts/apply_dbd_repairs.py
"""
from __future__ import annotations

import os

import pandas as pd

SRC = "data/processed/tf_pwm_training_v23.parquet"
DST = "data/processed/tf_pwm_training_v25.parquet"
P1 = "results/data_audit/short_dbd_repair_proposals.csv"
P2 = "results/data_audit/short_dbd_repair_pass2.csv"
REPORT = "results/data_audit/dbd_repair_applied.csv"


def main():
    df = pd.read_parquet(SRC)
    df["filename"] = df.filename.astype(str)
    n0 = len(df)

    fixes = {}
    for path, tag in [(P1, "pass1"), (P2, "pass2")]:
        p = pd.read_csv(path)
        ok = p[(p.status == "proposed") & p.new_seq.notna()]
        for r in ok.itertuples():
            # pass 2 supersedes pass 1 for the same row: it only ever revisits rows
            # pass 1 declined, but be explicit rather than relying on that
            fixes[str(r.filename)] = {
                "new_seq": r.new_seq, "pass": tag,
                "evidence": getattr(r, "evidence", None),
                "subs": getattr(r, "substitutions", None),
                "acc": getattr(r, "accession", None),
                "new_start": getattr(r, "new_start", None),
                "new_end": getattr(r, "new_end", None),
            }
        print(f"{tag}: {len(ok)} proposed rows")
    print(f"unique rows to repair: {len(fixes)}")

    df["repair_status"] = "unchanged"
    df["repair_old_seq"] = pd.NA
    df["repair_evidence"] = pd.NA
    df["repair_substitutions"] = pd.NA
    df["repair_accession"] = pd.NA
    df["repair_span"] = pd.NA

    idx = df.filename.isin(fixes)
    print(f"matched in the parquet: {int(idx.sum())} / {len(fixes)}")
    missing = set(fixes) - set(df.filename)
    if missing:
        print(f"  WARNING {len(missing)} proposal filenames absent from the table: "
              f"{sorted(missing)[:5]}")

    audit = []
    for i in df.index[idx]:
        fn = df.at[i, "filename"]
        f = fixes[fn]
        old = df.at[i, "sequence"]
        new = str(f["new_seq"])
        audit.append({"filename": fn, "gene": df.at[i, "gene_symbol"],
                      "old_len": len(old), "new_len": len(new),
                      "pass": f["pass"], "evidence": f["evidence"],
                      "substitutions": f["subs"], "accession": f["acc"]})
        df.at[i, "repair_old_seq"] = old
        df.at[i, "sequence"] = new
        df.at[i, "seq_length"] = len(new)
        df.at[i, "dbd_start"] = 0
        df.at[i, "dbd_end"] = len(new)
        df.at[i, "repair_status"] = f"repaired ({f['pass']})"
        df.at[i, "repair_evidence"] = f["evidence"]
        df.at[i, "repair_substitutions"] = f["subs"]
        df.at[i, "repair_accession"] = f["acc"]
        if pd.notna(f["new_start"]) and pd.notna(f["new_end"]):
            df.at[i, "repair_span"] = f"{int(f['new_start'])}-{int(f['new_end'])}"

    # the rows that stayed short and were not repaired
    L = df.sequence.str.len()
    gmax = df.gene_symbol.astype(str).str.upper().map(
        df.assign(_g=df.gene_symbol.astype(str).str.upper(),
                  _L=L).groupby("_g")._L.max())
    still = (L < 20) | (L / gmax < 0.40)
    df.loc[still & (df.repair_status == "unchanged"), "repair_status"] = \
        "short, unrepaired"
    n_short = int((df.repair_status == "short, unrepaired").sum())

    assert len(df) == n0, "row count changed"
    assert (df.seq_length == df.sequence.str.len()).all(), "seq_length out of sync"
    assert (df.dbd_end == df.sequence.str.len()).all(), "dbd_end out of sync"
    assert df.sequence.str.len().min() > 0, "empty sequence produced"

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    df.to_parquet(DST, index=False)
    pd.DataFrame(audit).to_csv(REPORT, index=False)

    print(f"\nrepaired      {int(idx.sum())} rows")
    print(f"still short   {n_short} rows (left as-is, flagged 'short, unrepaired')")
    print(f"crop length   median {pd.DataFrame(audit).old_len.median():.0f} -> "
          f"{pd.DataFrame(audit).new_len.median():.0f}")
    print(f"min length in the table: {int(L.min())} -> {int(df.sequence.str.len().min())}")
    print(f"\nsaved {DST}")
    print(f"saved {REPORT}")


if __name__ == "__main__":
    main()
