#!/usr/bin/env python
"""Put back the columns the unified-v25 merge dropped.

`build_unified_split_v25.py` concatenated the v25 and DeepPBS tables on the
INTERSECTION of their columns, which silently discarded fields the v24 recipe
needs:

  partner_sequence / partner_seqs / multichain_eligible / n_chains / is_dimer
        --two-chain-input feeds a heterodimer partner DBD as a second chain
  group_id
        --group-balanced-sampling samples group_id (gene+sequence+motif source)
        uniformly; without it the sampler silently falls back
  gene_key / motif_source / family_source / _set
        carried along for downstream analysis

This restores them for the v25 rows and fills sensible single-chain defaults for
the DeepPBS rows, which never had a partner. It does NOT touch the row set, the
split assignment, the sequences or the PWMs -- the frozen dataset is unchanged
(see `canonical-dataset-v25-frozen`); only columns that were lost in transit come
back.

  python scripts/restore_unified_v25_columns.py
"""
from __future__ import annotations

import pandas as pd

UNI = "data/processed/tf_pwm_unified_v25.parquet"
V25 = "data/processed/tf_pwm_training_v25.parquet"

# column -> value for the DeepPBS rows, which are single-chain and ungrouped
DEFAULTS = {
    "partner_sequence": None, "partner_seqs": None, "partner_gene": None,
    "multichain_eligible": False, "n_chains": 1, "is_dimer": False,
    "motif_source": "deeppbs", "family_source": None, "_set": "deeppbs",
}


def main():
    uni = pd.read_parquet(UNI); uni["filename"] = uni.filename.astype(str)
    v25 = pd.read_parquet(V25); v25["filename"] = v25.filename.astype(str)
    missing = [c for c in v25.columns if c not in uni.columns]
    print(f"restoring {len(missing)} columns: {missing}")

    n0, cols0 = len(uni), set(uni.columns)
    src = v25.set_index("filename")
    for c in missing:
        uni[c] = uni.filename.map(src[c])

    for c, default in DEFAULTS.items():
        if c in uni.columns and default is not None:
            uni[c] = uni[c].where(uni.source_table == "v25", default)

    # group_id must stay unique per group; the DeepPBS rows had none, so give each
    # its own group rather than lumping them into one that the sampler would then
    # draw as a single unit
    if "group_id" in uni.columns:
        m = uni.group_id.isna()
        uni.loc[m, "group_id"] = "deeppbs__" + uni.loc[m, "filename"].astype(str)
        print(f"  group_id: assigned {int(m.sum())} singleton groups to DeepPBS rows")

    assert len(uni) == n0, "row count changed"
    assert cols0 <= set(uni.columns), "an original column disappeared"
    assert uni.filename.is_unique, "duplicate filenames"
    for c in ("split", "sequence", "pwm", "is_deeppbs_bench"):
        assert uni[c].notna().all(), f"{c} has nulls"

    two = uni.partner_sequence.notna().sum() if "partner_sequence" in uni else 0
    print(f"\nrows {len(uni)} | columns {len(cols0)} -> {len(uni.columns)}")
    print(f"  with a partner_sequence : {two}")
    print(f"  distinct group_id       : {uni.group_id.nunique()}")
    print(uni.groupby('split').size().to_string())
    uni.to_parquet(UNI, index=False)
    print(f"\nsaved {UNI}")


if __name__ == "__main__":
    main()
