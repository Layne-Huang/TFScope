#!/usr/bin/env python
"""Add a partner_sequence column to the EXISTING merged training table without
re-clustering -- so the train/val/test split is byte-identical and the
single-chain vs two-chain runs are directly comparable.

build_training_table.py names structure rows str_0..str_{N-1} in the row order
of tf_pwm_deeppbs_v2_deduped.parquet; tf_pwm_deeppbs_v2_partner.parquet is that
same file with partner columns appended (same order), so str_i <-> partner row
i. Sequence-only rows (seq_*) get partner_sequence="".

Outputs data/processed/tf_pwm_training_v2p.parquet
"""
import pandas as pd

TRAIN = "data/processed/tf_pwm_training_v2.parquet"
PART = "data/processed/tf_pwm_deeppbs_v2_partner.parquet"
OUT = "data/processed/tf_pwm_training_v2p.parquet"


def main():
    tr = pd.read_parquet(TRAIN)
    pt = pd.read_parquet(PART).reset_index(drop=True)
    part_by_stridx = {f"str_{i}": s for i, s in enumerate(pt["partner_sequence"].tolist())}
    part_gene = {f"str_{i}": g for i, g in enumerate(pt["partner_gene_used"].tolist())}
    part_chain = {
        f"str_{i}": g
        for i, g in enumerate(pt.get("partner_chain_used", pd.Series([""] * len(pt))).tolist())
    }
    part_method = {
        f"str_{i}": g
        for i, g in enumerate(pt.get("partner_crop_method", pd.Series([""] * len(pt))).tolist())
    }

    tr["partner_sequence"] = tr["filename"].map(part_by_stridx).fillna("")
    tr["partner_gene_used"] = tr["filename"].map(part_gene).fillna("")
    tr["partner_chain_used"] = tr["filename"].map(part_chain).fillna("")
    tr["partner_crop_method"] = tr["filename"].map(part_method).fillna("")

    has = tr["partner_sequence"].str.len() > 0
    repeat_family = tr.get(
        "family_name", pd.Series(["Other"] * len(tr))
    ).isin({"bHLH", "bZIP", "Nuclear_Receptor"})
    p53 = tr["gene_symbol"].astype(str).str.upper().str.match(
        r"^(TP53|TP63|TP73|P53)$"
    )
    tr["multichain_eligible"] = has & (repeat_family | p53)
    print(f"training table {len(tr)} rows | partner_sequence set on {has.sum()} rows "
          f"({tr.loc[has, 'gene_symbol'].str.upper().nunique()} genes)")
    # sanity: only structure rows get partners
    assert (tr.loc[has, "filename"].str.startswith("str_")).all(), "seq row got a partner!"
    tr.to_parquet(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
