#!/usr/bin/env python
"""Build a leakage-controlled nearest-neighbour index over TF embeddings."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tfscope.data.split_hygiene import (
    SPLIT_NAMES,
    normalize_text,
    sequence_hash,
    specific_source_ids,
    write_json,
)


def build_index(
    embeddings: dict[str, np.ndarray],
    df: pd.DataFrame,
    split: dict,
    donor_splits: list[str],
    k: int,
    family_restricted: bool = False,
) -> tuple[dict, dict]:
    rows = df.set_index("filename", drop=False)
    identity_source_ids = specific_source_ids(df)
    query_filenames = []
    for split_name in SPLIT_NAMES:
        query_filenames.extend(split.get(split_name, []))
    query_filenames = list(dict.fromkeys(query_filenames))

    allowed_donors = set()
    for split_name in donor_splits:
        allowed_donors.update(split.get(split_name, []))

    all_embedding_filenames = sorted(embeddings)
    donor_filenames = [
        filename
        for filename in all_embedding_filenames
        if filename in allowed_donors and filename in rows.index
    ]
    if not donor_filenames:
        raise ValueError("no eligible donors have both metadata and embeddings")

    donor_matrix = np.stack([embeddings[filename] for filename in donor_filenames])
    donor_matrix = donor_matrix.astype(np.float32)
    donor_matrix /= np.linalg.norm(donor_matrix, axis=1, keepdims=True) + 1e-8
    donor_rows = rows.loc[donor_filenames]
    donor_filename_values = np.asarray(donor_filenames, dtype=object)
    donor_genes = donor_rows["gene_symbol"].map(normalize_text).to_numpy()
    donor_uniprots = donor_rows["uniprot_id"].map(normalize_text).to_numpy()
    donor_sources = donor_rows["source_id"].map(normalize_text).to_numpy()
    donor_sequences = donor_rows["sequence"].map(sequence_hash).to_numpy()
    donor_families = donor_rows["family_name"].map(normalize_text).to_numpy()

    exclusion_counts = Counter()
    missing_query_embeddings = []
    index = {}
    embedded_query_filenames = [
        filename
        for filename in query_filenames
        if filename in embeddings and filename in rows.index
    ]
    if not embedded_query_filenames:
        raise ValueError("no queries have both metadata and embeddings")
    query_matrix = np.stack(
        [embeddings[filename] for filename in embedded_query_filenames]
    ).astype(np.float32)
    query_matrix /= np.linalg.norm(query_matrix, axis=1, keepdims=True) + 1e-8
    all_similarities = query_matrix @ donor_matrix.T
    query_position = {
        filename: position
        for position, filename in enumerate(embedded_query_filenames)
    }

    for query_filename in query_filenames:
        if query_filename not in embeddings or query_filename not in rows.index:
            index[query_filename] = []
            missing_query_embeddings.append(query_filename)
            continue

        similarities = all_similarities[query_position[query_filename]]
        query_row = rows.loc[query_filename]
        query_family = normalize_text(query_row.get("family_name"))

        valid = np.ones(len(donor_filenames), dtype=bool)

        def exclude(reason: str, mask: np.ndarray) -> None:
            valid[mask] = False
            exclusion_counts[reason] += int(mask.sum())

        exclude("same_record", donor_filename_values == query_filename)
        query_gene = normalize_text(query_row.get("gene_symbol"))
        if query_gene:
            exclude("same_gene", donor_genes == query_gene)
        query_uniprot = normalize_text(query_row.get("uniprot_id"))
        if query_uniprot:
            exclude("same_uniprot", donor_uniprots == query_uniprot)
        query_source = normalize_text(query_row.get("source_id"))
        if query_source in identity_source_ids:
            exclude("same_source_id", donor_sources == query_source)
        query_sequence = sequence_hash(query_row.get("sequence"))
        if query_sequence:
            exclude("same_sequence", donor_sequences == query_sequence)
        if family_restricted:
            exclude("different_family", donor_families != query_family)

        candidate_indices = np.flatnonzero(valid)
        if candidate_indices.size == 0:
            index[query_filename] = []
            continue
        ranked = candidate_indices[
            np.argsort(-similarities[candidate_indices], kind="stable")[:k]
        ]
        index[query_filename] = [
            {
                "nn_filename": donor_filenames[int(donor_index)],
                "cos_sim": float(similarities[int(donor_index)]),
            }
            for donor_index in ranked
        ]

    manifest = {
        "donor_splits": donor_splits,
        "n_queries": len(query_filenames),
        "n_donor_records": len(donor_filenames),
        "k": k,
        "family_restricted": family_restricted,
        "exclusion_rules": [
            "same_record",
            "same_gene",
            "same_uniprot",
            "same_specific_source_id",
            "same_sequence",
        ],
        "n_specific_source_ids": len(identity_source_ids),
        "exclusion_counts": dict(exclusion_counts),
        "missing_query_embeddings": sorted(missing_query_embeddings),
    }
    return index, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings", default="data/processed/tf_dbd_embeddings_aug.npz"
    )
    parser.add_argument(
        "--split", default="data/processed/splits/cluster40_clean/split.json"
    )
    parser.add_argument(
        "--data", default="data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
    )
    parser.add_argument(
        "--out", default="data/processed/tf_nn_index_cluster40_clean.json"
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--donor-splits",
        nargs="+",
        choices=list(SPLIT_NAMES),
        default=["train"],
        help="Split partitions allowed in the donor bank. Default: train only.",
    )
    parser.add_argument("--family-restricted", action="store_true")
    args = parser.parse_args()

    with open(args.split) as handle:
        split = json.load(handle)
    df = pd.read_parquet(args.data)
    with np.load(args.embeddings) as embedding_file:
        embeddings = {
            filename: embedding_file[filename] for filename in embedding_file.files
        }

    index, manifest = build_index(
        embeddings=embeddings,
        df=df,
        split=split,
        donor_splits=args.donor_splits,
        k=args.k,
        family_restricted=args.family_restricted,
    )

    output = Path(args.out)
    write_json(output, index)
    manifest.update(
        {
            "data": args.data,
            "split": args.split,
            "embeddings": args.embeddings,
            "index": str(output),
        }
    )
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    write_json(manifest_path, manifest)

    test_top1 = [
        index[filename][0]["cos_sim"]
        for filename in split.get("test", [])
        if index.get(filename)
    ]
    print(
        f"Wrote {output} with {manifest['n_donor_records']} train-only donors "
        f"and K={args.k}"
    )
    print(f"Wrote manifest {manifest_path}")
    if test_top1:
        print(
            "Test top-1 cosine similarity: "
            f"mean={np.mean(test_top1):.3f}, "
            f"median={np.median(test_top1):.3f}, "
            f"min={np.min(test_top1):.3f}, "
            f"max={np.max(test_top1):.3f}"
        )


if __name__ == "__main__":
    main()
