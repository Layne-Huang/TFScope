"""Utilities for leakage-controlled dataset splits and retrieval banks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SPLIT_NAMES = ("train", "val", "test")
MAX_GENES_PER_SPECIFIC_SOURCE = 2


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def sequence_hash(sequence) -> str:
    sequence = normalize_text(sequence).replace(" ", "")
    if not sequence:
        return ""
    return hashlib.sha1(sequence.encode("ascii", errors="ignore")).hexdigest()


def specific_source_ids(
    df: pd.DataFrame,
    max_genes: int = MAX_GENES_PER_SPECIFIC_SOURCE,
) -> set[str]:
    """Return source IDs that behave like motif identities, not method labels."""
    work = df[["source_id", "gene_symbol"]].copy()
    work["_source_id"] = work["source_id"].map(normalize_text)
    work["_gene"] = work["gene_symbol"].map(normalize_text)
    work = work[(work["_source_id"] != "") & (work["_gene"] != "")]
    gene_counts = work.groupby("_source_id")["_gene"].nunique()
    return set(gene_counts[gene_counts <= max_genes].index)


class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_group_ids(
    df: pd.DataFrame,
    cluster_by_uniprot: dict[str, int] | None = None,
) -> pd.Series:
    """Build transitive leakage groups across all known record identities."""
    cluster_by_uniprot = cluster_by_uniprot or {}
    identity_source_ids = specific_source_ids(df)
    uf = UnionFind(df.index)
    first_seen: dict[tuple[str, str], int] = {}

    def connect(kind: str, value: str, row_index: int) -> None:
        if not value:
            return
        key = (kind, value)
        previous = first_seen.setdefault(key, row_index)
        uf.union(previous, row_index)

    for row_index, row in df.iterrows():
        gene = normalize_text(row.get("gene_symbol"))
        uniprot = normalize_text(row.get("uniprot_id"))
        source_id = normalize_text(row.get("source_id"))
        seq_hash = sequence_hash(row.get("sequence"))
        cluster = cluster_by_uniprot.get(uniprot)

        connect("gene", gene, row_index)
        connect("uniprot", uniprot, row_index)
        if source_id in identity_source_ids:
            connect("source_id", source_id, row_index)
        connect("sequence", seq_hash, row_index)
        if cluster is not None:
            connect("sequence_cluster", str(cluster), row_index)

    roots = sorted({uf.find(index) for index in df.index})
    root_to_group = {root: group_id for group_id, root in enumerate(roots)}
    return pd.Series(
        {index: root_to_group[uf.find(index)] for index in df.index},
        name="group_id",
        dtype="int64",
    )


def assign_groups(
    df: pd.DataFrame,
    group_ids: pd.Series,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[int, str]:
    """Assign whole groups while approximately balancing family row counts."""
    import numpy as np

    work = df.copy()
    work["group_id"] = group_ids
    group_info = []
    for group_id, rows in work.groupby("group_id", sort=True):
        family_counts = rows["family_id"].value_counts()
        family_id = int(family_counts.index[0])
        group_info.append(
            {
                "group_id": int(group_id),
                "family_id": family_id,
                "n_rows": int(len(rows)),
            }
        )

    rng = np.random.RandomState(seed)
    assignment: dict[int, str] = {}
    for family_id in sorted({item["family_id"] for item in group_info}):
        groups = [item for item in group_info if item["family_id"] == family_id]
        rng.shuffle(groups)
        groups.sort(key=lambda item: item["n_rows"], reverse=True)

        total_rows = sum(item["n_rows"] for item in groups)
        targets = {
            "train": total_rows * (1.0 - val_frac - test_frac),
            "val": total_rows * val_frac,
            "test": total_rows * test_frac,
        }
        assigned_rows = {name: 0 for name in SPLIT_NAMES}

        for item in groups:
            deficits = {
                name: targets[name] - assigned_rows[name] for name in SPLIT_NAMES
            }
            split_name = max(
                SPLIT_NAMES,
                key=lambda name: (
                    deficits[name] / max(targets[name], 1.0),
                    deficits[name],
                    name == "train",
                ),
            )
            assignment[item["group_id"]] = split_name
            assigned_rows[split_name] += item["n_rows"]

    return assignment


@dataclass
class HygieneReport:
    overlaps: dict[str, dict[str, list[str]]]
    missing_filenames: list[str]

    @property
    def clean(self) -> bool:
        return not self.missing_filenames and not any(
            values
            for category in self.overlaps.values()
            for values in category.values()
        )

    def to_dict(self) -> dict:
        return {
            "clean": self.clean,
            "missing_filenames": self.missing_filenames,
            "overlaps": self.overlaps,
        }


def audit_split(df: pd.DataFrame, split: dict) -> HygieneReport:
    """Check record identity fields for cross-split overlap."""
    filename_to_split = {}
    for split_name in SPLIT_NAMES:
        for filename in split.get(split_name, []):
            filename_to_split[filename] = split_name

    known = set(df["filename"])
    missing = sorted(set(filename_to_split) - known)
    work = df[df["filename"].isin(filename_to_split)].copy()
    work["_split"] = work["filename"].map(filename_to_split)
    work["_gene"] = work["gene_symbol"].map(normalize_text)
    work["_uniprot"] = work["uniprot_id"].map(normalize_text)
    work["_source_id"] = work["source_id"].map(normalize_text)
    identity_source_ids = specific_source_ids(df)
    work.loc[~work["_source_id"].isin(identity_source_ids), "_source_id"] = ""
    work["_sequence"] = work["sequence"].map(sequence_hash)

    overlaps: dict[str, dict[str, list[str]]] = {}
    for label, column in (
        ("gene_symbol", "_gene"),
        ("uniprot_id", "_uniprot"),
        ("source_id", "_source_id"),
        ("sequence", "_sequence"),
    ):
        by_split = {
            split_name: set(work.loc[work["_split"] == split_name, column]) - {""}
            for split_name in SPLIT_NAMES
        }
        overlaps[label] = {}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlaps[label][f"{left}-{right}"] = sorted(by_split[left] & by_split[right])

    return HygieneReport(overlaps=overlaps, missing_filenames=missing)


def donor_exclusion_reasons(
    query: pd.Series,
    donor: pd.Series,
    identity_source_ids: set[str] | None = None,
) -> list[str]:
    """Return identity reasons that make a donor invalid for a query."""
    reasons = []
    fields = [
        ("same_record", "filename"),
        ("same_gene", "gene_symbol"),
        ("same_uniprot", "uniprot_id"),
    ]
    query_source_id = normalize_text(query.get("source_id"))
    if identity_source_ids is None or query_source_id in identity_source_ids:
        fields.append(("same_source_id", "source_id"))
    for reason, column in fields:
        query_value = normalize_text(query.get(column))
        donor_value = normalize_text(donor.get(column))
        if query_value and query_value == donor_value:
            reasons.append(reason)
    query_sequence = sequence_hash(query.get("sequence"))
    donor_sequence = sequence_hash(donor.get("sequence"))
    if query_sequence and query_sequence == donor_sequence:
        reasons.append("same_sequence")
    return reasons


def summarize_split(df: pd.DataFrame, split: dict) -> dict:
    result = {}
    for split_name in SPLIT_NAMES:
        rows = df[df["filename"].isin(split.get(split_name, []))]
        result[split_name] = {
            "rows": int(len(rows)),
            "genes": int(rows["gene_symbol"].nunique()),
            "uniprot_ids": int(rows["uniprot_id"].nunique()),
            "families": dict(Counter(rows["family_name"])),
        }
    return result


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
