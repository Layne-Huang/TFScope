#!/usr/bin/env python
"""Build a leakage-controlled cluster40 train/validation/test split.

Rows are grouped transitively by:
  - CD-HIT sequence cluster;
  - gene symbol;
  - UniProt accession;
  - motif source ID;
  - exact sequence.

All rows in a connected component are assigned to the same split. This avoids
the historical failure mode where one representative per gene was clustered
and unmapped records were silently placed in training.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tfscope.data.split_hygiene import (
    SPLIT_NAMES,
    assign_groups,
    audit_split,
    build_group_ids,
    normalize_text,
    summarize_split,
    write_json,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATA = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
OUTDIR = "data/processed/splits/cluster40_clean"
EXISTING_CLUSTERS = "data/processed/splits/cluster40/cdhit_clusters.clstr"


def run_cdhit(fasta_path: str, out_prefix: str, identity: float, threads: int) -> str:
    executable = shutil.which("cd-hit")
    if executable is None:
        raise RuntimeError(
            "cd-hit is not installed. Pass --clusters with an existing .clstr file."
        )
    word_size = 2 if identity < 0.5 else (3 if identity < 0.6 else 4)
    command = [
        executable,
        "-i",
        fasta_path,
        "-o",
        out_prefix,
        "-c",
        str(identity),
        "-n",
        str(word_size),
        "-M",
        "4000",
        "-T",
        str(threads),
        "-d",
        "0",
        "-g",
        "1",
    ]
    log.info("Running: %s", " ".join(command))
    subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return out_prefix + ".clstr"


def parse_clstr(path: str) -> dict[str, int]:
    """Return normalized FASTA identifier -> cluster ID."""
    assignment = {}
    cluster_id = -1
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">Cluster"):
                cluster_id += 1
                continue
            if ">" not in line:
                continue
            identifier = line.split(">", 1)[1].split("...", 1)[0].strip()
            assignment[normalize_text(identifier)] = cluster_id
    return assignment


def write_unique_uniprot_fasta(df: pd.DataFrame, path: str) -> int:
    """Write one deterministic sequence per UniProt accession."""
    representatives = (
        df.assign(_uid=df["uniprot_id"].map(normalize_text))
        .sort_values(["_uid", "filename"])
        .drop_duplicates("_uid")
    )
    with open(path, "w") as handle:
        for _, row in representatives.iterrows():
            handle.write(f">{normalize_text(row['uniprot_id'])}\n{row['sequence']}\n")
    return len(representatives)


def build_group_manifest(df: pd.DataFrame, group_ids: pd.Series, assignment: dict) -> list:
    work = df.copy()
    work["group_id"] = group_ids
    work["split"] = work["group_id"].map(assignment)
    manifest = []
    for group_id, rows in work.groupby("group_id", sort=True):
        manifest.append(
            {
                "group_id": int(group_id),
                "split": rows["split"].iloc[0],
                "n_rows": int(len(rows)),
                "filenames": sorted(rows["filename"].tolist()),
                "genes": sorted({normalize_text(value) for value in rows["gene_symbol"]}),
                "uniprot_ids": sorted(
                    {normalize_text(value) for value in rows["uniprot_id"]}
                ),
                "source_ids": sorted(
                    {normalize_text(value) for value in rows["source_id"]}
                ),
                "family_ids": sorted({int(value) for value in rows["family_id"]}),
            }
        )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--clusters", default=EXISTING_CLUSTERS)
    parser.add_argument("--identity", type=float, default=0.4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if args.val_frac < 0 or args.test_frac < 0:
        raise ValueError("split fractions must be non-negative")
    if args.val_frac + args.test_frac >= 1:
        raise ValueError("val_frac + test_frac must be less than 1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data).reset_index(drop=True)
    log.info(
        "Loaded %d rows, %d genes, %d UniProt accessions",
        len(df),
        df["gene_symbol"].nunique(),
        df["uniprot_id"].nunique(),
    )

    cluster_path = args.clusters
    if not cluster_path or not os.path.exists(cluster_path):
        fasta_path = str(outdir / "sequences.fasta")
        n_sequences = write_unique_uniprot_fasta(df, fasta_path)
        log.info("Wrote %d unique UniProt sequences to %s", n_sequences, fasta_path)
        start = time.time()
        cluster_path = run_cdhit(
            fasta_path,
            str(outdir / "cdhit_clusters"),
            identity=args.identity,
            threads=args.threads,
        )
        log.info("CD-HIT completed in %.1f seconds", time.time() - start)
    else:
        log.info("Reusing CD-HIT assignments from %s", cluster_path)
        cluster_copy = outdir / "cdhit_clusters.clstr"
        if Path(cluster_path).resolve() != cluster_copy.resolve():
            shutil.copyfile(cluster_path, cluster_copy)

    cluster_by_uniprot = parse_clstr(cluster_path)
    n_unclustered = sum(
        normalize_text(value) not in cluster_by_uniprot for value in df["uniprot_id"]
    )
    if n_unclustered:
        log.warning(
            "%d rows have UniProt IDs absent from the supplied cluster file; "
            "they remain grouped by gene, accession, source, and exact sequence",
            n_unclustered,
        )

    group_ids = build_group_ids(df, cluster_by_uniprot)
    assignment = assign_groups(
        df,
        group_ids,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    row_splits = group_ids.map(assignment)

    split = {
        split_name: sorted(df.loc[row_splits == split_name, "filename"].tolist())
        for split_name in SPLIT_NAMES
    }
    split["metadata"] = {
        "method": "connected_group_cluster_split",
        "data": args.data,
        "identity_threshold": args.identity,
        "cluster_file": str(cluster_path),
        "n_sequence_clusters": len(set(cluster_by_uniprot.values())),
        "n_connected_groups": int(group_ids.nunique()),
        "group_keys": [
            "sequence_cluster",
            "gene_symbol",
            "uniprot_id",
            "source_id_shared_by_at_most_2_genes",
            "exact_sequence",
        ],
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
    }

    report = audit_split(df, split)
    if not report.clean:
        raise RuntimeError(
            "internal error: generated split failed hygiene audit\n"
            + json.dumps(report.to_dict(), indent=2)
        )

    manifest = build_group_manifest(df, group_ids, assignment)
    summary = summarize_split(df, split)
    split["metadata"]["summary"] = summary

    write_json(outdir / "split.json", split)
    write_json(outdir / "group_manifest.json", {"groups": manifest})
    write_json(
        outdir / "hygiene_report.json",
        {"split": report.to_dict(), "summary": summary},
    )

    for split_name in SPLIT_NAMES:
        values = summary[split_name]
        log.info(
            "%s: %d rows, %d genes, %d UniProt IDs",
            split_name,
            values["rows"],
            values["genes"],
            values["uniprot_ids"],
        )
    log.info("Saved clean split artifacts to %s", outdir)


if __name__ == "__main__":
    main()
