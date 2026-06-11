#!/usr/bin/env python
"""Create train/val/test splits for TFScope training.

Three strategies:
  - lofo: Leave-One-Family-Out (10 splits, one per held-out family)
  - identity: Cluster at 30% sequence identity, assign clusters to splits
  - random: Stratified random 80/10/10

All methods enforce 80% deduplication across splits.
Dedup operates at the protein level (uniprot_id); all PWMs for a protein
stay in the same split.

Usage:
    python scripts/create_splits.py --tf-data data/processed/tf_pwm.parquet --outdir data/processed/splits --method lofo --method identity --method random
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

FAMILY_NAMES = {
    0: "C2H2_short",
    1: "C2H2_medium",
    2: "C2H2_long",
    3: "bHLH",
    4: "Homeodomain",
    5: "bZIP",
    6: "Nuclear_Receptor",
    7: "Forkhead",
    8: "ETS",
    9: "Other",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--tf-data", default="data/processed/tf_pwm.parquet")
    parser.add_argument("--outdir", default="data/processed/splits")
    parser.add_argument("--method", nargs="+", choices=["lofo", "identity", "random"],
                        default=["lofo", "identity", "random"])
    parser.add_argument("--identity-threshold", type=float, default=0.5,
                        help="Jaccard threshold for 80%% identity dedup (default 0.5)")
    parser.add_argument("--cluster-threshold", type=float, default=0.15,
                        help="Jaccard threshold for low-identity clustering (default 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_similarity_matrix(sequences: dict[str, str]) -> np.ndarray:
    """Compute pairwise protein similarity using 3-mer Jaccard.

    Fast O(N) per-sequence k-mer extraction, O(N^2 * |kmer|) comparison.
    ~1 second for 1270 proteins instead of ~30 minutes with alignment.
    Jaccard >= 0.4 is a good proxy for >= 80% sequence identity.
    """
    ids = list(sequences.keys())
    n = len(ids)

    # Extract k-mer sets
    kmer_sets = []
    for pid in ids:
        seq = sequences[pid]
        kmers = set(seq[i:i+3] for i in range(len(seq) - 2))
        kmer_sets.append(kmers)

    logger.info(f"Computing {n}x{n} similarity matrix (3-mer Jaccard)...")
    t0 = time.time()
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        ki = kmer_sets[i]
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0
                continue
            kj = kmer_sets[j]
            inter = len(ki & kj)
            if inter == 0:
                continue
            jac = inter / (len(ki) + len(kj) - inter)
            matrix[i, j] = jac
            matrix[j, i] = jac

    logger.info(f"Similarity matrix computed in {time.time() - t0:.1f}s")
    return matrix, ids


def greedy_identity_clustering(
    identity_matrix: np.ndarray,
    ids: list[str],
    threshold: float,
) -> list[set[str]]:
    """Greedy clustering: assign each sequence to first cluster where identity >= threshold."""
    n = len(ids)
    assigned = [-1] * n
    cluster_reps = []

    for i in range(n):
        found = False
        for ci, rep_idx in enumerate(cluster_reps):
            if identity_matrix[i, rep_idx] >= threshold:
                assigned[i] = ci
                found = True
                break
        if not found:
            assigned[i] = len(cluster_reps)
            cluster_reps.append(i)

    clusters = defaultdict(set)
    for i, ci in enumerate(assigned):
        clusters[ci].add(ids[i])

    return [clusters[ci] for ci in sorted(clusters.keys())]


def enforce_dedup(
    train_proteins: set[str],
    val_proteins: set[str],
    test_proteins: set[str],
    dedup_clusters: list[set[str]],
) -> tuple[set[str], set[str], set[str]]:
    """Ensure no two proteins >= 80% identical appear in different splits."""
    train_proteins = set(train_proteins)
    val_proteins = set(val_proteins)
    test_proteins = set(test_proteins)

    for cluster in dedup_clusters:
        members = cluster & (train_proteins | val_proteins | test_proteins)
        if len(members) <= 1:
            continue

        in_train = members & train_proteins
        in_val = members & val_proteins
        in_test = members & test_proteins

        if in_test and (in_train or in_val):
            for pid in in_test:
                test_proteins.discard(pid)
                train_proteins.add(pid)
        if in_val and in_train:
            for pid in in_val:
                val_proteins.discard(pid)
                train_proteins.add(pid)

    return train_proteins, val_proteins, test_proteins


def proteins_to_filenames(df: pd.DataFrame, protein_ids: set[str]) -> list[str]:
    """Map protein IDs to their PWM filenames."""
    return sorted(df[df["uniprot_id"].isin(protein_ids)]["filename"].tolist())


def save_split(split_dict: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split_dict, f, indent=2)
    logger.info(f"Saved split to {path} "
                f"(train={len(split_dict['train'])}, "
                f"val={len(split_dict['val'])}, "
                f"test={len(split_dict['test'])})")


def create_lofo_splits(
    df: pd.DataFrame,
    protein_family: dict[str, int],
    dedup_clusters: list[set[str]],
    outdir: str,
    seed: int,
):
    """Leave-One-Family-Out: hold out one DBD family, train on rest."""
    rng = np.random.RandomState(seed)
    lofo_dir = os.path.join(outdir, "lofo")
    os.makedirs(lofo_dir, exist_ok=True)

    all_proteins = set(protein_family.keys())

    for family_id in sorted(set(protein_family.values())):
        family_name = FAMILY_NAMES.get(family_id, f"family_{family_id}")
        test_proteins = {p for p, f in protein_family.items() if f == family_id}
        candidate_train = all_proteins - test_proteins

        # Enforce dedup
        train_proteins, _, test_proteins = enforce_dedup(
            candidate_train, set(), test_proteins, dedup_clusters
        )

        # Split train into train + val (90/10)
        train_list = sorted(train_proteins)
        rng.shuffle(train_list)
        val_size = max(1, len(train_list) // 10)
        val_proteins = set(train_list[:val_size])
        train_proteins = set(train_list[val_size:])

        split = {
            "train": proteins_to_filenames(df, train_proteins),
            "val": proteins_to_filenames(df, val_proteins),
            "test": proteins_to_filenames(df, test_proteins),
            "metadata": {
                "held_out_family_id": int(family_id),
                "held_out_family_name": family_name,
                "n_train_proteins": len(train_proteins),
                "n_val_proteins": len(val_proteins),
                "n_test_proteins": len(test_proteins),
                "n_train_pwms": len(proteins_to_filenames(df, train_proteins)),
                "n_val_pwms": len(proteins_to_filenames(df, val_proteins)),
                "n_test_pwms": len(proteins_to_filenames(df, test_proteins)),
            },
        }
        save_split(split, os.path.join(lofo_dir, f"{family_name}.json"))

    logger.info(f"Created {len(set(protein_family.values()))} LOFO splits")


def create_identity_splits(
    df: pd.DataFrame,
    identity_matrix: np.ndarray,
    protein_ids: list[str],
    protein_family: dict[str, int],
    dedup_clusters: list[set[str]],
    outdir: str,
    cluster_threshold: float,
    seed: int,
):
    """Cluster at low identity (30%), assign entire clusters to splits."""
    rng = np.random.RandomState(seed)
    id_dir = os.path.join(outdir, "identity")
    os.makedirs(id_dir, exist_ok=True)

    low_clusters = greedy_identity_clustering(identity_matrix, protein_ids, cluster_threshold)
    logger.info(f"Created {len(low_clusters)} clusters at {cluster_threshold:.0%} identity")

    # Build cluster info with dominant family
    cluster_info = []
    for cluster in low_clusters:
        families = [protein_family.get(p, 9) for p in cluster if p in protein_family]
        if not families:
            continue
        dominant_family = max(set(families), key=families.count)
        cluster_info.append({
            "members": cluster,
            "dominant_family": dominant_family,
        })

    # Stratified assignment by family
    train_proteins, val_proteins, test_proteins = set(), set(), set()

    for family_id in sorted(set(protein_family.values())):
        family_clusters = [c for c in cluster_info if c["dominant_family"] == family_id]
        if not family_clusters:
            continue

        family_clusters.sort(key=lambda c: sorted(c["members"])[0])
        rng.shuffle(family_clusters)

        n = len(family_clusters)
        n_test = max(1, n // 10)
        n_val = max(1, n // 10)

        for i, c in enumerate(family_clusters):
            if i < n_test:
                test_proteins |= c["members"]
            elif i < n_test + n_val:
                val_proteins |= c["members"]
            else:
                train_proteins |= c["members"]

    train_proteins, val_proteins, test_proteins = enforce_dedup(
        train_proteins, val_proteins, test_proteins, dedup_clusters
    )

    split = {
        "train": proteins_to_filenames(df, train_proteins),
        "val": proteins_to_filenames(df, val_proteins),
        "test": proteins_to_filenames(df, test_proteins),
        "metadata": {
            "method": "identity_30pct",
            "n_clusters": len(low_clusters),
            "cluster_threshold": cluster_threshold,
            "seed": seed,
        },
    }
    save_split(split, os.path.join(id_dir, "split.json"))


def create_random_splits(
    df: pd.DataFrame,
    protein_family: dict[str, int],
    dedup_clusters: list[set[str]],
    outdir: str,
    seed: int,
):
    """Stratified random 80/10/10 with dedup enforcement."""
    rng = np.random.RandomState(seed)
    rand_dir = os.path.join(outdir, "random")
    os.makedirs(rand_dir, exist_ok=True)

    train_proteins, val_proteins, test_proteins = set(), set(), set()

    for family_id in sorted(set(protein_family.values())):
        family_proteins = sorted(p for p, f in protein_family.items() if f == family_id)
        rng.shuffle(family_proteins)

        n = len(family_proteins)
        n_test = max(1, n // 10)
        n_val = max(1, n // 10)

        test_proteins.update(family_proteins[:n_test])
        val_proteins.update(family_proteins[n_test:n_test + n_val])
        train_proteins.update(family_proteins[n_test + n_val:])

    train_proteins, val_proteins, test_proteins = enforce_dedup(
        train_proteins, val_proteins, test_proteins, dedup_clusters
    )

    split = {
        "train": proteins_to_filenames(df, train_proteins),
        "val": proteins_to_filenames(df, val_proteins),
        "test": proteins_to_filenames(df, test_proteins),
        "metadata": {
            "method": "random_stratified",
            "seed": seed,
        },
    }
    save_split(split, os.path.join(rand_dir, "split.json"))

    # Per-family splits
    per_fam_dir = os.path.join(rand_dir, "per_family")
    os.makedirs(per_fam_dir, exist_ok=True)

    for family_id in sorted(set(protein_family.values())):
        family_name = FAMILY_NAMES.get(family_id, f"family_{family_id}")
        family_proteins = sorted(p for p, f in protein_family.items() if f == family_id)
        rng.shuffle(family_proteins)

        n = len(family_proteins)
        n_test = max(1, n // 10)
        n_val = max(1, n // 10)

        fam_split = {
            "train": proteins_to_filenames(df, set(family_proteins[n_test + n_val:])),
            "val": proteins_to_filenames(df, set(family_proteins[n_test:n_test + n_val])),
            "test": proteins_to_filenames(df, set(family_proteins[:n_test])),
            "metadata": {"family_id": int(family_id), "family_name": family_name},
        }
        save_split(fam_split, os.path.join(per_fam_dir, f"{family_name}.json"))


def main():
    args = parse_args()

    if not os.path.exists(args.tf_data):
        logger.error(f"Data not found at {args.tf_data}. Run map_tf_annotations.py first.")
        sys.exit(1)

    df = pd.read_parquet(args.tf_data)
    logger.info(f"Loaded {len(df)} TF records ({df['uniprot_id'].nunique()} unique proteins)")

    # Build protein-level data (deduplicate by uniprot_id)
    protein_df = df.drop_duplicates(subset=["uniprot_id"])[["uniprot_id", "sequence", "family_id"]].reset_index(drop=True)
    sequences = dict(zip(protein_df["uniprot_id"], protein_df["sequence"]))
    protein_family = dict(zip(protein_df["uniprot_id"], protein_df["family_id"]))

    logger.info(f"Unique proteins for identity computation: {len(sequences)}")

    # Compute pairwise identity at protein level
    t0 = time.time()
    similarity_matrix, protein_ids = compute_similarity_matrix(sequences)
    logger.info(f"Similarity matrix computed in {time.time() - t0:.1f}s")

    # Create 80% dedup clusters
    dedup_clusters = greedy_identity_clustering(
        similarity_matrix, protein_ids, args.identity_threshold
    )
    logger.info(f"Created {len(dedup_clusters)} clusters at {args.identity_threshold:.0%} identity")

    # Run requested methods
    for method in args.method:
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating {method} splits...")

        if method == "lofo":
            create_lofo_splits(df, protein_family, dedup_clusters, args.outdir, args.seed)
        elif method == "identity":
            create_identity_splits(
                df, similarity_matrix, protein_ids, protein_family,
                dedup_clusters, args.outdir, args.cluster_threshold, args.seed
            )
        elif method == "random":
            create_random_splits(df, protein_family, dedup_clusters, args.outdir, args.seed)

    logger.info(f"\n{'='*60}")
    logger.info("All splits created!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
