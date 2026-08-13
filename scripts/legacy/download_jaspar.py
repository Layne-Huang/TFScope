#!/usr/bin/env python
"""Download JASPAR 2024 vertebrate TF profiles (PWM matrices + metadata).

Usage:
    python scripts/download_jaspar.py --species vertebrates --outdir data/raw/jaspar
    python scripts/download_jaspar.py --resume  # continue interrupted download
"""

import argparse
import json
import logging
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://jaspar.elixir.no/api/v1/"


def parse_args():
    parser = argparse.ArgumentParser(description="Download JASPAR 2024 TF profiles")
    parser.add_argument("--species", default="vertebrates")
    parser.add_argument("--collection", default="CORE")
    parser.add_argument("--outdir", default="data/raw/jaspar")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--resume", action="store_true", help="Skip already-downloaded profiles")
    parser.add_argument("--overwrite", action="store_true", help="Re-download everything")
    return parser.parse_args()


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_matrix_list(
    session: requests.Session,
    tax_group: str = "vertebrates",
    collection: str = "CORE",
    page_size: int = 200,
) -> list[dict]:
    """Paginate through /matrix/ endpoint to collect all profile metadata."""
    profiles = []
    url = f"{BASE_URL}matrix/?tax_group={tax_group}&collection={collection}&page_size={page_size}&format=json"

    while url:
        logger.info(f"Fetching page: {url}")
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        profiles.extend(results)
        count = data.get("count", "?")
        logger.info(f"  Got {len(results)} profiles (total so far: {len(profiles)}/{count})")
        url = data.get("next")
        time.sleep(0.2)

    logger.info(f"Total profiles fetched: {len(profiles)}")
    return profiles


def fetch_profile_detail(
    session: requests.Session, matrix_id: str
) -> dict | None:
    """Fetch full profile detail including PFM matrix."""
    url = f"{BASE_URL}matrix/{matrix_id}/?format=json"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch {matrix_id}: {e}")
        return None


def fetch_meme_format(session: requests.Session, matrix_id: str) -> str | None:
    """Fetch single profile in MEME format."""
    url = f"{BASE_URL}matrix/{matrix_id}/?format=meme"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch MEME for {matrix_id}: {e}")
        return None


def pfm_to_pwm(pfm: dict, pseudocount: float = 0.01) -> np.ndarray:
    """Convert position frequency matrix to probability matrix.

    Args:
        pfm: {"A": [counts], "C": [...], "G": [...], "T": [...]}
        pseudocount: added to each count before normalization.

    Returns:
        (4, L) float32 array with ACGT rows, probabilities summing to 1 per column.
    """
    order = ["A", "C", "G", "T"]
    counts = np.array([pfm[base] for base in order], dtype=np.float64)
    counts += pseudocount
    col_sums = counts.sum(axis=0, keepdims=True)
    pwm = counts / col_sums
    return pwm.astype(np.float32)


def load_progress(outdir: str) -> set[str]:
    path = os.path.join(outdir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return set(data.get("downloaded_ids", []))
    return set()


def save_progress(outdir: str, downloaded_ids: set[str], last_fetched: int):
    path = os.path.join(outdir, "progress.json")
    with open(path, "w") as f:
        json.dump({
            "downloaded_ids": sorted(downloaded_ids),
            "last_fetched": last_fetched,
        }, f)


def log_error(outdir: str, matrix_id: str, error: str):
    path = os.path.join(outdir, "errors.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({"matrix_id": matrix_id, "error": error}) + "\n")


def save_profiles(
    profiles_meta: list[dict],
    pwm_matrices: dict[str, np.ndarray],
    outdir: str,
):
    """Save metadata to parquet, PWMs to HDF5, MEMEs to files."""
    # Metadata parquet
    rows = []
    for p in profiles_meta:
        matrix_id = p.get("matrix_id", p.get("id", ""))
        pfm = p.get("pfm", {})
        motif_length = len(pfm.get("A", [])) if pfm else 0

        uniprot_ids = p.get("uniprot_ids", []) or []
        if isinstance(uniprot_ids, str):
            uniprot_ids = [uniprot_ids]

        jaspar_class = p.get("class", []) or []
        if isinstance(jaspar_class, str):
            jaspar_class = [jaspar_class]

        family = p.get("family", []) or []
        if isinstance(family, str):
            family = [family]

        species_list = []
        for sp in (p.get("species", []) or []):
            if isinstance(sp, dict):
                species_list.append(sp.get("name", ""))
            else:
                species_list.append(str(sp))

        tax_ids = []
        for sp in (p.get("species", []) or []):
            if isinstance(sp, dict):
                tax_ids.append(sp.get("tax_id", ""))
            else:
                tax_ids.append("")

        rows.append({
            "matrix_id": matrix_id,
            "base_id": p.get("base_id", ""),
            "version": p.get("version", 0),
            "name": p.get("name", ""),
            "symbol": p.get("symbol", ""),
            "uniprot_ids": uniprot_ids,
            "class": jaspar_class,
            "family": family,
            "species": species_list,
            "tax_ids": tax_ids,
            "motif_length": motif_length,
            "in_scope": 4 <= motif_length <= 20,
        })

    df = pd.DataFrame(rows)
    parquet_path = os.path.join(outdir, "profiles.parquet")
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved {len(df)} profiles to {parquet_path}")

    # PWMs to HDF5
    h5_path = os.path.join(outdir, "pwms.h5")
    with h5py.File(h5_path, "w") as f:
        for mid, pwm in pwm_matrices.items():
            f.create_dataset(mid, data=pwm)
    logger.info(f"Saved {len(pwm_matrices)} PWMs to {h5_path}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "memes"), exist_ok=True)

    session = make_session()

    # Check for existing progress
    downloaded_ids = set()
    if args.resume and not args.overwrite:
        downloaded_ids = load_progress(args.outdir)
        logger.info(f"Resuming: {len(downloaded_ids)} profiles already downloaded")

    if args.overwrite:
        downloaded_ids = set()
        logger.info("Overwrite mode: starting fresh")

    # Step 1: Fetch profile list
    logger.info("Fetching JASPAR profile list...")
    profile_list = fetch_matrix_list(
        session, args.species, args.collection, args.page_size
    )

    # Step 2: Fetch details for each profile
    all_meta = []
    pwm_matrices = {}
    total = len(profile_list)

    # Load already-fetched metadata if resuming
    existing_parquet = os.path.join(args.outdir, "profiles.parquet")
    existing_h5 = os.path.join(args.outdir, "pwms.h5")

    if args.resume and os.path.exists(existing_parquet) and os.path.exists(existing_h5):
        existing_df = pd.read_parquet(existing_parquet)
        for _, row in existing_df.iterrows():
            mid = row["matrix_id"]
            if mid in downloaded_ids:
                all_meta.append(row.to_dict())

        with h5py.File(existing_h5, "r") as f:
            for mid in downloaded_ids:
                if mid in f:
                    pwm_matrices[mid] = f[mid][()]

    for i, summary in enumerate(profile_list):
        matrix_id = summary.get("matrix_id", summary.get("id", ""))

        if matrix_id in downloaded_ids:
            logger.info(f"  [{i+1}/{total}] Skipping {matrix_id} (already downloaded)")
            continue

        logger.info(f"  [{i+1}/{total}] Fetching {matrix_id}...")
        detail = fetch_profile_detail(session, matrix_id)

        if detail is None:
            log_error(args.outdir, matrix_id, "Failed to fetch detail")
            time.sleep(0.2)
            continue

        pfm = detail.get("pfm", {})
        if pfm and pfm.get("A"):
            pwm = pfm_to_pwm(pfm)
            pwm_matrices[matrix_id] = pwm

        all_meta.append(detail)
        downloaded_ids.add(matrix_id)

        # Fetch and save MEME format
        meme_text = fetch_meme_format(session, matrix_id)
        if meme_text:
            meme_path = os.path.join(args.outdir, "memes", f"{matrix_id}.meme")
            with open(meme_path, "w") as f:
                f.write(meme_text)

        # Save progress checkpoint every 50 profiles
        if (i + 1) % 50 == 0:
            save_progress(args.outdir, downloaded_ids, i)
            save_profiles(all_meta, pwm_matrices, args.outdir)
            logger.info(f"  Checkpoint saved ({len(downloaded_ids)} profiles)")

        time.sleep(0.2)

    # Final save
    save_profiles(all_meta, pwm_matrices, args.outdir)
    save_progress(args.outdir, downloaded_ids, total)

    # Summary statistics
    df = pd.read_parquet(os.path.join(args.outdir, "profiles.parquet"))
    logger.info(f"\n{'='*60}")
    logger.info(f"Download complete!")
    logger.info(f"  Total profiles: {len(df)}")
    logger.info(f"  In scope (motif 4-20): {df['in_scope'].sum()}")
    logger.info(f"  Out of scope: {(~df['in_scope']).sum()}")
    logger.info(f"  Motif length distribution:")
    for ml in sorted(df['motif_length'].unique()):
        count = (df['motif_length'] == ml).sum()
        marker = " *" if 4 <= ml <= 20 else ""
        logger.info(f"    length {ml:2d}: {count}{marker}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
