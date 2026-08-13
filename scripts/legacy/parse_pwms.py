#!/usr/bin/env python
"""Parse the combined PWM archive (JASPAR + HOCOMOCO + CIS-BP) into processed parquet files.

Input: AllPWMs_JASPARV1_H13Core_CISBPv194w2.tar.gz containing ~3831 log-odds PWMs.
Output:
  - data/processed/pwm.parquet       : All PWMs as probability matrices + metadata
  - data/processed/pwm_summary.txt   : Human-readable summary

Usage:
    python scripts/parse_pwms.py --archive /path/to/AllPWMs_JASPARV1_H13Core_CISBPv194w2.tar.gz --outdir data/processed
"""

import argparse
import io
import logging
import os
import re
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# HOCOMOCO assay type decoding
HOCO_ASSAY = {
    "S": "SELEX",
    "P": "PBM",
    "SM": "SELEX+ChIP-seq",
    "PS": "PBM+SELEX",
    "PSM": "PBM+SELEX+ChIP-seq",
    "PM": "PBM+ChIP-seq",
    "M": "ChIP-seq model",
    "PSG": "PBM+SELEX+genomic",
    "SG": "SELEX+genomic",
    "PSGI": "PBM+SELEX+genomic+integrative",
    "SGI": "SELEX+genomic+integrative",
    "PG": "PBM+genomic",
    "PSGB": "PBM+SELEX+genomic+bacterial",
    "SGIB": "SELEX+genomic+integrative+bacterial",
    "PSGIB": "PBM+SELEX+genomic+integrative+bacterial",
    "SGB": "SELEX+genomic+bacterial",
    "PSIB": "PBM+SELEX+integrative+bacterial",
    "SB": "SELEX+bacterial",
    "PI": "PBM+integrative",
    "SI": "SELEX+integrative",
    "PSI": "PBM+SELEX+integrative",
    "G": "genomic",
    "I": "integrative",
    "B": "bacterial",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Parse combined PWM archive")
    parser.add_argument("--archive", required=True, help="Path to tar.gz archive")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--max-motif-length", type=int, default=20)
    parser.add_argument("--min-motif-length", type=int, default=4)
    return parser.parse_args()


def softmax_pwm(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds matrix (4, L) to probability matrix via softmax over rows."""
    # Numerical stability: subtract per-column max
    shifted = log_odds - log_odds.max(axis=0, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=0, keepdims=True)


def parse_filename(filename: str) -> dict:
    """Extract metadata from PWM filename.

    Patterns:
      TF_NAME.MAxxxx.x.txt          -> JASPAR
      TF_NAME.H13CORE.ver.{assay}.{quality}.txt  -> HOCOMOCO
      TF_NAME.Mxxxxx_x.94d.txt      -> CIS-BP
      TF_NAME.Mxxxxx_x.00d.txt      -> CIS-BP (another version)
      TF_NAME.RCADE.txt             -> RCADE
      TF_NAME.MEME.txt              -> Unknown/MEME
    """
    base = filename.replace(".txt", "")

    # JASPAR: contains MA pattern like MA0037.2
    jaspar_match = re.search(r"\.(MA\d+\.\d+)$", base)
    if jaspar_match:
        tf_name = base[: jaspar_match.start()]
        jaspar_id = jaspar_match.group(1)
        return {
            "tf_name": tf_name,
            "source": "JASPAR",
            "source_id": jaspar_id,
            "assay_type": "",
            "quality_grade": "",
        }

    # HOCOMOCO: contains H13CORE
    hoco_match = re.search(r"\.H13CORE", base)
    if hoco_match:
        tf_name = base[: hoco_match.start()]
        remainder = base[hoco_match.end() + 1:]  # after "H13CORE."
        parts = remainder.split(".")
        version = parts[0] if parts else "0"
        assay_code = parts[1] if len(parts) > 1 else ""
        quality = parts[2] if len(parts) > 2 else ""
        return {
            "tf_name": tf_name,
            "source": "HOCOMOCO",
            "source_id": f"H13CORE.{version}",
            "assay_type": HOCO_ASSAY.get(assay_code, assay_code),
            "quality_grade": quality,
        }

    # CIS-BP: contains pattern like M06179_1.94d or M00984_2.00d
    cisbp_match = re.search(r"\.(M\d+_\d+\.\d+d)$", base)
    if cisbp_match:
        tf_name = base[: cisbp_match.start()]
        cisbp_id = cisbp_match.group(1)
        return {
            "tf_name": tf_name,
            "source": "CISBP",
            "source_id": cisbp_id,
            "assay_type": "",
            "quality_grade": "",
        }

    # RCADE
    if ".RCADE" in base:
        tf_name = base.replace(".RCADE", "")
        return {
            "tf_name": tf_name,
            "source": "RCADE",
            "source_id": "RCADE",
            "assay_type": "",
            "quality_grade": "",
        }

    # MEME (unknown source)
    if ".MEME" in base:
        tf_name = base.replace(".MEME", "")
        return {
            "tf_name": tf_name,
            "source": "MEME",
            "source_id": "MEME",
            "assay_type": "",
            "quality_grade": "",
        }

    return {
        "tf_name": base,
        "source": "UNKNOWN",
        "source_id": "",
        "assay_type": "",
        "quality_grade": "",
    }


def read_pwm_from_tar(tar: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray | None:
    """Read a 4-row log-odds PWM from a tar archive member."""
    try:
        f = tar.extractfile(member)
        if f is None:
            return None
        content = f.read().decode("utf-8", errors="replace")
        lines = content.strip().split("\n")
        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split("\t")]
            rows.append(vals)

        if len(rows) != 4:
            return None

        # Verify all rows have same length
        lengths = [len(r) for r in rows]
        if len(set(lengths)) != 1:
            return None

        return np.array(rows, dtype=np.float64)
    except Exception as e:
        logger.warning(f"Failed to read {member.name}: {e}")
        return None


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.archive):
        logger.error(f"Archive not found: {args.archive}")
        sys.exit(1)

    logger.info(f"Opening archive: {args.archive}")
    records = []
    skipped = {"short": 0, "long": 0, "parse_error": 0}

    with tarfile.open(args.archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".txt") and m.isfile()]
        logger.info(f"Found {len(members)} PWM files")

        for i, member in enumerate(members):
            if (i + 1) % 500 == 0:
                logger.info(f"  Processed {i+1}/{len(members)} files...")

            filename = os.path.basename(member.name)
            meta = parse_filename(filename)

            log_odds = read_pwm_from_tar(tar, member)
            if log_odds is None:
                skipped["parse_error"] += 1
                continue

            motif_length = log_odds.shape[1]

            if motif_length < args.min_motif_length:
                skipped["short"] += 1
                continue

            # Truncate long motifs
            truncated = False
            if motif_length > args.max_motif_length:
                log_odds = log_odds[:, :args.max_motif_length]
                motif_length = args.max_motif_length
                truncated = True

            # Convert to probabilities
            pwm = softmax_pwm(log_odds).astype(np.float32)

            records.append({
                "tf_name": meta["tf_name"],
                "source": meta["source"],
                "source_id": meta["source_id"],
                "assay_type": meta["assay_type"],
                "quality_grade": meta["quality_grade"],
                "motif_length": motif_length,
                "original_motif_length": log_odds.shape[1] if truncated else motif_length,
                "truncated": truncated,
                "pwm": pwm.tobytes(),
                "filename": filename,
            })

    logger.info(f"Parsed {len(records)} PWMs, skipped: {skipped}")

    if not records:
        logger.error("No valid PWMs found!")
        sys.exit(1)

    df = pd.DataFrame(records)
    out_path = os.path.join(args.outdir, "pwm.parquet")
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"PWM Archive Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Total valid PWMs: {len(df)}")
    logger.info(f"Skipped (too short): {skipped['short']}")
    logger.info(f"Skipped (too long, truncated): {df['truncated'].sum()}")
    logger.info(f"Skipped (parse errors): {skipped['parse_error']}")
    logger.info(f"")

    logger.info(f"By source:")
    for src, count in df["source"].value_counts().items():
        logger.info(f"  {src:12s}: {count}")

    logger.info(f"")
    logger.info(f"Motif length distribution:")
    for ml in sorted(df["motif_length"].unique()):
        count = (df["motif_length"] == ml).sum()
        bar = "#" * min(count // 5, 50)
        logger.info(f"  {ml:2d}: {count:4d} {bar}")

    logger.info(f"")
    logger.info(f"HOCOMOCO quality grades:")
    hoco = df[df["source"] == "HOCOMOCO"]
    if not hoco.empty:
        for qg in sorted(hoco["quality_grade"].unique()):
            count = (hoco["quality_grade"] == qg).sum()
            logger.info(f"  {qg}: {count}")

    logger.info(f"")
    logger.info(f"Unique TF names: {df['tf_name'].nunique()}")
    logger.info(f"TFs with multiple profiles: {(df.groupby('tf_name').size() > 1).sum()}")
    logger.info(f"{'='*70}")

    # Save summary to file
    summary_path = os.path.join(args.outdir, "pwm_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total PWMs: {len(df)}\n")
        f.write(f"Sources: {dict(df['source'].value_counts())}\n")
        f.write(f"Unique TFs: {df['tf_name'].nunique()}\n")
        f.write(f"Motif lengths: {dict(df['motif_length'].value_counts().sort_index())}\n")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
