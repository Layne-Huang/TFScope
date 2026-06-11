#!/usr/bin/env python
"""Map TF names from parsed PWM archive to UniProt sequences and InterPro DBD annotations.

Works with the combined archive (JASPAR + HOCOMOCO + CIS-BP + RCADE).
Reads from data/processed/pwm.parquet (output of parse_pwms.py).

For each unique TF name:
  1. Resolve UniProt accession via gene symbol search
  2. Fetch protein sequence from UniProt REST API
  3. Fetch domain annotations from InterPro REST API
  4. Assign 10-category family label
  5. Merge with all PWM profiles for that TF

Usage:
    python scripts/map_tf_annotations.py --pwm-parquet data/processed/pwm.parquet --outdir data/processed
    python scripts/map_tf_annotations.py --resume  # skip already resolved TF names
"""

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Pfam IDs for DBD detection
DBD_PFAM = {
    "PF00096": "C2H2",
    "PF00010": "bHLH",
    "PF07916": "bHLH",
    "PF00046": "Homeodomain",
    "PF00170": "bZIP",
    "PF07716": "bZIP",
    "PF00105": "Nuclear_Receptor",
    "PF00104": "Nuclear_Receptor",
    "PF00250": "Forkhead",
    "PF00178": "ETS",
}

# InterPro IDs for DBD detection
DBD_INTERPRO = {
    "IPR001878": "C2H2",
    "IPR011598": "bHLH",
    "IPR001356": "Homeodomain",
    "IPR004827": "bZIP",
    "IPR000536": "Nuclear_Receptor",
    "IPR001766": "Forkhead",
    "IPR000418": "ETS",
}

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
    parser = argparse.ArgumentParser(description="Map TF names to UniProt sequences + InterPro DBD annotations")
    parser.add_argument("--pwm-parquet", default="data/processed/pwm.parquet",
                        help="Path to pwm.parquet from parse_pwms.py")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--species", default="Homo sapiens",
                        help="Preferred species for UniProt resolution")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def resolve_uniprot(session: requests.Session, tf_name: str, species: str = "") -> dict | None:
    """Resolve TF name to UniProt accession + sequence + organism.

    Strategy:
      1. Search UniProt by gene symbol + reviewed (Swiss-Prot) + species
      2. If no match, try without species filter
      3. If still no match, try protein name search
      4. If still no match, relax reviewed filter
    """
    species_filter = f" AND (organism_id:9606)" if "sapiens" in species.lower() else ""

    # Attempt 1: gene symbol + species + reviewed
    results = _search_uniprot(session, f"(gene:{tf_name}) AND (reviewed:true){species_filter}")
    if results:
        return _pick_best_result(results, species)

    # Attempt 2: gene symbol + reviewed, no species
    results = _search_uniprot(session, f"(gene:{tf_name}) AND (reviewed:true)")
    if results:
        return _pick_best_result(results, species)

    # Attempt 3: protein name search
    results = _search_uniprot(session, f"(protein_name:{tf_name}) AND (reviewed:true)")
    if results:
        return _pick_best_result(results, species)

    # Attempt 4: gene symbol without reviewed filter
    results = _search_uniprot(session, f"(gene:{tf_name}){species_filter}")
    if results:
        return _pick_best_result(results, species)

    return None


def _search_uniprot(session: requests.Session, query: str) -> list[dict]:
    """Search UniProt and return results with accession, sequence, organism."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "json",
        "size": 5,
        "fields": "accession,gene_names,protein_name,sequence,organism_name,organism_id,length",
    }
    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except requests.exceptions.RequestException as e:
        logger.warning(f"UniProt search failed for query '{query}': {e}")
        return []


def _pick_best_result(results: list[dict], preferred_species: str) -> dict | None:
    """Pick the best UniProt result, preferring preferred species."""
    preferred = preferred_species.lower()
    for r in results:
        org = r.get("organism", {}).get("scientificName", "").lower()
        if preferred in org:
            return _extract_uniprot_info(r)
    # Fall back to first result
    return _extract_uniprot_info(results[0])


def _extract_uniprot_info(entry: dict) -> dict | None:
    """Extract relevant fields from a UniProt entry."""
    seq_info = entry.get("sequence", {})
    organism = entry.get("organism", {})
    genes = entry.get("genes", [{}])

    gene_names = []
    for g in genes:
        name = g.get("geneName", {}).get("value")
        if name:
            gene_names.append(name)
        for syn in g.get("synonyms", []):
            val = syn.get("value")
            if val:
                gene_names.append(val)

    return {
        "accession": entry.get("primaryAccession", ""),
        "sequence": seq_info.get("value", ""),
        "length": seq_info.get("length", 0),
        "organism": organism.get("scientificName", ""),
        "gene_names": gene_names,
    }


def fetch_interpro_domains(session: requests.Session, uniprot_id: str) -> list[dict]:
    """Fetch domain annotations from InterPro REST API."""
    url = f"https://www.ebi.ac.uk/interpro/api/entry/InterPro/protein/UniProt/{uniprot_id}?format=json"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        domains = []

        for entry in data.get("results", []):
            metadata = entry.get("metadata", {})
            interpro_id = metadata.get("accession", "")
            name = metadata.get("name", "")

            member_dbs = metadata.get("member_databases", {})
            pfam_ids = []
            if "pfam" in member_dbs:
                pfam_entries = member_dbs["pfam"]
                if isinstance(pfam_entries, dict):
                    pfam_ids = list(pfam_entries.keys())
                elif isinstance(pfam_entries, list):
                    pfam_ids = [p.get("accession", "") for p in pfam_entries]

            proteins = entry.get("proteins", [])
            for prot in proteins:
                locations = prot.get("entry_protein_locations", [])
                for loc in locations:
                    for frag in loc.get("fragments", []):
                        domains.append({
                            "interpro_id": interpro_id,
                            "name": name,
                            "pfam_ids": pfam_ids,
                            "start": frag.get("start", 0),
                            "end": frag.get("end", 0),
                        })

        return domains
    except requests.exceptions.RequestException as e:
        logger.warning(f"InterPro fetch failed for {uniprot_id}: {e}")
        return []


def extract_dbd_regions(domains: list[dict], seq_length: int) -> list[tuple[int, int]]:
    """Extract DBD regions from InterPro domain annotations.

    Returns list of (start, end) tuples (0-indexed).
    Merges overlapping/adjacent domains within 10 residues.
    """
    dbd_domains = []
    for d in domains:
        is_dbd = False
        for pfam_id in d.get("pfam_ids", []):
            if pfam_id in DBD_PFAM:
                is_dbd = True
                break
        if d.get("interpro_id", "") in DBD_INTERPRO:
            is_dbd = True
        if is_dbd:
            dbd_domains.append((max(0, d["start"] - 1), min(seq_length, d["end"])))

    if not dbd_domains:
        return []

    dbd_domains.sort()
    merged = [dbd_domains[0]]
    for start, end in dbd_domains[1:]:
        if start <= merged[-1][1] + 10:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def count_c2h2_fingers(domains: list[dict]) -> int:
    """Count C2H2 zinc finger domains."""
    count = 0
    for d in domains:
        if d.get("interpro_id", "") == "IPR001878":
            count += 1
        for pfam_id in d.get("pfam_ids", []):
            if pfam_id == "PF00096":
                count += 1
    return count


def assign_family_label(domains: list[dict]) -> tuple[int, str]:
    """Assign one of 10 family labels based on InterPro domains.

    Priority: specific families first, then C2H2 subcategories, then Other.
    """
    # Check InterPro domains for specific families (non-C2H2)
    dbd_types = set()
    is_c2h2 = False
    for d in domains:
        for pfam_id in d.get("pfam_ids", []):
            if pfam_id in DBD_PFAM:
                dbd_type = DBD_PFAM[pfam_id]
                if dbd_type == "C2H2":
                    is_c2h2 = True
                else:
                    dbd_types.add(dbd_type)
        if d.get("interpro_id", "") in DBD_INTERPRO:
            dbd_type = DBD_INTERPRO[d["interpro_id"]]
            if dbd_type == "C2H2":
                is_c2h2 = True
            else:
                dbd_types.add(dbd_type)

    # Check specific families
    family_map = {
        "bHLH": 3, "Homeodomain": 4, "bZIP": 5,
        "Nuclear_Receptor": 6, "Forkhead": 7, "ETS": 8,
    }
    for dbd_type in dbd_types:
        if dbd_type in family_map:
            return family_map[dbd_type], dbd_type

    # C2H2 subcategories by finger count
    if is_c2h2:
        n_fingers = count_c2h2_fingers(domains)
        if n_fingers <= 3:
            return 0, "C2H2_short"
        elif n_fingers <= 6:
            return 1, "C2H2_medium"
        else:
            return 2, "C2H2_long"

    return 9, "Other"


def load_progress(outdir: str) -> set[str]:
    path = os.path.join(outdir, "mapping_progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f).get("resolved_tfnames", []))
    return set()


def save_progress(outdir: str, resolved_tfnames: set[str]):
    path = os.path.join(outdir, "mapping_progress.json")
    with open(path, "w") as f:
        json.dump({"resolved_tfnames": sorted(resolved_tfnames)}, f)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    session = make_session()

    # Load parsed PWM data
    if not os.path.exists(args.pwm_parquet):
        logger.error(f"PWM parquet not found at {args.pwm_parquet}. Run parse_pwms.py first.")
        sys.exit(1)

    df = pd.read_parquet(args.pwm_parquet)
    logger.info(f"Loaded {len(df)} PWMs from {args.pwm_parquet}")

    # Get unique TF names
    unique_tfnames = sorted(df["tf_name"].unique())
    logger.info(f"Found {len(unique_tfnames)} unique TF names to resolve")

    # Resume support
    resolved_tfnames = load_progress(args.outdir) if args.resume else set()
    if resolved_tfnames:
        logger.info(f"Resuming: {len(resolved_tfnames)} TF names already resolved")

    # Resolve each unique TF name -> UniProt + sequence + InterPro domains
    tf_info = {}  # tf_name -> {uniprot_id, sequence, seq_length, organism, domains, ...}
    unresolved = []

    for i, tf_name in enumerate(unique_tfnames):
        if tf_name in resolved_tfnames and tf_name in tf_info:
            continue

        if tf_name in resolved_tfnames:
            # Previously resolved but tf_info empty (loaded from checkpoint only)
            # Need to re-fetch — skip for simplicity, re-resolve
            pass

        logger.info(f"  [{i+1}/{len(unique_tfnames)}] Resolving {tf_name}...")

        uniprot_info = resolve_uniprot(session, tf_name, args.species)

        if uniprot_info is None or not uniprot_info.get("sequence"):
            unresolved.append({
                "tf_name": tf_name,
                "reason": "No UniProt match found",
            })
            time.sleep(0.1)
            continue

        uniprot_id = uniprot_info["accession"]
        sequence = uniprot_info["sequence"]
        seq_length = uniprot_info["length"]
        organism = uniprot_info["organism"]

        # Fetch InterPro domains
        domains = fetch_interpro_domains(session, uniprot_id)
        time.sleep(0.1)

        # Extract DBD regions
        dbd_regions = extract_dbd_regions(domains, seq_length)
        if dbd_regions:
            dbd_start = dbd_regions[0][0]
            dbd_end = dbd_regions[-1][1]
            dbd_count = len(dbd_regions)
        else:
            dbd_start = seq_length // 3
            dbd_end = 2 * seq_length // 3
            dbd_count = 0

        # Assign family label
        family_id, family_name = assign_family_label(domains)

        tf_info[tf_name] = {
            "uniprot_id": uniprot_id,
            "gene_symbol": tf_name,
            "organism": organism,
            "sequence": sequence,
            "seq_length": seq_length,
            "dbd_start": dbd_start,
            "dbd_end": dbd_end,
            "dbd_count": dbd_count,
            "family_id": family_id,
            "family_name": family_name,
            "domains": domains,
        }
        resolved_tfnames.add(tf_name)

        # Checkpoint every 20 TFs
        if len(tf_info) % 20 == 0:
            save_progress(args.outdir, resolved_tfnames)
            logger.info(f"  Checkpoint: {len(tf_info)} resolved")

    logger.info(f"Resolved {len(tf_info)}/{len(unique_tfnames)} TF names")

    # Merge TF info with PWM records
    tf_records = []
    skipped_no_info = 0

    for _, row in df.iterrows():
        tf_name = row["tf_name"]
        if tf_name not in tf_info:
            skipped_no_info += 1
            continue

        info = tf_info[tf_name]
        pwm_bytes = row["pwm"]
        if not isinstance(pwm_bytes, bytes):
            continue

        tf_records.append({
            "tf_name": tf_name,
            "uniprot_id": info["uniprot_id"],
            "gene_symbol": info["gene_symbol"],
            "organism": info["organism"],
            "sequence": info["sequence"],
            "seq_length": info["seq_length"],
            "dbd_start": info["dbd_start"],
            "dbd_end": info["dbd_end"],
            "dbd_count": info["dbd_count"],
            "family_id": info["family_id"],
            "family_name": info["family_name"],
            "motif_length": row["motif_length"],
            "pwm": row["pwm"],
            "source": row["source"],
            "source_id": row["source_id"],
            "assay_type": row.get("assay_type", ""),
            "quality_grade": row.get("quality_grade", ""),
            "filename": row["filename"],
        })

    logger.info(f"Merged {len(tf_records)} PWM records, skipped {skipped_no_info} without sequence")

    # Save outputs
    if tf_records:
        tf_df = pd.DataFrame(tf_records)
        tf_df.to_parquet(os.path.join(args.outdir, "tf_pwm.parquet"), index=False)
        logger.info(f"Saved {len(tf_df)} records to tf_pwm.parquet")

        # Deduplicated sequences
        seq_df = tf_df[["uniprot_id", "gene_symbol", "organism", "sequence", "seq_length",
                         "dbd_start", "dbd_end", "dbd_count", "family_id", "family_name"]].copy()
        seq_df = seq_df.drop_duplicates(subset=["uniprot_id"]).reset_index(drop=True)

        # Add domains JSON
        domains_map = {name: info["domains"] for name, info in tf_info.items()}
        gene_to_name = tf_df.groupby("gene_symbol")["tf_name"].first().to_dict()
        seq_df["domains_json"] = seq_df["gene_symbol"].map(
            lambda g: json.dumps([
                {"interpro_id": d["interpro_id"], "name": d["name"],
                 "pfam_ids": d["pfam_ids"], "start": d["start"], "end": d["end"]}
                for d in domains_map.get(gene_to_name.get(g, ""), [])
            ])
        )
        seq_df.to_parquet(os.path.join(args.outdir, "tf_sequences.parquet"), index=False)
        logger.info(f"Saved {len(seq_df)} unique sequences to tf_sequences.parquet")

    if unresolved:
        unres_df = pd.DataFrame(unresolved)
        unres_df.to_parquet(os.path.join(args.outdir, "unresolved.parquet"), index=False)
        logger.info(f"Saved {len(unres_df)} unresolved TFs to unresolved.parquet")

    save_progress(args.outdir, resolved_tfnames)

    # Summary
    if tf_records:
        tf_df = pd.DataFrame(tf_records)
        logger.info(f"\n{'='*60}")
        logger.info(f"Mapping complete!")
        logger.info(f"  Total PWM records: {len(tf_df)}")
        logger.info(f"  Unique proteins: {tf_df['uniprot_id'].nunique()}")
        logger.info(f"  Unresolved TFs: {len(unresolved)}")
        logger.info(f"")
        logger.info(f"  Family distribution:")
        for fid in sorted(tf_df["family_id"].unique()):
            name = FAMILY_NAMES.get(fid, "Unknown")
            count = (tf_df["family_id"] == fid).sum()
            n_tfs = tf_df[tf_df["family_id"] == fid]["uniprot_id"].nunique()
            logger.info(f"    {fid}: {name:20s} -> {count:4d} PWMs, {n_tfs} TFs")
        logger.info(f"")
        logger.info(f"  By source:")
        for src in sorted(tf_df["source"].unique()):
            count = (tf_df["source"] == src).sum()
            logger.info(f"    {src:12s}: {count}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
