#!/usr/bin/env python
"""Look up protein information for a TF by gene name.

Usage:
    python scripts/lookup_tf.py RAX2
    python scripts/lookup_tf.py RAX2 --species "Mus musculus"
    python scripts/lookup_tf.py RAX2 --json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.map_tf_annotations import (
    make_session,
    resolve_uniprot,
    fetch_interpro_domains,
    extract_dbd_regions,
    assign_family_label,
)


def lookup_tf(name: str, species: str = "Homo sapiens") -> dict | None:
    session = make_session()

    uniprot = resolve_uniprot(session, name, species)
    if uniprot is None:
        return None

    uniprot_id = uniprot["accession"]
    domains = fetch_interpro_domains(session, uniprot_id)
    seq_length = uniprot["length"]
    dbd_regions = extract_dbd_regions(domains, seq_length)
    family_id, family_name = assign_family_label(domains)

    if dbd_regions:
        dbd_start = dbd_regions[0][0]
        dbd_end   = dbd_regions[-1][1]
        dbd_count = len(dbd_regions)
    else:
        dbd_start = seq_length // 3
        dbd_end   = 2 * seq_length // 3
        dbd_count = 0

    return {
        "query":       name,
        "uniprot_id":  uniprot_id,
        "gene_names":  uniprot["gene_names"],
        "organism":    uniprot["organism"],
        "sequence":    uniprot["sequence"],
        "seq_length":  seq_length,
        "dbd_start":   dbd_start,
        "dbd_end":     dbd_end,
        "dbd_count":   dbd_count,
        "family_id":   family_id,
        "family_name": family_name,
        "domains":     domains,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="TF gene name, e.g. RAX2")
    parser.add_argument("--species", default="Homo sapiens")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Print full JSON output")
    args = parser.parse_args()

    info = lookup_tf(args.name, args.species)
    if info is None:
        print(f"No UniProt entry found for '{args.name}'")
        sys.exit(1)

    if args.as_json:
        print(json.dumps(info, indent=2))
        return

    seq_preview = info["sequence"][:60] + ("..." if info["seq_length"] > 60 else "")
    dbd_region = f"{info['dbd_start']}–{info['dbd_end']}"
    if info["dbd_count"] == 0:
        dbd_region += "  (estimated — no DBD domains found)"

    print(f"Query        : {info['query']}")
    print(f"UniProt ID   : {info['uniprot_id']}")
    print(f"Gene names   : {', '.join(info['gene_names']) or 'N/A'}")
    print(f"Organism     : {info['organism']}")
    print(f"Length       : {info['seq_length']} aa")
    print(f"Sequence     : {seq_preview}")
    print(f"Family       : {info['family_id']} ({info['family_name']})")
    print(f"DBD region   : {dbd_region}  ({info['dbd_count']} domain(s))")
    if info["domains"]:
        print("Domains      :")
        for d in info["domains"]:
            pfam = ", ".join(d["pfam_ids"]) or "—"
            print(f"  [{d['interpro_id']}] {d['name']}  pfam={pfam}  {d['start']}–{d['end']}")


if __name__ == "__main__":
    main()
