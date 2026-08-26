#!/usr/bin/env python
"""v26 Phase-1: fetch a FROZEN annotation snapshot (UniProt + InterPro + SIFTS).

Retires audit Finding L (live API calls, no release pin). Every downstream v26 script reads
this snapshot and never touches the network.

Stores RAW responses so the snapshot can be re-parsed without re-fetching:
  data/annotations_v26/uniprot.jsonl.gz     {acc: <full UniProtKB entry>}
  data/annotations_v26/interpro.jsonl.gz    {acc: <full InterPro entry-all response>}
  data/annotations_v26/sifts.jsonl.gz       {pdb_id: <PDBe SIFTS uniprot mapping>}
  data/annotations_v26/gene_resolution.jsonl.gz  {gene: <UniProt search response>}
  data/annotations_v26/SNAPSHOT.md          release strings + access date + counts

RESUMABLE: re-running skips keys already present in each .jsonl.gz. Safe to kill and relaunch.
Progress is printed every PROGRESS_EVERY items (elapsed, rate, ETA, failures).

  python scripts/v26/fetch_annotation_snapshot.py                # all sources
  python scripts/v26/fetch_annotation_snapshot.py --only uniprot
  python scripts/v26/fetch_annotation_snapshot.py --limit 20     # smoke test
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT = "data/annotations_v26"
V23 = "data/processed/tf_pwm_training_v23.parquet"
ORIG = "data/processed/tf_pwm.parquet"
STRUCT = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"

UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{acc}.json"
UNIPROT_SEARCH = ("https://rest.uniprot.org/uniprotkb/search"
                  "?query=gene_exact:{gene}&format=json&size=5")
INTERPRO = ("https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{acc}/"
            "?page_size=200")
SIFTS = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"

PROGRESS_EVERY = 25
SLEEP = 0.12          # politeness delay between requests


# ------------------------------------------------------------------ io helpers
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=0.6,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=8))
    s.headers.update({"User-Agent": "TFScope-v26-snapshot (leihuang@csail.mit.edu)"})
    return s


def _done_keys(path: str) -> set:
    """Keys already fetched. Tolerates a truncated final line from a killed run."""
    if not os.path.exists(path):
        return set()
    keys = set()
    with gzip.open(path, "rt") as fh:
        for line in fh:
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                continue
    return keys


def _append(path: str, key: str, payload, status: int):
    with gzip.open(path, "at") as fh:
        fh.write(json.dumps({"key": key, "status": status, "payload": payload}) + "\n")


def _fetch_many(sess, name, keys, url_tmpl, path, limit=None):
    """Generic resumable fetch loop with progress counter. Returns (n_ok, n_fail)."""
    done = _done_keys(path)
    todo = [k for k in keys if k not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{name}] {len(keys)} total, {len(done)} cached, {len(todo)} to fetch", flush=True)
    t0 = time.time()
    n_ok = n_fail = 0
    for i, k in enumerate(todo, 1):
        try:
            r = sess.get(url_tmpl.format(**{name_key(name): k}), timeout=45)
            if r.status_code == 200:
                _append(path, k, r.json(), 200)
                n_ok += 1
            else:
                _append(path, k, None, r.status_code)
                n_fail += 1
        except Exception as e:                                  # noqa: BLE001
            _append(path, k, {"error": str(e)[:300]}, -1)
            n_fail += 1
        if i % PROGRESS_EVERY == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / max(el, 1e-6)
            eta = (len(todo) - i) / max(rate, 1e-6)
            print(f"[{name}] {i}/{len(todo)}  ok={n_ok} fail={n_fail}  "
                  f"{rate:.1f}/s  elapsed={el/60:.1f}m  eta={eta/60:.1f}m", flush=True)
        time.sleep(SLEEP)
    return n_ok, n_fail


def name_key(name: str) -> str:
    return {"uniprot": "acc", "interpro": "acc", "sifts": "pdb", "gene": "gene"}[name]


# ------------------------------------------------------------------- key lists
def extra_accessions() -> list[str]:
    """Round-2 seed: accessions discovered by row resolution that the round-1 seed missed.

    Round 1 seeded from tf_pwm.parquet's gene->uniprot map, but SIFTS resolves structure
    chains to accessions outside that map (orthologs, isoforms, alternate entries), leaving
    163 accessions / 496 rows with no UniProt sequence or InterPro annotation.
    """
    p = "data/processed/v26/row_resolution.parquet"
    if not os.path.exists(p):
        return []
    return sorted({str(a) for a in pd.read_parquet(p).primary_accession.dropna()})


def sifts_accessions() -> list[str]:
    """Round-3 seed: every accession SIFTS assigns to any chain of a parsed structure.

    Round 2 seeded only from row resolution, i.e. PRIMARY chains. Partner chains in the same
    co-crystals map to accessions outside that set, leaving 766 chains / 10,644 contacts with no
    UniProt coordinate. Partner contacts are needed for partner-residue supervision and for the
    Phase-3 Assembly-OOD audit.
    """
    p = "data/processed/v26/sifts_mappings.parquet"
    if not os.path.exists(p):
        return []
    return sorted({str(a) for a in pd.read_parquet(p).accession.dropna()})


def collect_keys():
    v23 = pd.read_parquet(V23)
    orig = pd.read_parquet(ORIG)[["gene_symbol", "uniprot_id"]].dropna()
    orig["G"] = orig.gene_symbol.astype(str).str.upper()

    v23_genes = sorted({str(g).upper() for g in v23.gene_symbol})
    gene2acc = {}
    for r in orig.itertuples():
        gene2acc.setdefault(r.G, str(r.uniprot_id))

    accs = sorted({a for g, a in gene2acc.items() if g in set(v23_genes)})
    missing_genes = sorted(set(v23_genes) - set(gene2acc))

    pdbs = sorted({str(p).upper() for p in pd.read_parquet(STRUCT).pdb_id.dropna()})
    return accs, missing_genes, pdbs, gene2acc


# ---------------------------------------------------------------- release info
def capture_release(sess) -> dict:
    rel = {"access_date_utc": datetime.now(timezone.utc).isoformat()}
    try:
        r = sess.get(UNIPROT_ENTRY.format(acc="P01106"), timeout=30)
        rel["uniprot_release"] = r.headers.get("X-UniProt-Release", "unknown")
        rel["uniprot_release_date"] = r.headers.get("X-UniProt-Release-Date", "unknown")
    except Exception as e:                                      # noqa: BLE001
        rel["uniprot_release"] = f"error:{e}"
    for url, key in [("https://www.ebi.ac.uk/interpro/api/utils/release/", "interpro_release"),
                     ("https://www.ebi.ac.uk/pdbe/api/pdb/status/", "pdbe_status")]:
        try:
            r = sess.get(url, timeout=30)
            rel[key] = r.json() if r.status_code == 200 else f"http:{r.status_code}"
        except Exception as e:                                  # noqa: BLE001
            rel[key] = f"error:{str(e)[:120]}"
    return rel


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["uniprot", "interpro", "sifts", "gene"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--from-sifts", action="store_true",
                    help="also fetch every accession SIFTS assigns to a parsed chain (partners)")
    ap.add_argument("--round2", action="store_true",
                    help="also fetch accessions discovered by row resolution (see extra_accessions)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    sess = _session()
    accs, missing_genes, pdbs, gene2acc = collect_keys()
    if a.round2:
        extra = extra_accessions()
        new = sorted(set(extra) - set(accs))
        print(f"round2: +{len(new)} accessions discovered by row resolution", flush=True)
        accs = sorted(set(accs) | set(new))
    if a.from_sifts:
        extra = sifts_accessions()
        new = sorted(set(extra) - set(accs))
        print(f"from-sifts: +{len(new)} partner/other chain accessions", flush=True)
        accs = sorted(set(accs) | set(new))
    print(f"targets: {len(accs)} accessions | {len(missing_genes)} unresolved genes | "
          f"{len(pdbs)} PDB ids", flush=True)

    rel = capture_release(sess)
    print(f"release: uniprot={rel.get('uniprot_release')} "
          f"({rel.get('uniprot_release_date')})", flush=True)

    stats = {}
    if a.only in (None, "gene"):
        stats["gene"] = _fetch_many(sess, "gene", missing_genes, UNIPROT_SEARCH,
                                    f"{OUT}/gene_resolution.jsonl.gz", a.limit)
    if a.only in (None, "uniprot"):
        stats["uniprot"] = _fetch_many(sess, "uniprot", accs, UNIPROT_ENTRY,
                                       f"{OUT}/uniprot.jsonl.gz", a.limit)
    if a.only in (None, "interpro"):
        stats["interpro"] = _fetch_many(sess, "interpro", accs, INTERPRO,
                                        f"{OUT}/interpro.jsonl.gz", a.limit)
    if a.only in (None, "sifts"):
        stats["sifts"] = _fetch_many(sess, "sifts", pdbs, SIFTS,
                                     f"{OUT}/sifts.jsonl.gz", a.limit)

    with open(f"{OUT}/gene_to_accession_seed.json", "w") as fh:
        json.dump(gene2acc, fh, indent=1)
    with open(f"{OUT}/release.json", "w") as fh:
        json.dump(rel, fh, indent=2)

    lines = [
        "# v26 annotation snapshot", "",
        f"- access date (UTC): `{rel['access_date_utc']}`",
        f"- UniProt release: `{rel.get('uniprot_release')}` "
        f"(`{rel.get('uniprot_release_date')}`)",
        f"- InterPro release endpoint: `{json.dumps(rel.get('interpro_release'))[:200]}`",
        "", "## Counts", "",
        f"- accessions requested: {len(accs)}",
        f"- genes needing resolution: {len(missing_genes)}",
        f"- PDB ids (SIFTS): {len(pdbs)}", "",
        "## Fetch results (ok / fail)", "",
    ]
    for k, (ok, fail) in stats.items():
        lines.append(f"- {k}: {ok} ok, {fail} failed")
    lines += ["", "Raw responses are stored verbatim in the `*.jsonl.gz` files "
              "(`{key, status, payload}` per line). Re-running this script only fetches "
              "missing keys.", ""]
    with open(f"{OUT}/SNAPSHOT.md", "w") as fh:
        fh.write("\n".join(lines))

    print("\n=== snapshot complete ===")
    for k, (ok, fail) in stats.items():
        print(f"  {k}: ok={ok} fail={fail}")
    print(f"  wrote {OUT}/")


if __name__ == "__main__":
    sys.exit(main())
