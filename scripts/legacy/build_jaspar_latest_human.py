#!/usr/bin/env python
"""Fetch the CURRENT JASPAR CORE vertebrate matrices (all latest versions).

Our stored JASPAR records are badly stale: 463/571 matrix IDs (81%) have a
newer version available, some 3 versions behind (e.g. GATA2 MA0036.1 -> .4).
This pulls the current CORE vertebrates collection and keeps human entries.
"""
import json, time
import numpy as np
import pandas as pd
import requests

OUT = "data/processed/jaspar_latest_human_pwms.parquet"
MIN_LEN = 5


def fetch_all_core_vertebrates(session):
    """Return {base_id: (version, matrix_id, name, species)} for latest versions."""
    latest, url, page = {}, ("https://jaspar.elixir.no/api/v1/matrix/"
                             "?collection=CORE&tax_group=vertebrates&page_size=1000&format=json"), 0
    while url and page < 20:
        r = session.get(url, timeout=90); r.raise_for_status(); d = r.json()
        for m in d.get("results", []):
            mid = m["matrix_id"]
            base, ver = mid.rsplit(".", 1)
            ver = int(ver)
            if base not in latest or ver > latest[base][0]:
                latest[base] = (ver, mid, m.get("name"), m.get("species"))
        url = d.get("next"); page += 1
        print(f"  page {page}: {len(latest)} base ids so far", flush=True)
    return latest


def fetch_matrix(session, matrix_id):
    r = session.get(f"https://jaspar.elixir.no/api/v1/matrix/{matrix_id}/?format=json", timeout=45)
    r.raise_for_status()
    return r.json()


def main():
    s = requests.Session()
    latest = fetch_all_core_vertebrates(s)
    print(f"latest CORE vertebrate base ids: {len(latest)}", flush=True)

    rows, n_skip, n_err = [], 0, 0
    items = sorted(latest.items())
    for i, (base, (ver, mid, name, species)) in enumerate(items):
        try:
            d = fetch_matrix(s, mid)
        except Exception:
            n_err += 1
            continue
        # human only (tax id 9606)
        sp = d.get("species") or []
        if not any(str(x.get("tax_id")) == "9606" for x in sp):
            n_skip += 1
            continue
        pfm = d.get("pfm")
        if not pfm or not all(k in pfm for k in "ACGT"):
            n_skip += 1
            continue
        mat = np.array([pfm["A"], pfm["C"], pfm["G"], pfm["T"]], dtype=np.float32)
        if mat.shape[1] < MIN_LEN:
            n_skip += 1
            continue
        cs = mat.sum(axis=0, keepdims=True)
        if (cs <= 0).any():
            n_skip += 1
            continue
        mat = mat / cs
        uniprot = d.get("uniprot_ids") or []
        rows.append({
            "gene_symbol": d.get("name"),
            "motif_id": mid,
            "uniprot_id": uniprot[0] if uniprot else "",
            "motif_length": mat.shape[1],
            "pwm": mat.astype(np.float32).tobytes(),
            "source": "JASPAR_latest",
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(items)}] kept={len(rows)} skipped={n_skip} err={n_err}", flush=True)
        time.sleep(0.05)

    out = pd.DataFrame(rows)
    print(f"\nkept {len(out)} human matrices (skipped non-human/bad={n_skip}, errors={n_err})", flush=True)
    print(f"distinct genes: {out['gene_symbol'].str.upper().nunique()}", flush=True)
    out.to_parquet(OUT)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
