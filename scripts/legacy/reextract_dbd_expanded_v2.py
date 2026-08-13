#!/usr/bin/env python
"""Fixed version: fetches the REAL full-length UniProt sequence (by accession,
since we already have it) before computing/clipping domain coordinates --
the previous attempt incorrectly reused our own already-truncated stored
seq_length (the old middle-third fallback crop), corrupting the clip and
producing nonsensical start>end spans for ~1/3 of resolved genes.
"""
import sys, time, json
import requests
import pandas as pd

sys.path.insert(0, "scripts")
from map_tf_annotations import fetch_interpro_domains

EXPANDED_PFAM = {
    "PF00096": "C2H2", "PF00010": "bHLH", "PF07916": "bHLH",
    "PF00046": "Homeodomain", "PF00170": "bZIP", "PF07716": "bZIP",
    "PF00105": "Nuclear_Receptor", "PF00104": "Nuclear_Receptor",
    "PF00250": "Forkhead", "PF00178": "ETS",
    "PF00554": "Rel_homology", "PF02864": "STAT", "PF00319": "MADS_box",
    "PF00505": "HMG_box", "PF09011": "HMG_box", "PF00320": "GATA",
    "PF00907": "T_box", "PF00605": "IRF", "PF00870": "p53",
    "PF00853": "Runt", "PF03165": "SMAD", "PF00292": "Paired_domain",
    "PF01285": "TEA",
}
EXPANDED_INTERPRO = {
    "IPR001878": "C2H2", "IPR011598": "bHLH", "IPR001356": "Homeodomain",
    "IPR004827": "bZIP", "IPR000536": "Nuclear_Receptor",
    "IPR001766": "Forkhead", "IPR000418": "ETS",
    "IPR011539": "Rel_homology", "IPR013801": "STAT", "IPR002100": "MADS_box",
    "IPR009071": "HMG_box", "IPR000679": "GATA", "IPR046360": "T_box",
    "IPR001346": "IRF", "IPR011615": "p53", "IPR013524": "Runt",
    "IPR003619": "SMAD", "IPR001523": "Paired_domain", "IPR000818": "TEA",
}


def fetch_full_sequence(session, uniprot_id):
    resp = session.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json",
                        params={"fields": "sequence"}, timeout=20)
    resp.raise_for_status()
    return resp.json()["sequence"]["value"]


def extract_dbd_regions_expanded(domains, seq_length):
    dbd_domains, hit_families = [], set()
    for d in domains:
        is_dbd = False
        for pfam_id in d.get("pfam_ids", []):
            if pfam_id in EXPANDED_PFAM:
                is_dbd = True
                hit_families.add(EXPANDED_PFAM[pfam_id])
        if d.get("interpro_id", "") in EXPANDED_INTERPRO:
            is_dbd = True
            hit_families.add(EXPANDED_INTERPRO[d["interpro_id"]])
        if is_dbd:
            s, e = d["start"] - 1, d["end"]
            if 0 <= s < e <= seq_length:
                dbd_domains.append((s, e))
    if not dbd_domains:
        return [], hit_families
    dbd_domains.sort()
    merged = [dbd_domains[0]]
    for start, end in dbd_domains[1:]:
        if start <= merged[-1][1] + 10:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged, hit_families


aug_v2 = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet")
no_domain = aug_v2[aug_v2["dbd_count"] == 0]
unique_genes = no_domain[["gene_symbol", "uniprot_id"]].drop_duplicates(subset=["gene_symbol"])
print(f"distinct genes to re-check: {len(unique_genes)}", flush=True)

session = requests.Session()
results = {}
n_resolved = 0
for i, (_, row) in enumerate(unique_genes.iterrows()):
    gene, uid = row["gene_symbol"], row["uniprot_id"]
    try:
        full_seq = fetch_full_sequence(session, uid)
        real_len = len(full_seq)
        domains = fetch_interpro_domains(session, uid)
        merged, hit_families = extract_dbd_regions_expanded(domains, real_len)
    except Exception as e:
        results[gene] = {"error": str(e)}
        continue
    if merged:
        n_resolved += 1
        span_start, span_end = merged[0][0], merged[-1][1]
        results[gene] = {"resolved": True, "n_domains": len(merged),
                          "start": span_start, "end": span_end,
                          "real_full_len": real_len,
                          "new_len": span_end - span_start,
                          "families": sorted(hit_families),
                          "crop_seq": full_seq[span_start:span_end]}
    else:
        results[gene] = {"resolved": False, "real_full_len": real_len}
    time.sleep(0.2)
    if (i + 1) % 25 == 0 or (i + 1) == len(unique_genes):
        print(f"  [{i+1}/{len(unique_genes)}] resolved so far: {n_resolved}", flush=True)

print(f"\nDONE. {n_resolved}/{len(unique_genes)} genes now resolve to a real domain-based crop.", flush=True)
with open("/tmp/dbd_reextraction_v2_results.json", "w") as f:
    json.dump(results, f)
print("saved to /tmp/dbd_reextraction_v2_results.json", flush=True)
