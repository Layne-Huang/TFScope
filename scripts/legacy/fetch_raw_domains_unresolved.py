#!/usr/bin/env python
"""Fetch and save the RAW InterPro domain scan output for the 100 genes that
didn't resolve to any recognized DBD family, even under the expanded
whitelist -- so we can inspect what's actually annotated for them."""
import sys, time, json
import requests
import pandas as pd

sys.path.insert(0, "scripts")
from map_tf_annotations import fetch_interpro_domains

unresolved = json.load(open("/tmp/unresolved_genes.json"))
aug_v2 = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet")
gene_to_uid = dict(zip(aug_v2["gene_symbol"], aug_v2["uniprot_id"]))

session = requests.Session()
raw_results = {}
for i, gene in enumerate(unresolved):
    uid = gene_to_uid.get(gene)
    try:
        domains = fetch_interpro_domains(session, uid)
        # dedupe by interpro_id, keep name + pfam_ids only (drop repeated fragments)
        seen = {}
        for d in domains:
            seen[d["interpro_id"]] = {"name": d["name"], "pfam_ids": d["pfam_ids"]}
        raw_results[gene] = {"uniprot_id": uid, "domains": seen}
    except Exception as e:
        raw_results[gene] = {"uniprot_id": uid, "error": str(e)}
    time.sleep(0.15)
    if (i + 1) % 25 == 0 or (i + 1) == len(unresolved):
        print(f"  [{i+1}/{len(unresolved)}]", flush=True)

with open("/tmp/unresolved_raw_domains.json", "w") as f:
    json.dump(raw_results, f, indent=1)
print("saved to /tmp/unresolved_raw_domains.json", flush=True)
