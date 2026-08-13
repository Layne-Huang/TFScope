#!/usr/bin/env python
"""Stage 1: fetch raw InterPro DBD fragment coordinates for every aug-set gene
that could need cluster-aware cropping (dbd_count>=2, plus the held
dbd_count=0 cases). Writes one JSON line per gene, flushed immediately, so an
AFS-token drop only costs the current gene -- re-running resumes.

Deliberately fetch-only: clustering/cropping happens in a separate local
script so the gap threshold can be retuned without re-hitting the API.
"""
import json, os, sys, time
import requests
import pandas as pd

sys.path.insert(0, "scripts")
from map_tf_annotations import fetch_interpro_domains
from reextract_dbd_round3 import ROUND3_PFAM, ROUND3_INTERPRO

CKPT = "/tmp/domain_fragments_v2.jsonl"

# families added in round 4, plus round-3 set
EXTRA_PFAM = {
    "PF03859":"CG1","PF05044":"Prospero","PF04500":"FLYWCH","PF02946":"GTF2I",
    "PF02008":"CXXC","PF00642":"CCCH","PF13837":"Myb_SANT_like",
    "PF13873":"Myb_SANT_like","PF05224":"NDT80","PF10523":"BEN",
    "PF02178":"AT_hook","PF02437":"SKI_DAC",
}
EXTRA_IPR = {
    "IPR005559":"CG1","IPR023082":"Prospero","IPR007588":"FLYWCH",
    "IPR004212":"GTF2I","IPR002857":"CXXC","IPR000571":"CCCH",
    "IPR044822":"Myb_SANT_like","IPR028002":"Myb_SANT_like",
    "IPR024061":"NDT80","IPR018379":"BEN","IPR017956":"AT_hook",
    "IPR003380":"SKI_DAC",
}
PFAM = {**ROUND3_PFAM, **EXTRA_PFAM}
IPR = {**ROUND3_INTERPRO, **EXTRA_IPR}

# genes whose stored accession points at the WRONG protein (confirmed earlier)
FIX_UID = {
    "MYT1": "Q01538",   # was PKMYT1 (a kinase)
    "EVI1": "Q03112",   # was CCR7 (a chemokine receptor); correct gene = MECOM
    "ZEP1": "P15822",   # was a rice protein; correct gene = HIVEP1
}


def fetch_seq(session, uid):
    r = session.get(f"https://rest.uniprot.org/uniprotkb/{uid}.json",
                     params={"fields": "sequence"}, timeout=25)
    r.raise_for_status()
    return r.json()["sequence"]["value"]


def main():
    aug = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet")
    need = aug[(aug["dbd_count"] >= 2) | (aug["dbd_count"] == 0)]
    genes = need[["gene_symbol", "uniprot_id"]].drop_duplicates(subset=["gene_symbol"])
    print(f"genes to fetch: {len(genes)}", flush=True)

    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT):
            try: done.add(json.loads(line)["gene"])
            except Exception: pass
    todo = [(g, u) for g, u in genes.values if g not in done]
    print(f"already done: {len(done)}   remaining: {len(todo)}", flush=True)

    s = requests.Session()
    t0, n_ok, n_err = time.time(), 0, 0
    with open(CKPT, "a") as f:
        for i, (gene, uid) in enumerate(todo):
            uid = FIX_UID.get(gene, uid)
            try:
                seq = fetch_seq(s, uid)
                domains = fetch_interpro_domains(s, uid)
                frags, fams = [], set()
                for d in domains:
                    hit = None
                    for p in d.get("pfam_ids", []):
                        if p in PFAM: hit = PFAM[p]
                    if d.get("interpro_id", "") in IPR:
                        hit = IPR[d["interpro_id"]]
                    if hit:
                        a, b = d["start"] - 1, d["end"]
                        if 0 <= a < b <= len(seq):
                            # store the family PER FRAGMENT: mixed-family genes
                            # (ZFHX3 C2H2+Homeodomain, TRPS1 C2H2+GATA, CREB5
                            # C2H2+bZIP) need it to pick the right domain type
                            frags.append([a, b, hit]); fams.add(hit)
                f.write(json.dumps({"gene": gene, "uniprot_id": uid,
                                     "seq": seq, "frags": sorted(frags),
                                     "families": sorted(fams)}) + "\n")
                f.flush(); n_ok += 1
            except Exception as e:
                f.write(json.dumps({"gene": gene, "uniprot_id": uid, "error": str(e)}) + "\n")
                f.flush(); n_err += 1
            time.sleep(0.15)
            if (i + 1) % 25 == 0 or (i + 1) == len(todo):
                el = time.time() - t0
                rate = el / (i + 1)
                print(f"  [{i+1}/{len(todo)}] ok={n_ok} err={n_err} "
                      f"elapsed={el:.0f}s est_remaining={(len(todo)-i-1)*rate:.0f}s", flush=True)
    print(f"DONE. ok={n_ok} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
