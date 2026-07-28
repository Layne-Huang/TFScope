#!/usr/bin/env python
"""Round-3 DBD extraction: adds the families found by inspecting the raw
InterPro scan of the still-unresolved genes. All Pfam/InterPro IDs below were
verified live against the InterPro API (not recalled from memory).

Deliberately EXCLUDED (chromatin readers / non-sequence-specific / unrelated):
  PHD fingers, bromodomains, JmjC, RNase-H-like, IPT/Ig-fold, ankyrin repeats.
"""
import sys, time, json
import requests
import pandas as pd

sys.path.insert(0, "scripts")
from map_tf_annotations import fetch_interpro_domains

ROUND3_PFAM = {
    # original 7 families
    "PF00096": "C2H2", "PF00010": "bHLH", "PF07916": "bHLH",
    "PF00046": "Homeodomain", "PF00170": "bZIP", "PF07716": "bZIP",
    "PF00105": "Nuclear_Receptor", "PF00104": "Nuclear_Receptor",
    "PF00250": "Forkhead", "PF00178": "ETS",
    # round 2
    "PF00554": "Rel_homology", "PF02864": "STAT", "PF00319": "MADS_box",
    "PF00505": "HMG_box", "PF09011": "HMG_box", "PF00320": "GATA",
    "PF00907": "T_box", "PF00605": "IRF", "PF00870": "p53",
    "PF00853": "Runt", "PF03165": "SMAD", "PF00292": "Paired_domain",
    "PF01285": "TEA",
    # round 3 (all verified live)
    "PF01388": "ARID", "PF03299": "AP2_TF", "PF00447": "HSF",
    "PF02892": "BED_zf", "PF26664": "BED_zf", "PF03615": "GCM",
    "PF11680": "PUR", "PF04845": "PUR", "PF00249": "Myb_SANT",
    "PF23082": "Myb_SANT", "PF02319": "E2F_DP", "PF16421": "E2F_DP",
    "PF00751": "DM_domain", "PF02257": "RFX", "PF25416": "CP2_GRHL",
    "PF03221": "CenpB_HTH", "PF04218": "Psq_HTH", "PF05225": "Psq_HTH",
    "PF01342": "SAND", "PF01530": "C2H2C_type", "PF16422": "COE_EBF",
    "PF16423": "COE_EBF", "PF05485": "THAP",
}
ROUND3_INTERPRO = {
    "IPR001878": "C2H2", "IPR011598": "bHLH", "IPR001356": "Homeodomain",
    "IPR004827": "bZIP", "IPR000536": "Nuclear_Receptor",
    "IPR001766": "Forkhead", "IPR000418": "ETS",
    "IPR011539": "Rel_homology", "IPR013801": "STAT", "IPR002100": "MADS_box",
    "IPR009071": "HMG_box", "IPR000679": "GATA", "IPR046360": "T_box",
    "IPR001346": "IRF", "IPR011615": "p53", "IPR013524": "Runt",
    "IPR003619": "SMAD", "IPR001523": "Paired_domain", "IPR000818": "TEA",
    "IPR001606": "ARID", "IPR004979": "AP2_TF", "IPR013854": "AP2_TF",
    "IPR000232": "HSF", "IPR003656": "BED_zf", "IPR003902": "GCM",
    "IPR006628": "PUR", "IPR001005": "Myb_SANT", "IPR017930": "Myb_SANT",
    "IPR003316": "E2F_DP", "IPR015633": "E2F_DP", "IPR001275": "DM_domain",
    "IPR003150": "RFX", "IPR057520": "CP2_GRHL", "IPR006600": "CenpB_HTH",
    "IPR007889": "Psq_HTH", "IPR000770": "SAND", "IPR002515": "C2H2C_type",
    "IPR032200": "COE_EBF", "IPR032201": "COE_EBF", "IPR006612": "THAP",
    # coarse C2H2 superfamily -- catches ZNF487/ZNF788-style genes annotated
    # only at superfamily level, which the precise-family-only match missed
    "IPR036236": "C2H2",
}


def fetch_full_sequence(session, uid):
    r = session.get(f"https://rest.uniprot.org/uniprotkb/{uid}.json",
                     params={"fields": "sequence"}, timeout=20)
    r.raise_for_status()
    return r.json()["sequence"]["value"]


def extract(domains, seq_length):
    hits, fams = [], set()
    for d in domains:
        is_dbd = False
        for p in d.get("pfam_ids", []):
            if p in ROUND3_PFAM:
                is_dbd = True; fams.add(ROUND3_PFAM[p])
        if d.get("interpro_id", "") in ROUND3_INTERPRO:
            is_dbd = True; fams.add(ROUND3_INTERPRO[d["interpro_id"]])
        if is_dbd:
            s, e = d["start"] - 1, d["end"]
            if 0 <= s < e <= seq_length:
                hits.append((s, e))
    if not hits:
        return [], fams
    hits.sort()
    merged = [hits[0]]
    for s, e in hits[1:]:
        if s <= merged[-1][1] + 10:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged, fams


if __name__ == "__main__":
    aug = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet")
    todo = aug[aug["dbd_count"] == 0][["gene_symbol", "uniprot_id"]].drop_duplicates(subset=["gene_symbol"])
    print(f"genes to re-check: {len(todo)}", flush=True)

    session = requests.Session()
    results, n_ok = {}, 0
    for i, (_, row) in enumerate(todo.iterrows()):
        gene, uid = row["gene_symbol"], row["uniprot_id"]
        try:
            seq = fetch_full_sequence(session, uid)
            domains = fetch_interpro_domains(session, uid)
            merged, fams = extract(domains, len(seq))
        except Exception as e:
            results[gene] = {"error": str(e)}
            continue
        if merged:
            n_ok += 1
            s, e = merged[0][0], merged[-1][1]
            gaps = [merged[k+1][0] - merged[k][1] for k in range(len(merged)-1)]
            results[gene] = {"resolved": True, "uniprot_id": uid, "start": s, "end": e,
                              "new_len": e - s, "n_domains": len(merged), "gaps": gaps,
                              "families": sorted(fams), "crop_seq": seq[s:e],
                              "real_full_len": len(seq)}
        else:
            results[gene] = {"resolved": False}
        time.sleep(0.2)
        if (i+1) % 25 == 0 or (i+1) == len(todo):
            print(f"  [{i+1}/{len(todo)}] resolved={n_ok}", flush=True)

    print(f"\nDONE. {n_ok}/{len(todo)} resolved.", flush=True)
    json.dump(results, open("/tmp/dbd_round3_results.json", "w"))
    print("saved /tmp/dbd_round3_results.json", flush=True)
