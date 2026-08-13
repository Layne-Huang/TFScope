#!/usr/bin/env python
"""Add the genuinely-new CIS-BP v3.10 human genes to the sequence-only set.

Of 65 CIS-BP genes absent from both datasets:
  13 we DELIBERATELY dropped earlier (no DBD known in either database) -> stay out
   8 have DBDs=Unknown in CIS-BP -> fail our own criteria -> stay out
   4 are aliases/pseudogenes/non-genes -> excluded (see EXCLUDE below)
  40 are genuinely new with a curated DBD -> added here

Each new gene goes through exactly the same pipeline as the rest of the set:
UniProt sequence -> InterPro DBD fragments (expanded whitelist) -> cluster-aware
crop -> CIS-BP v3.10 PWM.
"""
import json, sys, time
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, "scripts")
from map_tf_annotations import fetch_interpro_domains
from reextract_dbd_round3 import ROUND3_PFAM, ROUND3_INTERPRO
from cluster_crop_v2 import cluster, choose, db_family_tokens, GAP_THRESHOLD

EXTRA_PFAM = {"PF03859":"CG1","PF05044":"Prospero","PF04500":"FLYWCH","PF02946":"GTF2I",
              "PF02008":"CXXC","PF00642":"CCCH","PF13837":"Myb_SANT_like",
              "PF13873":"Myb_SANT_like","PF05224":"NDT80","PF10523":"BEN",
              "PF02178":"AT_hook","PF02437":"SKI_DAC","PF03184":"CenpB_HTH",
              "PF00808":"CENPB_N","PF04851":"NFX","PF00313":"CSD","PF01388":"ARID"}
EXTRA_IPR = {"IPR005559":"CG1","IPR023082":"Prospero","IPR007588":"FLYWCH",
             "IPR004212":"GTF2I","IPR002857":"CXXC","IPR000571":"CCCH",
             "IPR044822":"Myb_SANT_like","IPR028002":"Myb_SANT_like",
             "IPR024061":"NDT80","IPR018379":"BEN","IPR017956":"AT_hook",
             "IPR003380":"SKI_DAC","IPR006600":"CenpB_HTH","IPR002514":"CENPB_N",
             "IPR000058":"NFX","IPR002059":"CSD","IPR001606":"ARID"}
PFAM = {**ROUND3_PFAM, **EXTRA_PFAM}
IPR = {**ROUND3_INTERPRO, **EXTRA_IPR}

# alias duplicates / pseudogenes / non-genes -- verified against UniProt
EXCLUDE = {
    "ZFTA":      "alias of C11orf95, already in dataset",
    "ZNF705EP":  "pseudogene; alias ZNF705E already in dataset",
    "ZNF788P":   "pseudogene (82aa); parent ZNF788 previously dropped",
    "AC092835":  "no UniProt entry (clone identifier, not a gene)",
}


def uniprot_lookup(session, gene):
    r = session.get("https://rest.uniprot.org/uniprotkb/search",
                     params={"query": f"(gene:{gene}) AND (organism_id:9606) AND (reviewed:true)",
                             "fields": "accession,sequence", "format": "json", "size": 1}, timeout=25)
    res = r.json().get("results", [])
    if not res:
        r = session.get("https://rest.uniprot.org/uniprotkb/search",
                         params={"query": f"(gene:{gene}) AND (organism_id:9606)",
                                 "fields": "accession,sequence", "format": "json", "size": 1}, timeout=25)
        res = r.json().get("results", [])
    if not res:
        return None, None
    return res[0]["primaryAccession"], res[0]["sequence"]["value"]


def main():
    cis = pd.read_parquet("data/processed/cisbp_v310_human_pwms.parquet")
    aug = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet")
    st = pd.read_parquet("data/processed/tf_pwm_deeppbs_v2_deduped.parquet")

    DROPPED = {"AHCTF1","BPTF","CEBPZ","CGGBP1","CPEB1","DACH1","DACH2","DNTTIP1",
               "GLYR1","LRRFIP1","NFKB","PHA2","PHF21A","SKOR1","SPZ1","TCF20",
               "FAM200B","ZNF788"}
    have = set(aug["gene_symbol"].str.upper()) | set(st["gene"].str.upper())
    cand = cis[~cis["gene_symbol"].str.upper().isin(have | DROPPED | set(EXCLUDE))]
    cand = cand[~cand["dbds_cisbp"].astype(str).str.upper().isin(["UNKNOWN", "NAN"])]
    genes = sorted(cand["gene_symbol"].str.upper().unique())
    print(f"genes to add: {len(genes)}", flush=True)

    s = requests.Session()
    rows, failed = [], []
    for i, g in enumerate(genes):
        sub = cand[cand["gene_symbol"].str.upper() == g]
        try:
            uid, seq = uniprot_lookup(s, g)
            if not seq:
                failed.append((g, "no uniprot")); continue
            doms = fetch_interpro_domains(s, uid)
            frags, fams = [], set()
            for d in doms:
                hit = None
                for p in d.get("pfam_ids", []):
                    if p in PFAM: hit = PFAM[p]
                if d.get("interpro_id", "") in IPR:
                    hit = IPR[d["interpro_id"]]
                if hit:
                    a, b = d["start"] - 1, d["end"]
                    if 0 <= a < b <= len(seq):
                        frags.append((a, b, hit)); fams.add(hit)
            if not frags:
                failed.append((g, "no DBD via InterPro")); continue
            frags.sort()
            cl = cluster(frags, GAP_THRESHOLD)
            pref = db_family_tokens(str(sub.iloc[0]["dbds_cisbp"]))
            a, b, amb, why = choose(cl, pref, None)
            crop = seq[a:b]
            for _, m in sub.iterrows():
                rows.append({
                    "tf_name": g, "gene_symbol": g, "uniprot_id": uid,
                    "organism": "Homo sapiens", "sequence": crop, "seq_length": len(crop),
                    "dbd_start": 0, "dbd_end": len(crop), "dbd_count": len(cl),
                    "family_name": "_".join(sorted(fams)), "family_id": -1,
                    "motif_length": int(m["motif_length"]), "pwm": m["pwm"],
                    "source": "CISBP_v3.10", "source_id": m["motif_id"],
                    "assay_type": "", "quality_grade": "",
                    "filename": f"CISBPv310__{g}.{m['motif_id']}.txt",
                    "origin": "cisbp_v310_new_gene",
                    "crop_method": why, "crop_ambiguous": amb, "gap_threshold": GAP_THRESHOLD,
                })
        except Exception as e:
            failed.append((g, str(e)[:60]))
        time.sleep(0.2)
        if (i + 1) % 10 == 0 or (i + 1) == len(genes):
            print(f"  [{i+1}/{len(genes)}] rows={len(rows)} failed={len(failed)}", flush=True)

    new = pd.DataFrame(rows)
    print(f"\nbuilt {len(new)} rows across {new['gene_symbol'].nunique() if len(new) else 0} genes")
    if failed:
        print(f"failed ({len(failed)}): {failed}")
    if len(new):
        print("\ncrop length:", new.drop_duplicates('gene_symbol')["seq_length"].describe()[["count","mean","50%","min","max"]].round(1).to_dict())
        new.to_parquet("/tmp/cisbp_new_genes_rows.parquet")
        print("saved /tmp/cisbp_new_genes_rows.parquet")


if __name__ == "__main__":
    main()
