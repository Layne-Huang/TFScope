#!/usr/bin/env python
"""v26 Phase-1 step 3b: triage accessions with no whitelisted DBD.

build_dbd_spans.py found 50 accessions (131 v23 rows) with no whitelisted sequence-specific
DBD. Dropping them silently would repeat the class of error this rebuild exists to fix, so each
is classified from the InterPro entry names ALREADY IN THE SNAPSHOT (authoritative, not recalled)
into one of:

  whitelist_gap        a real sequence-specific DBD family the curated whitelist misses
  excluded_by_policy   chromatin reader / non-sequence-specific (PHD, bromo, MBD, methyltransferase)
                       -- deliberately excluded, same policy as reextract_dbd_round3.py
  not_a_TF             accession is not a transcription factor -> gene-symbol mis-resolution
  unclassified         needs human review; never silently dropped

Emits:
  results/v26/missing_dbd_triage.csv              per-accession decision + evidence
  data/annotations_v26/whitelist_additions.json   proposed additions, with the InterPro entry
                                                  name that justifies each one

  python scripts/v26/triage_missing_dbd.py
  python scripts/v26/triage_missing_dbd.py --apply   # merge additions into the whitelist
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

OUTD = "data/processed/v26"
RESD = "results/v26"
WHITELIST = "data/annotations_v26/dbd_pfam_whitelist.json"
ADDITIONS = "data/annotations_v26/whitelist_additions.json"

# Positive evidence: InterPro entry names denoting a SEQUENCE-SPECIFIC DNA-binding domain.
# Each pattern -> family label. Matched case-insensitively against the InterPro entry_name.
DBD_NAME_PATTERNS = [
    (r"zinc finger,? c2h2", "C2H2"),
    (r"zinc finger c2h2", "C2H2"),
    (r"high mobility group protein hmga", "HMGA_AT_hook"),
    (r"\bat hook\b", "HMGA_AT_hook"),
    (r"casz1-like,? zinc finger", "CASZ1_zf"),
    (r"zinc finger, flywch", "FLYWCH"),
    (r"flywch zinc finger", "FLYWCH"),
    (r"transcription factor tfiid|tata-box.binding|tbp\b", "TBP"),
    (r"nuclear transcription factor y|nf-y[ab]|ccaat.binding", "NF_Y"),
    (r"lag1|rbpj|csl\b|lag-1", "CSL_RBPJ"),
    (r"zinc finger, cxxc", "CXXC_zf"),
    (r"cxxc zinc finger", "CXXC_zf"),
    (r"grainyhead|cp2\b", "CP2_GRHL"),
    (r"\bskn-1\b|skn1", "SKN1"),
    (r"cold.shock|nucleic acid.binding, ob-fold", "CSD_YBX"),
    # --- round 2: families surfaced by the first pass's `unclassified` bucket ---
    (r"myb/sant-like dna.binding", "Myb_SANT"),
    (r"nuclear respiratory factor 1|nrf1/ewg", "NRF1"),
    (r"homeo.prospero", "Homeo_prospero"),
    (r"transcription factor cbf/nf-y|nfyb/hap3|archaeal histone domain", "NF_Y"),
    (r"zinc finger, ccch|ccch zinc finger|zinc finger c-x8-c-x5-c-x3-h", "CCCH_zf"),
    (r"cg-1 dna.binding", "CG1_CAMTA"),
    (r"ndt80, dna.binding|p53-like transcription factor, dna.binding", "NDT80_p53like"),
    (r"sand-like domain", "SAND"),
    (r"gtf2i-like repeat", "GTF2I"),
    (r"\bben domain\b", "BEN"),
    # --- round 3: non-human orthologs surfaced after the round-2 accession fetch ---
    (r"zn\(2\)cys\(6\) fungal-type dna.binding|zn\(2\)-c6 fungal-type|binuclear cluster domain", "Zn2Cys6_fungal"),
    (r"wrky domain", "WRKY"),
    (r"b3 dna binding domain|dna.binding pseudobarrel", "B3_plant"),
    (r"brinker dna.binding", "Brinker"),
    (r"bes1/bzr1 plant transcription factor", "BZR1_plant"),
    (r"transcription factor aft|iron-regulated transcriptional activator aft", "AFT_fungal"),
]

# Negative evidence: deliberately excluded by policy (same list as reextract_dbd_round3.py docstring)
EXCLUDE_PATTERNS = [
    (r"phd.type|phd.finger", "PHD_reader"),
    (r"bromodomain", "bromodomain_reader"),
    (r"jmjc|jumonji", "JmjC_reader"),
    (r"methyl-cpg.binding|\bmbd\b", "MBD_reader"),
    (r"dna \(cytosine-5\)-methyltransferase|dna methylase", "DNA_methyltransferase"),
    (r"ring/fyve/phd|fyve/phd", "PHD_reader"),
    (r"leucine-rich repeat", "LRR_nonDBD"),
    (r"chromo domain", "chromo_reader"),
    (r"\bset domain\b", "SET_reader"),
]

# Accessions that are clearly not transcription factors (gene-symbol mis-resolution).
NOT_TF_PATTERNS = [
    (r"chemokine receptor", "GPCR_not_TF"),
    (r"pyruvate dehydrogenase", "metabolic_enzyme_not_TF"),
    (r"g-protein coupled receptor", "GPCR_not_TF"),
    (r"protein kinase domain|serine/threonine-protein kinase", "protein_kinase_not_TF"),
    (r"pre-rrna-processing|rix1", "rRNA_processing_not_TF"),
    (r"xpa/rad14|zinc finger, xpa-type", "DNA_repair_not_sequence_specific"),
    # Crystallisation fusion partners that SIFTS returns as the chain's accession.
    # These are NOT the TF -- the row needs a different chain/accession, not a whitelist entry.
    (r"maltose/cyclodextrin abc transporter|bacterial-type extracellular solute-binding",
     "MBP_fusion_tag_not_TF"),
    (r"green fluorescent protein", "GFP_fusion_tag_not_TF"),
    (r"hhh-gpd domain|dna glycosylase", "DNA_glycosylase_not_sequence_specific"),
    (r"cyclic gmp-amp synthase|mab-21-like", "cGAS_not_TF"),
]


def classify(names: list[str]) -> tuple[str, str, str]:
    """Return (decision, family_or_reason, evidence_entry_name)."""
    joined = [(n or "").lower() for n in names]
    for pats, decision in ((DBD_NAME_PATTERNS, "whitelist_gap"),):
        for pat, fam in pats:
            for n in joined:
                if re.search(pat, n):
                    return decision, fam, n
    for pat, reason in NOT_TF_PATTERNS:
        for n in joined:
            if re.search(pat, n):
                return "not_a_TF", reason, n
    for pat, reason in EXCLUDE_PATTERNS:
        for n in joined:
            if re.search(pat, n):
                return "excluded_by_policy", reason, n
    return "unclassified", "", "; ".join(joined[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="merge additions into the whitelist")
    a = ap.parse_args()
    os.makedirs(RESD, exist_ok=True)

    acc = pd.read_parquet(f"{OUTD}/accessions.parquet")
    cand = pd.read_parquet(f"{OUTD}/dbd_candidates.parquet")
    dom = pd.read_parquet(f"{OUTD}/domains.parquet")
    res = pd.read_parquet(f"{OUTD}/row_resolution.parquet")

    missing = sorted(set(acc.accession) - set(cand.accession))
    rows_per_acc = res.groupby("primary_accession").size().to_dict()
    gene_of = dict(zip(acc.accession, acc.gene))

    dom_ip = dom[dom.source_database.isin(["pfam", "interpro"])]
    by_acc = {k: v for k, v in dom_ip.groupby("accession")}

    out, additions = [], {}
    for accn in missing:
        sub = by_acc.get(accn)
        names = sub.entry_name.dropna().unique().tolist() if sub is not None else []
        ents = sub.entry_accession.dropna().unique().tolist() if sub is not None else []
        decision, fam, ev = classify(names)
        if decision == "whitelist_gap" and sub is not None:
            for _, r in sub.iterrows():
                nm = str(r.entry_name or "").lower()
                for pat, f in DBD_NAME_PATTERNS:
                    if re.search(pat, nm):
                        additions[str(r.entry_accession)] = {
                            "family": f, "entry_name": r.entry_name,
                            "source_database": r.source_database,
                            "evidence": "InterPro entry name in frozen snapshot",
                        }
                        break
        out.append({
            "accession": accn, "gene": gene_of.get(accn),
            "v23_rows": rows_per_acc.get(accn, 0),
            "decision": decision, "family_or_reason": fam,
            "evidence_entry_name": ev,
            "n_domain_entries": len(ents),
            "all_entries": ";".join(ents[:12]),
        })

    t = pd.DataFrame(out).sort_values(["decision", "v23_rows"], ascending=[True, False])
    t.to_csv(f"{RESD}/missing_dbd_triage.csv", index=False)
    json.dump({"note": "proposed whitelist additions justified by InterPro entry names in the "
                       "frozen snapshot; apply with triage_missing_dbd.py --apply",
               "n": len(additions), "entries": additions},
              open(ADDITIONS, "w"), indent=1)

    print(f"accessions with no whitelisted DBD: {len(missing)}")
    print(t.groupby("decision").agg(accessions=("accession", "size"),
                                    v23_rows=("v23_rows", "sum")).to_string())
    print(f"\nproposed whitelist additions: {len(additions)}")
    for k, v in sorted(additions.items()):
        print(f"  {k:12s} {v['family']:16s} {v['entry_name']}")
    unc = t[t.decision == "unclassified"]
    if len(unc):
        print(f"\nUNCLASSIFIED ({len(unc)} accessions, {int(unc.v23_rows.sum())} rows) "
              f"-- need review:")
        print(unc[["accession", "gene", "v23_rows", "evidence_entry_name"]].to_string(index=False))

    if a.apply:
        wl = json.load(open(WHITELIST))
        pf, ip = wl["pfam_to_family"], wl.get("interpro_to_family", {})
        n_pf = n_ip = 0
        for ent, meta in additions.items():
            if ent.startswith("PF"):
                if ent not in pf:
                    pf[ent] = meta["family"]; n_pf += 1
            elif ent.startswith("IPR"):
                if ent not in ip:
                    ip[ent] = meta["family"]; n_ip += 1
        wl["pfam_to_family"], wl["interpro_to_family"] = pf, ip
        wl["n_pfam"], wl["n_interpro"] = len(pf), len(ip)
        wl.setdefault("provenance", []).append(
            {"step": "triage_missing_dbd.py --apply",
             "added_pfam": n_pf, "added_interpro": n_ip,
             "justification": "InterPro entry names in data/annotations_v26 snapshot"})
        json.dump(wl, open(WHITELIST, "w"), indent=1)
        print(f"\nAPPLIED: +{n_pf} Pfam, +{n_ip} InterPro -> {WHITELIST}")
        print("  re-run: python scripts/v26/build_dbd_spans.py")


if __name__ == "__main__":
    main()
