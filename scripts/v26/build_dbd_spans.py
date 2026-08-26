#!/usr/bin/env python
"""v26 Phase-1 step 3: sequence-derived DBD candidate spans.

Retires audit Findings A (contact-defined boundaries) and I (motif-family oracle):
boundaries come ONLY from the frozen InterPro snapshot, and NO candidate is selected
here -- selection is a separate, explicitly-recorded decision made downstream.

Input : data/processed/v26/{domains,accessions}.parquet
        data/annotations_v26/dbd_pfam_whitelist.json
Output: data/processed/v26/dbd_candidates.parquet
          accession, candidate_idx, start, end (1-based inclusive, UniProt coords),
          n_fragments, families, entry_accessions, span_len, is_tandem_array
        results/v26/dbd_span_coverage.csv   per-accession candidate counts

Algorithm
  1. Keep InterPro fragments whose Pfam / InterPro accession is on the curated
     sequence-specific-DBD whitelist (chromatin readers excluded by construction).
  2. Merge fragments that overlap or abut.
  3. Cluster merged fragments into candidate DBDs with GAP_THRESHOLD=40 residues --
     the same biological argument as cluster_crop_v2.py (C2H2 TGEKP linkers are
     ~5-7 aa, loose arrays rarely exceed 20-30), so tandem arrays stay intact while
     genuinely separate domains split.
  4. Emit EVERY candidate. Multi-domain proteins yield multiple rows.

  python scripts/v26/build_dbd_spans.py [--gap 40]
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

OUTD = "data/processed/v26"
RESD = "results/v26"
WHITELIST = "data/annotations_v26/dbd_pfam_whitelist.json"
GAP_THRESHOLD = 40


def merge_intervals(iv):
    """iv: list of (start, end, family, entry). Merge overlapping/abutting."""
    if not iv:
        return []
    iv = sorted(iv, key=lambda x: (x[0], x[1]))
    out = [list(iv[0][:2]) + [{iv[0][2]}, {iv[0][3]}]]
    for s, e, fam, ent in iv[1:]:
        if s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2].add(fam)
            out[-1][3].add(ent)
        else:
            out.append([s, e, {fam}, {ent}])
    return out


def cluster_spans(merged, gap):
    """Group merged fragments separated by <= gap into one candidate DBD."""
    if not merged:
        return []
    groups = [[merged[0]]]
    for m in merged[1:]:
        if m[0] - groups[-1][-1][1] - 1 <= gap:
            groups[-1].append(m)
        else:
            groups.append([m])
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=GAP_THRESHOLD)
    a = ap.parse_args()
    os.makedirs(RESD, exist_ok=True)

    wl = json.load(open(WHITELIST))
    pf2fam = wl["pfam_to_family"]
    ip2fam = wl.get("interpro_to_family", {})
    # TIER 2 ("rescue"): broad / superfamily-level entries added to give a DBD to accessions that
    # have NO primary-tier hit at all. They are NOT allowed to redefine spans for accessions that
    # already have a primary DBD -- adding IPR013087 (Zinc finger C2H2-type) globally lengthened
    # 534/1357 existing spans, up to +429 aa on ZNF142, i.e. it silently changed the C2H2 DBD
    # definition. Rescue entries are applied in a second pass, per-accession, only where needed.
    rescue = set(wl.get("rescue_only_entries", []))
    pf1 = {k: v for k, v in pf2fam.items() if k not in rescue}
    ip1 = {k: v for k, v in ip2fam.items() if k not in rescue}
    print(f"whitelist tier1: {len(pf1)} Pfam + {len(ip1)} InterPro -> "
          f"{len(set(pf1.values()) | set(ip1.values()))} families; "
          f"tier2 rescue-only entries: {len(rescue)}", flush=True)

    dom = pd.read_parquet(f"{OUTD}/domains.parquet")
    acc = pd.read_parquet(f"{OUTD}/accessions.parquet")[["accession", "seq_len", "gene"]]
    acclen = dict(zip(acc.accession, acc.seq_len))

    # keep whitelisted entries: match either the member-database accession (PFxxxxx)
    # or the integrated InterPro accession (IPRxxxxxx)
    def _fam(d, pf, ip):
        f = d.entry_accession.map(pf)
        f = f.fillna(d.entry_accession.map(ip))
        return f.fillna(d.integrated.map(ip))

    dom["family_t1"] = _fam(dom, pf1, ip1)
    dom["family_t2"] = _fam(dom, pf2fam, ip2fam)

    keep = dom[dom.family_t1.notna()].copy()
    keep["family"] = keep.family_t1
    t1_acc = set(keep.accession)

    # tier-2 rescue pass: only accessions with zero tier-1 fragments
    resc = dom[dom.family_t2.notna() & ~dom.accession.isin(t1_acc)].copy()
    resc["family"] = resc.family_t2
    n_resc_acc = resc.accession.nunique()
    keep = pd.concat([keep, resc], ignore_index=True)
    keep["tier"] = keep.accession.map(lambda a: "tier1" if a in t1_acc else "tier2_rescue")

    print(f"domain fragments: {len(dom)} total -> {len(keep)} DBD fragments over "
          f"{keep.accession.nunique()} accessions "
          f"(tier1 {len(t1_acc)} acc, tier2 rescue {n_resc_acc} acc)", flush=True)

    rows, cov = [], []
    for accn, sub in keep.groupby("accession"):
        iv = [(int(r.start), int(r.end), str(r.family), str(r.entry_accession))
              for r in sub.itertuples()]
        merged = merge_intervals(iv)
        groups = cluster_spans(merged, a.gap)
        L = acclen.get(accn)
        for i, g in enumerate(groups):
            s = min(x[0] for x in g)
            e = max(x[1] for x in g)
            fams = sorted({f for x in g for f in x[2]})
            ents = sorted({t for x in g for t in x[3]})
            if L is not None:
                s, e = max(1, s), min(int(L), e)
            rows.append({
                "accession": accn, "candidate_idx": i,
                "start": int(s), "end": int(e), "span_len": int(e - s + 1),
                "n_fragments": len(g), "families": ";".join(fams),
                "entry_accessions": ";".join(ents),
                "is_tandem_array": len(g) > 1,
                "protein_len": int(L) if L is not None else None,
                "tier": "tier1" if accn in t1_acc else "tier2_rescue",
            })
        cov.append({"accession": accn, "n_candidates": len(groups),
                    "families": ";".join(sorted({f for m in merged for f in m[2]}))})

    cand = pd.DataFrame(rows)
    cand.to_parquet(f"{OUTD}/dbd_candidates.parquet", index=False)
    covd = pd.DataFrame(cov)
    covd.to_csv(f"{RESD}/dbd_span_coverage.csv", index=False)

    multi = covd[covd.n_candidates > 1]
    print(f"\ncandidates: {len(cand)} spans over {cand.accession.nunique()} accessions")
    print(f"  accessions with 1 candidate : {int((covd.n_candidates == 1).sum())}")
    print(f"  accessions with >1 candidate: {len(multi)}  "
          f"(these need an explicit selection MODE downstream)")
    print(f"  tandem-array spans (>1 merged fragment): {int(cand.is_tandem_array.sum())}")
    print(f"  span_len: median {int(cand.span_len.median())} "
          f"p10 {int(cand.span_len.quantile(.1))} p90 {int(cand.span_len.quantile(.9))} "
          f"max {int(cand.span_len.max())}")
    print(f"  family spread: {cand.families.str.split(';').explode().value_counts().head(8).to_dict()}")
    print(f"\n  wrote {OUTD}/dbd_candidates.parquet, {RESD}/dbd_span_coverage.csv")


if __name__ == "__main__":
    main()
