#!/usr/bin/env python
"""Generate a re-binned family_id column using ALL scanned Pfam families (no threshold).

Uses the cached InterPro domains from rebin_families_full_pfam.py. For every TF currently
in `Other` (family_id 9), assign its true DBD family from the full-Pfam scan:
  - if it maps to an already-named family (bZIP, Forkhead, ...) -> fold into that family
  - if it's a newly-discovered family (GATA, AP2/ERF, p53, HTH, ...) -> its own new class
  - if no DBD family resolved -> stays `Other`
TFs already in a named family (0-8) keep their label. Produces a new parquet with
`family_id_rebin` / `family_name_rebin`, a contiguous id space, and a mapping json.

Run from repo root in the `tfscope` env.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from rebin_families_full_pfam import dominant_family, CACHE

PARQUET = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
OUT_PARQUET = "data/processed/tf_pwm_aug_dbd_canon_trim_rebin.parquet"
OUTDIR = "results/family_rebin"

# discovered-name (from scan) -> existing named family it should fold into
FOLD_INTO_EXISTING = {
    "bZIP": "bZIP", "bHLH": "bHLH", "Homeodomain": "Homeodomain",
    "Nuclear_Receptor": "Nuclear_Receptor", "ETS": "ETS",
    "Forkhead/WH": "Forkhead",
    "C2H2": "C2H2_medium",          # tiny count; default to mid C2H2 subclass
}
# the 9 existing named families, in their original id order
EXISTING_ORDER = ["C2H2_short", "C2H2_medium", "C2H2_long", "bHLH", "Homeodomain",
                  "bZIP", "Nuclear_Receptor", "Forkhead", "ETS"]

def main():
    df = pd.read_parquet(PARQUET)
    cache = json.load(open(CACHE))

    # 1) resolve every Other TF to its scanned family name (or None)
    uid2fam = {}
    for uid in df["uniprot_id"].dropna().astype(str).unique():
        fam, _ev = dominant_family(cache.get(uid, []))
        uid2fam[uid] = fam

    def rebin_name(row):
        if row["family_id"] != 9:                 # already a named family -> keep
            return row["family_name"]
        scanned = uid2fam.get(str(row["uniprot_id"]))
        if scanned is None:
            return "Other"                         # genuinely unresolved
        return FOLD_INTO_EXISTING.get(scanned, scanned)  # fold or new class

    df["family_name_rebin"] = df.apply(rebin_name, axis=1)

    # 2) build a contiguous id space: existing 9 (same order) -> new families by motif
    #    count desc -> Other last
    counts = df["family_name_rebin"].value_counts()
    new_fams = [f for f in counts.index
                if f not in EXISTING_ORDER and f != "Other"]
    new_fams = sorted(new_fams, key=lambda f: -counts[f])
    ordered = EXISTING_ORDER + new_fams + ["Other"]
    name2id = {f: i for i, f in enumerate(ordered)}
    df["family_id_rebin"] = df["family_name_rebin"].map(name2id).astype(int)

    # 3) save
    df.to_parquet(OUT_PARQUET, index=False)
    os.makedirs(OUTDIR, exist_ok=True)
    gene_counts = df.groupby("family_name_rebin")["gene_symbol"].nunique()
    mapping = {"num_families": len(ordered),
               "id2name": {i: f for f, i in name2id.items()},
               "name2id": name2id,
               "counts": {f: {"motifs": int(counts.get(f, 0)),
                              "genes": int(gene_counts.get(f, 0))} for f in ordered}}
    json.dump(mapping, open(os.path.join(OUTDIR, "family_map_rebin.json"), "w"), indent=2)

    # 4) report
    print(f"OLD taxonomy: 10 families  (Other = {int((df['family_id']==9).sum())} motifs)")
    print(f"NEW taxonomy: {len(ordered)} families  "
          f"(Other = {int(counts.get('Other',0))} motifs)\n")
    print(f"{'id':>3s} {'family':20s} {'motifs':>7s} {'genes':>6s}  {'status'}")
    for f in ordered:
        i = name2id[f]
        status = ("existing" if f in EXISTING_ORDER else
                  "OTHER (unresolved)" if f == "Other" else "NEW")
        print(f"{i:3d} {f:20s} {int(counts.get(f,0)):7d} {int(gene_counts.get(f,0)):6d}  {status}")
    n_new = len(new_fams)
    print(f"\n-> {n_new} NEW family classes carved out of Other; "
          f"Other shrank {int((df['family_id']==9).sum())} -> {int(counts.get('Other',0))} motifs")
    print(f"saved -> {OUT_PARQUET}")
    print(f"saved -> {OUTDIR}/family_map_rebin.json")

if __name__ == "__main__":
    main()
