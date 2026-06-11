#!/usr/bin/env python
"""Full-Pfam family re-binning analysis for TFScope's MoE taxonomy.

The original `map_tf_annotations.py` assigns families by matching each TF against a
HARDCODED whitelist of 7 Pfam DBD families; everything else falls into `Other` (the
single largest bucket). This script asks: if we instead read EVERY DBD Pfam/InterPro
family present in the data and bin by abundance, how many distinct, learnable family
classes hide inside `Other`, and how much does `Other` shrink at motif-count thresholds?

Method
------
1. Take all training TFs currently labelled `Other` (family_id == 9) from the parquet.
2. For each unique UniProt id, fetch InterPro entries (REST API) and collect the DBD-type
   domains — using InterPro's own `type == "domain"/"family"` + name, NOT a whitelist.
3. Map each TF to its dominant DBD family name (longest / most-specific DBD entry).
4. Count motifs (rows, i.e. with augmentation) and unique genes per discovered family.
5. Report how many families clear ≥50 / ≥100 / ≥150 motifs (viable-expert thresholds)
   and the residual `Other` size at each.

Run from repo root in the `tfscope` env. Writes results/family_rebin/.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd
import requests

PARQUET = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
OUTDIR = "results/family_rebin"
CACHE = os.path.join(OUTDIR, "interpro_domains_cache.json")

# DBD-type InterPro entry names we treat as a DNA-binding fold. Broad, curated from the
# families known to occur in animal TFs — matched case-insensitively as substrings of the
# InterPro entry name. This is the OPEN vocabulary (vs the old 7-entry Pfam whitelist).
DBD_NAME_KEYS = [
    "homeobox", "homeodomain", "helix-turn-helix", "hth",
    "zinc finger, c2h2", "c2h2", "krab",
    "gata", "nuclear hormone receptor", "nuclear receptor",
    "basic-leucine zipper", "bzip", "leucine zipper",
    "helix-loop-helix", "bhlh",
    "fork head", "forkhead", "winged helix",
    "ets", "p53", "p53-like",
    "rel homology", "rhd", "nf-kappa", "nfat",
    "interferon regulatory factor", "irf",
    "runt", "t-box", "tbox",
    "mads", "srf", "mef2",
    "dm dna-binding", "dmrt", "doublesex",
    "e2f", "tdp", "dp",
    "methyl-cpg", "mbd",
    "grainyhead", "cp2", "lsf",
    "ap2", "erf", "sand", "arid", "bright",
    "heat shock factor", "hsf", "hsf-type",
    "stat", "rfx", "cut", "cenp-b", "centromere protein b",
    "high mobility group", "hmg", "sox", "tcf",
    "gcm", "csl", "rbpj", "tea", "tead", "cbf",
    "paired box", "pax", "ncu-g1", "thap", "myb", "sant",
    "ccaat", "nf-y", "cbf-b", "ndt80",
]

# canonical family display name <- substring detected
CANON = [
    (("homeobox","homeodomain"), "Homeodomain"),
    (("helix-turn-helix","hth"), "HTH"),
    (("c2h2","krab"), "C2H2"),
    (("gata",), "GATA"),
    (("nuclear hormone","nuclear receptor"), "Nuclear_Receptor"),
    (("bzip","basic-leucine","leucine zipper"), "bZIP"),
    (("helix-loop-helix","bhlh"), "bHLH"),
    (("fork head","forkhead","winged helix"), "Forkhead/WH"),
    (("ets",), "ETS"),
    (("p53",), "p53"),
    (("rel homology","rhd","nf-kappa","nfat"), "RHD/NFkB"),
    (("interferon regulatory","irf"), "IRF"),
    (("runt",), "Runt"),
    (("t-box","tbox"), "T-box"),
    (("mads","srf","mef2"), "MADS/SRF"),
    (("dm dna","dmrt","doublesex"), "DMRT"),
    (("e2f","tdp"," dp"), "E2F/DP"),
    (("methyl-cpg","mbd"), "MBD"),
    (("grainyhead","cp2","lsf"), "Grainyhead/CP2"),
    (("ap2","erf"), "AP2/ERF"),
    (("arid","bright","sand"), "ARID/SAND"),
    (("heat shock factor","hsf"), "HSF"),
    (("stat",), "STAT"),
    (("rfx",), "RFX"),
    (("cut",), "CUT"),
    (("cenp-b","centromere protein b"), "CENP-B"),
    (("high mobility group","hmg","sox","tcf"), "HMG/SOX"),
    (("gcm",), "GCM"),
    (("csl","rbpj"), "CSL"),
    (("tea","tead"), "TEA/TEAD"),
    (("paired box","pax"), "PAX"),
    (("thap",), "THAP"),
    (("myb","sant"), "MYB/SANT"),
    (("ccaat","nf-y","cbf-b"), "NF-Y/CBF"),
    (("ndt80",), "NDT80"),
]

def canon_family(name: str):
    n = name.lower()
    for keys, disp in CANON:
        if any(k in n for k in keys):
            return disp
    return None

def is_dbd(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in DBD_NAME_KEYS)

def fetch_domains(session, uid, retries=3):
    url = f"https://www.ebi.ac.uk/interpro/api/entry/InterPro/protein/UniProt/{uid}?format=json"
    for a in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 204:
                return []
            r.raise_for_status()
            out = []
            for entry in r.json().get("results", []):
                md = entry.get("metadata", {})
                nm = md.get("name", "") or ""
                typ = md.get("type", "")
                length = 0
                for prot in entry.get("proteins", []):
                    for loc in prot.get("entry_protein_locations", []):
                        for frag in loc.get("fragments", []):
                            length = max(length, frag.get("end", 0) - frag.get("start", 0))
                out.append({"name": nm, "type": typ, "len": length})
            return out
        except requests.exceptions.RequestException:
            time.sleep(1.5 * (a + 1))
    return None  # failed

def dominant_family(domains):
    """Pick the canonical DBD family from the longest DBD-type entry."""
    cand = [(d["len"], canon_family(d["name"]), d["name"]) for d in domains
            if is_dbd(d["name"]) and canon_family(d["name"])]
    if not cand:
        return None, None
    cand.sort(reverse=True)
    return cand[0][1], cand[0][2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["other", "all"], default="other",
                    help="re-scan only the Other bucket (default) or every TF")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #uniprot fetched")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_parquet(PARQUET)
    sub = df if args.scope == "all" else df[df["family_id"] == 9]
    uids = sorted(set(sub["uniprot_id"].dropna().astype(str)) - {"", "nan"})
    if args.limit:
        uids = uids[:args.limit]
    print(f"[rebin] scope={args.scope}  motifs={len(sub)}  unique UniProt={len(uids)}")

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    failed = []
    for i, uid in enumerate(uids):
        if uid in cache:
            continue
        d = fetch_domains(session, uid)
        if d is None:
            failed.append(uid); cache[uid] = []
        else:
            cache[uid] = d
        if (i + 1) % 25 == 0:
            json.dump(cache, open(CACHE, "w"))
            print(f"  fetched {i+1}/{len(uids)}", flush=True)
        time.sleep(0.15)
    json.dump(cache, open(CACHE, "w"))
    if failed:
        print(f"[rebin] {len(failed)} UniProt fetches failed (treated as unresolved)")

    # assign each motif row to a re-binned family via its uniprot
    uid2fam = {}
    for uid in uids:
        fam, ev = dominant_family(cache.get(uid, []))
        uid2fam[uid] = fam  # None -> stays unresolved
    sub = sub.copy()
    sub["rebin_family"] = sub["uniprot_id"].astype(str).map(uid2fam)

    # counts
    motif_counts = sub["rebin_family"].value_counts(dropna=False)
    gene_counts = (sub.dropna(subset=["rebin_family"])
                      .groupby("rebin_family")["gene_symbol"].nunique().sort_values(ascending=False))
    unresolved_motifs = int(sub["rebin_family"].isna().sum())

    print("\n=== Families discovered inside the OLD 'Other' bucket ===")
    print(f"{'family':18s} {'motifs':>7s} {'genes':>6s}")
    rows = []
    for fam, m in motif_counts.items():
        if fam is None or (isinstance(fam, float) and np.isnan(fam)):
            continue
        g = int(gene_counts.get(fam, 0))
        rows.append((fam, int(m), g))
        print(f"{str(fam):18s} {int(m):7d} {g:6d}")
    print(f"{'(unresolved)':18s} {unresolved_motifs:7d}")

    print("\n=== Viable-expert thresholds (how many NEW family classes clear each) ===")
    summary = {}
    for thr in (50, 100, 150):
        viable = [(f, m, g) for f, m, g in rows if m >= thr]
        new_other = unresolved_motifs + sum(m for f, m, g in rows if m < thr)
        print(f"  >= {thr:3d} motifs : {len(viable):2d} new families   "
              f"residual Other = {new_other} motifs")
        summary[f"thr_{thr}"] = dict(n_new_families=len(viable),
                                     residual_other_motifs=int(new_other),
                                     families=[f for f, m, g in viable])
    json.dump(dict(scope=args.scope, total_motifs_scanned=int(len(sub)),
                   unique_uniprot=len(uids), unresolved_motifs=unresolved_motifs,
                   discovered=[dict(family=f, motifs=m, genes=g) for f, m, g in rows],
                   thresholds=summary, failed_fetches=failed),
              open(os.path.join(OUTDIR, "rebin_summary.json"), "w"), indent=2)
    print(f"\nsaved -> {OUTDIR}/rebin_summary.json  (+ interpro_domains_cache.json)")

if __name__ == "__main__":
    main()
