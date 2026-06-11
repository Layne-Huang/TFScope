#!/usr/bin/env python
"""Download HT-SELEX FASTQ runs from ENA (resume-safe).

Reads data/processed/htselex/metadata.tsv (from build_htselex_metadata.py) and
downloads fastq.gz to the holylabs big-file area. Defaults to a small family-
spanning PILOT so the enrichment path can be validated before the full ~11 GB.

Usage:
  python scripts/download_htselex.py --pilot                 # ~6 TFs, fg+bg cycles
  python scripts/download_htselex.py --tfs CTCF EGR1 GATA1
  python scripts/download_htselex.py --all                   # everything resolved
"""
import argparse, os, subprocess, sys
import pandas as pd

META = "data/processed/htselex/metadata.tsv"
DEST = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/htselex/fastq"
PILOT_TFS = ["CTCF", "EGR1", "GATA1", "MAX", "FOXA2", "HNF4A"]


def pick_runs(m, tfs):
    """For each TF keep its highest cycle (foreground) + a background cycle
    (cycle 0 of same ligand if present, else lowest cycle), matching ligand
    design so constant primer flanks cancel in log-enrichment."""
    sel = []
    for tf in tfs:
        sub = m[(m["tf"].str.upper() == tf.upper()) & m["resolved"]
                & m["cycle"].notna() & m["fastq_url"].astype(str).str.startswith("ftp")]
        if sub.empty:
            print(f"  [warn] no resolved runs for {tf}"); continue
        # group by ligand design; pick the series with the most cycles.
        # fill NaN ligand so those runs still form a (single) group.
        sub = sub.assign(ligand=sub["ligand"].fillna("NA"))
        best = None
        for lig, g in sub.groupby("ligand"):
            if best is None or g["cycle"].nunique() > best[1]["cycle"].nunique():
                best = (lig, g)
        if best is None:
            print(f"  [warn] no usable ligand series for {tf}"); continue
        g = best[1]
        if g["cycle"].nunique() < 2:
            print(f"  [warn] {tf}: <2 cycles in best series, skip"); continue
        fg = g.loc[g["cycle"].idxmax()]
        bg_pool = g[g["cycle"] == 0]
        bg = bg_pool.iloc[0] if len(bg_pool) else g.loc[g["cycle"].idxmin()]
        for role, row in [("fg", fg), ("bg", bg)]:
            sel.append({**row.to_dict(), "role": role})
    return pd.DataFrame(sel)


def download(url, dest):
    if not url.startswith(("http", "ftp")):
        url = "https://" + url.lstrip("/")
    elif url.startswith("ftp."):
        url = "https://" + url
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "cached"
    tmp = dest + ".part"
    r = subprocess.run(["curl", "-sS", "-L", "-m", "600", "-o", tmp, url])
    if r.returncode == 0 and os.path.getsize(tmp) > 0:
        os.rename(tmp, dest); return "ok"
    if os.path.exists(tmp): os.remove(tmp)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=META)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--tfs", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(args.meta, sep="\t")
    if args.all:
        tfs = sorted(m.loc[m["resolved"], "tf"].str.upper().unique())
    elif args.tfs:
        tfs = args.tfs
    else:  # pilot default
        tfs = PILOT_TFS
    print(f"Selecting runs for {len(tfs)} TFs ...")
    runs = pick_runs(m, tfs)
    if runs.empty:
        print("No runs selected."); return
    print(f"{len(runs)} runs to fetch "
          f"(~{runs['fastq_bytes'].fillna(0).astype(float).sum()/1e9:.2f} GB)")

    os.makedirs(args.dest, exist_ok=True)
    manifest = []
    for _, r in runs.iterrows():
        fn = f"{r['tf']}__{r['run_accession']}__c{int(r['cycle'])}__{r['role']}.fastq.gz"
        dest = os.path.join(args.dest, fn)
        status = download(r["fastq_url"], dest)
        print(f"  {fn:<55} {status}")
        manifest.append({**r.to_dict(), "local_path": dest, "status": status})
    mf = pd.DataFrame(manifest)
    out = os.path.join(os.path.dirname(args.meta), "download_manifest.tsv")
    # append to any existing manifest, dedup on local_path
    if os.path.exists(out):
        prev = pd.read_csv(out, sep="\t")
        mf = pd.concat([prev, mf]).drop_duplicates("local_path", keep="last")
    mf.to_csv(out, sep="\t", index=False)
    ok = (mf["status"] != "fail").sum()
    print(f"\n{ok}/{len(mf)} files present. Manifest → {out}")


if __name__ == "__main__":
    main()
