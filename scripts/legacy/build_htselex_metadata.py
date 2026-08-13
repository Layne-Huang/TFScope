#!/usr/bin/env python
"""Build HT-SELEX run metadata for stage-A contrastive pretraining.

Source: Jolma et al. 2013 (Cell) HT-SELEX, ENA study PRJEB3289 (~2726 runs).
Optionally also Yin et al. 2017 (methyl-HT-SELEX) if --extra-study given.

Each ENA run has a sample_title encoding the experiment, e.g.
    CTCF_full_AJ_TAGCGA20NGCT_2
    Dlx2_TCGCCA20NCCT_AA_2
parsed as:
    tf        = first token (gene symbol)
    construct = 'full' | 'DBD' | 'DBDw' ... (if present)
    batch     = 2-letter code (e.g. AA, AJ)
    ligand    = <5'const><N-len>N<3'const>  (the randomized-region design)
    cycle     = trailing integer  (SELEX enrichment round; 0 = initial pool)

We map tf → uniprot_id + protein sequence + DBD coords by reusing the genes
already resolved in the TFScope parquet (case-insensitive gene_symbol). Misses
are reported for optional later UniProt resolution. Output is a per-run table;
the actual FASTQ download + k-mer enrichment happen in later scripts.

Output: data/processed/htselex/metadata.tsv
"""
import argparse, os, re, sys, subprocess, io
import pandas as pd

ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"
STUDY   = "PRJEB3289"          # Jolma 2013 HT-SELEX
OUTDIR  = "data/processed/htselex"
PARQUET = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"

LIGAND_RE = re.compile(r"^[ACGT]+\d+N[ACGT]+$", re.IGNORECASE)


def fetch_filereport(study):
    fields = "run_accession,sample_title,experiment_title,fastq_ftp,fastq_bytes,read_count,base_count"
    url = (f"{ENA_API}?accession={study}&result=read_run&fields={fields}"
           f"&format=tsv&limit=0")
    out = subprocess.run(["curl", "-sS", "-m", "180", url],
                         capture_output=True, text=True, check=True).stdout
    return pd.read_csv(io.StringIO(out), sep="\t")


def parse_title(title):
    """Return dict(tf, construct, batch, ligand, cycle) or None if unparseable."""
    if not isinstance(title, str) or not title:
        return None
    toks = title.split("_")
    if len(toks) < 2:
        return None
    tf = toks[0]
    cycle = None
    if toks[-1].isdigit():
        cycle = int(toks[-1])
    ligand = next((t for t in toks if LIGAND_RE.match(t)), None)
    construct = next((t for t in toks[1:] if t.lower() in
                      ("full", "dbd", "dbdw", "dbds", "ext")), None)
    batch = next((t for t in toks[1:-1]
                  if re.fullmatch(r"[A-Za-z]{2}", t) and t != construct), None)
    # ligand random-region length (e.g. 20 from TAGCGA20NGCT)
    n_len = None
    if ligand:
        m = re.search(r"(\d+)N", ligand, re.IGNORECASE)
        n_len = int(m.group(1)) if m else None
    return dict(tf=tf, construct=construct, batch=batch, ligand=ligand,
                n_len=n_len, cycle=cycle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=STUDY)
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Fetching ENA filereport for {args.study} ...", flush=True)
    rep = fetch_filereport(args.study)
    print(f"  {len(rep)} runs")

    parsed = rep["sample_title"].apply(parse_title)
    rep = rep[parsed.notna()].copy()
    pf = pd.json_normalize(parsed[parsed.notna()])
    pf.index = rep.index
    meta = pd.concat([rep, pf], axis=1)

    # single fastq url per run (some rows have ';'-joined paired files; keep first)
    meta["fastq_url"] = meta["fastq_ftp"].astype(str).str.split(";").str[0]

    # map tf -> uniprot/sequence/dbd from existing resolved parquet
    df = pd.read_parquet(args.parquet)
    cols = ["gene_symbol", "uniprot_id", "sequence", "dbd_start", "dbd_end",
            "family_name", "organism"]
    # prefer human, then longest sequence, as the canonical resolution per gene
    res = (df[cols].assign(gkey=lambda d: d["gene_symbol"].str.upper(),
                           _hs=lambda d: d["organism"].astype(str)
                               .str.contains("sapiens", case=False).astype(int),
                           _len=lambda d: d["sequence"].str.len())
           .sort_values(["_hs", "_len"], ascending=False)
           .drop_duplicates("gkey"))
    gmap = res.drop(columns=["_hs", "_len"]).set_index("gkey")

    meta["gkey"] = meta["tf"].str.upper()
    joined = meta.join(gmap, on="gkey", rsuffix="_res")
    joined["resolved"] = joined["uniprot_id"].notna()

    n_runs = len(joined)
    n_tf_total = joined["tf"].nunique()
    n_tf_res = joined.loc[joined["resolved"], "tf"].nunique()
    print(f"\nParsed runs: {n_runs}")
    print(f"Unique TFs (by title): {n_tf_total}")
    print(f"  resolved to UniProt via parquet: {n_tf_res}")
    print(f"  unresolved (need UniProt lookup): {n_tf_total - n_tf_res}")
    print(f"Cycle distribution:\n{joined['cycle'].value_counts().sort_index()}")
    print(f"Random-region length (n_len):\n{joined['n_len'].value_counts()}")

    out = os.path.join(args.outdir, "metadata.tsv")
    keep = ["run_accession", "tf", "gene_symbol", "uniprot_id", "family_name",
            "organism", "construct", "batch", "ligand", "n_len", "cycle",
            "read_count", "fastq_bytes", "fastq_url", "sequence",
            "dbd_start", "dbd_end", "resolved"]
    keep = [c for c in keep if c in joined.columns]
    joined[keep].to_csv(out, sep="\t", index=False)
    print(f"\nSaved {out}")

    # unresolved TF list for follow-up
    unres = sorted(joined.loc[~joined["resolved"], "tf"].unique())
    if unres:
        up = os.path.join(args.outdir, "unresolved_tfs.txt")
        open(up, "w").write("\n".join(unres) + "\n")
        print(f"Saved {len(unres)} unresolved TF names → {up}")


if __name__ == "__main__":
    main()
