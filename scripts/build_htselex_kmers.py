#!/usr/bin/env python
"""Compute k-mer enrichment from downloaded HT-SELEX runs → contrastive pairs.

For each TF with a foreground (late-cycle) and background (early-cycle) run of
the SAME ligand design, count k-mers in both and score
    enrichment(kmer) = log2( (c_fg/N_fg + eps) / (c_bg/N_bg + eps) )
The shared constant primer flanks cancel between fg and bg of the same design,
so counting over the full read is fine. Enriched k-mers ARE the binding
landscape — denser supervision than a single consensus PWM.

Outputs, under data/processed/htselex/:
  kmer_enrichment.parquet  — long table [uniprot_id, gene_symbol, sequence,
                             family_name, kmer, enrichment, c_fg, c_bg, k]
  topk_per_tf.tsv          — top-N enriched k-mers per TF (for eyeballing motifs)

These pairs feed stage-A contrastive pretraining (protein tower ↔ DNA k-mer
tower, InfoNCE; enrichment as soft affinity weight / positive selection).
"""
import argparse, gzip, os, sys
from collections import Counter
import numpy as np, pandas as pd

MANIFEST = "data/processed/htselex/download_manifest.tsv"
OUTDIR   = "data/processed/htselex"
COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s): return s.translate(COMP)[::-1]


def count_kmers(path, k, max_reads=None, both_strands=True):
    c = Counter(); n_reads = 0
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i % 4 != 1:       # sequence lines only
                continue
            seq = line.strip().upper()
            if not seq or set(seq) - set("ACGT"):
                # skip reads with N / non-ACGT
                seq = "".join(ch for ch in seq if ch in "ACGT")
                if len(seq) < k:
                    continue
            n_reads += 1
            for j in range(len(seq) - k + 1):
                km = seq[j:j + k]
                c[km] += 1
                if both_strands:          # strand-symmetric: also count revcomp
                    c[revcomp(km)] += 1
            if max_reads and n_reads >= max_reads:
                break
    return c, n_reads


def enrich_tf(fg_path, bg_path, k, eps=1.0, max_reads=None):
    cf, _ = count_kmers(fg_path, k, max_reads)
    cb, _ = count_kmers(bg_path, k, max_reads)
    Nf = sum(cf.values()) or 1
    Nb = sum(cb.values()) or 1
    kmers = set(cf) | set(cb)
    rows = []
    for km in kmers:
        ff = (cf.get(km, 0) + eps) / Nf
        fb = (cb.get(km, 0) + eps) / Nb
        rows.append((km, float(np.log2(ff / fb)), cf.get(km, 0), cb.get(km, 0)))
    df = pd.DataFrame(rows, columns=["kmer", "enrichment", "c_fg", "c_bg"])
    return df.sort_values("enrichment", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=100, help="enriched kmers kept per TF")
    ap.add_argument("--max-reads", type=int, default=None,
                    help="cap reads per run (speed; default all)")
    args = ap.parse_args()

    mf = pd.read_csv(args.manifest, sep="\t")
    mf = mf[mf["status"] != "fail"]
    all_rows, top_rows = [], []
    for tf, g in mf.groupby("tf"):
        fg = g[g["role"] == "fg"]; bg = g[g["role"] == "bg"]
        if fg.empty or bg.empty:
            print(f"  [skip] {tf}: missing fg/bg"); continue
        fg, bg = fg.iloc[0], bg.iloc[0]
        df = enrich_tf(fg["local_path"], bg["local_path"], args.k,
                       max_reads=args.max_reads)
        top = df.head(args.top_n).copy()
        for col in ["uniprot_id", "gene_symbol", "family_name", "sequence"]:
            top[col] = fg.get(col)
        top["k"] = args.k
        all_rows.append(top)
        consensus = top.iloc[0]["kmer"]
        print(f"  {tf:<10} fg_c{int(fg['cycle'])} bg_c{int(bg['cycle'])}  "
              f"top kmer {consensus} (E={top.iloc[0]['enrichment']:.2f})  "
              f"| next: {', '.join(top['kmer'].iloc[1:5])}")
        top_rows.append((tf, consensus, revcomp(consensus),
                         "; ".join(top["kmer"].head(8))))

    if not all_rows:
        print("No TF produced enrichment."); return
    out = pd.concat(all_rows, ignore_index=True)
    os.makedirs(args.outdir, exist_ok=True)
    pq = os.path.join(args.outdir, "kmer_enrichment.parquet")
    out.to_parquet(pq, index=False)
    tt = os.path.join(args.outdir, "topk_per_tf.tsv")
    pd.DataFrame(top_rows, columns=["tf", "top_kmer", "top_kmer_rc",
                                    "top8"]).to_csv(tt, sep="\t", index=False)
    print(f"\nSaved {len(out)} (TF,kmer) pairs over {out['gene_symbol'].nunique()} "
          f"TFs → {pq}\nTop-kmer summary → {tt}")


if __name__ == "__main__":
    main()
