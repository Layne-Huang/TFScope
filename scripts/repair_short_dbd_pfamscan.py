#!/usr/bin/env python
"""Second-pass repair for the rows UniProt annotation alone could not fix.

`repair_short_dbd_from_uniprot.py` leaves three residues of failure:

  5 rows  the UniProt entry exists but carries no DNA-binding / zinc-finger / DBD
          domain feature (HSF1, RELA, SPT15). Fixed here by running Pfam HMMs over the
          FULL UniProt sequence -- not over the 11-aa crop, which is far too short for
          any HMM to hit.
  9 rows  the gene symbol has no UniProt entry. Three distinct causes, handled
          separately below; they are not one problem.
  5 rows  the crop matches no UniProt sequence even approximately (POU6F1, RXRA).
          NOT touched here -- <60% identity is a different protein, and guessing would
          substitute one sequence for another.

The nine symbol failures:

  P53           a naming variant of TP53. Aliased and repaired normally.
  FOS::JUN      dimer composites. The stored `sequence` is ONE chain, not two: the
  POU5F1::SOX2  FOS::JUN crop is JUN's basic region and the POU5F1::SOX2 crop is
                POU5F1's POU-specific domain. Repaired against that chain's own entry,
                and flagged, because the partner chain is simply absent from the row.
  CG18619       Drosophila. Outside the human/mouse UniProt snapshot, and only a 15-aa
                crop is on hand, so there is nothing local to extend from. Left for a
                network fetch; reported, not guessed.

  PATH=/data1/leihuang/miniconda3/envs/multiflow/bin:$PATH \\
      python scripts/repair_short_dbd_pfamscan.py
"""
from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid

import pandas as pd

sys.path.insert(0, "scripts")
from repair_short_dbd_from_uniprot import (APPROX_MIN_ID, DBD_KEYWORDS,  # noqa: E402
                                           ZF_MERGE_GAP, candidates, load_uniprot,
                                           locate, merge)

# Pfam family names are a different vocabulary from UniProt's free-text descriptions,
# so reusing DBD_KEYWORDS on them silently drops real hits: RELA's `RHD_DNA_bind` has
# no substring in common with any UniProt-style keyword. Match Pfam's own convention
# (`*_DNA_bind`) as well, and explicitly drop the neighbouring non-DNA-binding modules
# that hmmscan reports alongside -- `RHD_dimer`, activation domains and the like.
PFAM_DBD_EXTRA = re.compile(r"DNA[-_ ]?bind", re.I)
PFAM_NOT_DBD = re.compile(r"dimer|oligo|tetramer|activation|transactiv|Vert_HS_TF", re.I)

PROPOSALS = "results/data_audit/short_dbd_repair_proposals.csv"
OUT = "results/data_audit/short_dbd_repair_pass2.csv"
PFAM_DIR = "/data1/leihuang/pfam_scan"

# gene symbol in the table -> the UniProt symbol that actually holds the sequence
ALIAS = {"P53": "TP53"}
# dimer composites: which chain the stored sequence belongs to
DIMER_CHAIN = {"FOS::JUN": "JUN", "POU5F1::SOX2": "POU5F1"}


def pfam_scan(seqs: dict[str, str]) -> pd.DataFrame:
    """Run the local Pfam HMM library over {name: sequence}."""
    tmp = os.path.join(PFAM_DIR, f"tmp_{uuid.uuid4().hex}.fasta")
    with open(tmp, "w") as f:
        for k, v in seqs.items():
            f.write(f">{k}\n{v}\n")
    try:
        r = subprocess.run(["./pfam_scan.py", os.path.basename(tmp), "pfamdb/",
                            "-cpu", "8"],
                           cwd=PFAM_DIR, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"pfam_scan failed:\n{r.stderr[-1500:]}")
        lines = [l for l in r.stdout.strip().split("\n") if l]
    finally:
        os.path.exists(tmp) and os.remove(tmp)
    if len(lines) < 2:
        return pd.DataFrame()
    df = pd.DataFrame([l.split(",") for l in lines[1:]], columns=lines[0].split(","))
    for c in ("aln_start", "aln_end", "env_start", "env_end"):
        df[c] = df[c].astype(int)
    df["score"] = df.score.astype(float)
    return df


def main():
    prop = pd.read_csv(PROPOSALS)
    up = load_uniprot()
    rows = []

    # ── (1) UniProt entry present, no DBD feature -> Pfam over the full sequence ──
    nodbd = prop[prop.status == "UniProt entry has no DNA-binding / zinc-finger / DBD domain"]
    full_of = {}
    for r in nodbd.drop_duplicates("gene").itertuples():
        for p in up.get(r.gene, []):
            if p.get("primaryAccession") == r.accession:
                full_of[r.gene] = p["sequence"]["value"]
    print(f"[1] pfam-scanning {len(full_of)} full sequences: {sorted(full_of)}")
    hits = pfam_scan(full_of) if full_of else pd.DataFrame()
    if not hits.empty:
        print(hits[["seq_id", "aln_start", "aln_end", "hmm_name", "type", "score",
                    "evalue"]].to_string(index=False))

    for r in nodbd.itertuples():
        rec = {"filename": r.filename, "gene": r.gene, "old_len": r.old_len,
               "accession": r.accession, "route": "pfam_scan(full UniProt seq)"}
        h = hits[hits.seq_id == r.gene] if not hits.empty else pd.DataFrame()
        # keep only DBD-like Pfam families, then merge neighbours into one span
        if not h.empty:
            keep = (h.hmm_name.str.contains(DBD_KEYWORDS) |
                    h.hmm_name.str.contains(PFAM_DBD_EXTRA))
            h = h[keep & ~h.hmm_name.str.contains(PFAM_NOT_DBD)]
        if h.empty:
            rec["status"] = "no DBD-like Pfam domain found in the full sequence"
        else:
            spans = merge(sorted((int(a), int(b), n) for a, b, n in
                                 zip(h.aln_start, h.aln_end, h.hmm_name)), ZF_MERGE_GAP)
            s, e, nm = max(spans, key=lambda x: x[1] - x[0])
            seq = full_of[r.gene]
            rec.update(new_start=s, new_end=e, new_len=e - s + 1, new_seq=seq[s - 1:e],
                       evidence=f"Pfam {nm}", n_candidate_spans=len(spans),
                       status="proposed")
        rows.append(rec)

    # ── (2) symbol not in UniProt: alias, dimer chain, or out of scope ───────────
    nosym = prop[prop.status == "no UniProt entry for this gene symbol"]
    for r in nosym.itertuples():
        tgt = ALIAS.get(r.gene) or DIMER_CHAIN.get(r.gene)
        rec = {"filename": r.filename, "gene": r.gene, "old_len": r.old_len,
               "route": (f"alias -> {tgt}" if r.gene in ALIAS else
                         f"dimer composite; sequence is the {tgt} chain"
                         if r.gene in DIMER_CHAIN else "out of snapshot scope"),
               "resolved_symbol": tgt,
               "is_dimer_composite": r.gene in DIMER_CHAIN}
        if not tgt:
            rec["status"] = ("non-human/mouse gene, absent from the snapshot and only a "
                             "15-aa crop available locally -- needs a UniProt fetch")
            rows.append(rec); continue

        best = (None, -1.0)
        for p in up.get(tgt, []):
            full = p["sequence"]["value"]
            i = full.find(r.old_seq)
            if i >= 0:
                best = ((p, (i, 1.0, [])), 1.0); break
            loc = locate(r.old_seq, full)
            if loc and loc[1] > best[1]:
                best = ((p, loc), loc[1])
        if not best[0] or best[1] < APPROX_MIN_ID:
            rec["status"] = (f"{tgt} found but the crop does not match it "
                             f"(best identity {best[1]:.2f})")
            rec["best_identity"] = round(best[1], 3) if best[0] else None
            rows.append(rec); continue

        p, (i, idn, mm) = best[0]
        full = p["sequence"]["value"]
        off = i + 1
        rec.update(accession=p.get("primaryAccession"), match_identity=round(idn, 4),
                   n_mismatch=len(mm), old_start=off, old_end=off + r.old_len - 1,
                   full_len=len(full))
        cands = candidates(p)
        if not cands:
            rec["status"] = f"{tgt} entry has no DBD feature either"
            rows.append(rec); continue
        olap = [(s, e, ev) for s, e, ev in cands
                if not (e < rec["old_start"] or s > rec["old_end"])]
        s, e, ev = max(olap or cands, key=lambda x: x[1] - x[0])
        new = list(full[s - 1:e]); subs = []
        for j in mm:                                  # keep the construct's residues
            pos = off + j
            if s <= pos <= e:
                k = pos - s
                subs.append(f"{new[k]}{pos}{r.old_seq[j]}")
                new[k] = r.old_seq[j]
        rec.update(new_start=s, new_end=e, new_len=e - s + 1, new_seq="".join(new),
                   substitutions=";".join(subs), evidence=ev,
                   overlaps_old_crop=bool(olap), status="proposed")
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print("\n=== pass-2 outcome ===")
    for k, v in out.status.value_counts().items():
        print(f"  {v:3d}  {k}")
    ok = out[out.status == "proposed"]
    if not ok.empty:
        print(f"\nrepaired: {len(ok)} rows / {ok.gene.nunique()} genes")
        cols = [c for c in ["gene", "resolved_symbol", "old_len", "new_len", "evidence",
                            "route", "substitutions"] if c in ok.columns]
        print(ok[cols].drop_duplicates("gene").to_string(index=False))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
