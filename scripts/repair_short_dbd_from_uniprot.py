#!/usr/bin/env python
"""Repair truncated DBD crops by re-deriving them from UniProt domain annotation.

The v23 table contains 159 rows whose protein crop is too short to carry the binding
information the label demands -- 53 of them under 20 aa (down to 5), and 120 that are
under 40% the length of the LONGEST crop of the SAME gene, i.e. the same protein cropped
two incompatible ways. Deleting them loses data; this re-derives the crop instead.

Evidence, in order of preference, from the v26 UniProt snapshot
(`data/annotations_v26/uniprot.jsonl.gz`, release 2026_02):

  1. `DNA binding` feature   -- UniProt's curated DNA-binding region. Definitive.
  2. `Zinc finger` features  -- individual C2H2 fingers; consecutive fingers separated
                                by < GAP residues are merged into one array, because a
                                zinc-finger DBD is the array, not one finger.
  3. `Domain` feature whose description matches a known DBD fold keyword.

Every proposal is gated on one check: the CURRENT crop must be an exact substring of the
UniProt full sequence. If it is not, the row and the accession describe different
proteins (wrong isoform, wrong species, a dimer composite) and NOTHING is proposed --
silently re-cropping those would substitute one protein for another.

This writes a review table. It does NOT modify the parquet: the substring test proves
identity but not that the chosen span is the one the motif was measured against, so the
proposals need eyeballing before they are applied.

  python scripts/repair_short_dbd_from_uniprot.py
"""
from __future__ import annotations

import gzip
import json
import os
import re

import pandas as pd

DATA = "data/processed/tf_pwm_training_v23.parquet"
UNIPROT = "data/annotations_v26/uniprot.jsonl.gz"
OUT = "results/data_audit/short_dbd_repair_proposals.csv"

MIN_LEN = 20          # rule A: nothing shorter than this can carry a motif
INTRA_FRAC = 0.40     # rule B: vs the longest crop of the SAME gene
ZF_MERGE_GAP = 30     # merge zinc fingers closer than this into one array
APPROX_MIN_ID = 0.90  # a crop this close to UniProt is the same protein, mutated

DBD_KEYWORDS = re.compile(
    r"homeo|home[ao]box|bzip|leucine zipper|basic|helix-loop-helix|bhlh|fork[- ]?head|"
    r"winged helix|ets|t-box|tbox|p53|pou|hmg|sox|gata|nuclear receptor|nr c4|"
    r"zinc finger|c2h2|rel|irf|runt|stat|myb|sant|arid|mads|rfx|grainyhead|cxxc|"
    r"tea|tead|ap-2|nfi|csl|ebf|wrky|thap|dmrt|gcm|sand|mbd|tbp|e2f|smad|hsf",
    re.I)


def load_uniprot():
    """gene symbol -> list of UniProt payloads."""
    by_gene = {}
    with gzip.open(UNIPROT, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d.get("status") != 200 or not isinstance(d.get("payload"), dict):
                continue
            p = d["payload"]
            for gn in p.get("genes", []):
                nm = gn.get("geneName", {}).get("value")
                if nm:
                    by_gene.setdefault(nm.upper(), []).append(p)
    return by_gene


def locate(crop, full):
    """(offset0, identity, mismatch_positions) of the best equal-length window.

    Several crops are not exact substrings because the deposited construct carries
    point mutations -- five of the eight affected genes are the conserved bZIP /
    homeodomain cysteine changed to serine, the standard trick to stop disulfide
    cross-linking during EMSA/SELEX. The motif was measured on the MUTANT, so the
    mutant residues are kept and only the coordinates are borrowed from UniProt.
    """
    L = len(crop)
    if L > len(full):
        return None
    best = (-1, -1.0, None)
    for i in range(len(full) - L + 1):
        w = full[i:i + L]
        mm = [j for j in range(L) if crop[j] != w[j]]
        idn = 1.0 - len(mm) / L
        if idn > best[1]:
            best = (i, idn, mm)
        if idn == 1.0:
            break
    return best


def feats(p, kind):
    out = []
    for ft in p.get("features", []):
        if ft.get("type") != kind:
            continue
        loc = ft.get("location", {})
        s = loc.get("start", {}).get("value")
        e = loc.get("end", {}).get("value")
        if s and e:
            out.append((int(s), int(e), str(ft.get("description") or "")))
    return sorted(out)


def merge(spans, gap):
    """Collapse spans separated by < gap into one."""
    out = []
    for s, e, d in spans:
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e), out[-1][2])
        else:
            out.append((s, e, d))
    return out


def candidates(p):
    """(start, end, evidence) proposals, best evidence first. 1-based inclusive."""
    dna = feats(p, "DNA binding")
    if dna:
        return [(s, e, "UniProt DNA binding") for s, e, _ in merge(dna, ZF_MERGE_GAP)]
    zf = feats(p, "Zinc finger")
    if zf:
        return [(s, e, f"UniProt zinc-finger array ({len(zf)} fingers)")
                for s, e, _ in merge(zf, ZF_MERGE_GAP)]
    dom = [(s, e, d) for s, e, d in feats(p, "Domain") if DBD_KEYWORDS.search(d)]
    return [(s, e, f"UniProt domain: {d}") for s, e, d in merge(dom, ZF_MERGE_GAP)]


def main():
    df = pd.read_parquet(DATA)
    df["filename"] = df.filename.astype(str)
    df["g"] = df.gene_symbol.astype(str).str.upper()
    df["L"] = df.sequence.str.len()
    df["gene_max_L"] = df.g.map(df.groupby("g").L.max())
    bad = df[(df.L < MIN_LEN) | (df.L / df.gene_max_L < INTRA_FRAC)].copy()
    print(f"bad rows: {len(bad)} / {bad.g.nunique()} genes "
          f"(A: L<{MIN_LEN} = {(df.L < MIN_LEN).sum()}; "
          f"B: intra-gene <{INTRA_FRAC} = {(df.L / df.gene_max_L < INTRA_FRAC).sum()})")

    up = load_uniprot()
    print(f"UniProt snapshot: {len(up)} gene symbols")

    rows = []
    for r in bad.itertuples():
        rec = {"filename": r.filename, "gene": r.g, "old_len": int(r.L),
               "gene_max_len": int(r.gene_max_L), "motif_source": r.motif_source,
               "old_seq": r.sequence}
        entries = up.get(r.g, [])
        if not entries:
            rec["status"] = "no UniProt entry for this gene symbol"
            rows.append(rec); continue

        # exact substring first; otherwise the best equal-length window, accepted only
        # above APPROX_MIN_ID -- below that it is a different sequence, not a mutant
        chosen = off = None; ident = 1.0; mism = []
        best_alt = (None, -1.0)
        for p in entries:
            full = p.get("sequence", {}).get("value", "")
            i = full.find(r.sequence)
            if i >= 0:
                chosen, off, ident, mism = p, i + 1, 1.0, []
                break
            loc = locate(r.sequence, full)
            if loc and loc[1] > best_alt[1]:
                best_alt = ((p, loc), loc[1])
        if chosen is None and best_alt[0] and best_alt[1] >= APPROX_MIN_ID:
            p, (i, idn, mm) = best_alt[0]
            chosen, off, ident, mism = p, i + 1, idn, mm
        if chosen is None:
            rec["status"] = ("current crop does not match any UniProt sequence for this "
                             f"symbol (best identity {best_alt[1]:.2f} < {APPROX_MIN_ID})"
                             if best_alt[0] else
                             "current crop is not a substring of any UniProt sequence")
            rec["accessions_tried"] = ";".join(p.get("primaryAccession", "?")
                                               for p in entries)
            rec["best_identity"] = round(best_alt[1], 3) if best_alt[0] else None
            rows.append(rec); continue

        full = chosen["sequence"]["value"]
        rec.update(accession=chosen.get("primaryAccession"),
                   full_len=len(full), old_start=off, old_end=off + int(r.L) - 1,
                   match_type="exact" if ident == 1.0 else "approx",
                   match_identity=round(ident, 4), n_mismatch=len(mism))

        cands = candidates(chosen)
        if not cands:
            rec["status"] = "UniProt entry has no DNA-binding / zinc-finger / DBD domain"
            rows.append(rec); continue

        # the span that actually overlaps where the current crop sits; else the longest
        olap = [(s, e, ev) for s, e, ev in cands
                if not (e < rec["old_start"] or s > rec["old_end"])]
        s, e, ev = max(olap or cands, key=lambda x: x[1] - x[0])
        # take the span from UniProt, then write the deposited construct's own residues
        # back over it: the label was measured on that construct, not on wild type
        new = list(full[s - 1:e])
        subs = []
        for j in mism:
            pos = off + j                      # 1-based UniProt coordinate
            if s <= pos <= e:
                k = pos - s
                subs.append(f"{new[k]}{pos}{r.sequence[j]}")
                new[k] = r.sequence[j]
        rec.update(new_start=s, new_end=e, new_len=e - s + 1,
                   new_seq="".join(new), substitutions=";".join(subs),
                   n_substitutions=len(subs), evidence=ev,
                   overlaps_old_crop=bool(olap),
                   n_candidate_spans=len(cands),
                   status="proposed")
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = [c for c in ["filename", "gene", "accession", "status", "match_type",
                        "match_identity", "n_mismatch", "substitutions",
                        "n_substitutions", "old_len", "new_len",
                        "gene_max_len", "full_len", "old_start", "old_end", "new_start",
                        "new_end", "evidence", "overlaps_old_crop", "n_candidate_spans",
                        "motif_source", "old_seq", "new_seq", "accessions_tried",
                        "best_identity"]
            if c in out.columns]
    out[cols].to_csv(OUT, index=False)

    print("\n=== outcome ===")
    for k, v in out.status.value_counts().items():
        print(f"  {v:4d} rows  {k}")
    ok = out[out.status == "proposed"]
    if not ok.empty:
        print(f"\nproposed repairs: {len(ok)} rows / {ok.gene.nunique()} genes")
        print(f"  crop length {ok.old_len.median():.0f} -> {ok.new_len.median():.0f} "
              f"(median); still <{MIN_LEN} aa after repair: {(ok.new_len < MIN_LEN).sum()}")
        print(f"  span overlaps the old crop: {ok.overlaps_old_crop.sum()}/{len(ok)}")
        if "match_type" in ok:
            print(f"  match: exact {(ok.match_type=='exact').sum()}, "
                  f"approx {(ok.match_type=='approx').sum()} "
                  f"(mutant constructs; residues preserved, "
                  f"{int(ok.n_substitutions.fillna(0).sum())} substitutions written back)")
        print("\n  evidence used:")
        for k, v in ok.evidence.str.replace(r" \(.*", "", regex=True).value_counts().items():
            print(f"    {v:4d}  {k}")
        print("\n  examples:")
        print(ok[["gene", "old_len", "new_len", "evidence"]]
              .drop_duplicates("gene").head(14).to_string(index=False))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
