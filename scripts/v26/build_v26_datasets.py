#!/usr/bin/env python
"""v26 Phase-1 step 4: emit the model-input datasets from sequence-derived DBD spans.

Produces v26_core / v26_flank20 / v26_flank32 from ONE span table, so the three variants differ
only in flank width -- the controlled ablation the audit showed v24-vs-v25flank could not support
(v23 clipped 1,034 contact links to zero while v25flank relocated 680 onto flanks, so the two
models optimised different objectives; docs/v26_audit.md §9).

Every crop records its UniProt coordinate window (`crop_unp_start/end`), which is what lets Phase 2
project contacts into crop space without ever moving or silently clipping an index.

Candidate selection for multi-domain proteins is EXPLICIT and recorded per row in
`dbd_selection_mode`; the motif database's family label is never used (audit Finding I):

  sole_candidate        only one sequence-derived DBD exists -> no choice was made
  legacy_crop_overlap   >1 candidate; the user-supplied annotated DBD (approximated here by the
                        legacy v23 crop) picks one. TARGET-DBD MODE.
  enumerated            >1 candidate; every candidate emitted as its own example, nothing selected.
                        CANDIDATE-ENUMERATION MODE.

Both modes are emitted so downstream experiments can choose and report which they used.

  python scripts/v26/build_v26_datasets.py
  python scripts/v26/build_v26_datasets.py --flanks 0 20 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

OUTD = "data/processed/v26"
RESD = "results/v26"
V23 = "data/processed/tf_pwm_training_v23.parquet"
STRUCT = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"
PROGRESS_EVERY = 1000


def _h(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def select_candidate(cands: pd.DataFrame, legacy_crop: str | None, full_seq: str | None):
    """Return (row, mode). Uses the legacy crop's LOCATION only, never its family label."""
    if len(cands) == 1:
        return cands.iloc[0], "sole_candidate"
    if legacy_crop and full_seq:
        pos = full_seq.find(legacy_crop)
        if pos >= 0:
            l0, l1 = pos + 1, pos + len(legacy_crop)          # 1-based inclusive
            ov = cands.apply(lambda r: _overlap(int(r.start), int(r.end), l0, l1), axis=1)
            if ov.max() > 0:
                return cands.loc[ov.idxmax()], "legacy_crop_overlap"
    # No annotation available -> longest candidate, flagged so it can be filtered out
    return cands.loc[cands.span_len.idxmax()], "longest_candidate_UNVERIFIED"


def build(flank: int, rows: list[dict], tag: str):
    df = pd.DataFrame(rows)
    p = f"{OUTD}/v26_{tag}.parquet"
    df.to_parquet(p, index=False)
    L = df.sequence.str.len()
    span = df.dbd_end - df.dbd_start
    print(f"  {tag:14s} rows={len(df):5d} accessions={df.primary_accession.nunique():4d} "
          f"medlen={int(L.median()):4d} dbd_frac={float((span / L).mean()):.3f} "
          f"dbd_start>0={float((df.dbd_start > 0).mean()):.2f} -> {p}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flanks", type=int, nargs="+", default=[0, 20, 32])
    a = ap.parse_args()
    os.makedirs(RESD, exist_ok=True)

    v = pd.read_parquet(V23).rename(columns={"_set": "set_label"})
    res = pd.read_parquet(f"{OUTD}/row_resolution.parquet").set_index("filename")
    acc = pd.read_parquet(f"{OUTD}/accessions.parquet").set_index("accession")
    cand = pd.read_parquet(f"{OUTD}/dbd_candidates.parquet")
    st = pd.read_parquet(STRUCT).reset_index(drop=True)
    by_acc = {k: g for k, g in cand.groupby("accession")}

    # legacy filename -> (pdb, chain) for structure metadata (metadata ONLY, never an input)
    pdb_of = {}
    for i, r in st.iterrows():
        pdb_of[f"str_{i}"] = (str(r.pdb_id).upper(), str(r.chain_id))

    per_flank = {f: [] for f in a.flanks}
    enumerated = []
    stats = {"no_accession": 0, "no_candidate": 0, "emitted": 0,
             "modes": {}, "composite_dimer": 0}

    for i, r in enumerate(v.itertuples(), 1):
        fn = str(r.filename)
        rr = res.loc[fn] if fn in res.index else None
        accn = None if rr is None else rr.primary_accession
        if accn is None or accn not in acc.index:
            stats["no_accession"] += 1
            continue
        arow = acc.loc[accn]
        full = str(arow.sequence)
        cs = by_acc.get(accn)
        if cs is None or not len(cs):
            stats["no_candidate"] += 1
            continue

        legacy_crop = str(r.sequence)
        sel, mode = select_candidate(cs, legacy_crop, full)
        stats["modes"][mode] = stats["modes"].get(mode, 0) + 1

        gene = str(r.gene_symbol).upper()
        is_composite = "::" in gene
        if is_composite:
            stats["composite_dimer"] += 1

        # partner entities: sequences carried from v23; accession resolution is best-effort and
        # lives in Phase 3 (needed for the Assembly-OOD audit), so store sequences + gene here.
        partners = []
        ps = r.partner_seqs
        if ps is not None:
            for j, pseq in enumerate(list(ps)):
                pseq = str(pseq)
                if len(pseq) >= 10:
                    partners.append({"index": j, "sequence": pseq,
                                     "sequence_hash": _h(pseq), "length": len(pseq),
                                     "gene_hint": str(r.partner_gene or "")})

        target_unit_id = _h(accn, int(sel.start), int(sel.end))
        motif_obs_id = _h(fn, str(r.motif_source))
        base = {
            "target_unit_id": target_unit_id,
            "motif_observation_id": motif_obs_id,
            "legacy_filename": fn,
            "legacy_set": str(r.set_label),
            "motif_source": str(r.motif_source),
            "primary_accession": accn,
            "primary_gene": str(arow.gene) if arow.gene else gene,
            "primary_gene_symbol_legacy": gene,
            "primary_sequence_hash": str(arow.sequence_hash),
            "primary_protein_len": int(arow.seq_len),
            "organism_taxid": arow.organism_taxid,
            "dbd_unp_start": int(sel.start), "dbd_unp_end": int(sel.end),
            "dbd_families_for_analysis_only": str(sel.families),
            "dbd_tier": str(sel.tier),
            "dbd_selection_mode": mode,
            "n_dbd_candidates": int(len(cs)),
            "is_composite_dimer_name": bool(is_composite),
            "partner_entities": json.dumps(partners),
            "n_partners": len(partners),
            "pwm": r.pwm, "motif_length": int(r.motif_length),
            "structure_id": (pdb_of.get(fn, (None, None))[0]),
            "structure_chain": (pdb_of.get(fn, (None, None))[1]),
        }

        for f in a.flanks:
            s = max(1, int(sel.start) - f)
            e = min(int(arow.seq_len), int(sel.end) + f)
            crop = full[s - 1:e]
            row = dict(base)
            row.update({
                "example_id": _h(motif_obs_id, accn, s, e),
                "sequence": crop,
                "crop_unp_start": s, "crop_unp_end": e,      # <- Phase-2 projection key
                "dbd_start": int(sel.start) - s,             # crop-local, 0-based
                "dbd_end": int(sel.end) - s + 1,
                "flank_width": f,
            })
            per_flank[f].append(row)
        stats["emitted"] += 1

        # candidate-enumeration mode: every candidate as its own example (flank 0 only)
        if len(cs) > 1:
            for _, c in cs.iterrows():
                s, e = int(c.start), int(c.end)
                enumerated.append({
                    **base,
                    "example_id": _h(motif_obs_id, accn, s, e, "enum"),
                    "target_unit_id": _h(accn, s, e),
                    "dbd_unp_start": s, "dbd_unp_end": e,
                    "dbd_selection_mode": "enumerated",
                    "dbd_families_for_analysis_only": str(c.families),
                    "sequence": full[s - 1:e],
                    "crop_unp_start": s, "crop_unp_end": e,
                    "dbd_start": 0, "dbd_end": e - s + 1,
                    "flank_width": 0,
                    "candidate_idx": int(c.candidate_idx),
                })

        if i % PROGRESS_EVERY == 0:
            print(f"  processed {i}/{len(v)} rows  emitted={stats['emitted']}", flush=True)

    print("\n=== datasets ===")
    names = {0: "core", 20: "flank20", 32: "flank32"}
    built = {}
    for f in a.flanks:
        built[f] = build(f, per_flank[f], names.get(f, f"flank{f}"))
    if enumerated:
        e = pd.DataFrame(enumerated)
        e.to_parquet(f"{OUTD}/v26_core_enumerated.parquet", index=False)
        print(f"  {'enumerated':14s} rows={len(e):5d} "
              f"accessions={e.primary_accession.nunique():4d} -> {OUTD}/v26_core_enumerated.parquet")

    # nesting invariant: core ⊂ flank20 ⊂ flank32 for the same example
    if 0 in built and 20 in built:
        k = ["motif_observation_id"]
        m = built[0][k + ["crop_unp_start", "crop_unp_end"]].merge(
            built[20][k + ["crop_unp_start", "crop_unp_end"]], on=k, suffixes=("_c", "_f"))
        bad = int(((m.crop_unp_start_f > m.crop_unp_start_c)
                   | (m.crop_unp_end_f < m.crop_unp_end_c)).sum())
        print(f"\n  nesting core ⊂ flank20: {'OK' if bad == 0 else f'{bad} VIOLATIONS'}")

    stats["modes"] = dict(sorted(stats["modes"].items(), key=lambda x: -x[1]))
    json.dump(stats, open(f"{RESD}/dataset_build_stats.json", "w"), indent=2)
    print(f"\n  selection modes: {stats['modes']}")
    print(f"  skipped: no_accession={stats['no_accession']} no_candidate={stats['no_candidate']}")
    print(f"  composite dimer names: {stats['composite_dimer']}")
    print(f"  wrote {RESD}/dataset_build_stats.json")


if __name__ == "__main__":
    main()
