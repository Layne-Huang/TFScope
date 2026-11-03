#!/usr/bin/env python
"""Re-frame the contact-supervision indices onto the frozen unified-v25 crops.

`contact_targets_v23.json` / `recognition_residues_v23.json` do NOT index the v23
crops, despite the name. Their residue indices live in the crop frame of
`tf_pwm_training_v25xtal.parquet` (304/309 rows agree on the stored `L`, against
15/309 for v23). v24 fed them straight into the v23 frame, so the supervision was
off-register: 181 of 309 rows carried residue indices past the end of their own
crop and were silently clipped at collate time.

Both files are crops of the same protein, so the mapping is a pure translation
wherever one crop contains the other -- 305/309 rows. The remaining 4 go through a
difflib opcode alignment. Indices that fall outside the new crop are dropped and
counted rather than clamped, because clamping would pile real contacts onto the
terminal residue and invent supervision that is not there.

This rewrites a DERIVED file to match the frozen dataset. It does not touch the
dataset itself -- see the `canonical-dataset-v25-frozen` note.

Out: data/contact_maps/contact_targets_unified_v25.json
     data/contact_maps/recognition_residues_unified_v25.json
     results/data_audit/contact_remap_v25.csv

  python scripts/remap_contact_targets_to_v25.py
"""
from __future__ import annotations

import json
import os
from difflib import SequenceMatcher

import pandas as pd

SRC_FRAME = "data/processed/tf_pwm_training_v25xtal.parquet"
DST_FRAME = "data/processed/tf_pwm_unified_v25.parquet"
CT_IN = "data/contact_maps/contact_targets_v23.json"
RP_IN = "data/contact_maps/recognition_residues_v23.json"
CT_OUT = "data/contact_maps/contact_targets_unified_v25.json"
RP_OUT = "data/contact_maps/recognition_residues_unified_v25.json"
REPORT = "results/data_audit/contact_remap_v25.csv"


def index_map(old: str, new: str) -> tuple[dict[int, int], str]:
    """old-crop position -> new-crop position, plus how it was derived."""
    i = new.find(old)
    if i >= 0:                                   # old sits inside new
        return {k: k + i for k in range(len(old))}, "shift"
    j = old.find(new)
    if j >= 0:                                   # new sits inside old
        return {k + j: k for k in range(len(new))}, "shift"
    m, route = {}, "align"
    for tag, a0, a1, b0, b1 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            for d in range(a1 - a0):
                m[a0 + d] = b0 + d
        elif tag == "replace" and (a1 - a0) == (b1 - b0):
            # same-length substitution: the repair wrote construct mutations back
            # over the reference span, so position still corresponds 1:1
            for d in range(a1 - a0):
                m[a0 + d] = b0 + d
    return m, route


def main():
    a = pd.read_parquet(SRC_FRAME, columns=["filename", "sequence"])
    b = pd.read_parquet(DST_FRAME, columns=["filename", "sequence"])
    OLD = dict(zip(a.filename.astype(str), a.sequence))
    NEW = dict(zip(b.filename.astype(str), b.sequence))

    ct = json.load(open(CT_IN))
    rp = json.load(open(RP_IN))
    ct_out, rp_out, rows = {}, {}, []

    for fn in sorted(set(ct) | set(rp)):
        old, new = OLD.get(fn), NEW.get(fn)
        rec = {"filename": fn, "in_src_frame": old is not None,
               "in_dst_table": new is not None}
        if old is None or new is None:
            rec["route"] = "dropped: absent from a frame"
            rows.append(rec); continue

        m, route = index_map(old, new)
        rec.update(route=route, old_len=len(old), new_len=len(new), mapped=len(m))

        if fn in ct:
            cols, kept, lost = {}, 0, 0
            for c, pairs in ct[fn]["cols"].items():
                keep = [[m[i], w] for i, w in pairs if i in m and 0 <= m[i] < len(new)]
                kept += len(keep); lost += len(pairs) - len(keep)
                if keep:
                    cols[c] = keep
            if cols:
                ct_out[fn] = {"L": len(new), "cols": cols}
            rec.update(ct_kept=kept, ct_lost=lost,
                       ct_old_max=max((i for p in ct[fn]["cols"].values() for i, _ in p),
                                      default=-1),
                       ct_old_L=ct[fn]["L"])

        if fn in rp:
            keep = sorted({m[i] for i in rp[fn] if i in m and 0 <= m[i] < len(new)})
            if keep:
                rp_out[fn] = keep
            rec.update(rp_kept=len(keep), rp_lost=len(rp[fn]) - len(keep))

        rows.append(rec)

    # every emitted index must be inside its own crop -- the whole point of this pass
    for fn, d in ct_out.items():
        assert d["L"] == len(NEW[fn])
        for pairs in d["cols"].values():
            assert all(0 <= i < d["L"] for i, _ in pairs), fn
    for fn, idx in rp_out.items():
        assert all(0 <= i < len(NEW[fn]) for i in idx), fn

    os.makedirs(os.path.dirname(CT_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    json.dump(ct_out, open(CT_OUT, "w"))
    json.dump(rp_out, open(RP_OUT, "w"))
    t = pd.DataFrame(rows)
    t.to_csv(REPORT, index=False)

    print(f"rows considered            {len(t)}")
    print(f"  by translation           {(t.route=='shift').sum()}")
    print(f"  by difflib alignment     {(t.route=='align').sum()}")
    print(f"  dropped                  {(t.route.astype(str).str.startswith('dropped')).sum()}")
    print(f"\ncontact_targets  {len(ct)} -> {len(ct_out)} rows, "
          f"weights kept {int(t.ct_kept.sum())} / lost {int(t.ct_lost.sum())}")
    print(f"recognition      {len(rp)} -> {len(rp_out)} rows, "
          f"residues kept {int(t.rp_kept.sum())} / lost {int(t.rp_lost.sum())}")
    before = int((t.ct_old_max >= t.new_len).sum())
    print(f"\nrows whose max residue index overran the crop: {before} -> 0 (verified)")
    print(f"\nsaved {CT_OUT}\nsaved {RP_OUT}\nsaved {REPORT}")


if __name__ == "__main__":
    main()
