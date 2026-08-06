#!/usr/bin/env python
"""Re-index the v23 recognition-residue prior and contact-distillation targets to
the FLANK-augmented coordinates. In v23 the DBD crop starts at 0, so the target
residue indices are DBD-relative == sequence-relative. In v25flank the DBD sits at
[dbd_start:dbd_end], so every PRIMARY residue index must be shifted by +dbd_start.
Partner indices are unaffected (the loader recomputes partner_start from len(chain1)
at runtime). dbd_only rows (dbd_start==0) pass through unchanged.
"""
import json
import pandas as pd

FLANK_PARQUET = "data/processed/tf_pwm_training_v25flank.parquet"
REC_IN = "data/contact_maps/recognition_residues_v23.json"
CON_IN = "data/contact_maps/contact_targets_v23.json"
REC_OUT = "data/contact_maps/recognition_residues_v25flank.json"
CON_OUT = "data/contact_maps/contact_targets_v25flank.json"

df = pd.read_parquet(FLANK_PARQUET)
ds = {r.filename: int(r.dbd_start) for r in df.itertuples()}
slen = {r.filename: int(r.seq_length) for r in df.itertuples()}

rec = json.load(open(REC_IN)); con = json.load(open(CON_IN))

def shift_primary_list(lst, off, L):
    return [i + off for i in lst if 0 <= i + off < L]

rec_out = {}
for fn, v in rec.items():
    off = ds.get(fn, 0); L = slen.get(fn, 10**9)
    if isinstance(v, dict):
        v2 = dict(v); v2["primary"] = shift_primary_list(v.get("primary", []), off, L)
        rec_out[fn] = v2                       # partner list unchanged
    else:
        rec_out[fn] = shift_primary_list(v, off, L)

con_out = {}
for fn, entry in con.items():
    off = ds.get(fn, 0); L = slen.get(fn, entry.get("L"))
    e2 = {"L": L, "cols": {}}
    for col, rows in entry.get("cols", {}).items():
        e2["cols"][col] = [[ridx + off, w] for ridx, w in rows if 0 <= ridx + off < L]
    if "partner_cols" in entry:
        e2["partner_cols"] = entry["partner_cols"]     # unchanged (partner_start recomputed)
    con_out[fn] = e2

json.dump(rec_out, open(REC_OUT, "w"))
json.dump(con_out, open(CON_OUT, "w"))
# sanity
nshift = sum(1 for fn in rec if ds.get(fn, 0) > 0)
print(f"recognition: {len(rec_out)} entries ({nshift} shifted) -> {REC_OUT}")
print(f"contact:     {len(con_out)} entries -> {CON_OUT}")
mk = [fn for fn, v in ds.items() if v > 0][:1]
if mk:
    fn = mk[0]
    print(f"example {fn}: dbd_start={ds[fn]}  rec_v23={rec.get(fn)}  rec_flank={rec_out.get(fn)}")
