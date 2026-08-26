#!/usr/bin/env python
"""v26 Phase-2 step 5: regenerate v24-SHAPED contact targets from the v26 canonical source.

Purpose: make the v24-vs-v26 ablation fair. If v26 wins while using different contact
supervision, the comparison is confounded (exactly the v24-vs-v25flank problem, audit Finding D).
This emits the SAME JSON shape v24's loader expects, but derived from canonical coordinates, so
both models can be trained on supervision that differs only in coordinate hygiene.

Writes to data/contacts_v26/ ONLY. It never touches data/contact_maps/, so the legacy v23/v25
targets remain byte-identical (asserted by tests/v26/test_legacy_untouched.py).

Shape (matches contact_targets_v23.json):
  { "<legacy_filename>": {"L": <seqlen>, "cols": {"<col>": [[res_idx, weight], ...]}} }
with res_idx in CROP coordinates of the chosen dataset.

  python scripts/v26/rebuild_v24_compatible_contacts.py --dataset core
"""
from __future__ import annotations
import argparse, json, os
import pandas as pd

CD = "data/contacts_v26"; V26D = "data/processed/v26"; RESD = "results/v26"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="core")
    ap.add_argument("--exclude-eval-only", action="store_true", default=True)
    a = ap.parse_args()
    os.makedirs(RESD, exist_ok=True)

    src = f"{CD}/contacts2d_{a.dataset}.parquet"
    if not os.path.exists(src):
        raise SystemExit(f"missing {src}; run build_contact_2d_columns.py first")
    df = pd.read_parquet(src)
    ex = pd.read_parquet(f"{V26D}/v26_{a.dataset}.parquet")
    seqlen = {r.example_id: len(str(r.sequence)) for r in ex.itertuples()}
    legacy = {r.example_id: r.legacy_filename for r in ex.itertuples()}

    use = df[(df.column_status == "assigned") & df.in_crop].copy()
    n_before = len(use)
    if a.exclude_eval_only:
        use = use[~use.eval_only]
    print(f"assigned+in_crop contacts: {n_before}; after eval_only exclusion: {len(use)}")

    out, dropped = {}, 0
    for eid, g in use.groupby("example_id"):
        fn = legacy.get(eid); L = seqlen.get(eid)
        if fn is None or L is None:
            dropped += len(g); continue
        cols = {}
        for col, gg in g.groupby("pwm_column"):
            # weight = closeness (nearer contact -> larger), normalised per column
            w = (4.5 - gg.min_distance.clip(upper=4.5)) / 4.5 + 0.1
            cols[str(int(col))] = [[int(r), round(float(x), 4)]
                                   for r, x in zip(gg.crop_residue_idx, w)]
        out[fn] = {"L": int(L), "cols": cols}

    p = f"{CD}/v24compat_contact_targets_{a.dataset}.json"
    json.dump(out, open(p, "w"))
    n_links = sum(len(v) for e in out.values() for v in e["cols"].values())
    n_cols = sum(len(e["cols"]) for e in out.values())
    summary = {"dataset": a.dataset, "entries": len(out), "columns": n_cols,
               "residue_links": n_links, "rows_dropped_no_legacy_name": dropped,
               "excluded_eval_only": bool(a.exclude_eval_only),
               "v23_reference_links": 10232, "v23_reference_entries": 309, "path": p}
    json.dump(summary, open(f"{RESD}/v24compat_contacts_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"  (v23 had 309 entries / 10,232 links, of which 1,034 were silently clipped)")

if __name__ == "__main__":
    main()
