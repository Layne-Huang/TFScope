#!/usr/bin/env python
"""Inset specs for the template-style panels d and e.

d  one zinc finger of the C2H2 case study, with the Zn-ligating Cys and His drawn as
   spheres in their experts' colours and the canonical recognition positions as sticks.
e  the bHLH case study, basic / DNA-reading segment against the dimerisation segment.

Writes results/moe_biochem/inset_spec_v2.json; render it with the pymol env:
  /data1/leihuang/miniconda3/envs/pymol/bin/python scripts/moe_interp/05_render_inset.py \
      --spec results/moe_biochem/inset_spec_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np

# expert accent palette, shared with the figure script
EXPERT_COLOR = {0: "#E9A13B", 1: "#C0392B", 2: "#2E8B8B", 3: "#3C6E9E",
                4: "#8E6BAF", 5: "#8A9A5B", 6: "#E07B1A", 7: "#CC6699"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/moe_biochem")
    ap.add_argument("--outdir", default="figures_v24/figure_moe_biochemical_programs")
    a = ap.parse_args()
    cases = json.load(open(os.path.join(a.indir, "case_studies.json")))
    specs = []

    # ── d: a single C2H2 finger ────────────────────────────────────────────────
    c = cases["c2h2"]
    seq = c["seq"]
    zn = np.array(c["zn_coord"])
    auth = c["auth_resid"]
    e1 = c["e1"]
    lig = np.where(zn)[0].tolist()
    # first CCHH module
    fingers, cur = [], []
    for i in lig:
        if cur and seq[i] == "C" and seq[cur[-1]] == "H":
            fingers.append(cur); cur = []
        cur.append(i)
    if cur:
        fingers.append(cur)
    f0 = fingers[0]
    lo_i, hi_i = max(f0[0] - 3, 0), min(f0[-1] + 3, len(seq) - 1)
    cys = [i for i in f0 if seq[i] == "C"]
    his = [i for i in f0 if seq[i] == "H"]
    # canonical recognition positions: helix starts 6 before the first His of the module
    h1 = his[0] if his else f0[-1]
    recog = [h1 - 7, h1 - 5, h1 - 4, h1 - 1]   # canonical -1, +2, +3, +6
    recog = [i for i in recog if lo_i <= i <= hi_i]

    specs.append(dict(
        out=os.path.join(a.outdir, "render_d_zincfinger.png"),
        pdb=c["pdb"], chain=c["chain"], buffer=1.2,
        resi_range=[auth[lo_i], auth[hi_i]],
        accents=[
            dict(expert=6, color=EXPERT_COLOR[6], style="spheres", cartoon=False,
                 resi=[auth[i] for i in cys]),
            dict(expert=4, color=EXPERT_COLOR[4], style="spheres", cartoon=False,
                 resi=[auth[i] for i in his]),
        ] + [
            # colour each recognition-helix residue by the expert it is actually routed to
            dict(expert=ex, color=EXPERT_COLOR[ex], style="sticks", cartoon=False,
                 resi=[auth[i] for i in recog if e1[i] == ex])
            for ex in sorted({e1[i] for i in recog})
        ]))
    print(f"d: {c['pdb']}/{c['chain']} finger 1 = auth {auth[lo_i]}-{auth[hi_i]}; "
          f"Cys {[seq[i] for i in cys]}@{[auth[i] for i in cys]}, "
          f"His@{[auth[i] for i in his]}, recognition@{[auth[i] for i in recog]} "
          f"(experts {[e1[i] for i in recog]})")

    # ── e: bHLH basic region vs dimerisation segment ───────────────────────────
    c = cases["bhlh"]
    bc = np.where(np.array(c["base_contact"]))[0]
    anchor = int(bc.max()) if len(bc) else len(c["seq"]) // 2
    auth = c["auth_resid"]
    specs.append(dict(
        out=os.path.join(a.outdir, "render_e_bhlh.png"),
        pdb=c["pdb"], chain=c["chain"], buffer=1.2,
        accents=[
            dict(expert=0, color=EXPERT_COLOR[0], style="sticks",
                 resi=[auth[i] for i in range(anchor + 1)]),
            dict(expert=3, color=EXPERT_COLOR[3], style="sticks",
                 resi=[auth[i] for i in range(anchor + 1, len(auth))]),
        ]))
    print(f"e: {c['pdb']}/{c['chain']} basic region = crop 0-{anchor} "
          f"(auth {auth[0]}-{auth[anchor]}), dimerisation = {anchor+1}-{len(auth)-1}")

    out = os.path.join(a.indir, "inset_spec_v2.json")
    json.dump(specs, open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
