#!/usr/bin/env python
"""Render a full-length TF (AlphaFold model) with the DBD highlighted, for the
TFScope architecture figure: whole chain gray, DBD residues purple.

Fetches the current AlphaFold DB model (queries the API for the latest version),
writes a .pml, and (optionally) you render headless with `pymol -cq <pml>`.

Usage:
  <py> scripts/render_tf_architecture.py --uniprot P61244 --gene MAX \
       --dbd-start 22 --dbd-end 80 --outdir results/pymol_investigation/arch_MAX
(dbd-start is 0-based as stored in tf_sequences.parquet; converted to 1-based resi.)
"""
from __future__ import annotations
import argparse, json, os, urllib.request
from pathlib import Path

PURPLE = "#7A3E9D"
GRAY = "#BEBEBE"


def fetch_af_pdb(uniprot, dest):
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
    with urllib.request.urlopen(api, timeout=30) as r:
        meta = json.load(r)[0]
    url = meta["pdbUrl"]
    urllib.request.urlretrieve(url, dest)
    return meta, url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniprot", required=True)
    ap.add_argument("--gene", required=True)
    ap.add_argument("--dbd-start", type=int, required=True, help="0-based start")
    ap.add_argument("--dbd-end", type=int, required=True, help="end (inclusive, 1-based)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    pdb = out / f"AF_{args.uniprot}.pdb"
    meta, url = fetch_af_pdb(args.uniprot, pdb)
    L = meta["sequenceEnd"]
    r0, r1 = args.dbd_start + 1, args.dbd_end   # -> 1-based inclusive resi range

    pml = f"""# TFScope architecture: {args.gene} ({args.uniprot}), full-length AF model, DBD highlighted
load {pdb.resolve()}, tf
bg_color white
hide everything
show cartoon
set cartoon_transparency, 0.0
set cartoon_fancy_helices, 1
set ray_shadows, 0
set ray_opaque_background, 1
set ambient, 0.5
set antialias, 2
set_color dbdpurple, [0.478, 0.243, 0.616]
set_color restgray, [0.745, 0.745, 0.745]
color restgray, tf
select dbd, tf and resi {r0}-{r1}
color dbdpurple, dbd
orient tf
zoom tf, 3
ray 1500, 1500
png {out.resolve()}/{args.gene}_arch.png, dpi=300
# a second view highlighting the DBD centered
orient dbd
zoom tf, 2
ray 1500, 1500
png {out.resolve()}/{args.gene}_arch_dbdview.png, dpi=300
"""
    (out / f"{args.gene}_arch.pml").write_text(pml)
    print(f"[af] {args.gene} {args.uniprot}  len={L}  DBD resi {r0}-{r1}  pLDDT={meta['globalMetricValue']}")
    print(f"[af] model: {url}")
    print(f"[af] wrote {pdb.name} and {args.gene}_arch.pml")


if __name__ == "__main__":
    main()
