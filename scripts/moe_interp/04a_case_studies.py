#!/usr/bin/env python
"""Assemble the two panel-D case studies and the PyMOL inset specification.

Picks one C2H2 zinc-finger row and one bHLH row from the co-crystal subset, joins the
per-residue routing to the structural annotation, and resolves PDB auth numbering (which
the annotation table does not carry) so the ribbon inset can address the same residues.

Usage:
  python scripts/moe_interp/04a_case_studies.py [--indir results/moe_biochem]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import protein_letters_3to1

warnings.filterwarnings("ignore")

CIF_DIR = "data/raw/pdb_cif_cache"
# chosen for structural clarity: three complete C2H2 fingers, and a bHLH crop that spans
# the basic region plus helix 1 of the dimerisation module
CASES = [
    dict(key="c2h2", filename="str_221", label="Egr1 (Zif268) C2H2 zinc fingers"),
    dict(key="bhlh", filename="str_685", label="MYC bHLH"),
]


def auth_ids(pdb_id, chain_id, offset, n):
    for nm in (f"{pdb_id.lower()}.cif", f"{pdb_id.upper()}.cif"):
        p = os.path.join(CIF_DIR, nm)
        if os.path.exists(p):
            break
    else:
        return None
    model = MMCIFParser(QUIET=True).get_structure(pdb_id, p)[0]
    res = [r for r in model[chain_id]
           if r.id[0] == " " and r.get_resname().strip() in protein_letters_3to1]
    return [int(r.id[1]) for r in res[offset:offset + n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/moe_biochem")
    ap.add_argument("--outdir", default="figures_v24/figure_moe_biochemical_programs",
                    help="where the rendered insets will be written")
    a = ap.parse_args()

    tok = pd.read_parquet(os.path.join(a.indir, "routing_seed42.parquet"))
    ptab = pd.read_parquet(os.path.join(a.indir, "protein_table.parquet"))
    ann = pd.read_parquet(os.path.join(a.indir, "structural_annotations.parquet"))
    mp = pd.read_csv(os.path.join(a.indir, "structure_mapping.csv"))

    out = {}
    for case in CASES:
        row = ptab[ptab.filename == case["filename"]]
        if not len(row):
            print(f"!! {case['filename']} not in the protein table -- skipped")
            continue
        row = row.iloc[0]
        m = mp[mp.filename == case["filename"]]
        if not len(m) or pd.isna(m.iloc[0].pdb):
            print(f"!! {case['filename']} has no structure mapping -- skipped")
            continue
        m = m.iloc[0]
        d = (ann[ann.prot == row.prot]
             .merge(tok[tok.prot == row.prot][["pos", "e1", "e2", "w1", "base_contact_v23"]],
                    on="pos", how="left")
             .sort_values("pos"))
        ids = auth_ids(m.pdb, m.chain, int(m.offset), len(d))
        d["auth_resid"] = ids if ids and len(ids) == len(d) else -1
        out[case["key"]] = dict(
            filename=case["filename"], gene=str(row.gene_symbol),
            family=str(row.family_name), label=case["label"],
            pdb=str(m.pdb), chain=str(m.chain), offset=int(m.offset),
            contact_recall=float(m.contact_recall),
            seq="".join(d.aa),
            pos=[int(x) for x in d.pos],
            auth_resid=[int(x) for x in d.auth_resid],
            e1=[int(x) for x in d.e1], e2=[int(x) for x in d.e2],
            base_contact=[bool(x) for x in d.base_contact],
            backbone_contact=[bool(x) for x in d.backbone_contact],
            major_groove=[bool(x) for x in d.major_groove],
            zn_coord=[bool(x) for x in d.zn_coord],
            buried_core=[bool(x) for x in d.buried_core],
            helix=[bool(x) for x in d.helix],
            ppi=[bool(x) for x in d.ppi],
            rsa=[round(float(x), 3) for x in d.rsa],
        )
        print(f"{case['key']}: {row.gene_symbol} {case['filename']} -> {m.pdb}/{m.chain} "
              f"offset={m.offset} recall={m.contact_recall} n={len(d)}  "
              f"Zn={int(d.zn_coord.sum())} base={int(d.base_contact.sum())} "
              f"ppi={int(d.ppi.sum())} helix={int(d.helix.sum())}")
        print(f"   experts along the chain: {''.join(str(x) for x in d.e1)}")

    with open(os.path.join(a.indir, "case_studies.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {a.indir}/case_studies.json")

    # PyMOL inset spec. Accents are chosen from the *structural-role* regressions, not
    # from the v23 contact axis: one expert with the strongest supported scaffold role
    # (Zn coordination where the domain has zinc, otherwise burial) and one with the
    # strongest supported DNA base-readout role, restricted to experts this chain uses.
    role = pd.read_csv(os.path.join(a.indir, "expert_role_stats.csv"))
    role = role[(role.fdr < 0.05) & (~role.unstable.astype(bool)) & (role.effect_size > 0)]

    def pick(feature, used):
        g = role[(role.feature == feature) & (role.expert.isin(used))]
        if not len(g):
            return None
        return int(g.loc[g.effect_size.idxmax()].expert)

    specs = []
    for key, c in out.items():
        e1 = np.array(c["e1"])
        used = [e for e in range(int(role.expert.max()) + 1) if (e1 == e).sum() >= 3]
        has_zn = any(c["zn_coord"])
        scaf = pick("zn_coord", used) if has_zn else pick("buried_core", used)
        read = pick("base_contact", used) or pick("backbone_contact", used)
        if scaf is None or read is None or scaf == read:
            print(f"{key}: no distinct scaffold/readout accent pair -- inset skipped")
            continue
        specs.append(dict(
            out=os.path.join(a.outdir, f"inset_{key}.png"),
            pdb=c["pdb"], chain=c["chain"], buffer=1.5,
            scaffold_expert=scaf, readout_expert=read,
            accents=[
                dict(expert=scaf, color="#2C5985",
                     resi=[c["auth_resid"][i] for i in np.where(e1 == scaf)[0]]),
                dict(expert=read, color="#AC4A31",
                     resi=[c["auth_resid"][i] for i in np.where(e1 == read)[0]]),
            ]))
        out[key]["_accents"] = dict(scaffold=scaf, readout=read)
        print(f"{key} inset: scaffold accent e{scaf} "
              f"({'Zn coordination' if has_zn else 'buried core'}), "
              f"readout accent e{read} (DNA contact)")

    with open(os.path.join(a.indir, "case_studies.json"), "w") as f:
        json.dump(out, f, indent=1)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.indir, "inset_spec.json"), "w") as f:
        json.dump(specs, f, indent=1)
    print(f"saved {a.indir}/inset_spec.json")


if __name__ == "__main__":
    main()
