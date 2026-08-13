#!/usr/bin/env python
"""Render a protein-DNA complex with ESM-predicted DNA-contact residues.

ESM-only companion to the Fig-2b PyMOL kit: instead of TFScope importance, the
per-residue *ESM contact-probe score* is written into the B-factor column (for a
`spectrum b` gradient), and the true DNA-contact residues are shown as orange
labeled sticks with dashed H-bonds to the DNA.

Usage:
  <tfscope-python> scripts/render_esm_contacts_pymol.py \
      --complex 1B72_0_B_WITH_DE \
      --residues results/esm_contact_diagnostic/case_study_residues.csv \
      --outdir results/pymol_investigation/1B72_ESM
Then render:  <pymol-python-env>/pymol -cq <outdir>/<name>_render.pml
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser, is_aa

PDBDIR = "/data1/leihuang/TFlow/data/TF_split_index"
NAME_RE = re.compile(r"^([0-9A-Za-z]{4})_(\d+)_([A-Za-z0-9])_WITH_([A-Za-z0-9]+)$")


def ordered_protein_resids(pdb_file, chain_id):
    st = PDBParser(QUIET=True).get_structure("x", pdb_file)
    model = next(st.get_models())
    out = []
    for res in model[chain_id].get_residues():
        if is_aa(res, standard=False) and not res.id[0].strip():
            out.append((int(res.id[1]), res.get_resname().upper()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", required=True)
    ap.add_argument("--residues", required=True,
                    help="CSV with position, amino_acid, label, score")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score-threshold", type=float, default=0.5)
    args = ap.parse_args()

    m = NAME_RE.match(args.complex)
    pdbid, _, pchain, dna = m.groups()
    dna_chains = list(dict.fromkeys(dna))
    pdb_file = f"{PDBDIR}/{args.complex}.pdb"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.residues).sort_values("position").reset_index(drop=True)
    resids = ordered_protein_resids(pdb_file, pchain)  # (author_resSeq, resname3)

    # DBD position p (1-based over the chain sequence) -> the p-th modeled residue.
    from Bio.Data.PDBData import protein_letters_3to1_extended as T2O
    score_by_resid, contact_resids = {}, []
    label_txt = []
    for _, r in df.iterrows():
        p = int(r["position"])
        author, resn3 = resids[p - 1]
        one = T2O.get(resn3, "X")
        assert one == r["amino_acid"], (
            f"aa mismatch at pos {p}: pdb {one} vs csv {r['amino_acid']}")
        score_by_resid[author] = float(r["score"])
        if int(r["label"]) == 1:
            contact_resids.append(author)
            label_txt.append(f"{one}{author}")

    # --- write B-factor = 100*score PDB (chain B), 0 elsewhere ---
    imp_pdb = out / f"{pdbid}_esm.pdb"
    with open(pdb_file) as fh, open(imp_pdb, "w") as w:
        for line in fh:
            if line[:6] in ("ATOM  ", "HETATM") and line[21] == pchain:
                author = int(line[22:26])
                b = 100.0 * score_by_resid.get(author, 0.0)
                line = line[:60] + f"{b:6.2f}" + line[66:]
            elif line[:6] in ("ATOM  ", "HETATM"):
                line = line[:60] + f"{0.0:6.2f}" + line[66:]
            w.write(line)

    contact_sel = "+".join(str(x) for x in contact_resids)
    dna_sel = "+".join(dna_chains)

    pml = f"""# ESM DNA-contact render -- {args.complex}  ({pdbid}:{pchain})  headless: pymol -cq this.pml
python
one_letter = {{'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}}
python end
load {imp_pdb.resolve()}, cplx
bg_color white
hide everything
set ray_shadows, 0
set ray_opaque_background, 1
set cartoon_fancy_helices, 1
set cartoon_transparency, 0.0
set cartoon_side_chain_helper, 1
set ambient, 0.5
set antialias, 2
set ray_trace_mode, 1
set ray_trace_color, grey40
set float_labels, 1
set label_size, 15
set label_color, black
set label_font_id, 7
set label_outline_color, white

set_color esmteal, [0.165, 0.616, 0.561]
set_color esmorange, [0.906, 0.435, 0.318]

# protein cartoon: solid teal
show cartoon, polymer.protein and chain {pchain}
color esmteal, polymer.protein and chain {pchain}

# DNA as grey ringed cartoon + faint sticks
show cartoon, polymer.nucleic
set cartoon_ring_mode, 3
set cartoon_ring_finder, 1
color grey75, polymer.nucleic
show sticks, polymer.nucleic
set stick_radius, 0.10, polymer.nucleic
color grey60, polymer.nucleic

# true DNA-contact residues: orange sticks + labels + H-bond dashes (applied LAST)
select contacts, chain {pchain} and resi {contact_sel}
show sticks, contacts and not name N+C+O
util.cnc("contacts")
color esmorange, contacts and elem C
set stick_radius, 0.26, contacts
distance hbonds, (contacts and (elem N+O)), (polymer.nucleic and chain {dna_sel} and (elem N+O)), 3.6, mode=2
color grey20, hbonds
hide labels, hbonds
set dash_width, 3.0

orient cplx
zoom cplx, 2

# (1) clean render: sticks + H-bonds, no text
ray 1800, 1300
png {out.resolve()}/{pdbid}_esm_render.png, dpi=300

# (2) labeled render: small residue labels on the alpha carbons
label contacts and name CA, "%s%s" % (one_letter[resn], resi)
set label_position, (1.2, 0.8, 2.5)
ray 1800, 1300
png {out.resolve()}/{pdbid}_esm_render_labeled.png, dpi=300
"""
    pml_path = out / f"{pdbid}_esm_render.pml"
    pml_path.write_text(pml)

    # residue table
    tab = df.copy()
    tab["author_resid"] = [resids[int(p) - 1][0] for p in tab["position"]]
    tab.to_csv(out / f"{pdbid}_esm_residues.csv", index=False)

    print(f"[render] complex {args.complex}  protein chain {pchain}  DNA {dna_chains}")
    print(f"[render] {len(contact_resids)} contact residues: {label_txt}")
    print(f"[render] wrote {imp_pdb.name}, {pml_path.name}")
    print(f"[render] now run: pymol -cq {pml_path.resolve()}")


if __name__ == "__main__":
    main()
