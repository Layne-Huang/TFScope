#!/usr/bin/env python
"""Small ribbon insets for panel D, rendered with PyMOL.

Draws the co-crystal the case-study row was mapped to: protein backbone in neutral
grey, DNA in a light neutral tone, Zn as a small sphere, and the residues routed to the
two accent experts as coloured sticks.  Deliberately flat/cartoon-like -- the inset is
an annotation, not a molecular-graphics showpiece.

Run with the pymol env, not tfscope:
  /data1/leihuang/miniconda3/envs/pymol/bin/python scripts/moe_interp/05_render_inset.py \
      --spec results/moe_biochem/inset_spec.json
"""
from __future__ import annotations

import argparse
import json
import os

import pymol
from pymol import cmd

CIF_DIR = "data/raw/pdb_cif_cache"


def render(spec, outpath, width=1100, height=1000):
    cmd.reinitialize()
    path = None
    for nm in (f"{spec['pdb'].lower()}.cif", f"{spec['pdb'].upper()}.cif"):
        p = os.path.join(CIF_DIR, nm)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise SystemExit(f"no CIF for {spec['pdb']}")
    cmd.load(path, "st")
    cmd.remove("solvent")
    cmd.remove("hydrogens")

    ch = spec["chain"]
    focus = f"polymer.protein and chain {ch}"
    # Drop crystallographic copies: keep a partner protein chain only if it forms a real
    # interface with the annotated chain. Trim DNA to the duplex around that chain.
    for c in {a.chain for a in cmd.get_model("polymer.protein").atom} - {ch}:
        n_iface = cmd.count_atoms(
            f"(polymer.protein and chain {c} and name CA) within 8 of ({focus})")
        if n_iface < 12:
            cmd.remove(f"polymer.protein and chain {c}")
    cmd.remove(f"byres (polymer.nucleic and not (polymer.nucleic within 14 of ({focus})))")
    cmd.remove(f"elem ZN and not (elem ZN within 4 of ({focus}))")

    cmd.hide("everything")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("specular", 0.15)
    cmd.set("ambient", 0.32)
    cmd.set("direct", 0.55)
    cmd.set("cartoon_sampling", 12)
    cmd.set("stick_radius", 0.24)
    cmd.set("sphere_scale", 0.45)
    cmd.set_color("dna_pale", [0.906, 0.882, 0.827])
    cmd.set_color("prot_grey", [0.478, 0.494, 0.510])

    prot = focus
    cmd.show("cartoon", "polymer.protein")
    cmd.color("prot_grey", "polymer.protein")
    cmd.set("cartoon_transparency", 0.6, f"polymer.protein and not chain {ch}")

    cmd.show("cartoon", "polymer.nucleic")
    cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.set("cartoon_ring_mode", 3, "polymer.nucleic")
    cmd.set("cartoon_ring_finder", 1)
    cmd.set("cartoon_ring_transparency", 0.35, "polymer.nucleic")
    cmd.color("dna_pale", "polymer.nucleic")

    cmd.show("spheres", "elem ZN")
    cmd.color("gray30", "elem ZN")

    # optionally crop to one module (e.g. a single zinc finger)
    if spec.get("resi_range"):
        lo, hi = spec["resi_range"]
        cmd.remove(f"polymer.protein and not chain {ch}")
        cmd.remove(f"({prot}) and not resi {lo}-{hi}")
        cmd.remove(f"byres (polymer.nucleic and not "
                   f"(polymer.nucleic within 12 of ({prot})))")
        cmd.remove(f"elem ZN and not (elem ZN within 4 of ({prot}))")

    # accent residues, addressed by auth numbering resolved upstream
    for gi, grp in enumerate(spec["accents"]):
        col = grp["color"].lstrip("#")
        cname = f"acc_{gi}"
        cmd.set_color(cname, [int(col[i:i + 2], 16) / 255 for i in (0, 2, 4)])
        resi = "+".join(str(x) for x in grp["resi"])
        if not resi:
            continue
        sel = f"({prot}) and resi {resi}"
        style = grp.get("style", "sticks")
        if style == "spheres":
            cmd.show("spheres", f"{sel} and sidechain")
            cmd.color(cname, f"{sel} and sidechain")
        else:
            cmd.show("sticks", f"{sel} and not (name C+N+O)")
            cmd.color(cname, f"{sel} and elem C")
        if grp.get("cartoon", True):
            cmd.show("cartoon", sel)
            cmd.set("cartoon_color", cname, sel)

    cmd.orient(f"({prot}) or polymer.nucleic")
    if spec.get("turn"):
        for ax, ang in spec["turn"]:
            cmd.turn(ax, ang)
    cmd.zoom(f"({prot}) or polymer.nucleic", spec.get("buffer", 1.0))
    cmd.ray(width, height)
    cmd.png(outpath, dpi=600)
    crop_alpha(outpath)
    print(f"wrote {outpath}")


def crop_alpha(path, pad=6):
    """Trim the transparent border so the inset fills its axes."""
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    a = np.array(im)[:, :, 3]
    ys, xs = np.where(a > 4)
    if not len(ys):
        return
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad + 1, a.shape[1]), min(ys.max() + pad + 1, a.shape[0]))
    im.crop(box).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="results/moe_biochem/inset_spec.json")
    a = ap.parse_args()
    pymol.finish_launching(["pymol", "-qc"])
    specs = json.load(open(a.spec))
    for s in specs:
        render(s, s["out"])


if __name__ == "__main__":
    main()
