"""Render a labelled high-res PyMOL panel for a TF (headless).
Headless PyMOL won't rasterise text labels, so we render TWICE at the SAME view:
  main.png    — the publication scene (importance cartoon + DNA + orange contact sticks + H-bond dashes)
  markers.png — only the hit CA atoms, each a unique flat colour
then composite residue labels with PIL at each marker's pixel centroid (step 2, tfscope env).

Step 1 (pymol env):  /data1/leihuang/miniconda3/envs/pymol/bin/python scripts/render_pymol_labeled.py GENE PDBID CHAIN
Step 2 (tfscope env): python scripts/render_pymol_labeled.py --compose GENE
"""
import os, sys, json, csv

GENE  = None; PDBID = None; CHAIN = "A"; COMPOSE = "--compose" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
GENE = args[0]
if not COMPOSE:
    PDBID, CHAIN = args[1], (args[2] if len(args) > 2 else "A")
OUT = f"results/pymol_investigation/{GENE}"
W, H = 2200, 1700

# distinct flat marker colours (R,G,B 0-255)
PALETTE = [(230,25,75),(60,180,75),(0,130,200),(245,130,48),(145,30,180),
           (70,240,240),(240,50,230),(210,245,60),(0,128,128),(170,110,40)]

def load_hits():
    rows = list(csv.DictReader(open(f"{OUT}/{GENE}_residues.csv")))
    hits = [r for r in rows if r["crystal_contact"] == "True" and r["top20"] == "True"]
    hits.sort(key=lambda r: int(r["rank"]))
    return [(int(r["resnum"]), f"{r['aa']}{r['resnum']}") for r in hits]

# ───────────────────────── step 1: render (pymol env) ─────────────────────────
def render():
    from pymol import cmd
    hits = load_hits()
    resi = "+".join(str(r) for r, _ in hits)
    cmd.load(os.path.abspath(f"{OUT}/{GENE}_importance.pdb"), GENE)
    cmd.bg_color("white")
    cmd.hide("everything")
    cmd.set("ray_shadows", 0); cmd.set("ray_opaque_background", 1)
    cmd.set("ray_trace_mode", 0); cmd.set("antialias", 2)
    cmd.set("cartoon_fancy_helices", 1); cmd.set("cartoon_transparency", 0.15)
    cmd.set("ambient", 0.45); cmd.set("spec_reflect", 0.15)
    cmd.show("cartoon", f"polymer.protein and chain {CHAIN}")
    cmd.spectrum("b", "blue_white_red", f"polymer.protein and chain {CHAIN}")
    cmd.show("cartoon", "polymer.nucleic")
    cmd.set("cartoon_ring_mode", 3); cmd.set("cartoon_ring_finder", 1)
    cmd.color("grey70", "polymer.nucleic")
    cmd.show("sticks", "polymer.nucleic"); cmd.set("stick_radius", 0.10, "polymer.nucleic")
    cmd.color("grey55", "polymer.nucleic")
    cmd.select("hits", f"chain {CHAIN} and resi {resi}")
    cmd.show("sticks", "hits and not name N+C+O")
    cmd.util.cnc("hits"); cmd.color("orange", "hits and elem C")
    cmd.set("stick_radius", 0.24, "hits")
    cmd.distance("hbonds", "(hits and (elem N+O))", "(polymer.nucleic and (elem N+O))", 3.6, mode=0)
    cmd.color("black", "hbonds"); cmd.hide("labels", "hbonds")
    cmd.set("dash_width", 3.5); cmd.set("dash_gap", 0.35); cmd.set("dash_radius", 0.06)
    cmd.deselect()
    cmd.orient("hits"); cmd.zoom("hits", 7.0); cmd.turn("y", 6)
    view = cmd.get_view()
    cmd.ray(W, H); cmd.png(f"{OUT}/{GENE}_render.png", dpi=300)

    # twin marker pass: only hit CAs, each a unique flat colour, same view
    cmd.hide("everything"); cmd.bg_color("white")
    cmd.set("antialias", 0); cmd.set("ray_shadows", 0)
    cmd.set("ambient", 1.0); cmd.set("spec_reflect", 0.0); cmd.set("specular", 0.0)
    cmap = {}
    for i, (rnum, name) in enumerate(hits):
        c = PALETTE[i % len(PALETTE)]
        cn = f"mk{i}"
        cmd.set_color(cn, [v / 255.0 for v in c])
        sel = f"chain {CHAIN} and resi {rnum} and name CA"
        cmd.show("spheres", sel); cmd.set("sphere_scale", 0.6, sel); cmd.color(cn, sel)
        cmap[name] = c
    cmd.set_view(view)
    cmd.ray(W, H); cmd.png(f"{OUT}/{GENE}_markers.png", dpi=300)
    json.dump({"size": [W, H], "labels": [{"name": n, "color": cmap[n]} for _, n in hits]},
              open(f"{OUT}/{GENE}_markers.json", "w"), indent=1)
    print(f"rendered {OUT}/{GENE}_render.png + markers (n={len(hits)})")

# ───────────────────────── step 2: compose labels (tfscope env) ───────────────
def compose():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    meta = json.load(open(f"{OUT}/{GENE}_markers.json"))
    main = Image.open(f"{OUT}/{GENE}_render.png").convert("RGB")
    mk = np.asarray(Image.open(f"{OUT}/{GENE}_markers.png").convert("RGB")).astype(int)
    draw = ImageDraw.Draw(main)
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/data1/leihuang/miniconda3/envs/tfscope/fonts/DejaVuSans-Bold.ttf"]:
        try:
            font = ImageFont.truetype(fp, 46); break
        except Exception:
            font = None
    if font is None:
        import matplotlib
        font = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__),
                                 "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"), 46)
    placed = []
    for lab in meta["labels"]:
        col = np.array(lab["color"])
        d = np.abs(mk - col).sum(2)
        ys, xs = np.where(d < 40)
        if len(xs) == 0: continue
        cx, cy = float(xs.mean()), float(ys.mean())
        # nudge label off the marker, avoid overlapping previous labels
        ox, oy = cx + 18, cy - 60
        for px, py in placed:
            if abs(ox - px) < 120 and abs(oy - py) < 50: oy -= 56
        placed.append((ox, oy))
        # leader line marker -> label
        draw.line([(cx, cy), (ox + 6, oy + 40)], fill=(30, 30, 30), width=3)
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(20, 20, 20))
        txt = lab["name"]
        draw.text((ox, oy), txt, font=font, fill=(0, 0, 0),
                  stroke_width=5, stroke_fill=(255, 255, 255))
    out = f"{OUT}/{GENE}_render_labeled.png"
    main.save(out)
    print(f"saved {out}")

render() if not COMPOSE else compose()
