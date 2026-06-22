# TFScope Fig 2b render — ZBTB7A (7N5V:A)  (headless: pymol -cq this.pml)
load /afs/csail.mit.edu/u/l/leihuang/project/TFScope/results/pymol_investigation/ZBTB7A/ZBTB7A_importance.pdb, ZBTB7A
bg_color white
hide everything
set ray_shadows, 0
set ray_opaque_background, 1
set ray_trace_mode, 0
set cartoon_fancy_helices, 1
set cartoon_transparency, 0.15
set spec_reflect, 0.15
set ambient, 0.45
set antialias, 2
set float_labels, 1
set label_size, 30
set label_color, black
set label_font_id, 7
set label_outline_color, white
# protein cartoon coloured low->high TFScope importance (B-factor); blue=low so it stays visible on white
show cartoon, polymer.protein and chain A
spectrum b, blue_white_red, polymer.protein and chain A
# DNA as grey ringed cartoon + faint sticks
show cartoon, polymer.nucleic
set cartoon_ring_mode, 3
set cartoon_ring_finder, 1
color grey70, polymer.nucleic
show sticks, polymer.nucleic
set stick_radius, 0.10, polymer.nucleic
color grey55, polymer.nucleic
# contact-making top hits: orange sticks + labels + H-bond dashes to DNA bases
select hits, chain A and resi 396+399+421+423+424
show sticks, hits and not name N+C+O
util.cnc("hits")
color orange, hits and elem C
set stick_radius, 0.24, hits
label hits and name CA, "%s%s" % (one_letter[resn], resi)
set label_position, (0, 0, 3)
distance hbonds, (hits and (elem N+O)), (polymer.nucleic and (elem N+O)), 3.6, mode=0
color black, hbonds
hide labels, hbonds
set dash_width, 3.5
set dash_gap, 0.35
set dash_radius, 0.06
deselect
# frame the protein-DNA interface (zoom on the hits, keep a little DNA context)
orient hits
zoom hits, 7.0
turn y, 6
ray 2200, 1700
png /afs/csail.mit.edu/u/l/leihuang/project/TFScope/results/pymol_investigation/ZBTB7A/ZBTB7A_render.png, dpi=300
