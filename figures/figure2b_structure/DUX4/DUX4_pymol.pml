# TFScope importance investigation — DUX4 (5ZFY:A)
load /afs/csail.mit.edu/u/l/leihuang/project/TFScope/results/pymol_investigation/DUX4/DUX4_importance.pdb, DUX4
bg_color white
hide everything
# protein coloured by TFScope importance (B-factor)
show cartoon, polymer.protein
spectrum b, white_red, polymer.protein and chain A
set cartoon_transparency, 0.1
# DNA
show cartoon, polymer.nucleic
set cartoon_ring_mode, 3
color grey70, polymer.nucleic
# TFScope top-20 important residues as sticks, labelled
select tfscope_top, chain A and resi 18+30+38+52+60+63+64+65+66+67+68+69+71+138+139+140+141+142+143+144
show sticks, tfscope_top and not name N+C+O
color orange, tfscope_top and name C*
label tfscope_top and name CA, "%s%s" % (one_letter[resn], resi)
set label_size, 14
deselect
orient
set ray_opaque_background, 0
# color scale: white (low importance) -> red (high). sticks = top-20 hits.
