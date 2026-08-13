# TFScope importance investigation — ZBTB7A (7N5V:A)
load /afs/csail.mit.edu/u/l/leihuang/project/TFScope/results/pymol_investigation/ZBTB7A/ZBTB7A_importance.pdb, ZBTB7A
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
select tfscope_top, chain A and resi 385+393+396+399+400+401+404+405+406+410+414+419+421+423+424+425+428+430+432+433
show sticks, tfscope_top and not name N+C+O
color orange, tfscope_top and name C*
label tfscope_top and name CA, "%s%s" % (one_letter[resn], resi)
set label_size, 14
deselect
orient
set ray_opaque_background, 0
# color scale: white (low importance) -> red (high). sticks = top-20 hits.
