# ESM DNA-contact render -- 1B72_0_B_WITH_DE  (1B72:B)  headless: pymol -cq this.pml
python
one_letter = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
python end
load /afs/csail.mit.edu/u/l/leihuang/project/TFScope/results/pymol_investigation/1B72_ESM/1B72_esm.pdb, cplx
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
show cartoon, polymer.protein and chain B
color esmteal, polymer.protein and chain B

# DNA as grey ringed cartoon + faint sticks
show cartoon, polymer.nucleic
set cartoon_ring_mode, 3
set cartoon_ring_finder, 1
color grey75, polymer.nucleic
show sticks, polymer.nucleic
set stick_radius, 0.10, polymer.nucleic
color grey60, polymer.nucleic

# true DNA-contact residues: orange sticks + labels + H-bond dashes (applied LAST)
select contacts, chain B and resi 260+266+279+282+283+285+286+288+289+290
show sticks, contacts and not name N+C+O
util.cnc("contacts")
color esmorange, contacts and elem C
set stick_radius, 0.26, contacts
distance hbonds, (contacts and (elem N+O)), (polymer.nucleic and chain D+E and (elem N+O)), 3.6, mode=2
color grey20, hbonds
hide labels, hbonds
set dash_width, 3.0

orient cplx
zoom cplx, 2

# (1) clean render: sticks + H-bonds, no text
ray 1800, 1300
png /afs/csail.mit.edu/u/l/leihuang/project/TFScope/results/pymol_investigation/1B72_ESM/1B72_esm_render.png, dpi=300

# (2) labeled render: small residue labels on the alpha carbons
label contacts and name CA, "%s%s" % (one_letter[resn], resi)
set label_position, (1.2, 0.8, 2.5)
ray 1800, 1300
png /afs/csail.mit.edu/u/l/leihuang/project/TFScope/results/pymol_investigation/1B72_ESM/1B72_esm_render_labeled.png, dpi=300
