# Fig. 2b — DeepPBS-style residue narrative (in-silico mutagenesis → interface chemistry)

These paragraphs mirror the DeepPBS p53 passage: pick a complex, show the model's
per-residue importance on the structure, then walk through the top residues and the
specific chemical interaction (H-bond / vdW / electrostatic) each one makes. Every
residue number, partner base and distance below is read directly from the crystal
structure by `scripts/make_pymol_investigation.py`; importance/rank are from the
TFScope in-silico alanine scan (`results/pymol_investigation/<GENE>/<GENE>_residues.csv`).

---

## ZBTB7A — C2H2 zinc finger (PDB 7N5V:A), AUROC 0.94, top-20 recovers 5/6 contacts

Whereas DeepPBS reads importance off the protein–DNA structure, TFScope is given **only
the amino-acid sequence**; its per-residue importance is obtained by an in-silico
alanine scan, the mean absolute change in the predicted motif when each residue is
muted (analogous to DeepPBS's edge-perturbation readout, but computed without any
structural input). As an example, we examined the protein–DNA interface of ZBTB7A
(LRF/Pokemon; PDB ID 7N5V), a C2H2 zinc-finger oncoprotein that represses tumour-
suppressor loci and binds a G-rich element through its tandem zinc fingers. We mapped
the TFScope importance scores (min–max normalised) onto the recognition helices
(Fig. 2b); residue spheres are sized and coloured by importance, and the residues that
actually contact a DNA base in the crystal are ringed. The single most important
residue identified from sequence alone, **Arg421**, donates a hydrogen bond to the
**N7 of a major-groove guanine (2.9 Å)** — the canonical arginine–guanine recognition
that anchors C2H2 readout. The network deems **Lys396–guanine O6 (2.8 Å)** and
**Arg399–guanine O6 (3.3 Å)** as two further strong drivers, jointly explaining the
G-rich core of the ZBTB7A motif. **Asp423** confers specificity for the opposite strand
by hydrogen-bonding the **N4 of cytosine (3.2 Å)**, reading the C that base-pairs the
recognised G, and **Lys424** packs against a third guanine O6 (3.8 Å). Five of the six
crystallographic base contacts fall within TFScope's top-20 residues, and the importance
peaks co-localise with the recognition helices at the major-groove interface even though
the model never saw the structure. The resulting specificity prediction aligns with the
known G-rich preference of ZBTB7A, indicating that the model's sequence-only importance
recovers the same base-contacting chemistry that a structure-based method reads off the
complex.

---

## DUX4 — double homeodomain (PDB 5ZFY:A), AUROC 0.76, top-20 recovers 6/12 contacts

As a second, mechanistically distinct example we examined DUX4 (PDB ID 5ZFY), a double-
homeodomain factor whose mis-expression drives facioscapulohumeral muscular dystrophy and
which activates the cleavage-stage zygotic programme. Each homeodomain inserts its
recognition helix into the major groove. TFScope's importance again concentrates on the
base-reading residues: **Asn144** hydrogen-bonds the **N6 of adenine (2.8 Å)** and
**Asn69** the **N7 of adenine (3.0 Å)** — the hallmark homeodomain asparagine that
specifies the invariant A of the TAAT/TGAT core — while **Gln68** and **Gln143** read the
adjacent thymine through van der Waals contact with its C7 methyl (3.7 Å), and **Ile65**
and **Ile140** make hydrophobic packing contacts against the adenine ring (3.8–4.0 Å).
The two flanking arginines (Arg18, Arg71) sit over the backbone phosphates, contributing
electrostatic affinity rather than base specificity. Six of the twelve base contacts lie
in the top-20, and the highest-importance residues reproduce the asparagine–adenine and
glutamine–thymine chemistry that defines homeodomain readout.

---

### How to reproduce / investigate in PyMOL
```
pymol results/pymol_investigation/ZBTB7A/ZBTB7A_pymol.pml
```
- protein cartoon is coloured white→red by TFScope importance (B-factor channel)
- DNA shown as grey rings; TFScope top-20 residues drawn as orange sticks + labelled
- cross-check each labelled residue against `<GENE>_residues.csv` (column `interaction`,
  `nearest_base`, `d_base`) to confirm the force type before writing it up
- to switch examples: `python scripts/make_pymol_investigation.py <GENE> <PDBID> <CHAIN>`
