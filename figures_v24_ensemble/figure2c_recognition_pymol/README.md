# Fig 2c (structural) — DNA-contact residues recognized by TFScope

**1B72** — human PBX1 homeodomain bound to DNA (Homeobox_KN family).

- **Teal cartoon:** TF DBD; recognition helix inserts into the DNA major groove.
- **Orange sticks + dashes:** residues that TFScope's frozen-ESM contact probe
  scores as DNA-contacting, shown with H-bond dashes to the DNA bases they contact.
  These match the true 4.5 Å co-crystal contacts (Y260, K266, R283, N286, ...).
- The contact probe is a property of the frozen ESM-2 backbone shared by **all
  5 ensemble members** (seed-independent), so no register-averaging is applied here.

**Metric provenance** (`esm_contact_probe_summary.json`): frozen ESM-2 650M layer-33
residue embeddings → linear contact probe, GroupKFold by protein sequence (no TF
across folds), 4.5 Å contact cutoff, 2324 complexes / 240k DBD residues.
Held-out **AUROC 0.95** for predicting DNA-contact residues (see
`results/esm_contact_diagnostic/contact_probe_auroc_auprc.pdf`).

**Files**
- `recognition_pymol_1B72_PBX1.png` — clean render (main panel)
- `recognition_pymol_1B72_PBX1_labeled.png` — with residue labels
- `recognition_pymol_1B72_PBX1.pml` — PyMOL script (reproduce: `pymol -cq *.pml`)
- `recognition_residues_1B72_raw.csv` — per-residue contact-probe score + true label
