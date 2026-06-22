# Fig. 2c — recognition-code validation (rCLAMPS / Wetzel, Zhang & Singh 2022)

Source data: rCLAMPS repo (github.com/jlwetzel-slab/rCLAMPS), cloned to
/data1/leihuang/rCLAMPS. Recognition positions: homeodomain = Pfam Homeobox HMM
match states {2,3,4,5,47,50,51,54,55}; C2H2 = canonical helix code {-1,+2,+3,+6}.
Builder: scripts/build_fig2c_recognition_code.py → results/per_family/fig2c_recognition_code.json.
Key numbers: homeodomain n=15, median per-TF AUROC 0.69, 93% above chance, recognition
z +0.62 vs rest -0.05. C2H2 (supplement): Zn-ligand z +0.54 > code z +0.11.

---

## Subsection: "Residue importance recapitulates an experimentally derived recognition code"

> The crystal-contact analysis (Fig. 2b) compares the model's importance against the
> physical interface of each specific complex. To test whether TFScope instead captures
> the transferable, family-level recognition logic — the mapping between base-contacting
> amino acids and the nucleotides they specify — we compared its residue importance to
> rCLAMPS, a probabilistic recognition code learned independently from DNA-binding
> specificities across each structural family (Wetzel, Zhang & Singh, 2022). This ground
> truth derives from binding data rather than from any individual structure, and is
> independent of TFScope's training. We aligned every test homeodomain to the Pfam
> Homeobox profile HMM used by rCLAMPS and projected the model's in-silico alanine-scan
> importance onto the alignment columns. Averaged across the 15 homeodomains, importance
> concentrates sharply on the recognition-helix-3 and N-terminal-arm columns that rCLAMPS
> identifies as base-reading, peaking at the canonical specificity residue Asn51 and its
> neighbour at position 50 (Fig. 2c, left); recognition columns carry a mean importance of
> +0.62 s.d. versus −0.05 s.d. for the remainder of the domain. At the level of individual
> factors, importance ranks the recognition-code positions above the rest of the domain with
> a median AUROC of 0.69 (93% of factors above chance), and does so consistently with the
> orthogonal crystal-contact labels of Fig. 2b (Fig. 2c, right). That the same sequence-only
> importance recovers two independent definitions of the interface — the physical contacts of
> a given complex and a binding-data-derived recognition code — indicates that TFScope has
> internalised the recognition logic of the family rather than memorising a particular
> structure.

**Optional Discussion limitation sentence (for the C2H2 supplement):**

> In metal-coordinating families the in-silico alanine scan additionally highlights
> structural residues: for C2H2 zinc fingers, the invariant Zn-coordinating cysteines and
> histidines receive the largest importance (Supplementary Fig. SX), because muting them is
> predicted to abolish binding altogether. The scan therefore conflates structural and
> base-reading roles in metalloproteins — a confound that accounts for the lower tail of the
> contact-recovery distribution in Fig. 2b and that the metal-free homeodomain analysis
> avoids; disentangling the two is a target for future attention-based attribution.

---

## Figure caption

> **(c)** TFScope residue importance recapitulates the rCLAMPS recognition code (Wetzel et
> al. 2022), an independent, binding-data-derived map of base-reading positions. **Left,**
> mean in-silico alanine-scan importance (z-scored per factor) across Pfam Homeobox HMM
> positions for the 15 test homeodomains; rCLAMPS recognition positions in red, peaking at
> recognition helix-3 (position 50/Asn51) and the N-terminal arm. **Right,** per-transcription-
> factor AUROC with which importance ranks the target positions above the rest of the domain,
> using two independent ground truths — crystal base contacts (Fig. 2b) and the rCLAMPS code;
> bars, medians; dotted line, chance.

## Supplementary caption

> **Supplementary Fig. SX.** Alanine-scan importance is confounded by structural metal
> ligands in C2H2 zinc fingers (n=8 test factors). Per-residue importance (z-score) at the
> canonical recognition-helix code positions (−1, +2, +3, +6), other domain positions, and the
> Zn-coordinating cysteines/histidines. The structural Zn ligands receive the highest
> importance, since predicted binding collapses when they are removed.
