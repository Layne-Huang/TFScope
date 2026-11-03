# Fig. X | Mixture-of-experts separates structural scaffolding from DNA-recognition chemistry

v24 seed42 (single model, `contact_v24_seed42/ckpt_best.pt`), residue-granularity MoE,
8 routed experts, top-2 routing, 323,751 routed DBD residues over 2,260 unique v23 DBDs.

---

## Files

Six independent subfigures, each written as PDF + SVG (text kept as text) + PNG @600 dpi:

| File | Panel | Size |
|---|---|---|
| `panel_a.{pdf,svg,png}` | Expert chemical signatures (amino-acid enrichment) | 5.3 × 2.75 in |
| `panel_b.{pdf,svg,png}` | Expert structural-role signatures | 4.2 × 2.85 in |
| `panel_c.{pdf,svg,png}` | Expert co-routing structure (top-2 routing) | 3.7 × 3.15 in |
| `panel_d.{pdf,svg,png}` | Case study: C2H2 zinc finger | 5.1 × 3.35 in |
| `panel_e.{pdf,svg,png}` | Case study: bZIP / bHLH | 4.3 × 3.35 in |
| `panel_f.{pdf,svg,png}` | Cross-family reuse of biochemical programs | 3.3 × 2.75 in |
| `figure_moe_biochemical_programs_full.{pdf,svg,png}` | all six assembled, 13.2 × 8.6 in | |
| `render_d_zincfinger.png`, `render_e_bhlh.png` | PyMOL ribbons embedded in panels d and e | |
| `superseded_2x2/` | the earlier 4-panel (a–d) layout, kept for reference | |

Statistics (in `results/moe_biochem/`):

| File | Contents |
|---|---|
| `expert_feature_stats.csv` | **the requested unified tidy table** — one row per expert × feature, both feature types |
| `expert_residue_stats.csv` | panel a: family-adjusted residue enrichment + family-stratified CMH odds ratios |
| `expert_role_stats.csv` | panel b: role ~ expert + amino acid + family logistic fits |
| `expert_role_stats_dedup.csv` | panel b refit on one crop per gene (sensitivity) |
| `aa_matched_controls.csv` | amino-acid-matched controls (the key falsification tests) |
| `corouting_stats.csv` | panel c: observed vs independence-expected top-2 pairs, permutation p |
| `corouting_by_family.csv` | per-family co-routing (supplementary) |
| `cross_seed_cosine.csv` | cross-seed Hungarian-matched profile cosine (supplementary) |
| `c2h2_register.csv` | panel d: expert usage per C2H2 recognition-helix register position |
| `bzip_bhlh_register.csv` | panel e: expert usage per bZIP/bHLH domain position |
| `family_reuse.csv` | panel f: per-family program consistency + usage enrichment, bootstrap CIs |
| `expert_summary.csv` | usage + both DNA-contact axes per expert |
| `structural_annotations.parquet` | 23,716 per-residue structural labels |
| `structure_mapping.csv` | v23 row → PDB chain, offset, contact recall, v23 index shift |
| `routing_seed42.parquet` | per-residue routing dump (top-1, top-2, gate weights) |
| `case_studies.json` | panel d per-residue records |

Pipeline (`scripts/moe_interp/`):

```
01_route_dump.py            per-residue routing dump (top-1, top-2, gate weights)
02_structural_annotations.py  crop -> co-crystal mapping + per-residue structural labels
03_stats.py                 panels a-c statistics + AA-matched controls
06_case_and_family_stats.py panels d-f statistics (registers, cross-family reuse)
04a_case_studies.py         panel d/e case-study records
08_case_renders.py          PyMOL specs for the panel d/e ribbons
05_render_inset.py          run with the pymol env to render them
07_subfigures.py            the six subfigures + assembled figure
```

---

## Figure caption

**Fig. X | Mixture-of-experts separates structural scaffolding from DNA-recognition
chemistry.**
**a,** Family-adjusted amino-acid enrichment across the eight routed experts,
log₂ P(aa | e) / Σ_f P(f | e) P(aa | f), so an expert's signature cannot be explained by
which TF families it happens to receive. Amino acids are grouped by biochemical class;
boxes mark each expert's two strongest residues. Sidebars give route usage and
DNA-contact enrichment; bars are the v23 contact-supervision set, dots the contact set
recomputed here from the co-crystals (see *Caveat*). Experts are ordered along the v23
contact axis, and the same order is used in **a**–**c**.
**b,** Structural-role enrichment on the co-crystal subset (23,716 residues, 273 crops,
126 genes, 195 PDB entries), from logistic regressions role ~ 1{expert} + C(amino acid)
+ C(TF family) with standard errors clustered by gene; a surviving effect is therefore
not attributable to residue composition or to family identity. Bubble colour and area
give the log₂ odds ratio and its magnitude, outlines mark FDR < 0.05, × marks
quasi-separated fits. Experts depleted from every DNA-contact class occupy buried,
metal-coordinating or loop roles, whereas contact-enriched experts localise to specific
DNA faces. Right-hand labels name only roles the regression supports at FDR < 0.05.
**c,** Top-2 co-routing relative to an independence expectation that preserves each
expert's marginal usage (2,000 permutations; all 28 pairs FDR < 0.05). Routing is
strongly structured: the strongest enriched pairs are e0–e1 (+1.79), e4–e7 (+1.52) and
e2–e3 (+1.32), and the strongest depletion is e1–e6 (−3.78).
**d,** C2H2 register. Every C-x(2,4)-C-x(12)-H-x(3,5)-H module in the routing table
(n = 5,475 fingers from 682 proteins) is anchored on its first His, defined as helix
position +7. Tracks give the fraction of fingers routing each register position to a
given expert. The Zn-ligating cysteines (−9, −6) go to e6 (0.67, 0.70), the Zn-ligating
histidines (+7, +11) to e4 (0.70, 0.75), and the recognition helix to e2 and e1 — so the
metal-scaffold and base-readout programs partition one domain by sub-function. The ribbon
shows one finger of PDB 1P47 with Cys (orange) and His (purple) spheres and the
recognition-helix side chains (teal) in the major groove.
**e,** bZIP / bHLH register, anchored structurally: position 0 is the last
DNA-base-contacting residue of each crop, i.e. the end of the basic region (n = 38
mapped domains). The major-groove-readout expert e0 concentrates in the basic region and
the helical-scaffold expert e3 in the dimerisation element. Ribbon: MYC in PDB 1NKP, with
the basic segment (orange) in the major groove and the dimerisation helix (blue) running
into the coiled coil; the MAX partner is faint grey.

**f,** Cross-family reuse. For each expert and TF family, the cosine between the
expert's family-controlled amino-acid profile computed *inside* that family and its
global profile: ≈1 means the same chemical operation is being reused in that clade, ≈0
means the expert is doing something unrelated there. Bars omitted where the expert has
< 200 residues in the family. Error bars are 95% protein-level bootstrap CIs (400
resamples). The helical/buried scaffold e3 is reused across all five families
(0.62–0.91), whereas the Cys–Zn program of e6 is essentially C2H2-specific
(0.83 in C2H2 vs 0.10–0.12 elsewhere).

---

## What the data support

Established (FDR < 0.05, controlled for amino-acid identity **and** TF family):

| Expert | Supported role | Amino-acid-matched control |
|---|---|---|
| e6 | Zn coordination (log₂ OR **+5.16**) | among **Cys only**: 79.7 % vs 2.9 % Zn ligands, log₂ OR +5.46, p = 3 × 10⁻⁹ |
| e4 | Zn coordination (**+7.32**) | among **His only**: 67.1 % vs 0.8 %, log₂ OR +9.55, p = 1 × 10⁻¹⁴; His base contacts *depleted* (−3.20, p = 0.002) |
| e3 | buried core (+0.65), α-helix (+0.40); depleted from all four DNA-contact classes | among **Leu only**: 70.9 % vs 41.2 % buried, +1.80, p = 2 × 10⁻⁵ |
| e5 | buried core (+0.60); strongest DNA depletion (base −3.10, major groove −3.60) | among **Gly only**: 88.6 % vs 73.1 % loop/turn, +1.79, p = 0.003 |
| e0 | major groove (+2.51), base contact (+2.05), backbone (+1.35), α-helix (+0.54) | among **Gln only**: base contact +2.62, p = 0.004 |
| e2 | base contact (+1.29), major groove (+1.22), β-strand (+0.87) | Pro/Ala/Thr loop/turn test **not significant** (p = 0.62) |
| e7 | minor groove (+3.16), β-strand (+1.50), loop (+1.08); base contact *depleted* (−1.25) | Phe/Tyr DNA-interface test **negative** (−0.55, p = 0.031) |
| e1 | DNA backbone (+0.69), buried core (+0.69), loop (+0.54); base contact n.s. | among **Arg/Lys only**: backbone 53.1 % vs 38.6 %, +1.10, p = 0.001; base contact n.s. (p = 0.37) |

Robustness: refitting panel b on one crop per gene (11,022 residues, 126 crops) preserves
the sign of **49/49** FDR-significant effects, effect sizes correlate r = 0.984, and 42/49
remain FDR < 0.05. Cross-seed Hungarian-matched chemistry profiles have median cosine
0.50 (per-seed means 0.38–0.54) against a permutation null of 0.18–0.22.

Deliberately **not** claimed:

* **e7 is not an "aromatic interface" expert.** Its raw Phe/Tyr DNA-interface rate is
  higher than background (42.8 % vs 31.2 %), but after controlling for family the effect
  reverses (log₂ OR −0.55, p = 0.031). e7 is described as minor-groove / β-element.
  No stacking geometry was computed, so no stacking claim is made.
* **e2's "local geometry / loop accommodation" hypothesis is unsupported** — the
  Pro/Ala/Thr loop-turn control is null. e2 is labelled by what does hold, base readout.
* **e4 is not labelled a Zn expert from His enrichment alone.** The label rests on the
  structural regression (+7.32) and the His-only control (+9.55), and e4 His residues are
  *depleted* for base contacts.
* **Panel c is titled "combines experts non-randomly", not "combines complementary
  programs".** Two of the four enriched pairs are complementary (e0 readout + e1 backbone;
  e2 readout + e3 scaffold) but e5–e6 pairs two scaffold programs, so the stronger claim
  is not made in the title.
* **Panel f does not claim that experts are generally family-agnostic.** Only e3 is
  reused across all five families. Two experts are strongly clade-restricted: e6's
  Cys–Zn program exists essentially only in C2H2, and e7 is used 5.4× more than average
  in homeodomains yet its chemistry there is unrelated to its global program
  (cosine −0.02) — same expert index, different operation. The MoE therefore mixes a few
  genuinely reusable programs with several family-specialised ones. The broader "not
  merely memorising families" claim rests on **a** and **b** (chemistry and structural
  roles survive family control) and on the family-embedding ablation (66% of tokens keep
  their expert when `family_id` is held constant), not on **f**.

---

## Methods — structural annotation criteria

Structures come from the local co-crystal cache (`data/raw/pdb_cif_cache`). v23 crops
carry no PDB pointer, so each crop was mapped by (i) collecting PDB chains whose
deposited sequence contains the crop verbatim, (ii) locating the crop inside that chain's
*observed* residue sequence, and (iii) keeping the candidate whose recomputed DNA-contact
set best matches the v23 supervision set for that row. 273/277 rows mapped (195 distinct
PDB entries), median contact recall 0.875. Position *i* of the annotation is by
construction the structural residue whose one-letter code is `crop[i]`, i.e. exactly the
residue the router saw at token *i*.

All distances are heavy-atom to heavy-atom; hydrogens ignored.

| Label | Criterion |
|---|---|
| DNA base contact | ≥ 1 residue heavy atom within **4.0 Å** of a nucleotide *base* atom (any atom not in {P, OP1–3, O5′, C5′, C4′, O4′, C3′, O3′, C2′, C1′, O2′}) |
| DNA backbone contact | ≥ 1 heavy atom within **4.0 Å** of a sugar/phosphate atom |
| major-groove contact | ≥ 1 heavy atom within **4.0 Å** of a major-groove edge atom {N7, O6, N6, C5, C7, C5M, O4, N4} |
| minor-groove contact | ≥ 1 heavy atom within **4.0 Å** of a minor-groove edge atom {N3, N2, O2, C2} |
| Zn coordination | ≥ 1 candidate ligand atom (element S, N or O) within **2.8 Å** of a Zn ion |
| buried core | relative SASA < **0.20**; Shrake–Rupley on the isolated chain, normalised by the Tien et al. 2013 theoretical maximum ASA |
| α-helix / β-strand / loop | pydssp three-state assignment (H / E / –) from N, CA, C, O backbone coordinates |
| protein–protein interface | ≥ 1 heavy atom within **4.0 Å** of a different protein chain |
| exposed non-DNA surface | relative SASA ≥ 0.20, no DNA contact, no protein–protein interface |

Prevalences over the 23,716 annotated residues: base contact 0.075, backbone 0.194,
major groove 0.061, minor groove 0.014, Zn coordination 0.029, buried core 0.266,
α-helix 0.395, β-strand 0.156, loop 0.449, protein interface 0.060, exposed non-DNA 0.493.

Statistics. Panel a plots the family-composition-controlled log₂ enrichment already used
for the v24 expert analysis (unchanged for the figure) and reports alongside it a
Cochran–Mantel–Haenszel odds ratio stratified by TF family. Panel b fits, per
(expert, role), `role ~ 1{expert} + C(amino acid) + C(family)` by binomial GLM with
cluster-robust SEs by gene; Benjamini–Hochberg FDR across all 88 tests. Panel c compares
observed unordered top-2 pair frequencies with E[{i,j}] ∝ p1_i p2_j + p1_j p2_i
renormalised over i<j, plus a 2,000-fold permutation of the second slot that preserves
marginals and rejects self-pairs.

---

## Caveat worth acting on: the v23 contact indices are frame-shifted

The expert ordering and panel a's right-hand bars use the v23 contact-supervision set,
because that is the axis the analysis was specified around. While mapping crops to
structures it became clear that **v23's stored contact indices sit in a shifted frame
relative to crop coordinates for 192 of 273 rows** (`v23_index_shift` in
`structure_mapping.csv`; shifts spread over −8…+8 and pile up at the ±8 search bound, so
some are larger still). Allowing a rigid shift raises median contact recall from 0.455 to
0.875, which is what a coordinate-convention offset looks like, not a wrong-structure
match.

Consequences visible in this figure:

* The recomputed contact axis agrees with the v23 axis only moderately
  (Spearman ρ = 0.62), and **e4 flips sign**: v23 +0.31 vs recomputed −0.25. Panel b then
  shows e4 has no DNA-contact enrichment at all and is a Zn-coordination expert. So the
  requested ordering puts e4 at the "most DNA-interface" end where the structures place it
  at the scaffold end. The two axes are drawn together in panel a (bars vs dots) rather
  than silently reconciled.
* Since v24 was trained on these indices, part of its contact supervision was
  systematically off by a few residues. That is worth checking against the v26 contact
  build independently of this figure.
