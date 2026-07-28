# TFScope Case Studies — Summary

_Last updated: 2026-06-19_

All four case studies share one template: **sequence-only, confidence-calibrated
DNA-binding-motif nomination for orphan transcription factors** — TFs with no curated motif
and no protein–DNA structure. Each follows the same claim discipline (a testable hypothesis,
never a validated binding event) and each is staged with its results tables, a publication
figure, and manuscript snippets. **None are yet inserted into `manuscript/TFScope_nature_methods.tex`.**

---

## Overview

| # | TF | Family | Checkpoint | RAG motif | Conf | Headline support |
|---|-----|--------|-----------|-----------|------|------------------|
| Fig 5 | **SOHLH1** | bHLH | `cluster40_v18a_rag` (clean orphan) | **CGCGTG** (E-box), 1.36 bits | 0.80 | r=0.76 vs SOHLH2 paralog; masked-SOHLH2 control r=0.81 |
| Fig 6 | **ZGLP1** | GATA | dual: `lofo_v18a/Other` (clean) + `cluster40_v18a_rag` (leaky, labelled) | **GATAAT**, 1.49 bits | 0.85 (High) | r=0.99 vs GATA3; double-dissociation control; divergent flanking |
| Fig 7 | **ADNP** | Homeodomain | `cluster40_v18a_rag` (clean orphan) | **TTGATAA** (TGAT-leaning), 1.58 bits | 0.69 (Med) | retrieves EXD/PBX1; 100% masked-HD control; ADNP2 diverges r=0.28 |
| Fig 8 | **ZHX3 / ZHX2 / ADNP2** | Homeodomain | `deeppbs_v18a_attnrepair` (benchmark ckpt; all clean) | ZHX3 **TTAATAA** 1.61; ZHX2 TGAATAAAA 1.27; ADNP2 AGAAACATGTGAAAA 1.20 | 0.90 / 0.72 / 0.70 | **ZHX3 = lead application figure**; ZHX2↔ZHX3 r=0.78; promoter enrichment |

---

## Fig 5 — SOHLH1 (bHLH, germ-cell orphan)

- **Result dir:** `results/sohlh1_case/` (also legacy `results/case_study_sohlh1/`)
- **Run:** `mamba run -n tfscope python scripts/case_study/run_all.py`
- noRAG → weak `CCCCCC`; leave-gene-out RAG sharpens into an **E-box `CGCGTG`** (1.36 bits).
- Retrieved neighbours (SOHLH1/SOHLH2 excluded): NPAS2, TCF4, CLOCK. Max DBD identity ≤50%.
- **Calibration** (cluster40 held-out, n=221): Spearman ρ=0.32; median oracle r rises 0.21→0.42
  across confidence bins; held-out bHLH median oracle r=0.45 → conf 0.80 maps to expected r≈0.42–0.45.
- **Positive control:** retrieval-masked SOHLH2 recovers JASPAR MA1560.1 E-box at r=0.81
  (IC-weighted 0.90). (Retrieval-masked, not fully train-masked — SOHLH2 stays in the encoder.)
- **Promoter scan = honest NEGATIVE (Extended Data):** after a dinucleotide-preserving shuffle
  control (GC + CpG fixed), the E-box shows no significant enrichment (AUROC 0.58, p=0.067;
  shuffled control 0.57). The naive AUROC 0.73 was a GC/CpG artifact. No occupancy claim.
- **Confidence definition:** `0.5·(meanIC/2) + 0.5·norm(gate)` — gate and IC are the held-out
  predictors of accuracy (ρ=+0.32, +0.29). Earlier similarity/cosine-weighted score was
  anti-calibrated and discarded.

## Fig 6 — ZGLP1 (GATA, oogenic orphan)

- **Result dir:** `results/zglp1_case/`
- **Run:** `mamba run -n tfscope python scripts/case_study/run_all_zglp1.py`
- **Not a clean orphan w.r.t. TFScope:** has a HOCOMOCO H13CORE motif in training and sits in
  the cluster40 **train** split → production checkpoint is encoder-leaky. Reported with **two
  checkpoints, labelled throughout:**
  - `lofo_v18a/Other` (family-masked, never saw any GATA) — clean de-novo generalization test.
  - `cluster40_v18a_rag` (production, LGO retrieval) — retrieval-supported nomination.
- Clean de-novo consensus: `CCTCCCCTCC` (no GATA, r=0.37). Production RAG: **`GATAAT`** (1.49 bits).
- Retrieved neighbours (ZGLP1 LGO): GATA3, GATA5, GATA1. r=0.99 vs GATA3; r=0.72 vs GATA1–6 mean.
- **Double dissociation:** retrieval-masked RAG recovers known GATA motifs **100%** (r=0.75, 0.75
  on GATA4/GATA6); family-masked de-novo **0%** → nomination is retrieval-driven family inference,
  not memorization (output is GATA family consensus, not ZGLP1's leaked divergent motif).
- **Divergence finding:** ZGLP1's experimental motif (ATGATCGAT) is a divergent GATA variant;
  high-IC core columns agree (per-column r=0.58), flanking diverges (overall r=0.49). Nominates
  GATA core + flags ZGLP1-specific flanking as a concrete testable refinement.
- **Promoter scan = WEAKLY POSITIVE:** GATA core enriched beyond composition (AUROC 0.58, p=0.048;
  shuffled control 0.51). GATA is AT-rich → avoids the SOHLH1 CpG artifact. Top candidates:
  STRA8, SYCP3, SMC1B, SYCE2, GDF9, OOSP2. Candidate-level only.

## Fig 7 — ADNP (homeodomain, neurodevelopmental orphan)

- **Result dir:** `results/adnp_case/`
- **Run:** `mamba run -n tfscope python scripts/case_study/run_all_adnp.py`
- Flagship autism / Helsmoortel–Van der Aa gene. **CLEAN orphan** (absent from all
  train/retrieval/benchmark tables; max DBD identity 62%, to KLF4) → single-checkpoint design
  like SOHLH1. (BHLHA9 was rejected — it is in cluster40 train, leaky.)
- noRAG → weak `AAAAAAA`; LGO-RAG → **`TTGATAA`** (1.58 bits), homeodomain-class but **TGAT-leaning**
  (equidistant from canonical TAAT and TALE/PBX1 TGAT grammar).
- Retrieved neighbours (ADNP/ADNP2 excluded): EXD, PBX1, HMG20A (real homeodomains).
- **Masked homeodomain control: 100% (4/4):** EN1→TAATTA r=0.82, PITX1→TAATCC r=1.00,
  ISL1→TAATTA r=0.61, PBX1→TGATTGA r=0.64. Both TAAT and TALE TGAT grammars recovered.
- ADNP2 (also clean orphan) companion: TGAAATATA, ADNP↔ADNP2 consistency r=0.28 (honest
  paralog divergence — not independent validation).
- **Promoter scan = honest NEGATIVE (Extended Data):** AT-rich motif shows no enrichment beyond
  composition (AUROC 0.51; shuffled control higher at 0.60). Consistent with ADNP acting in the
  chromatin ChAHP complex — sequence alone does not localize its motif.

## Fig 8 — ZHX3 / ZHX2 / ADNP2 (homeodomain, on the DeepPBS benchmark checkpoint)

- **Result dir:** `results/orphan_homeodomain_deeppbs/`
- **Run:** `mamba run -n tfscope python scripts/case_study/run_all_orphans.py`
- **Checkpoint:** `deeppbs_v18a_attnrepair/ckpt_best.pt` — v18a, full augmented data, **DeepPBS
  split** (`splits/deeppbs_aug_dbd/benchmark.json`, train+val = 4117 donors), honest LGO retrieval.
  This is the checkpoint behind the DeepPBS benchmark comparison. Sanity: EN1 (masked) → TAATTAA r=0.87.
- **Orphan status verified:** ADNP2, ZHX2, ZHX3 absent from the entire parquet → truly orphan.
- DBD windows: ADNP2 = single homeobox (1043–1102); ZHX2/ZHX3 carry 4–5 homeoboxes — all screened,
  **HD1** gave the sharpest/most canonical prediction in both.

| orphan | DBD | RAG motif | IC | conf | retrieves | r vs TAAT |
|--------|-----|-----------|----|----|-----------|-----------|
| **ZHX3** | HD1 (304–363) | **TTAATAA** | 1.61 | **0.90 (High)** | ANHX, SATB1, CUX2 | 0.74 |
| **ZHX2** | HD1 (263–324) | **TGAATAAAA** | 1.27 | 0.72 (High) | TGIF1, POU5F1, SOX2 | 0.56 |
| **ADNP2** | homeobox (1043–1102) | AGAAACATGTGAAAA | 1.20 | 0.70 (Med) | MATALPHA2, PBX1 | 0.46 |

- **Paralog consistency:** ZHX2↔ZHX3 r=0.78 (strong; ZHX1's training motif is degenerate "G" and
  cannot serve as reference). ADNP2↔ADNP r=0.61.
- **Calibration (deeppbs held-out):** Spearman ρ=0.41; held-out homeodomain median oracle r=0.79 (n=5).
- **Masked HD control:** EN1 (0.87), PBX1 (0.67) recover at r≥0.6; PITX1 (0.56), ISL1 (0.59) give
  the correct TAAT consensus but just below the 0.6 cutoff (2/4 at r≥0.6; 4/4 correct consensus).
- **Promoter scans (dinucleotide-shuffle control):**
  - **ZHX3:** hepatic / cell-cycle — AUROC 0.712, p=5e-4, control 0.607 → **enriched** (AFP, CCNA2, CCND1, CDKN1A).
  - **ZHX2:** hepatic / tumour-suppressor — AUROC 0.668, p=5e-3, control 0.636 → enriched (borderline; ALB, GPC3, AFP).
  - **ADNP2:** neurodevelopmental — AUROC 0.708, control 0.750 → n.s. (composition-confounded, like ADNP).

---

## Manuscript decision — ZHX3 is the lead application figure

`figures/figure8/Figure_ZHX3_application.pdf` is the chosen application figure: ZHX3 nomination
(TTAATAA, conf 0.90) + ZHX2 paralog consistency (r=0.78) + composition-independent promoter
enrichment (AFP/cell-cycle targets) + calibration + masked control.

Nominations are **checkpoint-dependent** — no single checkpoint makes every candidate look best:

| TF | `cluster40_v18a_rag` | `deeppbs_v18a_attnrepair` (benchmark) |
|----|----------------------|---------------------------------------|
| SOHLH1 | **E-box CGCGTG, r=0.76** ✓ | GCCCCCCG, r=0.38 ✗ |
| ZHX3 | AAAATAA, IC 0.97, r=0.62 | **TTAATAA, IC 1.61, r=0.74** ✓✓ |
| ZHX2 | ATAAATC, IC 0.43 ✗ | TGAATAAAA, IC 1.27 ✓ |

Per the recorded decision: lead the manuscript with **ZHX3 on the deeppbs benchmark checkpoint**
(strongest + benchmark-consistent + functional promoter support) and **drop SOHLH1 from that
figure**. SOHLH1 / ZGLP1 / ADNP remain cluster40-only examples (Figs 5–7), usable (if at all) as
clearly-labelled supplementary cross-family vignettes — not on the deeppbs figure.

---

## Cross-cutting design principles

1. **noRAG → RAG sharpening** — a weak/degenerate prior becomes a confident motif via leave-gene-out
   retrieval of genuine same-class neighbours.
2. **Confidence calibration** — score = `0.5·(meanIC/2) + 0.5·norm(gate)`, validated against
   held-out oracle r (per-split calibration table).
3. **Leakage audit** — every TF vetted for train/retrieval/benchmark membership; leaky cases
   (ZGLP1) get a dual-checkpoint clean+labelled treatment.
4. **Masked positive control** — same-family known TFs, retrieval-masked, must recover their
   curated motif (proves the retrieval pathway works on that family).
5. **Composition-controlled promoter scan** — dinucleotide-shuffle background (GC + CpG fixed);
   honest negatives (SOHLH1, ADNP, ADNP2) reported as Extended Data, positives (ZGLP1, ZHX2/ZHX3)
   as candidate-level support — never an occupancy claim.

## File map

| study | results dir | figure | manuscript snippets |
|-------|-------------|--------|---------------------|
| SOHLH1 | `results/sohlh1_case/` | `figures/figure5/Figure5_SOHLH1_case_study.pdf` | `results/sohlh1_case/manuscript/` |
| ZGLP1 | `results/zglp1_case/` | `figures/figure6/Figure6_ZGLP1_case_study.pdf` | `results/zglp1_case/manuscript/` |
| ADNP | `results/adnp_case/` | `figures/figure7/Figure7_ADNP_case_study.pdf` | `results/adnp_case/manuscript/` |
| ZHX3/ZHX2/ADNP2 | `results/orphan_homeodomain_deeppbs/` | `figures/figure8/Figure8_orphan_homeodomains.pdf`, `Figure_ZHX3_application.pdf` | `results/orphan_homeodomain_deeppbs/manuscript/` |
