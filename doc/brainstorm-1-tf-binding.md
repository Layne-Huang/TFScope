# Brainstorm Session #1: TF Binding Specificity & DBD Family Analysis

**Agent:** tf-binding-expert
**Date:** 2026-04-16
**Topic:** DBD family coverage, data sources, train/test splits, and biological feasibility

---

## 1. DBD Family Coverage

### Recommended 8-10 Core Families

Based on Lambert et al. 2018 classification of ~1,600 human TFs across ~60 DBD families, focusing on families with sufficient PWM data:

| # | DBD Family | Human TFs | Motif Length | Readout Mode | Key Notes |
|---|-----------|-----------|-------------|-------------|-----------|
| 1 | C2H2 Zinc Fingers | ~700 | 9-20bp (3bp/finger) | Base readout | Largest family; modular recognition code; variable finger count (1-40+) |
| 2 | bHLH | ~110 | 6-8bp (E-box: CANNTG) | Mixed base + shape | Dimerize; E-box variants; partners affect specificity |
| 3 | Homeodomain | ~200 | 6-8bp (TAAT-core) | Mixed; strong shape | Highly conserved fold; subtle specificity differences; HOX+TALE cofactors |
| 4 | bZIP | ~60 | 10-14bp (palindromic) | Base readout | Homo/heterodimers; CREB, AP-1 (Fos/Jun); palindromic motifs |
| 5 | Nuclear Receptor | ~50 | 12-15bp (direct/inverted repeats) | Base readout | Two zinc finger DBDs; DR4, IR0 motifs; ligand-dependent |
| 6 | Forkhead | ~45 | 12-17bp | Mixed; strong shape | Winged helix; FOX proteins; long flanking contacts |
| 7 | Sox (HMG) | ~25 | 7-9bp | Strong shape readout | Minor groove binding; DNA bending; Sox2/Oct4 cooperativity |
| 8 | ETS | ~30 | 9-11bp (GGAA-core) | Mixed base + shape | Conserved GGAA core; flanking variation; ETS1, PU.1 |

Plus catch-all "other structured" for: IRF (~10 TFs), AP-2 (~5 TFs), RFX (~8 TFs).

**These 8 families cover ~85% of human TFs with available PWMs.**

### C2H2 Subdivision (Critical)

C2H2 zinc fingers are ~45% of all human TFs. A single "C2H2 expert" is insufficient. Subdivide by finger count:

| Subgroup | Finger Count | Approx. TFs | Motif Length |
|----------|-------------|-------------|-------------|
| C2H2-short | 1-3 | ~200 | 3-9bp |
| C2H2-medium | 4-6 | ~250 | 12-18bp |
| C2H2-long | 7-10 | ~150 | 21-30bp |
| C2H2-extended | 11+ | ~100 | 33bp+ |

Recognition grammar and motif length scale differently across these groups.

---

## 2. Data Sources

### Comparison Table

| Database | PWM Count | Strengths | Weaknesses | Best For |
|----------|-----------|-----------|------------|----------|
| **JASPAR 2024** | ~2,000 | Curated, non-redundant; REST API; standardized format; TF class annotations | Smaller; over-represents well-studied TFs (mouse/human) | Core training + evaluation |
| **CIS-BP** | ~15,000+ | Comprehensive; DBD annotations from Pfam/InterPro; both experimental and inferred | **Critical: many PWMs are homology-transferred (data leakage risk)**; SELEX PWMs have higher info content than PBM | Broad coverage after removing inferred entries |
| **HOCOMOCO v11-12** | ~1,400 | Systematic quality scoring (A/B/C/D); consistent pipeline; both ChIP-seq and in-vitro | Limited to human/mouse | Independent evaluation set |

### Recommended Data Strategy

1. **Primary training set:** JASPAR 2024 core profiles + CIS-BP experimentally-determined (non-inferred) PWMs
2. **Deduplication:** 80% sequence identity clustering on full TF protein sequences (not just DBD)
3. **Assay metadata:** Include assay type (PBM, HT-SELEX, SELEX, ChIP-seq). PWM quality differs systematically by assay. Consider learned assay-type embedding.
4. **Evaluation set:** Reserve HOCOMOCO quality A/B motifs as independent test — consistent pipeline, fairest benchmark

### Data Leakage Warnings

- **CIS-BP homology transfer:** If TF-A's PWM was inferred from TF-B, these are NOT independent. Exclude all inferred PWMs from test set.
- **SELEX vs PBM bias:** SELEX-derived PWMs tend to have higher information content than PBM-derived PWMs for the same TF.
- **Multi-DBD TFs:** Some TFs have multiple DBD types — place in separate "multi-domain" category.
- **Dimerization partners:** bHLH and bZIP PWMs depend on dimer partner. Track dimer identities.

---

## 3. Train/Test Split Strategy

### Strategy A — Leave-One-Family-Out (LOFO) Cross-Validation (PRIMARY)

Hold out an entire DBD family, train on the rest. Directly tests cross-family generalization.

**This is the headline result for Nature Methods.** Report per-family results.

### Strategy B — Sequence-Identity-Based Split

Cluster all TF protein sequences at 30% identity. Assign clusters to train/test. No homologous TFs leak between splits, but different DBD families can appear in both sets.

### Strategy C — Family-Aware Random Split (baseline)

Within each family, random 80/10/10 split. Tests within-family generalization. Useful as performance ceiling.

**Recommendation: Report all three.** Lead with LOFO as headline. Strategy B as "fair but realistic." Strategy C as within-family ceiling.

---

## 4. Biological Concerns with Proposed Architecture

### Concern 1: Global Attention Pooling Dilutes Signal

The DBD is typically 60-100 residues out of 300-2000+ total. Global pooling is dominated by non-DBD residues.

**Fix:** Learned gate: `global_feature = alpha * all_residues_pool + (1-alpha) * dbd_residues_pool`, where alpha is predicted from family embedding. For compact DBDs (homeodomain, ETS), alpha should be low. For C2H2 with important linker regions, alpha should be higher.

### Concern 2: Motif Length Range [4-20bp] Is Mismatched

- C2H2 with 10+ fingers can bind 30bp motifs (20bp cap truncates them)
- Homeodomain/bHLH motifs are 6-8bp (predicting up to 20 is wasteful)

**Fix:** Family-conditional motif length prediction. Each MOE expert predicts family-appropriate range. For C2H2, predict finger count then derive length (~3bp/finger). For homeodomain, constrain to 6-10bp.

### Concern 3: Discrete Family Labels Miss Ambiguity

~10-15% of TFs have multiple DBD types or blur boundaries.

**Fix:** Soft family assignment — distribution over families rather than hard label. During inference, allow model to attend to multiple family experts.

### Concern 4: PWM Misses Inter-Position Dependencies

PWMs assume position independence. bZIP palindromic motifs and forkhead flanking positions have correlations.

**Fix:** Acceptable for seed model. Flag as known limitation. In full TFScope (Module 2), consider dinucleotide-weight matrix output.

### Concern 5: ESM3 May Not Capture Fine-Grained Specificity Determinants

Key specificity-determining residues (e.g., positions -1, 2, 3, 6 in C2H2 finger) are a tiny subset of the embedding.

**Fix:** DBD-focused attention pooling concentrates capacity on relevant residues. Consider fine-tuning last layers. Use Lambert 2018 annotations as auxiliary supervision.

### Concern 6: Dimerization Is Entirely Unmodeled

bHLH, bZIP, and nuclear receptor TFs bind as dimers. Motif depends on dimer partner.

**Fix:** Acknowledge limitation. Focus initial evaluation on monomers (homeodomain, ETS, forkhead, C2H2) or homodimers (most bZIP). Add dimerization module in full model.

---

## 5. Actionable Recommendations (Priority Order)

1. **Use 8-10 DBD families** with richest data + "other" catch-all. Subdivide C2H2 by finger count.
2. **Training data:** JASPAR 2024 + CIS-BP experimental-only. Deduplicate at 80% identity. Include assay type.
3. **Evaluation:** Lead with LOFO. HOCOMOCO quality A/B as independent test.
4. **Architecture modifications:**
   - Gated global/DBD feature combination (family-conditional alpha)
   - Family-conditional motif length prediction
   - Soft family routing in MOE
   - Self-attention in PWM head for inter-position dependencies
   - Acknowledge dimerization limitation; restrict seed model evaluation
   - Start frozen encoder, probe before fine-tuning
