---
name: project-state-v19
description: TFScope V19 current publication candidate, architecture, results, and next steps as of 2026-06-15
metadata:
  type: project
---

## Current state (2026-06-15)

The V19 publication candidate is a **two-model family composition** of:
- **E2** (corrected fixed-frame RAG model, seed 42): provides PWM frame
- **E5b** (family-register model, seed 42): provides motif content

Composition policy is validation-locked in
`results/v19_e9_model_composition/validation_composition_grid.json`.
Family weights: C2H2_long=1.0, C2H2_medium=0.25, Homeodomain=0.25, bHLH=0.10, others=0.0.

Final frozen test results in `results/v19_e9_publication/`.

**Why:** E6 donor reranker and E7 position-wise gating were both rejected on
validation. The composition was the best-performing candidate.

**How to apply:** The paper must report this as a two-model ensemble with 2× inference
cost; corrected E2 must be retained as the single-model baseline. Do not retune the
frozen family policy on test data.

## Test results (n=195 evaluable genes, 10,000 bootstrap replicates)

| Metric | E2 baseline | Composition | Delta | 95% CI | p |
|--------|-------------|-------------|-------|--------|---|
| panel-r | 0.4938 | **0.5454** | **+0.0516** | [+0.031, +0.074] | <1e-4 |
| canon-r | 0.1573 | 0.1527 | -0.0046 | [-0.022, +0.013] | 0.62 (NS) |
| fixed MAE | 1.1444 | **1.1168** | **-0.0276** | [-0.045, -0.011] | 0.001 |
| RMSE | 0.3186 | **0.3070** | **-0.0115** | [-0.018, -0.006] | <1e-4 |
| CE | 1.5619 | **1.4309** | **-0.131** | [-0.173, -0.092] | <1e-4 |
| KL | 1.1015 | **0.9729** | **-0.129** | [-0.169, -0.090] | <1e-4 |

Top-1/AUC/F1/MCC: no significant change (CI includes zero).

Main gain driven by C2H2_long (+0.2045 panel-r, p<1e-4) and bHLH (+0.0071, p=0.012).

## Architecture (brief)

- Backbone: ESM-2 650M frozen + LoRA (last 6 layers, rank 16, α 32)
- Pooling: Dual-stream gated attention (global + DBD-masked)
- MoE: 10 families, 12 routed SwiGLU experts + 1 shared, family-aware gating, FiLM
- Head: PWMHeadV18 (prior branch + contact-aware residual with softmax attention)
- RAG: TrustPredictor, K=16 train-only donors, retrieval dropout 0.15
- ~700M parameters per model

## Clean benchmark artifacts

- Split: `data/processed/splits/cluster40_clean/split.json`
  (train 2947 rows/914 genes, val 686/234, test 614/197; zero leakage)
- Index: `data/processed/tf_nn_index_cluster40_clean.json` (train-only K=16)
- E2 checkpoint: `/data1/leihuang/project/TFScope/checkpoints/v19_e2_gene_balanced_bf16_ddp3/rag_seed42/ckpt_best.pt`

## Next steps (from MACHINE_HANDOFF.md)

1. Compare against external published baselines with explicit protocol-difference statements
2. Report two-model inference cost in manuscript
3. Use `results/v19_e9_publication/` for final tables, CI, and motif-logo examples
4. Calibration, conformal abstention (E11/E12) still incomplete

## Known limitations

- Single seed (42) — user-fixed constraint, must be disclosed
- Canon-r not improved (registration is the dominant error for all models)
- DeepPBS uses structure-based split; direct leaderboard comparison is not valid
- No calibration/abstention metrics yet

Links: [[user-profile]], [[feedback-conventions]]
