# TFScope ICLR 2026 revision harness

Implements the experimental scaffolding of
`docs/ICLR2026_MODEL_REVISION_PLAN.md`. The plan's rule is strict:
**v24 stays the production model until the §7 replacement gate is passed.**
Nothing here promotes a new model; it produces the apples-to-apples evidence the
gate needs.

## What is implemented

| Piece | File | Status |
|---|---|---|
| MoE off-switch (B5) | `src/tfscope/config.py` (`use_moe`), `scripts/train.py` (`--no-moe`) | done, unit-tested |
| Mean-pool baseline (B2) | `pool_type="mean"` / `--mean-pool` | done, smoke-tested |
| Variant registry B0–B8 | `iclr/variants.py` | done |
| Training-free baselines B0/B1 | `iclr/baselines.py` | done, run on real 291-row test |
| Pre-registered §7 gate | `iclr/promotion_gate.py` | done, self-test |
| Candidate A (chain-set encoder) | `src/tfscope/models/chain_set_encoder.py` | done, equivariance-tested |
| Candidate B (interface pair head) | `src/tfscope/models/interface_pair.py` | done, unit-tested |
| Phase-I launcher | `iclr/run_phase1.sh` | done |

Candidates A/B are **modules only**. Per plan §9 they are *not* wired into the
production model until Phase I proves multimer and/or contact signal is real.

## Canonical benchmark (plan §2)

Everything uses one parquet + one split so preprocessing, registration, max
length, and eval code are identical across variants:

- data: `data/processed/tf_pwm_training_v23.parquet`
- split: `data/processed/splits/train_v22/split.json`
  - `train` (4794) → training
  - `test` (291 `str_*` rows) → **the immutable 291-row structure test set**
- monomer vs multimer: parquet `n_chains` / `is_dimer`
- primary endpoint: **gene-balanced coverage-aware r** (`gene_covR`); also report
  row `covR`, PWM MAE, top-base accuracy, coverage, gate-length error, and
  family / monomer / multimer breakdowns.

Frozen v24 reference checkpoint (`--v24-ckpt`, plan §2 rule 1, immutable):
canonical `contact_v24_seed42/ckpt_best.pt`; on this node the available copy is
`checkpoints/v24_e1_paired/pwmhead_ft.pt`.

## Running Phase I

```bash
# 1) training-free floors (CPU, minutes)
bash iclr/run_phase1.sh baselines

# 2) print the exact train/eval commands for B2–B8 (≥3 seeds)
SEEDS="42 1 7" bash iclr/run_phase1.sh commands
```

Variant map (plan §3):

| ID | variant | how it is produced |
|---|---|---|
| B0 | family-average PWM | `iclr/baselines.py` (training-free) |
| B1 | nearest training PWM | `iclr/baselines.py` (training-free) |
| B2 | frozen ESM + mean pool + MLP | `train.py --mean-pool --no-moe`, no v18/contacts/chain |
| B3 | frozen ESM + attention pool + MLP | B2 without `--mean-pool` |
| B4 | ESM + span gate | B3 with stronger gate-length weight |
| B5 | v24 without MoE | full v24 recipe **+ `--no-moe`** |
| B6 | v24 without contacts | full v24 recipe **− contact flags** |
| B7 | v24 N-chain, minimal head | N-chain input only (no MoE/v18/contacts) |
| B8 | complete v24 | frozen checkpoint, evaluate only |

Trained variants keep v24's LoRA/optimisation recipe (B5–B7) or a fully frozen
ESM (B2–B4). Match the trainable-parameter budget where practical (plan §3) and
report parameter / runtime accounting (plan §2 rule 6).

## Phase-I decision → §7 gate

After collecting per-seed, per-gene predictions for B0–B8 (and any candidate),
assemble a results JSON (schema in `iclr/promotion_gate.py`) and run:

```bash
python -m iclr.promotion_gate --results results/iclr_phase1/results.json \
    --out results/iclr_phase1/promotion_decision.json
```

The gate encodes all ten §7 conditions (absolute gain ≥ +0.02 gene_covR, paired
bootstrap CI > 0, all-seeds-positive, beats best simple baseline, monomer
preserved, multimer gain if chain-set is claimed, permutation invariance <
0.005, no single-family dependence, sequence-only headline, artifacts recorded)
and writes a machine-readable `promotion_decision.json`. If any condition fails,
keep v24 and record the candidate as an unsuccessful ablation (plan §7, §8).

## Candidate modules (only if Phase I justifies them)

- **Candidate A — `ChainSetEncoder`** (plan §4): permutation-equivariant
  Set-Transformer over DBD chains; preserves residue states, no chain-ID
  embedding (homomer-symmetric), permutation-consistency loss provided.
  `tests/test_chain_set_equivariance.py` verifies invariance to < 1e-5.
- **Candidate B — `InterfacePairHead`** (plan §5): explicit residue × latent
  DNA-position pair mixer with occupancy `C`, base-energy `E`, masked 2D
  contact distillation, 1D marginal ablation, and shuffled-contact control.
  `tests/test_interface_pair.py`.

Build the unified candidate (plan §6) only when Phase I supports both signals;
name it `tfscope_interface_set_candidate` (not `v25`) until §7 passes.
