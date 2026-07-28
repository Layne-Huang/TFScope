# TFScope — Current Architecture & Results (v24, as of 2026-07-26)

Sequence-only prediction of transcription-factor binding PWMs from the protein
DNA-binding domain (DBD). Supersedes the v19-era `TFSCOPE_ARCHITECTURE_v19_ARCHIVED.md`
and `ARCHITECTURE_AND_RESULTS.md` (both pre-date v20–v24).

Current best model: **v24** (`checkpoints/v24_contact/contact_v24_seed42`).

---

## 1. Problem & framing
Input = one TF DBD amino-acid sequence (+ optional partner chains for obligate
multimers). Output = a position weight matrix (4×L). TFScope is **sequence-based**
and needs no structure — its niche vs structure-based models (DeepPBS) is the vast
majority of TFs with no solved protein–DNA complex.

---

## 2. Architecture (forward path in `src/tfscope/models/tfscope.py::TFScopeModel`)

```
tokens ─► ESM-2 650M (frozen) + LoRA ─► [chain-ID emb, N-chain] ─► +DBD indicator
      ─► ResidueMoE (per-DBD-token) ─► pooling/projection
      ─► PositionGateHead (span: start+length)      ┐
      ─► PWMHeadV18 (contact-aware cross-attn PWM)   ┘─► (gate_logits, pwm_logits)
```

### 2.1 Encoder — `backbone.py`
Frozen ESM-2 `esm2_t33_650M_UR50D`; **LoRA** rank 16, alpha 32, on the last 6
layers (only trainable part of the backbone, ~0.5M params).

### 2.2 N-chain (order-aware multimer) input — `data/dataset.py` + `tfscope.py`
For obligate multimers, feed `chain1 + <eos> + protomer2 + <eos> + … ` up to
`max_chains=4`; `dbd_mask` covers all protomers. A per-token **chain-ID embedding**
(`chain_id = cumsum(<eos>).clamp(max_chains)`) tags which protomer each residue
belongs to. Handles dimers→tetramers (p53, HSF, NF-Y, IRF, NR/RXR, bZIP/MAF, POU-SOX).
Partner data built by `build_nchain_v23.py` (`find_dimer_partners` shared-primary-
duplex logic); training table `tf_pwm_training_v23.parquet` (n_chains {1:4436, 2:1375,
3:87, 4:114}).

### 2.3 Mixture-of-Experts — `moe.py::ResidueMoE`
Per-residue (not per-protein) MoE: **8 routed + 2 shared** SwiGLU experts, top-2.
Router sees token feature ++ **family embedding** (learned, 10 families, dim 64) and
a cosine bias to learned expert prototypes. Residue granularity is what stops the
routing collapse the protein-granularity MoE suffered. Diagnosed **specialized**:
NMI(expert,family)=0.556; bZIP→E5, NR→E2, p53/POU→E4 (E0/E1/E6 under-used). Family
embedding is a modest signal (+0.033 covR ablation) — ESM already encodes family.

### 2.4 Output heads — `heads.py`, `pwm_head_v18.py`
- **PositionGateHead (`gate_mode="span"`)**: predicts a **continuous** span
  (start + length), soft-rectangular occupancy via paired sigmoids (τ=0.5),
  `max_motif_length=42`. Contiguous by construction; length gradients flow.
  Replaced the legacy 42 independent per-position sigmoids (which gave the
  fragmented, fixed-~10 bp gate).
- **PWMHeadV18**: cosine, LayerNorm'd cross-attention from motif positions to DBD
  residues + amino-acid-identity values (mutation channel) + v14 prior branch.
  Contact pathway (see §2.5).

### 2.5 Contact grounding (the v24 change) — `pwm_head_v18.py`
Three distinct mechanisms (all ON in v24; were mis-wired/off before):
1. **Contact supervision** (`v18_contact_supervision`, weight 0.3): loss pulling
   cross-attention onto DNA-contact residues (1D per-residue target).
2. **Base-level contact distillation** (`contact_distill_weight=0.2`): KL between
   the attention matrix and the true 2D base×residue contact map (`contact_targets_v23.json`).
3. **Learnable contact-bias** (`v18_contact_bias_scale=1.0, learnable=True`):
   injectable additive attention bias; at inference accepts `contact_override` (true
   contacts) — this is the pathway that recovers de-novo-design CAC (§4).

⚠ **Key fix in v24**: contact maps were keyed by deeppbs-style filenames but the
training table uses `str_i/seq_i` → **supervision was a silent no-op in v22/v23**.
Remapped via (pdb,chain,gene) → `recognition_residues_v23.json` (362),
`contact_targets_v23.json` (309). v24 is the first run where contact learning is active.

### 2.6 Loss — `losses/tfscope_loss.py`
L1/IC PWM loss + IC-PCC + top-base; `gate_length_weight=0.05` (couples soft gate
length to GT); `pwm_cov_r_weight=0.25` (coverage-aware column correlation aligned to
the eval metric); contact-supervision + distillation terms; latent-registration
(train frame == eval frame, shift+RC).

---

## 3. Benchmark & metric
- **Leakage-free split** (`splits/train_v22/`): MMseqs2 40%-identity connected-component
  split; **test = 291 structure rows only** (so DeepPBS can run on the same structures);
  seq cluster-mates of test excluded. Test families (corrected labels): p53 101 (35%),
  Forkhead 65, ETS 56, POU 46, bZIP 14, NR 5, bHLH 4. Note p53/POU are ~zero-shot
  (0 train rows) — the test is a hard, structure-heavy slice.
- **Coverage-aware metric** (`eval_full_metrics.panel_full`, `train.py` selector):
  covR = per-column Pearson over the gate-active core **× coverage** (uncovered GT
  columns penalised). Fixes the length blind spot where a short gate is scored on
  fewer easy columns. `--legacy-oracle-r` restores the old overlap-only metric.

---

## 4. Results (v24, test = 291 rows, predicted_gate; 7-seed where noted)
| metric | row-mean | gene-balanced |
|---|---|---|
| **covR** (coverage-aware) | 0.461 | **0.523** |
| r (overlap Pearson) | 0.532 | 0.592 |
| PWM MAE | 0.210 | 0.192 |
| top-1 base acc | 0.542 | 0.598 |
| coverage | 0.779 | 0.837 |
| gate-length MAE / bias | 3.98 bp / −0.43 bp | — |
| covR @ oracle length (upper bound) | 0.502 | 0.575 |

- **v20→v24 progression**: v20 ~0.37 row / ~0.44 gene-bal → v24 0.461 / 0.523.
  v23 N-chain across 7 seeds: 0.495±0.031 gene-bal (stable).
- **N-chain multimer gains** (vs dimer-only): POU +0.062, p53 +0.048 (robust); NFE2L2
  gate extended 11→17 bp toward true 14; Forkhead the main cost.
- **Contact grounding**: v24 best covR; **MyoD1 WT E-box recovered (CAGCTG)** (v23 gave
  garbage); **de-novo design CAC 0/4 → 3/4 with true-contact injection** (trained bias
  pathway). Perception is the OOD ceiling: attention-on-true-contacts AUROC ~0.68 for
  bacterial-HTH designs (vs 0.95 in-distribution) → decoder works given contacts;
  finding the contacts is the bottleneck.

### Limitations (honest)
- **Mutation-blind**: on the Barrera-2016 55 WT/MUT homeodomain pairs, v24 predicts
  WT→MUT change 0.005 (measured 0.180), corr −0.05, directional 40%. Two fixes tried,
  both negative: (a) plain 55-pair fine-tune (Δpred 0.008, corr 0.01); (b) an explicit
  **paired delta objective** (E0/E1, `docs/MUTATION_EXPERIMENTS.md`) — E0 shows the
  signal reaches the logits but is routed to flanks; E1's paired loss appears to work
  when peeking at test (corr 0→0.53) but gives held-out corr **0.00** under gene-disjoint
  model selection (overfits 24 pairs; Δpred stays collapsed). Bottleneck is data scale
  (55 HD pairs, one family), not objective — needs full Barrera/DMS across families.
- **Long multimeric motifs** still under-covered (p53 tetramer: content up but length
  ~10 vs 14.7).
- **de-novo design** design assay: LOWER value = STRONGER binding (exp_pref was
  inverted); designs are weak binders (IC<1); zero-shot from sequence doesn't reach CAC.

### DeepPBS positioning
Fair cluster40 comparison (DeepPBS retrained on cluster40-train, both held-out):
DeepPBS (structure-based) panel_r ~0.60 > TFScope v19 ~0.46 — structure model leads on
structural tests (expected). A clean v24-vs-DeepPBS on the new 291-test needs a DeepPBS
retrain on train_v22 (no large leakage-clean subset exists between the two splits).
Framing: TFScope = the **sequence-only** alternative (no structure required).

---

## 5. File map
| path | role |
|---|---|
| `src/tfscope/models/tfscope.py` | top-level `TFScopeModel.forward` |
| `src/tfscope/models/backbone.py` | frozen ESM-2 + LoRA |
| `src/tfscope/models/moe.py` | `ResidueMoE` + family embeddings |
| `src/tfscope/models/heads.py` | `PositionGateHead` (span), `ContactPredHead` |
| `src/tfscope/models/pwm_head_v18.py` | contact-aware PWM head (attn/bias/distill) |
| `src/tfscope/config.py` | `TFScopeConfig` (all knobs) |
| `src/tfscope/data/dataset.py` | N-chain input, contact/recog priors |
| `src/tfscope/losses/tfscope_loss.py` | loss terms |
| `scripts/train.py` | training + coverage-aware selector |
| `scripts/eval_full_metrics.py` | coverage-aware metric |
| `scripts/build_nchain_v23.py` | N-chain partner data |
| `scripts/run_v24_contact_ddp.sh` | v24 launcher (6-GPU DDP) |
| `data/processed/tf_pwm_training_v23.parquet` | training table (N-chain) |
| `results/v22_ablation/`, `results/mutation_benchmark/` | eval outputs |

## 6. Reproduce
```bash
# v24 (6-GPU DDP; see multi-gpu-ddp-node-fix in memory: UUID-pin + NCCL_P2P_DISABLE)
bash scripts/run_v24_contact_ddp.sh
# test diagnostic (covR, family-stratified)
python scripts/eval_v22_diagnostics.py --checkpoint <ckpt>/ckpt_best.pt \
  --data data/processed/tf_pwm_training_v23.parquet \
  --split data/processed/splits/train_v22/split.json --split-name test --out <out>.json
```
