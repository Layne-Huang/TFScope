# TFScope Experiment Changelog

---

## Dataset Comparison: TFScope vs DeepPBS (2026-05-19)

### 1. Cluster-level overlap — fundamentally different test settings

**DeepPBS clustering protocol (from Supplementary):**
- CD-HIT v4.8.1 at 40% sequence similarity → 189 clusters
- CV dataset: up to 5 structures sampled per cluster, split into 5 cluster-stratified folds
- Blind benchmark: remaining structures from the **same 189 clusters**, resampled up to 5 per cluster

Because the blind benchmark is built from leftovers of the same clusters, DeepPBS trained on
cluster-mates (≥40% sequence similar proteins) for most blind benchmark entries.

| | TFScope | DeepPBS |
|---|---|---|
| Training samples | 3,257 rows | 523 structure entries |
| Unique training TF identities | 1,224 genes | 243 TF identities |
| Blind benchmark size | 129 PWMs | 130 structures |
| Exact blind structures in CV folds | **0 / 129 (0%)** | **0 / 130 (0%)** — verified |
| Blind TFs with a cluster-mate in CV training | **0 / 27 HOCOMOCO (0%)** | **49 / 55 TF identities (89%)** |

**Structure-level:** Both models are blind — no PDB entry in `id.txt` appears in any CV fold file.

**Cluster-level:** DeepPBS trained on 40%-similar cluster-mates for 89% of blind benchmark TFs,
giving it learned structural priors for those proteins. TFScope has seen zero benchmark gene symbols
in any form — the test TFs are entirely absent from the training set.

**Key distinction:** DeepPBS's blind benchmark tests generalization to *new structures of known TFs*
(same sequence cluster). TFScope's test set tests generalization to *completely unseen TF genes*.

### 2. Large family distribution shift in TFScope's split

| Family | Train % | Test % | Shift |
|---|---|---|---|
| C2H2_long | 25.2 | 1.8 | **−23.4** |
| ETS | 2.1 | 10.4 | **+8.3** |
| Other | 16.9 | 34.9 | **+18.0** |
| Nuclear_Receptor | 4.5 | 9.6 | +5.1 |
| Forkhead | 2.6 | 6.2 | +3.6 |
| Homeodomain | 21.4 | 17.5 | −3.9 |

C2H2 zinc fingers dominate TFScope's training set (39% combined) but represent only 7% of the
benchmark test set. ETS and Nuclear Receptor families are underrepresented during training
but are heavily tested.

### 3. Source distribution shift

| Source | Train % | Test % |
|---|---|---|
| HOCOMOCO | 44.0 | 38.8 |
| JASPAR | 21.3 | **35.1** (+13.8) |
| CisBP | 28.1 | 25.8 |

JASPAR PWMs are overrepresented in the test set, and JASPAR entries tend to have stricter
curation standards (potentially sharper PWMs) than CisBP.

### 4. Implications for comparison

- The Pearson r gap (0.185 vs 0.702 on blind benchmark) reflects three compounded factors:
  1. **Modality gap**: DeepPBS uses 3D co-crystal structure + DNA; TFScope uses sequence only
  2. **Gene-level familiarity**: DeepPBS has seen 89% of blind benchmark TFs via other crystal
     structures; TFScope has seen 0% of benchmark gene symbols during training
  3. **Family shift**: TFScope's training data is misaligned with the test family distribution

- **Fair framing for the paper**: TFScope's task is strictly harder — zero-shot generalization
  from training TFs to unseen benchmark TFs, using sequence alone with no structural information.

- **Mitigation options**:
  - Re-weight training loss by family to correct the C2H2 dominance
  - Oversample ETS / Nuclear Receptor / Forkhead training examples
  - Add family-aware augmentation

---

## Master Scoreboard

### A. DeepPBS CV validation set (n=613, TFScope re-split)

All TFScope metrics use the same evaluation formulas throughout.
DeepPBS MAE reported in both their scale (sum-4-bases/pos) and normalized (/4) for fair comparison.

| Version | Pearson r ↑ | MAE (per-elem) ↓ | IC-wtd Pearson ↑ | Length MAE ↓ | Notes |
|---|---|---|---|---|---|
| **DeepPBS** | **0.757** | **0.124** *(0.496/4)* | 0.496 † | — | 3D structure + DNA input |
| Baseline LoRA (ep.55) | 0.200 | 0.236 | 0.898 | 2.43 | KL+L1(0.5)+IC; label_smooth dead |
| v2 LoRA best | 0.212 | 0.239 | 0.901 | 2.35 | KL+L1(1.5)+IC+entropy; same arch |
| v3 best (ep.33) | 0.175 | 0.241 | 0.897 | 2.47 | new arch; KL still in loss → stuck |
| v4 best (ep.32) | 0.128 | 0.265 | 0.860 | 2.45 | L1-only; val-best overfit |
| **v4 ep.200** | **0.185** | **0.251** | **0.865** | **2.42** | L1-only; continued training |

† DeepPBS IC-Pearson uses IC-profile Pearson; TFScope uses IC-weighted PWM Pearson — not directly comparable.
IC-Pearson using DeepPBS's exact formula: v3 = 0.289 vs DeepPBS = 0.496.

### B. DeepPBS blind benchmark (n=130, `deeppbsmar24/run/folds/id.txt`) — **primary comparison**

This is the fully held-out benchmark described in the DeepPBS paper Methods: "a separate fully blindfold
benchmark set was kept aside" and never used during cross-validation training.
Both models evaluated on the same 130 protein chains with identical JASPAR/HOCOMOCO ground-truth PWMs.

All TFScope versions evaluated on DeepPBS blind benchmark. IC-Pearson uses TFScope formula
(IC-weighted PWM Pearson). MAE (DP ×4) converts TFScope per-element MAE to DeepPBS scale.

| Version | Pearson r ↑ | MAE (per-elem) ↓ | MAE (DP ×4) ↓ | IC-Pearson (TF) ↑ | Length MAE ↓ | Top-1 acc ↑ | n |
|---|---|---|---|---|---|---|---|
| **DeepPBS** | **0.702** | **0.138** | **0.553** | — | — | — | 130 |
| TFScope v4 ep.200 | 0.178 | 0.256 | 1.025 | 0.865 | 2.42 | 0.365 | 129 |
| TFScope v5 best (ep.27) | 0.129 | 0.270 | 1.081 | 0.864 | 2.62 | 0.320 | 129 |
| TFScope v5 ep.200 | 0.140 | 0.262 | 1.050 | 0.856 | 2.60 | 0.328 | 129 |
| TFScope v6 best (ep.105) | 0.323 | 0.246 | 0.984 | 0.880 | 1.75 | 0.477 | 130 |
| TFScope v6 ep.165 | 0.377 | 0.226 | 0.906 | 0.888 | 1.83 | 0.513 | 130 |
| **TFScope v6 ep.200** | **0.377** | **0.226** | **0.906** | **0.888** | **1.85** | **0.513** | 130 |

**Training data per version:**
- **v4**: original TFScope dataset (3,257 train), 0% gene overlap with blind benchmark
- **v5 augmented**: v4 dataset + 46 new JASPAR entries from DeepPBS training (3,303 train) — marginal
- **v6 DeepPBS-only**: PDB-derived sequences + NPZ PWMs, exact DeepPBS CV training (471 train) — fair

**Key findings:**

1. **Gene familiarity is the dominant fixable factor.** v6 achieves +0.199 Pearson r over v4 simply by
   training on the same TF identities DeepPBS sees (cluster-mates of blind benchmark TFs).

2. **Sparse augmentation (v5) does NOT help.** Adding only 46 entries to 3,257 (~1.4%) is too small
   to shift the training distribution. v5 < v4 across all metrics — the few new entries can't override
   the original dataset's bias.

3. **All-or-nothing effect.** Gene familiarity helps only when training is *dominated* by the same TF
   pool as the test set. Partial overlap dilutes; full coverage works.

4. **Remaining gap = pure modality.** v6 (0.377) vs DeepPBS (0.702) → **0.325 Pearson r** is the
   honest measure of what 3D structural input + DNA provides over protein sequence alone.

5. **v6 length prediction is much better.** Length MAE drops from 2.42 (v4) to 1.83 (v6) — training
   on actual co-crystal-aligned motif lengths teaches the model to predict motif width correctly.

**Results dirs:**
- DeepPBS: `results/deeppbs_blind_benchmark/`
- TFScope v4: `results/tfscope_v4_blind_benchmark/`
- TFScope v5 best: `results/tfscope_v5_augmented_best/`
- TFScope v5 ep.200: `results/tfscope_v5_augmented_ep200/`
- TFScope v6 best: `results/tfscope_v6_only_best/`
- TFScope v6 ep.165: `results/tfscope_v6_deeppbs_only/`
- TFScope v6 ep.200: `results/tfscope_v6_only_ep200/`

**Key finding:** Architecture changes alone (v3) did not help. The KL term in the loss creates a degenerate
flat-prediction attractor; removing it (v4) is the critical fix.

---

## v4 — Drop KL, plain L1 primary loss (2026-05-18)

**Checkpoint dir:** `checkpoints/deeppbs_benchmark_v4/`
**Job:** 13717674 (submitted 2026-05-18, running)
**Architecture:** identical to v3 (48.62M params)
**Results:** pending

### Problem diagnosed from v3 training logs
Val `L_pwm` was ~2.47 from epoch 1 through epoch 49 — completely flat. The PWM head never learned.
Root cause: `KL(uniform ∥ uniform) = 0`, so the model could reduce total loss while staying near
[0.25, 0.25, 0.25, 0.25] per position. Architecture improvements had no effect because the loss
landscape itself had a degenerate attractor.

### Key finding: DeepPBS loss (from source inspection)
File: `/n/home13/leihuang/project/DeepPBS/deeppbs/nn/trainer.py` + `run/config.json`
```python
# config.json: ic_loss_weight=0, mse_loss_weight=1
loss = L1(softmax(output), target)   # plain unweighted L1, nothing else
```
They train directly on the evaluation metric. No KL, no entropy, no IC loss in production.

### Results (training complete, 200 epochs)

| Metric | v4 best (ep.32) | v4 ep.200 | v3 ep.33 | DeepPBS |
|---|---|---|---|---|
| Pearson r | 0.128 | **0.185** | 0.175 | 0.757 |
| IC-Pearson (DP formula) | 0.352 | **0.452** | 0.289 | 0.496 |
| MAE per-element | 0.265 | **0.251** | 0.241 | 0.124 |
| MAE (DeepPBS scale) | 1.058 | 1.002 | 0.964 | 0.496 |
| Length MAE (bp) | 2.45 | 2.42 | 2.47 | — |

**Key finding:** val-loss-best (epoch 32) is a poor proxy — epoch 200 outperforms it on all test
metrics. L1-only loss improved IC-Pearson (0.289→0.452 on DeepPBS formula) substantially, but
Pearson r (0.175→0.185) and MAE (0.241→0.251) show only marginal change. The problem is no
longer the loss — it is generalization: the model can't transfer nucleotide specificity from
training TFs to the held-out benchmark TFs.

**Next direction:** either stronger LoRA (more layers / higher rank) to better adapt ESM2
representations, or data augmentation / multi-task learning to improve cross-family transfer.

### Changes

#### `src/tfscope/losses/tfscope_loss.py`
- Removed `L_kl` call and `pwm_kl` from metrics dict
- `L_pwm` is now: `pwm_l1_weight × L1 + pwm_ic_weight × |ΔIC| + pwm_entropy_weight × H(pred)`

#### `src/tfscope/config.py`
- `pwm_l1_weight`: 1.5 → **1.0** (matches DeepPBS's primary coefficient)
- Removed `label_smoothing` field (was only used in KL term)

### Full L_pwm after change
```
L_pwm = 1.0 × L1(softmax(pred), target)
      + 0.5 × |IC(target) − IC(pred)|
      + 0.1 × H(pred)
```

---

## v3 — Gated Pooling + SwiGLU MoE + Prototypes + Cross-Attn Decoder (2026-05-18)

**Checkpoint dir:** `checkpoints/deeppbs_benchmark_v3/`
**Job:** 13682939
**Model size:** 48.62M trainable params (vs 31.87M in v2)
**Results dir:** `results/deeppbs_v3_best/`

### Results (ckpt_best.pt = epoch 33, best val loss 0.9803)

| Metric | TFScope v3 | DeepPBS | Same scale? |
|---|---|---|---|
| Pearson r mean | 0.175 | 0.757 | ✓ |
| IC-wtd Pearson (TF formula) | 0.897 | — | — |
| IC-Pearson (DeepPBS formula) | 0.289 | 0.496 | ✓ |
| MAE per-element | 0.241 | 0.124 | ✓ |
| MAE DeepPBS scale (×4) | 0.964 | 0.496 | ✓ |
| Length MAE (bp) | 2.47 | — | — |

Val loss plateaued from epoch 1; best val loss epoch 34, overfitting after that.
Val L_pwm never moved (~2.47 throughout) — confirmed KL as root cause.

### Architecture changes (vs v2)

#### 1. Gated Attention Pooling
Source: *Gated Attention for LLMs* (NeurIPS 2024 Best Paper)
- `GatedAttentionPooling` replaces `AttentionPooling` for both global and DBD streams
- Gate: `gated_v = σ(W_g · x) ⊙ v` — prevents attention sinks
- Files: `src/tfscope/models/pooling.py`, `tfscope.py`

#### 2. DBD Position Indicator
Source: ESM-DBP (Nat. Commun. 2024)
- Learned 1280-dim embedding added to DBD residues before pooling, zero-init
- Files: `src/tfscope/models/tfscope.py`

#### 3. SwiGLU Routed Experts
Source: DeepSeek-V2/V3, LLaMA 2/3
- `ExpertMLP` (GELU) → `SwiGLUExpert`: `W_down(silu(W_gate·x) ⊙ W_up·x)`
- Files: `src/tfscope/models/moe.py`

#### 4. Shared Expert (DeepSeek MoE)
- 1 always-active SwiGLU expert alongside top-2 routed experts
- Files: `src/tfscope/models/moe.py`, `config.py`

#### 5. Prototype Dictionary
Source: Prototype Transformer (ProtoT), arXiv 2602.11852, Feb 2026
- `PrototypeDictionary(n_prototypes=32, hidden_dim=512)` — soft-attention retrieval
- `proto_weights (B, 32)` in `aux_dict` for interpretability
- Files: `src/tfscope/models/moe.py`, `config.py`

#### 6. Cross-Attention PWM Decoder
Source: AlphaFold / Perceiver IO
- Each PWM position cross-attends to ESM2 DBD residues (kdim=1280, DBD-masked)
- Files: `src/tfscope/models/heads.py`, `tfscope.py`

### Config additions
```python
n_shared_experts: int = 1
n_prototypes: int = 32
pwm_cross_attn: bool = True
```

---

## v2 — Sharpness loss fixes (2026-05-18)

**Checkpoint dir:** `checkpoints/deeppbs_benchmark_lora/` (same dir as baseline, continued training)
**Results dir:** `results/deeppbs_lora_best/`

### Results (ckpt_best.pt)

| Metric | v2 best | Baseline (ep.55) | Change |
|---|---|---|---|
| Pearson r mean | 0.212 | 0.200 | +0.012 |
| IC-wtd Pearson | 0.901 | 0.898 | +0.003 |
| MAE per-element | 0.239 | 0.236 | +0.003 |
| Length MAE (bp) | 2.35 | 2.43 | −0.08 |

Marginal improvement. Still flat predictions — root cause was the KL term, not these three fixes.

### Changes (vs baseline)

#### `src/tfscope/losses/tfscope_loss.py` + `src/tfscope/config.py`
1. **Wire up label smoothing** — was in config but never applied to KL target
   - `label_smoothing`: 0.1 → 0.02; applied as `t' = (1−α)t + α×0.25`
2. **Add entropy penalty** — new `pwm_entropy_weight = 0.1`; `L_entropy = mean H(pred)`
3. **Increase L1 weight** — `pwm_l1_weight`: 0.5 → 1.5

### Full L_pwm
```
L_pwm = KL(smoothed) + 1.5×L1 + 0.5×|ΔIC| + 0.1×H(pred)
```

---

## Baseline — LoRA fine-tuning (epoch 55)

**Checkpoint:** `checkpoints/deeppbs_benchmark_lora/ckpt_epoch055.pt`
**Split:** `data/processed/splits/deeppbs/benchmark.json` (2822 train / 308 val / 613 test)
**Results dir:** `results/deeppbs_lora_epoch055/`

### Results

| Metric | TFScope | DeepPBS |
|---|---|---|
| Pearson r mean | 0.200 | 0.757 |
| IC-wtd Pearson | 0.898 | — |
| MAE per-element | 0.236 | 0.124 |
| Length MAE (bp) | 2.43 | — |

### Config
```json
"pwm_l1_weight": 0.5,
"pwm_ic_weight": 0.5,
"label_smoothing": 0.1,
"lora_rank": 16,
"lora_n_layers": 6
```

### Diagnosis
Flat prediction: median max-nucleotide prob/position = 0.41 vs target 0.78.
High IC-weighted Pearson (0.898) means model ranks positions by importance correctly,
but assigns near-uniform nucleotide probabilities at each position (low Pearson r).

Root causes (in hindsight):
1. `label_smoothing = 0.1` in config but **never wired into the loss**
2. `pwm_l1_weight = 0.5` too weak
3. **KL term** — has flat-prediction local minimum KL(uniform∥uniform)=0 (identified later at v4)
