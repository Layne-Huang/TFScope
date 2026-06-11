# v10 提升计划：面向 outperform DeepPBS 的优先级路线图

**项目目标**  
在不依赖真实 TF–DNA crystal complex 的前提下，将当前 v10 seed/RAG 模型从 **mean Pearson r ≈ 0.542** 提升到接近或超过 **DeepPBS r ≈ 0.702**，同时保证结果在 **LSO / LGO / low-homology** 设置下可信。

**当前核心判断**

1. **v10 是当前最强 seed model**：v10 的 mean r、MAE、AUC、CE 均明显优于 v7/v11/v13，是最值得继续强化的主干。
2. **v10 的主要短板不是 motif core detection，而是 base composition**：IC-r 已经很高，说明模型知道哪些位置信息量高，但 A/C/G/T 的列分布还不够准确。
3. **retrieval 本身很强**：NN-LSO baseline 已经超过 DeepPBS，说明 v10 的最大机会在于更好地利用 retrieval prior。
4. **augmented data 不能直接混合训练**：v11 加入 4247 augmented samples 后 mean r 下降，说明存在 distribution shift 或 label noise。
5. **Rosetta 不能作为直接 PWM calibrator**：Rosetta calibration 明显伤害 v10；结构/能量模块应稍后以 gated auxiliary 的方式加入，而不是作为主提升路线。

---

# 总体策略

v10 的提升应分为两条线：

## 短期性能线

目标是直接提升 benchmark performance：

```text
v10
→ clean retrieval
→ motif-aligned retrieval
→ K=16/32 multi-neighbor retrieval
→ retrieval-dominant mixture gate
→ NN-teacher distillation
→ IC-weighted Pearson loss
→ augmented data as retrieval donor only
```

## 稍后机制线

目标是形成更强的论文贡献：

```text
v10-RAG++
→ structure-derived residue-base contact map
→ learned recognition-code calibrator
→ gated MM-GBSA refinement
```

短期不要把 MM-GBSA / Rosetta / FoldX 作为主提升手段。先把 v10 的 retrieval 使用效率提高到接近 NN-LSO upper bound。

---

# Part I. 优先实行计划

---

## P0. 先做 clean benchmark 和 leakage 控制

### 目的

先确认 v10 的真实性能，避免模型通过 same-source 或 same-gene retrieval 间接看到 test PWM。

### 当前问题

原始 DeepPBS benchmark 中存在大量 same-source overlap，因此 v10 当前结果不能完全代表 clean de novo generalization。

### 需要立刻生成的 split

```text
1. Original split
   - 保持现有 DeepPBS benchmark
   - 只用于和 DeepPBS 对齐比较

2. LSO split: Leave-Source-Out
   - test sample 的 source_id 不得出现在 train/retrieval index

3. LGO split: Leave-Gene-Out
   - test sample 的 gene / UniProt 不得出现在 train/retrieval index

4. Cluster30 / Cluster40 split
   - test DBD sequence 与所有 train DBD identity < 30% 或 < 40%
```

### 输出文件

```text
data/processed/splits/original.json
data/processed/splits/lso.json
data/processed/splits/lgo.json
data/processed/splits/cluster30.json

data/processed/tf_nn_index_original.json
data/processed/tf_nn_index_lso.json
data/processed/tf_nn_index_lgo.json
data/processed/tf_nn_index_cluster30.json
```

### 必须实现的过滤函数

```python
def allowed_neighbor(query, candidate, mode):
    if candidate.sample_id == query.sample_id:
        return False

    if mode in ["lso", "lgo", "cluster30"]:
        if candidate.source_id == query.source_id:
            return False

    if mode in ["lgo", "cluster30"]:
        if candidate.gene_symbol == query.gene_symbol:
            return False
        if candidate.uniprot_id == query.uniprot_id:
            return False

    if mode == "cluster30":
        if candidate.cluster30 == query.cluster30:
            return False

    return True
```

### 成功标准

每个 test TF 输出一张 leakage audit 表：

```text
test_id
gene_symbol
uniprot_id
source_id
dbd_family
max_train_identity
same_source_in_train
same_gene_in_train
nearest_neighbor_source
nearest_neighbor_gene
leakage_level
```

主文 claim 使用 LSO/LGO/cluster split；Original 只作为 DeepPBS-comparable benchmark。

---

## P1. 建立 v10-clean baseline

### 目的

先不要改架构，直接用 v10 架构在 clean retrieval index 上重新评估/训练，确认性能下降来自哪里。

### 实验

| ID | Model | Train retrieval | Test retrieval | 目的 |
|---|---|---|---|---|
| P1.1 | v10-original | same-source allowed | same-source allowed | 复现实验上限 |
| P1.2 | v10-eval-LSO | same-source allowed | LSO | 测试训练时依赖 leaky retrieval 的程度 |
| P1.3 | v10-train-LSO | LSO | LSO | clean RAG baseline |
| P1.4 | v10-train-LGO | LGO | LGO | 更严格泛化 |
| P1.5 | v10-noRAG | none | none | 真正 sequence-only baseline |

### 重点观察

```text
如果 P1.2 大幅下降，而 P1.3 恢复一部分：
    v10 确实学到了 leaky retrieval 依赖，但 RAG 仍然有效。

如果 P1.3 接近 P1.2：
    说明架构本身不能很好适应 clean retrieval，需要进入 P2-P5。

如果 P1.5 与 P1.3 差距小：
    说明 clean retrieval 没有被模型充分利用。
```

### 推荐命名

```text
v10.0_original
v10.1_clean_lso
v10.1_clean_lgo
```

---

## P2. Motif-aligned retrieval：最优先的结构性改动

### 背景

不同数据库/实验来源的 PWM 可能存在：

```text
offset shift
reverse-complement orientation
flanking length difference
core motif window difference
```

如果 retrieved PWM 没有和 query PWM 对齐，即使 neighbor 是正确的，也会降低 Pearson r。

### 核心思路

对每个 retrieved PWM，枚举所有 offset 和 orientation，用 v10 seed PWM 的 IC pattern 或 PWM similarity 找到最佳对齐。

### Alignment 目标函数

对 retrieved PWM \(M_k\)，寻找：

\[
R_k^* = \arg\max_R Sim(P_{\text{seed}}, R(M_k))
\]

其中 \(R\) 包括：

```text
orientation ∈ {forward, reverse-complement}
shift ∈ [-max_shift, ..., +max_shift]
trim / pad
```

### 推荐 similarity

优先使用组合分数：

\[
Sim = 
0.5 \cdot PCC(P_{\text{seed}}, R(M_k))
+ 0.3 \cdot PCC(IC_{\text{seed}}, IC_{R(M_k)})
+ 0.2 \cdot TopBaseAcc(P_{\text{seed}}, R(M_k))
\]

不要只用 raw PCC，因为 v10 的 IC-r 很高，IC pattern 对定位 core motif 很有帮助。

### 实现伪代码

```python
def align_pwm_to_seed(neighbor_pwm, seed_pwm, max_shift=6):
    candidates = []

    for orientation in ["forward", "revcomp"]:
        pwm_o = maybe_revcomp(neighbor_pwm, orientation)

        for shift in range(-max_shift, max_shift + 1):
            pwm_aligned = shift_and_pad(pwm_o, shift, target_len=seed_pwm.shape[-1])

            score = (
                0.5 * pwm_pcc(seed_pwm, pwm_aligned)
                + 0.3 * ic_pcc(seed_pwm, pwm_aligned)
                + 0.2 * topbase_acc(seed_pwm, pwm_aligned)
            )

            candidates.append((score, pwm_aligned, orientation, shift))

    return max(candidates, key=lambda x: x[0])
```

### 实验

| ID | 改动 | 预期 |
|---|---|---|
| P2.1 | v10 + alignment only | mean r 上升，CE 下降 |
| P2.2 | alignment by PWM PCC | baseline |
| P2.3 | alignment by IC pattern | 可能更稳 |
| P2.4 | alignment by combined score | 推荐最终版本 |

### 成功标准

```text
LSO mean r 至少 +0.03
Original mean r 至少 +0.03
高 IC positions 的 per-column r 明显提升
```

---

## P3. K=16/32 multi-neighbor retrieval

### 当前问题

v10 使用 K=3。K 太小，尤其在 LSO/LGO 设置下容易被单个 noisy neighbor 带偏。

### 改法

将 retrieval_k 从 3 扩展到：

```text
K = 16
K = 32
```

然后使用 learned attention 选择有效 neighbor，而不是简单平均。

### Retrieval branch

\[
P_{\text{retrieval}} = \sum_{k=1}^{K} \alpha_k \cdot Align(PWM_k)
\]

其中：

\[
\alpha_k = softmax(MLP(x_k))
\]

neighbor feature \(x_k\) 包括：

```text
DBD cosine similarity
full sequence cosine similarity
family match indicator
motif length compatibility
source quality score
trust score
alignment score
neighbor PWM IC
```

### 实验

| ID | K | Alignment | 目的 |
|---|---:|---|---|
| P3.1 | 3 | yes | 对照 |
| P3.2 | 8 | yes | 看是否稳定提升 |
| P3.3 | 16 | yes | 推荐 |
| P3.4 | 32 | yes | 检查是否过多 noisy neighbor |

### 成功标准

```text
K=16 优于 K=3
K=32 不明显过拟合
attention 权重集中在少数高 trust neighbors
```

---

## P4. Retrieval-dominant mixture gate

### 当前 v10 问题

v10 是 additive log-prior：

\[
z = z_{\text{denovo}} + \beta z_{\text{retrieval}}
\]

这会让 de novo logits 过度干扰强 retrieval prior。由于 NN-LSO 已经超过 DeepPBS，v10 应该在高 trust 情况下更接近 retrieval，而不是只把 retrieval 当作辅助项。

### 推荐新融合方式

使用 probability mixture：

\[
P_{\text{final},j}
=
w_j P_{\text{retrieval},j}
+
(1-w_j)P_{\text{denovo},j}
\]

或者 logit mixture：

\[
z_{\text{final},j}
=
w_j z_{\text{retrieval},j}
+
(1-w_j)z_{\text{denovo},j}
+
\delta z_j
\]

其中 \(w_j\) 是 position-specific gate。

### Gate 输入

```text
retrieval trust score
alignment score
top-1/top-2 neighbor similarity gap
retrieval PWM entropy
retrieval PWM IC
seed PWM entropy
seed/retrieval disagreement
DBD family
motif position embedding
```

### 推荐初始化

```text
high trust: w ≈ 0.8
medium trust: w ≈ 0.5
low trust: w ≈ 0.1–0.2
```

可以用 bias 初始化：

```python
gate_bias = logit(0.7)
```

但要让 low-trust case 可以 fallback。

### 正则

防止 gate 学成全 retrieval 或全 de novo：

\[
L_{\text{gate-reg}}
=
\lambda \cdot \left| mean(w) - w_0 \right|
\]

其中 \(w_0\) 初始可设为 0.5–0.7。

### 实验

| ID | Fusion | 目标 |
|---|---|---|
| P4.1 | v10 additive | baseline |
| P4.2 | probability mixture | 推荐 |
| P4.3 | logit mixture | 比较 |
| P4.4 | mixture + residual delta | 最终候选 |

### 成功标准

```text
模型性能接近 NN-LSO，但在 NN 错误样本上优于 NN-LSO。
高 trust 样本：prediction ≈ retrieval。
低 trust 样本：prediction ≈ denovo。
```

---

## P5. NN-teacher distillation

### 目的

因为 NN-LSO baseline 已经很强，应显式训练模型学习什么时候 copy NN，什么时候修正 NN。

### Teacher 构造

在 training set 内部，对每个 query 用 clean retrieval index 找 neighbor，计算：

\[
r_{\text{NN-target}} = PCC(PWM_{\text{NN}}, PWM_{\text{target}})
\]

然后定义 teacher weight：

```python
if r_nn_target > 0.75:
    teacher_weight = 1.0
elif r_nn_target > 0.6:
    teacher_weight = 0.5
else:
    teacher_weight = 0.0
```

### Loss

\[
L_{\text{teacher}}
=
teacher\_weight \cdot KL(P_{\text{retrieval-branch}} \| P_{\text{NN}})
\]

或：

\[
L_{\text{teacher}}
=
teacher\_weight \cdot KL(P_{\text{final}} \| P_{\text{NN}})
\]

推荐先作用在 retrieval branch，而不是 final output。这样 de novo branch 还能修正。

### 注意

test 阶段绝不能使用 target PWM 计算 trust。teacher 只用于 training 内部监督 trust/gate。

### 实验

| ID | Teacher target | 说明 |
|---|---|---|
| P5.1 | no teacher | baseline |
| P5.2 | retrieval branch only | 推荐 |
| P5.3 | final output | 可能过度 copy |
| P5.4 | gate supervision | 让 gate 学会 trust |

### 成功标准

```text
高质量 retrieval case: final r 接近 NN。
低质量 retrieval case: final r 不低于 denovo。
overall r 明显超过 v10。
```

---

## P6. 加入 IC-weighted Pearson loss

### 当前问题

v10 的 IC-r 高，但 mean Pearson r 不够，说明模型知道重要位置，却没学准每列 A/C/G/T 的比例。

### 新 loss

对每个 motif position \(j\)：

\[
L_{\text{PCC},j}
=
1 - PCC(P_{\text{pred},j}, P_{\text{target},j})
\]

用 target IC 加权：

\[
L_{\text{IC-PCC}}
=
\sum_j IC_{\text{target},j} \cdot L_{\text{PCC},j}
\]

### 总 loss

```text
L_total =
    CE/KL loss
  + λ_pcc * IC-weighted Pearson loss
  + λ_ic  * IC matching loss
  + λ_top * top-base margin loss
  + λ_gate * gate loss
  + λ_balance * MoE balance
  + λ_diversity * MoE diversity
```

### 推荐超参

```text
λ_pcc = 0.2, 0.5, 1.0
λ_top = 0.1
λ_ic = 0.3 或保留当前 0.5
entropy penalty 降低到 0.03–0.05，避免过度影响 base composition
```

### Top-base margin loss

\[
L_{\text{top}}
=
\max(0, m - z_{\text{true top}} + z_{\text{second}})
\]

只在 high-IC positions 上启用。

### 成功标准

```text
mean Pearson r 上升
AUC 不下降
CE/KL 不明显变差
top-base accuracy 上升
```

---

## P7. Augmented data 作为 retrieval donor，而不是 supervised target

### 当前问题

v11 直接用 augmented data 训练后 mean r 下降，因此 augmented data 不能无权重地混入 supervised loss。

### 推荐方案

```text
supervised training target: 520 structure-derived TFs
retrieval donor index: 4247 augmented TFs
```

即 augmented data 只作为 neighbor candidate。

### Donor quality score

为每个 donor 计算：

```text
source_quality
motif_IC_quality
family_annotation_confidence
motif_length_compatibility
duplicate_conflict_score
```

构成：

\[
q_k \in [0,1]
\]

并输入 retrieval attention：

\[
\alpha_k = softmax(MLP([... , q_k]))
\]

### 实验

| ID | Supervised set | Retrieval donor | 目的 |
|---|---|---|---|
| P7.1 | 520 | 520 | baseline |
| P7.2 | 520 | 4247 unweighted | 看 donor coverage 是否有用 |
| P7.3 | 520 | 4247 quality-weighted | 推荐 |
| P7.4 | 4247 pretrain → 520 finetune | 稍后尝试 |

### 成功标准

```text
P7.3 > P7.1
family coverage 提升
low-homology / LGO 样本提升
不再出现 v11 的 mean r 下降
```

---

# Part II. 优先实验矩阵

建议按以下顺序跑。不要一次性把所有改动混在一起，否则无法定位有效因素。

| 阶段 | 实验 ID | 改动 | 预计收益 | 风险 |
|---|---|---|---:|---|
| 0 | P1.1–P1.5 | clean eval/index | 可信结果 | 可能暴露性能下降 |
| 1 | P2.1–P2.4 | motif alignment | 高 | 实现细节影响大 |
| 2 | P3.1–P3.4 | K=16/32 | 中-高 | noisy neighbors |
| 3 | P4.1–P4.4 | mixture gate | 高 | gate collapse |
| 4 | P5.1–P5.4 | NN-teacher | 高 | 过度 copy retrieval |
| 5 | P6 | IC-PCC loss | 中 | AUC/CE tradeoff |
| 6 | P7 | augmented donor only | 中 | donor noise |

推荐组合版本：

```text
v10.0_original
v10.1_clean_lso
v10.2_align
v10.3_align_K16
v10.4_align_K16_mixgate
v10.5_align_K16_mixgate_teacher
v10.6_align_K16_mixgate_teacher_ICPCC
v10.7_align_K16_mixgate_teacher_ICPCC_augdonor
```

---

# Part III. 两周执行时间表

## Week 1

### Day 1–2: Clean benchmark

```text
- build leakage_audit.py
- build lso/lgo/cluster split
- build split-specific retrieval index
- run v10 eval under original/lso/lgo
```

产出：

```text
results/v10_leakage_audit.csv
results/v10_original_eval.json
results/v10_lso_eval.json
results/v10_lgo_eval.json
```

### Day 3–4: Motif alignment

```text
- implement align_pwm_to_seed()
- compare PCC-only / IC-only / combined alignment
- evaluate v10 + aligned retrieval
```

产出：

```text
results/v10_align_ablation.csv
plots/alignment_examples/
```

### Day 5–7: K=16/32 retrieval

```text
- rebuild index with K=16/32
- add neighbor attention features
- train/evaluate v10.3
```

产出：

```text
results/v10_k_ablation.csv
plots/neighbor_attention_weights/
```

## Week 2

### Day 8–10: Mixture gate

```text
- implement P_final = w P_retrieval + (1-w) P_denovo
- add position-specific gate
- evaluate gate behavior by trust score
```

产出：

```text
results/v10_mixgate.csv
plots/gate_vs_trust.png
plots/gate_by_family.png
```

### Day 11–12: NN-teacher

```text
- compute train-set NN-target r
- add teacher-weighted KL loss
- compare retrieval-branch vs final-output distillation
```

产出：

```text
results/v10_teacher.csv
plots/teacher_gain_by_trust.png
```

### Day 13–14: IC-PCC loss and final combination

```text
- add IC-weighted Pearson loss
- train final v10.6
- report original/lso/lgo metrics
```

产出：

```text
results/v10_final_report.md
results/v10_final_metrics.csv
```

---

# Part IV. 稍后施行计划

---

## L1. Contact-code calibrator

### 目的

提高 exact base composition，并让模型具有 residue-level interpretability。

### 形式

\[
z_{\text{final}}
=
z_{\text{v10-RAG++}}
+
\sum_{i:(i,j)\in C}
w_{ij} g_f(h_i,a_i,b)
\]

其中：

```text
C: Boltz/AF3/PDB-derived residue-base contact map
h_i: family-aligned DBD position
a_i: amino acid identity
g_f: family-specific recognition-code parameter
```

### 适用时机

等 v10.6/v10.7 稳定后再做。不要阻塞短期 RAG++ 提升。

---

## L2. MM-GBSA gated refinement

### 当前判断

MM-GBSA 比 Rosetta 有希望，但不能直接替换 v10 PWM。应作为 gated auxiliary。

### 推荐形式

\[
P_{\text{final}}
=
(1-w)P_{\text{v10-RAG++}}
+
wP_{\text{MMGBSA}}
\]

gate 输入：

```text
retrieval trust
seed entropy
Boltz iPTM
MM-GBSA energy spread
contact density
motif IC
```

### 必须避免

```text
single consensus DNA → Boltz → MM-GBSA → direct PWM replacement
```

因为会有 seed propagation problem。

### 更好方案

```text
sample top-N DNA beam from v10-RAG++
run Boltz/MSA for top candidates
select high-confidence structures
ensemble MM-GBSA PWM
gated fusion into v10 output
```

---

## L3. Augmented pretrain → 520 finetune

### 目的

让 augmented data 学到 broad family grammar，但最终回到 high-quality 520 distribution。

### 方案

```text
Stage 1:
    pretrain on 4247 augmented samples
    lower learning rate
    stronger dropout
    quality-weighted loss

Stage 2:
    finetune on 520 structure-derived samples
    freeze ESM/LoRA partially
    tune RAG gate + PWM head + trust predictor

Stage 3:
    evaluate original/lso/lgo/cluster30
```

---

## L4. Low-homology / leave-family benchmark

### 目的

形成更强论文 claim：

```text
not merely homology transfer
not merely motif retrieval
can generalize in low-homology or family-held-out cases
```

### 设置

```text
cluster30 split
cluster40 split
leave-subfamily-out
leave-family-out
ortholog split
```

---

# Part V. 关键诊断图

每一轮实验都应该输出以下分析图，而不是只看 overall mean r。

## Retrieval 诊断

```text
1. NN similarity vs final improvement
2. trust score vs actual NN-target r
3. gate weight vs trust score
4. attention weight distribution over K neighbors
5. aligned vs unaligned retrieval PWM examples
```

## Error 诊断

```text
1. Δr by DBD family
2. Δr by motif length
3. Δr by source database
4. Δr by IC level
5. worst 20 cases
6. v10 better than NN cases
7. NN better than v10 cases
```

## Leakage 诊断

```text
1. original vs LSO vs LGO performance
2. same-source cases vs clean cases
3. same-gene cases vs novel-gene cases
4. performance vs max train identity
```

---

# Part VI. 最终推荐配置

如果只能训练一个最值得尝试的版本，建议配置如下：

```yaml
model_name: v10.6_align_K16_mixgate_teacher_ICPCC

backbone:
  esm_model: esm2_t33_650M_UR50D
  lora_layers: last_6
  lora_rank: 16
  lora_alpha: 32

retrieval:
  mode: LSO
  k: 16
  alignment: combined_pwm_ic_topbase
  retrieval_dropout: 0.15
  augmented_donors: false  # first run false; later true
  same_source_filter: true
  same_gene_filter: optional_for_LGO

fusion:
  type: probability_mixture
  position_specific_gate: true
  residual_delta: true
  gate_inputs:
    - trust_score
    - alignment_score
    - retrieval_entropy
    - retrieval_ic
    - seed_entropy
    - seed_retrieval_disagreement
    - family_embedding
    - motif_position_embedding

loss:
  pwm_ce_weight: 1.0
  ic_pcc_weight: 0.5
  ic_match_weight: 0.3
  topbase_margin_weight: 0.1
  trust_loss_weight: 0.5
  nn_teacher_weight: 0.3
  moe_balance_weight: keep_current
  moe_diversity_weight: keep_current

training:
  batch_size: 32_or_64
  lr: 3e-4_to_6e-4
  warmup_steps: 500
  early_stop_metric: LSO_mean_r
  eval_splits:
    - original
    - LSO
    - LGO
```

---

# Part VII. 成功判据

短期成功：

```text
Original split mean r > 0.70
LSO split mean r > 0.65
AUC remains ≥ 0.78
CE < v10 CE
v10-RAG++ > NN-LSO on at least a subset of low-trust NN cases
```

中期成功：

```text
LSO mean r ≥ DeepPBS original r
LGO mean r close to or above DeepPBS
augmented donor improves LGO/low-homology cases
contact-code improves high-IC contact positions
```

论文级成功：

```text
Clean retrieval-controlled model beats or matches DeepPBS without crystal structures.
Model explains when it copies homologous motifs and when it performs de novo correction.
Structure is used as mapping/interpretability rather than raw energy replacement.
```

---

# 最后结论

v10 的提升不应该从“继续加数据”或“继续物理能量校准”开始，而应该从 **更强、更干净、更可控的 retrieval fusion** 开始。

最优先路线是：

```text
clean LSO/LGO index
→ motif-aligned retrieval
→ K=16 multi-neighbor retrieval
→ retrieval-dominant mixture gate
→ NN-teacher distillation
→ IC-weighted Pearson loss
→ augmented data as retrieval donor only
```

这条路线最有希望把 v10 从当前 **r≈0.54** 推向 **r>0.70**。  
contact-code 和 MM-GBSA 应该作为稍后增强，用于机制解释和局部 refinement，而不是阻塞当前 v10 性能提升。
