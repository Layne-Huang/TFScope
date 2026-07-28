# TFScope：TF-selective DNA sequence design 最小可行计划

## 1. 核心目标

将当前“从随机 DNA 收敛到目标 PWM consensus”的 in-silico SELEX，升级为一个真正需要优化的任务：

> 给定一个目标 TF，设计一组对目标 TF 高分、对相似 off-target TF 低分、满足基本序列约束，并在独立实验 PWM 下仍保持选择性的 DNA 序列。

该分析不再只证明遗传算法能够找回 PWM 的逐列 argmax，而是检验 TFScope 是否能够支持 **factor-selective forward DNA design**。

---

## 2. 最小可行范围

### 2.1 目标 TF

优先使用已有 Figure 3 中的四个代表性 TF：

| Target TF | Family | 预期核心 motif |
|---|---|---|
| LHX5 | Homeodomain | TAAT |
| MYOG | bHLH | CAGCTG / E-box |
| CREB3L2 | bZIP | ACGT |
| ELK1 | ETS | GGAA |

LHX5 作为主案例，其余三个作为跨家族复现。

### 2.2 Off-target TF

每个 target 选择 **5–10 个 hardest off-targets**，优先选择：

1. 同家族 TF；
2. 与 target 的 TFScope 预测 PWM 最相似的 TF；
3. 有独立 curated experimental PWM 的 TF；
4. 如有表达信息，优先选择与 target 在相同细胞环境中可能共存的 TF。

自动选择方式：

1. 将 target PWM 与所有候选 TF PWM 做正反链和 offset 对齐；
2. 计算最大 PWM correlation；
3. 选择相似度最高的前 5–10 个 TF。

保存每个 target 的 off-target 列表及选择依据。

---

## 3. 输入数据

每个 TF 至少需要：

- TFScope 预测 PWM；
- curated experimental PWM，用于独立验证；
- TF family 注释；
- motif 长度；
- 可选：细胞类型或表达背景。

建议目录：

```text
data/
├── predicted_pwms/
│   ├── LHX5.meme
│   ├── MYOG.meme
│   ├── CREB3L2.meme
│   └── ELK1.meme
├── experimental_pwms/
│   ├── LHX5.meme
│   ├── MYOG.meme
│   ├── CREB3L2.meme
│   └── ELK1.meme
└── tf_metadata.tsv
```

在开始优化前检查：

- 每个位置的碱基概率和约为 1；
- 没有 0 概率导致 `log(0)`；
- 预测和实验 PWM 的方向、长度和背景定义已记录；
- experimental PWM 没有参与优化。

---

## 4. DNA 设计空间

不要只设计刚好等于 motif 长度的序列。

MVP 建议：

```yaml
sequence_length: 24
alphabet: [A, C, G, T]
gc_min: 0.35
gc_max: 0.65
max_homopolymer_length: 3
minimum_pairwise_hamming_distance: 5
```

每条 24-bp 序列在正链和反链上滑窗扫描。对于 TF \(k\)，定义：

\[
S_k(s)
=
\max_{\text{window, strand}}
\sum_{j=1}^{L_k}
\log
\frac{P_{k,j}(s_j)+\epsilon}
{P_{\mathrm{bg}}(s_j)}
\]

其中：

- \(P_{k,j}\) 为 TF \(k\) 的 PWM；
- \(P_{\mathrm{bg}}\) 默认设为 0.25，或使用项目统一背景；
- \(\epsilon\) 为小 pseudocount；
- 取所有窗口和两条链上的最大分数。

这样优化需要同时决定：

- motif 出现在 24-bp 序列中的位置；
- motif 使用正链还是反链；
- flanking bases；
- 是否意外形成 off-target motif。

---

## 5. 分数标准化

不同 TF 的 PWM 长度和信息量不同，不能直接比较 raw score。

对于每个 TF \(k\)，随机生成至少 100,000 条满足 GC 约束的 24-bp 背景序列，得到：

\[
\mu_k=\mathbb E[S_k(s)]
\]

\[
\sigma_k=\operatorname{SD}[S_k(s)]
\]

标准化分数：

\[
Z_k(s)
=
\frac{S_k(s)-\mu_k}
{\sigma_k+\epsilon}
\]

此后所有 target 和 off-target 分数均使用 \(Z\)-score。

---

## 6. 最小优化目标

### 6.1 核心 specificity margin

对于目标 TF \(t\) 和其 off-target 集合 \(\mathcal O_t\)：

\[
M_t(s)
=
Z_t(s)
-
\max_{o\in\mathcal O_t} Z_o(s)
\]

它直接衡量：

> 目标 TF 分数相对于最危险 off-target 的优势。

### 6.2 约束惩罚

定义：

\[
C(s)
=
C_{\mathrm{GC}}
+
C_{\mathrm{homopolymer}}
+
C_{\mathrm{forbidden}}
\]

MVP 必须包含：

- GC 含量 35%–65%；
- 不允许 4 个或以上相同碱基连续出现；
- 不允许指定限制性酶切位点；
- 不允许出现 ambiguous bases。

推荐将明显非法序列直接判为无效，而不是仅给予较小惩罚。

### 6.3 最终单目标 fitness

MVP 阶段先使用简单、可解释的目标：

\[
J_t(s)
=
Z_t(s)
-
\lambda \max_{o\in\mathcal O_t}Z_o(s)
-
\alpha C(s)
\]

初始参数：

```yaml
lambda_off_target: 1.0
alpha_constraint: 10.0
```

由于 \(C(s)\) 对非法序列应较大，合法序列优先于高分但不可合成的序列。

MVP 不要求一开始就实现 NSGA-II。先验证 specificity-aware objective 是否优于 consensus 和 target-only GA。

---

## 7. 优化方法

使用简单遗传算法。

### 7.1 推荐参数

```yaml
population_size: 1000
generations: 50
elite_fraction: 0.05
parent_fraction: 0.20
mutation_rate_per_base: 0.04
crossover_probability: 0.30
independent_seeds: 10
final_designs_per_tf: 20
```

### 7.2 每代流程

```text
初始化满足约束的 24-bp DNA population
              ↓
计算 target Z-score
              ↓
计算所有 off-target Z-scores
              ↓
计算 specificity margin 和 fitness
              ↓
保留 elite
              ↓
从高 fitness parents 中采样
              ↓
crossover + point mutation
              ↓
修复或删除违反约束的序列
              ↓
进入下一代
```

### 7.3 最终设计筛选

从所有 seed 的最终候选中：

1. 按 \(M_t(s)\) 降序排列；
2. 选择最高分序列；
3. 删除与已选序列 Hamming distance 小于 5 的候选；
4. 重复直到获得 20 条设计。

这样避免输出大量几乎完全相同的 consensus 变体。

---

## 8. 必须比较的基线

每个 target TF 比较四类序列。

### Baseline 1：Random

随机生成满足相同约束的 24-bp DNA。

### Baseline 2：Consensus embedding

将 target PWM 的逐列 argmax consensus 嵌入 24-bp 背景中。

为避免背景影响，应：

- 尝试所有可能插入位置；
- 尝试正链和反链；
- 在满足约束的候选中选 target score 最高者。

### Baseline 3：Target-only GA

优化：

\[
J_{\mathrm{target-only}}(s)=Z_t(s)
\]

不考虑 off-target。

### Proposed：Specificity-aware GA

优化：

\[
J_{\mathrm{proposed}}(s)
=
Z_t(s)-\max_o Z_o(s)-\alpha C(s)
\]

核心比较不是谁的 target score 最高，而是谁的 specificity margin 最高。

---

## 9. 评价指标

### 9.1 Oracle 内部评价

使用 TFScope 预测 PWM：

- target \(Z_t\)；
- maximum off-target \(Z_o\)；
- specificity margin \(M_t\)；
- 约束通过率；
- 独立 seed 的收敛稳定性；
- 最终设计间 Hamming diversity。

### 9.2 独立实验 PWM 评价

这是最关键的外部验证。

用 curated experimental PWMs 重新计算：

\[
Z_t^{\mathrm{exp}}(s)
\]

\[
M_t^{\mathrm{exp}}(s)
=
Z_t^{\mathrm{exp}}(s)
-
\max_o Z_o^{\mathrm{exp}}(s)
\]

优化过程中不能使用这些 experimental PWMs。

### 9.3 主要终点

主要终点建议定义为：

> Proposed designs 在 independent experimental PWMs 下的 median specificity margin，是否高于 consensus 和 target-only GA。

次要终点：

- target experimental score 是否仍处于高水平；
- off-target experimental score 是否显著下降；
- 四个 TF 中有多少个重复该趋势。

---

## 10. 最小成功标准

将该分析视为值得进入主文，需要至少满足：

1. 四个 target 中至少三个：
   - proposed experimental specificity margin 高于 consensus；
   - proposed experimental specificity margin 高于 target-only GA。
2. Proposed designs 的 target experimental score 不应大幅下降：
   - 建议保持在 consensus target score 的前 80%–90%。
3. 每个 TF 至少获得 10 条满足约束且相互不同的设计。
4. 结果在 10 个随机 seed 中方向一致。
5. 不能只在 TFScope 预测 PWM 上改善，而在 experimental PWM 上完全消失。

如果只在 oracle 内部改善，应定位为方法演示或 Supplementary analysis，而不是独立验证。

---

## 11. 推荐 Figure 布局

### Panel a：设计任务

```text
Target PWM
+
motif-similar off-target PWMs
+
sequence constraints
          ↓
specificity-aware DNA optimization
```

### Panel b：Target–off-target trade-off

横轴：

\[
\max Z_{\mathrm{off-target}}
\]

纵轴：

\[
Z_{\mathrm{target}}
\]

显示：

- random；
- consensus；
- target-only GA；
- proposed designs。

理想序列位于左上角。

### Panel c：Specificity margin

对四个 TF 比较：

- consensus；
- target-only GA；
- proposed。

显示预测 PWM 和实验 PWM 下的 margin，最好使用 paired points 或 boxplots。

### Panel d：Score heatmap

行：最终设计序列。  
列：target TF 和 off-target TF。  
颜色：experimental PWM \(Z\)-score。

理想结果：

- target 列高；
- off-target 列低。

### 可选 Supplementary Panel

- GA convergence；
- seed stability；
- sequence logos；
- mutation robustness；
- 各项约束统计。

---

## 12. 最小代码结构

```text
scripts/
├── select_offtargets.py
├── build_pwm_score_models.py
├── run_specificity_ga.py
├── evaluate_designs.py
└── plot_specificity_design.py

configs/
└── specificity_design.yaml

results/
└── specificity_design/
    ├── off_target_selection.tsv
    ├── random_background_stats.tsv
    ├── all_candidates.tsv
    ├── final_designs.tsv
    ├── oracle_evaluation.tsv
    ├── experimental_pwm_evaluation.tsv
    ├── convergence/
    └── figures/
```

`final_designs.tsv` 至少包含：

```text
target_tf
sequence
seed
rank
target_z_pred
max_offtarget_z_pred
margin_pred
target_z_exp
max_offtarget_z_exp
margin_exp
gc_fraction
best_target_position
best_target_strand
worst_offtarget_tf
minimum_hamming_distance
```

---

## 13. 推荐实施顺序

### Phase 1：单个 LHX5 原型

- 选择 5–10 个 homeodomain off-target；
- 完成 scoring、GA、constraints 和 baseline；
- 验证 proposed 是否优于 consensus；
- 检查结果是否只是 motif 方向或长度处理错误。

### Phase 2：扩展到四个 TF

- MYOG；
- CREB3L2；
- ELK1；
- 使用完全相同的 pipeline 和预定义参数。

不要为每个 TF 单独调参，否则容易产生 cherry-picking。

### Phase 3：独立 PWM 验证和作图

- 用实验 PWM 统一重新打分；
- 统计四个 TF 的 margin 改善；
- 生成主图和补充图。

---

## 14. 预计时间

在预测 PWM 和实验 PWM 已整理好的前提下：

| 任务 | 预计时间 |
|---|---:|
| PWM 解析与 off-target 选择 | 1 天 |
| 评分函数与背景标准化 | 1 天 |
| GA 与 constraints | 1–2 天 |
| Baselines 与多 seed 运行 | 1 天 |
| 独立 PWM 评价 | 1 天 |
| 作图与 QC | 1–2 天 |

总计约 **1 周** 可完成 MVP。

---

## 15. 必须避免的过度表述

可以说：

> TFScope supports specificity-aware optimization of DNA sequences that favor a target TF over motif-similar off-target factors.

不要直接说：

- experimentally validated selective binding；
- designed DNA binds only the target TF；
- de novo discovery of novel specificity；
- full protein–DNA affinity design。

因为当前仍然基于 PWM，且没有湿实验。

---

## 16. 最终推荐结论

如果结果成功，最稳妥的结论是：

> By jointly optimizing target occupancy, off-target avoidance, and basic sequence constraints, TFScope generated diverse DNA sequences with higher predicted target–off-target specificity margins than consensus and target-only designs. The improvement was retained when the sequences were rescored using independent curated experimental PWMs, supporting the use of TFScope for specificity-aware forward DNA sequence design.

中文：

> 通过联合优化目标 TF 得分、相似 off-target TF 回避以及基本序列约束，TFScope 能够生成一组比 consensus 和单目标优化具有更高目标–非目标选择性差值的多样化 DNA 序列；这一改善在独立实验 PWM 重新评分后仍能保持，从而支持 TFScope 用于具有选择性感知的前向 DNA 序列设计。
