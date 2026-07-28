# TFScope 孤儿转录因子 motif 的最小可行计算验证方案

**项目目标**  
利用公开的 TF-specific ChIP-seq 数据，验证 TFScope 仅根据蛋白序列预测的 orphan-TF motif 是否能够解释对应转录因子的真实染色质结合区域。

**方案定位**  
这是一个完全基于公开数据的计算验证，不需要重新处理 FASTQ，也不需要开展任何湿实验。第一阶段只使用已发布的 processed ChIP-seq peak 文件，重点回答：

> TFScope 预测的 motif 是否在对应 TF 的 ChIP-seq peaks 中富集，并集中于 peak summit 附近？

---

## 1. 最小研究范围

第一阶段只分析三个拥有公开 TF-specific ChIP-seq 数据的转录因子：

| TF | 细胞系 | 公开数据 | 推荐用途 |
|---|---|---|---|
| ADNP | K562 | GEO **GSE105573** / ENCODE **ENCSR440VKE** | 主验证 |
| ZHX2 | HepG2 | ENCODE **ENCSR407BEZ** | 主验证 |
| ZHX3 | HepG2 | GEO **GSE170636** / ENCODE **ENCSR367KYL** | 主验证 |

### 可选的低成本复现数据

| TF | 细胞系 | 数据 | 用途 |
|---|---|---|---|
| ZHX2 | MCF-7 | GEO **GSE96441** / ENCODE **ENCSR876UYH** | 跨细胞系复现 |
| ZHX2 | MDA-MB-231 | GEO **GSE175487** | ChIP-seq + ZHX2 knockdown RNA-seq 的后续增强分析 |
| ADNP | mouse ESC | GEO **GSE97945** | 跨物种辅助验证 |

第一轮不建议强行纳入 SOHLH1、ADNP2 和 ZGLP1，因为目前缺少同等强度、易直接使用的 TF-specific occupancy 数据。它们可以保留现有 cCRE enrichment，后续再做组织匹配的 ATAC-seq 或扰动转录组分析。

---

## 2. 本方案能证明什么、不能证明什么

### 可以支持的结论

- 预测 motif 在对应 TF 的 ChIP-seq peaks 中比组成匹配背景更常见。
- motif 高分位点更靠近 ChIP peak summit。
- 预测 motif 对对应 TF peaks 具有一定区分能力。
- 预测 motif 的表现优于保持长度、GC 倾向和信息量的随机 motif。

### 不能直接证明的结论

- 不能证明每个 motif hit 都被该 TF 实际结合。
- 不能证明所有 motif-positive enhancer/promoter 都是该 TF 的直接靶点。
- 如果多个同家族 TF 识别相似 motif，不能仅凭 motif 区分具体 paralog。
- ChIP peak 可能包含间接结合、共因子结合或抗体/标签相关偏差。

推荐措辞：

> Public TF-specific occupancy data support the biological plausibility of the predicted motifs.

避免使用：

> The predicted motifs prove the direct binding specificity of the orphan TFs.

---

## 3. 需要准备的输入

### 3.1 TFScope 预测 motif

每个 TF 准备一个标准 PWM，推荐 MEME motif format：

```text
motifs/
├── ADNP.meme
├── ZHX2.meme
└── ZHX3.meme
```

同时保存：

- motif 长度；
- 每列 A/C/G/T 概率；
- reverse-complement 处理方式；
- motif 来源模型和 checkpoint；
- 是否为 combined、no-RAG 版本；
- 预测生成日期。

分析开始后冻结 PWM，不要根据 ChIP 结果人工修改 motif。

### 3.2 Processed ChIP-seq peaks

优先下载：

- GRCh38/hg38；
- conservative IDR thresholded peaks；
- BED、narrowPeak 或 bigBed 格式；
- 如有多个 processed peak 集，优先选择 ENCODE 推荐的 replicated/IDR peak 集。

不要在第一阶段下载 FASTQ、BAM 或自行 call peaks。

### 3.3 参考文件

- hg38 reference FASTA；
- hg38 chromosome sizes；
- ENCODE blacklist；
- 可选：JASPAR/HOCOMOCO 中同 family motif，用于 family-matched control。

---

## 4. 分析目录结构

```text
orphan_tf_validation/
├── README.md
├── config.yaml
├── motifs/
│   ├── ADNP.meme
│   ├── ZHX2.meme
│   └── ZHX3.meme
├── data/
│   ├── raw_peaks/
│   ├── processed_peaks/
│   ├── genome/
│   └── controls/
├── results/
│   ├── motif_scan/
│   ├── enrichment/
│   ├── centrality/
│   ├── classification/
│   ├── null_pwm/
│   └── figures/
└── scripts/
    ├── 01_prepare_peaks.py
    ├── 02_extract_sequences.sh
    ├── 03_generate_controls.py
    ├── 04_scan_motifs.sh
    ├── 05_compute_statistics.py
    └── 06_make_figures.py
```

---

## 5. 核心分析流程

### Step 0：冻结分析配置

建议预先固定以下参数：

```yaml
genome: hg38
peak_window_bp: 500
shuffle_per_peak: 20
genomic_negative_ratio: 1
fimo_pvalue_threshold: 1e-4
column_shuffled_pwm_number: 100
bootstrap_iterations: 1000
random_seed: 42
```

阈值可以调整，但必须在查看最终结果前确定，并在所有 TF 中一致使用。

### Step 1：标准化 ChIP peaks

对每个 TF：

1. 下载 processed conservative IDR peaks。
2. 确认 genome assembly 为 hg38。
3. 删除非标准染色体、ENCODE blacklist 区域和超出染色体边界的区域。
4. 使用 summit 作为中心；如果 peak 文件没有 summit 字段，则使用 peak midpoint。
5. 从中心截取固定长度窗口：

\[
-250\ \text{bp} \;\text{to}\; +250\ \text{bp}
\]

使用固定 500 bp 窗口可以避免不同 peak 长度引起 motif hit 数量偏差。

输出：

```text
data/processed_peaks/ADNP.hg38.500bp.bed
data/processed_peaks/ZHX2.hg38.500bp.bed
data/processed_peaks/ZHX3.hg38.500bp.bed
```

### Step 2：提取真实 peak 序列

使用 `bedtools getfasta` 提取 hg38 序列：

```bash
bedtools getfasta \
  -fi hg38.fa \
  -bed ADNP.hg38.500bp.bed \
  -fo ADNP.peaks.fa
```

对 ZHX2 和 ZHX3 重复相同操作。

### Step 3：构造背景序列

#### 必做背景：dinucleotide-shuffled peaks

对每条真实 peak 序列生成 20 条 dinucleotide shuffle：

- 保持序列长度；
- 尽量保持二核苷酸频率；
- 因此同时控制 GC 含量和局部序列组成；
- 打乱真实 motif 的位置顺序。

该背景回答：

> 在相同序列组成下，真实 ChIP peaks 的碱基排列是否更支持预测 motif？

#### 推荐背景：GC 和长度匹配的 genomic negatives

为每个真实 peak 选择一个同染色体、等长、GC 相近、不与 ChIP peaks 或 blacklist 重叠的随机基因组区域。

该背景用于：

- peak vs non-peak classification；
- AUROC/AUPRC；
- Fisher’s exact test；
- 验证结果不只来自 shuffle 算法。

第一轮时间非常有限时，可以先只完成 dinucleotide shuffle；GC-matched genomic negatives 作为第二轮增强。

### Step 4：扫描预测 motif

推荐使用 MEME Suite：

- **FIMO**：输出 motif hits、位置、strand、score 和 p-value；
- **CentriMo**：检测 motif 是否集中在 peak 中心；
- 可选 **AME**：进行 sequence-level motif enrichment。

对以下序列使用同一 PWM、同一阈值和同一正反链设置：

1. 真实 ChIP peak sequences；
2. dinucleotide-shuffled sequences；
3. GC/长度匹配 genomic negatives。

每条序列至少保留：

- 是否存在显著 motif hit；
- motif hit 数量；
- 最高 PWM/FIMO score；
- 最佳 hit 与 peak summit 的距离；
- 所有 hit 的位置。

---

## 6. 必做统计指标

### 6.1 Composition-controlled enrichment

定义：

\[
E =
\frac{\text{hits per kb in real peaks}}
{\text{mean hits per kb in dinucleotide shuffles}}
\]

绘图使用：

\[
\log_2 E
\]

解释：

| \(\log_2 E\) | 含义 |
|---:|---|
| 0 | 与 shuffle 背景相同 |
| 1 | 2 倍富集 |
| 2 | 4 倍富集 |
| 4 | 16 倍富集 |
| < 0 | 相对背景耗竭 |

同时报告 shuffle null distribution：

\[
z =
\frac{C_{\text{observed}}-\mu_{\text{shuffle}}}
{\sigma_{\text{shuffle}}}
\]

推荐同时给经验 p-value：

\[
p_{\text{empirical}} =
\frac{1+\#(C_{\text{shuffle}}\ge C_{\text{observed}})}
{1+N_{\text{shuffle}}}
\]

如果每条 peak 独立 shuffle，统计时应以完整 shuffle replicate 为单位，避免把所有 shuffled sequences 错误地当成完全独立样本。

### 6.2 Hit-positive peak 比例与 odds ratio

把每个 peak 定义为 motif-positive 或 motif-negative，比较 real peaks 与 matched genomic negatives：

\[
OR =
\frac{
n_{\text{peak, hit}}/n_{\text{peak, no hit}}
}{
n_{\text{background, hit}}/n_{\text{background, no hit}}
}
\]

使用 Fisher’s exact test，并对三个 TF 的 p-value 做 Benjamini–Hochberg FDR 校正。

### 6.3 Peak/background classification

对每条序列使用最高 motif score：

\[
S_i = \max_j \operatorname{PWMscore}_{ij}
\]

区分真实 ChIP peaks 与 GC/长度匹配背景，并报告：

- AUROC；
- AUPRC；
- 1,000 次 bootstrap 95% confidence interval。

这里不训练额外分类器，直接使用 PWM score，避免增加不必要的模型复杂度。

### 6.4 Summit-centered enrichment

对每个 motif hit 计算：

\[
d_i =
\text{motif center}_i-\text{peak summit}_i
\]

画出 \(-250\) 到 \(+250\) bp 的 hit density，并报告：

- \(|d|\le 50\) bp 内的 hit 比例；
- \(|d|\le 100\) bp 内的 hit 比例；
- 与均匀位置分布或 shuffled sequences 的比较；
- CentriMo enrichment p-value。

真正的直接识别 motif 通常在 peak summit 附近形成中心峰。只有总体富集而没有中心富集，可能反映间接结合、共因子 motif 或一般序列组成。

---

## 7. 必须加入的负对照

### 7.1 Column-shuffled PWMs

对每个 TFScope PWM 随机打乱列顺序 100 次，尽量保持 motif 长度、每列信息量和总体 GC/AT 偏好，但破坏位置顺序。

对每个随机 PWM 重复相同 enrichment 分析，得到 null distribution。

成功标准之一：

\[
E_{\text{TFScope}}
>
P_{95}(E_{\text{column-shuffled}})
\]

### 7.2 Family-matched motifs

从 JASPAR 或 HOCOMOCO 选取与目标 TF 同 family、长度和 GC 倾向接近的 motif。

目的不是要求 TFScope motif 一定超过所有同 family motif，而是区分：

- TF-specific signal；
- 仅仅学到了泛化的 family prior；
- 任何 TAAT-rich 或 GC-rich motif 都能产生的富集。

重点报告 TFScope motif 在该分布中的 percentile rank。

### 7.3 Cross-TF specificity matrix

用三个 motif 扫描三个 ChIP peak sets：

| motif \\ peaks | ADNP peaks | ZHX2 peaks | ZHX3 peaks |
|---|---:|---:|---:|
| ADNP motif |  |  |  |
| ZHX2 motif |  |  |  |
| ZHX3 motif |  |  |  |

每格填入 log2 enrichment、AUROC 和 summit-centered score。

理想情况是对角线更强，但由于 TF 间可能共享 motif，该分析作为增强证据，不作为硬性成功条件。

---

## 8. 预先定义的成功标准

### 单个 TF 的强支持

满足至少三项：

1. real peaks vs dinucleotide shuffle 的 \(\log_2 E>0\)，且经验 FDR < 0.05；
2. enrichment 超过 95% column-shuffled PWMs；
3. peak/background AUROC 的 95% CI 下界高于 0.5；
4. motif hits 在 summit 附近显著集中；
5. 在独立细胞系或数据集中复现。

### 最小项目成功

以下任一条件即可支持进入论文：

- 三个 TF 中至少两个获得强支持；
- 或一个 TF 获得很强的 enrichment、centrality 和独立数据复现；
- 即使部分 TF 阴性，只要分析设计和负对照完整，也可以诚实报告 motif 可验证性存在 TF-dependent heterogeneity。

不要把“所有 TF 都必须显著”设为项目成功条件。

---

## 9. 最小结果图设计

### Panel A：预测 motif logos

ADNP、ZHX2、ZHX3 三个 TFScope PWM。

### Panel B：composition-controlled enrichment

每个 TF 显示 real peaks vs dinucleotide shuffle 的 log2 enrichment、95% null interval 和 empirical FDR。

### Panel C：peak/background score distribution

每个 TF 显示 ChIP peaks、GC-matched genomic background、AUROC 和 95% CI。

### Panel D：summit-centered motif density

横轴：\(-250\) 至 \(+250\) bp；纵轴：motif hit density。

### Panel E：null PWM percentile

展示真实 TFScope PWM 在 100 个 column-shuffled PWM 中的 percentile。

### 可选 Panel F：cross-TF specificity heatmap

3 motifs × 3 peak sets。

---

## 10. 最小结果表

| TF | Dataset | Number of peaks | log2 enrichment vs shuffle | Empirical FDR | AUROC | AUPRC | Central enrichment | Null-PWM percentile |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ADNP | GSE105573 / ENCSR440VKE |  |  |  |  |  |  |  |
| ZHX2 | ENCSR407BEZ |  |  |  |  |  |  |  |
| ZHX3 | GSE170636 / ENCSR367KYL |  |  |  |  |  |  |  |

---

## 11. 推荐执行顺序

### Phase 1：真正的 MVP

1. 整理三个 TFScope PWMs；
2. 下载三个 processed ChIP peak 文件；
3. 统一到 hg38；
4. 取 summit ±250 bp；
5. 生成 dinucleotide shuffles；
6. FIMO 扫描；
7. 计算 log2 enrichment；
8. CentriMo 中心富集；
9. 生成 100 个 column-shuffled PWMs；
10. 输出一张 summary figure 和一张结果表。

### Phase 2：低成本增强

11. 构造 GC/长度匹配 genomic negatives；
12. 计算 AUROC/AUPRC；
13. 加入 family-matched motifs；
14. 做 3 × 3 cross-TF specificity matrix。

### Phase 3：有明确阳性信号后再扩展

15. 用 ZHX2 MCF-7 数据做独立复现；
16. 用 ZHX2 MDA-MB-231 knockdown RNA-seq 检查 motif-linked genes 是否响应扰动；
17. 用 mouse ADNP ChIP-seq 做跨物种辅助验证；
18. 再考虑 SOHLH1、ZGLP1 的 perturbation RNA-seq 分析。

---

## 12. 预计工作量

在不处理原始测序数据的前提下：

| 工作 | 预计时间 |
|---|---:|
| 下载并整理 peaks/PWMs | 0.5 天 |
| peak 标准化与序列提取 | 0.5 天 |
| shuffle/control 生成 | 0.5–1 天 |
| FIMO/CentriMo 扫描 | 0.5 天 |
| 统计与绘图 | 1–2 天 |
| QC、重复运行和方法整理 | 1 天 |

**MVP 总计约 3–5 个工作日。**

---

## 13. 主要风险及处理方式

### 风险 1：预测 motif 在 peaks 中不富集

可能原因：预测 motif 不准确、ChIP 主要反映间接结合、TF 通过共因子招募、标签/抗体/细胞系改变 occupancy，或 motif 只在特定子集 peaks 中有效。

处理：检查 summit-centered subset；按 promoter/enhancer 或 chromatin state 分层；与 de novo motif 比较；阴性结果如实报告。

### 风险 2：只看到泛 family motif 信号

处理：加入 family-matched motif controls；报告 percentile，而非只给最相似 motif；不声称 TF-specific discrimination。

### 风险 3：ADNP 数据为 tagged/engineered system

处理：在 Methods 和 Limitations 中明确；将 ADNP 视为 public occupancy support，而不是天然状态的最终证据；后续使用 mouse ESC ADNP 数据辅助验证。

### 风险 4：peak 文件没有 summit

处理：优先下载 narrowPeak/IDR 文件；没有 summit 时使用 midpoint；明确说明 centrality 结果会更保守。

---

## 14. 推荐的软件环境

```text
Python >= 3.10
bedtools
MEME Suite
numpy
pandas
scipy
scikit-learn
statsmodels
matplotlib
pyfaidx
ushuffle or fasta-dinucleotide-shuffle
```

可选：

```text
HOMER
gimmemotifs
pybedtools
```

为了可复现，建议提供 Conda environment 或 Docker/Singularity 配置。

---

## 15. 可直接用于论文 Methods 的简短模板

> To evaluate the biological plausibility of TFScope-predicted motifs for orphan transcription factors, we analyzed publicly available TF-specific ChIP-seq datasets for ADNP, ZHX2, and ZHX3. Conservative replicated peak sets were obtained from GEO/ENCODE and standardized to the GRCh38 assembly. Fixed 500-bp windows centered on peak summits were extracted, with peak midpoints used when summit coordinates were unavailable. Predicted PWMs were scanned on both strands using identical thresholds across real peaks, dinucleotide-shuffled peak sequences, and GC- and length-matched genomic background regions. We quantified composition-controlled motif enrichment, peak-versus-background discrimination, and positional enrichment relative to ChIP peak summits. Column-shuffled PWMs preserving motif length and per-column information content were used to construct empirical null distributions. These analyses were interpreted as public occupancy-based support for motif plausibility rather than definitive proof of direct TF–DNA binding.

---

## 16. 数据来源

- [ADNP ChIP-seq in K562 — GEO GSE105573](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE105573)
- [ADNP ENCODE experiment — ENCSR440VKE](https://www.encodeproject.org/experiments/ENCSR440VKE/)
- [ZHX2 ChIP-seq in HepG2 — ENCSR407BEZ](https://www.encodeproject.org/experiments/ENCSR407BEZ/)
- [ZHX2 ChIP-seq in MCF-7 — ENCSR876UYH](https://www.encodeproject.org/experiments/ENCSR876UYH/)
- [ZHX2 ChIP-seq and knockdown RNA-seq — GEO GSE175487](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175487)
- [ZHX3 ChIP-seq in HepG2 — GEO GSE170636](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE170636)
- [ZHX3 ENCODE experiment — ENCSR367KYL](https://www.encodeproject.org/experiments/ENCSR367KYL/)
- [MEME Suite](https://meme-suite.org/)
- [ENCODE blacklist](https://www.encodeproject.org/annotations/ENCSR636HFF/)

---

## 17. 最终决策建议

当前最值得实施的最小方案是：

> **使用 ADNP、ZHX2 和 ZHX3 的公开 processed ChIP-seq peaks，完成 composition-controlled enrichment、summit centrality 和 column-shuffled PWM null control。**

这一步不需要 GPU，不需要处理 FASTQ，也不需要重新训练 TFScope。它能以最低成本把现有的“motif 在泛 cCRE 中富集”提升为“motif 能解释对应 TF 的公开 occupancy 数据”。

只有在该 MVP 得到至少一个明确阳性结果后，才建议继续投入时间做跨细胞系、扰动转录组和 enhancer–gene linking。
