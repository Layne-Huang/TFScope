# TFScope Seed Model

Predicting TF-DNA binding specificity from protein amino acid sequence using ESM C encoder + Mixture-of-Experts conditioning on DBD family identity.

## Setup

### 1. Create environment

Create the project environment:

```bash
mamba env create -f environment.yml
mamba activate tfscope
```

This installs Python 3.10, CD-HIT, and the packages listed in
`requirements.txt`. PyTorch is intentionally not pinned because its build must
match the target machine.

Install the appropriate CUDA or CPU PyTorch build separately:

```bash
# Use the command generated for your machine:
# https://pytorch.org/get-started/locally/

# Optional, only for entmax/sparsemax attention experiments:
pip install entmax
```

### 2. Verify

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import esm; print('ESM OK')"
```

---

## Dataset

### Overview

Training data comes from three sources:

| Source | Role | Contents |
|--------|------|----------|
| **JASPAR 2024** | Primary training | ~2,000 curated, non-redundant PWM profiles |
| **CIS-BP** | Supplementary training | ~15,000+ PWMs (use experimental-only, exclude inferred) |
| **HOCOMOCO v11** | Independent evaluation | ~1,400 human/mouse motifs with quality scores |

### Directory structure

```
data/
├── raw/
│   ├── jaspar/           # Downloaded JASPAR profiles
│   ├── cisbp/            # Downloaded CIS-BP data
│   └── hocomoco/         # Downloaded HOCOMOCO motifs
├── processed/
│   ├── tf_pwm.parquet    # Merged PWM table (TF name, PWM matrix, family, source)
│   ├── tf_sequences.parquet  # UniProt sequences + DBD annotations
│   └── splits/           # Train/val/test split files
│       ├── lofo/         # Leave-one-family-out splits
│       ├── identity/     # 30% identity-based splits
│       └── random/       # Within-family random splits
└── embeddings/           # Cached ESM C embeddings (computed once)
```

### Download instructions

#### JASPAR 2024 (primary data)

```bash
mkdir -p data/raw/jaspar

# Download all vertebrate TF profiles via REST API
python scripts/download_jaspar.py --species vertebrates --outdir data/raw/jaspar

# Or manually via API:
# curl https://jaspar.elixir.no/api/v1/matrix/?tax_group=vertebrates&format=json > data/raw/jaspar/vertebrates.json
# curl https://jaspar.elixir.no/api/v1/matrix/?tax_group=vertebrates&format=meme > data/raw/jaspar/vertebrates.meme
```

This fetches:
- PWM matrices (4 × L position frequency matrices)
- TF names and UniProt IDs
- DBD family class labels

#### CIS-BP (supplementary)

```bash
mkdir -p data/raw/cisbp

# Download from http://cisbp.ccbr.utoronto.ca/
# Get the "Complete Set" for Homo sapiens

# IMPORTANT: when processing, filter to experimentally determined PWMs only:
python scripts/process_cisbp.py \
    --indir data/raw/cisbp \
    --outdir data/processed \
    --experimental-only
```

**Warning:** CIS-BP includes many homology-transferred (inferred) PWMs. These must be excluded from training to prevent data leakage. The `--experimental-only` flag filters to PWMs determined by direct assays (SELEX, PBM, HT-SELEX).

#### HOCOMOCO v11 (evaluation)

```bash
mkdir -p data/raw/hocomoco

# Download from https://hocomoco.org/
wget -P data/raw/hocomoco https://hocomoco.org/api/v1/demo/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme

# Filter to quality A/B motifs for evaluation:
python scripts/process_hocomoco.py \
    --indir data/raw/hocomoco \
    --outdir data/processed \
    --min-quality B
```

### Map TFs to UniProt sequences and DBD annotations

```bash
python scripts/map_tf_annotations.py \
    --pwm-dir data/processed \
    --outdir data/processed \
    --uniprot-sparql   # fetches sequences from UniProt
```

This step:
1. Maps TF names to UniProt accessions
2. Downloads full protein sequences from UniProt
3. Fetches DBD boundary annotations from InterPro/Pfam
4. Assigns DBD family labels (the 10 categories below)

### DBD family labels

| ID | Family | Expected TFs |
|----|--------|-------------|
| 0 | C2H2 Zinc Finger (1-3 fingers) | ~200 |
| 1 | C2H2 Zinc Finger (4-6 fingers) | ~250 |
| 2 | C2H2 Zinc Finger (7+ fingers) | ~250 |
| 3 | bHLH | ~110 |
| 4 | Homeodomain | ~200 |
| 5 | bZIP | ~60 |
| 6 | Nuclear Receptor | ~50 |
| 7 | Forkhead | ~45 |
| 8 | ETS | ~30 |
| 9 | Other (Sox/HMG, IRF, RFX, AP-2, etc.) | ~100 |

### Create train/test splits

```bash
python scripts/create_splits.py \
    --tf-data data/processed/tf_pwm.parquet \
    --outdir data/processed/splits \
    --method lofo      # leave-one-family-out
    --method identity  # 30% sequence identity clustering
    --method random    # within-family 80/10/10
```

**Data deduplication:** All TFs are clustered at 80% full-protein sequence identity. No two TFs above this threshold appear in different splits.

---

## Training

### Quick test with synthetic data (no dataset needed)

Verify the model architecture works before downloading real data:

```bash
python scripts/synthetic_test.py
```

This runs three tests:
1. **Forward pass shape check** — verifies output dimensions
2. **Loss + backward** — verifies gradients flow to all parameters
3. **Overfit test** — 200 steps on 10 random samples, loss should decrease

### Train with real data

```bash
python scripts/train.py \
    --config default \
    --data-dir data/processed \
    --split-dir data/processed/splits/identity \
    --output-dir outputs/run_001 \
    --total-steps 20000 \
    --batch-size 32 \
    --device cuda
```

Key training parameters (in `src/tfscope/config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | For new parameters (pooling, MOE, heads) |
| `warmup_steps` | 2000 | Linear warmup |
| `total_steps` | 20000 | Total training steps |
| `batch_size` | 32 | Per-GPU batch size |
| `num_experts` | 12 | MOE experts |
| `top_k` | 2 | Top-k routing |
| `balance_loss_weight` | 0.01 | Load balance auxiliary loss |
| `diversity_loss_weight` | 0.005 | Family diversity auxiliary loss |

### Monitoring

During training, log every 50 steps:
- Total loss, per-task losses (length, PWM)
- Learned uncertainty weights (sigma_length, sigma_pwm)
- Per-expert utilization (watch for expert collapse)
- Learning rate

### Multi-GPU (optional)

Not yet implemented. For single-GPU training, the model fits comfortably on an A100 (~6-8 GB with frozen encoder, ~12-16 GB with LoRA).

---

## Evaluation

### Run evaluation

```bash
python scripts/eval.py \
    --checkpoint outputs/run_001/checkpoint_best.pt \
    --data-dir data/processed \
    --split-dir data/processed/splits \
    --method lofo \
    --outdir results/run_001
```

### Metrics

| Metric | What it measures | Implementation |
|--------|-----------------|----------------|
| **Tomtom p-value** | PWM similarity against known motifs | MEME Suite `tomtom` |
| **Pearson correlation** | Per-position nucleotide preference correlation | `scipy.stats.pearsonr` |
| **KL divergence** | Position-wise probability distribution distance | `scipy.stats.entropy` |
| **Motif length accuracy** | Classification accuracy | `sklearn.metrics.accuracy_score` |
| **Per-family breakdown** | All metrics split by DBD family | custom |

### LOFO cross-validation (primary evaluation)

Leave-one-family-out is the headline result for the paper:

```bash
# Evaluate holding out each family in turn
for family in c2h2_short c2h2_med c2h2_long bhlh homeodomain bzip nr forkhead ets other; do
    python scripts/eval.py \
        --checkpoint outputs/lofo_${family}/checkpoint_best.pt \
        --split-dir data/processed/splits/lofo \
        --held-out-family $family \
        --outdir results/lofo_${family}
done

# Aggregate results
python scripts/aggregate_lofo.py --results-dir results --outdir results/lofo_summary
```

### Comparison baselines

To compare against DeepPBS and other methods:

```bash
python scripts/benchmark_baselines.py \
    --methods deeppbs homology_transfer nearest_neighbor \
    --data-dir data/processed \
    --split-dir data/processed/splits/identity \
    --outdir results/baselines
```

---

## Project structure

```
TFScope/
├── CLAUDE.md                    # Project instructions
├── requirements.txt             # Python dependencies
├── data/                        # Dataset (gitignored)
│   ├── raw/                     # Downloaded source files
│   ├── processed/               # Cleaned, merged tables
│   └── embeddings/              # Cached ESM C embeddings
├── doc/                         # Brainstorm documents
├── papers/                      # Reference literature
├── agents/                      # Specialist agent configs
├── skills/                      # Agent generator skill
├── src/tfscope/                 # Source code
│   ├── config.py                # Hyperparameters
│   ├── models/                  # Model components
│   ├── losses/                  # Loss functions
│   ├── data/                    # Dataset class
│   └── train/                   # Training loop
├── scripts/                     # Executable scripts
├── outputs/                     # Training outputs (gitignored)
├── results/                     # Evaluation results (gitignored)
└── figures/                     # Publication figures
```

## Reference

- **TFScope.pdf** — Full project proposal (3 pages)
- **doc/brainstorm-0-architecture-overview.md** — Architecture design doc
- **doc/brainstorm-1-tf-binding.md** — DBD family and data analysis
- **doc/brainstorm-2-encoder-design.md** — ESM C encoder specifications
- **doc/brainstorm-3-moe-design.md** — MOE architecture and training details
