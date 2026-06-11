# TFScope Codex Instructions

Before planning or modifying this repository:

1. Read `docs/MACHINE_HANDOFF.md`.
2. Read `docs/TFSCOPE_V19_IMPROVEMENT_PLAN.md`.
3. Read the relevant sections of `docs/ARCHITECTURE_AND_RESULTS.md`.
4. Preserve the historical split and retrieval artifacts. New V19 comparisons
   must use:
   - `data/processed/splits/cluster40_clean/split.json`
   - `data/processed/tf_nn_index_cluster40_clean.json`
5. Keep validation and test retrieval train-only.
6. Run the leakage audit and focused tests after changing data, splits, or
   retrieval:

```bash
PYTHONPATH=src:scripts python -m unittest discover -s tests -v

python scripts/audit_split_hygiene.py \
  --data data/processed/tf_pwm_aug_dbd_canon_trim.parquet \
  --split data/processed/splits/cluster40_clean/split.json \
  --index data/processed/tf_nn_index_cluster40_clean.json \
  --fail-on-leakage
```

Current next step: run and evaluate E1 clean no-RAG/RAG baselines before
implementing registration or larger architecture changes.

PyTorch is intentionally excluded from `environment.yml` and
`requirements.txt`; install the build appropriate for the current machine.
