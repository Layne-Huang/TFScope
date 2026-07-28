---
name: feedback-conventions
description: Repository conventions and hard rules to follow in TFScope development
metadata:
  type: feedback
---

## Hard guardrails

Never tune the frozen family composition policy on test data. It was selected on
validation and must remain locked.

**Why:** test leakage would invalidate the publication claim.

**How to apply:** The policy file `results/v19_e9_model_composition/validation_composition_grid.json`
is read-only. Only read it for reporting.

---

Do not launch seeds 43/44 without explicit new user instruction.

**Why:** User explicitly fixed the scope to seed 42 for all V19 experiments.

---

All V19 comparisons must use the clean split and train-only donor policy. Never
compare against the historical cluster40 artifacts as if they represent V19 gains.

**Why:** The old split had cross-split gene leakage and same-gene retrieval.

---

PyTorch is intentionally excluded from `environment.yml`. Install the
CUDA/CPU-appropriate build manually. Do not add it to the yml.

---

Use `mamba`, not `conda`.

---

Big files (checkpoints, large results) → `/n/holylabs/lpinello_lab/Lab/leihuang/TFScope`.
Cache → `/n/holylabs/lpinello_lab/Lab/leihuang/.cache`.

---

Always check the AGENTS.md at the start of any new session — it points to
MACHINE_HANDOFF.md and the V19 improvement plan which are the authoritative
state docs.

Links: [[project-state-v19]]
