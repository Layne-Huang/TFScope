#!/bin/bash
#SBATCH -J af3_dpbs_seed
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 00:40:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/dpbs_seed_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/dpbs_seed_%j.err

# Fold protein + DeepPBS-predicted-consensus DNA with AF3 (MSA reused from WT data.json),
# so the deployable Rosetta single-mutation scan can be seeded from DeepPBS's prediction
# (parallel to the TFScope-RAG-seeded Rosetta-on-AF3).
set -eo pipefail
source /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3tools_venv/bin/activate
cd /n/home13/leihuang/project/TFScope
IN=results/af3_deeppbs_seed/inputs
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/deeppbs_seed
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME | $(ls $IN/*.json | wc -l) folds"; nvidia-smi || true

python - "$IN" "$OUT" <<'PY'
import sys, glob, os, time
sys.path.insert(0, "/n/holylabs/lpinello_lab/Lab/leihuang/af3_mmseqs_scripts/af3_mmseqs2")
from alphafold3 import run_alphafold3
IN, OUT = sys.argv[1], sys.argv[2]
MODEL = "/n/holylabs/lpinello_lab/Lab/leihuang/AF3_parameter"
DB    = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/alphafold3/public_databases"
for i, j in enumerate(sorted(glob.glob(os.path.join(IN, "*.json"))), 1):
    name = os.path.basename(j)[:-5]
    if os.path.exists(os.path.join(OUT, name, f"{name}_model.cif")):
        print(f"[{i}] {name} done", flush=True); continue
    t0 = time.time(); print(f"[{i}] folding {name} ...", flush=True)
    try:
        run_alphafold3(j, OUT, MODEL, DB); print(f"    {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"    FAILED {name}: {e}", flush=True)
PY
echo "Done at $(date)"
