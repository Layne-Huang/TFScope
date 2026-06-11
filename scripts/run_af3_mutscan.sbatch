#!/bin/bash
#SBATCH -J af3_mutscan
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -t 02:30:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/mutscan_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/mutscan_%j.err

# AF3-per-mutation PWM workflow: fold every single-base DNA mutant of a ZF with AF3
# (MSA reused from the WT data.json => pure inference, --norun_data_pipeline). The CIFs
# are later scored by Rosetta interface ΔΔG vs WT -> Boltzmann PWM.
# Usage: sbatch scripts/run_af3_mutscan.sbatch zf21   [or zf93/zf92/zf129]
set -eo pipefail
ZF=${1:?give a ZF name, e.g. zf21}
source /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3tools_venv/bin/activate
cd /n/home13/leihuang/project/TFScope
IN=results/af3_mutscan/$ZF/inputs
OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/af3_runs/mutscan/$ZF
mkdir -p "$OUT"
echo "Job $SLURM_JOB_ID on $SLURM_NODENAME | ZF=$ZF | $(ls $IN/*.json | wc -l) folds"; nvidia-smi || true

python - "$IN" "$OUT" <<'PY'
import sys, glob, os, time
sys.path.insert(0, "/n/holylabs/lpinello_lab/Lab/leihuang/af3_mmseqs_scripts/af3_mmseqs2")
from alphafold3 import run_alphafold3

IN, OUT = sys.argv[1], sys.argv[2]
MODEL = "/n/holylabs/lpinello_lab/Lab/leihuang/AF3_parameter"
DB    = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/alphafold3/public_databases"
jsons = sorted(glob.glob(os.path.join(IN, "*.json")))
for i, j in enumerate(jsons, 1):
    name = os.path.basename(j)[:-5]
    done = os.path.join(OUT, name, f"{name}_model.cif")
    if os.path.exists(done):
        print(f"[{i}/{len(jsons)}] {name} already done", flush=True); continue
    t0 = time.time()
    print(f"[{i}/{len(jsons)}] folding {name} ...", flush=True)
    try:
        run_alphafold3(j, OUT, MODEL, DB)
        print(f"    done in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"    FAILED {name}: {e}", flush=True)
PY
echo "Done at $(date)"
