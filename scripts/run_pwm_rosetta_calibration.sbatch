#!/bin/bash
#SBATCH -J pwm_rosetta_cal
#SBATCH -p sapphire,shared
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/aug_v7/slurm_%A_%a.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/aug_v7/slurm_%A_%a.err
#SBATCH --array=0-12

# pwm_rosetta calibration of the 130 Boltz-predicted protein-DNA complexes.
# 13 array tasks × 10 TFs each = 130 (last task gets the remainder).
# CPU-only — PyRosetta SNP scanning per complex takes ~30-40 min.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
set -eo pipefail

ROOT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/aug_v7
mkdir -p $ROOT

cd /n/home13/leihuang/project/TFScope

# Pick this task's slice of 10 af3_names from the manifest
SHARD=$SLURM_ARRAY_TASK_ID
NAMES=$(python3 -c "
import json
m = json.load(open('results/boltz_pipeline/manifest_all.json'))
names = [r['af3_name'] for r in m]      # 130 entries
shard, size = $SHARD, 10
chunk = names[shard*size:(shard+1)*size]
print(' '.join(chunk))
")
echo "Shard $SHARD will calibrate:"
for n in $NAMES; do echo "  $n"; done

python scripts/run_pwm_rosetta_calibration.py \
    --af3-out-dir /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/out_all/boltz_results_inputs_all/predictions \
    --manifest    /n/home13/leihuang/project/TFScope/results/boltz_pipeline/manifest_all.json \
    --out-dir     $ROOT \
    --only        $NAMES

echo "Shard $SHARD complete at $(date)"
