#!/bin/bash
set -eo pipefail

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"

PROJ=/n/home13/leihuang/project/TFScope
BOLTZ_OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/rela_dna_scan
PYROS_OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/rela_dna_scan
CACHE=/n/holylabs/lpinello_lab/Lab/leihuang/.cache

cd "$PROJ"
mkdir -p "$BOLTZ_OUT" "$PYROS_OUT"

# ── Step 1: Generate 37 FASTAs (WT + 36 DNA mutants, reusing cached MSA) ────
echo "[$(date)] Generating FASTA inputs..."
python scripts/build_rela_dna_mutant_inputs.py

# ── Step 2: Boltz-2 fold all 37 complexes ───────────────────────────────────
echo "[$(date)] Running Boltz-2 (37 complexes, no MSA server)..."
boltz predict \
    results/boltz_rela_dna_scan/inputs \
    --out_dir        "$BOLTZ_OUT" \
    --cache          "$CACHE" \
    --output_format  mmcif \
    --no_kernels

echo "[$(date)] Boltz-2 done. CIFs found:"
find "$BOLTZ_OUT" -name "*.cif" | sort

# ── Step 3: PyRosetta interface ΔΔG → PWM ───────────────────────────────────
echo "[$(date)] Running PyRosetta interface ΔΔG..."
python scripts/run_pyrosetta_interface_ddg.py \
    --manifest  results/boltz_rela_dna_scan/mutation_manifest.json \
    --boltz-out "$BOLTZ_OUT" \
    --out-dir   "$PYROS_OUT" \
    --wt-name   rela_WT

echo "[$(date)] All done. Results in $PYROS_OUT"
