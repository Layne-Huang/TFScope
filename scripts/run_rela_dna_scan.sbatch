#!/bin/bash
#SBATCH -J rela_dna_scan
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120g
#SBATCH -t 06:00:00
#SBATCH -o /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/rela_dna_scan/slurm_%j.out
#SBATCH -e /n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/rela_dna_scan/slurm_%j.err

# Pipeline: 37 independent Boltz-2 folds (WT + 36 DNA mutants) → PyRosetta interface ΔΔG → PWM
# MSA is reused from cached rela_msa_0.csv — no ColabFold server calls needed.

export PATH="/n/home13/leihuang/.conda/envs/tfscope/bin:$PATH"
set -eo pipefail

PROJ=/n/home13/leihuang/project/TFScope
BOLTZ_OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/boltz_runs/rela_dna_scan
PYROS_OUT=/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/pwm_rosetta_runs/rela_dna_scan
CACHE=/n/holylabs/lpinello_lab/Lab/leihuang/.cache

mkdir -p "$BOLTZ_OUT" "$PYROS_OUT"
cd "$PROJ"

# ── STEP 1: Boltz-2 fold all 37 FASTAs (reusing cached MSA, no server) ──────
echo "[$(date)] Starting Boltz-2 folds..."
boltz predict \
    results/boltz_rela_dna_scan/inputs \
    --out_dir        "$BOLTZ_OUT" \
    --cache          "$CACHE" \
    --output_format  mmcif \
    --no_kernels \
    --recycling_steps 3 \
    --sampling_steps  200

echo "[$(date)] Boltz-2 done. CIFs:"
find "$BOLTZ_OUT" -name "*.cif" | sort

# ── STEP 2: Build manifest for PyRosetta ─────────────────────────────────────
echo "[$(date)] Building PyRosetta manifest..."
python3 - <<'PYEOF'
import json, os, glob

manifest_in  = "results/boltz_rela_dna_scan/mutation_manifest.json"
boltz_out    = os.environ["BOLTZ_OUT"]
pyros_out    = os.environ["PYROS_OUT"]
core_offset  = 1
core         = "GGAAATTCCC"

muts = json.load(open(manifest_in))
records = []
for m in muts:
    name     = m["name"]
    pred_dir = os.path.join(boltz_out, f"boltz_results_inputs/{predictions}/{name}")
    # Boltz output path: {out_dir}/boltz_results_{input_dir_name}/predictions/{name}/{name}_model_0.cif
    cif_pats = glob.glob(f"{boltz_out}/**/predictions/{name}/{name}_model_0.cif", recursive=True)
    if not cif_pats:
        print(f"  [SKIP] no CIF for {name}")
        continue
    cif = cif_pats[0]
    records.append({
        "af3_name":    name,
        "filename":    "1ram_B_RELA.TF65_MOUSE.H11MO.0.A.txt",
        "fwd_dna":     m["fwd_dna"],
        "core":        core,
        "core_offset": core_offset,
        "pred_len":    len(core),
        "true_len":    len(core),
        "cif_path":    cif,
    })

out_manifest = os.path.join(pyros_out, "pyros_manifest.json")
with open(out_manifest, "w") as f:
    json.dump(records, f, indent=2)
print(f"Manifest: {len(records)} entries → {out_manifest}")
PYEOF

# ── STEP 3: PyRosetta interface ΔΔG for all 37 complexes ────────────────────
echo "[$(date)] Starting PyRosetta scans..."
python3 scripts/run_pyrosetta_interface_ddg.py \
    --manifest   "$PYROS_OUT/pyros_manifest.json" \
    --out-dir    "$PYROS_OUT" \
    --wt-name    rela_WT

echo "[$(date)] All done."
