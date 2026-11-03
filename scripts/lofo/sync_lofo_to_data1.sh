#!/usr/bin/env bash
# Mirror everything a LOFO job needs onto /data1, so an 18-hour training run never
# touches AFS.
#
# AFS tokens on this host drop every ~10 minutes; a long detached job that reads code
# or input parquets from AFS dies partway with a confusing `Permission denied`.
# Re-run this after ANY edit to scripts/lofo/, scripts/train.py or src/tfscope/, then
# launch with scripts/lofo/launch_lofo_wave.sh (which passes --mirror).
#
#   scripts/lofo/sync_lofo_to_data1.sh
set -euo pipefail

REPO=/afs/csail.mit.edu/u/l/leihuang/project/TFScope
STORE=/data1/leihuang/TFScope_store/lofo_v24
MIRROR=$STORE/run

mkdir -p "$MIRROR"/{scripts,src,data/processed/splits,data/contact_maps,results}

echo "=== code ==="
# scripts/v26 is needed too: run_detached.sh is the launcher the queue scheduler calls,
# and the scheduler itself runs from inside this mirror for days at a time.
for d in src/tfscope scripts/lofo scripts/v26 scripts/case_study iclr; do
  mkdir -p "$MIRROR/$(dirname "$d")"
  rsync -a --delete "$REPO/$d/" "$MIRROR/$d/"
  echo "  synced $d ($(find "$MIRROR/$d" -type f | wc -l) files)"
done

# Every top-level analysis/figure script, not a hand-kept list. Enumerating them one
# by one has now failed three times: a new script is written, the job is launched
# --mirror, and dies with "No such file". 44 files / 450 KB, so just take them all.
echo "=== top-level scripts (all of scripts/*.py) ==="
mkdir -p "$MIRROR/scripts"
rsync -a "$REPO"/scripts/*.py "$MIRROR/scripts/"
echo "  $(ls "$MIRROR"/scripts/*.py | wc -l) files"

echo "=== single-file inputs (read-only; never written by a LOFO job) ==="
INPUTS=(
  data/processed/tf_pwm_training_v23.parquet
  data/processed/tf_pwm_deeppbs_only_canon_trim.parquet
  data/contact_maps/contact_targets_v23.json
  data/contact_maps/recognition_residues_v23.json
  data/processed/splits/train_v22/split.json
  data/processed/splits/deeppbs_cluster40/split.json
)
for f in "${INPUTS[@]}"; do
  if [ -f "$REPO/$f" ]; then
    mkdir -p "$MIRROR/$(dirname "$f")"
    rsync -a "$REPO/$f" "$MIRROR/$f"
    echo "  $f  $(stat -c %s "$MIRROR/$f") bytes"
  else
    echo "  MISSING (skipped): $f"; exit 21
  fi
done

echo "=== LOFO splits ==="
rsync -a --delete "$REPO/data/processed/splits/lofo_v24/" "$MIRROR/data/processed/splits/lofo_v24/"
echo "  $(ls "$MIRROR/data/processed/splits/lofo_v24" | wc -l) split files"

echo "=== writable outputs: symlink to their /data1 homes ==="
link() {  # $1 = repo-relative path, $2 = /data1 target
  mkdir -p "$2"; rm -rf "$MIRROR/$1"; mkdir -p "$(dirname "$MIRROR/$1")"; ln -s "$2" "$MIRROR/$1"
  echo "  $1 -> $2"
}
link checkpoints           "$STORE/checkpoints"
link results/family_lofo   "$STORE/results_family_lofo"
link results/moe_expert_interpretation "$STORE/results_moe_expert_interpretation"
link figures_v24_ensemble  "$STORE/figures_v24_ensemble"
link figures_v24           "$STORE/figures_v24"

echo "=== verify: mirror resolves with NO AFS access ==="
for p in scripts/train.py scripts/lofo/build_lofo_splits_v24.py \
         scripts/lofo/run_lofo_queue.sh scripts/v26/run_detached.sh \
         scripts/analyze_v24_ens_expert_residues.py \
         data/processed/tf_pwm_training_v23.parquet \
         data/contact_maps/contact_targets_v23.json \
         data/processed/splits/lofo_v24/_manifest.json \
         data/processed/splits/lofo_v24/C2H2.json \
         src/tfscope/models/tfscope.py iclr/unified_eval.py; do
  [ -r "$MIRROR/$p" ] || { echo "ABORT: unreadable in mirror: $p"; exit 20; }
  echo "  ok  $p"
done
echo "mirror ready: $MIRROR"
