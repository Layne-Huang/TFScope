#!/usr/bin/env bash
# Mirror everything a v24-ablation job needs onto /data1, so a 15-hour training run
# never touches AFS.
#
# AFS tokens on this host drop every ~10 minutes; a long detached job that reads code
# or input parquets from AFS dies partway with a confusing `Permission denied`.
# Re-run this after ANY edit to scripts/ablation_v24/, scripts/train.py or src/tfscope/,
# then launch with scripts/ablation_v24/run_ablation_queue.sh (which passes --mirror).
#
#   scripts/ablation_v24/sync_ablation_to_data1.sh
set -euo pipefail

REPO=/afs/csail.mit.edu/u/l/leihuang/project/TFScope
STORE=/data1/leihuang/TFScope_store/ablation_v24
MIRROR=$STORE/run

mkdir -p "$MIRROR"/{scripts,src,data/processed/splits,data/contact_maps,results}

echo "=== code ==="
# scripts/v26 is needed too: run_detached.sh is the launcher the queue scheduler calls,
# and the scheduler itself runs from inside this mirror for days at a time.
for d in src/tfscope scripts/ablation_v24 scripts/v26 iclr; do
  mkdir -p "$MIRROR/$(dirname "$d")"
  rsync -a --delete "$REPO/$d/" "$MIRROR/$d/"
  echo "  synced $d ($(find "$MIRROR/$d" -type f | wc -l) files)"
done

echo "=== single-file inputs (read-only; never written by an ablation job) ==="
INPUTS=(
  scripts/train.py
  data/processed/tf_pwm_training_v23.parquet
  data/processed/tf_pwm_deeppbs_fold0.parquet
  data/contact_maps/contact_targets_deeppbsfold0.json
  data/contact_maps/recognition_residues_deeppbsfold0.json
  data/contact_maps/contact_targets_v23.json
  data/contact_maps/recognition_residues_v23.json
  data/contact_maps/contact_targets_v23_shuffled.json
  data/contact_maps/recognition_residues_v23_shuffled.json
  data/processed/splits/train_v22/split.json
)
for f in "${INPUTS[@]}"; do
  if [ -f "$REPO/$f" ]; then
    mkdir -p "$MIRROR/$(dirname "$f")"
    rsync -a "$REPO/$f" "$MIRROR/$f"
    echo "  $f  $(stat -c %s "$MIRROR/$f") bytes"
  else
    echo "  MISSING (skipped): $f  -- run scripts/ablation_v24/build_ablation_assets.py"; exit 21
  fi
done

echo "=== ablation splits ==="
rsync -a --delete "$REPO/data/processed/splits/ablation_v24/" "$MIRROR/data/processed/splits/ablation_v24/"
echo "  $(ls "$MIRROR/data/processed/splits/ablation_v24" | wc -l) split files"

echo "=== writable outputs: symlink to their /data1 homes ==="
link() {  # $1 = repo-relative path, $2 = /data1 target
  mkdir -p "$2"; rm -rf "$MIRROR/$1"; mkdir -p "$(dirname "$MIRROR/$1")"; ln -s "$2" "$MIRROR/$1"
  echo "  $1 -> $2"
}
link checkpoints              "$STORE/checkpoints"
link results/ablation_v24     "$STORE/results_ablation_v24"

echo "=== verify: mirror resolves with NO AFS access ==="
for p in scripts/train.py scripts/ablation_v24/train_ablation_v24.sh \
         scripts/ablation_v24/run_ablation_queue.sh scripts/v26/run_detached.sh \
         data/processed/tf_pwm_training_v23.parquet \
         data/contact_maps/contact_targets_v23_shuffled.json \
         data/processed/splits/ablation_v24/structonly.json \
         data/processed/tf_pwm_deeppbs_fold0.parquet \
         data/processed/splits/ablation_v24/deeppbs_fold0.json \
         data/contact_maps/contact_targets_deeppbsfold0.json \
         data/contact_maps/recognition_residues_deeppbsfold0.json \
         data/processed/splits/train_v22/split.json \
         src/tfscope/models/tfscope.py iclr/unified_eval.py; do
  [ -r "$MIRROR/$p" ] || { echo "ABORT: unreadable in mirror: $p"; exit 20; }
  echo "  ok  $p"
done
echo "mirror ready: $MIRROR"
