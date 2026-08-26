#!/usr/bin/env bash
# Mirror everything a v26 detached job needs onto /data1, so jobs never touch AFS.
#
# AFS tokens on this machine drop roughly every 10 minutes (observed 2026-08-14: valid 09:46,
# gone 09:58, despite a stated expiry of 21:44). A multi-hour Phase-5 training run reading code
# or input parquets from AFS will die partway with a confusing `Permission denied`. So:
#
#   code + read-only legacy inputs -> copied to $MIRROR
#   v26 data/results/logs          -> already on /data1, symlinked into $MIRROR
#
# Run this after any edit to scripts/v26/ or src/tfscope/, then launch jobs with
#   scripts/v26/run_detached.sh --mirror <job> <cmd...>
#
#   scripts/v26/sync_to_data1.sh
set -euo pipefail

REPO=/afs/csail.mit.edu/u/l/leihuang/project/TFScope
STORE=/data1/leihuang/TFScope_store/v26
MIRROR=$STORE/run

mkdir -p "$MIRROR"/{scripts,src,tests,data/processed,data/contact_maps,data/processed/splits,results}

echo "=== code ==="
for d in scripts/v26 src/tfscope tests/v26 configs/v26 iclr; do
  mkdir -p "$MIRROR/$(dirname "$d")"
  rsync -a --delete "$REPO/$d/" "$MIRROR/$d/"
  echo "  synced $d ($(find "$MIRROR/$d" -type f | wc -l) files)"
done

echo "=== read-only legacy inputs (copied; never written by v26) ==="
INPUTS=(
  data/processed/tf_pwm_training_v23.parquet
  data/processed/tf_pwm_training_v22.parquet
  data/processed/tf_pwm.parquet
  data/processed/tf_pwm_deeppbs_v2_deduped.parquet
  data/processed/tf_pwm_training_v25flank.parquet
  data/processed/tf_pwm_training_v25xtal.parquet
  data/contact_maps/contact_targets_v23.json
  data/contact_maps/recognition_residues_v23.json
  data/processed/splits/train_v22/split.json
  data/processed/splits/train_v22/assignments.parquet
  scripts/train.py
  results/mutation_benchmark/barrera_pairs.json
  results/mutation_benchmark/heldout_genes.json
)
for f in "${INPUTS[@]}"; do
  if [ -f "$REPO/$f" ]; then
    mkdir -p "$MIRROR/$(dirname "$f")"
    rsync -a "$REPO/$f" "$MIRROR/$f"
    echo "  $f  $(stat -c %s "$MIRROR/$f") bytes"
  else
    echo "  MISSING (skipped): $f"
  fi
done

echo "=== v26 data/results: symlink to their /data1 homes ==="
link() {  # $1 = repo-relative path, $2 = /data1 target
  rm -rf "$MIRROR/$1"; mkdir -p "$(dirname "$MIRROR/$1")"; ln -s "$2" "$MIRROR/$1"
  echo "  $1 -> $2"
}
link data/processed/v26   "$STORE/data_processed_v26"
link data/annotations_v26 "$STORE/data_annotations_v26"
link data/contacts_v26    "$STORE/data_contacts_v26"
link data/processed/splits/v26 "$STORE/data_processed_splits_v26"
link results/v26          "$STORE/results_v26"
link results/v26_audit    "$STORE/results_v26_audit"
# The mmCIF cache is already on /data1; symlink it so Phase-2 contact parsing needs no AFS.
link data/raw/pdb_cif_cache "/data1/leihuang/TFScope_store/data/raw/pdb_cif_cache"

echo "=== verify: mirror is readable with NO AFS access ==="
# Coverage check: every path a v26 job reads must resolve inside the mirror. Three separate
# failures came from forgetting a directory here (pdb_cif_cache, results/mutation_benchmark,
# configs/v26), so the list is explicit and the sync fails loudly if one is missing.
for p in scripts/v26/build_canonical_targets.py data/processed/tf_pwm_training_v23.parquet \
         data/processed/v26/dbd_candidates.parquet data/raw/pdb_cif_cache/10eh.cif \
         configs/v26/core.yaml results/mutation_benchmark/barrera_pairs.json \
         data/processed/splits/v26/manifest.parquet tests/v26/test_model_invariants.py \
         src/tfscope/v26/model.py iclr/score_checkpoint_unified.py scripts/train.py; do
  [ -r "$MIRROR/$p" ] || { echo "ABORT: unreadable in mirror: $p"; exit 20; }
  echo "  ok  $p"
done
echo "mirror ready: $MIRROR"
