#!/usr/bin/env bash
# Mirror everything the v25 runs read onto /data1.
#
# AFS tokens on this host expire every ~10 min and a detached job cannot renew them:
# directory listings keep working while file reads start failing, so a job launched
# from AFS dies hours in with a confusing "No such file". Long runs therefore execute
# from this mirror and touch AFS not at all.
#
# Re-run after ANY edit to scripts/ or src/ -- the mirror is a copy, not a link.
set -euo pipefail
SRC="/afs/csail.mit.edu/u/l/leihuang/project/TFScope"
DST="/data1/leihuang/TFScope_store/v25/run"

mkdir -p "$DST"/{scripts,src,data/processed/splits,data/contact_maps,iclr}
rsync -a --delete "$SRC/src/" "$DST/src/"
rsync -a --delete "$SRC/scripts/" "$DST/scripts/"
rsync -a "$SRC/iclr/" "$DST/iclr/"
rsync -a "$SRC/data/processed/tf_pwm_unified_v25.parquet" "$DST/data/processed/"
rsync -a "$SRC/data/processed/tf_pwm_training_v25.parquet" "$DST/data/processed/"
rsync -a --delete "$SRC/data/processed/splits/unified_v25/" \
                  "$DST/data/processed/splits/unified_v25/"
rsync -a "$SRC"/data/contact_maps/*unified_v25.json "$DST/data/contact_maps/"

echo "mirror -> $DST"
du -sh "$DST" | sed 's/^/  /'
ls "$DST/data/processed/splits/unified_v25/" | sed 's/^/  split: /'
