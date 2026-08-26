#!/usr/bin/env bash
# Launch a wave of LOFO trainings, one per free GPU, fully detached.
#
#   scripts/lofo/launch_lofo_wave.sh C2H2 Homeodomain bHLH
#   DRY_RUN=1 scripts/lofo/launch_lofo_wave.sh C2H2            # print, do not launch
#
# GPU selection: cards with < 2 GiB used, EXCLUDING nvidia-smi index 0 (dead on this
# node). Cards are pinned by UUID, never by index: CUDA does not enumerate the dead
# card at all, so the CUDA index is nvidia-smi index MINUS ONE. Passing an index makes
# every job land one card off the one you asked for, and `--gpu 9` silently resolves to
# a nonexistent 11th device -> "Device: cpu" -> "bf16 requires CUDA". UUIDs are exact.
# Refuses to launch if there are fewer free GPUs than families, rather than
# oversubscribing a card.
#
# Progress:  scripts/lofo/lofo_status.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"

FAMS=("$@")
[[ ${#FAMS[@]} -gt 0 ]] || { echo "usage: $0 <FAMILY_TAG>..." >&2; exit 2; }

EXCLUDE="${EXCLUDE_GPUS:-0}"
LIMIT="${MEM_LIMIT_MIB:-2000}"
mapfile -t FREE < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
  awk -F', ' -v ex="$EXCLUDE" -v lim="$LIMIT" '
    BEGIN{n=split(ex,e,/[ ,]+/); for(i=1;i<=n;i++) bad[e[i]]=1}
    !($1 in bad) && $3+0 < lim {print $1"|"$2}')

echo "free GPUs (excluding ${EXCLUDE}): ${FREE[*]:-<none>}"
if [[ ${#FREE[@]} -lt ${#FAMS[@]} ]]; then
  echo "ABORT: ${#FAMS[@]} families requested but only ${#FREE[@]} free GPU(s)." >&2
  echo "       Launch a smaller wave, or wait for a card to free up." >&2
  exit 3
fi

echo "=== syncing /data1 mirror (jobs must never read AFS) ==="
scripts/lofo/sync_lofo_to_data1.sh | tail -3

for i in "${!FAMS[@]}"; do
  fam="${FAMS[$i]}"
  idx="${FREE[$i]%%|*}"; uuid="${FREE[$i]#*|}"
  [[ -f "data/processed/splits/lofo_v24/${fam}.json" ]] || {
    echo "ABORT: no split for '${fam}'" >&2; exit 4; }
  job="lofo_v24_${fam}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "would launch: $job on nvidia-smi GPU $idx ($uuid)"; continue
  fi
  echo "--> $fam on nvidia-smi GPU $idx"
  REPO_MIRROR=/data1/leihuang/TFScope_store/lofo_v24/run \
    scripts/v26/run_detached.sh --mirror --gpu "$uuid" "$job" \
    bash scripts/lofo/train_lofo_v24.sh "$fam"
done

echo
echo "watch with: scripts/lofo/lofo_status.sh"
