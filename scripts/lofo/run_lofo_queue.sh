#!/usr/bin/env bash
# Drain a queue of LOFO families onto free GPUs, one at a time, unattended.
#
# The 12 LOFO runs take 7-17 h each and only 3 cards are usable, so the wave has to be
# refilled roughly every half day. This scheduler does that: it polls for a free GPU and
# launches the next queued family, until the queue is empty.
#
#   scripts/lofo/run_lofo_queue.sh bZIP Nuclear_Receptor HMG-SOX IRF T-box ...
#   DRY_RUN=1 scripts/lofo/run_lofo_queue.sh bZIP            # print, do not launch
#
# Launch the scheduler ITSELF detached, so it outlives the shell that started it:
#   scripts/v26/run_detached.sh lofo_v24_queue bash scripts/lofo/run_lofo_queue.sh <fams...>
#
# Safety:
#   * never starts a family whose job is already RUNNING or already has a ckpt_best.pt
#   * one launch per poll, then a settle wait, so a card is never double-booked while
#     the freshly launched job is still importing torch and has claimed no memory yet
#   * pins by UUID (CUDA index != nvidia-smi index on this node) and skips the dead GPU 0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"

QUEUE=("$@")
[[ ${#QUEUE[@]} -gt 0 ]] || { echo "usage: $0 <FAMILY_TAG>..." >&2; exit 2; }

POLL="${POLL_SECONDS:-300}"          # how often to look for a free card
SETTLE="${SETTLE_SECONDS:-420}"      # wait after a launch before polling again
EXCLUDE="${EXCLUDE_GPUS:-0}"
LIMIT="${MEM_LIMIT_MIB:-2000}"
LOGROOT="/data1/leihuang/TFScope_store/v26_logs"
CKROOT="/data1/leihuang/TFScope_store/lofo_v24/checkpoints/lofo_v24"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

free_uuid() {
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
  awk -F', ' -v ex="$EXCLUDE" -v lim="$LIMIT" '
    BEGIN{n=split(ex,e,/[ ,]+/); for(i=1;i<=n;i++) bad[e[i]]=1}
    !($1 in bad) && $3+0 < lim {print $1"|"$2; exit}'
}

already_running() {   # $1 = family tag
  local d="$LOGROOT/lofo_v24_$1"
  [[ -f "$d/STATUS" && "$(cat "$d/STATUS")" == "RUNNING" ]] &&
    kill -0 "$(cat "$d/pid" 2>/dev/null)" 2>/dev/null
}

echo "[$(stamp)] queue (${#QUEUE[@]}): ${QUEUE[*]}"
echo "[$(stamp)] poll=${POLL}s settle=${SETTLE}s exclude_gpus=${EXCLUDE}"

i=0
while (( i < ${#QUEUE[@]} )); do
  fam="${QUEUE[$i]}"

  if [[ ! -f "data/processed/splits/lofo_v24/${fam}.json" ]]; then
    echo "[$(stamp)] SKIP ${fam}: no split file"; ((i++)); continue
  fi
  if [[ -f "$CKROOT/${fam}/ckpt_best.pt" ]]; then
    echo "[$(stamp)] SKIP ${fam}: ckpt_best.pt already exists"; ((i++)); continue
  fi
  if already_running "$fam"; then
    echo "[$(stamp)] SKIP ${fam}: already RUNNING"; ((i++)); continue
  fi

  slot="$(free_uuid)"
  if [[ -z "$slot" ]]; then
    sleep "$POLL"; continue
  fi
  idx="${slot%%|*}"; uuid="${slot#*|}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[$(stamp)] would launch ${fam} on nvidia-smi GPU ${idx}"; ((i++)); continue
  fi

  echo "[$(stamp)] launching ${fam} on nvidia-smi GPU ${idx} (${uuid})"
  # Re-sync the mirror before each launch so a mid-queue code edit is picked up.
  scripts/lofo/sync_lofo_to_data1.sh >/dev/null 2>&1 || echo "[$(stamp)] WARN: mirror sync failed"
  if REPO_MIRROR=/data1/leihuang/TFScope_store/lofo_v24/run \
       scripts/v26/run_detached.sh --mirror --gpu "$uuid" "lofo_v24_${fam}" \
       bash scripts/lofo/train_lofo_v24.sh "$fam"; then
    ((i++))
    echo "[$(stamp)] settling ${SETTLE}s before next poll"
    sleep "$SETTLE"
  else
    echo "[$(stamp)] launch of ${fam} FAILED; retrying after ${POLL}s"
    sleep "$POLL"
  fi
done

echo "[$(stamp)] queue drained; ${#QUEUE[@]} families dispatched"
echo "[$(stamp)] NOTE: dispatched != finished -- check scripts/lofo/lofo_status.sh"
