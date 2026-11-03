#!/usr/bin/env bash
# Drain a queue of v24 ablation runs onto free GPUs, one at a time, unattended.
#
# Each run is 3-15 h on one card and only a couple of cards are usually free, so the
# queue has to be refilled for days. This scheduler does that: it polls for a free GPU
# and launches the next queued <TAG>:<SEED>, until the queue is empty.
#
#   scripts/ablation_v24/run_ablation_queue.sh A1_structonly:42 A2_nodistill:42 ...
#   DRY_RUN=1 scripts/ablation_v24/run_ablation_queue.sh A1_structonly:42
#
# Launch the scheduler ITSELF detached, so it outlives the shell that started it:
#   scripts/v26/run_detached.sh abl_v24_queue bash scripts/ablation_v24/run_ablation_queue.sh <jobs...>
#
# Safety:
#   * never starts a job that is already RUNNING or already has a ckpt_best.pt
#   * one launch per poll, then a settle wait, so a card is never double-booked while
#     the freshly launched job is still importing torch and has claimed no memory yet
#   * pins by UUID (CUDA index != nvidia-smi index on this node) and skips the dead GPU 0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"

QUEUE=("$@")
[[ ${#QUEUE[@]} -gt 0 ]] || { echo "usage: $0 <TAG>:<SEED>..." >&2; exit 2; }

POLL="${POLL_SECONDS:-300}"          # how often to look for a free card
SETTLE="${SETTLE_SECONDS:-420}"      # wait after a launch before polling again
EXCLUDE="${EXCLUDE_GPUS:-0}"
LIMIT="${MEM_LIMIT_MIB:-2000}"
LOGROOT="/data1/leihuang/TFScope_store/v26_logs"
CKROOT="/data1/leihuang/TFScope_store/ablation_v24/checkpoints/ablation_v24"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

free_uuid() {
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
  awk -F', ' -v ex="$EXCLUDE" -v lim="$LIMIT" '
    BEGIN{n=split(ex,e,/[ ,]+/); for(i=1;i<=n;i++) bad[e[i]]=1}
    !($1 in bad) && $3+0 < lim {print $1"|"$2; exit}'
}

already_running() {   # $1 = job name
  local d="$LOGROOT/$1"
  [[ -f "$d/STATUS" && "$(cat "$d/STATUS")" == "RUNNING" ]] &&
    kill -0 "$(cat "$d/pid" 2>/dev/null)" 2>/dev/null
}

echo "[$(stamp)] queue (${#QUEUE[@]}): ${QUEUE[*]}"
echo "[$(stamp)] poll=${POLL}s settle=${SETTLE}s exclude_gpus=${EXCLUDE}"

i=0
while (( i < ${#QUEUE[@]} )); do
  spec="${QUEUE[$i]}"
  tag="${spec%%:*}"; seed="${spec##*:}"
  [[ "$seed" == "$spec" ]] && seed=42
  job="abl_v24_${tag}_s${seed}"

  if [[ -f "$CKROOT/${tag}/seed${seed}/ckpt_best.pt" ]]; then
    echo "[$(stamp)] SKIP ${spec}: ckpt_best.pt already exists"; ((i++)); continue
  fi
  if already_running "$job"; then
    echo "[$(stamp)] SKIP ${spec}: already RUNNING"; ((i++)); continue
  fi

  slot="$(free_uuid)"
  if [[ -z "$slot" ]]; then
    sleep "$POLL"; continue
  fi
  idx="${slot%%|*}"; uuid="${slot#*|}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[$(stamp)] would launch ${spec} on nvidia-smi GPU ${idx}"; ((i++)); continue
  fi

  echo "[$(stamp)] launching ${spec} on nvidia-smi GPU ${idx} (${uuid})"
  # Re-sync the mirror before each launch so a mid-queue code edit is picked up.
  scripts/ablation_v24/sync_ablation_to_data1.sh >/dev/null 2>&1 ||
    echo "[$(stamp)] WARN: mirror sync failed"
  if REPO_MIRROR=/data1/leihuang/TFScope_store/ablation_v24/run \
       scripts/v26/run_detached.sh --mirror --gpu "$uuid" "$job" \
       bash scripts/ablation_v24/train_ablation_v24.sh "$tag" "$seed"; then
    ((i++))
    echo "[$(stamp)] settling ${SETTLE}s before next poll"
    sleep "$SETTLE"
  else
    echo "[$(stamp)] launch of ${spec} FAILED; retrying after ${POLL}s"
    sleep "$POLL"
  fi
done

echo "[$(stamp)] queue drained; ${#QUEUE[@]} jobs dispatched"
echo "[$(stamp)] NOTE: dispatched != finished -- check scripts/ablation_v24/ablation_status.sh"
