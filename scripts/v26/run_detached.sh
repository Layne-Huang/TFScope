#!/usr/bin/env bash
# Launch a v26 job fully detached from the calling shell / SSH session / agent session.
#
#   scripts/v26/run_detached.sh <job_name> <command...>
#   scripts/v26/run_detached.sh --gpu 4 <job_name> <command...>
#
# The job survives terminal close, SSH drop and agent-session termination:
#   * setsid   -> new session, no controlling terminal
#   * nohup    -> immune to SIGHUP
#   * </dev/null, >log 2>&1, disown
#
# Per-job state lives in $ROOT/<job_name>/ :
#   cmd.txt   exact command + env
#   pid       PID of the detached process group leader
#   log.txt   merged stdout/stderr (line-buffered)
#   STATUS    RUNNING -> DONE | FAILED:<exit_code>
#   started_at / finished_at
#
# Inspect:  scripts/v26/job_status.sh [job_name]
# Kill:     kill -- -$(cat <ROOT>/<job>/pid)      # negative PID = whole process group

set -uo pipefail

REPO_AFS="/afs/csail.mit.edu/u/l/leihuang/project/TFScope"
# Overridable so other experiment families (e.g. scripts/lofo) can point --mirror at
# their own self-contained /data1 mirror without duplicating this launcher.
REPO_MIRROR="${REPO_MIRROR:-/data1/leihuang/TFScope_store/v26/run}"
ROOT="${JOB_ROOT:-/data1/leihuang/TFScope_store/v26_logs}"
PY="/data1/leihuang/miniconda3/envs/tfscope/bin/python"

GPU=""
MIRROR=""
while true; do
  case "${1:-}" in
    --gpu)    GPU="$2"; shift 2 ;;
    # Run from the /data1 mirror instead of AFS, so the job needs ZERO AFS access.
    # AFS tokens on this host drop every ~10 min; use this for anything long-running.
    --mirror) MIRROR=1; shift ;;
    *) break ;;
  esac
done

JOB="${1:-}"; shift || true
if [[ -z "$JOB" || $# -eq 0 ]]; then
  echo "usage: $0 [--gpu N] <job_name> <command...>" >&2; exit 2
fi

REPO="${MIRROR:+$REPO_MIRROR}"; REPO="${REPO:-$REPO_AFS}"
DIR="$ROOT/$JOB"
if [[ -f "$DIR/STATUS" && "$(cat "$DIR/STATUS")" == "RUNNING" ]]; then
  if kill -0 "$(cat "$DIR/pid" 2>/dev/null)" 2>/dev/null; then
    echo "REFUSING: job '$JOB' is already RUNNING (pid $(cat "$DIR/pid"))" >&2; exit 3
  fi
  echo "WARN: stale RUNNING marker for '$JOB' (process gone); overwriting" >&2
fi
mkdir -p "$DIR"

{
  echo "job:      $JOB"
  echo "cwd:      $REPO"
  echo "gpu:      ${GPU:-<none>}"
  echo "python:   $PY"
  echo "command:  $*"
} > "$DIR/cmd.txt"

date -Iseconds > "$DIR/started_at"
echo RUNNING > "$DIR/STATUS"

# The wrapper runs the payload, then records the exit status. Detached via setsid+nohup.
setsid nohup bash -c '
  cd "'"$REPO"'" || exit 90
  export PYTHONUNBUFFERED=1
  export PYTHONPATH=src
  export TORCH_HOME=/data1/leihuang/.cache/torch
  export HF_HOME=/data1/leihuang/.cache/huggingface
  export OMP_NUM_THREADS=4
  # CUDA enumerates "fastest first" by default, which does NOT match nvidia-smi
  # indices on this node. Pin PCI order so --gpu N means physical GPU N.
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  '"${GPU:+export CUDA_VISIBLE_DEVICES=$GPU}"'
  "$@"
  rc=$?
  date -Iseconds > "'"$DIR"'/finished_at"
  if [[ $rc -eq 0 ]]; then echo DONE > "'"$DIR"'/STATUS"; else echo "FAILED:$rc" > "'"$DIR"'/STATUS"; fi
  exit $rc
' _ "$@" </dev/null >"$DIR/log.txt" 2>&1 &

PID=$!
echo "$PID" > "$DIR/pid"
disown "$PID" 2>/dev/null || true

echo "launched '$JOB' pid=$PID"
echo "  log:    $DIR/log.txt"
echo "  status: $DIR/STATUS"
