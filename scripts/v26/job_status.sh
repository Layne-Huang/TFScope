#!/usr/bin/env bash
# Status of detached v26 jobs.
#   scripts/v26/job_status.sh              # table of all jobs
#   scripts/v26/job_status.sh <job> [n]    # last n log lines of one job (default 40)
set -uo pipefail
ROOT="/data1/leihuang/TFScope_store/v26_logs"

if [[ $# -ge 1 ]]; then
  JOB="$1"; N="${2:-40}"; D="$ROOT/$JOB"
  [[ -d "$D" ]] || { echo "no such job: $JOB" >&2; exit 1; }
  echo "=== $JOB : $(cat "$D/STATUS" 2>/dev/null) (started $(cat "$D/started_at" 2>/dev/null))"
  sed -n '1,10p' "$D/cmd.txt"
  echo "--- last $N log lines ---"
  tail -n "$N" "$D/log.txt" 2>/dev/null
  exit 0
fi

printf "%-34s %-14s %-10s %s\n" JOB STATUS PID STARTED
for d in "$ROOT"/*/; do
  [[ -d "$d" ]] || continue
  j=$(basename "$d"); s=$(cat "$d/STATUS" 2>/dev/null || echo "?")
  p=$(cat "$d/pid" 2>/dev/null || echo "-")
  if [[ "$s" == "RUNNING" ]] && ! kill -0 "$p" 2>/dev/null; then s="STALE"; fi
  printf "%-34s %-14s %-10s %s\n" "$j" "$s" "$p" "$(cat "$d/started_at" 2>/dev/null)"
done
