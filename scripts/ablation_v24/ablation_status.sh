#!/usr/bin/env bash
# Progress of all v24 ablation trainings: status, last epoch, best oracle-r, wall clock.
#
#   scripts/ablation_v24/ablation_status.sh                 # one line per run
#   scripts/ablation_v24/ablation_status.sh A2_nodistill_s42 60   # last 60 log lines
set -uo pipefail
LOGROOT="/data1/leihuang/TFScope_store/v26_logs"
CKROOT="/data1/leihuang/TFScope_store/ablation_v24/checkpoints/ablation_v24"

if [[ $# -ge 1 ]]; then
  exec scripts/v26/job_status.sh "abl_v24_$1" "${2:-40}"
fi

printf "%-20s %-6s %-12s %-7s %-10s %-8s %s\n" VARIANT SEED STATUS EPOCH BEST_ORCL CKPT_MB STARTED
for d in "$LOGROOT"/abl_v24_*/; do
  [[ -d "$d" ]] || continue
  j=$(basename "$d"); spec="${j#abl_v24_}"
  [[ "$spec" == "queue" ]] && continue
  seed="${spec##*_s}"; tag="${spec%_s*}"
  s=$(cat "$d/STATUS" 2>/dev/null || echo "?")
  p=$(cat "$d/pid" 2>/dev/null || echo "-")
  [[ "$s" == "RUNNING" ]] && ! kill -0 "$p" 2>/dev/null && s="STALE"
  ep=$(grep -oE '^ *[0-9]+ +[0-9]+\.[0-9]+' "$d/log.txt" 2>/dev/null | tail -1 | awk '{print $1}')
  best=$(grep -oE 'oracle[_ ]r[^0-9-]*(-?[0-9]+\.[0-9]+)' "$d/log.txt" 2>/dev/null |
         grep -oE '(-?[0-9]+\.[0-9]+)$' | sort -g | tail -1)
  ck=$(du -m "$CKROOT/$tag/seed$seed/ckpt_best.pt" 2>/dev/null | cut -f1)
  printf "%-20s %-6s %-12s %-7s %-10s %-8s %s\n" "$tag" "$seed" "$s" "${ep:--}" \
    "${best:--}" "${ck:--}" "$(cat "$d/started_at" 2>/dev/null)"
done
