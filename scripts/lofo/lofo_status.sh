#!/usr/bin/env bash
# Progress of all LOFO v24 trainings: status, last epoch, best oracle-r, wall clock.
#
#   scripts/lofo/lofo_status.sh            # one line per family
#   scripts/lofo/lofo_status.sh C2H2 60    # last 60 log lines of one family
set -uo pipefail
LOGROOT="/data1/leihuang/TFScope_store/v26_logs"
CKROOT="/data1/leihuang/TFScope_store/lofo_v24/checkpoints/lofo_v24"

if [[ $# -ge 1 ]]; then
  exec scripts/v26/job_status.sh "lofo_v24_$1" "${2:-40}"
fi

printf "%-18s %-12s %-7s %-10s %-10s %s\n" FAMILY STATUS EPOCH BEST_ORCL CKPT_MB STARTED
for d in "$LOGROOT"/lofo_v24_*/; do
  [[ -d "$d" ]] || continue
  j=$(basename "$d"); fam="${j#lofo_v24_}"
  s=$(cat "$d/STATUS" 2>/dev/null || echo "?")
  p=$(cat "$d/pid" 2>/dev/null || echo "-")
  [[ "$s" == "RUNNING" ]] && ! kill -0 "$p" 2>/dev/null && s="STALE"
  # train.py prints "  <epoch>  <train loss>  <val loss> ..." per epoch
  ep=$(grep -oE '^ *[0-9]+ +[0-9]+\.[0-9]+' "$d/log.txt" 2>/dev/null | tail -1 | awk '{print $1}')
  best=$(grep -oE 'oracle[_ ]r[^0-9-]*(-?[0-9]+\.[0-9]+)' "$d/log.txt" 2>/dev/null |
         grep -oE '(-?[0-9]+\.[0-9]+)$' | sort -g | tail -1)
  ck=$(du -m "$CKROOT/$fam/ckpt_best.pt" 2>/dev/null | cut -f1)
  printf "%-18s %-12s %-7s %-10s %-10s %s\n" "$fam" "$s" "${ep:--}" "${best:--}" \
    "${ck:--}" "$(cat "$d/started_at" 2>/dev/null)"
done
