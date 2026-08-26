#!/usr/bin/env bash
# Progress of every v26 sweep run: best validation cov_r so far, epoch, and last log line.
CK=/data1/leihuang/TFScope_store/checkpoints/v26
printf "%-22s %-5s %-8s %-9s %-6s %s\n" CONFIG SEED STATUS BEST_COVR EPOCH LAST
shopt -s nullglob
for d in "$CK"/*/seed*/; do
  name=$(basename "$(dirname "$d")"); seed=$(basename "$d" | sed 's/seed//')
  b="-"; e="-"
  if [ -f "$d/DONE.json" ]; then
    st=DONE
    read -r b e < <(python3 -c "
import json;d=json.load(open('$d/DONE.json'))
print(round(d['best_cov_r'],4), d['best_epoch'])" 2>/dev/null)
  elif [ -f "$d/history.json" ]; then
    st=RUNNING
    read -r b e < <(python3 -c "
import json
h=json.load(open('$d/history.json'))
m=max(h,key=lambda x:x.get('cov_r',-9)) if h else {}
print(round(m.get('cov_r',0),4), m.get('epoch',0))" 2>/dev/null)
  elif [ -f "$d/train.log" ]; then
    st=STARTING
  else
    st=PENDING
  fi
  last=$(tail -1 "$d/train.log" 2>/dev/null | cut -c1-56)
  printf "%-22s %-5s %-8s %-9s %-6s %s\n" "$name" "$seed" "$st" "$b" "$e" "$last"
done
