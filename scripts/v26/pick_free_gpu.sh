#!/usr/bin/env bash
# Print the UUID of a free GPU, EXCLUDING index 0.
# GPU 0 on this node is dead: PyTorch reports "No CUDA GPUs are available" even when the UUID is
# pinned explicitly, and nvidia-smi still shows it as idle -- so any "first free card" heuristic
# silently selects it and the job hangs. Verified 2026-08-14 and again 2026-08-18.
set -uo pipefail
EXCLUDE="${EXCLUDE_GPUS:-0}"
LIMIT="${MEM_LIMIT_MIB:-2000}"
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
awk -F', ' -v ex="$EXCLUDE" -v lim="$LIMIT" '
  BEGIN{n=split(ex,e,/[ ,]+/); for(i=1;i<=n;i++) bad[e[i]]=1}
  !($1 in bad) && $3+0 < lim {print $2; exit}'
