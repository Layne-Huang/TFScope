#!/usr/bin/env bash
# Start one v22 stage/seed in a detached tmux session that survives Cursor/SSH.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 <stage> <seed> <cuda-visible-device>" >&2
  exit 2
fi

STAGE="$1"
SEED="$2"
GPU="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="tfscope-v22-${STAGE}-seed${SEED}"
TMUX_SOCKET="${TFSCOPE_TMUX_SOCKET:-tfscope-v22}"
OUT_ROOT="${OUT_ROOT:-/data1/leihuang/project/TFScope/checkpoints/v22_ablation}"
TMUX_LOG_DIR="$OUT_ROOT/tmux"
mkdir -p "$TMUX_LOG_DIR"

if tmux -L "$TMUX_SOCKET" has-session -t "$SESSION" 2>/dev/null; then
  echo "$SESSION is already active" >&2
  exit 1
fi

printf -v command \
  'cd %q && env STAGES=%q SEEDS=%q CUDA_VISIBLE_DEVICES=%q OUT_ROOT=%q RESUME_AUTO=1 TORCH_HOME=%q /usr/bin/bash %q >>%q 2>&1' \
  "$ROOT" "$STAGE" "$SEED" "$GPU" "$OUT_ROOT" \
  "/data1/leihuang/.cache/torch" "$ROOT/scripts/run_v22_ablation.sh" \
  "$TMUX_LOG_DIR/$SESSION.log"

tmux -L "$TMUX_SOCKET" new-session -d -s "$SESSION" "$command"
echo "started tmux session $SESSION (GPU $GPU)"
echo "attach: tmux -L $TMUX_SOCKET attach -t $SESSION"
