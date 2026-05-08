#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG="python/rl/configs/stage2_graph_cpu_i5_12500h.json"

if [ -x "$ROOT/.conda/trilibgo-rl/bin/python" ]; then
    PYTHON_EXE="$ROOT/.conda/trilibgo-rl/bin/python"
else
    PYTHON_EXE="python3"
fi

echo "[TriLibGo] Stage-2 RL training"
echo "[TriLibGo] Config: $CONFIG"

if [ $# -eq 0 ]; then
    echo "[TriLibGo] Starting fresh run"
    "$PYTHON_EXE" -m python.rl.train --config "$CONFIG"
else
    echo "[TriLibGo] Resuming from: $1"
    "$PYTHON_EXE" -m python.rl.train --config "$CONFIG" --resume "$1"
fi
