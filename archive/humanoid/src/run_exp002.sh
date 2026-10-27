#!/bin/bash
# EXP-002 launcher (see RESEARCH_LOG.md): the full PPO baseline = EXP-001 C3 config x 5 seeds x 10M steps.
# Each job writes experiments/EXP-002/<name>/ and experiments/EXP-002/logs/<name>.log.
# Seeds 0-2 also re-check determinism: up to 3M their eval.csv must equal EXP-001 C3_zoo_3M_s<seed> (constant LR).
# Usage (keeps running if the terminal closes; caffeinate stops the Mac from sleeping):
#   nohup caffeinate -is bash src/run_exp002.sh > experiments/EXP-002/logs/launcher.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
OUT=experiments/EXP-002
PARALLEL=${PARALLEL:-5}  # 4 cores / 8 threads; runs are single-threaded and deterministic, so this changes wall-clock only
mkdir -p "$OUT/logs"

run_job() {  # args: config seed name
  if "$PY" src/train.py --config "$1" --seed "$2" --out "$OUT/$3" > "$OUT/logs/$3.log" 2>&1; then
    echo "$(date '+%F %T') OK   $3"
  else
    echo "$(date '+%F %T') FAIL $3 (see $OUT/logs/$3.log)"
  fi
}
export -f run_job
export PY OUT

echo "$(date '+%F %T') START EXP-002"
for s in 0 1 2 3 4; do echo "configs/exp002_baseline.yaml $s C3_zoo_10M_s$s"; done | xargs -P "$PARALLEL" -L 1 bash -c 'run_job "$@"' _
echo "$(date '+%F %T') END EXP-002"
