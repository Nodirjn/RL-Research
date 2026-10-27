#!/bin/bash
# EXP-001 launcher (see RESEARCH_LOG.md). Runs jobs 3 at a time; each job writes
# experiments/EXP-001/<name>/ and experiments/EXP-001/logs/<name>.log.
#   stage a: C2 x 3 seeds x 1M        (benchmark gate vs published 1M PPO results)
#   stage b: C1, C2, C3 x 3 seeds x 3M (baseline-config selection; longest jobs first)
# Usage (keeps running if the terminal closes; caffeinate stops the Mac from sleeping):
#   nohup caffeinate -is bash src/run_exp001.sh a > experiments/EXP-001/logs/stage_a.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
OUT=experiments/EXP-001
PARALLEL=${PARALLEL:-3}  # runs are single-threaded and deterministic, so this changes wall-clock only
mkdir -p "$OUT/logs"

jobs_for_stage() {
  case "$1" in
    a) for s in 0 1 2; do echo "configs/exp001_c2_cleanrl.yaml $s C2_cleanrl_1M_s$s 1000000"; done ;;
    b) for c in c3_zoo:C3_zoo c2_cleanrl:C2_cleanrl c1_hist:C1_hist; do
         for s in 0 1 2; do echo "configs/exp001_${c%%:*}.yaml $s ${c##*:}_3M_s$s 3000000"; done
       done ;;
    *) echo "unknown stage '$1' (use a or b)" >&2; exit 1 ;;
  esac
}

run_job() {  # args: config seed name total_timesteps
  if "$PY" src/train.py --config "$1" --seed "$2" --out "$OUT/$3" --total-timesteps "$4" > "$OUT/logs/$3.log" 2>&1; then
    echo "$(date '+%F %T') OK   $3"
  else
    echo "$(date '+%F %T') FAIL $3 (see $OUT/logs/$3.log)"
  fi
}
export -f run_job
export PY OUT

echo "$(date '+%F %T') START stage $1"
jobs_for_stage "$1" | xargs -P "$PARALLEL" -L 1 bash -c 'run_job "$@"' _
echo "$(date '+%F %T') END stage $1"
