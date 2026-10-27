#!/bin/bash
# EXP-011 launcher (see RESEARCH_LOG.md): full PPO baseline on Minigrid, original sparse reward,
# SimpleCrossingS9N1 and LavaCrossingS9N1 x 5 seeds x 3M steps.
# Each job writes experiments/EXP-011/<name>/ and experiments/EXP-011/logs/<name>.log.
# After training, the greedy (deterministic) evaluation of each final model is added as final_eval_greedy.json.
# Usage (keeps running if the terminal closes; caffeinate stops the Mac from sleeping):
#   nohup caffeinate -is bash src/run_exp011.sh > experiments/EXP-011/logs/launcher.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
OUT=experiments/EXP-011
PARALLEL=${PARALLEL:-5}  # 4 cores / 8 threads; runs are single-threaded and deterministic, so this changes wall-clock only
mkdir -p "$OUT/logs"

run_job() {  # args: config seed name
  if "$PY" src/train.py --config "$1" --seed "$2" --out "$OUT/$3" > "$OUT/logs/$3.log" 2>&1 &&
     "$PY" src/evaluate.py --run-dir "$OUT/$3" --episodes 100 --seed-offset 20000 \
       --out "$OUT/$3/final_eval_greedy.json" >> "$OUT/logs/$3.log" 2>&1; then
    echo "$(date '+%F %T') OK   $3"
  else
    echo "$(date '+%F %T') FAIL $3 (see $OUT/logs/$3.log)"
  fi
}
export -f run_job
export PY OUT

echo "$(date '+%F %T') START EXP-011"
for task in simplecrossing lavacrossing; do
  case $task in simplecrossing) T=SimpleCrossingS9N1;; lavacrossing) T=LavaCrossingS9N1;; esac
  for s in 0 1 2 3 4; do echo "configs/exp011_$task.yaml $s ${T}_sparse_3M_s$s"; done
done | xargs -P "$PARALLEL" -L 1 bash -c 'run_job "$@"' _
echo "$(date '+%F %T') END EXP-011"
