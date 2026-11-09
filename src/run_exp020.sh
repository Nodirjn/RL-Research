#!/bin/bash
# EXP-020 launcher (pre-registration: documentation/stages/08_shaping_experiment.md): PBRS conditions
# against the sparse baseline, 7x7 view, 3M steps per run.
#   core       LavaCrossing (euclidean, euclidean_lava, bfs, naive: seeds 0-9; sparse: seeds 5-9), then
#              SimpleCrossing (euclidean, bfs, naive: seeds 0-9; sparse: seeds 5-9). 80 runs.
#   extension  LavaCrossing shifted potential, seeds 0-9. 10 runs.
# Jobs are ordered seed by seed, so stopping early leaves every condition with about the same number of seeds.
# Each job writes experiments/EXP-020/<name>/ and experiments/EXP-020/logs/<name>.log, then adds the greedy
# evaluation of the final model as final_eval_greedy.json. Finished runs are skipped, so a restart resumes.
# Usage (keeps running if the terminal closes; caffeinate stops the Mac from sleeping):
#   nohup caffeinate -is bash src/run_exp020.sh core > experiments/EXP-020/logs/launcher_core.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PHASE=${1:?usage: run_exp020.sh core|extension}
PY=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
OUT=experiments/EXP-020
PARALLEL=${PARALLEL:-5}  # 4 cores / 8 threads; runs are single-threaded and deterministic, so this changes wall-clock only
mkdir -p "$OUT/logs"

run_job() {  # args: config seed name
  if [ -f "$OUT/$3/final_eval_greedy.json" ]; then
    echo "$(date '+%F %T') SKIP $3 (finished)"
  elif "$PY" src/train.py --config "$1" --seed "$2" --out "$OUT/$3" > "$OUT/logs/$3.log" 2>&1 &&
       "$PY" src/evaluate.py --run-dir "$OUT/$3" --episodes 100 --seed-offset 20000 \
         --out "$OUT/$3/final_eval_greedy.json" >> "$OUT/logs/$3.log" 2>&1; then
    echo "$(date '+%F %T') OK   $3"
  else
    echo "$(date '+%F %T') FAIL $3 (see $OUT/logs/$3.log)"
  fi
}
export -f run_job
export PY OUT

jobs_for() {  # args: task conditions...
  local task=$1; shift
  local T
  case $task in simplecrossing) T=SimpleCrossingS9N1;; lavacrossing) T=LavaCrossingS9N1;; esac
  for s in 0 1 2 3 4 5 6 7 8 9; do
    for cond in "$@"; do
      if [ "$cond" = sparse ] && [ "$s" -lt 5 ]; then continue; fi  # seeds 0-4 of the baseline are EXP-011
      echo "configs/exp020_${task}_${cond}.yaml $s ${T}_${cond}_3M_s$s"
    done
  done
}

echo "$(date '+%F %T') START EXP-020 $PHASE"
case $PHASE in
  core)
    { jobs_for lavacrossing euclidean euclidean_lava bfs naive sparse
      jobs_for simplecrossing euclidean bfs naive sparse; } ;;
  extension)
    jobs_for lavacrossing shifted ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac | xargs -P "$PARALLEL" -L 1 bash -c 'run_job "$@"' _
echo "$(date '+%F %T') END EXP-020 $PHASE"
