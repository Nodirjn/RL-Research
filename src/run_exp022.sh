#!/bin/bash
# EXP-022 launcher (pre-registration: documentation/stages/12_shifted_potential_experiment.md): the shifted BFS
# potential Phi = 1 - b/B on SimpleCrossing with a 5x5 view, seeds 0-9 (the seeds of the EXP-021 BFS runs),
# 3M steps per run. 10 runs.
# Each job writes experiments/EXP-022/<name>/ and experiments/EXP-022/logs/<name>.log, then adds the greedy
# evaluation of the final model as final_eval_greedy.json. Finished runs are skipped, so a restart resumes.
# Usage (keeps running if the terminal closes; caffeinate stops the Mac from sleeping):
#   nohup caffeinate -is bash src/run_exp022.sh > experiments/EXP-022/logs/launcher.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
OUT=experiments/EXP-022
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

echo "$(date '+%F %T') START EXP-022"
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "configs/exp022_simplecrossing_bfs_shifted.yaml $s SimpleCrossingS9N1_view5_bfs_shifted_3M_s$s"
done | xargs -P "$PARALLEL" -L 1 bash -c 'run_job "$@"' _
echo "$(date '+%F %T') END EXP-022"
