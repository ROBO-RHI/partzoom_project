#!/bin/bash
# Window "status": refreshed overview.
cd ~/3d_diffuser_actor && source partzoom/env.sh 2>/dev/null
echo "=== $(date) ===   (Ctrl-b n / p to switch windows, Ctrl-b d to detach)"
echo; echo "--- pipeline timeline (last 8) ---"; tail -8 logs_partzoom/pipeline.log
echo; echo "--- GPU / RAM ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
free -g | awk 'NR==2{print "RAM used "$3"G / "$2"G"}'
echo; echo "--- running jobs ---"
ps -eo etime,cmd | grep -E "[m]ain_partzoom.py -u|[p]ython -u partzoom/main_partzoom|[p]ython partzoom/(eval_rollouts|offline_eval|bench_infer)" \
  | grep -v "bash -c" | sed -E 's/--variations [0-9 ]+//' | awk '!seen[$0]++' | cut -c1-170 | head -6
echo; echo "--- training progress ---"
for f in $(ls -t logs_partzoom/train_*.log | head -1); do
  echo "$f: $(tail -c 400 $f | tr '\r' '\n' | grep -E 'it\]|s/it' | tail -1 | cut -c1-90)"
  grep -E "^\[val\]" $f | tail -2 | cut -c1-150
done
echo; echo "--- eval progress (latest line per running log) ---"
for f in $(ls -t logs_partzoom/eval_*.log | head -2); do echo "$f: $(grep success= $f | tail -1 | cut -c1-140)"; done
echo; echo "--- results so far (success %, max 12 steps) ---"
python partzoom/analyze.py eval_logs/partzoom 2>/dev/null
