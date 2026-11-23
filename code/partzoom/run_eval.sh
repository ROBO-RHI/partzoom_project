#!/bin/bash
# usage: run_eval.sh NAME CKPT TASK EPISODES [extra eval_rollouts args...]
cd "$(dirname "$0")/.." && source partzoom/env.sh
NAME=$1; CKPT=$2; TASK=$3; EPS=$4; shift 4
mkdir -p eval_logs/partzoom logs_partzoom
xvfb-run -a -s "-screen 0 1024x768x24" python partzoom/eval_rollouts.py \
  --checkpoint $CKPT --tasks $TASK --episodes $EPS \
  --seed_dir data/partzoom/seeds/test \
  --output eval_logs/partzoom/$NAME/$TASK.jsonl "$@" \
  > logs_partzoom/eval_${NAME}_${TASK}.log 2>&1
