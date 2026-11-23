#!/bin/bash
# Sequential training queue; each finished variant gets its rollout evals
# launched in the background (one eval chain at a time, serialized via flock).
# usage: pipeline.sh ITERS   (baseline assumed already started separately)
cd "$(dirname "$0")/.."
ITERS=${1:-4000}
TASKS="insert_onto_square_peg place_shape_in_shape_sorter slide_block_to_color_target"
EVAL_ARGS="--max_steps 12"

eval_variant() {  # NAME then eval flags
  local NAME=$1; shift
  (
    flock 9
    while pgrep -f "eval_logs/partzoom/released" >/dev/null; do sleep 60; done
    for T in $TASKS; do
      partzoom/run_eval.sh $NAME train_logs/partzoom/$NAME/last.pth $T 0:50 $EVAL_ARGS "$@"
    done
    echo "EVAL_DONE $NAME $(date)" >> logs_partzoom/pipeline.log
  ) 9>/tmp/partzoom_eval.lock &
}

train() {  # NAME then flags shared by train+eval
  local NAME=$1; shift
  echo "TRAIN_START $NAME $(date)" >> logs_partzoom/pipeline.log
  partzoom/train_variant.sh $NAME $ITERS "$@"
  echo "TRAIN_DONE $NAME $(date)" >> logs_partzoom/pipeline.log
}

# baseline: wait for the separately-launched run
while pgrep -f "run_log_dir baseline" >/dev/null; do sleep 60; done
echo "TRAIN_DONE baseline $(date)" >> logs_partzoom/pipeline.log
eval_variant baseline --token_mode fps --region_source none

train part --token_mode part
eval_variant part --token_mode part

train zoom --zoom 1
eval_variant zoom --zoom 1

train both --token_mode part --zoom 1
eval_variant both --token_mode part --zoom 1

train zoom_object --zoom 1 --region_source object
eval_variant zoom_object --zoom 1 --region_source object

train zoom_random --zoom 1 --region_source random
eval_variant zoom_random --zoom 1 --region_source random

wait
echo "PIPELINE_DONE $(date)" >> logs_partzoom/pipeline.log
