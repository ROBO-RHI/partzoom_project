#!/bin/bash
# Round 2: takes over from pipeline.sh after "both" finishes training.
# 1) zero-shot evals of released weights + part sampling / zoom (no training)
# 2) low-LR (2e-5) fine-tunes of baseline and both
# 3) secondary comparisons (object / random zoom) with the round-1 recipe
cd "$(dirname "$0")/.."
TASKS="insert_onto_square_peg place_shape_in_shape_sorter slide_block_to_color_target"
LOG=logs_partzoom/pipeline.log
REL=train_logs/diffuser_actor_peract.pth

# --- take over ---
until grep -q "TRAIN_DONE both" $LOG; do sleep 5; done
sleep 3
kill 69456 2>/dev/null                      # round-1 driver (eval subshell for 'both' keeps running)
pkill -f "run_log_dir [z]oom_object" 2>/dev/null
sleep 10; rm -rf train_logs/partzoom/zoom_object logs_partzoom/train_zoom_object.log
echo "ROUND2_START $(date)" >> $LOG

eval_ckpt() {  # NAME CKPT flags...
  local NAME=$1 CKPT=$2; shift 2
  (
    flock 9
    for T in $TASKS; do
      partzoom/run_eval.sh $NAME $CKPT $T 0:50 --max_steps 12 "$@"
    done
    echo "EVAL_DONE $NAME $(date)" >> $LOG
  ) 9>/tmp/partzoom_eval.lock &
  sleep 2
}

train() {  # NAME flags...
  local NAME=$1; shift
  echo "TRAIN_START $NAME $(date)" >> $LOG
  partzoom/train_variant.sh $NAME 4000 "$@"
  echo "TRAIN_DONE $NAME $(date)" >> $LOG
}

eval_ckpt released_part $REL --token_mode part
eval_ckpt released_zoom $REL --zoom 1

train baseline_lr2e5 --token_mode fps --lr 2e-5
eval_ckpt baseline_lr2e5 train_logs/partzoom/baseline_lr2e5/last.pth --token_mode fps --region_source none

train both_lr2e5 --token_mode part --zoom 1 --lr 2e-5
eval_ckpt both_lr2e5 train_logs/partzoom/both_lr2e5/last.pth --token_mode part --zoom 1

train zoom_object --zoom 1 --region_source object
eval_ckpt zoom_object train_logs/partzoom/zoom_object/last.pth --zoom 1 --region_source object

train zoom_random --zoom 1 --region_source random
eval_ckpt zoom_random train_logs/partzoom/zoom_random/last.pth --zoom 1 --region_source random

wait
echo "PIPELINE_DONE $(date)" >> $LOG
