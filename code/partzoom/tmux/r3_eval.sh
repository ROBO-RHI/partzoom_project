#!/bin/bash
# Round 3, windows "r3eval1"/"r3eval2": closed-loop evals on all 100 test scenes
# (episodes 0-99; already-finished episodes 0-49 are skipped automatically).
# Worker 2 waits for training to finish (RAM). Jobs claimed atomically via mkdir.
source ~/3d_diffuser_actor/partzoom/tmux/common.sh
W=$1
REL=train_logs/diffuser_actor_peract.pth
T=train_logs/partzoom
JOBS=(
  "released|$REL|-|--token_mode fps --region_source none"
  "both_lr2e5|$T/both_lr2e5/last.pth|-|--token_mode part --zoom 1"
  "baseline_lr2e5|$T/baseline_lr2e5/last.pth|-|--token_mode fps --region_source none"
  "baseline_lr1e5|$T/baseline_lr1e5/last.pth|TRAIN_DONE baseline_lr1e5|--token_mode fps --region_source none"
  "both_lr1e5|$T/both_lr1e5/last.pth|TRAIN_DONE both_lr1e5|--token_mode part --zoom 1"
  "released_part|$REL|-|--token_mode part"
)
mkdir -p logs_partzoom/claims_r3
if [ "$W" != 1 ]; then echo "worker $W waits for R3TRAIN_DONE"; until has R3TRAIN_DONE; do sleep 60; done; fi
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME CKPT PRE FLAGS <<< "$job"
  has "R3EVAL_DONE $NAME" && continue
  mkdir logs_partzoom/claims_r3/$NAME 2>/dev/null || continue
  if [ "$PRE" != "-" ]; then echo "[$NAME] waiting for: $PRE"; until has "$PRE"; do sleep 60; done; fi
  mark R3EVAL_START $NAME worker$W
  for TK in $TASKS; do
    echo "=== $NAME / $TK (episodes 0-99) ==="
    xvfb-run -a -s "-screen 0 1024x768x24" python partzoom/eval_rollouts.py \
      --checkpoint $CKPT --tasks $TK --episodes 0:100 --seed_dir data/partzoom/seeds/test \
      --output eval_logs/partzoom/$NAME/$TK.jsonl --max_steps 12 $FLAGS 2>&1 \
      | tee -a logs_partzoom/eval_${NAME}_${TK}.log | grep --line-buffered -E "success=|Traceback|Error"
  done
  mark R3EVAL_DONE $NAME
done
mark R3EVALQ_DONE worker$W
echo "round-3 eval worker $W finished"; exec bash
