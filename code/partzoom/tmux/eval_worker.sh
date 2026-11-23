#!/bin/bash
# Window "eval1"/"eval2": rollout evals in the foreground. Jobs are claimed
# atomically (mkdir) so two workers never run the same job.
# usage: eval_worker.sh WORKER_ID   (worker 2 waits until training is over: RAM)
source ~/3d_diffuser_actor/partzoom/tmux/common.sh
W=$1
REL=train_logs/diffuser_actor_peract.pth
T=train_logs/partzoom
JOBS=(
  "released_part|$REL|-|--token_mode part"
  "released_zoom|$REL|-|--zoom 1"
  "baseline_lr2e5|$T/baseline_lr2e5/last.pth|TRAIN_DONE baseline_lr2e5|--token_mode fps --region_source none"
  "both_lr2e5|$T/both_lr2e5/last.pth|TRAIN_DONE both_lr2e5|--token_mode part --zoom 1"
  "zoom_object|$T/zoom_object/last.pth|TRAIN_DONE zoom_object|--zoom 1 --region_source object"
  "zoom_random|$T/zoom_random/last.pth|TRAIN_DONE zoom_random|--zoom 1 --region_source random"
)
mkdir -p logs_partzoom/claims
if [ "$W" = 1 ] && pgrep -f "[b]oth/last.pth" >/dev/null; then
  echo "== following eval of 'both' (started before tmux switch) =="
  tail -n 0 -F logs_partzoom/eval_both_*.log 2>/dev/null | grep --line-buffered -E "success=|Traceback|Error" &
  TP=$!
  while pgrep -f "[b]oth/last.pth" >/dev/null; do sleep 20; done
  kill $TP
fi
if [ "$W" != 1 ]; then echo "worker $W waits for TRAINQ_DONE"; until has TRAINQ_DONE; do sleep 60; done; fi
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME CKPT PRE FLAGS <<< "$job"
  has "EVAL_DONE $NAME" && continue
  mkdir logs_partzoom/claims/$NAME 2>/dev/null || continue
  if [ "$PRE" != "-" ]; then echo "[$NAME] waiting for: $PRE"; until has "$PRE"; do sleep 60; done; fi
  mark EVAL_START $NAME worker$W
  for TK in $TASKS; do
    echo "=== $NAME / $TK ==="
    xvfb-run -a -s "-screen 0 1024x768x24" python partzoom/eval_rollouts.py \
      --checkpoint $CKPT --tasks $TK --episodes 0:50 --seed_dir data/partzoom/seeds/test \
      --output eval_logs/partzoom/$NAME/$TK.jsonl --max_steps 12 $FLAGS 2>&1 \
      | tee -a logs_partzoom/eval_${NAME}_${TK}.log | grep --line-buffered -E "success=|Traceback|Error"
  done
  mark EVAL_DONE $NAME
done
mark EVALQ_DONE worker$W
echo "eval worker $W finished"; exec bash
