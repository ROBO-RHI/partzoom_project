#!/bin/bash
# Final stage: after all rollouts finish, run offline per-keyframe val evals,
# eval-time sweeps, clean inference benchmark, and rebuild the package.
cd "$(dirname "$0")/.." && source partzoom/env.sh
LOG=logs_partzoom/pipeline.log
TASKS="insert_onto_square_peg place_shape_in_shape_sorter slide_block_to_color_target"
until grep -q "PIPELINE_DONE" $LOG; do sleep 60; done
echo "OFFLINE_START $(date)" >> $LOG
REL=train_logs/diffuser_actor_peract.pth
off() {  # NAME CKPT flags
  local N=$1 C=$2; shift 2
  [ -f eval_logs/offline/$N.json ] && return
  python partzoom/offline_eval.py --checkpoint $C --tasks $TASKS \
    --output eval_logs/offline/$N.json "$@" > logs_partzoom/offline_$N.log 2>&1
}
T=train_logs/partzoom
# FPS models get part regions only for token-retention statistics (FPS ignores them)
off released $REL --token_mode fps --region_source part
off baseline $T/baseline/last.pth --token_mode fps --region_source part
off part $T/part/last.pth --token_mode part
off zoom $T/zoom/last.pth --zoom 1
off both $T/both/last.pth --token_mode part --zoom 1
off released_part $REL --token_mode part
off released_zoom $REL --zoom 1
off baseline_lr2e5 $T/baseline_lr2e5/last.pth --token_mode fps --region_source part
off both_lr2e5 $T/both_lr2e5/last.pth --token_mode part --zoom 1
off zoom_object $T/zoom_object/last.pth --zoom 1 --region_source object
off zoom_random $T/zoom_random/last.pth --zoom 1 --region_source random
# eval-time sweeps on the trained zoom / part / both models
for nz in 0.01 0.02 0.04; do
  off zoom_noise$nz $T/zoom/last.pth --zoom 1 --region_noise $nz
  off part_noise$nz $T/part/last.pth --token_mode part --region_noise $nz
  off both_noise$nz $T/both/last.pth --token_mode part --zoom 1 --region_noise $nz
done
for cm in 0.0 0.1 0.5 1.0; do off zoom_margin$cm $T/zoom/last.pth --zoom 1 --crop_margin $cm; done
for zt in 128 384; do off zoom_tokens$zt $T/zoom/last.pth --zoom 1 --zoom_tokens $zt; done
for rc in 0.25 0.75; do off part_cap$rc $T/part/last.pth --token_mode part --rel_cap $rc; done
python partzoom/bench_infer.py > logs_partzoom/bench_infer.log 2>&1
echo "OFFLINE_DONE $(date)" >> $LOG
python partzoom/make_package.py > logs_partzoom/make_package.log 2>&1
echo "PACKAGE_DONE $(date)" >> $LOG
