#!/bin/bash
# Window "offline": after all rollouts, offline val evals + sweeps + benchmark + package.
source ~/3d_diffuser_actor/partzoom/tmux/common.sh
echo "waiting for both eval workers (EVALQ_DONE x2) and EVAL_DONE both ..."
until [ $(grep -c "^EVALQ_DONE" $LOG) -ge 2 ] && has "EVAL_DONE both"; do sleep 60; done
sed -e 's/^until grep -q "PIPELINE_DONE".*$//' -e 's/^cd ~\/3d_diffuser_actor.*$//' partzoom/pipeline3.sh > /tmp/partzoom_offline_body.sh
# run each offline job with live output
sed -i 's#> logs_partzoom/offline_$N.log 2>&1#2>\&1 | tee logs_partzoom/offline_$N.log | grep --line-buffered -vE "Warning|warn"#' /tmp/partzoom_offline_body.sh
sed -i 's#> logs_partzoom/bench_infer.log 2>&1#2>\&1 | tee logs_partzoom/bench_infer.log#; s#> logs_partzoom/make_package.log 2>&1#2>\&1 | tee logs_partzoom/make_package.log | tail -40#' /tmp/partzoom_offline_body.sh
source /tmp/partzoom_offline_body.sh
echo "offline stage finished"; exec bash
