#!/bin/bash
# Round 3, window "r3final": offline val evals for the lr-1e-5 models, then tables + package.
source ~/3d_diffuser_actor/partzoom/tmux/common.sh
OUT=$HOME/harsh/partzoom_project
echo "waiting for both round-3 eval workers ..."
until [ $(grep -c "^R3EVALQ_DONE" $LOG) -ge 2 ]; do sleep 60; done
mark R3OFFLINE_START
T=train_logs/partzoom
for spec in "baseline_lr1e5|--token_mode fps --region_source part" "both_lr1e5|--token_mode part --zoom 1"; do
  IFS='|' read -r N FLAGS <<< "$spec"
  [ -f eval_logs/offline/$N.json ] && continue
  python partzoom/offline_eval.py --checkpoint $T/$N/last.pth --tasks $TASKS \
    --output eval_logs/offline/$N.json $FLAGS 2>&1 | tee logs_partzoom/offline_$N.log | grep -vE "Warning|warn"
done
python partzoom/make_package.py --out $OUT 2>&1 | tail -5
python partzoom/extra_tables.py --out $OUT 2>&1 | tail -3
python partzoom/ext_tables.py --out $OUT
python partzoom/make_fig.py --out $OUT 2>&1 | tail -3
mark R3_PACKAGE_DONE
echo "round-3 final stage finished"; exec bash
