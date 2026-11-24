#!/bin/bash
# Round 3, window "r3train": gentler matched pair at lr 1e-5 (same recipe otherwise).
source ~/3d_diffuser_actor/partzoom/tmux/common.sh
source <(sed -n '/^train() {/,/^}/p' ~/3d_diffuser_actor/partzoom/tmux/train_queue.sh)
train baseline_lr1e5 --token_mode fps --lr 1e-5
train both_lr1e5 --token_mode part --zoom 1 --lr 1e-5
mark R3TRAIN_DONE
echo "round-3 training finished"; exec bash
