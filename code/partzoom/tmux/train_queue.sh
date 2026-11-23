#!/bin/bash
# Window "train": fine-tunes run here in the foreground (live tqdm + val lines).
source ~/3d_diffuser_actor/partzoom/tmux/common.sh
train() {  # NAME flags...  (same flags as partzoom/train_variant.sh)
  local NAME=$1; shift
  if has "TRAIN_DONE $NAME"; then echo "skip $NAME (done)"; return; fi
  mark TRAIN_START $NAME
  torchrun --nproc_per_node 1 --master_port $((29500 + RANDOM % 1000)) partzoom/main_partzoom.py \
    --tasks $TASKS \
    --dataset data/partzoom/packaged/train --valset data/partzoom/packaged/val \
    --instructions instructions/peract/instructions.pkl \
    --gripper_loc_bounds tasks/18_peract_tasks_location_bounds.json --gripper_loc_bounds_buffer 0.04 \
    --num_workers 2 --train_iters 4000 --val_freq 1000 --val_iters 8 \
    --embedding_dim 120 --use_instruction 1 --rotation_parametrization 6D \
    --diffusion_timesteps 100 --dense_interpolation 1 --interpolation_length 2 \
    --batch_size 4 --batch_size_val 4 --cache_size 0 --cache_size_val 0 \
    --keypose_only 1 --variations $(seq -s ' ' 0 199) --lr 1e-4 --num_history 3 \
    --cameras left_shoulder right_shoulder wrist front --max_episodes_per_task -1 \
    --quaternion_format wxyz --seed 0 \
    --init_weights train_logs/diffuser_actor_peract.pth \
    --exp_log_dir partzoom --run_log_dir $NAME "$@" 2>&1 | tee logs_partzoom/train_$NAME.log
  mark TRAIN_DONE $NAME
}

# baseline_lr2e5 was started under nohup before the switch to tmux: follow it
if pgrep -f "run_log_dir [b]aseline_lr2e5" >/dev/null; then
  echo "== following baseline_lr2e5 (started before tmux switch) =="
  tail -n 0 -F logs_partzoom/train_baseline_lr2e5.log &
  TP=$!
  while pgrep -f "run_log_dir [b]aseline_lr2e5" >/dev/null; do sleep 20; done
  kill $TP
  mark TRAIN_DONE baseline_lr2e5
fi

train both_lr2e5 --token_mode part --zoom 1 --lr 2e-5
train zoom_object --zoom 1 --region_source object
train zoom_random --zoom 1 --region_source random
mark TRAINQ_DONE
echo "training queue finished"; exec bash
