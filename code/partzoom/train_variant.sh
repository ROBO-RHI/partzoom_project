#!/bin/bash
# usage: train_variant.sh NAME ITERS [extra main_partzoom args, e.g. --token_mode part --zoom 1]
cd "$(dirname "$0")/.." && source partzoom/env.sh
NAME=$1; ITERS=$2; shift 2
mkdir -p logs_partzoom
torchrun --nproc_per_node 1 --master_port $((29500 + RANDOM % 1000)) partzoom/main_partzoom.py \
  --tasks insert_onto_square_peg place_shape_in_shape_sorter slide_block_to_color_target \
  --dataset data/partzoom/packaged/train --valset data/partzoom/packaged/val \
  --instructions instructions/peract/instructions.pkl \
  --gripper_loc_bounds tasks/18_peract_tasks_location_bounds.json --gripper_loc_bounds_buffer 0.04 \
  --num_workers 2 --train_iters $ITERS --val_freq 1000 --val_iters 8 \
  --embedding_dim 120 --use_instruction 1 --rotation_parametrization 6D \
  --diffusion_timesteps 100 --dense_interpolation 1 --interpolation_length 2 \
  --batch_size 4 --batch_size_val 4 --cache_size 0 --cache_size_val 0 \
  --keypose_only 1 --variations $(seq -s ' ' 0 199) --lr 1e-4 --num_history 3 \
  --cameras left_shoulder right_shoulder wrist front --max_episodes_per_task -1 \
  --quaternion_format wxyz --seed 0 \
  --init_weights train_logs/diffuser_actor_peract.pth \
  --exp_log_dir partzoom --run_log_dir $NAME "$@" \
  > logs_partzoom/train_$NAME.log 2>&1
