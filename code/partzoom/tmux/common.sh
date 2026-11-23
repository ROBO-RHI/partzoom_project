# shared helpers for the tmux job windows
cd ~/3d_diffuser_actor && source partzoom/env.sh
LOG=logs_partzoom/pipeline.log
TASKS="insert_onto_square_peg place_shape_in_shape_sorter slide_block_to_color_target"
mark() { echo "$* $(date)" | tee -a $LOG; }
has() { grep -q "^$1 " $LOG; }
