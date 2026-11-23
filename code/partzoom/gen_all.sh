#!/bin/bash
# Full data generation: 3 parallel workers. Seeds: train 1000, val 2000, test 3000.
cd "$(dirname "$0")/.." && source partzoom/env.sh
mkdir -p logs_partzoom
G="xvfb-run -a -s '-screen 0 1024x768x24' python partzoom/generate_demos.py --out data/partzoom"
gen() { eval $G "$@"; }
(
  for t in insert_onto_square_peg place_shape_in_shape_sorter slide_block_to_color_target; do
    gen --task $t --split test --num 100 --seed_base 3000 --no_images
  done
  gen --task slide_block_to_color_target --split train --num 100 --seed_base 1000
  gen --task slide_block_to_color_target --split val --num 20 --seed_base 2000
) > logs_partzoom/gen_w1.log 2>&1 &
(
  gen --task insert_onto_square_peg --split train --num 100 --seed_base 1000
  gen --task insert_onto_square_peg --split val --num 20 --seed_base 2000
) > logs_partzoom/gen_w2.log 2>&1 &
(
  gen --task place_shape_in_shape_sorter --split train --num 100 --seed_base 1000
  gen --task place_shape_in_shape_sorter --split val --num 20 --seed_base 2000
) > logs_partzoom/gen_w3.log 2>&1 &
wait
echo GEN_ALL_DONE
