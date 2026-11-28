# Part-aware token sampling and part zoom for 3D Diffuser Actor

This folder holds the experiments for the 16-831 project proposal *"Part-aware sampling and part zoom for 3D Diffuser Actor"* (Harshvardhan Aditya, Ishita Srivastava, Raghav Mehta). Everything needed to read, check or rerun the results is here.

- **Results:** [RESULTS.md](RESULTS.md). **Status: final after round 3 (2026-11-24).** Rounds 1–2: 11 variants × 3 tasks × 50 closed-loop episodes, the offline evaluations, the eval-time sweeps and the inference benchmark. Round 3: 100 test scenes per task for six variants, including a new matched pair at lr 1e-5. Headline: **nothing beats the released checkpoint.** On 100 scenes it scores 63% (189/300), and every configuration tested is within 2 points of it (all p ≥ 0.60). **Part sampling + zoom does not beat its matched baseline:** −1.7 (p = 0.64) at lr 2e-5 and −3.0 (p = 0.33) at lr 1e-5 on 100 scenes, with insert peg consistently worse (−13 and −17; the latter p = 0.009 uncorrected, not significant after multiple-comparison correction). The +3 to +5 point gains at lr 1e-4 (50 scenes, n.s.) most likely reflect a poorly tuned fine-tuning recipe.
- **How the data was made:** [DATASET.md](DATASET.md)

## What was tested

The base model is **3D Diffuser Actor (3DDA)** with its released 18-task PerAct checkpoint. The diffusion head, the losses and the cross-attention of trajectory queries to all 4096 scene tokens are unchanged. The only thing that changes is **which 819 visual tokens** (N/5, the fixed budget) the trajectory tokens see in the relative self-attention layers. Upstream picks them by farthest-point sampling (FPS) in feature space.

| variant | tokens in the 819 budget |
|---|---|
| FPS (upstream) | 819 coarse tokens by FPS |
| **part-aware sampling** | coarse tokens inside the oracle part regions first (capped at 50% of the budget, FPS within the region), the rest by FPS over the scene |
| **part zoom** | 563 coarse tokens by FPS + 256 **fine** tokens. Fine tokens come from re-encoding 128 × 128 RGB + point-cloud crops around each part in every camera (2 regions × 4 cameras), are lifted into the same 3D frame, and in-region fine tokens are kept first |
| **both** | part-aware coarse selection (563) + 256 fine zoom tokens |
| whole-object zoom | zoom around the whole object (e.g. the full peg base) instead of the part |
| random-region zoom | zoom crops with the same box sizes at random workspace locations |

The total token budget is identical (819) for every variant. Regions are **oracle (privileged) simulator information**; see DATASET.md §5.

Each method was fine-tuned from the released checkpoint with the same recipe. FPS, part, zoom, both and the two zoom controls were trained at lr 1e-4. Matched FPS/both pairs were also trained at lr 2e-5 (`baseline_lr2e5`, `both_lr2e5`) and lr 1e-5 (`baseline_lr1e5`, `both_lr1e5`; round 3). The released weights were also run with part sampling or zoom applied zero-shot. The full list of runs is in RESULTS.md §1.

## Visuals

All images and videos are in [`figs/visuals/`](figs/visuals). Full-resolution MP4s for slides are in [`figs/visuals/videos/`](figs/visuals/videos). Part regions in every visual are **oracle (privileged) simulator information**.

### Where the token budget goes

Each method keeps 819 visual tokens for the trajectory's self-attention. Upstream FPS spreads them across the whole scene. Part-aware sampling and part zoom concentrate them on the manipulated part and the receiving part, which is where the end effector interacts.

![token selection, insert peg](figs/visuals/tokens_insert_onto_square_peg.png)

*Held-out insert-peg demo. Columns: scene with oracle part regions (cyan), then the tokens kept by upstream FPS, part-aware sampling, and part-aware + zoom (dots; heat = token density; light-cyan dots = fine zoom tokens), then the two 128-px zoom crops per camera. Same figure for the other tasks: [sort shape](figs/visuals/tokens_place_shape_in_shape_sorter.png), [slide block](figs/visuals/tokens_slide_block_to_color_target.png).*

| upstream FPS vs ours, over one demo (front and wrist cameras) | |
|---|---|
| ![](figs/visuals/token_focus_insert_onto_square_peg.gif) | ![](figs/visuals/token_focus_place_shape_in_shape_sorter.gif) |

### How close the tokens are to the end effector

![token distance](figs/visuals/token_distance.png)

Measured over 260 held-out keyframes from the 3 tasks:

| tokens | median distance to end effector | within 5 cm | within 10 cm | inside oracle parts |
|---|---|---|---|---|
| all 4096 scene tokens (reference) | 54.0 cm | 4.8% | 10.6% | 5.1% |
| upstream FPS (819) | 29.9 cm | 10.3% | 17.5% | 10.8% |
| part-aware sampling (819) | 27.2 cm | 15.9% | 26.0% | 23.2% |
| **part-aware + part zoom (819)** | **22.3 cm** | **20.3%** | **31.4%** | **50.8%** |

Raw numbers are in [`figs/visuals/token_distance.json`](figs/visuals/token_distance.json).

### Closed-loop rollouts: matched baseline vs ours on the same test scene

Each video runs `baseline_lr1e5` (upstream FPS) and `both_lr1e5` (part-aware + zoom) on the same held-out scene, with the same seed. The insets show the tokens each policy used at its latest decision (white ring = predicted next keypose). The policy sees only its usual four 256-px cameras; the large view comes from an extra camera added only for the video.

| ours succeeds, baseline fails (insert peg, scene 8) |
|---|
| ![](figs/visuals/rollout_insert_onto_square_peg_ep8_fail_ok.gif) |
| **ours succeeds, baseline fails (sort shape, scene 27)** |
| ![](figs/visuals/rollout_place_shape_in_shape_sorter_ep27_fail_ok.gif) |
| **baseline succeeds, ours fails (insert peg, scene 4)** |
| ![](figs/visuals/rollout_insert_onto_square_peg_ep4_ok_fail.gif) |

These clips are **selected examples, not a summary of performance**. Over 100 scenes per task the two models are statistically tied (RESULTS.md, Round 3), and insert peg is where ours does worse. [`figs/visuals/videos/`](figs/visuals/videos) has 18 rollouts across all three tasks, covering wins, losses and ties for both models. Each video is a fresh re-run; on borderline scenes the outcome can differ from the logged evaluation, because of GPU-level numerical noise. File names end in `<baseline>_<ours>` outcome (`ok`/`fail`) for that run.

## Folder layout

```
README.md  RESULTS.md  DATASET.md
results/
  tables.md            auto-generated tables, 50 test scenes per task (make_package.py)
  tables_extra.md      lr-2e-5 / lr-1e-5 / zoom-control paired tests, offline token retention,
                       bootstrap CIs, eval-time sweeps, benchmark (extra_tables.py)
  tables_100scenes.md  round 3: 100 test scenes per task + paired McNemar tests (ext_tables.py)
                       (all numbers in RESULTS.md come from these three files)
  summary.json         machine-readable success counts + validation curves
  rollouts/<variant>/<task>.jsonl   one line per test episode: success, steps, failure type,
                                    per-step predicted keyposes, token statistics, inference time
  offline/<name>.json  per-keyframe val errors for every variant and sweep; bench_infer.json
checkpoints/released/diffuser_actor_peract.pth   released 3DDA PerAct checkpoint (git LFS, 161 MB)
checkpoints/<variant>/weights_delta.pth          every fine-tuned model as a compact delta (git LFS, ~12 MB)
checkpoints/<variant>/hparams.json               full training configuration and command line
logs/                  training logs (val metrics every 1000 it), eval logs, generation logs,
                       pipeline.log (timeline), tensorboard/
figs/regions_peg_ep0.png   oracle regions and zoom crops, checked visually
figs/success_rates.png     success per task and variant with Wilson CIs (make_fig.py)
figs/visuals/              token-selection figures, token-distance plot, GIFs; videos/ = MP4 rollouts for slides
data/seeds/            exact episode definitions for train/val/test (reproduces the test scenes)
data/packaged -> symlink to the 6.2 GB of packaged train/val demos
code/                  runnable snapshot: upstream 3DDA modules + partzoom/ (our code)
```

### Our code (`code/partzoom/`)

| file | purpose |
|---|---|
| `regions.py` | oracle part/object boxes from the simulator, and the observation hook |
| `generate_demos.py` | demo generation in 3DDA's packaged format, plus regions and seed files |
| `model.py` | `PartZoomDiffuserActor`: token selection, crops, fine-token encoding, statistics |
| `main_partzoom.py` | fine-tuning entry point (dataset with regions, random-region control) |
| `eval_rollouts.py` | closed-loop RLBench evaluation (mirrors upstream `evaluate_policy.py`) |
| `offline_eval.py` | per-keyframe position and rotation errors on val demos |
| `analyze.py`, `make_package.py` | tables, Wilson CIs, exact McNemar tests; rebuilds this folder |
| `extra_tables.py`, `make_fig.py` | extra tables (`results/tables_extra.md`) and `figs/success_rates.png`; run after `make_package.py` |
| `ext_tables.py` | round-3 100-scene table and paired tests (`results/tables_100scenes.md`) |
| `bench_infer.py` | clean inference-time and memory benchmark |
| `train_variant.sh`, `run_eval.sh`, `pipeline*.sh`, `gen_all.sh` | the exact commands that were run |
| `tmux/r3_train.sh`, `tmux/r3_eval.sh`, `tmux/r3_final.sh` | round 3: lr-1e-5 training, 100-scene evals, offline evals + tables |

## Rerunning

```bash
cd code && source partzoom/env.sh       # conda env 3d_diffuser_actor, CoppeliaSim 4.1, xvfb
# fine-tune one variant (4000 it, batch 4, ~2.2 h on a T4):
bash partzoom/train_variant.sh zoom 4000 --zoom 1
# closed-loop eval on the 50 fixed test scenes:
bash partzoom/run_eval.sh zoom train_logs/partzoom/zoom/last.pth insert_onto_square_peg 0:50 --max_steps 12 --zoom 1
# tables:
python partzoom/make_package.py --no_ckpt --out ..
python partzoom/extra_tables.py --out .. && python partzoom/ext_tables.py --out .. && python partzoom/make_fig.py --out ..
```

The flags for every variant, at training and evaluation, are listed in `code/partzoom/pipeline.sh`, `pipeline2.sh` and `tmux/eval_worker.sh`; the offline evaluations, sweeps and benchmark are in `pipeline3.sh`. Each training run's full command line is saved in `checkpoints/<variant>/hparams.json` under `reproducibility.command_line`.

Environment: AWS instance, one NVIDIA T4 (16 GB), 4 vCPUs, 15 GB RAM. Software: conda env `3d_diffuser_actor` (PyTorch with CUDA, DGL 1.1.2 cu116, CLIP), CoppeliaSim 4.1, and the PyRep and RLBench PerAct forks installed in `~/PyRep` and `~/RLBench`.

## Checkpoints (Git LFS)

The checkpoints are stored with [Git LFS](https://git-lfs.com). After cloning, run `git lfs install && git lfs pull`, or install git-lfs before cloning.

- `checkpoints/released/diffuser_actor_peract.pth` is the released 3D Diffuser Actor PerAct checkpoint, unchanged. Every fine-tuned model starts from it.
- `checkpoints/<variant>/weights_delta.pth` holds **only the tensors that changed during fine-tuning** (about 11 MB of the 160 MB of weights; the frozen CLIP conv weights are identical) plus the SHA-256 of the released checkpoint. On load it is overlaid on the released checkpoint, which rebuilds the exact full weights. Every delta was checked to rebuild a state dict bit-for-bit identical to the original `last.pth`. Optimizer state is not included, so it is enough for evaluation and further fine-tuning but not for resuming the exact training run.
- `code/partzoom/model.py: load_released_weights()` accepts either format, so all scripts take a delta path directly. For example: `run_eval.sh both_lr1e5 ../checkpoints/both_lr1e5/weights_delta.pth insert_onto_square_peg 0:100 --max_steps 12 --token_mode part --zoom 1`. To write out a full checkpoint instead:
  ```python
  from partzoom.delta_ckpt import load_full_state_dict
  import torch; torch.save({"weight": load_full_state_dict("checkpoints/zoom/weights_delta.pth")}, "zoom_full.pth")
  ```
