# Part-aware token sampling and part zoom for 3D Diffuser Actor

This folder holds the experiments for the 16-831 project proposal *"Part-aware sampling and part zoom for 3D Diffuser Actor"* (Harshvardhan Aditya, Ishita Srivastava, Raghav Mehta). Everything needed to read, check or rerun the results is here.

- **Results:** [RESULTS.md](RESULTS.md). **Status: final (2026-11-23).** All 11 variants × 3 tasks × 50 closed-loop episodes, the offline evaluations, the eval-time sweeps and the inference benchmark are complete. Headline: the released checkpoint remains best (65% pooled). The best fine-tuned model is part sampling + zoom at lr 2e-5 (63%, n.s. vs released, +2 n.s. vs its matched baseline). At lr 1e-4 the methods are +3 to +5 points over the matched baseline, also not significant.
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

## Folder layout

```
README.md  RESULTS.md  DATASET.md
results/
  tables.md            auto-generated tables (make_package.py)
  tables_extra.md      lr-2e-5 / zoom-control paired tests, offline token retention,
                       bootstrap CIs, eval-time sweeps, benchmark (extra_tables.py)
                       (all numbers in RESULTS.md come from these two files)
  summary.json         machine-readable success counts + validation curves
  rollouts/<variant>/<task>.jsonl   one line per test episode: success, steps, failure type,
                                    per-step predicted keyposes, token statistics, inference time
  offline/<name>.json  per-keyframe val errors for every variant and sweep; bench_infer.json
checkpoints/<variant>/last.pth (+ hparams.json)   all fine-tuned models + released checkpoint
logs/                  training logs (val metrics every 1000 it), eval logs, generation logs,
                       pipeline.log (timeline), tensorboard/
figs/regions_peg_ep0.png   oracle regions and zoom crops, checked visually
figs/success_rates.png     success per task and variant with Wilson CIs (make_fig.py)
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
| `bench_infer.py` | clean inference-time and memory benchmark |
| `train_variant.sh`, `run_eval.sh`, `pipeline*.sh`, `gen_all.sh` | the exact commands that were run |

## Rerunning

```bash
cd code && source partzoom/env.sh       # conda env 3d_diffuser_actor, CoppeliaSim 4.1, xvfb
# fine-tune one variant (4000 it, batch 4, ~2.2 h on a T4):
bash partzoom/train_variant.sh zoom 4000 --zoom 1
# closed-loop eval on the 50 fixed test scenes:
bash partzoom/run_eval.sh zoom train_logs/partzoom/zoom/last.pth insert_onto_square_peg 0:50 --max_steps 12 --zoom 1
# tables:
python partzoom/make_package.py --no_ckpt --out ..
python partzoom/extra_tables.py --out .. && python partzoom/make_fig.py --out ..
```

The flags for every variant, at training and evaluation, are listed in `code/partzoom/pipeline.sh`, `pipeline2.sh` and `tmux/eval_worker.sh`; the offline evaluations, sweeps and benchmark are in `pipeline3.sh`. Each training run's full command line is saved in `checkpoints/<variant>/hparams.json` under `reproducibility.command_line`.

Environment: AWS instance, one NVIDIA T4 (16 GB), 4 vCPUs, 15 GB RAM. Software: conda env `3d_diffuser_actor` (PyTorch with CUDA, DGL 1.1.2 cu116, CLIP), CoppeliaSim 4.1, and the PyRep and RLBench PerAct forks installed in `~/PyRep` and `~/RLBench`.
