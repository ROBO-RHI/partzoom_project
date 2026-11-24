# Extra tables (partzoom/extra_tables.py)

All computed from eval_logs/partzoom/*.jsonl and eval_logs/offline/*.json. '*' marks p < 0.05 (McNemar) or a 95% CI excluding 0 (bootstrap).

## paired_extra (closed-loop success, same 50 scenes per task)

| variant | reference | task | variant SR | reference SR | Δ (pts) | wins | losses | exact McNemar p |
|---|---|---|---|---|---|---|---|---|
| `both_lr2e5` | `baseline_lr2e5` | insert peg | 42 | 48 | -6.0 | 7 | 10 | 0.629 |
| `both_lr2e5` | `baseline_lr2e5` | sort shape | 50 | 42 | +8.0 | 10 | 6 | 0.454 |
| `both_lr2e5` | `baseline_lr2e5` | slide block | 98 | 94 | +4.0 | 3 | 1 | 0.625 |
| `both_lr2e5` | `baseline_lr2e5` | pooled | 63 | 61 | +2.0 | 20 | 17 | 0.743 |
| `both_lr1e5` | `baseline_lr1e5` | insert peg | 40 | 56 | -16.0 | 6 | 14 | 0.115 |
| `both_lr1e5` | `baseline_lr1e5` | sort shape | 46 | 44 | +2.0 | 4 | 3 | 1.000 |
| `both_lr1e5` | `baseline_lr1e5` | slide block | 98 | 94 | +4.0 | 3 | 1 | 0.625 |
| `both_lr1e5` | `baseline_lr1e5` | pooled | 61 | 65 | -3.3 | 13 | 18 | 0.473 |
| `baseline_lr1e5` | `baseline_lr2e5` | insert peg | 56 | 48 | +8.0 | 11 | 7 | 0.481 |
| `baseline_lr1e5` | `baseline_lr2e5` | sort shape | 44 | 42 | +2.0 | 7 | 6 | 1.000 |
| `baseline_lr1e5` | `baseline_lr2e5` | slide block | 94 | 94 | +0.0 | 3 | 3 | 1.000 |
| `baseline_lr1e5` | `baseline_lr2e5` | pooled | 65 | 61 | +3.3 | 21 | 16 | 0.511 |
| `both_lr1e5` | `released` | insert peg | 40 | 56 | -16.0 | 5 | 13 | 0.096 |
| `both_lr1e5` | `released` | sort shape | 46 | 44 | +2.0 | 5 | 4 | 1.000 |
| `both_lr1e5` | `released` | slide block | 98 | 96 | +2.0 | 2 | 1 | 1.000 |
| `both_lr1e5` | `released` | pooled | 61 | 65 | -4.0 | 12 | 18 | 0.362 |
| `baseline_lr2e5` | `baseline` | insert peg | 48 | 26 | +22.0 | 13 | 2 | 0.007 * |
| `baseline_lr2e5` | `baseline` | sort shape | 42 | 34 | +8.0 | 7 | 3 | 0.344 |
| `baseline_lr2e5` | `baseline` | slide block | 94 | 94 | +0.0 | 2 | 2 | 1.000 |
| `baseline_lr2e5` | `baseline` | pooled | 61 | 51 | +10.0 | 22 | 7 | 0.008 * |
| `both_lr2e5` | `both` | insert peg | 42 | 36 | +6.0 | 14 | 11 | 0.690 |
| `both_lr2e5` | `both` | sort shape | 50 | 38 | +12.0 | 10 | 4 | 0.180 |
| `both_lr2e5` | `both` | slide block | 98 | 96 | +2.0 | 2 | 1 | 1.000 |
| `both_lr2e5` | `both` | pooled | 63 | 57 | +6.7 | 26 | 16 | 0.164 |
| `zoom` | `zoom_object` | insert peg | 34 | 44 | -10.0 | 8 | 13 | 0.383 |
| `zoom` | `zoom_object` | sort shape | 34 | 38 | -4.0 | 7 | 9 | 0.804 |
| `zoom` | `zoom_object` | slide block | 96 | 92 | +4.0 | 4 | 2 | 0.688 |
| `zoom` | `zoom_object` | pooled | 55 | 58 | -3.3 | 19 | 24 | 0.542 |
| `zoom` | `zoom_random` | insert peg | 34 | 24 | +10.0 | 12 | 7 | 0.359 |
| `zoom` | `zoom_random` | sort shape | 34 | 26 | +8.0 | 9 | 5 | 0.424 |
| `zoom` | `zoom_random` | slide block | 96 | 96 | +0.0 | 2 | 2 | 1.000 |
| `zoom` | `zoom_random` | pooled | 55 | 49 | +6.0 | 23 | 14 | 0.188 |
| `zoom_object` | `zoom_random` | insert peg | 44 | 24 | +20.0 | 16 | 6 | 0.052 |
| `zoom_object` | `zoom_random` | sort shape | 38 | 26 | +12.0 | 8 | 2 | 0.109 |
| `zoom_object` | `zoom_random` | slide block | 92 | 96 | -4.0 | 1 | 3 | 0.625 |
| `zoom_object` | `zoom_random` | pooled | 58 | 49 | +9.3 | 25 | 11 | 0.029 * |

## tokens_offline (val keyframes, 2 seeds; FPS runs are given part regions only for these statistics)

| offline run | token mode / zoom / regions | in-region coarse available | kept | retention | share of 819 budget on parts | in-region fine kept | crop magnification | crops with part visible |
|---|---|---|---|---|---|---|---|---|
| `released` | fps / – / part | 217 | 91 | 60% | 11% | – | – | – |
| `baseline` | fps / – / part | 217 | 91 | 60% | 11% | – | – | – |
| `baseline_lr2e5` | fps / – / part | 217 | 91 | 60% | 11% | – | – | – |
| `part` | part / – / part | 217 | 198 | 97% | 24% | – | – | – |
| `released_part` | part / – / part | 217 | 198 | 97% | 24% | – | – | – |
| `zoom` | fps / zoom / part | 217 | 78 | 54% | 10% | 251 | 2.53× | 81% |
| `released_zoom` | fps / zoom / part | 217 | 69 | 49% | 8% | 251 | 2.53× | 81% |
| `both` | part / zoom / part | 217 | 169 | 90% | 21% | 251 | 2.53× | 81% |
| `both_lr2e5` | part / zoom / part | 217 | 169 | 90% | 21% | 251 | 2.53× | 81% |
| `zoom_object` | fps / zoom / object | 1085 | 278 | 33% | 34% | 256 | 1.69× | 89% |
| `zoom_random` | fps / zoom / random | 30 | 4 | 12% | 1% | 155 | 1.68× | 41% |
| `baseline_lr1e5` | fps / – / part | 217 | 89 | 59% | 11% | – | – | – |
| `both_lr1e5` | part / zoom / part | 217 | 169 | 90% | 21% | 251 | 2.53× | 81% |

## offline_paired_bootstrap (difference in position error, paired by keyframe, CI by resampling val episodes)

| variant | reference | Δ pos err cm, all kf [95% CI] | Δ pos err cm, interaction kf [95% CI] | episodes |
|---|---|---|---|---|
| `part` | `baseline` | +0.02 [-0.12, +0.14] | -0.05 [-0.30, +0.16] | 60 |
| `zoom` | `baseline` | -0.09 [-0.31, +0.05] | -0.17 [-0.55, +0.06] | 60 |
| `both` | `baseline` | -0.12 [-0.35, +0.00] | -0.23 [-0.60, -0.01] * | 60 |
| `both_lr2e5` | `baseline_lr2e5` | -0.05 [-0.23, +0.09] | -0.16 [-0.43, +0.05] | 60 |
| `baseline` | `released` | +0.04 [-0.08, +0.14] | -0.01 [-0.22, +0.15] | 60 |
| `baseline_lr2e5` | `released` | -0.06 [-0.21, +0.09] | -0.09 [-0.36, +0.15] | 60 |
| `both_lr2e5` | `released` | -0.11 [-0.31, +0.06] | -0.25 [-0.57, -0.00] * | 60 |
| `released_part` | `released` | +0.03 [+0.02, +0.05] * | +0.04 [+0.02, +0.07] * | 60 |
| `released_zoom` | `released` | +0.03 [-0.07, +0.09] | +0.00 [-0.16, +0.10] | 60 |
| `zoom` | `zoom_object` | -0.03 [-0.12, +0.04] | -0.02 [-0.13, +0.06] | 60 |
| `zoom` | `zoom_random` | -0.18 [-0.33, -0.06] * | -0.31 [-0.57, -0.10] * | 60 |
| `both_lr1e5` | `baseline_lr1e5` | -0.04 [-0.13, +0.04] | -0.07 [-0.22, +0.06] | 60 |
| `baseline_lr1e5` | `released` | -0.11 [-0.26, +0.01] | -0.18 [-0.42, +0.02] | 60 |
| `both_lr1e5` | `released` | -0.15 [-0.34, -0.01] * | -0.25 [-0.56, -0.01] * | 60 |

## sweeps (eval-time only: models trained with default settings; offline val errors)

### localisation noise (std of a per-region 3D box shift, m)

| model | setting | pos err cm (all kf) | rot err ° (all kf) | pos err cm (interaction kf) | rot err ° (interaction kf) | insert-peg pos err cm (interaction kf) | pos < 1 cm | retention of in-region coarse | crops with part visible |
|---|---|---|---|---|---|---|---|---|---|
| `zoom` | 0 (default) | 0.39 | 15.7 | 0.40 | 3.3 | 0.30 | 96% | 54% | 81% |
| `zoom` | 0.01 | 0.40 | 15.7 | 0.42 | 3.3 | 0.31 | 96% | 54% | 79% |
| `zoom` | 0.02 | 0.44 | 15.8 | 0.48 | 3.6 | 0.42 | 95% | 55% | 73% |
| `zoom` | 0.04 | 0.53 | 16.2 | 0.63 | 3.8 | 0.62 | 93% | 54% | 57% |
| `part` | 0 (default) | 0.50 | 14.9 | 0.52 | 3.3 | 0.60 | 93% | 97% | – |
| `part` | 0.01 | 0.50 | 15.0 | 0.52 | 3.3 | 0.59 | 93% | 97% | – |
| `part` | 0.02 | 0.50 | 15.0 | 0.52 | 3.4 | 0.60 | 93% | 98% | – |
| `part` | 0.04 | 0.52 | 15.1 | 0.55 | 3.6 | 0.67 | 93% | 95% | – |
| `both` | 0 (default) | 0.35 | 15.1 | 0.35 | 3.6 | 0.23 | 95% | 90% | 81% |
| `both` | 0.01 | 0.37 | 15.4 | 0.37 | 4.0 | 0.24 | 96% | 93% | 79% |
| `both` | 0.02 | 0.40 | 15.0 | 0.42 | 4.0 | 0.28 | 95% | 95% | 73% |
| `both` | 0.04 | 0.46 | 16.3 | 0.52 | 4.3 | 0.34 | 94% | 93% | 57% |

### crop margin (fraction of box size added around the part)

| model | setting | pos err cm (all kf) | rot err ° (all kf) | pos err cm (interaction kf) | rot err ° (interaction kf) | insert-peg pos err cm (interaction kf) | pos < 1 cm | retention of in-region coarse | crops with part visible |
|---|---|---|---|---|---|---|---|---|---|
| `zoom` | 0.0 | 0.42 | 15.8 | 0.44 | 3.6 | 0.41 | 95% | 54% | 81% |
| `zoom` | 0.1 | 0.39 | 15.5 | 0.40 | 3.3 | 0.31 | 96% | 54% | 81% |
| `zoom` | 0.25 (default) | 0.39 | 15.7 | 0.40 | 3.3 | 0.30 | 96% | 54% | 81% |
| `zoom` | 0.5 | 0.41 | 15.7 | 0.43 | 3.3 | 0.30 | 95% | 54% | 81% |
| `zoom` | 1.0 | 0.44 | 15.6 | 0.48 | 3.4 | 0.40 | 94% | 54% | 81% |

### zoom token budget (fine tokens out of 819)

| model | setting | pos err cm (all kf) | rot err ° (all kf) | pos err cm (interaction kf) | rot err ° (interaction kf) | insert-peg pos err cm (interaction kf) | pos < 1 cm | retention of in-region coarse | crops with part visible |
|---|---|---|---|---|---|---|---|---|---|
| `zoom` | 128 | 0.38 | 15.7 | 0.40 | 3.5 | 0.30 | 96% | 59% | 81% |
| `zoom` | 256 (default) | 0.39 | 15.7 | 0.40 | 3.3 | 0.30 | 96% | 54% | 81% |
| `zoom` | 384 | 0.40 | 16.3 | 0.41 | 3.6 | 0.30 | 96% | 47% | 81% |

### rel_cap (max share of budget for in-region coarse tokens)

| model | setting | pos err cm (all kf) | rot err ° (all kf) | pos err cm (interaction kf) | rot err ° (interaction kf) | insert-peg pos err cm (interaction kf) | pos < 1 cm | retention of in-region coarse | crops with part visible |
|---|---|---|---|---|---|---|---|---|---|
| `part` | 0.25 | 0.50 | 15.0 | 0.51 | 3.3 | 0.60 | 93% | 84% | – |
| `part` | 0.5 (default) | 0.50 | 14.9 | 0.52 | 3.3 | 0.60 | 93% | 97% | – |
| `part` | 0.75 | 0.50 | 14.9 | 0.52 | 3.2 | 0.60 | 93% | 99% | – |

## bench (clean benchmark, idle T4, batch 1, released weights, 100 denoising steps per step)

| configuration | s/step mean | s/step min | peak GPU memory allocated (MiB) | vs FPS |
|---|---|---|---|---|
| fps | 1.86 | 1.84 | 352 | +0% |
| part | 2.03 | 2.01 | 352 | +9% |
| zoom | 1.88 | 1.87 | 352 | +1% |
| both | 2.01 | 1.99 | 352 | +8% |
