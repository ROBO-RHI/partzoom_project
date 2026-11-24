# Closed-loop results on the extended test set (100 scenes per task, episodes 0–99)

Success % (successes/episodes) [Wilson 95% CI], max 12 steps. Only variants with all 300 episodes are listed. Episodes 0–49 are the same scenes as the main tables.

| variant | insert peg | sort shape | slide block | pooled (300) |
|---|---|---|---|---|
| `released` | **51** (51/100) [41–61] | **40** (40/100) [31–50] | **98** (98/100) [93–99] | **63** (189/300) [57–68] |
| `released_part` | **51** (51/100) [41–61] | **39** (39/100) [30–49] | **100** (100/100) [96–100] | **63** (190/300) [58–69] |
| `baseline_lr2e5` | **53** (53/100) [43–62] | **41** (41/100) [32–51] | **95** (95/100) [89–98] | **63** (189/300) [57–68] |
| `both_lr2e5` | **40** (40/100) [31–50] | **46** (46/100) [37–56] | **98** (98/100) [93–99] | **61** (184/300) [56–67] |
| `baseline_lr1e5` | **59** (59/100) [49–68] | **38** (38/100) [29–48] | **97** (97/100) [92–99] | **65** (194/300) [59–70] |
| `both_lr1e5` | **42** (42/100) [33–52] | **45** (45/100) [36–55] | **98** (98/100) [93–99] | **62** (185/300) [56–67] |

## Paired comparisons (exact two-sided McNemar, same 100 scenes per task)

| variant | reference | task | Δ SR (pts) | wins | losses | p |
|---|---|---|---|---|---|---|
| `both_lr2e5` | `baseline_lr2e5` | insert peg | -13.0 | 14 | 27 | 0.060 |
| `both_lr2e5` | `baseline_lr2e5` | sort shape | +5.0 | 16 | 11 | 0.442 |
| `both_lr2e5` | `baseline_lr2e5` | slide block | +3.0 | 5 | 2 | 0.453 |
| `both_lr2e5` | `baseline_lr2e5` | pooled | -1.7 | 35 | 40 | 0.644 |
| `both_lr1e5` | `baseline_lr1e5` | insert peg | -17.0 | 11 | 28 | 0.009 * |
| `both_lr1e5` | `baseline_lr1e5` | sort shape | +7.0 | 15 | 8 | 0.210 |
| `both_lr1e5` | `baseline_lr1e5` | slide block | +1.0 | 3 | 2 | 1.000 |
| `both_lr1e5` | `baseline_lr1e5` | pooled | -3.0 | 29 | 38 | 0.328 |
| `both_lr2e5` | `released` | insert peg | -11.0 | 15 | 26 | 0.117 |
| `both_lr2e5` | `released` | sort shape | +6.0 | 17 | 11 | 0.345 |
| `both_lr2e5` | `released` | slide block | +0.0 | 2 | 2 | 1.000 |
| `both_lr2e5` | `released` | pooled | -1.7 | 34 | 39 | 0.640 |
| `both_lr1e5` | `released` | insert peg | -9.0 | 17 | 26 | 0.222 |
| `both_lr1e5` | `released` | sort shape | +5.0 | 13 | 8 | 0.383 |
| `both_lr1e5` | `released` | slide block | +0.0 | 2 | 2 | 1.000 |
| `both_lr1e5` | `released` | pooled | -1.3 | 32 | 36 | 0.716 |
| `baseline_lr2e5` | `released` | insert peg | +2.0 | 15 | 13 | 0.851 |
| `baseline_lr2e5` | `released` | sort shape | +1.0 | 12 | 11 | 1.000 |
| `baseline_lr2e5` | `released` | slide block | -3.0 | 1 | 4 | 0.375 |
| `baseline_lr2e5` | `released` | pooled | +0.0 | 28 | 28 | 1.000 |
| `baseline_lr1e5` | `released` | insert peg | +8.0 | 22 | 14 | 0.243 |
| `baseline_lr1e5` | `released` | sort shape | -2.0 | 9 | 11 | 0.824 |
| `baseline_lr1e5` | `released` | slide block | -1.0 | 1 | 2 | 1.000 |
| `baseline_lr1e5` | `released` | pooled | +1.7 | 32 | 27 | 0.603 |
| `released_part` | `released` | insert peg | +0.0 | 10 | 10 | 1.000 |
| `released_part` | `released` | sort shape | -1.0 | 8 | 9 | 1.000 |
| `released_part` | `released` | slide block | +2.0 | 2 | 0 | 0.500 |
| `released_part` | `released` | pooled | +0.3 | 20 | 19 | 1.000 |
| `both_lr1e5` | `both_lr2e5` | insert peg | +2.0 | 16 | 14 | 0.856 |
| `both_lr1e5` | `both_lr2e5` | sort shape | -1.0 | 12 | 13 | 1.000 |
| `both_lr1e5` | `both_lr2e5` | slide block | +0.0 | 1 | 1 | 1.000 |
| `both_lr1e5` | `both_lr2e5` | pooled | +0.3 | 29 | 28 | 1.000 |
| `baseline_lr1e5` | `baseline_lr2e5` | insert peg | +6.0 | 17 | 11 | 0.345 |
| `baseline_lr1e5` | `baseline_lr2e5` | sort shape | -3.0 | 10 | 13 | 0.678 |
| `baseline_lr1e5` | `baseline_lr2e5` | slide block | +2.0 | 5 | 3 | 0.727 |
| `baseline_lr1e5` | `baseline_lr2e5` | pooled | +1.7 | 32 | 27 | 0.603 |