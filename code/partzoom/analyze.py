"""Aggregate rollout and offline results into tables (markdown + json)."""
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = ["insert_onto_square_peg", "place_shape_in_shape_sorter",
         "slide_block_to_color_target"]
SHORT = {"insert_onto_square_peg": "insert peg",
         "place_shape_in_shape_sorter": "sort shape",
         "slide_block_to_color_target": "slide block"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load_rollouts(root, step_cap=None):
    out = {}
    for d in sorted(glob.glob(f"{root}/*")):
        name = Path(d).name
        recs = {}
        for f in glob.glob(f"{d}/*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                if step_cap is not None:
                    r["success"] = r["success"] and r["steps"] <= step_cap
                recs[(r["task"], r["episode"])] = r
        out[name] = recs
    return out


def rollout_table(runs, names, ref=None):
    lines = ["| variant | " + " | ".join(SHORT[t] for t in TASKS) +
             " | mean | infer s/step |", "|" + "---|" * (len(TASKS) + 3)]
    summary = {}
    for n in names:
        if n not in runs:
            continue
        recs = runs[n]
        cells, srs = [], []
        for t in TASKS:
            rs = [r for (tt, _), r in recs.items() if tt == t]
            k = sum(r["success"] for r in rs)
            lo, hi = wilson(k, len(rs))
            srs.append(k / max(len(rs), 1))
            cells.append(f"{100*k/max(len(rs),1):.0f} ({k}/{len(rs)}) "
                         f"[{100*lo:.0f}–{100*hi:.0f}]" if rs else "–")
            summary.setdefault(n, {})[t] = {"k": k, "n": len(rs)}
        allr = list(recs.values())
        it = np.mean([r["infer_time_mean"] for r in allr]) if allr else float("nan")
        lines.append(f"| {n} | " + " | ".join(cells) +
                     f" | {100*np.mean(srs):.1f} | {it:.2f} |")
    return "\n".join(lines), summary


def paired_table(runs, names, ref):
    lines = ["| variant vs " + ref + " | task | both ✓ | only variant ✓ | "
             "only ref ✓ | both ✗ | Δ SR (pts) | McNemar p |",
             "|---|---|---|---|---|---|---|---|"]
    for n in names:
        if n == ref or n not in runs or ref not in runs:
            continue
        for t in TASKS + ["all"]:
            keys = [k for k in runs[n] if k in runs[ref] and (t == "all" or k[0] == t)]
            if not keys:
                continue
            a = np.array([runs[n][k]["success"] for k in keys])
            b = np.array([runs[ref][k]["success"] for k in keys])
            b10 = int((a & ~b).sum()); b01 = int((~a & b).sum())
            lines.append(f"| {n} | {SHORT.get(t, t)} | {int((a & b).sum())} | "
                         f"{b10} | {b01} | {int((~a & ~b).sum())} | "
                         f"{100*(a.mean()-b.mean()):+.1f} | "
                         f"{mcnemar_exact(b10, b01):.3f} |")
    return "\n".join(lines)


def fail_table(runs, names):
    lines = ["| variant | success | timeout (wrong poses) | planner error |",
             "|---|---|---|---|"]
    for n in names:
        if n not in runs:
            continue
        c = defaultdict(int)
        for r in runs[n].values():
            ft = r["fail_type"]
            c["planning" if ft.startswith("planning") else ft] += 1
        tot = sum(c.values())
        lines.append(f"| {n} | {c['success']}/{tot} | {c['timeout']} | "
                     f"{c['planning']} |")
    return "\n".join(lines)


def token_table(runs, names):
    keys = ["coarse_rel_tokens", "coarse_rel_kept", "coarse_rel_retention",
            "fine_rel_kept", "crop_zoom", "crop_valid_frac"]
    lines = ["| variant | " + " | ".join(keys) + " |",
             "|" + "---|" * (len(keys) + 1)]
    for n in names:
        if n not in runs:
            continue
        vals = []
        for k in keys:
            v = [r["stats_mean"].get(k) for r in runs[n].values()
                 if r["stats_mean"].get(k) is not None]
            vals.append(f"{np.mean(v):.2f}" if v else "–")
        lines.append(f"| {n} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def offline_table(root, names):
    lines = ["| variant | task | pos err cm (all kf) | rot err deg (all kf) | "
             "pos err cm (interaction kf) | rot err deg (interaction kf) | "
             "pos <1cm % | gripper acc % |", "|---|---|---|---|---|---|---|---|"]
    res = {}
    for n in names:
        f = Path(root) / f"{n}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        rows = d["rows"]
        for t in TASKS + ["all"]:
            rs = [r for r in rows if t == "all" or r["task"] == t]
            if not rs:
                continue
            # interaction keyframes: second half of each episode (place/insert)
            inter = [r for r in rs if r["kf"] >= r["T"] // 2]
            row = {
                "pos": np.mean([r["pos_cm"] for r in rs]),
                "rot": np.mean([r["rot_deg"] for r in rs]),
                "pos_i": np.mean([r["pos_cm"] for r in inter]),
                "rot_i": np.mean([r["rot_deg"] for r in inter]),
                "acc1": 100 * np.mean([r["pos_cm"] < 1 for r in rs]),
                "grip": 100 * np.mean([r["grip_ok"] for r in rs]),
            }
            res[(n, t)] = row
            lines.append(f"| {n} | {SHORT.get(t, t)} | {row['pos']:.2f} | "
                         f"{row['rot']:.2f} | {row['pos_i']:.2f} | "
                         f"{row['rot_i']:.2f} | {row['acc1']:.1f} | "
                         f"{row['grip']:.1f} |")
    return "\n".join(lines), res


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "eval_logs/partzoom"
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    runs = load_rollouts(root)
    names = names or sorted(runs)
    print(rollout_table(runs, names)[0])
