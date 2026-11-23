"""Assemble the self-contained project folder and regenerate all result tables.

usage: python partzoom/make_package.py [--out ~/partzoom_project] [--no_ckpt]

Re-running is safe: files are re-copied/overwritten, tables regenerated from
the raw per-episode logs. Hand-written docs (README/RESULTS/DATASET.md) are
not touched; tables land in results/tables.md for pasting/reference.
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "partzoom"))
from analyze import (TASKS, SHORT, wilson, mcnemar_exact, load_rollouts)  # noqa

STEP_CAP = 12  # all variants evaluated with max_steps=12 (released peg ran 25;
               # its successes all occurred by step 6, so capping changes nothing)

ORDER = ["released", "released_part", "released_zoom",
         "baseline", "part", "zoom", "both",
         "baseline_lr2e5", "both_lr2e5",
         "zoom_object", "zoom_random"]
DESC = {
    "released": "released 3DDA PerAct checkpoint, no fine-tuning (FPS tokens)",
    "released_part": "released weights + part-aware sampling, zero-shot",
    "released_zoom": "released weights + part zoom, zero-shot",
    "baseline": "fine-tuned, upstream FPS token sampling (matched baseline)",
    "part": "fine-tuned, part-aware sampling",
    "zoom": "fine-tuned, part zoom (FPS for coarse tokens)",
    "both": "fine-tuned, part-aware sampling + part zoom",
    "baseline_lr2e5": "fine-tuned baseline, lr 2e-5",
    "both_lr2e5": "fine-tuned part sampling + zoom, lr 2e-5",
    "zoom_object": "fine-tuned, whole-object zoom (secondary comparison)",
    "zoom_random": "fine-tuned, random-region zoom (secondary comparison)",
}


def complete(recs):
    return {t: sum(1 for (tt, _) in recs if tt == t) for t in TASKS}


def rollout_rows(runs):
    rows = []
    for n in ORDER:
        if n not in runs:
            continue
        recs = runs[n]
        row = {"variant": n, "tasks": {}}
        for t in TASKS:
            rs = [r for (tt, _), r in recs.items() if tt == t]
            k = sum(r["success"] for r in rs)
            row["tasks"][t] = {"k": k, "n": len(rs), "ci": wilson(k, len(rs))}
        allr = list(recs.values())
        k = sum(r["success"] for r in allr)
        row["all"] = {"k": k, "n": len(allr), "ci": wilson(k, len(allr))}
        row["infer"] = float(np.mean([r["infer_time_mean"] for r in allr]))
        rows.append(row)
    return rows


def fmt_cell(c):
    if c["n"] == 0:
        return "–"
    lo, hi = c["ci"]
    return f"**{100*c['k']/c['n']:.0f}** ({c['k']}/{c['n']}) [{100*lo:.0f}–{100*hi:.0f}]"


def md_rollout(rows):
    out = ["| variant | insert peg | sort shape | slide block | pooled (150) | mean of tasks |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        mean = np.mean([r["tasks"][t]["k"] / max(r["tasks"][t]["n"], 1)
                        for t in TASKS])
        partial = any(r["tasks"][t]["n"] < 50 for t in TASKS)
        out.append(f"| `{r['variant']}`{' (running)' if partial else ''} | " +
                   " | ".join(fmt_cell(r["tasks"][t]) for t in TASKS) +
                   f" | {fmt_cell(r['all'])} | {100*mean:.1f} |")
    return "\n".join(out)


def md_paired(runs, ref):
    out = [f"| variant | task | Δ SR vs `{ref}` (pts) | wins | losses | exact McNemar p |",
           "|---|---|---|---|---|---|"]
    for n in ORDER:
        if n == ref or n not in runs or ref not in runs:
            continue
        for t in TASKS + ["all"]:
            keys = [k for k in runs[n] if k in runs[ref] and (t == "all" or k[0] == t)]
            if not keys:
                continue
            a = np.array([runs[n][k]["success"] for k in keys])
            b = np.array([runs[ref][k]["success"] for k in keys])
            w, l = int((a & ~b).sum()), int((~a & b).sum())
            p = mcnemar_exact(w, l)
            out.append(f"| `{n}` | {SHORT.get(t, 'pooled')} | {100*(a.mean()-b.mean()):+.1f} | "
                       f"{w} | {l} | {p:.3f}{' *' if p < 0.05 else ''} |")
    return "\n".join(out)


def md_fail(runs):
    out = ["| variant | task | success | timeout (12 steps, wrong poses) | planner error | mean steps to success |",
           "|---|---|---|---|---|---|"]
    for n in ORDER:
        if n not in runs:
            continue
        for t in TASKS:
            rs = [r for (tt, _), r in runs[n].items() if tt == t]
            if not rs:
                continue
            c = defaultdict(int)
            for r in rs:
                ft = r["fail_type"] if r["success"] or r["fail_type"] != "success" else "timeout"
                c["planning" if ft.startswith("planning") else ft] += 1
            st = [r["steps"] for r in rs if r["success"]]
            out.append(f"| `{n}` | {SHORT[t]} | {c['success']} | {c['timeout']} | "
                       f"{c['planning']} | {np.mean(st):.2f} |" if st else
                       f"| `{n}` | {SHORT[t]} | 0 | {c['timeout']} | {c['planning']} | – |")
    return "\n".join(out)


def md_tokens(runs):
    keys = [("coarse_rel_tokens", "in-region coarse tokens available"),
            ("coarse_rel_kept", "in-region coarse tokens kept"),
            ("coarse_rel_retention", "retention of in-region coarse tokens"),
            ("frac_budget_rel", "share of 819-token budget on parts"),
            ("fine_rel_kept", "in-region fine (zoom) tokens kept"),
            ("crop_zoom", "mean crop magnification"),
            ("crop_valid_frac", "crops with ≥4 in-region pixels")]
    out = ["| variant | " + " | ".join(k[1] for k in keys) + " |",
           "|---|" + "---|" * len(keys)]
    for n in ORDER:
        if n not in runs:
            continue
        if all(r["stats_mean"].get("coarse_rel_tokens", 0) == 0
               for r in runs[n].values()):
            continue  # no regions given to the model (FPS variants)
        vals = []
        for k, _ in keys:
            v = [r["stats_mean"].get(k) for r in runs[n].values()
                 if r["stats_mean"].get(k) is not None]
            if not v:
                vals.append("–")
            elif k in ("coarse_rel_retention", "frac_budget_rel", "crop_valid_frac"):
                vals.append(f"{100*np.mean(v):.0f}%")
            elif k == "crop_zoom":
                vals.append(f"{np.mean(v):.2f}×")
            else:
                vals.append(f"{np.mean(v):.0f}")
        out.append(f"| `{n}` | " + " | ".join(vals) + " |")
    return "\n".join(out)


VAL_RE = re.compile(r"^\[val\] step (\d+): (.*)$")


def val_curves(log_dir):
    curves = {}
    for f in sorted(glob.glob(f"{log_dir}/train_*.log")):
        name = Path(f).stem[len("train_"):]
        pts = []
        for line in open(f, errors="ignore"):
            m = VAL_RE.match(line.strip())
            if m:
                d = {k: float(v) for k, v in
                     (kv.split("=") for kv in m.group(2).split(", "))}
                d["step"] = int(m.group(1)) + 1
                pts.append(d)
        if pts:
            curves[name] = pts
    return curves


def md_val(curves):
    out = ["| variant | iters | final-keypose pos err (mm) | pos < 1 cm | quaternion L1 | quat L1 < 0.025 | gripper acc |",
           "|---|---|---|---|---|---|---|"]
    for n in ORDER:
        if n not in curves:
            continue
        d = curves[n][-1]
        out.append(f"| `{n}` | {d['step']} | {1000*d['pos_l2_final']:.1f} | "
                   f"{100*d['pos_l2_final<0.01']:.1f}% | {d['rot_l1']:.3f} | "
                   f"{100*d['rot_l1<0025']:.1f}% | {100*d['traj_gripper']:.1f}% |")
    return "\n".join(out)


def md_offline(off_dir):
    out = ["| variant | task | pos err cm (all kf) | rot err ° (all kf) | pos err cm (interaction kf) | rot err ° (interaction kf) | pos < 1 cm | gripper acc |",
           "|---|---|---|---|---|---|---|---|"]
    any_ = False
    for n in ORDER:
        f = Path(off_dir) / f"{n}.json"
        if not f.exists():
            continue
        any_ = True
        rows = json.load(open(f))["rows"]
        for t in TASKS + ["all"]:
            rs = [r for r in rows if t == "all" or r["task"] == t]
            if not rs:
                continue
            it = [r for r in rs if r["kf"] >= r["T"] // 2]
            out.append(
                f"| `{n}` | {SHORT.get(t, 'pooled')} | {np.mean([r['pos_cm'] for r in rs]):.2f} | "
                f"{np.mean([r['rot_deg'] for r in rs]):.1f} | {np.mean([r['pos_cm'] for r in it]):.2f} | "
                f"{np.mean([r['rot_deg'] for r in it]):.1f} | "
                f"{100*np.mean([r['pos_cm'] < 1 for r in rs]):.0f}% | "
                f"{100*np.mean([r['grip_ok'] for r in rs]):.0f}% |")
    return "\n".join(out) if any_ else "_(not run yet)_"


def copy(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path.home() / "partzoom_project"))
    p.add_argument("--no_ckpt", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- code: upstream 3DDA subset needed to train/eval + our partzoom package
    up = out / "code"
    for d in ["diffuser_actor", "datasets", "utils", "tasks",
              "online_evaluation_rlbench", "partzoom"]:
        copy(REPO / d, up / d)
    for f in ["engine.py", "main_trajectory.py", "LICENSE", "environment.yaml"]:
        if (REPO / f).exists():
            copy(REPO / f, up / f)
    shutil.rmtree(up / "partzoom" / "figs", ignore_errors=True)

    # ---- data: seeds (tiny, exact episode definitions) + pointer to packaged demos
    copy(REPO / "data/partzoom/seeds", out / "data/seeds")
    link = out / "data/packaged"
    if not link.exists():
        os.symlink(REPO / "data/partzoom/packaged", link)

    # ---- checkpoints
    if not a.no_ckpt:
        ck = out / "checkpoints"
        copy(REPO / "train_logs/diffuser_actor_peract.pth", ck / "released/diffuser_actor_peract.pth")
        for d in sorted(glob.glob(str(REPO / "train_logs/partzoom/*"))):
            n = Path(d).name
            for f in ["last.pth", "hparams.json"]:
                if (Path(d) / f).exists():
                    copy(Path(d) / f, ck / n / f)
            for f in glob.glob(f"{d}/events.out.tfevents*"):
                copy(f, out / "logs/tensorboard" / n / Path(f).name)

    # ---- raw results + logs
    copy(REPO / "eval_logs/partzoom", out / "results/rollouts")
    if (REPO / "eval_logs/offline").exists():
        copy(REPO / "eval_logs/offline", out / "results/offline")
    for f in glob.glob(str(REPO / "logs_partzoom/*")):
        name = Path(f).name
        if name.startswith("train_"):
            # strip tqdm progress noise
            lines = [l for l in open(f, errors="ignore").read().replace("\r", "\n").split("\n")
                     if "it/s]" not in l and "s/it]" not in l]
            (out / "logs").mkdir(parents=True, exist_ok=True)
            (out / "logs" / name).write_text("\n".join(lines))
        else:
            copy(f, out / "logs" / name)
    copy(REPO / "partzoom/figs", out / "figs")

    # ---- make code/ runnable in place (relative paths + links into package)
    for f in glob.glob(str(up / "partzoom/*.sh")):
        s = open(f).read()
        s = s.replace("cd ~/3d_diffuser_actor", 'cd "$(dirname "$0")/.."')
        open(f, "w").write(s)
    env = (up / "partzoom/env.sh").read_text().replace(
        "export PYTHONPATH=$HOME/3d_diffuser_actor:$PYTHONPATH",
        'export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)":$PYTHONPATH')
    (up / "partzoom/env.sh").write_text(env)
    inst = up / "instructions/peract/instructions.pkl"
    if not inst.exists():
        import pickle
        full = pickle.load(open(REPO / "instructions/peract/instructions.pkl", "rb"))
        inst.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({t: full[t] for t in TASKS}, open(inst, "wb"))
    for lnk, tgt in [(up / "data/partzoom/packaged", REPO / "data/partzoom/packaged"),
                     (up / "data/partzoom/seeds", out / "data/seeds"),
                     (up / "train_logs/diffuser_actor_peract.pth",
                      out / "checkpoints/released/diffuser_actor_peract.pth")]:
        lnk.parent.mkdir(parents=True, exist_ok=True)
        if not lnk.is_symlink():
            os.symlink(os.path.relpath(tgt, lnk.parent), lnk)

    # ---- tables
    runs = load_rollouts(str(REPO / "eval_logs/partzoom"), step_cap=STEP_CAP)
    rows = rollout_rows(runs)
    curves = val_curves(REPO / "logs_partzoom")
    tables = {
        "rollout": md_rollout(rows),
        "paired_baseline": md_paired(runs, "baseline"),
        "paired_released": md_paired(runs, "released"),
        "fail": md_fail(runs),
        "tokens": md_tokens(runs),
        "val": md_val(curves),
        "offline": md_offline(REPO / "eval_logs/offline"),
    }
    md = ["# Auto-generated tables (partzoom/make_package.py)\n",
          f"Success rates: % (successes/episodes) [Wilson 95% CI]; max 12 steps per episode.\n"]
    for k, v in tables.items():
        md += [f"## {k}\n", v, ""]
    (out / "results/tables.md").write_text("\n".join(md))
    json.dump({"rollout": rows, "val_curves": curves,
               "variants": {n: DESC.get(n, "") for n in runs}},
              open(out / "results/summary.json", "w"), indent=1, default=float)
    print("\n\n".join(tables.values()))


if __name__ == "__main__":
    main()
