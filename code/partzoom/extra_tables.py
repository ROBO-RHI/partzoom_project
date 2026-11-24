"""Extra tables for the final RESULTS.md, computed from the raw logs.

Complements results/tables.md (make_package.py) with:
  - paired exact-McNemar comparisons for the lr-2e-5 matched pair and the
    zoom controls (part vs whole-object vs random-region zoom),
  - offline token retention (incl. FPS retention, measured with part regions),
  - eval-time sweeps on the trained models (offline per-keyframe errors),
  - paired episode-bootstrap CIs for offline error differences,
  - the clean inference benchmark.

usage: python partzoom/extra_tables.py [--out ~/partzoom_project]
Writes <out>/results/tables_extra.md.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "partzoom"))
from analyze import TASKS, SHORT, mcnemar_exact, load_rollouts  # noqa

OFF = REPO / "eval_logs/offline"
PAIRS = [  # (variant, reference)
    ("both_lr2e5", "baseline_lr2e5"),
    ("both_lr1e5", "baseline_lr1e5"),
    ("baseline_lr1e5", "baseline_lr2e5"),
    ("both_lr1e5", "released"),
    ("baseline_lr2e5", "baseline"),
    ("both_lr2e5", "both"),
    ("zoom", "zoom_object"),
    ("zoom", "zoom_random"),
    ("zoom_object", "zoom_random"),
]


def md_pairs(runs):
    out = ["| variant | reference | task | variant SR | reference SR | Δ (pts) | wins | losses | exact McNemar p |",
           "|---|---|---|---|---|---|---|---|---|"]
    for n, ref in PAIRS:
        for t in TASKS + ["all"]:
            keys = sorted(k for k in runs[n] if k in runs[ref] and (t == "all" or k[0] == t))
            a = np.array([runs[n][k]["success"] for k in keys])
            b = np.array([runs[ref][k]["success"] for k in keys])
            w, l = int((a & ~b).sum()), int((~a & b).sum())
            p = mcnemar_exact(w, l)
            out.append(f"| `{n}` | `{ref}` | {SHORT.get(t, 'pooled')} | {100*a.mean():.0f} | "
                       f"{100*b.mean():.0f} | {100*(a.mean()-b.mean()):+.1f} | {w} | {l} | "
                       f"{p:.3f}{' *' if p < 0.05 else ''} |")
    return "\n".join(out)


def load_off(name):
    return json.load(open(OFF / f"{name}.json"))


def err_summary(rows):
    it = [r for r in rows if r["kf"] >= r["T"] // 2]
    peg_it = [r for r in it if r["task"] == TASKS[0]]
    return {"pos": np.mean([r["pos_cm"] for r in rows]),
            "rot": np.mean([r["rot_deg"] for r in rows]),
            "pos_i": np.mean([r["pos_cm"] for r in it]),
            "rot_i": np.mean([r["rot_deg"] for r in it]),
            "peg_pos_i": np.mean([r["pos_cm"] for r in peg_it]),
            "acc1": 100 * np.mean([r["pos_cm"] < 1 for r in rows]),
            "grip": 100 * np.mean([r["grip_ok"] for r in rows])}


def md_tokens_offline(names):
    out = ["| offline run | token mode / zoom / regions | in-region coarse available | kept | retention | share of 819 budget on parts | in-region fine kept | crop magnification | crops with part visible |",
           "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        d = load_off(n)
        a, s = d["args"], d["token_stats"]
        cfg = f"{a['token_mode']} / {'zoom' if a['zoom'] else '–'} / {a['region_source']}"
        f = lambda k, fmt: (fmt.format(s[k]) if k in s else "–")  # noqa
        out.append(f"| `{n}` | {cfg} | {f('coarse_rel_tokens', '{:.0f}')} | "
                   f"{f('coarse_rel_kept', '{:.0f}')} | "
                   f"{f('coarse_rel_retention', '{:.0%}')} | {f('frac_budget_rel', '{:.0%}')} | "
                   f"{f('fine_rel_kept', '{:.0f}')} | {f('crop_zoom', '{:.2f}×')} | "
                   f"{f('crop_valid_frac', '{:.0%}')} |")
    return "\n".join(out)


SWEEPS = [
    ("localisation noise (std of a per-region 3D box shift, m)", [
        ("zoom", "zoom", "0 (default)"), ("zoom", "zoom_noise0.01", "0.01"),
        ("zoom", "zoom_noise0.02", "0.02"), ("zoom", "zoom_noise0.04", "0.04"),
        ("part", "part", "0 (default)"), ("part", "part_noise0.01", "0.01"),
        ("part", "part_noise0.02", "0.02"), ("part", "part_noise0.04", "0.04"),
        ("both", "both", "0 (default)"), ("both", "both_noise0.01", "0.01"),
        ("both", "both_noise0.02", "0.02"), ("both", "both_noise0.04", "0.04")]),
    ("crop margin (fraction of box size added around the part)", [
        ("zoom", "zoom_margin0.0", "0.0"), ("zoom", "zoom_margin0.1", "0.1"),
        ("zoom", "zoom", "0.25 (default)"), ("zoom", "zoom_margin0.5", "0.5"),
        ("zoom", "zoom_margin1.0", "1.0")]),
    ("zoom token budget (fine tokens out of 819)", [
        ("zoom", "zoom_tokens128", "128"), ("zoom", "zoom", "256 (default)"),
        ("zoom", "zoom_tokens384", "384")]),
    ("rel_cap (max share of budget for in-region coarse tokens)", [
        ("part", "part_cap0.25", "0.25"), ("part", "part", "0.5 (default)"),
        ("part", "part_cap0.75", "0.75")]),
]


def md_sweeps():
    blocks = []
    for title, items in SWEEPS:
        out = [f"### {title}\n",
               "| model | setting | pos err cm (all kf) | rot err ° (all kf) | pos err cm (interaction kf) | rot err ° (interaction kf) | insert-peg pos err cm (interaction kf) | pos < 1 cm | retention of in-region coarse | crops with part visible |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for model, name, setting in items:
            d = load_off(name)
            e, s = err_summary(d["rows"]), d["token_stats"]
            cv = f"{s['crop_valid_frac']:.0%}" if "crop_valid_frac" in s else "–"
            out.append(f"| `{model}` | {setting} | {e['pos']:.2f} | {e['rot']:.1f} | {e['pos_i']:.2f} | "
                       f"{e['rot_i']:.1f} | {e['peg_pos_i']:.2f} | {e['acc1']:.0f}% | "
                       f"{s['coarse_rel_retention']:.0%} | {cv} |")
        blocks.append("\n".join(out))
    return "\n\n".join(blocks)


def paired_boot(name, ref, inter, n_boot=10000, seed=0):
    """Mean difference in pos err (cm) name - ref, CI by resampling episodes."""
    def idx(rows):
        return {(r["task"], r["episode"], r["seed"], r["kf"]): r["pos_cm"] for r in rows
                if not inter or r["kf"] >= r["T"] // 2}
    a, b = idx(load_off(name)["rows"]), idx(load_off(ref)["rows"])
    by_ep = defaultdict(list)
    for k in a:
        if k in b:
            by_ep[(k[0], k[1])].append(a[k] - b[k])
    eps = list(by_ep)
    sums = np.array([sum(by_ep[e]) for e in eps])
    cnts = np.array([len(by_ep[e]) for e in eps])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(eps), len(eps))
        bs.append(sums[i].sum() / cnts[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return sums.sum() / cnts.sum(), lo, hi, len(eps)


def md_boot():
    comps = [("part", "baseline"), ("zoom", "baseline"), ("both", "baseline"),
             ("both_lr2e5", "baseline_lr2e5"), ("baseline", "released"),
             ("baseline_lr2e5", "released"), ("both_lr2e5", "released"),
             ("released_part", "released"), ("released_zoom", "released"),
             ("zoom", "zoom_object"), ("zoom", "zoom_random"),
             ("both_lr1e5", "baseline_lr1e5"), ("baseline_lr1e5", "released"),
             ("both_lr1e5", "released")]
    out = ["| variant | reference | Δ pos err cm, all kf [95% CI] | Δ pos err cm, interaction kf [95% CI] | episodes |",
           "|---|---|---|---|---|"]
    for n, ref in comps:
        m, lo, hi, ne = paired_boot(n, ref, False)
        mi, loi, hii, _ = paired_boot(n, ref, True)
        sig = lambda l, h: " *" if (l > 0 or h < 0) else ""  # noqa
        out.append(f"| `{n}` | `{ref}` | {m:+.2f} [{lo:+.2f}, {hi:+.2f}]{sig(lo, hi)} | "
                   f"{mi:+.2f} [{loi:+.2f}, {hii:+.2f}]{sig(loi, hii)} | {ne} |")
    return "\n".join(out)


def md_bench():
    d = json.load(open(OFF / "bench_infer.json"))
    out = ["| configuration | s/step mean | s/step min | peak GPU memory allocated (MiB) | vs FPS |",
           "|---|---|---|---|---|"]
    base = d["fps"]["s_per_step_mean"]
    for k, v in d.items():
        out.append(f"| {k} | {v['s_per_step_mean']:.2f} | {v['s_per_step_min']:.2f} | "
                   f"{v['peak_mem_mib']:.0f} | {100*(v['s_per_step_mean']/base-1):+.0f}% |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path.home() / "partzoom_project"))
    a = p.parse_args()
    runs = load_rollouts(str(REPO / "eval_logs/partzoom"), step_cap=12)
    off_names = ["released", "baseline", "baseline_lr2e5", "part", "released_part",
                 "zoom", "released_zoom", "both", "both_lr2e5", "zoom_object", "zoom_random",
                 "baseline_lr1e5", "both_lr1e5"]
    md = ["# Extra tables (partzoom/extra_tables.py)\n",
          "All computed from eval_logs/partzoom/*.jsonl and eval_logs/offline/*.json. "
          "'*' marks p < 0.05 (McNemar) or a 95% CI excluding 0 (bootstrap).\n",
          "## paired_extra (closed-loop success, same 50 scenes per task)\n", md_pairs(runs), "",
          "## tokens_offline (val keyframes, 2 seeds; FPS runs are given part regions only for these statistics)\n",
          md_tokens_offline(off_names), "",
          "## offline_paired_bootstrap (difference in position error, paired by keyframe, CI by resampling val episodes)\n",
          md_boot(), "",
          "## sweeps (eval-time only: models trained with default settings; offline val errors)\n",
          md_sweeps(), "",
          "## bench (clean benchmark, idle T4, batch 1, released weights, 100 denoising steps per step)\n",
          md_bench(), ""]
    (Path(a.out) / "results/tables_extra.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
