"""Success-rate figure: one panel per task (+ pooled), bars per variant with
Wilson 95% CIs. Reads <out>/results/summary.json written by make_package.py.

usage: python partzoom/make_fig.py [--out ~/partzoom_project]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

ORDER = ["released", "released_part", "released_zoom",
         "baseline", "part", "zoom", "both",
         "baseline_lr2e5", "both_lr2e5",
         "baseline_lr1e5", "both_lr1e5",
         "zoom_object", "zoom_random"]
GROUP = {"released": 0, "released_part": 0, "released_zoom": 0,
         "baseline": 1, "part": 1, "zoom": 1, "both": 1,
         "baseline_lr2e5": 2, "both_lr2e5": 2,
         "baseline_lr1e5": 4, "both_lr1e5": 4,
         "zoom_object": 3, "zoom_random": 3}
GROUP_NAME = ["released weights (no fine-tuning)", "fine-tuned, lr 1e-4",
              "fine-tuned, lr 2e-5", "zoom controls (lr 1e-4)",
              "fine-tuned, lr 1e-5"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8c8b86", "#7a4fd1"]
PANELS = [("insert_onto_square_peg", "insert peg (n=50)"),
          ("place_shape_in_shape_sorter", "sort shape (n=50)"),
          ("slide_block_to_color_target", "slide block (n=50)"),
          ("all", "pooled (n=150)")]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path.home() / "partzoom_project"))
    a = p.parse_args()
    out = Path(a.out)
    rows = {r["variant"]: r for r in json.load(open(out / "results/summary.json"))["rollout"]}
    names = [n for n in ORDER if n in rows]
    # y positions with a gap between groups
    ys, y = [], 0.0
    for i, n in enumerate(names):
        if i and GROUP[n] != GROUP[names[i - 1]]:
            y += 0.6
        ys.append(y)
        y += 1.0

    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": TEXT})
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    for ax, (task, title) in zip(axes, PANELS):
        ref = None
        for n, yy in zip(names, ys):
            c = rows[n]["all"] if task == "all" else rows[n]["tasks"][task]
            sr = 100 * c["k"] / c["n"]
            lo, hi = 100 * c["ci"][0], 100 * c["ci"][1]
            if n == "released":
                ref = sr
            ax.barh(yy, sr, height=0.72, color=COLORS[GROUP[n]], zorder=2)
            ax.errorbar(sr, yy, xerr=[[sr - lo], [hi - sr]], fmt="none",
                        ecolor=TEXT, elinewidth=1, capsize=2.5, zorder=3)
            ax.text(117, yy, f"{sr:.0f}", va="center", ha="right", fontsize=8, color=MUTED)
        if ref is not None:
            ax.axvline(ref, color=COLORS[0], lw=1, ls="--", zorder=1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("success rate (%)")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    labels = GROUP_NAME + []
    handles.append(plt.Line2D([0], [0], color=COLORS[0], lw=1, ls="--"))
    labels.append("released checkpoint")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle("Closed-loop success on 50 fixed test scenes per task "
                 "(bars: success %, whiskers: Wilson 95% CI; max 12 steps)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    (out / "figs").mkdir(exist_ok=True)
    fig.savefig(out / "figs/success_rates.png", dpi=160, facecolor="white")
    print("wrote", out / "figs/success_rates.png")


if __name__ == "__main__":
    main()
