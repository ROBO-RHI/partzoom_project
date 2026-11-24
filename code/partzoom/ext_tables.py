"""Round 3: 100-scene closed-loop tables (test episodes 0-99) for the variants
that were evaluated on the extended test set. Main tables elsewhere use 0-49.

usage: python partzoom/ext_tables.py [--out ~/harsh/partzoom_project]
Writes <out>/results/tables_100scenes.md and prints it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "partzoom"))
from analyze import TASKS, SHORT, wilson, mcnemar_exact, load_rollouts  # noqa

ORDER = ["released", "released_part", "baseline_lr2e5", "both_lr2e5",
         "baseline_lr1e5", "both_lr1e5"]
PAIRS = [("both_lr2e5", "baseline_lr2e5"), ("both_lr1e5", "baseline_lr1e5"),
         ("both_lr2e5", "released"), ("both_lr1e5", "released"),
         ("baseline_lr2e5", "released"), ("baseline_lr1e5", "released"),
         ("released_part", "released"), ("both_lr1e5", "both_lr2e5"),
         ("baseline_lr1e5", "baseline_lr2e5")]
N = 100


def cell(rs):
    k, n = sum(r["success"] for r in rs), len(rs)
    if n == 0:
        return "–"
    lo, hi = wilson(k, n)
    return f"**{100*k/n:.0f}** ({k}/{n}) [{100*lo:.0f}–{100*hi:.0f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path.home() / "harsh/partzoom_project"))
    a = p.parse_args()
    runs = load_rollouts(str(REPO / "eval_logs/partzoom"), step_cap=12,
                         episodes=(0, N))
    full = [n for n in ORDER if n in runs and all(
        sum(1 for (t, _) in runs[n] if t == tk) == N for tk in TASKS)]
    md = ["# Closed-loop results on the extended test set (100 scenes per task, episodes 0–99)\n",
          "Success % (successes/episodes) [Wilson 95% CI], max 12 steps. Only variants "
          "with all 300 episodes are listed. Episodes 0–49 are the same scenes as the main tables.\n",
          "| variant | insert peg | sort shape | slide block | pooled (300) |",
          "|---|---|---|---|---|"]
    for n in full:
        recs = runs[n]
        md.append(f"| `{n}` | " + " | ".join(
            cell([r for (t, _), r in recs.items() if t == tk]) for tk in TASKS)
            + f" | {cell(list(recs.values()))} |")
    md += ["", "## Paired comparisons (exact two-sided McNemar, same 100 scenes per task)\n",
           "| variant | reference | task | Δ SR (pts) | wins | losses | p |",
           "|---|---|---|---|---|---|---|"]
    for n, ref in PAIRS:
        if n not in full or ref not in full:
            continue
        for t in TASKS + ["all"]:
            keys = [k for k in runs[n] if k in runs[ref] and (t == "all" or k[0] == t)]
            x = np.array([runs[n][k]["success"] for k in keys])
            y = np.array([runs[ref][k]["success"] for k in keys])
            w, l = int((x & ~y).sum()), int((~x & y).sum())
            pv = mcnemar_exact(w, l)
            md.append(f"| `{n}` | `{ref}` | {SHORT.get(t, 'pooled')} | "
                      f"{100*(x.mean()-y.mean()):+.1f} | {w} | {l} | "
                      f"{pv:.3f}{' *' if pv < 0.05 else ''} |")
    txt = "\n".join(md)
    out = Path(a.out) / "results/tables_100scenes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
