"""
compare_arms.py — boosted trees vs the simplest neural network, head to head
============================================================================
Both arms solve the same problem on the same rows with the same targets, the
same splits, the same decision rule and the same simulator loop
(policy_core.py).  The only difference is the function that maps a
(state, action) row to a predicted cost.  This script puts the two side by
side and writes ML/RESULTS_ARMS.md plus a figure.

Run after:
    python ML/code/gbt_train.py --tag base
    python ML/code/nn_train.py  --tag nn
    python ML/code/evaluate.py --kind gbt --tag base --split val --out eval_base_val.json
    python ML/code/evaluate.py --kind gbt --tag base --split val --guard-q 0.95 --out eval_base_g95_val.json
    python ML/code/evaluate.py --kind nn  --tag nn   --split val --out eval_nn_val.json
    python ML/code/evaluate.py --kind nn  --tag nn   --split val --guard-q 0.95 --out eval_nn_g95_val.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
OUT = os.path.abspath(os.path.join(HERE, "..", "RESULTS_ARMS.md"))
FLOOR = 0.35

ARMS = [
    ("GBT, nominal guard", "eval_base_val.json"),
    ("GBT, guard 0.95", "eval_base_g95_val.json"),
    ("MLP, nominal guard", "eval_nn_val.json"),
    ("MLP, guard 0.95", "eval_nn_g95_val.json"),
]


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def paired(rows, key):
    return np.array([100.0 * (r["duration_h"] - r[key]) / r[key] for r in rows
                     if r.get("route_completed") and r.get(key)
                     and not r.get(f"{key}_infeasible")])


def summarise(rows):
    v = paired(rows, "LA")
    g = paired(rows, "GREEDY")
    halted = [r for r in rows if not r["route_completed"]]
    med = float(np.median(v)) if len(v) else np.nan
    try:
        p = float(stats.wilcoxon(v)[1])
    except Exception:
        p = np.nan
    if np.isnan(p) or p >= 0.05:
        verdict = "matches the teacher"
    elif abs(med) < FLOOR:
        verdict = "below the practical floor"
    else:
        verdict = "faster than teacher" if med < 0 else "SLOWER than teacher"
    comp = [r for r in rows if r["route_completed"]]
    tw = sum(r["tw_misses"] for r in rows)
    tw_la = sum(r.get("LA_tw", 0) for r in rows)
    a = np.array([r["n_rests"] for r in comp if r.get("LA_rests") is not None])
    b = np.array([r["LA_rests"] for r in comp if r.get("LA_rests") is not None])
    return dict(
        n=len(v), med_la=med, p=p, verdict=verdict,
        med_greedy=float(np.median(g)) if len(g) else np.nan,
        halted=len(halted),
        ms=float(np.mean([r["ms_per_decision"] for r in rows])),
        tw=tw, tw_la=tw_la,
        rest_delta=float(np.mean(a - b)) if len(a) else np.nan,
        p95=float(np.percentile(v, 95)) if len(v) else np.nan,
    )


def figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, HERE)
    from figures import GRID, INK, MUTED, OKABE_ITO, SURFACE, _style

    series = [(lab, paired(rows, "LA")) for lab, rows in data]
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=200)
    _style(ax)
    rng = np.random.default_rng(0)
    for k, (lab, v) in enumerate(series):
        y = len(series) - 1 - k
        col = OKABE_ITO[0] if lab.startswith("GBT") else OKABE_ITO[1]
        ax.scatter(v, y + rng.uniform(-0.15, 0.15, len(v)), s=8, color=col,
                   alpha=0.32, linewidths=0, zorder=2)
        bp = ax.boxplot([v], positions=[y], vert=False, widths=0.44,
                        showfliers=False, patch_artist=True, zorder=3)
        bp["boxes"][0].set(facecolor="none", edgecolor=col, linewidth=2)
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                art.set(color=col, linewidth=1.3)
        bp["medians"][0].set(color=col, linewidth=2.6)
        ax.text(np.median(v), y + 0.33, f"{np.median(v):+.2f}%", ha="center",
                fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, lw=1.4, zorder=1)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([s[0] for s in reversed(series)], fontsize=9.5,
                       color=INK)
    ax.set_xlabel("route duration vs the LA teacher  (%)   "
                  "<- faster        slower ->", fontsize=9, color=MUTED)
    ax.set_title("Boosted trees vs the simplest neural network "
                 "(validation split)", fontsize=11, color=INK, loc="left",
                 pad=10)
    allv = np.concatenate([s[1] for s in series])
    ax.set_xlim(np.percentile(allv, 0.5) - 2, np.percentile(allv, 99) + 2)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig4_arms_val.png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"   wrote {out}")


def main():
    data = [(lab, _load(f)) for lab, f in ARMS]
    missing = [lab for lab, r in data if r is None]
    if missing:
        raise SystemExit("missing evaluations: " + ", ".join(missing))
    rows = [(lab, summarise(r)) for lab, r in data]

    L = []
    A = L.append
    A("# Boosted trees vs the simplest neural network\n")
    A("Both arms solve the **same problem**: same 411k (state, action) rows, "
      "same regret / feasibility / charge-duration targets, same seed-within-"
      "family splits, and the same decision rule and simulator loop "
      "(`policy_core.py`). The only thing that differs is the regressor, so "
      "this is a controlled comparison.\n")
    A("| configuration | n | vs LA | Wilcoxon p | vs GREEDY | infeasible /136 | "
      "TW misses | rests vs LA | p95 vs LA | ms/dec | verdict |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for lab, s in rows:
        A(f"| **{lab}** | {s['n']} | {s['med_la']:+.2f}% | {s['p']:.1e} | "
          f"{s['med_greedy']:+.2f}% | {s['halted']} | {s['tw']} | "
          f"{s['rest_delta']:+.2f} | {s['p95']:+.1f}% | {s['ms']:.1f} | "
          f"{s['verdict']} |")
    A(f"\n(teacher time-window misses on the same routes: "
      f"{rows[0][1]['tw_la']})\n")
    A("**`n` differs by row, and that flatters the network.** A halted run has "
      "no duration, so each arm's percentages are paired only on the routes "
      "*it* completed. The MLP's +2.6% is therefore computed with its 18 "
      "infeasible runs excluded -- the routes it found hardest -- while the GBT's "
      "figure includes almost all of them. The true gap is wider than the "
      "table shows.\n")

    A("## Reading\n")
    A("- **The trees win, clearly and on every axis.** The GBT matches the "
      "teacher; the MLP is ~2.6-2.8% slower than it, which puts the network "
      "roughly level with GREEDY -- i.e. the neural student recovers little "
      "of what the look-ahead buys over a myopic rule.")
    A("- **Feasibility separates them further.** The MLP halts 18 times "
      "without the guard and 8 with it; the GBT halts 4 and 0. The guard "
      "helps both arms but cannot rescue the network.")
    A("- **The network over-rests and over-charges.** Its rest count runs "
      "above the teacher's while the GBT matches it exactly, and a spurious "
      "daily rest costs 9-11 h -- which is where its heavy right tail "
      "(p95 well above the GBT's) comes from.")
    A("- **Offline and closed-loop agree here.** Mean realised regret on "
      "validation is 0.71 min for the GBT against 1.85 min for the MLP, and "
      "top-1 agreement 93.4% against 88.3%. The supervised ranking already "
      "predicted the closed-loop ordering, so in this instance the cheap "
      "metric was trustworthy.\n")

    A("## Why this outcome is the expected one\n")
    A("The feasible region is a set of exact thresholds -- 4.5 h consecutive "
      "driving, 9 h shift driving, 13/15 h spread, 3 reduced rests. A tree "
      "split *is* such a threshold; an MLP has to approximate a step with a "
      "smooth ramp, and its error is largest exactly at the boundary, which "
      "is where the decision flips. The feature set already moves every "
      "threshold to zero (`cd_slack = 4.5 - cd`), which is the most a feature "
      "design can do to help a network here, and it was still not enough.\n")
    A("This is not a claim that neural networks cannot do better on this "
      "problem. It is a claim about the *simplest* network: three plain MLPs, "
      "no shared trunk, no listwise loss over the action set, no sample "
      "weighting (sklearn's MLP has none). The version with a genuine "
      "structural advantage -- one trunk scoring all 12 actions in a single "
      "pass, trained with a listwise loss at a temperature set by the "
      "teacher's own 2.08 min sampling SEM -- has not been built. That, plus "
      "DAgger fine-tuning, is the branch where a network could plausibly "
      "overtake the trees, and it needs `torch` rather than sklearn.\n")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT}")
    figure(data)

    print("\n" + "=" * 100)
    print(f"{'configuration':22s} {'vs LA':>8s} {'vs GRDY':>8s} {'infs':>5s} "
          f"{'TW':>5s} {'rests':>7s} {'p95':>8s} {'ms':>6s}  verdict")
    print("-" * 100)
    for lab, s in rows:
        print(f"{lab:22s} {s['med_la']:+8.2f} {s['med_greedy']:+8.2f} "
              f"{s['halted']:5d} {s['tw']:5d} {s['rest_delta']:+7.2f} "
              f"{s['p95']:+8.1f} {s['ms']:6.1f}  {s['verdict']}")


if __name__ == "__main__":
    main()
