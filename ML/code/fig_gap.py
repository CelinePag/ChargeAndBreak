"""
fig_gap.py — every method and every configuration on ONE axis
=============================================================
The manuscript's metric (gap to the hindsight oracle), in the manuscript's
style (`ml_style.py`, extending `src/plot/paper_style.py`).  "% vs LA" puts the
teacher at the origin and hides where things sit relative to the optimum; this
puts the ORACLE at the origin, so every policy -- the solver baselines already
in `solutions/basecase` and every learned configuration in `configs.py` --
occupies its own position on a common scale.

Reads `ML/results/all_test.json`, so it stays in step with whatever `run_all.py`
last produced rather than depending on a list of file names.

Two encodings, kept apart exactly as the paper does:

  box colour    METHOD IDENTITY.  Greedy is the same blue as in
                `figures/basecase/paper_gap_box.png`, LA the same green, 2SP
                the same purple.  All four learned arms share the one free
                Okabe-Ito slot and are told apart by HATCH, the DET/DETg
                precedent -- so "learned" reads at a glance and identity is on
                the y axis, never colour alone.
  strip colour  INFEASIBILITY RATE, on the manuscript's green -> yellow ->
                vermillion ramp.  A violation ends the run, so an infeasible
                run has no duration: a policy is not better because it posts a
                good median over the routes it happened to complete.

  fig_gap_test.png       the headline: everything, on the held-out test batch
  fig_gap_arms.png       the learned arms only, for the arm comparison

RO spans 4-50%, overlapping every other method rather than sitting in a clean
high band, so a broken axis would drop real points into the gap.  It is left
off the scale and named in the caption.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_style import (COLOR, GRID, HATCH, INK_MUTED, INK_PRIMARY,  # noqa: E402
                      LBL, apply_rc, infeas_color, shade, style_axes, tint)
from paper_link import collect_gaps              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
OFF_SCALE = 15.0

BASELINE_KEY = {"LA (look-ahead MILP)": "LA", "GREEDY": "greedy",
                "2SP": "2SP", "RO": "RO"}


def collect(rows, with_baselines=True):
    """[(label, style_key, gaps, n_infeasible)] for every configuration."""
    out, base_done = [], False
    for r in rows:
        f = r["eval_file"]
        if not os.path.exists(os.path.join(RESULTS, f)):
            continue
        g = collect_gaps(f, student_label=r["display"])
        a, _p, n_inf = g[r["display"]]
        if len(a):
            out.append((r["display"], r["arm"], a, n_inf))
        if with_baselines and not base_done:
            for k, key in BASELINE_KEY.items():
                if k in g and len(g[k][0]):
                    out.append((LBL.get(key, k), key, g[k][0], g[k][2]))
            base_done = True
    return out


def draw(series, out, title, subtitle):
    series = sorted(series, key=lambda s: np.median(s[2]))
    shown = [s for s in series if np.median(s[2]) < OFF_SCALE]
    off = [s for s in series if np.median(s[2]) >= OFF_SCALE]

    apply_rc()
    fig = plt.figure(figsize=(7.6, 0.46 * len(shown) + 1.9), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[11, 1], wspace=0.035)
    ax = fig.add_subplot(gs[0, 0])
    axf = fig.add_subplot(gs[0, 1], sharey=ax)
    style_axes(ax)
    style_axes(axf, xgrid=False)

    hi = 0.0
    for _l, _k, v, _n in shown:
        q1, q3 = np.percentile(v, [25, 75])
        w = v[v <= q3 + 1.5 * (q3 - q1)]
        hi = max(hi, float(w.max()) if len(w) else q3)
    ax.set_xlim(-0.4, hi * 1.10)

    rng = np.random.default_rng(0)
    for i, (label, key, v, n_inf) in enumerate(shown):
        y = len(shown) - 1 - i
        col = COLOR.get(key, COLOR["gbt"])
        ax.scatter(v, y + rng.uniform(-0.12, 0.12, len(v)), s=4, color=col,
                   alpha=0.25, linewidths=0, zorder=2)
        bp = ax.boxplot([v], positions=[y], vert=False, widths=0.42,
                        showfliers=False, patch_artist=True, zorder=3)
        bp["boxes"][0].set(facecolor=tint(col, 0.55), edgecolor=shade(col),
                           linewidth=0.9, hatch=HATCH.get(key, ""))
        for part in ("whiskers", "caps"):
            for art in bp[part]:
                art.set(color=shade(col), linewidth=0.8)
        bp["medians"][0].set(color=shade(col, 0.55), linewidth=1.8)
        ax.text(float(np.median(v)), y + 0.28, f"{np.median(v):+.2f}%",
                ha="center", fontsize=7, color=INK_PRIMARY, zorder=7)

        rate = n_inf / max(len(v) + n_inf, 1)
        axf.barh([y], [1.0], height=0.42, color=infeas_color(rate),
                 edgecolor=INK_MUTED, linewidth=0.4)
        axf.text(0.5, y, f"{100*rate:.0f}%", ha="center", va="center",
                 fontsize=6.3,
                 color=INK_PRIMARY if rate < 0.12 else "#ffffff")

    ax.axvline(0, color=INK_PRIMARY, lw=1.1, zorder=1)
    ax.set_ylim(-0.7, len(shown) - 0.3)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([f"{l}  (n={len(v)})" for l, _k, v, _n in reversed(shown)],
                       fontsize=7.4)
    ax.set_xlabel("gap to the hindsight oracle  (%, lower is better)")
    axf.set_xticks([])
    axf.set_xlim(0, 1)
    axf.set_xlabel("infeas.", fontsize=7, color=INK_MUTED, labelpad=4)
    axf.spines["left"].set_visible(False)
    axf.spines["bottom"].set_visible(False)
    axf.tick_params(left=False, labelleft=False)

    ax.set_title(title, fontsize=9.5, color=INK_PRIMARY, loc="left", pad=28)
    note = subtitle
    if off:
        note += "\n" + "Off scale: " + ", ".join(
            f"{l} {np.median(v):+.1f}%" for l, _k, v, _n in off)
    ax.text(0, 1.012, note, transform=ax.transAxes, fontsize=7,
            color=INK_MUTED, ha="left", va="bottom", linespacing=1.5)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {out}")


def main():
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(RESULTS, "all_test.json")
    if not os.path.exists(p):
        # run_all.py writes the real table only at the end; a provisional one
        # lets the figure be drawn from whatever has finished so far
        p = os.path.join(RESULTS, "all_test_partial.json")
    if not os.path.exists(p):
        raise SystemExit("no all_test.json yet — run ML/code/run_all.py")
    print(f"   (table: {os.path.basename(p)})")
    with open(p) as fh:
        rows = json.load(fh)

    everything = collect(rows, with_baselines=True)
    draw(everything, os.path.join(FIGS, "fig_gap_test.png"),
         "Gap to the hindsight optimum — every method and configuration",
         "held-out test batch (seeds 22-25, 125 routes); learned policies use "
         "no solver at deployment")

    arms_only = [s for s in collect(rows, with_baselines=False)]
    # keep LA as the reference line in the arm comparison
    la = [s for s in everything if s[1] == "LA"]
    draw(arms_only + la, os.path.join(FIGS, "fig_gap_arms.png"),
         "Learned configurations, against the teacher",
         "held-out test batch; every row is an independent model fitted on "
         "seeds 1-19")


if __name__ == "__main__":
    main()
