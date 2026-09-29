"""
fig_ood.py — base-case models on instances they were never trained for
======================================================================
One group of rows per instance family: the base case first, as the reference
the models were trained for, then each physics shift.  Within a group every
method gets one row: the three learned arms (their best base-case
configuration, median training seed), the teacher LA and Greedy.  The marker
is the median gap to the oracle solved FOR THOSE instances; the bar is the
interquartile range.  The right strip is the infeasibility rate on the
manuscript's green -> yellow -> vermillion ramp.

Read each group against the base-case group at the top: that is the
degradation caused by the shift alone, since the route seeds (22-25) are the
same as the base-case test.

    python ML/code/fig_ood.py                     -> fig_ood.png
    python ML/code/fig_ood.py --variant g99sr     -> fig_ood_g99sr.png: the same
        models with the spread-room check and a 0.99 drive guard
        (ood_eval.py --guard-q 0.99 --spread-room)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_style import (COLOR, INK_MUTED, INK_PRIMARY, apply_rc,   # noqa: E402
                      infeas_color, shade, style_axes)
from ood_eval import AXES, LEARNED, MODELS                       # noqa: E402
from paper_link import collect_gaps                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))

METHODS = LEARNED + ["LA", "Greedy"]
_ARM_MARK = {"gbt": "o", "clf": "s", "nn": "^"}
MARK = {lab: _ARM_MARK[k] for k, _t, lab in MODELS}
if sum(k == "nn" for k, _t, _l in MODELS) > 1:          # second MLP: its own marker
    MARK[[lab for k, _t, lab in MODELS if k == "nn"][1]] = ">"
MARK.update(LA="D", Greedy="v")
KEY = {lab: {"gbt": "gbt", "clf": "clf", "nn": "mlp"}[k] for k, _t, lab in MODELS}
KEY.update(LA="LA", Greedy="greedy")
XCAP = 25.0        # % -- the x axis stops here; medians beyond it are labelled


def base_reference(variant="g95"):
    """The same three models + LA / Greedy on the BASE-CASE test batch."""
    out = {}
    for kind, tag, lab in MODELS:
        g = collect_gaps(f"eval_{tag}_{variant}_test.json", student_label=lab)
        a, _p, n_inf = g[lab]
        out[lab] = (a, n_inf, len(a) + n_inf)
        for src, dst in (("LA (look-ahead MILP)", "LA"), ("GREEDY", "Greedy")):
            if dst not in out and src in g:
                b, _bp, bi = g[src]
                out[dst] = (b, bi, len(b) + bi)
    return out


def ood_groups(variant="g95"):
    tail = "" if variant == "g95" else f"_{variant}"
    with open(os.path.join(RESULTS, f"ood_test{tail}.json")) as fh:
        rows = json.load(fh)
    groups = []
    for axis, (_, _, label) in AXES.items():
        sub = [r for r in rows if r["axis"] == axis]
        if not sub:
            continue
        g = {}
        for m in METHODS:
            mr = [r for r in sub if r["method"] == m]
            if not mr:
                continue
            v = np.array([r["gap"] for r in mr
                          if r["completed"] and r["gap"] is not None])
            g[m] = (v, sum(1 for r in mr if not r["completed"]), len(mr))
        groups.append((label, g))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="g95",
                    help="g95 (as trained), g95sr, g99sr -- see ood_eval.py")
    variant = ap.parse_args().variant
    groups = ([("Base case (trained on)", base_reference(variant))]
              + ood_groups(variant))
    rows = []                                  # (y, group, method, stats)
    y = 0.0
    ticks, ticklabels, bands = [], [], []
    headers = []
    for gi, (glab, g) in enumerate(groups):
        headers.append((y, glab, gi))           # header row
        y += 0.95
        y0 = y
        for m in METHODS:
            if m in g:
                rows.append((y, glab, m, g[m]))
                y += 1.0
        bands.append((y0 - 0.5, y - 0.5, glab, gi))
        y += 0.5                                # gap between groups

    apply_rc()
    fig = plt.figure(figsize=(7.4, 0.24 * len(rows) + 1.6), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[11, 1], wspace=0.035)
    ax = fig.add_subplot(gs[0, 0])
    axf = fig.add_subplot(gs[0, 1], sharey=ax)
    style_axes(ax)
    style_axes(axf, xgrid=False)

    for (ylo, yhi, glab, gi) in bands:
        if gi % 2 == 0:
            ax.axhspan(ylo, yhi, color="#f4f4f4", zorder=0, lw=0)
    for (yh, glab, gi) in headers:
        ax.text(0.0, yh, glab, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.6, color=INK_PRIMARY,
                fontweight="bold")

    # The axis is capped so that one far-off row (the MLP on the charger
    # shifts reaches +40 %) does not squeeze every other row into a corner; a
    # median beyond the cap is drawn AT the edge and labelled with its value.
    all_q3 = [np.percentile(v, 75) for _y, _g, _m, (v, _i, _n) in rows if len(v)]
    xmax = min(np.percentile(all_q3, 97) * 1.25 if all_q3 else 10, XCAP)
    for yy, glab, m, (v, n_inf, n) in rows:
        col = COLOR[KEY[m]]
        if len(v):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([q1, q3], [yy, yy], color=shade(col, 0.1), lw=2.2,
                    solid_capstyle="round", zorder=2)
            off = med > xmax
            ax.plot([min(med, xmax)], [yy], marker=MARK[m], ms=5.2,
                    color=shade(col, 0.25), mfc="white" if m in LEARNED else col,
                    mew=1.3, zorder=3, clip_on=False)
            if off:
                ax.text(xmax * 0.975, yy, f"{med:+.1f} →", ha="right", va="center",
                        fontsize=5.8, color=INK_PRIMARY, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
            else:
                on_bar = q3 > xmax * 0.88          # the label would sit on the bar
                ax.text(min(q3, xmax * 0.88), yy, f"  {med:+.1f}", va="center",
                        fontsize=5.8, color=INK_MUTED, clip_on=True, zorder=4,
                        bbox=(dict(boxstyle="square,pad=0.1", fc="white", ec="none")
                              if on_bar else None))
        else:
            ax.text(0.3, yy, "all runs infeasible", va="center", fontsize=5.8,
                    color=INK_MUTED, style="italic")
        rate = n_inf / max(n, 1)
        axf.barh([yy], [1.0], height=0.78, color=infeas_color(rate),
                 edgecolor="white", linewidth=0.4)
        axf.text(0.5, yy, f"{100*rate:.0f}", ha="center", va="center",
                 fontsize=5.4, color=INK_PRIMARY if rate < 0.12 else "white")

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[2] for r in rows], fontsize=6.2, color=INK_MUTED)
    ax.tick_params(axis="y", pad=2, length=0)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_PRIMARY, lw=0.9, zorder=1)
    ax.set_xlim(-0.8, xmax)
    ax.set_xlabel("gap to the hindsight oracle (%, lower is better)   "
                  "marker = median, bar = interquartile range")
    extra = {"g95sr": "  (+ spread-room check)",
             "g99sr": "  (+ spread-room check, 0.99 drive guard)"}.get(variant, "")
    ax.set_title("Trained on the base case — tested on shifted physics" + extra,
                 fontsize=9.5, color=INK_PRIMARY, loc="left", pad=6)
    # group names sit left of the method names: widen the left margin
    ax.yaxis.set_label_coords(-0.3, 0.5)
    axf.set_xticks([])
    axf.set_xlim(0, 1)
    axf.set_xlabel("infeas.\n%", fontsize=6.5, color=INK_MUTED)
    axf.spines["left"].set_visible(False)
    axf.spines["bottom"].set_visible(False)
    axf.tick_params(left=False, labelleft=False)
    fig.text(0.01, 0.005,
             "Learned policies (hollow markers) were fitted only on base-case "
             "routes; route seeds 22-25 are identical across groups, so each "
             "shift is the only change.", fontsize=6.4, color=INK_MUTED)
    out = os.path.join(FIGS, "fig_ood.png" if variant == "g95"
                       else f"fig_ood_{variant}.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
