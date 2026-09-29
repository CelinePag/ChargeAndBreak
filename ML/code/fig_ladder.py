"""
fig_ladder.py — how many inputs should each model get?
=====================================================
One line per arm across the feature-set ladder (C compact < D dedup < F full <
L full + raw lookahead), x = the number of inputs the model actually consumes,
y = median route duration against the teacher, mean over training seeds with
the seed standard deviation as error bars.  A second panel shows infeasible
runs on the same x -- two panels rather than a second y axis, as in the
paper's own figures.

All three arms share the "learned" hue from ml_style, so they are told apart
by marker and line style AND labelled directly at the end of each line --
identity is never colour alone.

    python ML/code/fig_ladder.py
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

from ml_style import (COLOR, INK_MUTED, INK_PRIMARY, apply_rc,   # noqa: E402
                      shade, style_axes)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
FLOOR = 0.35

STYLE = {  # arm -> (label, marker, linestyle)
    "gbt": ("Trees", "o", "-"),
    "clf": ("Classifier", "s", "--"),
    "mlp": ("MLP", "^", ":"),
}
SET_ORDER = {"C": 0, "D": 1, "F": 2, "L": 3}


def cells(rows):
    out = {}
    for r in rows:
        out.setdefault((r["arm"], r["fset"]), []).append(r)
    return out


def main():
    with open(os.path.join(RESULTS, "ladder_test.json")) as fh:
        rows = json.load(fh)
    cs = cells(rows)

    apply_rc()
    fig, (ax, axi) = plt.subplots(
        1, 2, figsize=(8.4, 3.5), dpi=200,
        gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.28))
    for a in (ax, axi):
        style_axes(a)
        a.grid(True, axis="y", color="#e0e0e0", lw=0.6)
        a.set_xscale("log")

    # the practical floor: differences inside it are not evidence
    ax.axhspan(-FLOOR, FLOOR, color="#e8e8e8", zorder=0, lw=0)
    ax.axhline(0, color=INK_PRIMARY, lw=0.9, zorder=1)
    ax.text(0.01, -FLOOR, "± 0.35% practical floor", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6.4, color=INK_MUTED)

    col = shade(COLOR["gbt"], 0.15)
    for arm, (lab, mk, ls) in STYLE.items():
        pts = sorted(((fs, v) for (a, fs), v in cs.items() if a == arm),
                     key=lambda p: SET_ORDER.get(p[0], 9))
        if not pts:
            continue
        xs = [v[0]["n_features"] for _, v in pts]
        la = [np.array([r["med_la"] for r in v]) for _, v in pts]
        inf = [np.array([r["infeasible"] for r in v], float) for _, v in pts]
        ax.errorbar(xs, [m.mean() for m in la], yerr=[m.std() for m in la],
                    color=col, marker=mk, ls=ls, lw=1.4, ms=5.5, capsize=2.5,
                    mfc="white", mec=col, mew=1.3, zorder=3)
        axi.errorbar(xs, [m.mean() for m in inf], yerr=[m.std() for m in inf],
                     color=col, marker=mk, ls=ls, lw=1.4, ms=5.5, capsize=2.5,
                     mfc="white", mec=col, mew=1.3, zorder=3)
        # direct label at the right end of each line
        ax.annotate(lab, (xs[-1], la[-1].mean()), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=INK_PRIMARY)
        # the set label at every point, on a different side per arm: trees and
        # the MLP share their input counts (C40, D77, F95, L215), so one goes
        # below-right and the other above-left of its marker; the classifier's
        # sit under its error bar
        for (fs, v), x, m in zip(pts, xs, la):
            if arm == "clf":
                at, off, ha, va = (x, m.mean() - m.std()), (0, -3), "center", "top"
            elif arm == "gbt":
                at, off, ha, va = (x, m.mean()), (6, -5), "left", "top"
            else:
                at, off, ha, va = (x, m.mean()), (-6, 5), "right", "bottom"
            ax.annotate(v[0]["fset_label"], at, xytext=off, textcoords="offset points",
                        ha=ha, va=va, fontsize=6, color=INK_MUTED)

    ax.set_xlabel("inputs the model consumes (log scale)")
    ax.set_ylabel("route duration vs teacher (%)\n← faster      slower →")
    ax.set_title("Performance across the feature-set ladder",
                 fontsize=9, color=INK_PRIMARY, loc="left")
    axi.set_xlabel("inputs the model consumes (log scale)")
    axi.set_ylabel("infeasible runs (of 125)")
    axi.set_title("Infeasibility", fontsize=9, color=INK_PRIMARY, loc="left")
    axi.set_ylim(bottom=0)
    for a in (ax, axi):
        a.set_xlim(22, 290)
        a.set_xticks([30, 50, 100, 200])
        a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.tick_params(axis="x", labelsize=6.5)
        a.minorticks_off()
    fig.text(0.01, -0.04,
             "C compact · D dedup · F full engineered · L full + raw 20-stop "
             "lookahead.  Mean ± sd over training seeds, test batch (125 routes).",
             fontsize=6.6, color=INK_MUTED)
    out = os.path.join(FIGS, "fig_ladder.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
