"""
fig_effort.py -- Fig. 1 of the CPAIOR draft (redrawn 2026-10-04 for clarity).

One row per method; hollow dot = mean gap to the hindsight oracle on the 32
test routes with UNIFORM chargers, filled dot = the same routes with chargers
of MIXED power.  The PyTorch student's rows differ only in what it was trained
on, and the label says how many LA decisions that cost.  Means are over routes
(each route averaged over the 3 training seeds); the paired statistics are in
Table 3.  Numbers come from the result stores via ML/scratch/cmp_curve.py.

    python ML/paper/fig_effort.py        -> ML/paper/fig_effort.pdf (+ .png preview)
"""
import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scratch"))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.lines import Line2D                                  # noqa: E402

from ml_style import apply_rc, INK_MUTED                             # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import cmp_curve as cc                                           # noqa: E402

GREEN, GREY, SKY, VERM = "#009E73", "#8C8C8C", "#56B4E9", "#D55E00"   # Okabe-Ito


def means(name, v="pmix"):
    """Mean gap (%) on the uniform and the mixed version of the same routes."""
    labels = cc.CONFIGS[name][1]
    u, x = [], []
    for b in sorted({b for (_m, vv, b) in cc.G if vv == v}):
        a, c = cc.mean_over_seeds(labels, "uniform", b), cc.mean_over_seeds(labels, v, b)
        if a and c:
            u.append(a[0])
            x.append(c[0])
    return np.mean(u), np.mean(x)


# top to bottom
ROWS = [("LA (re-solves at every stop)", "LA (the teacher)", GREEN),
        ("Student, uniform routes only", "torch, no mixed data", GREY),
        ("+ 47 mixed routes from the LA\n(2,279 LA decisions)", "torch, + pilot (47 routes)", SKY),
        ("+ 121 mixed routes from the LA\n(8,246 LA decisions)", "torch, + 121 mixed routes", SKY),
        ("+ DAgger, 1 round\n(1,781 LA calls)", "torch, + DAgger", VERM),
        ("+ DAgger, 2 rounds\n(3,557 LA calls)", "torch, + DAgger x2", VERM)]

apply_rc()
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # no Type 3 fonts (LNCS)
fig, ax = plt.subplots(figsize=(4.8, 2.6))
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK_MUTED)
ax.spines["bottom"].set_linewidth(0.6)
ax.grid(True, axis="x", color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)

ys = np.arange(len(ROWS))[::-1]
for y, (label, name, col) in zip(ys, ROWS):
    u, x = means(name)
    ax.plot([u, x], [y, y], color=col, lw=2.0, alpha=0.55, solid_capstyle="round", zorder=2)
    ax.scatter([u], [y], s=34, facecolor="white", edgecolor=col, linewidth=1.4, zorder=3)
    ax.scatter([x], [y], s=34, facecolor=col, edgecolor=col, linewidth=1.4, zorder=4)
    ax.text(max(u, x) + 0.25, y, f"{x - u:+.1f}".replace("-", "−"), va="center",
            ha="left", fontsize=7, color=INK_MUTED)

ax.axhline(ys[0] - 0.5, color="#bbbbbb", lw=0.6)          # LA above, students below
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=7)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 8.6)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlabel("Gap to the hindsight optimum (%), mean over 32 test routes")
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK_MUTED, ms=5.5,
                          label="chargers all alike"),
                   Line2D([], [], marker="o", ls="", mfc=INK_MUTED, mec=INK_MUTED, ms=5.5,
                          label="chargers of mixed power")],
          frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.0),
          ncol=2, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
fig.savefig(os.path.join(HERE, "fig_effort.pdf"))
fig.savefig(os.path.join(HERE, "fig_effort.png"), dpi=220)
print("wrote", os.path.join(HERE, "fig_effort.pdf"))
