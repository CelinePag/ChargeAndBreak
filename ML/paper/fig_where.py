"""
fig_where.py — "where the improvement comes from", two panels (paper figure)
============================================================================
(a) The schedules of one test route around its third and fourth customers
    (hours 40-72), from fig_schedule.json (fig_schedule.py re-drives the
    policies and checks them against the stored test runs): oracle, LA, the
    PyTorch student and its rollout; daily rests, charging, customer service,
    time windows, missed windows, decisions changed by the rollout.
(b) Predicted against measured cost of resting now, over the student's
    action, at the decisive states of the trees' rollout labels
    (data/rl_api_r1_gbt; ML/scratch/rest_mispricing.py): rests that add a
    daily rest to the route, and rests that keep the number of rests, by the
    driving allowance they give up.

Writes fig_where.pdf/.png.

    python ML/paper/fig_where.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scratch"))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from fig_schedule import CACHE, COL                                  # noqa: E402

ROWS = [("Oracle (hindsight)", "Oracle"), ("LA", "LA"), ("Learned network", "Network"),
        ("Network + rollout", "Network + rollout")]
X0, X1 = 40.0, 72.0
GROUPS = [("Adds a daily rest", 0, 4, 11), ("Same rests,\n4 to 11 h given up", 1, 4, 11),
          ("Same rests,\n2 to 4 h given up", 1, 2, 4)]


def rest_pricing():
    import rest_mispricing as rm
    a = rm.candidates("r1_gbt")
    out = []
    for lab, neu, lo, hi in GROUPS:
        n, pred, meas, se = rm.cell(a, neu, lo, hi)
        out.append(dict(label=lab, n=n, pred=pred, meas=meas, se=se))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.0, "pdf.fonttype": 42})
    data = json.load(open(CACHE))
    rows = {r["name"]: r for r in data["rows"]}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.8, 1.95), gridspec_kw=dict(width_ratios=[1.5, 1.0]))

    # (a) schedules, hours X0-X1
    n = len(ROWS)
    for _c, a, b in data["windows"]:
        if b > X0 and a < X1:
            ax.axvspan(max(a, X0), min(b, X1), color=COL["window"], alpha=0.13, lw=0, zorder=0)
    for y, (name, short) in zip(range(n - 1, -1, -1), ROWS):
        r = rows[name]
        ax.barh(y, X1 - X0, left=X0, height=0.16, color=COL["drive"], lw=0, zorder=1)
        for s in r["stops"]:
            dw = s["td"] - s["ta"]
            if dw <= 1e-4 or s["td"] < X0 or s["ta"] > X1:
                continue
            c = (COL["rest"] if s["taur"] > 0.5 else COL["charge"] if s["tauc"] > 1e-6
                 else COL["brk"] if s["taub"] > 1e-6 else COL["service"] if s["cust"] else None)
            if c is None:
                continue
            lo, hi = max(s["ta"], X0), min(s["td"], X1)
            ax.barh(y, hi - lo, left=lo, height=0.5, color=c, lw=0, zorder=2)
        for s in r["stops"]:
            if s["cust"] and s["miss"] and X0 <= s["ta"] <= X1:
                ax.plot(s["ta"], y + 0.4, marker="x", ms=4.0, mew=1.1, color=COL["miss"], ls="none", zorder=4)
        t = {s["stop"]: s["ta"] for s in r["stops"]}
        for k in r["changed"]:
            if X0 <= t[k] <= X1:
                ax.plot(t[k], y - 0.4, marker="^", ms=3.0, color="#000000", ls="none", zorder=4)
    ax.set_yticks(range(n - 1, -1, -1))
    ax.set_yticklabels([s for _n, s in ROWS])
    ax.set_ylim(-0.75, n - 0.35)
    ax.set_xlim(X0, X1)
    ax.set_xticks(range(int(X0), int(X1) + 1, 8))
    ax.set_xlabel("Hours since departure")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [Patch(color=COL["charge"], label="Charging"), Patch(color=COL["rest"], label="Daily rest"),
               Patch(color=COL["service"], label="Customer"),
               Patch(color=COL["window"], alpha=0.13, label="Time window"),
               Line2D([], [], marker="x", ls="none", color=COL["miss"], mew=1.1, ms=4.0, label="Missed"),
               Line2D([], [], marker="^", ls="none", color="#000000", ms=3.0, label="Changed by rollout")]
    ax.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False,
              fontsize=6.5, handlelength=1.1, columnspacing=0.8, handletextpad=0.35, borderaxespad=0.1)
    ax.set_title("(a)", loc="left", x=-0.33, y=1.0, fontsize=8, pad=14)

    # (b) cost of resting now: predicted vs measured
    g = rest_pricing()
    ys = list(range(len(g) - 1, -1, -1))
    h = 0.36
    for y, row in zip(ys, g):
        bx.barh(y + h / 2, row["pred"], height=h, color="#BDBDBD", lw=0)
        bx.barh(y - h / 2, row["meas"], height=h, color=COL["charge"], lw=0,
                xerr=row["se"], error_kw=dict(lw=0.7, capsize=1.5, ecolor="#222222"))
        bx.text(row["pred"] + 0.25, y + h / 2, f"+{row['pred']:.1f}", va="center", fontsize=6.5, color="#222222")
        bx.text(row["meas"] + row["se"] + 0.45, y - h / 2, f"+{row['meas']:.1f}", va="center", fontsize=6.5,
                color="#222222")
    bx.set_yticks(ys)
    bx.set_yticklabels([row["label"] for row in g], fontsize=6.8)
    bx.set_xlim(0, 12.5)
    bx.set_xticks([0, 4, 8, 12])
    bx.set_xlabel("Cost of resting now (h)")
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.legend(handles=[Patch(color="#BDBDBD", label="Predicted"), Patch(color=COL["charge"], label="Measured")],
              ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, fontsize=6.5,
              handlelength=1.1, columnspacing=0.8, handletextpad=0.35, borderaxespad=0.1)
    bx.set_title("(b)", loc="left", x=-0.62, y=1.0, fontsize=8, pad=14)

    fig.tight_layout(pad=0.25, w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_where.{ext}"), dpi=300)
    print("wrote fig_where.pdf/.png;", ", ".join(f"{r['label'].splitlines()[0]} n={r['n']}" for r in g))


if __name__ == "__main__":
    main()
