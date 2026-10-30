"""
fig_where.py — the two "where the improvement comes from" figures (paper)
=========================================================================
fig_schedule_zoom.pdf: the schedules of one test route around its third and
    fourth customers (hours 40-72), from fig_schedule.json (fig_schedule.py
    re-drives the policies and checks them against the stored test runs):
    oracle, LA, the PyTorch student and its rollout; daily rests, charging,
    customer service (drawn before a rest taken at the same stop), time
    windows, missed windows, decisions changed by the rollout.
fig_rest_cost.pdf: extra hours to the destination when the driver rests now
    instead of taking the student's action, at the decisive states of the
    trees' rollout labels (data/rl_api_r1_gbt; ML/scratch/rest_mispricing.py):
    as estimated by the trees (trained on the LA's scores) and as simulated to
    the destination, for rests that add a daily rest to the route and rests
    that keep the number of daily rests.

Each file is drawn at its printed size (the manuscript sets them side by side
in two minipages of 0.58 and 0.40 \\textwidth), so the fonts print at 7 pt.

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

TEXTWIDTH_IN = 4.80            # LNCS \textwidth, 122 mm
HEIGHT_IN = 1.85
ROWS = [("Oracle (hindsight)", "Oracle"), ("LA", "LA"), ("Learned network", "Network"),
        ("Network + rollout", "Network + rollout")]
X0, X1 = 40.0, 72.0
GROUPS = [("Rest adds a\ndaily rest", 0), ("Same number\nof daily rests", 1)]


def _style(plt):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.0, "pdf.fonttype": 42})


def schedule(plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from rl_env import load_route

    data = json.load(open(CACHE))
    fd = load_route(data["instance"])[0]
    serv = {int(k): float(v) for k, v in fd["S"].items()}
    rows = {r["name"]: r for r in data["rows"]}
    fig, ax = plt.subplots(figsize=(0.58 * TEXTWIDTH_IN, HEIGHT_IN))
    n = len(ROWS)
    for _c, a, b in data["windows"]:
        if b > X0 and a < X1:
            ax.axvspan(max(a, X0), min(b, X1), color=COL["window"], alpha=0.13, lw=0, zorder=0)
    for y, (name, _short) in zip(range(n - 1, -1, -1), ROWS):
        r = rows[name]
        ax.barh(y, X1 - X0, left=X0, height=0.16, color=COL["drive"], lw=0, zorder=1)
        for s in r["stops"]:
            if s["td"] - s["ta"] <= 1e-4 or s["td"] < X0 or s["ta"] > X1:
                continue
            segs = []
            t = s["ta"]
            if s["cust"]:                   # service starts on arrival, before a rest there
                sv = serv.get(s["stop"], 0.0)
                segs.append((t, t + sv, COL["service"]))
                t += sv
            c = (COL["rest"] if s["taur"] > 0.5 else COL["charge"] if s["tauc"] > 1e-6
                 else COL["brk"] if s["taub"] > 1e-6 else None)
            if c is not None and s["td"] > t + 1e-4:
                segs.append((t, s["td"], c))
            for lo, hi, col in segs:
                lo, hi = max(lo, X0), min(hi, X1)
                if hi > lo:
                    ax.barh(y, hi - lo, left=lo, height=0.5, color=col, lw=0, zorder=2 if col != COL["service"] else 3)
        for s in r["stops"]:
            if s["cust"] and s["miss"] and X0 <= s["ta"] <= X1:
                ax.plot(s["ta"] + 0.25, y + 0.42, marker="x", ms=3.8, mew=1.0, color=COL["miss"], ls="none",
                        zorder=4)
        t = {s["stop"]: s["ta"] for s in r["stops"]}
        for k in r["changed"]:
            if X0 <= t[k] <= X1:
                ax.plot(t[k], y - 0.42, marker="^", ms=2.8, color="#000000", ls="none", zorder=4)
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
               Line2D([], [], marker="x", ls="none", color=COL["miss"], mew=1.0, ms=3.8, label="Missed"),
               Line2D([], [], marker="^", ls="none", color="#000000", ms=2.8, label="Rollout change")]
    ax.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(-0.42, 1.0), frameon=False,
              fontsize=6.3, handlelength=1.0, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.1)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_schedule_zoom.{ext}"), dpi=300)
    plt.close(fig)


def rest_cost(plt):
    from matplotlib.patches import Patch
    import rest_mispricing as rm

    a = rm.candidates("r1_gbt")
    g = []
    for lab, neu in GROUPS:                 # all allowances given up (0-11 h)
        n, pred, meas, se = rm.cell(a, neu, 0, 12)
        g.append(dict(label=lab, n=n, pred=pred, meas=meas, se=se))
    fig, bx = plt.subplots(figsize=(0.40 * TEXTWIDTH_IN, HEIGHT_IN))
    ys = [1, 0]
    h = 0.34
    for y, r in zip(ys, g):
        bx.barh(y + h / 2 + 0.02, r["pred"], height=h, color="#BDBDBD", lw=0)
        bx.barh(y - h / 2 - 0.02, r["meas"], height=h, color=COL["charge"], lw=0,
                xerr=r["se"], error_kw=dict(lw=0.7, capsize=1.5, ecolor="#222222"))
        bx.text(r["pred"] + 0.25, y + h / 2 + 0.02, f"{r['pred']:.1f} h", va="center", fontsize=6.5,
                color="#222222")
        bx.text(r["meas"] + r["se"] + 0.35, y - h / 2 - 0.02, f"{r['meas']:.1f} h", va="center",
                fontsize=6.5, color="#222222")
    bx.set_yticks(ys)
    bx.set_yticklabels([r["label"] for r in g])
    bx.set_ylim(-0.6, 1.6)
    bx.set_xlim(0, 12.5)
    bx.set_xticks([0, 4, 8, 12])
    bx.set_xlabel("Extra hours to the destination")
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.legend(handles=[Patch(color="#BDBDBD", label="Estimated from the LA"),
                       Patch(color=COL["charge"], label="Simulated")],
              ncol=1, loc="lower left", bbox_to_anchor=(-0.05, 1.0), frameon=False, fontsize=6.5,
              handlelength=1.0, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_rest_cost.{ext}"), dpi=300)
    plt.close(fig)
    return g


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    schedule(plt)
    g = rest_cost(plt)
    print("wrote fig_schedule_zoom.pdf and fig_rest_cost.pdf;",
          ", ".join(f"{r['label'].replace(chr(10), ' ')}: n={r['n']}" for r in g))


if __name__ == "__main__":
    main()
