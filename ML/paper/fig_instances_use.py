"""
fig_instances_use.py — which routes are solved by what and used for what
========================================================================
A schematic of the paper's data (no results): three groups of routes
(base case, one-at-a-time variants, mixed powers) by route-seed split
(training 1-19, validation 20-21, test 22-25 of each instance class).  For
each set: how many routes, which solvers ran on it (LA, oracle, greedy), and
what the paper uses it for.

Encoding:
  * solid box          = the routes with their STORED travel times (the
                         realization the LA and the oracle were run on);
  * orange dashed box  = the same routes under NEW travel-time draws, run in
                         the simulator only (no solver);
  * chips              = the solvers run on that set.

Counts were checked on 2026-10-30 against data/dataset*.npz (routes with an
LA run), instances/, instances_sens/ and ML/instances_mixed/; uses against
hpc/run_jobs.sh (rl-exit, rl-exit-test, rl-exit-x) and the model metas.
Writes fig_instances_use.pdf/.png next to this file.

    python ML/paper/fig_instances_use.py
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

ORANGE, LA_C, ORACLE_C, GREEDY_C = "#E69F00", "#009E73", "#3A3A3A", "#0072B2"
INK, MUTED, LIGHT = "#1a1a1a", "#666666", "#d4d4d4"
CHIP = {"LA": LA_C, "oracle": ORACLE_C, "greedy": GREEDY_C}

W = 7.4
X_LANE = 0.10
COLS = [1.66, 3.59, 5.52]                 # left edge of the three split columns
CW = 1.80                                 # box width
LINE = 0.135                              # text line height (in)


def _tint(hex_c, a):
    h = hex_c.lstrip("#")
    r, g, b = (int(h[k:k + 2], 16) for k in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(255 - (255 - v) * a) for v in (r, g, b))


# Each cell is a list of items drawn top-down:
#   ("box", count, note, [chips])        stored travel times
#   ("draw", title, note, [chips])       same routes, new travel-time draws
#   ("use", [lines], muted?)             what the set is used for
LANES = [
    dict(title="Base case",
         sub=["36 classes × 25 routes = 900,", "solved in the earlier paper.",
              "70 routes without a feasible", "LA run are left out."],
         h=2.62,
         cells=[
             [("box", "639 routes", "of 684 with a feasible LA run", ["LA"]),
              ("use", ["train the network and the trees on", "the LA's cost of every action"]),
              ("use", ["also train the two networks of", "the mixed-power section"]),
              ("draw", "same 639 routes × 2 new draws", "the network drives, no solver", []),
              ("use", ["expert iteration: the rollout's", "comparisons are the labels"]),
              ("use", ["seeds 17–19 held out to pick", "the training epoch"])],
             [("box", "66 routes", "of 72 with a feasible LA run", ["LA"]),
              ("use", ["stop the training of the network", "and the trees"]),
              ("use", ["check the rollout and each", "expert-iteration round"]),
              ("draw", "same 66 routes × 16 new draws", "simulator only, no solver", []),
              ("use", ["choose the expert-iteration", "round (round 2)"])],
             [("box", "125 routes", "of 144 with a feasible LA run", ["LA", "oracle", "greedy"]),
              ("use", ["main results: gap to the oracle", "and difference to the LA"]),
              ("draw", "same 125 routes × 16 new draws", "simulator only, no solver", []),
              ("use", ["compare the learned policies", "with each other, more precisely"])],
         ]),
    dict(title="One-at-a-time\nvariants",
         sub=["base routes with one change", "for the whole route: battery", "300/700/900 kWh, chargers",
              "150/700/1000 kW or 30/100 km", "apart (8 variants, 12 classes)"],
         h=1.12,
         cells=[
             [("box", "1,274 routes", "8 variants with an LA run", ["LA"]),
              ("use", ["train the two networks of the", "mixed-power section"])],
             [("box", "136 routes", "8 variants with an LA run", ["LA"]),
              ("use", ["stop their training"])],
             [("box", "277 routes", "8 variants with an LA run", ["LA"]),
              ("use", ["not used in the paper"], True)],
         ]),
    dict(title="Mixed powers",
         sub=["base routes (short and medium,", "8 classes) whose chargers get", "different powers; on half of",
              "them, also a new spacing"],
         h=1.60,
         cells=[
             [("box", "238 routes", "of 248 with an LA run", ["LA"]),
              ("use", ["train the mixed-powers network"]),
              ("draw", "all 248 routes × 2 new draws", "the network drives, no solver", []),
              ("use", ["expert iteration of that network", "(run, not in the paper yet)"], True)],
             [("box", "32 routes", "16 per variant", []),
              ("use", ["not used in the paper"], True)],
             [("box", "64 routes", "oracle within 1% on 61 of them", ["LA", "oracle", "greedy"]),
              ("use", ["results on mixed powers:", "gap and where energy is charged"])],
         ]),
]


class Fig:
    def __init__(self, height):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        self.plt, self.Box = plt, FancyBboxPatch
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                             "font.size": 7.5, "pdf.fonttype": 42})
        self.fig = plt.figure(figsize=(W, height))
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, height)
        ax.axis("off")
        self.ax = ax

    def text(self, x, y, s, **kw):
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        return self.ax.text(x, y, s, **kw)

    def rbox(self, x, y, w, h, fc, ec, ls="-", lw=0.9):
        self.ax.add_patch(self.Box((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                   fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))

    def chips(self, x_left, y, names):
        """Solver chips, left to right from x_left."""
        x = x_left
        for nm in names:
            w = 0.052 * len(nm) + 0.12
            self.rbox(x, y - 0.07, w, 0.14, _tint(CHIP[nm], 0.22), CHIP[nm], lw=0.7)
            self.text(x + w / 2, y, nm, ha="center", fontsize=6.6, color=INK, zorder=3)
            x += w + 0.05

    def cell(self, x, y, items, draw=True):
        """Draw a cell top-down from y; return the y where it ends."""
        for it in items:
            kind = it[0]
            if kind in ("box", "draw"):
                _, title, note, chips = it
                h = 0.56 if kind == "box" else 0.40
                y -= 0.04
                if draw:
                    if kind == "box":
                        self.rbox(x, y - h, CW, h, "#ffffff", INK)
                    else:
                        self.rbox(x, y - h, CW, h, _tint(ORANGE, 0.10), ORANGE, ls=(0, (3, 1.6)), lw=1.1)
                    self.text(x + 0.07, y - 0.13, title, fontsize=8 if kind == "box" else 7.4,
                              fontweight="bold")
                    self.text(x + 0.07, y - 0.28, note, fontsize=6.8, color=MUTED)
                    if kind == "box":
                        if chips:
                            self.text(x + 0.07, y - 0.44, "solved by", fontsize=6.8, color=MUTED)
                            self.chips(x + 0.55, y - 0.44, chips)
                        else:
                            self.text(x + 0.07, y - 0.44, "no solver run", fontsize=6.8, color=MUTED)
                y -= h + 0.05
            else:
                lines = it[1]
                muted = len(it) > 2 and it[2]
                col = MUTED if muted else INK
                for k, ln in enumerate(lines):
                    y -= LINE
                    if draw:
                        if k == 0:
                            self.text(x + 0.03, y, "\u2192", fontsize=7.5, color=col)
                        self.text(x + 0.17, y, ln, fontsize=7.2, color=col,
                                  style="italic" if muted else "normal")
                y -= 0.035
        return y


def lane_height(f, lane):
    n = lane["title"].count("\n")
    label = 0.34 + 0.16 * n + 0.125 * len(lane["sub"])
    cells = max(-f.cell(0, 0, items, draw=False) for items in lane["cells"])
    return max(label, cells) + 0.14


def main():
    head = 1.02
    probe = Fig(1.0)
    hs = [lane_height(probe, lane) for lane in LANES]
    probe.plt.close(probe.fig)
    height = head + sum(hs) + 0.02
    f = Fig(height)

    top = height - 0.17
    titles = [("Training routes", "seeds 1\u201319 of each class"),
              ("Validation routes", "seeds 20\u201321"),
              ("Test routes", "seeds 22\u201325")]
    for x, (t, s_) in zip(COLS, titles):
        f.text(x, top, t, fontsize=9, fontweight="bold")
        f.text(x, top - 0.18, s_, fontsize=7.2, color=MUTED)
    f.text(X_LANE, top, "Routes", fontsize=9, fontweight="bold")

    yl = top - 0.46                                       # legend, two lines
    f.rbox(COLS[0], yl - 0.07, 0.28, 0.14, "#ffffff", INK)
    f.text(COLS[0] + 0.36, yl, "the routes with their stored travel times (the ones the LA and "
           "the oracle ran on)", fontsize=7, color=MUTED)
    yl -= 0.22
    f.rbox(COLS[0], yl - 0.07, 0.28, 0.14, _tint(ORANGE, 0.10), ORANGE, ls=(0, (3, 1.6)), lw=1.1)
    f.text(COLS[0] + 0.36, yl, "the same routes under new travel-time draws, run in the "
           "simulator only (no solver)", fontsize=7, color=MUTED)
    f.ax.plot([X_LANE, W - 0.08], [height - head + 0.02] * 2, color=LIGHT, lw=0.7)

    y = height - head
    for k, (lane, h) in enumerate(zip(LANES, hs)):
        n = lane["title"].count("\n")
        f.text(X_LANE, y - 0.14 - 0.08 * n, lane["title"], fontsize=9, fontweight="bold",
               linespacing=1.1)
        ys = y - 0.34 - 0.16 * n
        for ln in lane["sub"]:
            f.text(X_LANE, ys, ln, fontsize=6.8, color=MUTED)
            ys -= 0.125
        for x, items in zip(COLS, lane["cells"]):
            f.cell(x, y - 0.02, items)
        y -= h
        if k < len(LANES) - 1:
            f.ax.plot([X_LANE, W - 0.08], [y + 0.02, y + 0.02], color=LIGHT, lw=0.7)

    for ext in ("pdf", "png"):
        f.fig.savefig(os.path.join(HERE, f"fig_instances_use.{ext}"), dpi=220)
    print("wrote fig_instances_use.pdf/.png")


if __name__ == "__main__":
    main()
