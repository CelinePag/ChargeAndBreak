"""
fig_methods_flow.py — how each policy decides at a stop (from the user's sketch)
===============================================================================
A schematic (no data), redrawn from the user's hand sketch of 2026-10-30 and
extended with expert iteration. One row per method, top to bottom in the
order the paper builds them: the LA, the learned network (or trees), its
rollout, expert iteration (training, offline), and the network after expert
iteration, alone and with its rollout.

Encoding, as in the sketch:
  * orange hop  = the simulator drives one leg (real stop-to-stop progress,
                  or a simulated leg inside a rollout future);
  * thin black  = one LA scenario, optimized by a MILP with its travel times
                  known, up to 24 h ahead (no simulator);
  * dot         = a stop where the policy decides;
  * box         = the learned model called at stop i.
Grey arrows between rows say what each method is trained on.

Numbers in the notes are the paper's (main.tex, draft of 2026-10-30).
Writes fig_methods_flow.pdf/.png next to this file.

    python ML/paper/fig_methods_flow.py
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ORANGE, LA_C, SKY, EI_C = "#E69F00", "#009E73", "#56B4E9", "#CC79A7"
INK, MUTED, LIGHT = "#1a1a1a", "#666666", "#cfcfcf"

W = 7.2
X_NAME = 0.10
X0, XI, X24, XEND = 1.62, 2.42, 3.62, 4.92     # route start, stop i, +24 h, end of route
X_NOTE = 5.12
STEP = (XI - X0) / 2                           # leg length on the drawn route


def _tint(hex_c, a):
    h = hex_c.lstrip("#")
    r, g, b = (int(h[k:k + 2], 16) for k in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(255 - (255 - v) * a) for v in (r, g, b))


class Fig:
    def __init__(self, height):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        self.plt, self.Arrow, self.Box = plt, FancyArrowPatch, FancyBboxPatch
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                             "font.size": 8, "pdf.fonttype": 42, "mathtext.fontset": "stix"})
        self.fig = plt.figure(figsize=(W, height))
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, height)
        ax.axis("off")
        self.ax = ax

    # ── primitives ────────────────────────────────────────────────────────────
    def hop(self, x0, x1, y0, y1=None, lw=1.5, color=ORANGE, rad=-0.55, head=True, alpha=1.0, z=3):
        y1 = y0 if y1 is None else y1
        style = "-|>,head_length=3.2,head_width=1.8" if head else "-"
        self.ax.add_patch(self.Arrow((x0 + 0.025, y0 + 0.012), (x1 - 0.025, y1 + 0.012),
                                     connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                                     color=color, lw=lw, alpha=alpha, zorder=z, shrinkA=0, shrinkB=0))

    def arrow(self, xy0, xy1, color=MUTED, lw=0.9, rad=0.0, z=2, head="-|>,head_length=4,head_width=2.2"):
        self.ax.add_patch(self.Arrow(xy0, xy1, connectionstyle=f"arc3,rad={rad}", arrowstyle=head,
                                     color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))

    def dot(self, x, y, ms=3.2, color=INK, z=4):
        self.ax.plot(x, y, marker="o", ms=ms, color=color, zorder=z, ls="none")

    def text(self, x, y, s, **kw):
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        return self.ax.text(x, y, s, **kw)

    def box(self, x, y, w, h, label, fc, ec, fs=7.2):
        self.ax.add_patch(self.Box((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                   fc=fc, ec=ec, lw=0.9, zorder=5))
        self.text(x, y, label, ha="center", fontsize=fs, zorder=6)

    # ── building blocks ───────────────────────────────────────────────────────
    def route(self, y, end_label=True):
        """Stops start, 1, i driven by the simulator; faint rest of the route to its end."""
        ax = self.ax
        ax.plot([XI, XEND], [y, y], color=LIGHT, lw=0.9, zorder=1)
        ax.plot([X0, XI], [y, y], color=INK, lw=1.0, zorder=2)
        xs = [X0, X0 + STEP, XI]
        for a, b in zip(xs[:-1], xs[1:]):
            self.hop(a, b, y)
        for x in xs:
            self.dot(x, y)
        self.text(XI, y - 0.11, "$i$", ha="center", fontsize=8.5)
        ax.plot([XEND, XEND], [y - 0.05, y + 0.05], color=MUTED, lw=1.0, zorder=2)
        if end_label:
            self.text(XEND, y - 0.12, "end", ha="center", fontsize=6.5, color=MUTED)

    def model(self, y, label, colour, x=XI, dy=0.33):
        self.box(x, y + dy, 0.66, 0.20, label, _tint(colour, 0.30), colour)
        self.arrow((XI, y + dy - 0.10), (XI, y + 0.05), color=INK, lw=0.8)

    def fan(self, y, spread, x_to, kind, n_lines=3, labels=None, gap=0.36):
        """Candidate actions from stop i, each followed by `n_lines` futures up to x_to.
        kind 'milp' = thin solid black scenarios; 'sim' = simulated futures with
        orange hops and the policy's decisions as dots."""
        xa = XI + gap
        for k, dy in enumerate(spread):
            ya = y + dy
            self.ax.plot([XI, xa], [y, ya], color=INK, lw=0.9, zorder=2)
            self.dot(xa, ya, ms=2.6)
            if labels:
                if dy >= 0:
                    self.text(xa - 0.04, ya + 0.075, labels[k], ha="right", fontsize=7.2)
                else:
                    self.text(xa + 0.02, ya - 0.10, labels[k], ha="left", fontsize=7.2)
            offs = np.linspace(-0.055, 0.055, n_lines) if n_lines > 1 else [0.0]
            for d in offs:
                if kind == "milp":
                    self.ax.plot([xa, x_to], [ya, ya + d * 1.6], color=INK, lw=0.45, zorder=2)
                else:
                    self.ax.plot([xa, x_to], [ya, ya + d * 1.6], color=_tint(ORANGE, 0.55), lw=0.5,
                                 zorder=2, ls=(0, (2.0, 1.4)))
            if kind == "sim":                       # one future drawn in full: legs + decisions
                xs = np.linspace(xa, x_to, 6)
                for a, b in zip(xs[:-1], xs[1:]):
                    self.hop(a, b, ya, lw=1.1, rad=-0.6)
                for x in xs[1:-1]:
                    self.dot(x, ya, ms=2.2)

    def name(self, y, title, sub, colour=INK):
        n = title.count("\n")
        self.text(X_NAME, y + 0.08 + 0.08 * n, title, fontsize=9, fontweight="bold", color=colour,
                  linespacing=1.15)
        self.text(X_NAME, y - 0.10 - 0.02 * n, sub, fontsize=7.2, color=MUTED, linespacing=1.3)

    def note(self, y, lines):
        self.text(X_NOTE, y, "\n".join(lines), fontsize=7.2, linespacing=1.35)

    def link(self, y_top, y_bot, label):
        """Grey 'trained on' arrow between two rows, in the name column."""
        x = 0.30
        self.arrow((x, y_top), (x, y_bot), color=MUTED, lw=1.0)
        self.text(x + 0.12, (y_top + y_bot) / 2, label, fontsize=7.2, color=MUTED, style="italic")


# ── the rows ──────────────────────────────────────────────────────────────────
def row_la(f, y):
    f.name(y, "Look-ahead (LA)", "77 s per stop", LA_C)
    f.route(y)
    f.fan(y, [0.30, 0.08, -0.30], X24, "milp", n_lines=5, labels=["$a_1$", "$a_2$", "$a_n$"])
    f.text(XI + 0.32, y - 0.11, "$\\vdots$", ha="center", fontsize=8)
    f.ax.plot([X24, X24], [y - 0.42, y + 0.42], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    f.text(X24, y + 0.50, "+24 h", ha="center", fontsize=7, color=MUTED)
    f.text(X24 + 0.06, y + 0.31, "25 scenarios", fontsize=6.8, color=MUTED, va="center")
    f.note(y, ["For every possible action (about 6),",
               "solve one MILP per travel-time scenario,",
               "with that scenario's travel times known,",
               "up to 24 h ahead. Execute the action",
               "with the lowest average cost."])


def row_net(f, y, label, colour, title, sub, note):
    f.name(y, title, sub, colour)
    f.route(y)
    f.model(y, label, colour)
    f.note(y, note)


def row_roll(f, y, label, colour, title, sub, note):
    f.name(y, title, sub, colour)
    f.route(y)
    f.model(y, label, colour, x=XI - 0.14, dy=0.56)
    f.fan(y, [0.30, 0.02, -0.30], XEND, "sim", n_lines=3, labels=["$a_1$", "$a_2$", "$a_4$"])
    f.text(XI + 0.32, y - 0.13, "$\\vdots$", ha="center", fontsize=8)
    f.text((XI + XEND) / 2 + 0.38, y + 0.50, "32 sampled futures, the same for every candidate",
           ha="center", fontsize=6.8, color=MUTED)
    f.note(y, note)


def row_exit(f, y):
    f.name(y, "Expert iteration", "training, 2 rounds\n3 h on 32 cores", EI_C)
    ax = f.ax
    # the network drives a whole training route; its own choice is always executed
    xs = np.linspace(X0, XEND, 7)
    ax.plot([X0, XEND], [y, y], color=INK, lw=1.0, zorder=2)
    for a, b in zip(xs[:-1], xs[1:]):
        f.hop(a, b, y)
    for x in xs:
        f.dot(x, y)
    # decisive stops: the rollout compares the candidates, as labels only
    for xd in (xs[2], xs[4]):
        ax.plot(xd, y, marker="o", ms=7.5, mfc="none", mec=EI_C, mew=1.3, zorder=5, ls="none")
        for dy in (0.26, 0.40, 0.54):
            ax.plot([xd, xd + 0.55], [y + 0.04, y + dy], color=_tint(ORANGE, 0.75), lw=0.8,
                    ls=(0, (2.2, 1.4)), zorder=2)
    f.text((xs[2] + xs[4]) / 2 + 0.25, y + 0.66, "at each decisive stop (circled), the rollout "
           "compares the candidates", fontsize=6.8, color=MUTED, ha="center")
    f.text(xs[3], y - 0.15, "the network's own choice is executed", ha="center", fontsize=6.8,
           color=MUTED)
    # labels -> train the network further -> next round with the new network
    xb, yb = XEND - 0.60, y - 0.48
    f.box(xb, yb, 1.10, 0.22, "train the network further", _tint(EI_C, 0.30), EI_C, fs=6.8)
    f.arrow((XEND, y - 0.07), (xb + 0.56, yb), color=INK, lw=0.8, rad=-0.35)
    f.arrow((xb - 0.56, yb), (X0, y - 0.07), color=EI_C, lw=0.9, rad=-0.12)
    f.text((X0 + xb) / 2 - 0.10, yb - 0.17, "next round: the new network drives", ha="center",
           fontsize=6.8, color=EI_C)
    f.note(y, ["Each round, the network drives the 639",
               "training routes (2 new travel-time draws).",
               "At every decisive stop, the rollout compares",
               "its 4 candidates, but the network's own",
               "choice is executed. Its cost model is then",
               "trained further on the comparisons of all",
               "rounds so far."])


def main():
    rows = [("la", 1.25), ("link", 0.40), ("net", 0.80), ("roll", 1.35), ("link", 0.40),
            ("exit", 1.75), ("link", 0.40), ("net2", 0.80), ("roll2", 1.35)]
    head = 0.55
    height = head + sum(h for _, h in rows) + 0.10
    f = Fig(height)

    top = height - 0.18
    f.text(X_NAME, top, "Method", fontsize=9, fontweight="bold")
    f.text((X0 + XEND) / 2, top, "Deciding at stop $i$", fontsize=9, fontweight="bold", ha="center")
    f.text(X_NOTE, top, "What it does", fontsize=9, fontweight="bold")
    yl = top - 0.24                                    # legend
    f.hop(X0 - 0.05, X0 + 0.30, yl - 0.03, lw=1.3)
    f.text(X0 + 0.36, yl, "simulator drives one leg", fontsize=7, color=MUTED)
    f.ax.plot([X0 + 1.55, X0 + 1.85], [yl, yl], color=INK, lw=0.6)
    f.text(X0 + 1.91, yl, "MILP of one scenario", fontsize=7, color=MUTED)
    f.ax.plot([X0 - 0.15, W - 0.10], [top - 0.40, top - 0.40], color=LIGHT, lw=0.6)

    y = height - head
    links = iter(["trained on the LA's cost of every action at every stop (639 training routes)",
                  "the rollout's comparisons become training labels",
                  "after 2 rounds: the new network"])
    for kind, h in rows:
        yc = y - h / 2
        if kind == "la":
            row_la(f, yc)
        elif kind == "net":
            row_net(f, yc, "network or trees", SKY, "Network or trees", "1.4 ms / 12 ms per stop",
                    ["Predict, for every possible action, how much",
                     "more the LA would expect it to cost than the",
                     "best one, from features of the stop and of",
                     "the action. A safety filter removes unsafe",
                     "actions; execute the cheapest one."])
        elif kind == "roll":
            row_roll(f, yc, "network or trees", SKY, "  + rollout", "about 1 s per stop (trees 2 s)",
                     ["Call the model. If its best actions are",
                      "close (38% of stops), simulate its 4 best",
                      "candidates until the end of the route, the",
                      "model deciding at every simulated stop,",
                      "and execute the one with the lowest",
                      "average. Otherwise, the model's choice."])
        elif kind == "exit":
            f.ax.add_patch(f.Box((0.04, y - h + 0.04), W - 0.08, h - 0.08,
                                 boxstyle="round,pad=0,rounding_size=0.06",
                                 fc="#f6f6f6", ec="none", zorder=0))
            row_exit(f, yc + 0.08)
        elif kind == "net2":
            row_net(f, yc, "new network", EI_C, "Network after\nexpert iteration", "1.4 ms per stop",
                    ["As the network above, with the cost",
                     "model trained further by expert iteration."])
        elif kind == "roll2":
            row_roll(f, yc, "new network", EI_C, "  + rollout", "about 1 s per stop",
                     ["As the rollout above, with the new",
                      "network choosing the candidates and",
                      "deciding at every simulated stop."])
        elif kind == "link":
            f.link(y - 0.02, y - h + 0.02, next(links))
        y -= h

    for ext in ("pdf", "png"):
        f.fig.savefig(os.path.join(HERE, f"fig_methods_flow.{ext}"), dpi=220)
    print("wrote fig_methods_flow.pdf/.png")


if __name__ == "__main__":
    main()
