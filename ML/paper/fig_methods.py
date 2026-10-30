"""
fig_methods.py — what each policy of the paper uses, how far it looks, its class
================================================================================
A schematic (no data): one row per policy, in the order the paper introduces
them (greedy, LA, the network learned from the LA, its rollout, the network
trained by expert iteration on the rollout, the oracle); a horizon glyph from
the current stop i towards the destination (solid lines: travel times known to
the optimizer; dashed lines: sampled futures that the simulated policy does
not know when it decides; a dot: no look-ahead at decision time); what the
policy decides with; what it is trained on; and its class in Powell's
framework (PFA / VFA / DLA).  Drawn at the printed size (\\textwidth).
Writes fig_methods.pdf/.png.

    python ML/paper/fig_methods.py
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 4.80, 1.45

ROWS = [
    ("Greedy heuristic", "greedy", "fixed rules", "--", "PFA"),
    ("LA", "la", "MILP, 25 scenarios", "--", "DLA"),
    ("Learned network", "dot", "learned action costs", "LA costs", "VFA"),
    ("Rollout", "roll", "network, 32 futures", "--", "DLA"),
    ("Expert iteration", "dot", "learned action costs", "rollout", "VFA"),
    ("Oracle", "oracle", "MILP, realized times", "--", "benchmark"),
]
G0, G1 = 1.02, 2.52                    # horizon glyph, inches
X_USES, X_TRAIN, X_CLASS = 2.66, 3.80, 4.52
U_DEP, U_I, U_24, U_END = 0.0, 2.0, 4.6, 10.0
INK, MUTED = "#222222", "#666666"


def gx(u):
    return G0 + (G1 - G0) * u / 10.0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    top = H - 0.10
    ax.text(0.03, top, "Policy", fontweight="bold", va="center")
    ax.text((G0 + G1) / 2, top, "Looks ahead from stop $i$", fontweight="bold", va="center", ha="center")
    ax.text(X_USES, top, "Decides with", fontweight="bold", va="center")
    ax.text(X_TRAIN, top, "Trained on", fontweight="bold", va="center")
    ax.text(X_CLASS, top, "Class", fontweight="bold", va="center", ha="center")
    ya = top - 0.17
    for u, lab in ((U_DEP, "start"), (U_I, "$i$"), (U_24, "+24 h"), (U_END, "end")):
        ax.text(gx(u), ya, lab, ha="center", va="center", fontsize=6.3, color=MUTED)
        ax.plot([gx(u), gx(u)], [0.05, ya - 0.07], color="#DDDDDD", lw=0.5, zorder=0)

    step = (ya - 0.12) / len(ROWS)
    for k, (name, kind, uses, trained, cls) in enumerate(ROWS):
        y = ya - 0.12 - step * (k + 0.5)
        ax.text(0.03, y, name, va="center")
        ax.text(X_USES, y, uses, va="center", fontsize=6.6)
        ax.text(X_TRAIN, y, trained if trained != "--" else "\u2013", va="center", fontsize=6.6, color=INK if trained != "--" else MUTED)
        ax.text(X_CLASS, y, cls, va="center", ha="center", fontsize=7 if len(cls) < 5 else 6.3)
        xi = gx(U_I)
        if kind in ("dot", "greedy", "la", "roll"):
            ax.plot(xi, y, marker="o", ms=2.6, color=INK, zorder=3)
        if kind == "greedy":
            ax.plot([xi, gx(2.9)], [y, y], color=INK, lw=0.9)
            ax.text(gx(3.05), y, "next charger", va="center", fontsize=6.0, color=MUTED)
        elif kind == "la":
            for d in (-0.044, -0.022, 0.0, 0.022, 0.044):
                ax.plot([xi, gx(U_24)], [y, y + d], color=INK, lw=0.6)
            ax.text(gx(U_24) + 0.04, y, "known per scenario", va="center", fontsize=6.0, color=MUTED)
        elif kind == "roll":
            for d in (-0.044, -0.022, 0.0, 0.022, 0.044):
                ax.plot([xi, gx(U_END)], [y, y + d], color=INK, lw=0.6, ls=(0, (2.2, 1.4)))
            ax.text(gx(6.3), y - 0.080, "unknown when deciding", ha="center", va="center",
                    fontsize=6.0, color=MUTED)
        elif kind == "oracle":
            ax.plot([gx(U_DEP), gx(U_END)], [y, y], color=INK, lw=0.9)
            ax.plot(gx(U_DEP), y, marker="o", ms=2.6, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_methods.{ext}"), dpi=300)
    print("wrote fig_methods.pdf/.png")


if __name__ == "__main__":
    main()
