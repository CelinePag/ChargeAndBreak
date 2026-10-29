"""
fig_research_map.py — where the ML work stands: every experiment, by thread and time
=====================================================================================
One lane per research thread, one mark per experiment, left to right in time.
The outcome is shown by the marker's SHAPE and colour (fixed status colours,
never colour alone) and named in the legend; thin grey arrows show what was
continued into what.  Numbers are against the LA unless stated, from the
result files and ML/STATUS.html as of the date in the title.

    python ML/code/fig_research_map.py      # -> ML/figures/research_map.png / .pdf
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.lines import Line2D                                  # noqa: E402
from matplotlib.patches import FancyArrowPatch                       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE, PLANE = "#e1e0d9", "#fcfcfb", "#f4f3ef"
STATUS = {   # fixed status palette + a distinct marker per outcome
    "worked":  dict(c="#0ca30c", m="o", lab="Worked, kept"),
    "partly":  dict(c="#fab219", m="D", lab="Helped, a gap remains"),
    "failed":  dict(c="#d03b3b", m="X", lab="Did not work, given up"),
    "hold":    dict(c="#898781", m="s", lab="Set aside or on hold"),
    "running": dict(c="#2a78d6", m=">", lab="Running now"),
}
PHASES = ["Aug - Sep", "29 Sep", "30 Sep - 1 Oct", "2 - 6 Oct", "20 Oct", "21 - 23 Oct",
          "26 Oct", "27 Oct", "29 Oct (now)"]
LANES = ["Learning from\nthe LA", "Safety layer", "Other routes\nand trucks",
         "Chargers of\nmixed power", "Calling the\nsolver when\nit pays",
         "Beyond the\nteacher:\nplanning and RL"]

# (id, lane, phase, row, title, result, status)
ITEMS = [
    ("clf", 0, 0, 0, "Classifier, sklearn nets", "+1.0 / +1.6 pp; dropped", "failed"),
    ("trees", 0, 0, 1, "Trees learn each action's cost", "kept: +0.7 pp vs LA (test)", "worked"),
    ("pt1", 0, 2, 0, "PyTorch, first try", "+2.4 pp: extra daily rests", "failed"),
    ("pt2", 0, 2, 1, "PyTorch + ranking loss", "kept: +0.5 pp vs LA (test)", "worked"),
    ("orc", 0, 4, 0, "Copy the oracle instead", "worse: +2 / +6 pp vs LA", "failed"),

    ("shield", 1, 0, 0, "Shield: rules + 0.99 guard", "kept: 0 broken runs", "worked"),
    ("noshield", 1, 3, 0, "No shield at all", "5-10 % of runs break a rule", "failed"),
    ("nominal", 1, 3, 1, "Inputs at nominal times", "kept: trees +0.48 → +0.21 pp", "worked"),
    ("mseq", 1, 4, 0, "Pay charge-then-rest time", "kept: fair against the LA", "worked"),
    ("strict", 1, 6, 0, "Strict 13 h rest-start rule", "set aside: 12-22 % break it", "hold"),

    ("long", 2, 0, 0, "Longer routes than trained", "trees: −0.3 pp vs LA", "worked"),
    ("phys", 2, 0, 1, "Other battery or charger kW", "150 kW: +10 to +18 pp", "failed"),
    ("ferry", 2, 0, 2, "Ferries (Arendal tour)", "the only failures left", "failed"),
    ("allphys", 2, 2, 0, "Train on all truck types", "seen ≈ LA; unseen 150 kW +5.5", "partly"),
    ("allphys_pt", 2, 4, 0, "Same with PyTorch", "seen ≈ LA; unseen 150 kW +7.3", "partly"),

    ("mix0", 3, 2, 0, "Uniform-trained, mixed power", "every student loses 5-9 pp", "failed"),
    ("kwin", 3, 2, 1, "Charger-power inputs", "trees got worse", "failed"),
    ("chnet", 3, 3, 0, "Charger-comparison network", "did not replicate on test", "failed"),
    ("mixdata", 3, 3, 1, "LA data on mixed routes", "kept: +2.8 pp vs LA (test)", "partly"),
    ("dagger", 3, 3, 2, "DAgger: ask the LA", "+2.1 pp vs LA, plateau", "partly"),
    ("mixroll", 3, 7, 0, "Rollout, mixed student", "−0.0 pp vs LA (test)", "worked"),
    ("eqroll", 3, 7, 1, "Rollout, equal-power student", "+2.3 pp: data still needed", "partly"),

    ("margin", 4, 3, 0, "Call LA at close rest calls", "+1.7 pp vs LA, 12 calls/route", "partly"),
    ("gate", 4, 4, 0, "Learned gate (offline)", "finds 65-87 % of the loss", "worked"),
    ("gate_cl", 4, 5, 0, "Gate, closed loop", "hand-offs add daily rests", "failed"),
    ("check", 4, 5, 1, "Gate + rollout check", "−0.4 pp vs LA, 8 calls (test)", "worked"),
    ("gate_mix", 4, 5, 2, "Gate on mixed routes", "validation only, on hold", "hold"),

    ("roll1", 5, 1, 0, "Rollout at every stop", "+7.6 %: chased noise", "failed"),
    ("fpi", 5, 1, 1, "Fitted policy iteration", "no gain (+0.004 h)", "failed"),
    ("endgame", 5, 1, 2, "Exact MILP, last shifts", "worse on 8 of 10 routes", "failed"),
    ("flat", 5, 6, 0, "One correction everywhere", "no gain on 639 routes", "failed"),
    ("labels", 5, 6, 1, "Rollout labels at close calls", "they replicate: corr. 0.95", "worked"),
    ("G", 5, 6, 2, "Learned correction, trees", "no held-out gain; on hold", "hold"),
    ("dsr", 5, 7, 0, "Rollout at close calls", "−0.7 pp vs LA (test), no solver", "worked"),
    ("rl1", 5, 7, 1, "RL round 1", "−0.24 h/route vs student", "worked"),
    ("rl2", 5, 8, 1, "RL round 2", "≈ LA alone; −1.2 pp w. rollout", "worked"),
    ("win", 5, 8, 0, "Rollout + window stops", "no gain vs plain rollout (val)", "failed"),
]
ARROWS = [("roll1", "check"), ("check", "dsr"), ("fpi", "G"), ("labels", "dsr"),
          ("labels", "rl1"), ("rl1", "rl2"), ("dsr", "mixroll"), ("mixdata", "mixroll"),
          ("dsr", "win"), ("pt1", "pt2"), ("gate", "gate_cl"), ("gate_cl", "check")]


def main():
    colw, rowh, lanew = 2.45, 0.5, 1.75
    rows = {l: 1 + max(r for _i, ll, _p, r, *_ in ITEMS if ll == l) for l in range(len(LANES))}
    lane_h = {l: rows[l] * rowh + 0.25 for l in rows}
    top_pad, bottom_pad = 1.15, 0.75
    W = lanew + colw * len(PHASES) + 0.2
    H = top_pad + sum(lane_h.values()) + bottom_pad
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(0.25, H - 0.38, "Where the ML work stands, 29 October 2026", fontsize=17,
            color=INK, weight="bold", va="center")
    ax.text(0.25, H - 0.72, "One mark per experiment, left to right in time; the shape and "
            "colour give the outcome. Numbers are against the LA (gap to the oracle, pp) unless "
            "stated. Grey arrows: continued into.", fontsize=10, color=INK2, va="center")

    # phase headers and separators
    y0 = H - top_pad
    for p, name in enumerate(PHASES):
        x = lanew + p * colw
        ax.text(x + 0.12, y0 + 0.12, name, fontsize=9.5, color=MUTED, va="bottom")
        ax.plot([x, x], [bottom_pad, y0], color=GRID, lw=0.6, zorder=0)

    pos = {}
    y = y0
    for l, name in enumerate(LANES):
        h = lane_h[l]
        if l % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - h), W, h, color=PLANE, zorder=-1, lw=0))
        ax.text(0.25, y - h / 2, name, fontsize=10.5, color=INK, weight="bold", va="center")
        for iid, ll, p, r, title, res, st in ITEMS:
            if ll != l:
                continue
            x = lanew + p * colw + 0.22
            yy = y - 0.12 - (r + 0.5) * rowh
            s = STATUS[st]
            ax.scatter([x], [yy], s=95 if s["m"] != "X" else 110, marker=s["m"], color=s["c"],
                       edgecolors=INK2, linewidths=0.6, zorder=3)
            ax.text(x + 0.17, yy + 0.085, title, fontsize=9.2, color=INK, va="center", zorder=4)
            ax.text(x + 0.17, yy - 0.115, res, fontsize=8.3, color=INK2, va="center", zorder=4)
            pos[iid] = (x, yy)
        y -= h

    phase = {iid: p for iid, _l, p, *_ in ITEMS}
    for a, b in ARROWS:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        if phase[b] > phase[a]:
            # leave from the right edge of the source's cell, not across its label
            start, end = (lanew + (phase[a] + 1) * colw - 0.15, y1), (x2 - 0.1, y2)
        else:
            # same column: run just left of the markers
            start, end = (x1 - 0.13, y1), (x2 - 0.13, y2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                     color=MUTED, lw=0.9, alpha=0.75, zorder=2,
                                     shrinkA=2, shrinkB=4))

    handles = [Line2D([0], [0], marker=s["m"], color="none", markerfacecolor=s["c"],
                      markeredgecolor=INK2, markersize=10, label=s["lab"])
               for s in STATUS.values()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.005), ncol=5,
              frameon=False, fontsize=10, handletextpad=0.4, columnspacing=1.8)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"research_map.{ext}"), dpi=130, facecolor=SURFACE)
    print("wrote", os.path.join(OUT, "research_map.png"))


if __name__ == "__main__":
    main()
