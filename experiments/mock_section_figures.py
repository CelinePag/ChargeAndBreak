"""
mock_section_figures.py — PLACEHOLDER figures for paper Sections 8.3-8.5.

All data here is RANDOM / HAND-SHAPED — these figures exist only to agree on
the visual language and the narrative of each results subsection before the
real runs finish.  Every figure carries a "MOCK DATA" watermark.  The real
versions will be produced from compiled runs (compile_solutions + a section
in paper_figures.py) once the additional_analysis.py blocks have run.

Outputs -> figures/mock/*.png

  fig_sens_battery_power   8.3 headline — battery x charger-power grid
  fig_sens_tornado         8.3 — one-at-a-time axis effects
  fig_diesel_gap           8.4 — naive vs realized electrification penalty
  fig_vss_evpi             8.5 — EEV / RP / WS ladder (VSS + EVPI)
  fig_gamma_frontier       8.5 — ROBU budget frontier (gap vs feasibility)

Style follows paper_figures.py conventions: Okabe-Ito hues, neutral gray
chrome, thin marks, direct labels.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = os.path.join("figures", "mock")
os.makedirs(OUT, exist_ok=True)

# ── paper chrome (mirrors paper_figures.py) ──────────────────────────────────
INK, MUT, GRID = "#000000", "#555555", "#e0e0e0"
BLUE, VERM, ORAN, GREEN, PURP, SKY = ("#0072B2", "#D55E00", "#E69F00",
                                      "#009E73", "#CC79A7", "#56B4E9")
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUT, "axes.linewidth": 0.6,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.color": MUT, "ytick.color": MUT,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
rng = np.random.default_rng(7)


def _watermark(fig):
    fig.text(0.5, 0.5, "MOCK DATA", fontsize=34, color="#cc0000",
             alpha=0.14, ha="center", va="center", rotation=20, weight="bold")


def _save(fig, name):
    _watermark(fig)
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ══════════════════════════════════════════════════════════════════════════════
# 8.3 — battery x charger power grid (headline sensitivity figure)
# ══════════════════════════════════════════════════════════════════════════════

def fig_sens_battery_power():
    batteries = [400, 500, 600, 750]          # kWh (rows, bottom-up)
    powers    = [150, 200, 350, 1000]         # kW  (cols)

    # hand-shaped: penalty falls in both axes and saturates once a full
    # charge hides inside the 45-min break; coupling rises to a plateau.
    penalty = np.array([[14.2, 11.8,  8.9,  8.1],
                        [11.5,  9.3,  6.4,  5.6],
                        [ 9.7,  7.6,  4.9,  4.3],
                        [ 8.2,  6.1,  3.8,  3.4]])
    coupling = np.array([[38, 47, 61, 66],
                         [46, 58, 74, 79],
                         [55, 66, 83, 88],
                         [61, 73, 90, 93]], dtype=float)
    penalty  += rng.normal(0, 0.15, penalty.shape)
    coupling += rng.normal(0, 1.0, coupling.shape)

    cmaps = [LinearSegmentedColormap.from_list("pen", ["#ffffff", VERM]),
             LinearSegmentedColormap.from_list("cpl", ["#ffffff", BLUE])]
    titles = ["(a) Route-duration increase vs diesel (%)",
              "(b) Coupling fraction  $\\Sigma g_i/\\Sigma\\tau^c_i$  (%)"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, Z, cmap, title in zip(axes, [penalty, coupling], cmaps, titles):
        im = ax.imshow(Z, cmap=cmap, origin="lower", aspect="auto")
        ax.set_xticks(range(len(powers)),
                      [f"{p}" for p in powers])
        ax.set_yticks(range(len(batteries)), [f"{b}" for b in batteries])
        ax.set_xlabel("Charger power (kW)")
        ax.set_ylabel("Battery capacity (kWh)")
        ax.set_title(title, loc="left")
        vmax = Z.max()
        for r in range(Z.shape[0]):
            for c in range(Z.shape[1]):
                dark = Z[r, c] > 0.55 * vmax
                ax.text(c, r, f"{Z[r, c]:.1f}", ha="center", va="center",
                        fontsize=7, color="white" if dark else INK)
        # base case marker (500 kWh, 200 kW)
        ax.add_patch(plt.Rectangle((1 - 0.5, 1 - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=1.4))
        ax.text(1, 1 - 0.38, "base", ha="center", va="top",
                fontsize=6, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    fig.suptitle("Sensitivity 8.3 — vehicle/infrastructure grid "
                 "(MCS returns vanish once charging hides in the break)",
                 fontsize=8.5, y=1.04)
    _save(fig, "fig_sens_battery_power.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8.3 — one-at-a-time tornado
# ══════════════════════════════════════════════════════════════════════════════

def fig_sens_tornado():
    # (axis label, low-setting delta %, high-setting delta %); None = n/a
    rows = [
        ("Charger power (1000 / 150 kW)",   -1.8,  6.5),
        ("CS spacing (30 / 90 km)",         -2.1,  4.8),
        ("Battery (750 / 400 kWh)",         -2.4,  3.9),
        ("Travel-time CV (0.10 / 0.25)",    -1.2,  3.1),
        ("No split break (45 min only)",    None,  2.7),
        ("TW penalty beta (1 / 5 h)",       -0.6,  1.9),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lbl, lo, hi) in zip(y, rows):
        if lo is not None:
            ax.barh(yi, lo, height=0.55, color=SKY, edgecolor="white")
            ax.text(lo - 0.15, yi, f"{lo:+.1f}", ha="right", va="center",
                    fontsize=7, color=INK)
        ax.barh(yi, hi, height=0.55, color=ORAN, edgecolor="white")
        ax.text(hi + 0.15, yi, f"{hi:+.1f}", ha="left", va="center",
                fontsize=7, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Change in mean route duration vs base case (%)")
    ax.set_xlim(-4.5, 8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SKY),
                       plt.Rectangle((0, 0), 1, 1, color=ORAN)],
              labels=["favourable setting", "adverse setting"],
              frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Sensitivity 8.3 — one-at-a-time effects "
                 "(oracle objective, short/medium grid)", loc="left")
    _save(fig, "fig_sens_tornado.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8.4 — electrification penalty vs diesel: naive vs realized
# ══════════════════════════════════════════════════════════════════════════════

def fig_diesel_gap():
    spacings = [30, 60, 90]
    naive    = {"Short route":  [11.8, 12.4, 13.6],
                "Medium route": [12.9, 13.5, 14.8]}
    realized = {"Short route":  [2.1, 3.4, 6.2],
                "Medium route": [3.0, 4.5, 7.9]}
    coupled  = {"Short route":  [88, 79, 61],
                "Medium route": [84, 74, 55]}   # % of charge inside breaks

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    w = 0.36
    x = np.arange(len(spacings))
    for ax, cls in zip(axes, naive):
        ax.bar(x - w / 2, naive[cls], w, color="#b0b0b0",
               edgecolor="white", label="naive (+ total charging time)")
        ax.bar(x + w / 2, realized[cls], w, color=GREEN,
               edgecolor="white", label="realized (optimized schedule)")
        for xi, (nv, rv, cp) in enumerate(zip(naive[cls], realized[cls],
                                              coupled[cls])):
            ax.text(xi - w / 2, nv + 0.25, f"{nv:.0f}", ha="center",
                    fontsize=7, color=MUT)
            ax.text(xi + w / 2, rv + 0.25, f"{rv:.0f}", ha="center",
                    fontsize=7, color=INK)
            ax.text(xi, -0.16, f"{cp}% coupled", ha="center", va="top",
                    fontsize=6, color=MUT,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks(x, [f"{s} km" for s in spacings])
        ax.set_title(cls, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 19)
    axes[0].set_ylabel("Route duration vs diesel (%)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("8.4 — breaks absorb charging: the effective electrification "
                 "penalty is a fraction of the naive estimate",
                 fontsize=8.5, y=1.04)
    _save(fig, "fig_diesel_gap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8.5 — VSS / EVPI ladder
# ══════════════════════════════════════════════════════════════════════════════

def fig_vss_evpi():
    classes = ["RshortCfew", "RshortCmany", "RmediumCfew", "RmediumCmany"]
    ws  = np.array([26.5, 31.3, 54.4, 60.2])
    rp  = ws + np.array([1.0, 1.5, 4.3, 6.4])
    eev = rp + np.array([1.8, 2.6, 5.1, 7.2])

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    y = np.arange(len(classes))[::-1]
    for yi, w_, r_, e_ in zip(y, ws, rp, eev):
        ax.plot([w_, e_], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.scatter(w_, yi, s=34, color=INK,  zorder=3, label="_")
        ax.scatter(r_, yi, s=34, color=PURP, zorder=3, label="_")
        ax.scatter(e_, yi, s=34, facecolor="white", edgecolor=MUT,
                   linewidth=1.2, zorder=3, label="_")
    # annotate the decomposition on the widest row (bottom, RmediumCmany)
    yi = y[-1]
    ax.annotate("EVPI", ((ws[-1] + rp[-1]) / 2, yi + 0.28), ha="center",
                fontsize=7, color=PURP)
    ax.annotate("VSS", ((rp[-1] + eev[-1]) / 2, yi + 0.28), ha="center",
                fontsize=7, color=MUT)
    ax.set_yticks(y, classes)
    ax.set_ylim(-0.7, 3.6)
    ax.set_xlabel("Expected objective over common scenarios (h)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=INK,
                          label="WS (perfect hindsight)"),
               plt.Line2D([], [], marker="o", ls="", color=PURP,
                          label="RP (2SP plan + recourse)"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec=MUT,
                          label="EEV (nominal plan + recourse)")]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper center")
    ax.set_title("8.5 — value of the stochastic solution (VSS) and of perfect "
                 "information (EVPI)", loc="left")
    _save(fig, "fig_vss_evpi.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8.5 — ROBU budget frontier
# ══════════════════════════════════════════════════════════════════════════════

def fig_gamma_frontier():
    gammas = np.array([0, 1, 2, 4, 8, 12])          # 12 ~ N (box)
    gap    = np.array([4.1, 6.3, 9.0, 13.2, 17.8, 20.9])   # % vs oracle
    infeas = np.array([21.0, 12.0, 6.0, 2.0, 0.0, 0.0])    # % of runs
    g_sqrt = 3.5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 3.6), sharex=True,
                                   height_ratios=[3, 2])
    ax1.plot(gammas, gap, "-o", color=ORAN, lw=1.6, ms=4, label="ROBU")
    ax1.axhline(20.9, color=VERM, ls="--", lw=1, label="RO (box)")
    ax1.set_ylabel("Realized gap to oracle (%)")
    ax2.plot(gammas, infeas, "-o", color=ORAN, lw=1.6, ms=4)
    ax2.set_ylabel("Infeasible runs (%)")
    ax2.set_xlabel("Uncertainty budget $\\Gamma$")
    for ax in (ax1, ax2):
        ax.axvline(g_sqrt, color=MUT, ls=":", lw=1)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax1.text(g_sqrt + 0.2, 5.0,
             "$\\Gamma=\\sqrt{N}$ (base)", fontsize=7, color=MUT)
    ax1.legend(frameon=False, fontsize=7, loc="lower right")
    ax1.set_title("8.5 — price of robustness: budget $\\Gamma$ trades\n"
                  "nominal performance against feasibility", loc="left")
    _save(fig, "fig_gamma_frontier.png")


if __name__ == "__main__":
    fig_sens_battery_power()
    fig_sens_tornado()
    fig_diesel_gap()
    fig_vss_evpi()
    fig_gamma_frontier()
