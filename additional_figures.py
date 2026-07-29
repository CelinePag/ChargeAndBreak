"""
additional_figures.py — REAL tables & figures for the additional analyses
(paper Sections 8.3-8.5), built from whatever runs currently exist.

Unlike experiments/mock_section_figures.py (random placeholder data), this
script reads solutions/ (+ oracle caches, results_vss/) and renders the
paper artefacts with the data available NOW; cells or panels whose runs have
not finished yet are shown explicitly as "pending", mirroring the
paper_figures.py convention of drawing the full grid with empty slots.

Outputs
  figures/additional_diesel_gap.png|pdf     §8.4 figure
  figures/additional_diesel_stats.csv       §8.4 per-class detail
  tables/additional_diesel.tex              §8.4 table
  figures/additional_sens_effects.png|pdf   §8.3 one-at-a-time (preliminary)
  figures/additional_sens_stats.csv
  tables/additional_sensitivity.tex
  figures/additional_gamma_frontier.png|pdf §8.5 frontier (endpoints only yet)
  tables/additional_vss.tex                 §8.5 VSS/EVPI (skeleton until
  figures/additional_vss_stats.csv                results_vss/ fills up)

Usage
  python additional_figures.py                 # all sections
  python additional_figures.py --section diesel|sensitivity|gamma|vss
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from settings import T_START

# ── paper chrome (mirrors paper_figures.py) ──────────────────────────────────
INK, MUT, GRID = "#000000", "#555555", "#e0e0e0"
BLUE, VERM, ORAN, GREEN, PURP = ("#0072B2", "#D55E00", "#E69F00",
                                 "#009E73", "#CC79A7")
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUT, "axes.linewidth": 0.6,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.color": MUT, "ytick.color": MUT,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

COMBOS = [("short", "few"), ("short", "many"), ("medium", "few"),
          ("medium", "many")]
TWS    = ["tight", "medium", "large", "none"]
SEEDS  = range(1, 11)

_RTAG = {"short": "Rshort", "medium": "Rmedium", "long": "Rlong"}
_CTAG = {"few": "Cfew", "medium": "Cmedium", "many": "Cmany"}


def _stem(route, cust, tw, seed) -> str:
    return f"{_RTAG[route]}{_CTAG[cust]}T{tw}_{seed}"


def _latest(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def _load(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _greedy(stem: str, tag: str | None = None) -> dict | None:
    """Latest greedy solution for a (possibly tagged) instance.  Accepts both
    the '__tag' stem (orchestrator batch) and the runner-normalised '_tag'."""
    pats = ([f"solutions/{stem}_GREEDY_*.json"] if tag is None else
            [f"solutions/{stem}__{tag}_GREEDY_*.json",
             f"solutions/{stem}_{tag}_GREEDY_*.json"])
    for p in pats:
        f = _latest(p)
        if f:
            d = _load(f)
            if d and d.get("duration_h") is not None:
                infeas = bool((d.get("metrics") or {}).get("run_infeasible"))
                return dict(duration=float(d["duration_h"]), infeasible=infeas)
    return None


def _oracle(stem: str, tag: str | None = None) -> dict | None:
    """Oracle cache -> duration (h), total/coupled charging time (h)."""
    names = ([f"solutions/oracle_{stem}.json"] if tag is None else
             [f"solutions/oracle_{stem}__{tag}.json",
              f"solutions/oracle_{stem}_{tag}.json"])
    for n in names:
        d = _load(n)
        if d and d.get("feasible") and d.get("sol"):
            sol = d["sol"]
            ta_N = float(sol[-1]["ta"])
            tauc = sum(float(s.get("tauc") or 0.0) for s in sol)
            g    = sum(float(s.get("g")    or 0.0) for s in sol)
            return dict(duration=ta_N - T_START, tauc=tauc, g=g,
                        gap=float(d.get("gap") or 0.0))
    return None


def _fmt(x, spec=".1f", dash="--"):
    return format(x, spec) if x is not None and np.isfinite(x) else dash


def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join("figures", f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Figure    : figures/{name}.png|pdf")


def _write_csv(name, header, rows):
    path = os.path.join("figures", name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Stats CSV : {path}")


def _write_tex(name, text):
    os.makedirs("tables", exist_ok=True)
    path = os.path.join("tables", name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  LaTeX     : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# §8.4 — DIESEL
# ══════════════════════════════════════════════════════════════════════════════

def section_diesel():
    print("== Sec 8.4 diesel ==")
    per_class: dict[str, dict[str, list]] = {}
    detail = []
    for route, cust in COMBOS:
        for tw in TWS:
            for seed in SEEDS:
                st       = _stem(route, cust, tw, seed)
                ev_o     = _oracle(st)
                di_o     = _oracle(st, "diesel")
                ev_g     = _greedy(st)
                di_g     = _greedy(st, "diesel")

                pen_o = pen_g = naive = coup = None
                if ev_o and di_o and di_o["duration"] > 0:
                    pen_o = 100 * (ev_o["duration"] / di_o["duration"] - 1)
                    naive = 100 * (ev_o["tauc"] / di_o["duration"])
                    coup  = (100 * ev_o["g"] / ev_o["tauc"]
                             if ev_o["tauc"] > 1e-6 else None)
                if (ev_g and di_g and not ev_g["infeasible"]
                        and not di_g["infeasible"] and di_g["duration"] > 0):
                    pen_g = 100 * (ev_g["duration"] / di_g["duration"] - 1)

                detail.append([route, cust, tw, seed,
                               _fmt(di_o and di_o["duration"], ".3f", ""),
                               _fmt(ev_o and ev_o["duration"], ".3f", ""),
                               _fmt(pen_o, ".2f", ""), _fmt(pen_g, ".2f", ""),
                               _fmt(naive, ".2f", ""), _fmt(coup, ".1f", "")])
                d = per_class.setdefault(route, dict(pen_o=[], pen_g=[],
                                                     naive=[], coup=[],
                                                     dur_d=[], dur_e=[]))
                d["pen_o"].append(pen_o); d["pen_g"].append(pen_g)
                d["naive"].append(naive); d["coup"].append(coup)
                d["dur_d"].append(di_o and di_o["duration"])
                d["dur_e"].append(ev_o and ev_o["duration"])

    _write_csv("additional_diesel_stats.csv",
               ["route", "cust", "tw", "seed", "diesel_oracle_h",
                "ev_oracle_h", "pen_oracle_%", "pen_greedy_%",
                "naive_pen_%", "coupling_%"], detail)

    # ── figure: naive vs greedy vs oracle penalty, per route class ───────────
    routes = [r for r in ("short", "medium") if r in per_class]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    w, x = 0.26, np.arange(len(routes))
    series = [("naive (+ total charging time)", "naive", "#b0b0b0"),
              ("greedy EV vs greedy diesel",    "pen_g", BLUE),
              ("oracle EV vs oracle diesel",    "pen_o", INK)]
    for k, (lbl, key, col) in enumerate(series):
        vals = [_mean(per_class[r][key]) for r in routes]
        pos  = x + (k - 1) * w
        ax.bar(pos, [v or 0 for v in vals], w, color=col,
               edgecolor="white", label=lbl)
        for p, v in zip(pos, vals):
            if v is not None:
                ax.text(p, v + 0.25, f"{v:.1f}", ha="center", fontsize=7,
                        color=INK)
    for xi, r in enumerate(routes):
        c = _mean(per_class[r]["coup"])
        n = sum(1 for v in per_class[r]["pen_o"] if v is not None)
        ax.text(xi, -0.14, f"{_fmt(c, '.0f')}% coupled  (n={n})",
                ha="center", va="top", fontsize=6.5, color=MUT,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(x, [f"{r.capitalize()} route" for r in routes])
    ax.set_ylabel("Route duration vs diesel (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Electrification penalty: naive estimate vs optimized "
                 "schedules", loc="left")
    _save(fig, "additional_diesel_gap")

    # ── LaTeX table ──────────────────────────────────────────────────────────
    lines = [
        r"\begin{table}[ht]\centering",
        r"\caption{EV vs diesel route duration (\%), same instances and "
        r"realizations; naive = diesel + total EV charging time. "
        r"Coupling = share of charging time credited inside a mandatory "
        r"break ($\Sigma g_i/\Sigma\tau^c_i$, hindsight optimum).}",
        r"\label{tab:diesel}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Route & Diesel (h) & EV (h) & Naive (\%) & Greedy (\%) & "
        r"Oracle (\%) & Coupling (\%) \\",
        r"\hline",
    ]
    for r in routes:
        d = per_class[r]
        lines.append(
            f"{r.capitalize()} & {_fmt(_mean(d['dur_d']))} & "
            f"{_fmt(_mean(d['dur_e']))} & {_fmt(_mean(d['naive']))} & "
            f"{_fmt(_mean(d['pen_g']))} & {_fmt(_mean(d['pen_o']))} & "
            f"{_fmt(_mean(d['coup']), '.0f')} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_diesel.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §8.3 — SENSITIVITY (one-at-a-time)
# ══════════════════════════════════════════════════════════════════════════════

# (axis label, variant tag, planned?) — extend as new sweeps land
_SENS_ROWS = [
    ("CS spacing 30 km",        "cs30",   True),
    ("CS spacing 90 km",        "cs90",   True),
    ("Charger power 150 kW",    "kw150",  True),
    ("Charger power 350 kW",    "kw350",  True),
    ("Charger power 1000 kW",   "kw1000", True),
    ("Travel-time CV 0.10",     "cv0.1",  True),
    ("Travel-time CV 0.25",     "cv0.25", True),
    ("TW penalty beta 5 h",     "beta5",  True),
    ("Battery 400 kWh",         "bat400", False),   # gated on plumbing
    ("Battery 750 kWh",         "bat750", False),
    ("No split break",          "nosplit", False),
]


def section_sensitivity():
    print("== Sec 8.3 sensitivity ==")
    rows_out, fig_rows = [], []
    for label, tag, planned in _SENS_ROWS:
        dg, do, n_g, n_o = [], [], 0, 0
        for route, cust in COMBOS:
            for tw in TWS:
                for seed in SEEDS:
                    st = _stem(route, cust, tw, seed)
                    bg, vg = _greedy(st), _greedy(st, tag)
                    if (bg and vg and not bg["infeasible"]
                            and not vg["infeasible"] and bg["duration"] > 0):
                        dg.append(100 * (vg["duration"] / bg["duration"] - 1))
                        n_g += 1
                    bo, vo = _oracle(st), _oracle(st, tag)
                    if bo and vo and bo["duration"] > 0:
                        do.append(100 * (vo["duration"] / bo["duration"] - 1))
                        n_o += 1
        g_mean, o_mean = _mean(dg), _mean(do)
        status = ("pending (needs code)" if not planned and n_g == 0 else
                  "pending" if n_g == 0 else
                  f"greedy n={n_g}" + (f", oracle n={n_o}" if n_o else
                                       ", oracle pending"))
        rows_out.append([label, tag, _fmt(g_mean, ".2f", ""), n_g,
                         _fmt(o_mean, ".2f", ""), n_o, status])
        fig_rows.append((label, g_mean, o_mean, n_g, planned))

    _write_csv("additional_sens_stats.csv",
               ["axis", "tag", "greedy_delta_%", "n_greedy",
                "oracle_delta_%", "n_oracle", "status"], rows_out)

    # ── figure: available bars + explicit pending slots ──────────────────────
    fig, ax = plt.subplots(figsize=(5.8, 0.34 * len(fig_rows) + 1.2))
    y = np.arange(len(fig_rows))[::-1]

    def _annot(yi, v, txt, col):
        # small-magnitude bars annotate to the RIGHT of zero so the text
        # never spills over the y-axis labels
        if abs(v) < 1.0:
            ax.text(0.25, yi, txt, ha="left", va="center", fontsize=6.5,
                    color=col)
        else:
            ax.text(v + np.sign(v) * 0.15, yi, txt,
                    ha="left" if v >= 0 else "right", va="center",
                    fontsize=6.5, color=col)

    vals = [v for (_, g, o, _, _) in fig_rows
            for v in (g, o) if v is not None]
    for yi, (label, g, o, n_g, planned) in zip(y, fig_rows):
        if o is not None:
            ax.barh(yi, o, height=0.55, color=INK, edgecolor="white")
            _annot(yi, o, f"{o:+.1f}", INK)
        elif g is not None:
            ax.barh(yi, g, height=0.55, color=BLUE, alpha=0.55,
                    edgecolor="white")
            _annot(yi, g, f"{g:+.1f} (greedy, n={n_g})", MUT)
        else:
            note = "pending" if planned else "pending (needs code)"
            ax.text(0.25, yi, note, ha="left", va="center",
                    fontsize=6.5, color=MUT, style="italic")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(min(-2.0, (min(vals) if vals else 0) - 1),
                max(4.0, (max(vals) if vals else 0) + 4))
    ax.set_yticks(y, [r[0] for r in fig_rows])
    ax.set_xlabel("Change in mean route duration vs base case (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=INK),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.55)]
    ax.legend(handles, ["oracle", "greedy (preliminary)"],
              frameon=False, fontsize=7, loc="lower right")
    ax.set_title("One-at-a-time sensitivity (data available so far)",
                 loc="left")
    _save(fig, "additional_sens_effects")

    lines = [
        r"\begin{table}[ht]\centering",
        r"\caption{One-at-a-time sensitivity: mean change in route duration "
        r"vs the base case (\%). Preliminary cells use greedy; final values "
        r"use the hindsight optimum.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{lrrl}", r"\hline",
        r"Axis & Greedy $\Delta$ (\%) & Oracle $\Delta$ (\%) & Status \\",
        r"\hline",
    ]
    for label, tag, g, n_g, o, n_o, status in rows_out:
        lines.append(f"{label} & {g or '--'} & {o or '--'} & {status} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_sensitivity.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §8.5 — GAMMA FRONTIER (endpoints from base case until the sweep runs)
# ══════════════════════════════════════════════════════════════════════════════

def section_gamma():
    print("== Sec 8.5 gamma frontier ==")
    # sweep points (from tagged runs, none yet) --------------------------------
    sweep = {}
    for gam in (0, 1, 2, 4, 8):
        gaps, infeas, tot = [], 0, 0
        for route, cust in COMBOS:
            if route != "short":
                continue
            for tw in TWS:
                for seed in SEEDS:
                    st = _stem(route, cust, tw, seed)
                    f = (_latest(f"solutions/{st}__g{gam}_ROBU_*.json")
                         or _latest(f"solutions/{st}_g{gam}_ROBU_*.json"))
                    if not f:
                        continue
                    d = _load(f) or {}
                    m = d.get("metrics") or {}
                    tot += 1
                    if m.get("run_infeasible"):
                        infeas += 1
                    o = _oracle(st)
                    if o and d.get("duration_h") and not m.get(
                            "run_infeasible"):
                        gaps.append(100 * (d["duration_h"] / o["duration"]
                                           - 1))
        if tot:
            sweep[gam] = (_mean(gaps), 100 * infeas / tot, tot)

    # base-case endpoints from paper_gap_stats.csv -----------------------------
    end = {}
    try:
        with open("figures/paper_gap_stats.csv", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["route_class"] != "short":
                    continue
                m = row["method"]
                if m not in ("RO", "ROBU"):
                    continue
                nf = int(row["n_feasible"]); ni = int(
                    row["n_infeasible_excluded"])
                e = end.setdefault(m, dict(g=[], w=[], inf=0, tot=0))
                if row["gap_mean_%"]:
                    e["g"].append(float(row["gap_mean_%"])); e["w"].append(nf)
                e["inf"] += ni; e["tot"] += nf + ni
    except OSError:
        pass

    cats = ["0", "1", "2", "4", "8", r"$\sqrt{N}$" + "\n(base ROBU)",
            r"$N$" + "\n(box = RO)"]
    xs   = np.arange(len(cats))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 3.6), sharex=True,
                                   height_ratios=[3, 2])
    for i, gam in enumerate((0, 1, 2, 4, 8)):
        if gam in sweep:
            ax1.plot(xs[i], sweep[gam][0], "o", color=ORAN, ms=5)
            ax2.plot(xs[i], sweep[gam][1], "o", color=ORAN, ms=5)
        else:
            for ax in (ax1, ax2):
                ax.text(xs[i], 0.5, "pending", rotation=90, ha="center",
                        va="center", fontsize=6.5, color=MUT, style="italic",
                        transform=ax.get_xaxis_transform())
    for m, xi, col in (("ROBU", xs[5], ORAN), ("RO", xs[6], VERM)):
        e = end.get(m)
        if e and e["g"]:
            gap = float(np.average(e["g"], weights=e["w"]))
            inf = 100 * e["inf"] / e["tot"] if e["tot"] else None
            ax1.plot(xi, gap, "s", color=col, ms=6)
            ax1.annotate(f"{gap:.1f}", (xi, gap), xytext=(-4, -10),
                         textcoords="offset points", ha="right", fontsize=7)
            if inf is not None:
                ax2.plot(xi, inf, "s", color=col, ms=6)
                ax2.annotate(f"{inf:.1f}", (xi, inf), xytext=(-4, 4),
                             textcoords="offset points", ha="right",
                             fontsize=7)
    ax1.margins(y=0.2)
    ax2.margins(y=0.2)
    ax1.set_ylabel("Realized gap to oracle (%)")
    ax2.set_ylabel("Infeasible runs (%)")
    ax2.set_xticks(xs, cats)
    ax2.set_xlabel(r"Uncertainty budget $\Gamma$")
    for ax in (ax1, ax2):
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax1.set_title("Price of robustness (short routes) — sweep pending, "
                  "endpoints from base case", loc="left")
    _save(fig, "additional_gamma_frontier")


# ══════════════════════════════════════════════════════════════════════════════
# §8.5 — VSS / EVPI
# ══════════════════════════════════════════════════════════════════════════════

def section_vss():
    print("== Sec 8.5 vss/evpi ==")
    agg: dict[tuple, dict[str, list]] = {}
    for f in glob.glob("results_vss/*_vss.json"):
        d = _load(f)
        if not d:
            continue
        s = d.get("summary", {})
        inst = s.get("instance", "")
        cls  = inst.split("T")[0] if "T" in inst else inst
        a = agg.setdefault(cls, dict(ws=[], rp=[], eev=[], vss=[], evpi=[]))
        for k in a:
            a[k].append(s.get(f"{k}_mean") if k in ("ws", "rp", "eev")
                        else s.get(k))

    rows, lines = [], [
        r"\begin{table}[ht]\centering",
        r"\caption{Value of the stochastic solution (VSS) and expected value "
        r"of perfect information (EVPI), common-random scenarios.}",
        r"\label{tab:vss}",
        r"\begin{tabular}{lrrrrrr}", r"\hline",
        r"Class & WS (h) & RP (h) & EEV (h) & VSS (h) & EVPI (h) & n \\",
        r"\hline",
    ]
    for route, cust in COMBOS:
        cls = f"{_RTAG[route]}{_CTAG[cust]}"
        a = agg.get(cls)
        n = len(a["ws"]) if a else 0
        vals = [(_mean(a[k]) if a else None)
                for k in ("ws", "rp", "eev", "vss", "evpi")]
        rows.append([cls] + [_fmt(v, ".2f", "") for v in vals] + [n])
        cells = " & ".join(_fmt(v, ".2f") for v in vals)
        lines.append(f"{cls} & {cells} & {n or '--'} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_vss.tex", "\n".join(lines))
    _write_csv("additional_vss_stats.csv",
               ["class", "ws_h", "rp_h", "eev_h", "vss_h", "evpi_h", "n"],
               rows)
    if not agg:
        print("  (results_vss/ is empty — table is a skeleton; run "
              "'python additional_analysis.py vss')")


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = dict(diesel=section_diesel, sensitivity=section_sensitivity,
                 gamma=section_gamma, vss=section_vss)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real tables/figures for the "
                                             "additional analyses (8.3-8.5)")
    ap.add_argument("--section", default="all",
                    choices=["all", *_SECTIONS])
    args = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    for name, fn in _SECTIONS.items():
        if args.section in ("all", name):
            fn()
