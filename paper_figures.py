"""
paper_figures.py — Publication-quality aggregate figures from solutions/
=========================================================================
Compiles every finished run in solutions/ into one figure showing, per
instance family (route x customers x time-window class) and per method,
the distribution of the gap to the hindsight oracle across seeds.

Reuses the loading + gap annotation logic of compile_solutions.py, so the
numbers are identical to the Excel/LaTeX tables:
  gap_pen   — duration-based gap with window penalties on both sides
              (the paper's objective-function gap; default)
  gap_nopen — pure route-duration gap, window penalties excluded

Figure layout
-------------
  facet rows    : route class (short / medium / long) — only classes present
  facet columns : customers class (few / medium / many)
  x axis        : time-window class (none / tight / medium / large)
  series (hue)  : method, fixed colour per method across all figures
  y axis        : gap to oracle (%)

Three variants are produced so the best-looking one can be picked for the
paper (all carry the same data):
  box    — median + IQR box, 1.5 IQR whiskers, mean diamond   (recommended)
  bar    — mean bar with ±1 std whisker
  violin — kernel-density violin with median + quartile ticks

Infeasible runs (stranding / HoS breach) carry no meaningful gap and are
EXCLUDED from the distributions — the per-group counts are printed to the
console and written to the stats CSV so the caption can report them.

Usage
-----
  python paper_figures.py                     # all three variants
  python paper_figures.py --kind box          # one variant
  python paper_figures.py --metric gap_nopen  # penalty-free gap
  python paper_figures.py --dir solutions --out-dir figures

Outputs (default): figures/paper_gap_<kind>.pdf + .png and
figures/paper_gap_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

import compile_solutions as cs

# ── canonical axes orders ─────────────────────────────────────────────────────
# The figure always renders the FULL grid (all route/customer/TW classes and
# all methods) so the paper layout is visible before every run exists; slots
# without runs stay empty.  Pass --present-only to draw only levels with data.
_ROUTE_ORDER = ["short", "medium", "long"]
_CUST_ORDER  = ["few", "medium", "many"]
_TW_ORDER    = ["tight", "medium", "large", "none"]

_ROUTE_LBL = {"short": "Short route", "medium": "Medium route", "long": "Long route"}
_CUST_LBL  = {"few": "Few customers", "medium": "Medium customers", "many": "Many customers"}
_TW_LBL    = {"none": "None", "tight": "Tight", "medium": "Medium", "large": "Large"}

# ── fixed method -> colour assignment (Okabe–Ito colourblind-safe palette) ───
# Colour follows the method identity, never the number of methods present in a
# given figure, so Greedy is the same blue in every figure of the paper.
_METHOD_ORDER  = ["greedy", "RO", "ROBU", "LA", "2SP"]
_METHOD_LBL    = {"greedy": "Greedy", "RO": "RO", "ROBU": "ROBU",
                  "LA": "LA", "2SP": "2SP"}
_METHOD_COLOR  = {
    "greedy": "#0072B2",   # blue
    "RO":     "#D55E00",   # vermillion
    "ROBU":   "#E69F00",   # orange (budgeted robust, Bertsimas-Sim)
    "LA":     "#009E73",   # bluish green
    "2SP":    "#CC79A7",   # reddish purple
}

# chart chrome — neutral journal-figure grays
_INK_PRIMARY = "#000000"
_INK_MUTED   = "#555555"
_GRID        = "#e0e0e0"
_BASELINE    = "#333333"


def _tint(color: str, frac: float = 0.80) -> tuple:
    """Blend a colour toward white (frac = white share) for box fills."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def _shade(color: str, frac: float = 0.35) -> tuple:
    """Blend a colour toward black (frac = black share) for median lines."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r * (1 - frac), g * (1 - frac), b * (1 - frac))


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def _dedup_latest(rows):
    """
    When the same instance was solved several times with the same method
    (e.g. a rerun batch), keep only the LATEST run — ranked by the
    timestamp+index in the run_id (file name as fallback) — and drop the
    rest.  Returns (kept_rows, n_dropped).
    """
    def _rank(r):
        m = cs._RUN_ID_RE.match(r.get("run_id") or "")
        if m:
            return (m.group("ts"), int(m.group("idx")))
        return (r.get("_file", ""), 0)

    best = {}
    n_dup = 0
    for r in rows:
        key = (r.get("instance"), r.get("method"), bool(r.get("supervised")))
        if key in best:
            n_dup += 1
            if _rank(r) > _rank(best[key]):
                best[key] = r
        else:
            best[key] = r
    return list(best.values()), n_dup


def collect_gaps(solutions_dir: str, metric: str = "gap_pen"):
    """
    Load all finished runs, keep only the latest run per (instance, method),
    and pool the chosen gap metric per (route, customers, window, method)
    cell.

    Returns
    -------
    gaps    : dict[(route, cust, tw, method)] -> list of gaps (%)
    n_infe  : dict[same key] -> count of infeasible runs (excluded from gaps)
    """
    rows = cs.load_solutions(solutions_dir)
    cs._annotate_instance_tags(rows)
    cs._annotate_gap_to_oracle(rows)
    rows, n_dup = _dedup_latest(rows)
    if n_dup:
        print(f"  Dropped {n_dup} superseded duplicate run(s) "
              f"(same instance + method, older timestamp)")

    gaps   = defaultdict(list)
    n_infe = defaultdict(int)
    for r in rows:
        if r.get("status") != "OK":
            continue
        route, cust, tw = (r.get("route_class"), r.get("customers_class"),
                           r.get("window_class"))
        method = r.get("method")
        if not (route and cust and tw and method):
            continue
        key = (route, cust, tw, method)
        if not cs._is_feasible(r):
            n_infe[key] += 1
            continue
        g = r.get(metric)
        if g is not None:
            gaps[key].append(100.0 * g)
    return gaps, n_infe


def write_stats_csv(gaps, n_infe, path: str):
    """One row per (route, cust, tw, method): n, mean, std, median, quartiles."""
    keys = sorted(set(gaps) | set(n_infe))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["route_class", "customers_class", "window_class", "method",
                    "n_feasible", "n_infeasible_excluded",
                    "gap_mean_%", "gap_std_%", "gap_median_%",
                    "gap_q1_%", "gap_q3_%", "gap_min_%", "gap_max_%"])
        for key in keys:
            vals = np.asarray(gaps.get(key, []), dtype=float)
            stats = ([f"{v:.3f}" for v in (
                          vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0,
                          np.median(vals), np.percentile(vals, 25),
                          np.percentile(vals, 75), vals.min(), vals.max())]
                     if len(vals) else [""] * 7)
            w.writerow(list(key) + [len(vals), n_infe.get(key, 0)] + stats)
    print(f"  Stats CSV : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _present(levels, order):
    s = set(levels)
    return [v for v in order if v in s]


def _style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=_GRID, lw=0.6)
    ax.tick_params(axis="x", colors=_INK_PRIMARY, labelsize=7.5, length=2.5)
    ax.tick_params(axis="y", colors=_INK_PRIMARY, labelsize=7.5, length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
        ax.spines[side].set_linewidth(0.7)


_TW_SHORT   = {"tight": "T", "medium": "M", "large": "L", "none": "N"}
_CUST_SHORT = {"few": "Few", "medium": "Medium", "many": "Many"}

# TW class -> white-blend fraction of the method colour (row layout):
# tighter window = darker box
_TW_TINT = {"tight": 0.10, "medium": 0.40, "large": 0.62, "none": 0.84}


def _draw_group_marks(ax, kind, data, x_pos, col, mark_w):
    """Draw one method's marks (box / bar / violin) at the given positions."""
    if kind == "box":
        ax.boxplot(
            data, positions=x_pos, widths=mark_w,
            showmeans=True, patch_artist=True,
            boxprops=dict(facecolor=_tint(col), edgecolor=col, lw=1.0),
            whiskerprops=dict(color=col, lw=1.0),
            capprops=dict(color=col, lw=1.0),
            medianprops=dict(color=col, lw=1.6),
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor=col, markeredgewidth=1.0,
                           markersize=3.5),
            flierprops=dict(marker="o", markersize=2.0,
                            markerfacecolor="none", markeredgecolor=col,
                            markeredgewidth=0.6, alpha=0.6),
        )
    elif kind == "bar":
        means = [d.mean() for d in data]
        stds  = [d.std(ddof=1) if len(d) > 1 else 0.0 for d in data]
        ax.bar(x_pos, means, width=mark_w, color=col, alpha=0.85,
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.errorbar(x_pos, means, yerr=stds, fmt="none",
                    ecolor=_INK_PRIMARY, elinewidth=0.9,
                    capsize=2.0, capthick=0.9, zorder=4)
    elif kind == "violin":
        vp = ax.violinplot(data, positions=x_pos, widths=mark_w,
                           showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(col)
            body.set_edgecolor(col)
            body.set_alpha(0.40)
            body.set_linewidth(0.8)
        vp["cmedians"].set_color(col)
        vp["cmedians"].set_linewidth(1.3)
        q1s = [np.percentile(d, 25) for d in data]
        q3s = [np.percentile(d, 75) for d in data]
        ax.vlines(x_pos, q1s, q3s, color=col, lw=2.2, alpha=0.9)
    else:
        raise ValueError(f"unknown kind '{kind}'")


def plot_gap_figure(gaps, n_infe, kind: str = "box",
                    metric: str = "gap_pen", out_dir: str = "figures",
                    annotate_n: bool = True, full_grid: bool = True,
                    layout: str = "row", inner: str = "tw") -> list:
    """
    Render the gap-distribution figure and save PDF + PNG; returns the paths.

    layout="row"  — ONE figure, one row of three panels (one per route
                    class).  Inside each panel the x axis nests customer
                    class -> method -> TW class: for every customer group,
                    four method blocks sit side by side, each holding its
                    four TW boxes (T-M-L-N order).  Method = colour, TW
                    class = shade of that colour (Tight darkest -> None
                    lightest).  Outlier dots are hidden at this density;
                    whiskers still span 1.5 IQR.  ~7.2 x 2.5 in -> fits
                    ``\\includegraphics[width=\\textwidth]``.
    layout="grid" — everything in one 3 x 3 facet grid figure (route rows
                    x customer columns), coarser but larger marks.

    full_grid=True draws the complete canonical layout (every route,
    customer, TW class and method) even where no runs exist yet, so the
    final paper figure is visible from the start; False draws only the
    levels present in the data.
    """
    if full_grid:
        routes, custs, tws, methods = (_ROUTE_ORDER, _CUST_ORDER,
                                       _TW_ORDER, _METHOD_ORDER)
    else:
        routes  = _present((k[0] for k in gaps), _ROUTE_ORDER)
        custs   = _present((k[1] for k in gaps), _CUST_ORDER)
        tws     = _present((k[2] for k in gaps), _TW_ORDER)
        methods = _present((k[3] for k in gaps), _METHOD_ORDER)
    if not (routes and custs and tws and methods):
        raise SystemExit("no plottable runs found")

    # per-method x offsets inside each window-class slot
    group_w = 0.78
    slot_w  = group_w / len(methods)
    offsets = {m: (mi - (len(methods) - 1) / 2) * slot_w
               for mi, m in enumerate(methods)}
    mark_w  = slot_w * 0.85                       # leaves a gap between marks

    def _panel(ax, route, cust):
        """Draw one (route, cust) panel: all TW groups x all methods."""
        _style_axes(ax)
        for m in methods:
            x_pos, data = [], []
            for ti, tw in enumerate(tws):
                vals = gaps.get((route, cust, tw, m), [])
                if vals:
                    x_pos.append(ti + offsets[m])
                    data.append(np.asarray(vals))
            if data:
                _draw_group_marks(ax, kind, data, x_pos,
                                  _METHOD_COLOR[m], mark_w)

        ns = {(ti, m): len(gaps.get((route, cust, tw, m), []))
              for ti, tw in enumerate(tws) for m in methods}
        if not any(ns.values()):
            ax.text(0.5, 0.5, "no runs yet",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7.5, style="italic", color=_INK_MUTED)
        # Flag only groups whose seed count deviates >10% from the panel's
        # typical (modal) count — a full row of identical n values is noise.
        elif annotate_n:
            nonzero = [n for n in ns.values() if n]
            n_mode  = max(set(nonzero), key=nonzero.count)
            oddball = {k: n for k, n in ns.items()
                       if n and abs(n - n_mode) > 0.1 * n_mode}
            for (ti, m), n in oddball.items():
                # stagger adjacent method slots on two lines
                dy = -18 - 6 * (methods.index(m) % 2)
                ax.annotate(
                    f"{n}", xy=(ti + offsets[m], 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="top",
                    fontsize=5.5, color=_METHOD_COLOR[m])

        ax.axhline(0.0, color=_BASELINE, lw=0.8, zorder=1)
        ax.set_xticks(range(len(tws)))
        ax.set_xticklabels([_TW_LBL[t] for t in tws])
        ax.set_xlim(-0.55, len(tws) - 0.45)
        ax.set_xlabel("")

    os.makedirs(out_dir, exist_ok=True)

    # common y scale across every figure so route classes compare directly;
    # bars only reach mean+std, so their scale hugs that instead of the
    # raw per-run maximum
    if kind == "bar":
        tops = [np.mean(v) + (np.std(v, ddof=1) if len(v) > 1 else 0.0)
                for v in gaps.values() if v]
    else:
        tops = [v for vals in gaps.values() for v in vals]
    y_top = 1.06 * max(tops) if tops else 1.0

    def _legend(fig, tw_shades: bool = False):
        """One legend per figure: methods by colour, optionally TW shades."""
        if kind == "bar":
            handles = [Patch(facecolor=_METHOD_COLOR[m], alpha=0.85,
                             label=_METHOD_LBL[m]) for m in methods]
        else:
            handles = [Patch(facecolor=_tint(_METHOD_COLOR[m]),
                             edgecolor=_METHOD_COLOR[m],
                             label=_METHOD_LBL[m]) for m in methods]
        if tw_shades:
            handles += [Patch(facecolor=_tint("#4d4d4d", _TW_TINT[t]),
                              edgecolor="#4d4d4d",
                              label=f"TW {_TW_LBL[t]}") for t in tws]
        if kind == "box":
            handles.append(Line2D([], [], marker="D", linestyle="none",
                                  markerfacecolor=_INK_MUTED,
                                  markeredgecolor="white",
                                  markeredgewidth=0.6, markersize=4.5,
                                  label="mean"))
        fig.legend(handles=handles, loc="upper center",
                   ncol=len(handles), frameon=False, fontsize=6.8,
                   handlelength=1.2, handletextpad=0.5, columnspacing=1.0,
                   bbox_to_anchor=(0.5, 1.0))

    metric_sfx = "" if metric == "gap_pen" else f"_{metric}"

    def _save(fig, name_sfx):
        out = []
        for ext in ("pdf", "png"):
            p = os.path.join(out_dir,
                             f"paper_gap_{kind}{metric_sfx}{name_sfx}.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            out.append(p)
        plt.close(fig)
        return out

    paths = []
    if layout == "grid":
        n_rows, n_cols = len(routes), len(custs)
        fig_w = min(7.0, 1.4 + 1.9 * n_cols)      # \textwidth ≈ 7 in
        fig_h = 0.55 + 1.5 * n_rows               # wide, flat panels
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                 sharey=True, squeeze=False)
        for ri, route in enumerate(routes):
            for ci, cust in enumerate(custs):
                ax = axes[ri][ci]
                _panel(ax, route, cust)
                if ri == 0:
                    ax.set_title(_CUST_LBL[cust], fontsize=8.5,
                                 color=_INK_PRIMARY, pad=5)
                if ci == n_cols - 1 and n_rows > 1:
                    ax.annotate(_ROUTE_LBL[route],
                                xy=(1.03, 0.5), xycoords="axes fraction",
                                rotation=270, ha="center", va="center",
                                fontsize=8.5, color=_INK_PRIMARY)
                if ci == 0:
                    ax.set_ylabel("Gap to oracle (%)", fontsize=8,
                                  color=_INK_PRIMARY)
        axes[0][0].set_ylim(-0.02 * y_top, y_top)
        _legend(fig)
        right = 0.96 if n_rows > 1 else 1.0       # room for route labels
        fig.tight_layout(rect=(0, 0, right, 0.95))
        paths += _save(fig, "_grid")
    else:
        # ── ONE figure: 3 route panels, x = customer -> method -> TW ─────
        # inner grouping is switchable:
        #   inner="tw"     — method blocks, TW boxes inside (shade = TW)
        #   inner="method" — TW blocks, methods side by side (plain colours)
        inner_is_tw = (inner == "tw")
        outer_list  = methods if inner_is_tw else tws
        inner_list  = tws if inner_is_tw else methods
        n_i, n_o    = len(inner_list), len(outer_list)
        gap_b, gap_c = 1.0, 2.2
        blk = n_i + gap_b                     # stride between blocks
        grp = n_o * blk - gap_b + gap_c       # stride between cust groups

        def _x(ci, oi, ii):
            return ci * grp + oi * blk + ii

        x_max = _x(len(custs) - 1, n_o - 1, n_i - 1)

        # infeasibility heat strip, drawn below every panel so a method with a
        # good gap but a high stranding/HoS-breach rate is visible at a glance.
        # A cell that is 100% infeasible draws no box above it, so the strip is
        # the ONLY signal for it; a not-run cell is left blank.
        from matplotlib.patches import Rectangle as _Rect
        from matplotlib.colors import LinearSegmentedColormap as _LSC
        # traffic-light ramp built from the project's Okabe-Ito palette:
        # bluish-green (all feasible) -> yellow -> vermillion (all infeasible)
        _reds = _LSC.from_list(
            "infeas", ["#009E73", "#F0E442", "#D55E00"])

        def _infeas_frac(route, cust, tw, m):
            nf  = len(gaps.get((route, cust, tw, m), []))
            ni  = n_infe.get((route, cust, tw, m), 0)
            tot = nf + ni
            return (ni / tot) if tot else None

        # scale the ramp to the WORST rate actually observed, so red marks the
        # most-infeasible cell in this figure rather than a hypothetical 100%
        _all_fracs = [f for f in
                      (_infeas_frac(r, c, t, m)
                       for r in routes for c in custs
                       for t in tws for m in methods)
                      if f]
        _fmax = max(_all_fracs) if _all_fracs else 1.0
        _fmax = _fmax if _fmax > 1e-9 else 1.0

        # main panels (row 0) + a dedicated infeasibility strip row (row 1)
        fig = plt.figure(figsize=(7.2, 2.7))
        gs  = fig.add_gridspec(2, len(routes), height_ratios=[1.0, 0.11],
                               hspace=0.18, wspace=0.10)
        main_axes, strip_axes = [], []
        for ri in range(len(routes)):
            axm  = fig.add_subplot(gs[0, ri],
                                   sharey=main_axes[0] if main_axes else None)
            axst = fig.add_subplot(gs[1, ri])   # x aligned via identical xlim
            main_axes.append(axm)
            strip_axes.append(axst)

        for ri, route in enumerate(routes):
            ax   = main_axes[ri]
            axst = strip_axes[ri]
            _style_axes(ax)
            route_empty = True
            for ci, cust in enumerate(custs):
                for oi, ov in enumerate(outer_list):
                    for ii, iv in enumerate(inner_list):
                        m, tw = (ov, iv) if inner_is_tw else (iv, ov)
                        vals = gaps.get((route, cust, tw, m), [])
                        if not vals:
                            continue
                        route_empty = False
                        d    = np.asarray(vals)
                        px   = _x(ci, oi, ii)
                        col  = _METHOD_COLOR[m]
                        fill = (_tint(col, _TW_TINT[tw]) if inner_is_tw
                                else _tint(col, 0.55))
                        if kind == "bar":
                            ax.bar([px], [d.mean()], width=0.85,
                                   color=[fill] if inner_is_tw else col,
                                   edgecolor=col, linewidth=0.5, zorder=3)
                            ax.errorbar([px], [d.mean()],
                                        yerr=[d.std(ddof=1) if len(d) > 1
                                              else 0.0],
                                        fmt="none", ecolor="#aaaaaa",
                                        elinewidth=0.5, capsize=1.0,
                                        capthick=0.5, zorder=4)
                        else:   # box (violin is too dense at this width)
                            bp = ax.boxplot(
                                [d], positions=[px], widths=0.92,
                                showmeans=True, showfliers=False,
                                patch_artist=True,
                                boxprops=dict(edgecolor=col, lw=0.6),
                                whiskerprops=dict(color=col, lw=0.55,
                                                  alpha=0.85),
                                capprops=dict(lw=0),      # no whisker caps
                                medianprops=dict(color=_shade(col, 0.45),
                                                 lw=1.2),
                                meanprops=dict(marker="D",
                                               markerfacecolor=col,
                                               markeredgecolor="white",
                                               markeredgewidth=0.4,
                                               markersize=2.8),
                            )
                            bp["boxes"][0].set_facecolor(fill)

            if route_empty:
                ax.text(0.5, 0.5, "no runs yet", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7.5,
                        style="italic", color=_INK_MUTED)

            slot_pos = [_x(ci, oi, ii)
                        for ci in range(len(custs))
                        for oi in range(n_o)
                        for ii in range(n_i)]
            if inner_is_tw:
                # a T/M/L/N letter under every slot (methods are the blocks
                # and carry colour); single letters stay horizontal — at
                # this slot pitch any rotation makes glyphs collide
                slot_lbl = [_TW_SHORT[tws[ii]]
                            for _ in range(len(custs) * n_o)
                            for ii in range(n_i)]
                ax.set_xticks(slot_pos)
                ax.set_xticklabels(slot_lbl)
                ax.tick_params(axis="x", length=1.8, width=0.5, pad=1.5,
                               color=_INK_MUTED,
                               labelsize=6.0, labelcolor=_INK_PRIMARY)
                # faint vertical guide from every tick to the top border
                ax.grid(True, axis="x", color="#ececec", lw=0.35, zorder=0)
            else:
                # methods sit side by side inside each TW block: one larger
                # T/M/L/N letter per block, small minor ticks per slot
                blk_pos = [_x(ci, oi, (n_i - 1) / 2)
                           for ci in range(len(custs))
                           for oi in range(n_o)]
                blk_lbl = [_TW_SHORT[tws[oi]]
                           for _ in range(len(custs))
                           for oi in range(n_o)]
                ax.set_xticks(blk_pos)
                ax.set_xticklabels(blk_lbl)
                ax.tick_params(axis="x", length=1.8, width=0.5, pad=1.5,
                               color=_INK_MUTED,
                               labelsize=6.0, labelcolor=_INK_MUTED)
                ax.set_xticks(slot_pos, minor=True)
                ax.tick_params(axis="x", which="minor", length=1.2,
                               width=0.4, color=_INK_MUTED)
                ax.grid(True, axis="x", which="minor",
                        color="#ececec", lw=0.35, zorder=0)
            # stronger separators between customer-class groups
            for ci in range(len(custs) - 1):
                mid = (ci * grp + n_o * blk - gap_b - 1
                       + (ci + 1) * grp) / 2
                ax.axvline(mid, color="#bbbbbb", lw=0.7, zorder=0.5)
            ax.set_xlim(-1.2, x_max + 1.2)
            ax.set_title(_ROUTE_LBL[route], fontsize=8.5,
                         color=_INK_PRIMARY, pad=5)

            # ── infeasibility strip (own axes; x aligned to the panel) ───────
            axst.set_xlim(-1.2, x_max + 1.2)
            axst.set_ylim(0, 1)
            axst.set_xticks([]); axst.set_yticks([])
            for s in axst.spines.values():
                s.set_visible(False)
            for ci, cust in enumerate(custs):
                for oi, ov in enumerate(outer_list):
                    for ii, iv in enumerate(inner_list):
                        m, tw = (ov, iv) if inner_is_tw else (iv, ov)
                        frac = _infeas_frac(route, cust, tw, m)
                        if frac is None:
                            continue        # not run -> leave blank
                        axst.add_patch(_Rect(
                            (_x(ci, oi, ii) - 0.46, 0.30), 0.92, 0.60,
                            facecolor=_reds(min(1.0, frac / _fmax)),
                            edgecolor="#8a8a8a", lw=0.35, zorder=3))
            # customer-class labels sit UNDER the strip (no overlap now)
            centers = [ci * grp + (n_o * blk - gap_b - 1) / 2
                       for ci in range(len(custs))]
            for c_x, cust in zip(centers, custs):
                axst.annotate(_CUST_SHORT[cust], xy=(c_x, 0.30),
                              xycoords=("data", "axes fraction"),
                              xytext=(0, -3), textcoords="offset points",
                              ha="center", va="top", fontsize=7.5,
                              color=_INK_PRIMARY)
            if ri == 0:
                axst.annotate("Infeas.\nrate", xy=(0, 0.60),
                              xycoords="axes fraction",
                              xytext=(-4, 0), textcoords="offset points",
                              ha="right", va="center", fontsize=5.2,
                              linespacing=0.9, color=_INK_MUTED)

        # one shared y scale for all main panels; minor gridlines every 5%
        main_axes[0].set_ylim(0, y_top)       # y = 0 sits on the axis line
        for ax in main_axes:
            ax.yaxis.set_minor_locator(MultipleLocator(5))
            ax.grid(which="minor", axis="y", color="#f2f2f2", lw=0.4)
            ax.tick_params(axis="y", which="minor", length=1.5, width=0.5,
                           color=_INK_MUTED)
        for ax in main_axes[1:]:
            ax.tick_params(labelleft=False)
        main_axes[0].set_ylabel("Gap to oracle (%)", fontsize=8,
                                color=_INK_PRIMARY)
        _legend(fig)     # TW classes are labelled on the axis, not the legend
        fig.tight_layout(rect=(0, 0.0, 1, 0.90))

        # compact colour key, tucked to the right of the strip row
        import matplotlib.cm as _cm
        from matplotlib.colors import Normalize as _Norm
        _tmax = 100.0 * _fmax
        _sm = _cm.ScalarMappable(norm=_Norm(0, _tmax), cmap=_reds)
        _sm.set_array([])
        _p = strip_axes[-1].get_position()
        _cax = fig.add_axes([_p.x1 + 0.008, _p.y0, 0.009, _p.height])
        _cb  = fig.colorbar(_sm, cax=_cax, orientation="vertical",
                            ticks=[0, _tmax])
        _cb.ax.set_yticklabels(["0", f"{_tmax:.0f}"])
        _cb.outline.set_linewidth(0.3)
        _cb.set_label("Infeas. %", fontsize=5, labelpad=1)
        _cb.ax.tick_params(labelsize=4.4, length=1.5, width=0.3, pad=1)

        paths += _save(fig, "" if inner_is_tw else "_methods")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate gap-to-oracle figures (mean/spread across "
                    "seeds) per instance family and method, for the paper.")
    parser.add_argument("--dir", default="solutions",
                        help="solutions directory (default: solutions)")
    parser.add_argument("--out-dir", default="figures",
                        help="output directory (default: figures)")
    parser.add_argument("--kind", default="bar",
                        choices=["box", "bar", "violin", "all"],
                        help="figure variant (default: bar — means with "
                             "discreet ±1 std whiskers; box/violin get "
                             "unreadable at full nesting density)")
    parser.add_argument("--metric", default="gap_pen",
                        choices=["gap_pen", "gap_nopen"],
                        help="gap definition (default: gap_pen, window "
                             "penalties included on both sides)")
    parser.add_argument("--present-only", action="store_true", default=False,
                        help="draw only route/customer/TW classes and "
                             "methods that have runs, instead of the full "
                             "canonical paper layout with empty slots")
    parser.add_argument("--all", action="store_true", default=False,
                        help="regenerate the full paper set in one go: "
                             "box + bar, each in both inner orderings "
                             "(tw and method), plus the stats CSV; "
                             "--kind/--inner are ignored")
    parser.add_argument("--inner", default="tw", choices=["tw", "method"],
                        help="row layout inner grouping: 'tw' = method "
                             "blocks holding their four TW boxes (default); "
                             "'method' = TW blocks holding the four methods "
                             "side by side (files get a '_methods' suffix)")
    parser.add_argument("--layout", default="row", choices=["row", "grid"],
                        help="'row' = all data in one horizontal band "
                             "(9 panels side by side, default); "
                             "'grid' = 3x3 facet grid")
    args = parser.parse_args()

    gaps, n_infe = collect_gaps(args.dir, metric=args.metric)

    n_groups = len(gaps)
    n_runs   = sum(len(v) for v in gaps.values())
    n_exc    = sum(n_infe.values())
    print(f"  Pooled {n_runs} feasible run(s) into {n_groups} "
          f"(family x method) group(s); excluded {n_exc} infeasible run(s)")
    for key in sorted(n_infe):
        if n_infe[key]:
            print(f"    excluded {n_infe[key]:>2} infeasible: "
                  f"{'/'.join(key[:3])} [{key[3]}]")

    os.makedirs(args.out_dir, exist_ok=True)
    write_stats_csv(gaps, n_infe,
                    os.path.join(args.out_dir, "paper_gap_stats.csv"))

    if args.all:
        combos = [(k, i) for k in ("box", "bar") for i in ("tw", "method")]
    else:
        kinds  = (["box", "bar", "violin"] if args.kind == "all"
                  else [args.kind])
        combos = [(k, args.inner) for k in kinds]
    for kind, inner in combos:
        for p in plot_gap_figure(gaps, n_infe, kind=kind,
                                 metric=args.metric, out_dir=args.out_dir,
                                 full_grid=not args.present_only,
                                 layout=args.layout, inner=inner):
            print(f"  Figure    : {p}")
