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

Four variants carry the same data; box is the one reported in the paper and
is what a bare run produces:
  box    — median + IQR box, 1.5 IQR whiskers, mean diamond   (DEFAULT)
  bar    — mean bar with ±1 std whisker
  violin — kernel-density violin with median + quartile ticks
  line   — same slots as box, but the four TW marks inside each method block
           are CONNECTED: median (solid) and mean (dashed) lines over an IQR
           band, so the slope across the window classes is read directly
           (--line-no-band drops the bands)

Infeasible runs (stranding / HoS breach) carry no meaningful gap and are
EXCLUDED from the distributions — the per-group counts are printed to the
console and written to the stats CSV so the caption can report them.

Usage
-----
  python -m src.plot.paper_figures                     # box (paper figure)
  python -m src.plot.paper_figures --kind all          # box + bar + violin
  python -m src.plot.paper_figures --all               # box + bar x tw/method
  python -m src.plot.paper_figures --metric gap_nopen  # penalty-free gap
  python -m src.plot.paper_figures --dir solutions --out-dir figures

Outputs (default): figures/paper_gap_<kind>.pdf + .png and
data_output/paper_gap_stats.csv
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

from src.output_analysis import compile_solutions as cs

# ── canonical axes orders ─────────────────────────────────────────────────────
# The figure always renders the FULL grid (all route/customer/TW classes and
# all methods) so the paper layout is visible before every run exists; slots
# without runs stay empty.  Pass --present-only to draw only levels with data.
# Orders, labels, colours and chrome all live in paper_style so this script and
# additional_figures.py cannot drift apart (they had already started to).
from src.plot import paper_style as ps
from src import paths as _paths

_ROUTE_ORDER = ps.ROUTE_ORDER
_CUST_ORDER  = ps.CUST_ORDER
_TW_ORDER    = ps.TW_ORDER

_ROUTE_LBL = ps.ROUTE_LBL
_CUST_LBL  = ps.CUST_LBL
# Copy, not an alias: the pooled-TW level below adds a key, and mutating
# paper_style's dict would leak that synthetic level into every other module
# that imports it.
_TW_LBL    = dict(ps.TW_LBL)

# ── fixed method -> colour assignment (Okabe–Ito colourblind-safe palette) ───
# Colour follows the method identity, never the number of methods present in a
# given figure, so Greedy is the same blue in every figure of the paper.
# the five simulated policies (oracle is a bound, not a plotted method here)
_METHOD_ORDER  = [m for m in ps.METHOD_ORDER if m != "oracle"]
_METHOD_LBL    = ps.METHOD_LBL
_METHOD_COLOR  = ps.METHOD_COLOR

# chart chrome — neutral journal-figure grays
_INK_PRIMARY = ps.INK_PRIMARY
_INK_MUTED   = ps.INK_MUTED
_GRID        = ps.GRID
_BASELINE    = ps.BASELINE


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

# Dedup lives in compile_solutions so the Excel tables and these figures share
# one definition of "latest run per (instance, method)".
_dedup_latest = cs._dedup_latest


def collect_gaps(solutions_dir: str, metric: str = "gap_pen"):
    """
    Load all finished runs, keep only the latest run per (instance, method),
    and pool the chosen gap metric per (route, customers, window, method)
    cell.

    Runs are split three ways (see compile_solutions.classify_outcome):
      feasible   -> contributes a gap sample
      infeasible -> a GENUINE failure (certified plan still fails); counted in
                    n_infe, drives the infeasibility heat strip
      unsolved   -> the offline solver never certified the plan (ROBU C&CG did
                    not converge); counted separately in n_unsl and excluded
                    from BOTH the gaps and the infeasibility rate, so a
                    non-converged solve is not mislabelled as infeasible

    Returns
    -------
    gaps    : dict[(route, cust, tw, method)] -> list of gaps (%)
    n_infe  : dict[same key] -> count of genuinely-infeasible runs
    n_unsl  : dict[same key] -> count of unsolved (uncertified) runs
    """
    rows = cs.load_solutions(solutions_dir)
    cs._annotate_instance_tags(rows)
    cs._annotate_gap_to_oracle(rows, solutions_dir)
    cs._annotate_outcome(rows)
    rows, n_dup = _dedup_latest(rows)
    if n_dup:
        print(f"  Dropped {n_dup} superseded duplicate run(s) "
              f"(same instance + method, older timestamp)")

    # Method-configuration sweeps (--variant) run on the BASE instances, so they
    # carry a valid route/customers/window class and would pool straight into
    # the published figures.  Unlike the "__tag" instance variants, nothing else
    # filters them out — this line is the only thing that does.
    n_var = sum(1 for r in rows if r.get("variant"))
    if n_var:
        rows = [r for r in rows if not r.get("variant")]
        print(f"  Excluded {n_var} method-variant run(s) from the paper "
              f"figures (base case only)")

    gaps   = defaultdict(list)
    n_infe = defaultdict(int)
    n_unsl = defaultdict(int)
    n_feas = defaultdict(int)
    for r in rows:
        if r.get("status") != "OK":
            continue
        route, cust, tw = (r.get("route_class"), r.get("customers_class"),
                           r.get("window_class"))
        method = r.get("method")
        if not (route and cust and tw and method):
            continue
        key = (route, cust, tw, method)
        outcome = r.get("outcome")
        if outcome == "unsolved":
            n_unsl[key] += 1
            continue
        if outcome == "infeasible":
            n_infe[key] += 1
            continue
        # Every feasible run counts toward reliability, even when no oracle
        # bound exists for its instance (such a run yields no gap sample but
        # is NOT a failure).  Keeping the two counters separate stops
        # oracle-poor cells from reading as near-total infeasibility.
        n_feas[key] += 1
        g = r.get(metric)
        if g is not None:
            gaps[key].append(100.0 * g)
    return gaps, n_infe, n_unsl, n_feas


def write_stats_csv(gaps, n_infe, n_unsl, n_feas, path: str):
    """One row per (route, cust, tw, method): n, mean, std, median, quartiles.

    n_feasible counts every feasible run; n_gap_samples counts those that also
    had an oracle bound (the mean/quartiles rest on those only)."""
    keys = sorted(set(gaps) | set(n_infe) | set(n_unsl) | set(n_feas))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["route_class", "customers_class", "window_class", "method",
                    "n_feasible", "n_gap_samples",
                    "n_infeasible_excluded", "n_unsolved_excluded",
                    "gap_mean_%", "gap_std_%", "gap_median_%",
                    "gap_q1_%", "gap_q3_%", "gap_min_%", "gap_max_%"])
        for key in keys:
            vals = np.asarray(gaps.get(key, []), dtype=float)
            stats = ([f"{v:.3f}" for v in (
                          vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0,
                          np.median(vals), np.percentile(vals, 25),
                          np.percentile(vals, 75), vals.min(), vals.max())]
                     if len(vals) else [""] * 7)
            w.writerow(list(key)
                       + [n_feas.get(key, len(vals)), len(vals),
                          n_infe.get(key, 0), n_unsl.get(key, 0)]
                       + stats)
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

# Synthetic window level used when the four TW classes are pooled into one box
# per method.  It is a real key in the (route, cust, tw, method) dicts rather
# than a special case threaded through the layout, so every downstream consumer
# — panel drawing, heat strip, n annotations, stats CSV — works unchanged with
# a one-element TW axis.  Full colour (no tint): with a single level there is
# no shade ladder left to read.
POOLED_TW = "all"
_TW_TINT[POOLED_TW]  = 0.0
_TW_LBL[POOLED_TW]   = "All windows"
_TW_SHORT[POOLED_TW] = "A"


def pool_tw(*dicts):
    """Collapse the TW axis of (route, cust, tw, method)-keyed dicts.

    Lists are concatenated and counters summed, so a pooled box holds every
    instance of that (route, cust, method) cell regardless of window class.
    Returns one new dict per input, in the same order.
    """
    out = []
    for d in dicts:
        pooled: dict = {}
        for (route, cust, _tw, meth), v in d.items():
            key = (route, cust, POOLED_TW, meth)
            if isinstance(v, list):
                pooled.setdefault(key, []).extend(v)
            else:
                pooled[key] = pooled.get(key, 0) + v
        out.append(pooled)
    return out


def pool_cust(*dicts):
    """Collapse the CUSTOMER axis of (route, cust, tw, method)-keyed dicts.

    Used by the window-response figure, where the question is about the window
    class and the customer count is a nuisance dimension: pooling it triples
    the n behind each window mark and drops the figure from nine panels to
    three, so the tight-vs-loose contrast is read directly.
    """
    out = []
    for d in dicts:
        pooled: dict = {}
        for (route, _cust, tw, meth), v in d.items():
            key = (route, "all", tw, meth)
            if isinstance(v, list):
                pooled.setdefault(key, []).extend(v)
            else:
                pooled[key] = pooled.get(key, 0) + v
        out.append(pooled)
    return out


def plot_tw_response(gaps, metric: str = "gap_pen",
                     out_dir: str = _paths.figures()) -> list:
    """How each method responds to window tightness — one panel per route.

    A dumbbell per method: the median gap under the LOOSEST window class
    present, the median under the TIGHTEST, joined by a line, with the shift
    annotated.  Thin bars carry the IQR of each end so the reader can see
    whether a shift in the median is large against the spread.

    Why not a paired per-instance delta, which would be stronger: the window
    class is folded into the geometry seed (see instance_io._geometry_seed), so
    RshortCfewTnone_1 and RshortCfewTtight_1 are DIFFERENT routes with
    different customers and different realisations — not one route under two
    window regimes.  Nothing is paired across the window axis, so the honest
    comparison is distributional, and the shift is a difference of medians
    between two independent samples rather than a mean of per-instance deltas.

    Note the metric matters here more than anywhere else: gap_pen carries the
    out-of-window penalty on both sides, so a tight-window cell is scored on
    the objective the model actually optimises; --metric gap_nopen isolates
    the pure duration effect.
    """
    (gaps,) = pool_cust(gaps)
    routes = _present((k[0] for k in gaps), _ROUTE_ORDER)
    tws    = _present((k[2] for k in gaps), _TW_ORDER)   # tight -> ... -> none
    methods = _present((k[3] for k in gaps), _METHOD_ORDER)
    if not (routes and tws and methods):
        raise SystemExit("no plottable runs found")
    if len(tws) < 2:
        raise SystemExit(f"window-response needs >=2 window classes with runs, "
                         f"found only {tws}")
    tightest, loosest = tws[0], tws[-1]

    fig, axes = plt.subplots(1, len(routes), figsize=(7.2, 2.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(methods), dtype=float)

    for ax, route in zip(axes, routes):
        _style_axes(ax)
        for xi, m in zip(x, methods):
            col = _METHOD_COLOR[m]
            ends = {}
            for tw in (loosest, tightest):
                vals = gaps.get((route, "all", tw, m), [])
                if len(vals):
                    a = np.asarray(vals)
                    ends[tw] = (np.median(a), np.percentile(a, 25),
                                np.percentile(a, 75), len(a))
            if len(ends) < 2:
                continue
            (lo_med, lo_q1, lo_q3, _), (ti_med, ti_q1, ti_q3, _) = \
                ends[loosest], ends[tightest]
            # IQR bars first, so the medians and the connector sit on top
            for xoff, (q1, q3) in ((-0.16, (lo_q1, lo_q3)),
                                   (0.16, (ti_q1, ti_q3))):
                ax.plot([xi + xoff] * 2, [q1, q3], color=_tint(col, 0.55),
                        lw=3.0, solid_capstyle="butt", zorder=2)
            ax.plot([xi - 0.16, xi + 0.16], [lo_med, ti_med],
                    color=col, lw=1.2, zorder=3)
            # Hollow = loose, filled = tight: the fill follows the constraint,
            # matching "tighter window = darker" everywhere else in the paper.
            ax.plot(xi - 0.16, lo_med, "o", mfc="white", mec=col, mew=1.2,
                    ms=5, zorder=4)
            ax.plot(xi + 0.16, ti_med, "o", mfc=col, mec=col, ms=5, zorder=4)
            d = ti_med - lo_med
            ax.annotate(f"{d:+.1f}", xy=(xi, max(lo_med, ti_med)),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.8,
                        color=_shade(col, 0.25))
        ax.set_xticks(x, [_METHOD_LBL[m] for m in methods],
                      rotation=30, ha="right")
        ax.set_title(_ROUTE_LBL[route], fontsize=8, color=_INK_PRIMARY)
        ax.set_xlim(-0.6, len(methods) - 0.4)

    axes[0].set_ylabel(f"Gap to oracle (%)  [{metric}]", fontsize=7.5)
    handles = [Line2D([], [], marker="o", mfc="white", mec=_INK_MUTED,
                      mew=1.2, ls="none", ms=5,
                      label=f"{_TW_LBL[loosest]} (loosest)"),
               Line2D([], [], marker="o", mfc=_INK_MUTED, mec=_INK_MUTED,
                      ls="none", ms=5, label=f"{_TW_LBL[tightest]} (tightest)"),
               Line2D([], [], color=_tint(_INK_MUTED, 0.55), lw=3,
                      label="IQR")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    metric_sfx = "" if metric == "gap_pen" else f"_{metric}"
    out = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"paper_tw_response{metric_sfx}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


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


def plot_gap_figure(gaps, n_infe, n_unsl=None, n_feas=None, kind: str = "box",
                    metric: str = "gap_pen", out_dir: str = _paths.figures(),
                    annotate_n: bool = True, full_grid: bool = True,
                    layout: str = "row", inner: str = "tw",
                    line_band: bool = True) -> list:
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

    The infeasibility heat strip below each panel shades the GENUINE
    infeasibility rate (certified plans that still fail).  Unsolved
    (uncertified) runs are not drawn at all — they are reported on the console
    and in the stats CSV only — so a cell without an assessable rate stays
    blank rather than carrying a marker of its own.
    """
    if n_unsl is None:
        n_unsl = {}
    if n_feas is None:
        n_feas = {}
    pooled_tw = (inner == "pooled")
    if pooled_tw:
        # Collapse first, then let the whole layout run on a one-element TW
        # axis.  Doing it here rather than at the call site keeps the stats CSV
        # and the heat strip consistent with the boxes by construction.
        gaps, n_infe, n_unsl, n_feas = pool_tw(gaps, n_infe, n_unsl, n_feas)
    if full_grid:
        routes, custs, methods = _ROUTE_ORDER, _CUST_ORDER, _METHOD_ORDER
        tws = [POOLED_TW] if pooled_tw else _TW_ORDER
    else:
        routes  = _present((k[0] for k in gaps), _ROUTE_ORDER)
        custs   = _present((k[1] for k in gaps), _CUST_ORDER)
        tws     = ([POOLED_TW] if pooled_tw
                   else _present((k[2] for k in gaps), _TW_ORDER))
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
    elif kind == "box":
        # Boxes are drawn with showfliers=False, so the raw maximum would
        # scale the axis to outliers that are never plotted (leaving a large
        # empty band above the data).  The highest mark actually drawn is the
        # upper whisker: the largest point within Q3 + 1.5 IQR.  The mean
        # diamond is always inside the box, so it cannot exceed it.
        tops = []
        for v in gaps.values():
            if not v:
                continue
            a = np.asarray(v, dtype=float)
            q1, q3 = np.percentile(a, [25, 75])
            inside = a[a <= q3 + 1.5 * (q3 - q1)]
            tops.append(float(inside.max()) if inside.size else float(a.max()))
    elif kind == "line":
        # highest mark drawn is Q3, or the mean where the tail pulls it above
        tops = []
        for v in gaps.values():
            if not v:
                continue
            a = np.asarray(v, dtype=float)
            tops.append(max(float(np.percentile(a, 75)), float(a.mean())))
    else:
        # violin draws the full distribution, extremes included
        tops = [v for vals in gaps.values() for v in vals]
    y_top = 1.06 * max(tops) if tops else 1.0

    def _legend(fig, tw_shades: bool = False):
        """Single legend row: method colours, then (when ``tw_shades`` is set)
        a "TW:" lead-in and the four shade swatches keyed T=Tight ... N=None
        (dark = Tight -> light = None), so the window class is readable from the
        legend instead of from illegible per-bar axis letters.  Keeping it to
        ONE row avoids colliding with the centred route titles below."""
        if kind == "bar":
            handles = [Patch(facecolor=_METHOD_COLOR[m], alpha=0.85,
                             label=_METHOD_LBL[m]) for m in methods]
        elif kind == "line":
            handles = [Line2D([], [], color=_METHOD_COLOR[m], lw=1.6,
                              label=_METHOD_LBL[m]) for m in methods]
        else:
            handles = [Patch(facecolor=_tint(_METHOD_COLOR[m]),
                             edgecolor=_METHOD_COLOR[m],
                             label=_METHOD_LBL[m]) for m in methods]
        if kind == "line":
            # TW is read from POSITION inside the block here, not from a shade,
            # so the swatch key is replaced by the summary marks.
            handles += [
                Line2D([], [], color=_INK_MUTED, lw=1.2, marker="o", ms=2.6,
                       label="median"),
                Line2D([], [], color=_INK_MUTED, lw=0.9, ls="--", marker="D",
                       ms=2.4, markerfacecolor="white", label="mean"),
            ]
            if line_band:
                handles.append(Patch(facecolor=_tint(_INK_MUTED, 0.80),
                                     edgecolor="none", label="IQR"))
        if kind == "box":
            handles.append(Line2D([], [], marker="D", linestyle="none",
                                  markerfacecolor=_INK_MUTED,
                                  markeredgecolor="white",
                                  markeredgewidth=0.6, markersize=4.5,
                                  label="mean"))
        if tw_shades:
            handles.append(Patch(facecolor="none", edgecolor="none",
                                 label="   TW:"))    # spacer / row lead-in
            handles += [Patch(facecolor=_tint("#4d4d4d", _TW_TINT[t]),
                              edgecolor="#4d4d4d",
                              label=f"{_TW_SHORT[t]}={_TW_LBL[t]}") for t in tws]
        fig.legend(handles=handles, loc="upper center",
                   ncol=len(handles), frameon=False, fontsize=6.5,
                   handlelength=1.1, handletextpad=0.4, columnspacing=0.9,
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
        #   inner="pooled" — the TW axis was collapsed above, so this is the
        #                    inner="tw" layout with a single-element inner
        #                    list: one box per method, holding every window
        #                    class.  It takes the tw branch deliberately —
        #                    method must stay the OUTER key, or the one box
        #                    would sit in a TW block instead of a method one.
        inner_is_tw = (inner in ("tw", "pooled"))
        outer_list  = methods if inner_is_tw else tws
        inner_list  = tws if inner_is_tw else methods
        n_i, n_o    = len(inner_list), len(outer_list)
        # Block gap is sized for a block that HOLDS several marks.  Pooled to
        # one box per method, the same gap leaves each method floating in a
        # 2-unit block around a 0.85-wide box, so the customer groups stop
        # reading as groups.  Tighten it so the methods sit as a compact row
        # while gap_c still separates the customer groups.
        gap_b, gap_c = (0.35, 1.6) if pooled_tw else (1.0, 2.2)
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
            key = (route, cust, tw, m)
            # denominator = ALL feasible runs (fall back to gap samples for
            # callers that pass no n_feas), never just the ones with a bound
            nf  = (n_feas or {}).get(key, len(gaps.get(key, [])))
            ni  = n_infe.get(key, 0)
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
            if kind == "line":
                # Same slots as the boxes, different mark: within each method
                # block the four TW slots are CONNECTED instead of drawn as
                # four independent boxes, so the reader sees the slope (how
                # much a method gains as the windows loosen) rather than having
                # to compare four heights across gaps.  Requires inner="tw" —
                # methods are categorical, so a line across them would imply an
                # order that does not exist.
                for ci, cust in enumerate(custs):
                    for oi, m in enumerate(outer_list):
                        px, med, avg, q1, q3 = [], [], [], [], []
                        for ii, tw in enumerate(inner_list):
                            vals = gaps.get((route, cust, tw, m), [])
                            px.append(_x(ci, oi, ii))
                            if not vals:
                                med.append(np.nan); avg.append(np.nan)
                                q1.append(np.nan);  q3.append(np.nan)
                                continue
                            route_empty = False
                            d = np.asarray(vals, dtype=float)
                            med.append(float(np.median(d)))
                            avg.append(float(d.mean()))
                            _a, _b = np.percentile(d, [25, 75])
                            q1.append(float(_a)); q3.append(float(_b))
                        if all(np.isnan(v) for v in med):
                            continue
                        col = _METHOD_COLOR[m]
                        px = np.asarray(px, dtype=float)
                        q1 = np.asarray(q1); q3 = np.asarray(q3)
                        # A TW class with no runs breaks the line (NaN) rather
                        # than interpolating across it.
                        if line_band:
                            ax.fill_between(px, q1, q3, color=col, alpha=0.16,
                                            linewidth=0, zorder=2)
                            for edge in (q1, q3):
                                ax.plot(px, edge, "-", color=col, lw=0.5,
                                        alpha=0.65, zorder=3)
                        ax.plot(px, med, "-", color=col, lw=1.2, marker="o",
                                ms=2.4, mfc=col, mec=col, zorder=5)
                        ax.plot(px, avg, "--", color=col, lw=0.9, marker="D",
                                ms=2.2, mfc="white", mec=col, mew=0.6,
                                zorder=6)
            for ci, cust in enumerate(custs):
                if kind == "line":
                    break
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
                # Per-bar T/M/L/N letters are illegible at this slot pitch
                # (up to 60 bars/panel), so the TW class is read from the bar
                # SHADE keyed in the legend instead.  Keep only faint tick
                # guides at each slot — no crowded per-bar text.
                ax.set_xticks(slot_pos)
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=1.8, width=0.5,
                               color=_INK_MUTED)
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
                            continue        # nothing assessable -> leave blank
                        # green -> red over the GENUINE infeasibility rate
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
        # inner="tw": TW is read from the shade key in the legend; inner=
        # "method": TW is the on-axis block, so no shade key needed.
        # A pooled figure has one shade per method, so the TW swatch key
        # would advertise a distinction the boxes no longer carry.
        _legend(fig, tw_shades=inner_is_tw and kind != "line"
                and not pooled_tw)
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

        paths += _save(fig, "_pooledtw" if pooled_tw
                       else ("" if inner_is_tw else "_methods"))
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate gap-to-oracle figures (mean/spread across "
                    "seeds) per instance family and method, for the paper.")
    parser.add_argument("--dir", default=_paths.solutions(),
                        help="solutions directory (default: solutions)")
    parser.add_argument("--out-dir", default=_paths.figures(),
                        help="output directory (default: figures)")
    parser.add_argument("--kind", default=None,
                        choices=["box", "bar", "violin", "line", "all"],
                        help="figure variant ('box' — median + IQR, the "
                             "variant reported in the paper; 'bar' gives "
                             "means with ±1 std whiskers; 'line' puts the TW "
                             "class on the x axis and draws median/mean lines "
                             "with an IQR band per method; 'all' every "
                             "variant).  With neither --kind nor --inner nor "
                             "--tw-response nor --all given, the full paper "
                             "box set is regenerated in one go: "
                             "paper_gap_box + paper_gap_box_pooledtw + "
                             "paper_tw_response")
    parser.add_argument("--metric", default="gap_pen",
                        choices=["gap_pen", "gap_nopen"],
                        help="gap definition (default: gap_pen, window "
                             "penalties included on both sides)")
    parser.add_argument("--tw-response", dest="tw_response",
                        action="store_true", default=False,
                        help="window-response figure: one dumbbell per method "
                             "from the loosest to the tightest window class, "
                             "customers pooled, one panel per route "
                             "(-> paper_tw_response.pdf/.png).  Ignores "
                             "--kind/--inner/--layout.")
    parser.add_argument("--present-only", action="store_true", default=False,
                        help="draw only route/customer/TW classes and "
                             "methods that have runs, instead of the full "
                             "canonical paper layout with empty slots")
    parser.add_argument("--all", action="store_true", default=False,
                        help="regenerate the full paper set in one go: "
                             "box + bar, each in both inner orderings "
                             "(tw and method), plus the stats CSV; "
                             "--kind/--inner are ignored")
    parser.add_argument("--inner", default=None,
                        choices=["tw", "method", "pooled"],
                        help="row layout inner grouping: 'tw' = method "
                             "blocks holding their four TW boxes (default); "
                             "'method' = TW blocks holding the four methods "
                             "side by side (files get a '_methods' suffix); "
                             "'pooled' = the four TW classes merged into ONE "
                             "box per method, so a box holds every instance of "
                             "its (route, customers, method) cell "
                             "(files get a '_pooledtw' suffix)")
    parser.add_argument("--line-no-band", dest="line_band",
                        action="store_false", default=True,
                        help="--kind line: drop the IQR bands and draw only "
                             "the median and mean lines (five overlapping "
                             "bands can be muddy in a dense panel)")
    parser.add_argument("--layout", default="row", choices=["row", "grid"],
                        help="'row' = all data in one horizontal band "
                             "(9 panels side by side, default); "
                             "'grid' = 3x3 facet grid")
    args = parser.parse_args()

    gaps, n_infe, n_unsl, n_feas = collect_gaps(args.dir, metric=args.metric)

    n_groups = len(gaps)
    n_runs   = sum(len(v) for v in gaps.values())
    n_exc    = sum(n_infe.values())
    n_uns    = sum(n_unsl.values())
    print(f"  Pooled {n_runs} feasible run(s) into {n_groups} "
          f"(family x method) group(s); excluded {n_exc} infeasible and "
          f"{n_uns} unsolved (uncertified) run(s)")
    for key in sorted(n_infe):
        if n_infe[key]:
            print(f"    excluded {n_infe[key]:>2} infeasible: "
                  f"{'/'.join(key[:3])} [{key[3]}]")
    for key in sorted(n_unsl):
        if n_unsl[key]:
            print(f"    excluded {n_unsl[key]:>2} unsolved  : "
                  f"{'/'.join(key[:3])} [{key[3]}]")

    os.makedirs(args.out_dir, exist_ok=True)
    # tabular exports live in data_output/, not alongside the .pdf/.png figures
    write_stats_csv(gaps, n_infe, n_unsl, n_feas,
                    _paths.data_output("paper_gap_stats.csv"))

    if args.tw_response:
        # Its own layout (methods on x, customers pooled), so it does not go
        # through the kind/inner combo loop below.  SystemExit(0), not return:
        # this block is module level under __main__, not a function.
        for p in plot_tw_response(gaps, metric=args.metric,
                                  out_dir=args.out_dir):
            print(f"  Figure    : {p}")
        raise SystemExit(0)

    # Bare invocation (no --kind/--inner/--all): every box figure the paper
    # uses, plus the window-response figure, in one command.
    paper_set = (args.kind is None and args.inner is None and not args.all)

    if args.all:
        combos = [(k, i) for k in ("box", "bar") for i in ("tw", "method")]
    elif paper_set:
        combos = [("box", "tw"), ("box", "pooled")]
    else:
        kind  = args.kind  or "box"
        inner = args.inner or "tw"
        kinds = (["box", "bar", "violin", "line"] if kind == "all"
                 else [kind])
        combos = [(k, inner) for k in kinds]
    # The line mark connects the four TW slots inside a method block, so it
    # only means anything with inner="tw": methods are categorical and a line
    # across them would imply an order that does not exist.
    combos = [(k, "tw" if k == "line" else i) for k, i in combos]
    for kind, inner in combos:
        for p in plot_gap_figure(gaps, n_infe, n_unsl, n_feas, kind=kind,
                                 metric=args.metric, out_dir=args.out_dir,
                                 full_grid=not args.present_only,
                                 layout=args.layout, inner=inner,
                                 line_band=args.line_band):
            print(f"  Figure    : {p}")

    if paper_set:
        for p in plot_tw_response(gaps, metric=args.metric,
                                  out_dir=args.out_dir):
            print(f"  Figure    : {p}")
