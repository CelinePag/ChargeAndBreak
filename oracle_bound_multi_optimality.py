"""
oracle_bound_multi_optimality.py — Per-class oracle bound convergence,
coloured by optimality, faded instances + full-colour group average
=====================================================================
A variant of oracle_bound_multi.py.  Same normalized per-class convergence
plot (upper = incumbent, lower = best bound, both as % distance from that
instance's best solution found), but:

  * instances are coloured by whether the oracle reached optimality —
        black — final gap < threshold  (proved / effectively optimal)
        red   — final gap >= threshold (stopped without closing)
    where "optimal" is decided from the trace's LAST ub/lb, not the cache
    (many instances close in cutting planes, so the last node-table line still
    shows a gap while the solve actually ended at ub = lb).
  * individual instances are drawn faded; a full-colour AVERAGE curve per
    optimality group is overlaid (interpolated onto a shared log-time grid).
  * axes: x starts at 1 s and ends at the max time present in the data (per
    class); y is fixed to [-5, 5] %.  Both overridable on the CLI.

Data source
-----------
Curves come from the Gurobi B&B logs written by the pipeline / the one-off
tracer (logs/oracle_*_gurobi.log and logs/oracle_trace_*.log).  If those were
deleted, regenerate them first, e.g.:

    python oracle_bound_trace.py instances/RlongCfewTlarge_12.json --time_limit 900

Usage
-----
  python oracle_bound_multi_optimality.py
  python oracle_bound_multi_optimality.py --opt-gap-pct 0.5 --ylim -8 8
  python oracle_bound_multi_optimality.py --glob "logs/oracle_trace_Rlong*.log"
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from oracle_bound_trace import parse_gurobi_log

_CLASS_ORDER = ["short", "medium", "long"]
_CLASS_TITLE = {"short": "Short", "medium": "Medium", "long": "Long"}
_PROD_CAP_S = 2 * 3600     # ORACLE default wall-clock cap (runner_dispatch)

# optimality colours
_COL_OPT    = "#000000"   # final gap < threshold
_COL_NONOPT = "#D62728"   # final gap >= threshold

# defaults (overridable on the CLI)
_DEF_OPT_GAP_PCT = 0.5    # final-gap threshold in % for "optimal" (= oracle MIPGap)
_DEF_XMIN        = 1.0    # s — Gurobi logs node times in whole seconds, so
                          # anything faster is "within the first second"
_DEF_YLIM        = (-10.0, 5.0)   # asymmetric: stalled bounds sit near -8%,
                                  # incumbents converge to 0 from just above


def _route_class(name: str) -> str:
    m = re.search(r"R(short|medium|long)", name)
    return m.group(1) if m else "long"


def _instance_from_log(path: str) -> str:
    """Recover the instance id from an oracle Gurobi-log filename (real-run
    oracle_<inst>_gurobi.log or one-off oracle_trace_<inst>.log)."""
    base = re.sub(r"\.log$", "", os.path.basename(path))
    if base.startswith("oracle_trace_"):
        return base[len("oracle_trace_"):]
    base = re.sub(r"_gurobi$", "", base)
    if base.startswith("oracle_"):
        return base[len("oracle_"):]
    return base


def _final_gap_pct(inc: list, bnd: list) -> float:
    """Final optimality gap (%) from the last incumbent/bound in the trace."""
    if not inc or not bnd or inc[-1] == 0:
        return float("inf")
    return abs(inc[-1] - bnd[-1]) / abs(inc[-1]) * 100.0


def collect(glob_pat: str, opt_gap_pct: float):
    """instance -> (route_class, times, incumbents, bounds, is_optimal).

    Several comma-separated globs may be pooled; when two logs map to the same
    instance the one with more samples wins.  Optimality is decided from the
    trace's final gap versus opt_gap_pct."""
    out = {}
    for pat in glob_pat.split(","):
        for path in sorted(glob.glob(pat.strip())):
            inst = _instance_from_log(path)
            t, inc, bnd = parse_gurobi_log(path)
            if len(t) < 2:
                continue
            if inst in out and len(out[inst][1]) >= len(t):
                continue
            is_opt = _final_gap_pct(inc, bnd) < opt_gap_pct
            out[inst] = (_route_class(inst), t, inc, bnd, is_opt)
    return out


def _group_average(group: list, grid_log: np.ndarray):
    """Mean UB% and LB% across a group's instances on a shared log-time grid.

    Each instance's curve is interpolated onto grid_log; np.interp clamps to the
    endpoints, so before an instance's first sample it holds the first value and
    after it terminates it holds its final value (a finished solve keeps its last
    incumbent/bound).  group: list of (x, ub, lb)."""
    ubs, lbs = [], []
    for x, ub, lb in group:
        lx = np.log10(np.asarray(x, dtype=float))
        ubs.append(np.interp(grid_log, lx, ub))
        lbs.append(np.interp(grid_log, lx, lb))
    return np.mean(ubs, axis=0), np.mean(lbs, axis=0)


def plot_class(route_class: str, items: list, out_base: str,
               xlim: tuple, ylim: tuple):
    """items: list of (instance, times, incumbents, bounds, is_optimal)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    n_opt = n_non = 0
    groups: dict = {}   # colour -> list of (x, ub, lb)
    for _inst, t, inc, bnd, opt in items:
        col = _COL_OPT if opt else _COL_NONOPT
        n_opt += bool(opt)
        n_non += (not opt)
        ref = min(inc)                       # best solution found for this inst
        if ref <= 0:
            continue
        ub = [(u / ref - 1.0) * 100.0 for u in inc]
        lb = [(b / ref - 1.0) * 100.0 for b in bnd]
        x  = [max(ti, xlim[0]) for ti in t]
        # faded individual curves
        ax.plot(x, ub, color=col, lw=1.0, alpha=0.22, solid_capstyle="round")
        ax.plot(x, lb, color=col, lw=1.0, alpha=0.22, ls="--", dashes=(4, 2))
        groups.setdefault(col, []).append((x, ub, lb))

    # full-colour average per optimality group, on a shared log-time grid.
    # The average only exists once EVERY member has produced a sample, so
    # start the grid at the group's latest first-sample time — otherwise the
    # left edge mixes real data with values clamped back from later samples.
    for col, group in groups.items():
        t_start  = max(g[0][0] for g in group)
        grid_log = np.linspace(np.log10(max(t_start, xlim[0])),
                               np.log10(xlim[1]), 240)
        grid_x   = 10.0 ** grid_log
        avg_ub, avg_lb = _group_average(group, grid_log)
        ax.fill_between(grid_x, avg_lb, avg_ub, color=col, alpha=0.08)
        ax.plot(grid_x, avg_ub, color=col, lw=2.4, alpha=1.0,
                solid_capstyle="round", zorder=6)
        ax.plot(grid_x, avg_lb, color=col, lw=2.4, alpha=1.0,
                ls="--", dashes=(4, 2), zorder=6)

    ax.axhline(0.0, color="#333", lw=0.9, zorder=1)
    ax.text(xlim[0] * 1.15, 0.0, " optimum", va="bottom", ha="left",
            fontsize=7.5, color="#555")
    if xlim[0] < _PROD_CAP_S < xlim[1]:
        ax.axvline(_PROD_CAP_S, color="#999", lw=1.0, ls=":")
        ax.text(_PROD_CAP_S, ylim[1], "2 h cap ",
                rotation=90, va="top", ha="right", fontsize=7.2, color="#999")

    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("wall-clock time (s, log scale)")
    ax.set_ylabel("distance from best solution (%)")
    ax.set_title(f"{_CLASS_TITLE[route_class]} routes — oracle bound "
                 f"convergence (n={len(items)})\n"
                 f"black = optimal, red = no proof   "
                 f"(bold = group average, faded = instances)",
                 fontsize=9.5)
    ax.grid(True, which="both", color="#ececec", lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    status_handles = [
        Line2D([], [], color=_COL_OPT, lw=2.4,
               label=f"optimal (n={n_opt})"),
        Line2D([], [], color=_COL_NONOPT, lw=2.4,
               label=f"no proof (n={n_non})"),
    ]
    style_handles = [
        Line2D([], [], color="#555", lw=1.3, label="incumbent (UB)"),
        Line2D([], [], color="#555", lw=1.3, ls="--", dashes=(4, 2),
               label="best bound (LB)"),
    ]
    leg1 = ax.legend(handles=status_handles, fontsize=7.5, frameon=False,
                     loc="upper right", title="optimality",
                     title_fontsize=7.5)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=7.5, frameon=False,
              loc="lower right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = f"{out_base}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Per-class oracle bound convergence: instances faded, "
                    "full-colour group average, coloured by final-gap "
                    "optimality (black optimal, red no proof).")
    ap.add_argument("--glob",
                    default="logs/oracle_*_gurobi.log,logs/oracle_trace_*.log",
                    help="comma-separated glob(s) of oracle Gurobi logs "
                         "(default: real-run oracle logs + one-off trace logs)")
    ap.add_argument("--out-dir", default="figures",
                    help="output directory (default: figures)")
    ap.add_argument("--opt-gap-pct", type=float, default=_DEF_OPT_GAP_PCT,
                    help="final gap (%%) at or below which an instance counts "
                         f"as optimal (default: {_DEF_OPT_GAP_PCT}, = the oracle "
                         "MIPGap). Lower it for a stricter proof criterion.")
    ap.add_argument("--xmin", type=float, default=_DEF_XMIN,
                    help=f"x-axis start in s (default: {_DEF_XMIN}). All class "
                         "figures share one x-axis so they can be compared.")
    ap.add_argument("--xmax", type=float, default=None,
                    help="x-axis end in s (default: max time across ALL "
                         "classes, so every figure uses the same axis).")
    ap.add_argument("--ylim", type=float, nargs=2, default=list(_DEF_YLIM),
                    metavar=("YMIN", "YMAX"),
                    help=f"y-axis (%% from best). Default {_DEF_YLIM}")
    args = ap.parse_args()

    data = collect(args.glob, args.opt_gap_pct)
    if not data:
        raise SystemExit(
            f"no usable trace logs matched {args.glob}\n"
            "The oracle Gurobi logs may have been deleted — regenerate them "
            "with:  python oracle_bound_trace.py instances/<inst>.json")

    by_class = {c: [] for c in _CLASS_ORDER}
    for inst, tup in sorted(data.items()):
        by_class.setdefault(tup[0], []).append((inst, *tup[1:]))

    ylim = tuple(args.ylim)
    # ONE x-axis for every class figure: the three plots are meant to be read
    # side by side, so a per-class axis would make short routes (which finish
    # in ~10 s) look identical to long ones (hours) and would also reshape
    # every figure whenever a new log is added.  Span the whole dataset.
    data_xmax = max(max(t) for _, t, *_ in data.values())
    xmax = args.xmax if args.xmax is not None else data_xmax
    xmin = max(args.xmin, 1e-3)
    if xmax <= xmin:
        xmax = xmin * 10
    xlim = (xmin, xmax)
    print(f"  shared x-axis: [{xmin:g}, {xmax:g}] s")
    for rc in _CLASS_ORDER:
        items = by_class.get(rc)
        if not items:
            continue
        out_base = os.path.join(args.out_dir, f"oracle_bounds_opt_{rc}")
        print(f"  {rc:6}: {len(items)} instance(s) -> {out_base}.png")
        for p in plot_class(rc, items, out_base, xlim, ylim):
            print(f"    {p}")
