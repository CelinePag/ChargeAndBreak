"""
oracle_bound_multi.py — Per-route-class oracle bound convergence, normalized
============================================================================
Reads the Gurobi B&B logs written by oracle_bound_trace.py
(logs/oracle_trace_<instance>.log) and produces ONE figure per route class
(short / medium / long).  Each figure overlays every instance of that class,
showing BOTH bounds expressed as percentage distance from that instance's best
solution found, so all instances share the same axis:

    upper (incumbent)  UB%(t) = (incumbent(t) / ref - 1) * 100   >= 0
    lower (best bound) LB%(t) = (best_bound(t) / ref - 1) * 100  <= 0
    ref = best incumbent found for that instance (= the optimum when the solve
          proved optimal; the best feasible otherwise)

Both bounds squeeze toward 0% at optimality, so a solve that closes shows both
curves meeting at 0, while a solve that stalls leaves the lower curve short.

Usage
-----
  python -m src.output_analysis.oracle_bound_multi                       # all classes present
  python -m src.output_analysis.oracle_bound_multi --glob "oracle_trace_Rlong*.log"
  python -m src.output_analysis.oracle_bound_multi --out-dir figures
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.output_analysis.oracle_bound_trace import parse_gurobi_log
from src import paths as _paths

_CLASS_ORDER = ["short", "medium", "long"]
_CLASS_TITLE = {"short": "Short", "medium": "Medium", "long": "Long"}
# per-instance qualitative palette (Okabe-Ito, colourblind-safe)
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
            "#E69F00", "#56B4E9", "#8a6d3b", "#333333"]
_PROD_CAP_S = 12 * 3600


def _route_class(name: str) -> str:
    m = re.search(r"R(short|medium|long)", name)
    return m.group(1) if m else "long"


def _instance_from_log(path: str) -> str:
    """Recover the instance id from an oracle Gurobi-log filename, handling both
    the real-run pipeline (oracle_<instance>_gurobi.log) and the one-off tracer
    (oracle_trace_<instance>.log)."""
    base = os.path.basename(path)
    base = re.sub(r"\.log$", "", base)
    if base.startswith("oracle_trace_"):
        return base[len("oracle_trace_"):]
    base = re.sub(r"_gurobi$", "", base)
    if base.startswith("oracle_"):
        return base[len("oracle_"):]
    return base


def collect(glob_pat: str):
    """instance -> (route_class, times, incumbents, bounds).

    Accepts a single glob or several comma-separated globs, so the real-run
    oracle logs and the one-off trace logs can be pooled together.  When two
    logs map to the same instance, the one with more samples wins."""
    out = {}
    for path in _paths.expand_logs(glob_pat):
        inst = _instance_from_log(path)
        t, inc, bnd = parse_gurobi_log(path)
        if len(t) < 2:
            continue
        if inst in out and len(out[inst][1]) >= len(t):
            continue
        out[inst] = (_route_class(inst), t, inc, bnd)
    return out


def plot_class(route_class: str, items: list, out_base: str):
    """items: list of (instance, times, incumbents, bounds)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for k, (inst, t, inc, bnd) in enumerate(items):
        col = _PALETTE[k % len(_PALETTE)]
        ref = min(inc)                       # best solution found for this inst
        if ref <= 0:
            continue
        ub = [(u / ref - 1.0) * 100.0 for u in inc]
        lb = [(b / ref - 1.0) * 100.0 for b in bnd]
        x  = [max(ti, 0.1) for ti in t]
        ax.fill_between(x, lb, ub, color=col, alpha=0.07, step=None)
        ax.plot(x, ub, color=col, lw=1.4, solid_capstyle="round")
        ax.plot(x, lb, color=col, lw=1.4, ls="--", dashes=(4, 2))
        ax.scatter([x[-1]], [ub[-1]], color=col, s=13, zorder=5)
        ax.scatter([x[-1]], [lb[-1]], color=col, s=13, zorder=5,
                   facecolors="none", edgecolors=col, linewidths=1.0)

    ax.axhline(0.0, color="#333", lw=0.9, zorder=1)
    ax.text(0.12, 0.0, " optimum", va="bottom", ha="left",
            fontsize=7.5, color="#555")
    ax.axvline(_PROD_CAP_S, color="#999", lw=1.0, ls=":")
    ax.text(_PROD_CAP_S, ax.get_ylim()[1], "12 h cap ",
            rotation=90, va="top", ha="right", fontsize=7.2, color="#999")

    ax.set_xscale("log")
    ax.set_xlim(0.1, _PROD_CAP_S * 1.6)
    ax.set_xlabel("wall-clock time (s, log scale)")
    ax.set_ylabel("distance from best solution (%)")
    ax.set_title(f"{_CLASS_TITLE[route_class]} routes — oracle bound "
                 f"convergence (n={len(items)})\n"
                 f"upper = incumbent, lower = best bound", fontsize=10)
    ax.grid(True, which="both", color="#ececec", lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    inst_handles = [Line2D([], [], color=_PALETTE[k % len(_PALETTE)], lw=2,
                           label=inst)
                    for k, (inst, *_ ) in enumerate(items)]
    style_handles = [
        Line2D([], [], color="#555", lw=1.4, label="incumbent (UB)"),
        Line2D([], [], color="#555", lw=1.4, ls="--", dashes=(4, 2),
               label="best bound (LB)"),
    ]
    leg1 = ax.legend(handles=inst_handles, fontsize=7.5, frameon=False,
                     loc="upper right", title="instance",
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
        description="Per-class oracle bound convergence (UB/LB as % from opt).")
    ap.add_argument("--glob",
                    default="oracle_*_gurobi.log,oracle_trace_*.log",
                    help="comma-separated glob(s) of oracle Gurobi logs, "
                         "searched across logs/ and its experiment buckets "
                         "(default: real-run oracle logs + one-off traces)")
    ap.add_argument("--out-dir", default=_paths.figures(),
                    help="output directory (default: figures)")
    args = ap.parse_args()

    data = collect(args.glob)
    if not data:
        raise SystemExit(f"no usable trace logs matched {args.glob}")

    by_class = {c: [] for c in _CLASS_ORDER}
    for inst, (rc, t, inc, bnd) in sorted(data.items()):
        by_class.setdefault(rc, []).append((inst, t, inc, bnd))

    for rc in _CLASS_ORDER:
        items = by_class.get(rc)
        if not items:
            continue
        out_base = os.path.join(args.out_dir, f"oracle_bounds_{rc}")
        print(f"  {rc:6}: {len(items)} instance(s) -> {out_base}.png")
        for p in plot_class(rc, items, out_base):
            print(f"    {p}")
