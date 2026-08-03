"""
oracle_gap_optimality.py — oracle bound convergence, coloured by optimality
===========================================================================
Reads oracle Gurobi bound logs and overlays every instance's bounds as
percentage distance from its best solution, on a fixed axis so all instances
share one frame:

    upper (incumbent)  UB%(t) = (incumbent(t) / ref - 1) * 100   >= 0
    lower (best bound) LB%(t) = (best_bound(t) / ref - 1) * 100  <= 0
    ref = best incumbent found for that instance

Lines are BLACK for instances whose solve reached optimality and RED for
instances that did not (hit the time limit).  Instances whose log was
interrupted (no Gurobi termination line) are skipped.

Usage
-----
  python -m src.output_analysis.oracle_gap_optimality
  python -m src.output_analysis.oracle_gap_optimality --glob "logs/oracle_RmediumC*_gurobi.log"
  python -m src.output_analysis.oracle_gap_optimality --xmax 300 --ymin -25 --ymax 50
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.output_analysis.oracle_bound_trace import parse_gurobi_log
from src import paths as _paths


def _termination(path: str):
    """'optimal', 'timelimit', or None (interrupted) from the Gurobi log tail."""
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return None
    if "Optimal solution found" in txt:
        return "optimal"
    if "Time limit reached" in txt:
        return "timelimit"
    return None


def build(glob_pat: str, out_base: str, xmax: float, ymin: float, ymax: float):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    n_opt = n_not = n_skip = 0
    for path in sorted(glob.glob(glob_pat)):
        term = _termination(path)
        if term is None:
            n_skip += 1
            continue
        t, inc, bnd = parse_gurobi_log(path)
        if len(t) < 2:
            n_skip += 1
            continue
        ref = min(inc)
        if ref <= 0:
            continue
        ub = [(u / ref - 1.0) * 100.0 for u in inc]
        lb = [(b / ref - 1.0) * 100.0 for b in bnd]
        col = "black" if term == "optimal" else "red"
        if term == "optimal":
            n_opt += 1
        else:
            n_not += 1
        ax.plot(t, ub, color=col, lw=0.8, alpha=0.45, solid_capstyle="round")
        ax.plot(t, lb, color=col, lw=0.8, alpha=0.45, solid_capstyle="round")

    ax.axhline(0.0, color="#333333", lw=1.0, zorder=3)
    ax.text(xmax, 0.0, " optimum", va="bottom", ha="right",
            fontsize=8, color="#555555")
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("distance from optimum (%)")
    ax.set_title("Oracle bound convergence by optimality\n"
                 "upper = incumbent, lower = best bound", fontsize=10)
    ax.grid(True, color="#ececec", lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D([], [], color="black", lw=2,
               label=f"reached optimality (n={n_opt})"),
        Line2D([], [], color="red", lw=2,
               label=f"did not — time limit (n={n_not})"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = f"{out_base}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    print(f"  {n_opt} optimal (black), {n_not} time-limit (red), "
          f"{n_skip} skipped (interrupted/empty)")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Oracle bounds coloured by optimality.")
    ap.add_argument("--glob", default=_paths.logs("oracle_RmediumC*_gurobi.log"),
                    help="glob of oracle Gurobi logs (default: medium)")
    ap.add_argument("--out", default=_paths.figures("oracle_gap_optimality"),
                    help="output path base")
    ap.add_argument("--xmax", type=float, default=300.0)
    ap.add_argument("--ymin", type=float, default=-25.0)
    ap.add_argument("--ymax", type=float, default=50.0)
    args = ap.parse_args()
    for p in build(args.glob, args.out, args.xmax, args.ymin, args.ymax):
        print(f"  Figure : {p}")
