"""
oracle_bound_trace.py — Trace and plot the oracle MIP bound evolution
=====================================================================
The pipeline never saves the oracle solver log, so there is no historical
bound-evolution data to plot.  This script RUNS one oracle solve with Gurobi's
branch-and-bound log captured to a file, parses the node table, and plots the
incumbent (best feasible arrival, an upper bound) against the best bound (lower
bound) over wall-clock time — the shaded region between them is the optimality
gap that stays open on the hard long-route instances.

The realised travel times are the instance's precomputed D_real — exactly what
the real oracle solves on — so the traced difficulty matches the pipeline.

Usage
-----
  python oracle_bound_trace.py instances/RlongCfewTlarge_12.json
  python oracle_bound_trace.py instances/RlongCfewTlarge_12.json --time_limit 900
  python oracle_bound_trace.py <inst> --mip_gap 0.0 --out figures/bound_trace
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from instance_io import load_instance_json
from oracle import oracle_solve


# ── Gurobi B&B log parser ─────────────────────────────────────────────────────
# A node line ends in a time token like "  120s" and carries a gap "3.35%"; the
# two numeric tokens immediately before the gap are Incumbent then BestBd, e.g.
#   *  100    50        45   104.500000  100.00000  4.31%   6.0   40s
_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)s\s*$")
_GAP_RE  = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
_FLOAT   = re.compile(r"[-+]?\d+\.\d+")
_SUMMARY = re.compile(
    r"Best objective\s+([0-9.eE+-]+),\s+best bound\s+([0-9.eE+-]+)")
# total solve time from Gurobi's "Explored N nodes ... in 42.34 seconds" line,
# used to place the closure point (many instances close in cutting planes, so
# the last node-table line still shows a gap while the solve ended at ub = lb)
_RUNTIME = re.compile(r"\bin\s+([0-9]+(?:\.[0-9]+)?)\s+seconds\b")


_RUN_START = re.compile(r"logging started")


def parse_gurobi_log(path: str):
    """Return (times, incumbents, bounds) sampled from the node table, with a
    final closure point appended from the summary line so proved-optimal solves
    end at ub = lb even when they closed in cutting planes.

    Gurobi APPENDS to LogFile, so re-solving an instance concatenates several
    B&B logs into one file.  Only the MOST RECENT run is parsed: otherwise the
    series jumps backwards in time and the incumbent appears to increase,
    which corrupts both the per-instance curve and any average taken over it.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    starts = [k for k, ln in enumerate(lines) if _RUN_START.search(ln)]
    if starts:
        lines = lines[starts[-1]:]

    times, inc, bnd = [], [], []
    final_time = None
    for line in lines:
        mt = _TIME_RE.search(line.rstrip())
        mg = _GAP_RE.search(line)
        if not (mt and mg):
            mr = _RUNTIME.search(line)
            if mr:
                try:
                    final_time = float(mr.group(1))
                except ValueError:
                    pass
            # final summary line (proves-optimal / final incumbent+bound)
            ms = _SUMMARY.search(line)
            if ms:
                try:
                    ub, lb = float(ms.group(1)), float(ms.group(2))
                    # prefer the true solve time; never go backwards
                    t = final_time if final_time is not None else 0.0
                    if times:
                        t = max(t, times[-1])
                    times.append(t); inc.append(ub); bnd.append(lb)
                except ValueError:
                    pass
            continue
        t = float(mt.group(1))
        head = line[:mg.start()]
        floats = _FLOAT.findall(head)
        if len(floats) < 2:
            continue
        ub, lb = float(floats[-2]), float(floats[-1])
        if ub < lb - 1e-6:            # mis-parse (UB must dominate LB); skip
            continue
        times.append(t); inc.append(ub); bnd.append(lb)
    return times, inc, bnd


def plot_trace(times, inc, bnd, meta: dict, out_base: str):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if times:
        ax.fill_between(times, bnd, inc, color="#E69F00", alpha=0.18,
                        step="post", label="optimality gap")
        ax.step(times, inc, where="post", color="#D55E00", lw=1.6,
                label="incumbent (best feasible, UB)")
        ax.step(times, bnd, where="post", color="#0072B2", lw=1.6,
                label="best bound (LB)")
        ax.scatter([times[-1]], [inc[-1]], color="#D55E00", s=18, zorder=5)
        ax.scatter([times[-1]], [bnd[-1]], color="#0072B2", s=18, zorder=5)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("arrival time (h)")
    gap_txt = (f"{meta['final_gap']*100:.2f}%"
               if meta.get("final_gap") is not None else "n/a")
    ax.set_title(f"Oracle MIP bound evolution — {meta['instance']}\n"
                 f"N={meta['N']} stops, status={meta['status']}, "
                 f"final gap {gap_txt}", fontsize=10)
    ax.grid(True, color="#e0e0e0", lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
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
    ap = argparse.ArgumentParser(description="Trace + plot oracle MIP bounds.")
    ap.add_argument("instance", help="instance JSON path")
    ap.add_argument("--time_limit", type=int, default=600,
                    help="solver time cap in seconds (default 600). Note: the "
                         "oracle's MIPGap is 0.005, so an easy instance may "
                         "stop before the cap once it hits 0.5%.")
    ap.add_argument("--out", default=None,
                    help="output path base (default figures/bound_trace_<inst>)")
    ap.add_argument("--warmstart", action="store_true", default=False,
                    help="seed the MIP with a quiet greedy run first, as the "
                         "ORACLE pipeline does (default: cold start)")
    ap.add_argument("--tee", action="store_true", default=False,
                    help="show the live Gurobi log on the console as well as "
                         "writing it to the log file")
    args = ap.parse_args()

    full_data, D_real, E_real, _cv = load_instance_json(args.instance)
    N = full_data["N"]
    stem = os.path.splitext(os.path.basename(args.instance))[0]
    out_base = args.out or os.path.join("figures", f"bound_trace_{stem}")
    log_path = os.path.join("logs", f"oracle_trace_{stem}.log")
    os.makedirs("logs", exist_ok=True)

    # the instance's precomputed realised travel times — exactly what the real
    # oracle solves on
    D_actual_list = list(D_real)

    sim_results = None
    if args.warmstart:
        from greedy import run_greedy
        print(f"  Greedy warm-start run for {stem} ...")
        sim_results = run_greedy(full_data, D_real, E_real,
                                 verbose=False, oracle_tee=False,
                                 persist=False)   # warm start, not a result

    print(f"  Solving oracle for {stem} (N={N}) with a {args.time_limit}s cap, "
          f"Gurobi log -> {log_path}")
    res = oracle_solve(full_data, D_actual_list, sim_results=sim_results,
                       time_limit=args.time_limit, tee=args.tee, verbose=True,
                       log_file=log_path)

    times, inc, bnd = parse_gurobi_log(log_path)
    meta = dict(instance=stem, N=N, status=res.get("status"),
                final_gap=(None if not bnd or not inc or inc[-1] == 0
                           else abs(inc[-1] - bnd[-1]) / abs(inc[-1])))
    print(f"  Parsed {len(times)} node-log sample(s); "
          f"status={meta['status']}  obj={res.get('obj')}")
    if not times:
        print("  WARNING: no node-table lines parsed (solve may have finished "
              "in presolve, or the log format differs).")
    paths = plot_trace(times, inc, bnd, meta, out_base)
    for p in paths:
        print(f"  Figure : {p}")
