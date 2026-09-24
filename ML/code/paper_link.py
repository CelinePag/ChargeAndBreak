"""
paper_link.py — put the student into the MAIN PAPER's own comparison table
==========================================================================
The ML tree reports "% vs LA" because the look-ahead is the teacher.  The
manuscript reports something else: **gap to the hindsight ORACLE**, per method,
in `data_output/paper_gap_stats.csv` and the box plots in
`figures/basecase/paper_gap_box*.png`.  Those are not the same number, and the
approximation `ORACLE.obj - 8.0` used elsewhere in this tree is not either --
`obj` is an absolute clock value that ALSO carries the beta*miss penalty.

This module recomputes the gap exactly as `compile_solutions._annotate_gap_to_oracle`
does, so the student can be dropped into the manuscript's own table:

    oracle_duration      = ta_N - t0            (ta_N from the oracle schedule)
    oracle_duration_pen  = oracle_duration + (obj - ta_N)      [+ beta*misses]
    gap_nopen            = (duration     - oracle_duration)     / oracle_duration
    gap_pen              = (duration_pen - oracle_duration_pen) / oracle_duration_pen

`t0` is recovered per run as `sim_arrival_h - duration_h`, exactly as the
manuscript's pipeline does, rather than assumed to be 8.0.

Every baseline is read from the SAME stored runs the manuscript uses, and
restricted to the instances the student was evaluated on, so the comparison is
paired rather than a table lookup.

Read-only with respect to the main tree; writes only ML/RESULTS_PAPER.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SOL = os.path.join(_ROOT, "solutions", "basecase")
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RESULTS_PAPER.md"))
BETA = 0.5

METHODS = ["LA_MIPTAIL", "GREEDY", "2SP", "RO"]
LABEL = {"LA_MIPTAIL": "LA (look-ahead MILP)", "GREEDY": "GREEDY",
         "2SP": "2SP", "RO": "RO"}


# Reading the oracle and four baselines for every instance, once per
# CONFIGURATION, is ~5k JSON loads per figure.  They do not change between
# configurations, so cache them per process.
_ORACLE_CACHE: dict = {}
_BASELINE_CACHE: dict = {}


def oracle_facts(inst):
    if inst in _ORACLE_CACHE:
        return _ORACLE_CACHE[inst]
    v = _oracle_facts_uncached(inst)
    _ORACLE_CACHE[inst] = v
    return v


def _oracle_facts_uncached(inst):
    """(oracle_duration_pen_offset, ta_N) -- everything but t0, which is per run."""
    p = os.path.join(SOL, f"oracle_{inst}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        o = json.load(fh)
    if not o.get("feasible"):
        return None
    sol = o.get("sol") or []
    if not sol:
        return None
    ta_N = sol[-1].get("ta")
    misses = sum(int(s.get("delta") or 0) for s in sol)
    obj = o.get("obj")
    if ta_N is None or obj is None:
        return None
    return dict(ta_N=float(ta_N), misses=int(misses),
                pen=float(obj) - float(ta_N))      # = beta * misses


def gaps(duration_h, tw_misses, arrival_h, orc):
    """The manuscript's two gaps, for one run."""
    if duration_h is None or orc is None or arrival_h is None:
        return None, None
    t0 = arrival_h - duration_h
    ora_dur = orc["ta_N"] - t0
    if ora_dur <= 0:
        return None, None
    ora_pen = ora_dur + orc["pen"]
    dur_pen = duration_h + BETA * tw_misses
    return (100.0 * (duration_h - ora_dur) / ora_dur,
            100.0 * (dur_pen - ora_pen) / ora_pen)


def _latest(pat):
    fs = sorted(glob.glob(pat))
    return fs[-1] if fs else None


def baseline_gaps(inst, orc):
    if inst in _BASELINE_CACHE:
        return _BASELINE_CACHE[inst]
    out = _baseline_gaps_uncached(inst, orc)
    _BASELINE_CACHE[inst] = out
    return out


def _baseline_gaps_uncached(inst, orc):
    out = {}
    for m in METHODS:
        f = _latest(os.path.join(SOL, f"{inst}_{m}_*.json"))
        if not f:
            continue
        with open(f) as fh:
            s = json.load(fh)
        met = s.get("metrics", {})
        if met.get("run_infeasible") or s.get("duration_h") is None:
            out[m] = (None, None)
            continue
        out[m] = gaps(s.get("duration_h"), met.get("tw_n_misses", 0),
                      s.get("sim_arrival_h"), orc)
    return out


def collect_gaps(eval_file, student_label="ML student (GBT)"):
    """{method: (gaps, pen_gaps, n_infeasible)} for one evaluation, paired on
    the instances that evaluation covers.

    Each method keeps its OWN completed set, as the manuscript's per-cell
    tables do (a method that halts contributes no duration, never a zero).
    The count is reported so the reader can see it.
    """
    with open(os.path.join(RESULTS, eval_file)) as fh:
        rows = json.load(fh)
    out = {student_label: ([], [], 0)}
    for m in METHODS:
        out[LABEL[m]] = ([], [], 0)
    for r in rows:
        orc = oracle_facts(r["instance"])
        if orc is None:
            continue
        f = _latest(os.path.join(SOL, f"{r['instance']}_LA_MIPTAIL_*.json"))
        t0 = 8.0
        if f:
            with open(f) as fh:
                s_ = json.load(fh)
            if s_.get("duration_h") is not None and s_.get("sim_arrival_h"):
                t0 = s_["sim_arrival_h"] - s_["duration_h"]
        if r.get("duration_h") is None:              # halted: no duration
            out[student_label] = (out[student_label][0], out[student_label][1],
                                  out[student_label][2] + 1)
        else:
            g, gp = gaps(r["duration_h"], r.get("tw_misses", 0),
                         r["duration_h"] + t0, orc)
            if g is not None:
                out[student_label][0].append(g)
                out[student_label][1].append(gp)
        for m, (a, b) in baseline_gaps(r["instance"], orc).items():
            k = LABEL[m]
            if a is None:
                out[k] = (out[k][0], out[k][1], out[k][2] + 1)
            else:
                out[k][0].append(a)
                out[k][1].append(b)
    return {k: (np.array(v[0]), np.array(v[1]), v[2]) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval_base_test.json")
    ap.add_argument("--label", default="ML student (GBT)")
    args = ap.parse_args()

    with open(os.path.join(RESULTS, args.file)) as fh:
        rows = json.load(fh)

    per = {m: [] for m in METHODS}
    per_pen = {m: [] for m in METHODS}
    ml, ml_pen = [], []
    n_no_oracle = 0
    for r in rows:
        orc = oracle_facts(r["instance"])
        if orc is None:
            n_no_oracle += 1
            continue
        arr = (r["duration_h"] + 8.0) if r.get("duration_h") is not None else None
        # student runs store duration only; t0 is the instance's T_START, which
        # the baselines expose as sim_arrival_h - duration_h.  Take it from the
        # LA run of the same instance so both sides use the identical t0.
        f = _latest(os.path.join(SOL, f"{r['instance']}_LA_MIPTAIL_*.json"))
        if f:
            with open(f) as fh:
                s = json.load(fh)
            if s.get("duration_h") is not None and s.get("sim_arrival_h"):
                t0 = s["sim_arrival_h"] - s["duration_h"]
                arr = (r["duration_h"] + t0) if r.get("duration_h") is not None else None
        g, gp = gaps(r.get("duration_h"), r.get("tw_misses", 0), arr, orc)
        if g is not None:
            ml.append(g)
            ml_pen.append(gp)
        for m, (a, b) in baseline_gaps(r["instance"], orc).items():
            if a is not None:
                per[m].append(a)
                per_pen[m].append(b)

    L = []
    A = L.append
    A("# The student inside the manuscript's own comparison\n")
    A("The ML tree reports \"% vs LA\" because the look-ahead is the teacher. "
      "The manuscript reports **gap to the hindsight ORACLE** "
      "(`data_output/paper_gap_stats.csv`, `figures/basecase/paper_gap_box*.png`). "
      "This table recomputes that gap for the student using "
      "`compile_solutions._annotate_gap_to_oracle`'s exact definition, on the "
      f"same instances.\n")
    A(f"Source: `ML/results/{args.file}` — {len(ml)} instances with a usable "
      f"oracle" + (f" ({n_no_oracle} lacked one)" if n_no_oracle else "") + ".\n")
    A("| method | n | gap to oracle (median) | mean | penalised gap (median) |")
    A("|---|---:|---:|---:|---:|")
    A(f"| **{args.label}** | {len(ml)} | **{np.median(ml):+.2f}%** | "
      f"{np.mean(ml):+.2f}% | {np.median(ml_pen):+.2f}% |")
    for m in METHODS:
        if per[m]:
            A(f"| {LABEL[m]} | {len(per[m])} | {np.median(per[m]):+.2f}% | "
              f"{np.mean(per[m]):+.2f}% | {np.median(per_pen[m]):+.2f}% |")
    A("")
    A("Lower is better; 0% would be the hindsight optimum. The student is a "
      "solver-free policy, so its row belongs next to GREEDY in cost and next "
      "to LA in quality.\n")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT}\n")
    print(f"{'method':24s} {'n':>4s} {'gap median':>11s} {'mean':>8s} "
          f"{'pen median':>11s}")
    print("-" * 62)
    print(f"{args.label:24s} {len(ml):4d} {np.median(ml):+10.2f}% "
          f"{np.mean(ml):+7.2f}% {np.median(ml_pen):+10.2f}%")
    for m in METHODS:
        if per[m]:
            print(f"{LABEL[m]:24s} {len(per[m]):4d} {np.median(per[m]):+10.2f}% "
                  f"{np.mean(per[m]):+7.2f}% {np.median(per_pen[m]):+10.2f}%")


if __name__ == "__main__":
    main()
