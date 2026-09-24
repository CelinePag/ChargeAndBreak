"""
stats.py — is the student actually different from the teacher?
==============================================================
The headline is a median percentage difference over ~130 paired routes.  A
median near zero can mean "matches the teacher" or "we have not measured
anything"; those need separating before any claim is made.

Reported per baseline:
  * Wilcoxon signed-rank on the paired differences (non-parametric: the
    percentage differences are heavy-tailed, so a t-test would be wrong)
  * a bootstrap 95% CI on the MEDIAN percentage difference
  * the sign test count, which is what "faster on N of M routes" really is

A practical-significance floor is also applied.  The look-ahead is itself
non-deterministic run to run, so a difference smaller than that floor is not
evidence of anything even when it is statistically significant.  The floor is
supplied by --floor (default 0.35%, the median run-to-run spread previously
measured for LA on this simulator); pass --floor 0 to disable it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def paired(rows, key):
    out = []
    for r in rows:
        if not r.get("route_completed"):
            continue
        b = r.get(key)
        if b is None or r.get(f"{key}_infeasible"):
            continue
        out.append(100.0 * (r["duration_h"] - b) / b)
    return np.array(out)


def boot_median_ci(v, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    meds = np.median(v[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def describe(v, label, floor):
    if len(v) < 5:
        print(f"{label}: too few paired routes ({len(v)})")
        return
    med = float(np.median(v))
    lo, hi = boot_median_ci(v)
    try:
        w_stat, w_p = stats.wilcoxon(v, alternative="two-sided")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    n_faster = int((v < 0).sum())
    sign_p = float(stats.binomtest(n_faster, len(v), 0.5).pvalue)

    print(f"\n{label}   n = {len(v)} paired routes")
    print(f"   median            {med:+7.3f} %   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"   mean              {v.mean():+7.3f} %")
    print(f"   Wilcoxon          p = {w_p:.2e}")
    print(f"   faster on         {n_faster}/{len(v)}   sign test p = {sign_p:.2e}")

    sig = w_p < 0.05
    practical = abs(med) >= floor
    if not sig:
        verdict = "indistinguishable from the baseline"
    elif not practical:
        verdict = (f"statistically detectable but SMALLER than the {floor:.2f} % "
                   f"practical floor -- not a real difference")
    else:
        verdict = ("faster than the baseline" if med < 0
                   else "slower than the baseline")
    print(f"   -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval_gbt_base_g95_test.json",
                    help="an evaluation written by run_all.py")
    ap.add_argument("--floor", type=float, default=0.35,
                    help="practical-significance floor in %% (0 disables)")
    args = ap.parse_args()

    p = os.path.join(RESULTS, args.file)
    if not os.path.exists(p):
        raise SystemExit(f"not found: {p}")
    with open(p) as fh:
        rows = json.load(fh)

    print("=" * 66)
    print(f"SIGNIFICANCE  —  {args.file}")
    print(f"practical floor: {args.floor:.2f} %  "
          f"(LA's own run-to-run median spread)")
    print("=" * 66)

    describe(paired(rows, "LA"), "vs LA (teacher)", args.floor)
    describe(paired(rows, "GREEDY"), "vs GREEDY", args.floor)

    # gap to the oracle uses the manuscript's exact definition, not the
    # `obj - 8.0` shortcut: `obj` is an absolute clock value that also
    # carries the beta*miss penalty, and T_START is not always 8.0.
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    from paper_link import collect_gaps
    g = collect_gaps(args.file, student_label="student")
    describe(g["student"][0], "vs ORACLE (manuscript definition)", args.floor)

    # halts are counted, never averaged
    n = len(rows)
    halted = [r for r in rows if not r.get("route_completed")]
    la_inf = sum(1 for r in rows if r.get("LA_infeasible") or r.get("LA") is None)
    print(f"\nfeasibility:  student halted {len(halted)}/{n}, "
          f"teacher unusable {la_inf}/{n}")
    if halted:
        why = {}
        for r in halted:
            why[r["halt_reason"]] = why.get(r["halt_reason"], 0) + 1
        print("   student violation types: "
              + ", ".join(f"{k}={v}" for k, v in sorted(why.items())))
        # Fisher exact on halt rates, student vs teacher
        tbl = [[len(halted), n - len(halted)], [la_inf, n - la_inf]]
        try:
            _, p_f = stats.fisher_exact(tbl)
            print(f"   student vs teacher infeasible rate: Fisher exact p = {p_f:.3f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
