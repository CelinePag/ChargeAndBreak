"""
rl_report.py — paired summary of an rl_search run
=================================================
For every variant: the paired difference to the plain student on the SAME
draws, averaged over a route's draws first and then over routes (± one
standard error over routes), overall and by route length.  Cost = duration +
0.5 h per missed window.  A variant run that breaks a rule while the student
does not counts as the student's cost + FAIL_H on that draw (both broken:
0); failures are also listed.

    python ML/code/rl_report.py v1_gbt [--realised-only | --fresh-only]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FAIL_H = 24.0


def load(name):
    with open(os.path.join(RESULTS, f"rl_search_{name}.jsonl")) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def paired(base, var):
    """Per draw: (d_cost h, d_cost % of student, d_rests, d_misses, var failed,
    student failed, changed, d_overrun)."""
    out = []
    for b, v in zip(base, var):
        if b["failed"] and v["failed"]:
            dc = 0.0
        elif v["failed"]:
            dc = FAIL_H
        elif b["failed"]:
            dc = -FAIL_H
        else:
            dc = v["cost"] - b["cost"]
        ref = b["cost"] if b["cost"] else (v["cost"] or 1.0)
        out.append((dc, 100.0 * dc / ref, v["rests"] - b["rests"],
                    v["misses"] - b["misses"], int(v["failed"]), int(b["failed"]),
                    v["changed"] / max(v["decisions"], 1), v["overrun"] - b["overrun"]))
    return np.array(out, dtype=float)


def headroom(recs):
    """How much of the student's cost is the rest-count cliff, from the
    student's own runs on fresh draws (variant 0), by route length:
      knife-edge  routes whose rest count differs between draws
      sd cost     sd of the route cost across its draws (h), mean over routes
      fewer       draws on which SOME variant took fewer daily rests than the
                  student on the same draw (an upper bound on what a
                  coordinated change could avoid on that draw, in hindsight:
                  the variant is not chosen per draw by a real policy)."""
    print("\nheadroom (student, fresh draws):")
    print(f"{'length':8s} {'routes':>6s} {'knife-edge':>11s} {'sd cost h':>10s} "
          f"{'rests':>6s} {'fewer by any variant':>21s}")
    by = defaultdict(lambda: dict(n=0, knife=0, sd=[], rests=[], fewer=0, draws=0))
    for r in recs:
        res = r["res"]
        lo = 1 if r.get("realised_first") else 0
        st = [x for x in res[0][lo:]]
        ln = ("short" if r["instance"].startswith("Rshort") else
              "medium" if r["instance"].startswith("Rmedium") else "long")
        b = by[ln]
        b["n"] += 1
        rc = [x["rests"] for x in st if not x["failed"]]
        b["knife"] += int(len(set(rc)) > 1)
        cs = [x["cost"] for x in st if not x["failed"]]
        if len(cs) > 1:
            b["sd"].append(np.std(cs, ddof=1))
        b["rests"] += rc
        for d, x in enumerate(st):
            b["draws"] += 1
            others = [res[v][lo + d] for v in range(1, len(res))]
            b["fewer"] += int(any((not o["failed"]) and o["rests"] < x["rests"] for o in others))
    for ln in ("short", "medium", "long"):
        if ln in by:
            b = by[ln]
            print(f"{ln:8s} {b['n']:6d} {100 * b['knife'] / b['n']:10.1f}% "
                  f"{np.mean(b['sd']):10.2f} {np.mean(b['rests']):6.2f} "
                  f"{100 * b['fewer'] / b['draws']:20.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--realised-only", action="store_true")
    g.add_argument("--fresh-only", action="store_true")
    ap.add_argument("--headroom", action="store_true")
    a = ap.parse_args()
    recs = load(a.name)
    if a.headroom:
        headroom(recs)
    variants = recs[0]["variants"]
    per = defaultdict(lambda: defaultdict(list))      # variant -> length -> route rows
    base_cost = defaultdict(list)
    for r in recs:
        res = r["res"]
        sel = slice(None)
        if r.get("realised_first"):
            sel = slice(0, 1) if a.realised_only else slice(1, None) if a.fresh_only else slice(None)
        b = res[0][sel]
        ln = "short" if r["instance"].startswith("Rshort") else "medium" if r["instance"].startswith("Rmedium") else "long"
        base_cost[ln].append(np.mean([x["cost"] for x in b if not x["failed"]] or [np.nan]))
        for vi, vn in enumerate(variants[1:], 1):
            p = paired(b, res[vi][sel])
            per[vn][ln].append(p.mean(axis=0))
            per[vn]["all"].append(p.mean(axis=0))
    nd = recs[0]["n_draws"]
    print(f"{a.name}: {len(recs)} routes x {nd} draws"
          f"{' (realised only)' if a.realised_only else ' (fresh only)' if a.fresh_only else ''};"
          f" student cost (h): " + ", ".join(f"{k} {np.nanmean(v):.1f}" for k, v in base_cost.items()))
    hdr = (f"{'variant':22s} {'d cost h':>16s} {'d %':>14s} {'short %':>8s} {'medium %':>8s} "
           f"{'long %':>8s} {'d rests':>8s} {'d miss':>7s} {'chg %':>6s} {'fail':>5s} {'d ovr':>6s}")
    print(hdr)
    for vn in variants[1:]:
        A = np.array(per[vn]["all"])
        n = len(A)
        se = lambda c: A[:, c].std(ddof=1) / np.sqrt(n)
        by = {ln: (np.mean([x[1] for x in per[vn][ln]]) if per[vn][ln] else np.nan)
              for ln in ("short", "medium", "long")}
        print(f"{vn:22s} {A[:, 0].mean():+7.3f} ± {se(0):5.3f} {A[:, 1].mean():+6.2f} ± {se(1):4.2f} "
              f"{by['short']:+8.2f} {by['medium']:+8.2f} {by['long']:+8.2f} "
              f"{A[:, 2].mean():+8.3f} {A[:, 3].mean():+7.3f} {100 * A[:, 6].mean():6.2f} "
              f"{A[:, 4].sum() * nd:5.0f} {A[:, 7].mean():+6.3f}")


if __name__ == "__main__":
    main()
