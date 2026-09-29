"""
spread_compare.py — the same models, with and without the spread-room check
===========================================================================
`halt_state.py` traced nearly every infeasible run -- on long routes, on slow
chargers, on the real route -- to one gap in the safety layer: the shared
legality check leaves the charge and the stop overhead out of the 15 h spread,
and a learned policy chooses its charge only AFTER that check.
`policy_core.spread_room` closes the gap.  This lines the versions up on the
SAME routes, model by model, so the effect of the fix is separated from
everything else:

    g95      as trained: drive-time guard at the 0.95 quantile, no spread room
    g95sr    + the spread-room check
    g99sr    + the spread-room check, guard at the 0.99 quantile (the residual
             failures after g95sr are all realised drives beyond the guard)

For every model and route set:

    infeasible      runs that broke a rule (a violation ends the run)
    gap to oracle   median over the routes the model completed, as in the paper
    paired          median change in duration on routes BOTH versions completed:
                    what the fix costs where it was not needed

Rows are per training seed in the JSON; the printed table averages the seeds.

    python ML/code/spread_compare.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_link import gaps, oracle_facts                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
STORE = os.path.join(RESULTS, "spread_room.json")
VARIANTS = ("g95sr", "g99sr")

# arm -> (label, sets trained on short+medium, set of the all-length reference)
ARMS = {"gbt": ("Trees", ("F95", "R88"), "F95"),
        "clf": ("Classifier", ("F77", "R70"), "F77"),
        "mlp": ("MLP", ("F95", "R88"), "F95")}


def length_of(inst):
    return inst[1:].split("C")[0]


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def gap_of(r):
    orc = oracle_facts(r["instance"])
    if not r.get("route_completed") or r.get("duration_h") is None or orc is None:
        return None
    g, _ = gaps(r["duration_h"], r.get("tw_misses", 0), r["duration_h"] + 8.0, orc)
    return g


def side(rows):
    inf = [r for r in rows if not r.get("route_completed")]
    g = [x for x in (gap_of(r) for r in rows) if x is not None]
    return dict(routes=len(rows), infeasible=len(inf),
                rules=dict(collections.Counter(r.get("halt_reason") for r in inf)),
                gap=float(np.median(g)) if g else None)


def compare(before, after):
    a = {r["instance"]: r for r in before}
    b = {r["instance"]: r for r in after}
    both = [i for i in a if i in b and a[i].get("route_completed")
            and b[i].get("route_completed")]
    d = [100.0 * (b[i]["duration_h"] - a[i]["duration_h"]) / a[i]["duration_h"]
         for i in both]
    return dict(before=side(before), after=side(after), paired_n=len(both),
                paired_med=float(np.median(d)) if d else None,
                paired_changed=int(sum(abs(x) > 1e-9 for x in d)))


def not_long(i):
    return length_of(i) != "long"


def is_long(i):
    return length_of(i) == "long"


def cases():
    """(model label, trained on, tag, seed, [(route set, file stem, filter)])."""
    for arm, (lab, sm_sets, ref) in ARMS.items():
        for sd in range(3):
            yield (f"{lab} {ref}", "all lengths", f"{arm}_{ref}_base_s{sd}", sd,
                   [("short+medium test", "test", not_long),
                    ("long test (30)", "test", is_long)])
        for fs in sm_sets:
            for sd in range(3):
                yield (f"{lab} {fs}", "short+medium", f"{arm}_{fs}_base_SM_s{sd}", sd,
                       [("short+medium test", "test", not_long),
                        ("long test (30)", "test", is_long),
                        ("long, all 239", "longall", None)])


def collect():
    out = []
    for lab, who, tag, sd, sets in cases():
        for rs, stem, keep in sets:
            before = load(f"eval_{tag}_g95_{stem}.json")
            if before is None:
                continue
            if keep is not None:
                before = [r for r in before if keep(r["instance"])]
            for var in VARIANTS:
                after = load(f"eval_{tag}_{var}_{stem}.json")
                if after is None:
                    continue
                if keep is not None:
                    after = [r for r in after if keep(r["instance"])]
                c = compare(before, after)
                c.update(model=lab, trained=who, tag=tag, seed=sd, routes=rs,
                         variant=var)
                out.append(c)
    return out


def aggregate(rows):
    """Mean over training seeds, per (model, trained on, route set, variant)."""
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault((r["model"], r["trained"], r["routes"], r["variant"]),
                          []).append(r)
    agg = []
    for (model, who, rs, var), v in groups.items():
        def m(f):
            x = [f(r) for r in v if f(r) is not None]
            return float(np.mean(x)) if x else None
        agg.append(dict(model=model, trained=who, routes=rs, variant=var,
                        seeds=len(v), n=v[0]["before"]["routes"],
                        inf_before=m(lambda r: r["before"]["infeasible"]),
                        inf_after=m(lambda r: r["after"]["infeasible"]),
                        gap_before=m(lambda r: r["before"]["gap"]),
                        gap_after=m(lambda r: r["after"]["gap"]),
                        paired_med=m(lambda r: r["paired_med"]),
                        changed=m(lambda r: r["paired_changed"] / max(r["paired_n"], 1))))
    return agg


def main():
    rows = collect()
    with open(STORE, "w") as fh:
        json.dump(rows, fh, indent=1)
    agg = aggregate(rows)
    print("mean over training seeds; infeasible = runs of the route set\n")
    print(f"{'model':15s} {'trained on':13s} {'routes':18s} {'var':6s} "
          f"{'infeasible':>15s} {'gap to oracle %':>17s} {'paired change':>14s} "
          f"{'routes changed':>15s}")
    print("-" * 110)
    last = None
    for a in agg:
        if last and (a["model"], a["trained"]) != last:
            print()
        last = (a["model"], a["trained"])
        print(f"{a['model']:15s} {a['trained']:13s} {a['routes']:18s} {a['variant']:6s} "
              f"{a['inf_before']:5.1f} -> {a['inf_after']:4.1f}/{a['n']:<3d} "
              f"{a['gap_before']:+7.2f} -> {a['gap_after']:+5.2f} "
              f"{a['paired_med']:+13.3f}% {100 * a['changed']:13.0f}%")
    print(f"\nsaved: {STORE}")


if __name__ == "__main__":
    main()
