"""
ood_eval.py — base-case models on instances they were never trained for
=======================================================================
Every learned policy here was fitted on the BASE CASE only: a 500 kWh pack,
350 kW chargers every ~60 km, no ferries.  This drives the best model of each
arm on instances where that physics changes, and compares it with the
teacher (LA), Greedy and the hindsight oracle solved FOR THOSE instances.

    axis       what differs from training
    kwh300     battery 300 kWh              (training: 500)
    kwh900     battery 900 kWh
    kw150      charger 150 kW -> full charge 3.3 h   (training: 1.6 h)
    kw1000     charger 1000 kW -> full charge 0.9 h
    cs100      chargers ~100 km apart -> 34 instead of 56
    usecase    real Arendal-NL/DE tour: 7 customers, 2 ferry crossings

Only route seeds 22-25 are used on the synthetic axes -- the base-case TEST
seeds -- so the ONLY difference from the base-case test result is the physics.
Any drop is attributable to the shift, not to unseen route geometry.

Why this is a hard test for trees in particular: a tree is piecewise-constant
and cannot extrapolate.  At 1000 kW, `charge_rate_now_kw` is far above
anything seen at 350 kW; every such row falls into whatever leaf covered the
largest training value.  The engineered features partly absorb the shift --
`charge_time_to_full` and `e_margin_next_cs_wc` are recomputed from each
instance's own physics -- which is what this experiment actually measures.

The median TRAINING seed of each arm's best configuration is used, not the
best seed: picking the luckiest seed would flatter every arm.

    python ML/code/ood_eval.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.instance_gen.instance_io import load_instance_json      # noqa: E402

from policy_core import load_policy, run_student                  # noqa: E402

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
BETA = 0.5
TEST_SEEDS = {22, 23, 24, 25}

AXES = {
    "kwh300": ("instances_sens/battery_300", "sensitivity", "Battery 300 kWh"),
    "kwh900": ("instances_sens/battery_900", "sensitivity", "Battery 900 kWh"),
    "kw150": ("instances_sens/charger_power_150", "sensitivity", "Charger 150 kW"),
    "kw1000": ("instances_sens/charger_power_1000", "sensitivity", "Charger 1000 kW"),
    "cs100": ("instances_sens/cs_spacing_100", "sensitivity", "Chargers every 100 km"),
    "usecase": ("instances_usecase", "usecase", "Real route (use case)"),
}

# best configuration per arm on the base case (the feature-set ladder,
# results/ladder_test.json); MEDIAN training seed of it.  The MLP appears
# twice: D77 is its best base-case set (+0.09 vs LA over 3 seeds), F95 the one
# used before its ladder finished -- and D77 drops columns that are duplicates
# only under base-case physics (e.g. soc_frac = soc_kwh / 500 only while the
# pack is 500 kWh), so the pair shows what that selection costs under shift.
MODELS = [
    ("gbt", "gbt_F95_base_s1", "Trees F95"),
    ("clf", "clf_F77_base_s1", "Classifier F77"),
    ("nn", "mlp_D77_base_s1", "MLP D77"),
    ("nn", "mlp_F95_base_s2", "MLP F95"),
]
LEARNED = [lab for _k, _t, lab in MODELS]


def _latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def oracle(bucket, inst):
    p = os.path.join(_ROOT, "solutions", bucket, f"oracle_{inst}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        o = json.load(fh)
    sol = o.get("sol") or []
    if not o.get("feasible") or not sol or o.get("obj") is None:
        return None
    ta_N = float(sol[-1]["ta"])
    return dict(ta_N=ta_N, pen=float(o["obj"]) - ta_N)


def gap(duration_h, tw, t0, orc):
    """The manuscript's gap to the oracle (compile_solutions' definition)."""
    if duration_h is None or orc is None:
        return None, None
    od = orc["ta_N"] - t0
    if od <= 0:
        return None, None
    return (100.0 * (duration_h - od) / od,
            100.0 * (duration_h + BETA * tw - od - orc["pen"]) / (od + orc["pen"]))


def baseline(bucket, inst, method, orc, t0_default):
    f = _latest(os.path.join(_ROOT, "solutions", bucket, f"{inst}_{method}_*.json"))
    if not f:
        return None
    with open(f) as fh:
        s = json.load(fh)
    m = s.get("metrics", {})
    if m.get("run_infeasible") or s.get("duration_h") is None:
        return dict(completed=False, gap=None, gap_pen=None,
                    tw=int(m.get("tw_n_misses", 0)))
    t0 = (s["sim_arrival_h"] - s["duration_h"]) if s.get("sim_arrival_h") else t0_default
    g, gp = gap(s["duration_h"], int(m.get("tw_n_misses", 0)), t0, orc)
    return dict(completed=True, gap=g, gap_pen=gp,
                tw=int(m.get("tw_n_misses", 0)))


def variant_tail(guard_q, spread_room):
    """'' for the default (guard 0.95, no spread room), else '_g99sr' etc. --
    the same variant names as run_length.py."""
    v = f"g{round(100 * guard_q)}{'sr' if spread_room else ''}"
    return "" if v == "g95" else f"_{v}"


def instances(axis):
    d, _, _ = AXES[axis]
    out = []
    for p in sorted(glob.glob(os.path.join(_ROOT, d, "*.json"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if axis != "usecase":
            base = name.split("__")[0]
            if int(base.rsplit("_", 1)[1]) not in TEST_SEEDS:
                continue
        out.append((name, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default=",".join(AXES))
    ap.add_argument("--guard-q", type=float, default=0.95)
    ap.add_argument("--spread-room", action="store_true",
                    help="count the charge and stop overhead against the 15 h "
                         "spread (policy_core.spread_room)")
    args = ap.parse_args()
    store = os.path.join(RESULTS, f"ood_test{variant_tail(args.guard_q, args.spread_room)}.json")

    pols = [(lab, tag, load_policy(kind, tag, guard_q=args.guard_q,
                                   spread_room=args.spread_room))
            for kind, tag, lab in MODELS]
    rows = []
    t_all = time.time()
    for axis in args.axes.split(","):
        _, bucket, axis_label = AXES[axis]
        insts = instances(axis)
        print(f"\n=== {axis}: {axis_label} — {len(insts)} instances ===", flush=True)
        t0a = time.time()
        for name, path in insts:
            fd, D_real, E_real, cv = load_instance_json(path)
            fd["_horizon_h"] = 24.0
            t0 = float(fd.get("T_START", 8.0))
            orc = oracle(bucket, name)
            for lab, tag, pol in pols:
                r = run_student(fd, D_real, E_real, pol, cv=cv)
                g, gp = gap(r["duration_h"], r["tw_misses"], t0, orc)
                rows.append(dict(axis=axis, axis_label=axis_label, instance=name,
                                 method=lab, tag=tag, completed=r["route_completed"],
                                 gap=g, gap_pen=gp, tw=r["tw_misses"],
                                 violation=r.get("halt_reason")))
            for meth, key in (("LA", "LA"), ("Greedy", "GREEDY")):
                b = baseline(bucket, name, key, orc, t0)
                if b is not None:
                    rows.append(dict(axis=axis, axis_label=axis_label,
                                     instance=name, method=meth, tag=meth, **b))
        print(f"   done in {time.time()-t0a:.0f}s", flush=True)
        with open(store, "w") as fh:
            json.dump(rows, fh, indent=1)
    report(rows)
    print(f"\n[total {time.time()-t_all:.0f}s]  saved: {store}")


def report(rows):
    methods = LEARNED + ["LA", "Greedy"]
    print("\n" + "=" * 96)
    print("OUT OF DISTRIBUTION — median gap to oracle (%)  [infeasible / routes]")
    print("=" * 96)
    print(f"{'axis':24s}" + "".join(f"{m:>14s}" for m in methods))
    print("-" * 96)
    for axis, (_, _, lab) in AXES.items():
        sub = [r for r in rows if r["axis"] == axis]
        if not sub:
            continue
        line = f"{lab:24s}"
        for m in methods:
            mr = [r for r in sub if r["method"] == m]
            if not mr:
                line += f"{'—':>14s}"
                continue
            g = [r["gap"] for r in mr if r["completed"] and r["gap"] is not None]
            inf = sum(1 for r in mr if not r["completed"])
            med = f"{np.median(g):+.2f}" if g else "n/a"
            line += f"{med + f' [{inf}/{len(mr)}]':>14s}"
        print(line)


if __name__ == "__main__":
    main()
