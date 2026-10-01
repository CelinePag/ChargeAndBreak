"""
phys_eval.py — trees trained across physics, driven on held-out routes
======================================================================
Closed-loop companion of run_phys.py.  Every model drives the TEST route
seeds (22-25) of each physics value; the only thing that changes between
axes is the physics, so any difference between models on one axis is what
their training data taught them about it.

    model                      axes it drives
    gbt_F95_base_s1            every axis   (the base-only control, ood_eval)
    gbt_F95_phys_s1            every axis   (all physics seen)
    gbt_F95_phys_LO<v>_s1      axis <v>     (<v> never seen)

Gap to the oracle is ood_eval.gap -- the manuscript's definition, the one
every other table in RESULTS.md uses.  The teacher and Greedy are read from
their stored runs on the same routes; comparisons with them are PAIRED, on
the routes where both sides completed.

One jsonl row per (model, variant, instance); a rerun skips what is stored.

    python ML/code/phys_eval.py [--jobs 4] [--guard-q 0.99 --spread-room]
    python ML/code/phys_eval.py --report
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

from ood_eval import baseline, gap, oracle, variant_tail          # noqa: E402
from run_phys import HOLD_OUT, tag_of                             # noqa: E402

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
TEST_SEEDS = {22, 23, 24, 25}

# axis -> (instance dir, solution bucket, label, teacher's method name)
AXES = {
    "base": ("instances", "basecase", "Base case (500 kWh, 350 kW, 60 km)", "LA_MIPTAIL"),
    "kwh300": ("instances_sens/battery_300", "sensitivity", "Battery 300 kWh", "LA"),
    "kwh700": ("instances_sens/battery_700", "sensitivity", "Battery 700 kWh", "LA"),
    "kwh900": ("instances_sens/battery_900", "sensitivity", "Battery 900 kWh", "LA"),
    "kw150": ("instances_sens/charger_power_150", "sensitivity", "Charger 150 kW", "LA"),
    "kw700": ("instances_sens/charger_power_700", "sensitivity", "Charger 700 kW", "LA"),
    "kw1000": ("instances_sens/charger_power_1000", "sensitivity", "Charger 1000 kW", "LA"),
    "cs30": ("instances_sens/cs_spacing_30", "sensitivity", "Chargers every 30 km", "LA"),
    "cs100": ("instances_sens/cs_spacing_100", "sensitivity", "Chargers every 100 km", "LA"),
}
REGIME = {"kwh700": "interpolation", "kw700": "interpolation",
          "kwh300": "extrapolation", "kwh900": "extrapolation",
          "kw150": "extrapolation", "kw1000": "extrapolation",
          "cs100": "extrapolation", "cs30": "extrapolation", "base": "seen"}

BASE_TAG = "gbt_F95_base_s1"
ALL_TAG = tag_of("phys")


def plan(seed=1):
    """(model tag, axis) pairs to drive -- see the module docstring."""
    out = [(BASE_TAG, a) for a in AXES] + [(tag_of("phys", seed=seed), a) for a in AXES]
    out += [(tag_of(f"phys_LO{v}", seed=seed), v) for v in HOLD_OUT]
    out += [(tag_of("physR3k", seed=seed), a) for a in AXES]     # 3000-round cap
    out += [(f"gbt_P102_phys_s{seed}", a) for a in AXES]          # + charger power
    return out


def instances(axis):
    d = AXES[axis][0]
    out = []
    for p in sorted(glob.glob(os.path.join(_ROOT, d, "*.json"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if int(name.split("__")[0].rsplit("_", 1)[1]) in TEST_SEEDS:
            out.append((name, p))
    return out


_POL = {}


def _drive(job):
    """Worker: one model on one route.  Policies are cached per process."""
    tag, axis, name, path, guard_q, spread_room = job
    from policy_core import load_policy, run_student
    if tag not in _POL:
        _POL[tag] = load_policy("gbt", tag, guard_q=guard_q, spread_room=spread_room)
    fd, D_real, E_real, cv = load_instance_json(path)
    fd["_horizon_h"] = 24.0
    t0 = float(fd.get("T_START", 8.0))
    r = run_student(fd, D_real, E_real, _POL[tag], cv=cv)
    orc = oracle(AXES[axis][1], name)
    g, gp = gap(r["duration_h"], r["tw_misses"], t0, orc)
    rests = sum(1 for k in r["actions"] if k.endswith(("_r1", "_r2")))
    return dict(tag=tag, axis=axis, instance=name, completed=r["route_completed"],
                duration_h=r["duration_h"], gap=g, gap_pen=gp, tw=r["tw_misses"],
                rests=rests, violation=r.get("halt_reason"),
                ms_per_decision=r["ms_per_decision"])


def store_path(guard_q, spread_room):
    v = f"g{round(100 * guard_q)}{'sr' if spread_room else ''}"
    return os.path.join(RESULTS, f"phys_{v}.jsonl")


def read_store(path):
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--guard-q", type=float, default=0.99)
    ap.add_argument("--spread-room", action="store_true", default=True)
    ap.add_argument("--no-spread-room", dest="spread_room", action="store_false")
    ap.add_argument("--models", default="", help="restrict to these tags")
    ap.add_argument("--axes", default="", help="restrict to these axes")
    ap.add_argument("--limit", type=int, default=0, help="routes per (model, axis)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--report", action="store_true", help="report only")
    args = ap.parse_args()
    store = store_path(args.guard_q, args.spread_room)
    if args.report:
        report(read_store(store), store)
        return

    done = {(r["tag"], r["instance"]) for r in read_store(store)}
    jobs, missing = [], set()
    for tag, axis in plan(args.seed):
        if args.models and tag not in args.models.split(","):
            continue
        if args.axes and axis not in args.axes.split(","):
            continue
        if not os.path.exists(os.path.join(RESULTS, "..", "models", f"{tag}_meta.json")):
            missing.add(tag)
            continue
        insts = instances(axis)[: args.limit or None]
        jobs += [(tag, axis, n, p, args.guard_q, args.spread_room)
                 for n, p in insts if (tag, n) not in done]
    if missing:
        print(f"[wait] not trained yet, skipped: {', '.join(sorted(missing))}")
    print(f"[phys_eval] {len(jobs)} routes to drive -> {store}", flush=True)

    t0 = time.time()
    with open(store, "a") as fh:
        if args.jobs > 1:
            from multiprocessing import Pool
            with Pool(args.jobs) as pool:
                for i, row in enumerate(pool.imap_unordered(_drive, jobs, 2)):
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    if (i + 1) % 50 == 0:
                        print(f"  {i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
        else:
            for i, job in enumerate(jobs):
                fh.write(json.dumps(_drive(job)) + "\n")
                fh.flush()
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print(f"[phys_eval] done in {time.time()-t0:.0f}s")
    report(read_store(store), store)


# ── reporting ────────────────────────────────────────────────────────────────

REST_DWELL_H = 8.9      # a stop dwell this long is a daily rest (9-11 h)


def oracle_rests(bucket, name):
    p = os.path.join(_ROOT, "solutions", bucket, f"oracle_{name}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        o = json.load(fh)
    if not o.get("feasible") or not o.get("sol"):
        return None
    return int(sum(round(s.get("rho1", 0)) + round(s.get("rho2", 0)) for s in o["sol"]))


def stored_rests(bucket, name, method):
    """Daily rests a stored run EXECUTED, read from its dwells: its `actions`
    are what the LA selected, which the nominal re-solve may have moved."""
    fs = sorted(glob.glob(os.path.join(_ROOT, "solutions", bucket, f"{name}_{method}_*.json")))
    if not fs:
        return None
    with open(fs[-1]) as fh:
        s = json.load(fh)
    tr, td = s.get("sim_trajectory") or [], s.get("td_list") or []
    return sum(1 for k in range(min(len(tr), len(td)))
               if float(td[k]) - float(tr[k]["t_arr"]) >= REST_DWELL_H)


def baselines(axis, names):
    """Stored teacher / Greedy runs on these routes: {method: {inst: row}}."""
    d, bucket, _, la = AXES[axis]
    out = {"LA": {}, "Greedy": {}}
    for name in names:
        orc = oracle(bucket, name)
        fd = load_instance_json(os.path.join(_ROOT, d, name + ".json"))[0]
        t0 = float(fd.get("T_START", 8.0))
        for meth, key in (("LA", la), ("Greedy", "GREEDY")):
            b = baseline(bucket, name, key, orc, t0)
            if b is not None:
                b["rests"] = stored_rests(bucket, name, key)
                out[meth][name] = b
    return out


def _paired(a, b):
    """Paired gap difference a - b (pp) over routes where both completed."""
    ks = [k for k in a if k in b and a[k]["completed"] and b[k]["completed"]
          and a[k]["gap"] is not None and b[k]["gap"] is not None]
    if not ks:
        return None
    diff = np.array([a[k]["gap"] - b[k]["gap"] for k in ks])
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    return dict(n=len(ks), mean=float(diff.mean()), se=float(se),
                median=float(np.median(diff)), win=float((diff < 0).mean()))


def summarise(rows):
    """{axis: {method: summary}} -- models by tag, plus LA and Greedy."""
    out = {}
    for axis in AXES:
        sub = [r for r in rows if r["axis"] == axis]
        if not sub:
            continue
        names = sorted({r["instance"] for r in sub})
        base = baselines(axis, names)
        o_rests = {n: oracle_rests(AXES[axis][1], n) for n in names}
        per = {m: base[m] for m in base}
        for tag in sorted({r["tag"] for r in sub}):
            per[tag] = {r["instance"]: r for r in sub if r["tag"] == tag}
        res = {}
        for m, runs in per.items():
            g = [r["gap"] for r in runs.values() if r["completed"] and r["gap"] is not None]
            cmp_ = [(r.get("rests"), o_rests.get(k)) for k, r in runs.items()
                    if r["completed"] and r.get("rests") is not None
                    and o_rests.get(k) is not None]
            res[m] = dict(
                extra_rests=sum(1 for a, b in cmp_ if a > b), rest_n=len(cmp_),
                n=len(runs), infeasible=sum(1 for r in runs.values() if not r["completed"]),
                median=float(np.median(g)) if g else None,
                mean=float(np.mean(g)) if g else None,
                tw=int(sum(r.get("tw", 0) for r in runs.values())),
                vs_LA=_paired(runs, per["LA"]) if m != "LA" else None,
                vs_base=(_paired(runs, per[BASE_TAG])
                         if BASE_TAG in per and m not in (BASE_TAG, "LA", "Greedy") else None))
        out[axis] = res
    return out


def _label(m):
    if m == BASE_TAG:
        return "trees, base only"
    if m.startswith("gbt_P102_phys"):
        return "trees, all + power"
    if m.startswith("gbt_F95_physR3k"):
        return "trees, all, 3000 rnd"
    if m.startswith("gbt_F95_phys_LO"):
        return "trees, all but this"
    if m.startswith("gbt_F95_phys_"):
        return "trees, all physics"
    return m


def report(rows, store=None):
    S = summarise(rows)
    print("\n" + "=" * 110)
    print("TREES ACROSS PHYSICS - gap to oracle (%), test seeds 22-25; paired vs LA = mean +/- se "
          "of (model - LA) in pp, routes where both completed")
    print("=" * 110)
    hdr = f"{'axis':26s} {'model':22s} {'n':>4s} {'inf':>4s} {'median':>8s} {'mean':>8s} {'TW':>5s} {'+rest':>6s}   {'vs LA (pp)':>22s} {'win':>5s}   {'vs base-only':>18s}"
    print(hdr)
    for axis, res in S.items():
        print("-" * 110)
        order = [BASE_TAG] + sorted(m for m in res if m.startswith(("gbt_F95_phys", "gbt_P102_phys")) and "LO" not in m) \
            + sorted(m for m in res if "LO" in m) + ["LA", "Greedy"]
        first = True
        for m in order:
            if m not in res:
                continue
            r = res[m]
            med = f"{r['median']:+.2f}" if r["median"] is not None else "-"
            mean = f"{r['mean']:+.2f}" if r["mean"] is not None else "-"
            p = r["vs_LA"]
            pl = f"{p['mean']:+.2f} +/- {p['se']:.2f} (n={p['n']})" if p else ""
            wn = f"{100*p['win']:.0f}%" if p else ""
            q = r["vs_base"]
            ql = f"{q['mean']:+.2f} +/- {q['se']:.2f}" if q else ""
            lab = f"{AXES[axis][2]} [{REGIME[axis][:6]}]" if first else ""
            first = False
            print(f"{lab:26s} {_label(m):22s} {r['n']:4d} {r['infeasible']:4d} {med:>8s} {mean:>8s} "
                  f"{r['tw']:5d} {str(r['extra_rests']) + '/' + str(r['rest_n']):>6s}   {pl:>22s} {wn:>5s}   {ql:>18s}")
    if store:
        with open(store.replace(".jsonl", "_summary.json"), "w") as fh:
            json.dump(S, fh, indent=1)


if __name__ == "__main__":
    main()
