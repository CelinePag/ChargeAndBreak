"""
fpi_eval.py — the improved student against the student it corrects, and the LA
==============================================================================
    python ML/code/fpi_eval.py --tag fpi1_gbt_F95_base_s1_g99sr --set test --draws 8

Both comparisons are paired, the two policies facing identical travel times:

  realised : every route of the set on its own realised travel times, with the
             LA and the hindsight oracle from their stored runs -- the numbers
             the paper reports (gap to the oracle, extra daily rests, window
             misses);
  draws    : --draws fresh draws per route from the same distribution (seeds
             that fpi_collect.py never uses).  There is no LA or oracle run
             there, but the paired difference is measured on --draws times as
             many routes, which is what a one-round gain of tenths of an hour
             needs to be seen at all.

--set test     : the 125 test routes (seeds 22-25), never seen by the student
                 or by G
--set la-extra : the base-case routes on which the LA rests more than the
                 oracle (mostly training routes: read as a paired effect)
Output: ML/results/fpi_<tag>_<set>[_m<min_gain>].json, with a .partial.jsonl
that lets an interrupted run resume.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one LightGBM thread per worker

import argparse                                                     # noqa: E402
import json                                                         # noqa: E402
import statistics as st                                             # noqa: E402
import sys                                                          # noqa: E402
import time                                                         # noqa: E402
from multiprocessing import Pool                                    # noqa: E402

import numpy as np                                                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.simulation.scenarios import generate_scenarios            # noqa: E402

from fpi_policy import FPIPolicy, crc                               # noqa: E402
from paper_link import oracle_facts                                 # noqa: E402
from policy_core import MODELS, load_policy, run_student            # noqa: E402
from run_endgame import la_extra_routes                             # noqa: E402
from run_rollout import (RESULTS, la_run, load, n_rests,            # noqa: E402
                         oracle_rests, route_class, summarise, test_routes)

BETA = 0.5
_BASE = None
_POL = None
_A = None


def _init(args):
    global _BASE, _POL, _A
    _A = args
    _BASE = load_policy("gbt", args.meta["base"], guard_q=args.meta["guard_q"],
                        spread_room=args.meta["spread_room"])
    _POL = FPIPolicy(_BASE, args.tag, min_gain=args.min_gain)


def _pen(r):
    return (r["duration_h"] + BETA * r["tw_misses"]) if r["route_completed"] else None


def _one(inst):
    fd, D, E, cv = load(inst)
    T0 = float(fd.get("T_START", 8.0))
    orc = oracle_facts(inst)
    n_stops = int(fd["N"])
    out = dict(instance=inst, route=route_class(inst),
               oracle_rests=oracle_rests(inst), LA=la_run(inst))
    out["student"] = summarise(run_student(fd, D, E, _BASE, cv=cv), orc, T0)
    _POL.reset()
    out["fpi"] = summarise(run_student(fd, D, E, _POL, cv=cv), orc, T0)
    out["fpi"].update(changed=_POL.n_changed, scored=_POL.n_scored,
                      decisions=_POL.n_decisions)
    draws = []
    for k in range(_A.draws):
        sc = generate_scenarios(fd, 0, n_stops, 1, cv,
                                seed=crc(f"{inst}|ev|{k}|{_A.seed}"))[0]
        Dk = [sc["D"][leg] for leg in range(n_stops)]
        Ek = [sc["E"][leg] for leg in range(n_stops)]
        rs = run_student(fd, Dk, Ek, _BASE, cv=cv)
        _POL.reset()
        rf = run_student(fd, Dk, Ek, _POL, cv=cv)
        draws.append(dict(student=_pen(rs), fpi=_pen(rf),
                          rests_s=n_rests(rs["actions"]), rests_f=n_rests(rf["actions"]),
                          changed=_POL.n_changed))
    out["draws"] = draws
    return out


def _paired(rows, a, b):
    """Penalised-duration differences b - a, route by route (realised)."""
    out = []
    for r in rows:
        x, y = r.get(a) or {}, r.get(b) or {}
        if x.get("completed") and y.get("completed"):
            out.append((y["duration"] + BETA * y["tw"]) - (x["duration"] + BETA * x["tw"]))
    return out


def report(rows):
    print(f"\nREALISED travel times, {len(rows)} routes")
    for key in ("LA", "student", "fpi"):
        ok = [r for r in rows if (r[key] or {}).get("completed")
              and r["oracle_rests"] is not None]
        fails = sum(1 for r in rows if r[key] and not r[key].get("completed"))
        g = [r[key]["gap_pen"] for r in ok if r[key].get("gap_pen") is not None]
        extra = sum(1 for r in ok if r[key]["rests"] > r["oracle_rests"])
        if g:
            print(f"  {key:8s} gap_pen median {st.median(g):5.2f} %  mean {st.mean(g):5.2f} %"
                  f"  extra-rest routes {extra:3d}/{len(ok)}  TW misses "
                  f"{sum(r[key]['tw'] for r in ok):4d}  infeasible {fails}")
    for cls in ("long", "medium", "short"):
        sub = [r for r in rows if r["route"] == cls]
        parts = []
        for key in ("LA", "student", "fpi"):
            g = [r[key]["gap_pen"] for r in sub if (r[key] or {}).get("completed")
                 and r[key].get("gap_pen") is not None]
            if g:
                parts.append(f"{key} {st.mean(g):5.2f}")
        if parts:
            print(f"    {cls:7s} mean gap_pen %: " + "   ".join(parts))
    for a, b in (("student", "fpi"), ("LA", "fpi"), ("LA", "student")):
        p = _paired(rows, a, b)
        if p:
            print(f"  {b} - {a}: mean {st.mean(p):+.3f} h, median {st.median(p):+.3f} h, "
                  f"better on {sum(x < -1e-6 for x in p)}, worse on "
                  f"{sum(x > 1e-6 for x in p)} of {len(p)}")
    ch = [r["fpi"]["changed"] / max(r["fpi"]["decisions"], 1) for r in rows]
    print(f"  decisions changed by G: {100 * st.mean(ch):.1f}% on average")

    pairs = [(r["route"], r["instance"], d) for r in rows for d in r.get("draws", [])]
    if not pairs:
        return
    both = [(c, i, d["fpi"] - d["student"], d["rests_f"] - d["rests_s"])
            for c, i, d in pairs if d["fpi"] is not None and d["student"] is not None]
    fails_s = sum(1 for _, _, d in pairs if d["student"] is None)
    fails_f = sum(1 for _, _, d in pairs if d["fpi"] is None)
    print(f"\nFRESH DRAWS, {len(pairs)} (route, draw) pairs; infeasible: student "
          f"{fails_s}, fpi {fails_f}")
    for cls in ("all", "long", "medium", "short"):
        sub = [x for x in both if cls == "all" or x[0] == cls]
        if not sub:
            continue
        diffs = np.array([x[2] for x in sub])
        # SE over ROUTE means: draws of one route are not independent evidence
        per_route = {}
        for c, i, dd, _ in sub:
            per_route.setdefault(i, []).append(dd)
        means = np.array([np.mean(v) for v in per_route.values()])
        se = means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else float("nan")
        dr = np.array([x[3] for x in sub])
        print(f"  {cls:7s} fpi - student: mean {diffs.mean():+.3f} h (+- {se:.3f} over "
              f"{len(means)} routes)  better {np.mean(diffs < -1e-6) * 100:4.1f}%  "
              f"worse {np.mean(diffs > 1e-6) * 100:4.1f}%  rests fewer {np.sum(dr < 0)}"
              f" / more {np.sum(dr > 0)}  of {len(sub)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="test", choices=["test", "la-extra"])
    ap.add_argument("--routes", default=None, help="comma-separated instances")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    with open(os.path.join(MODELS, f"{args.tag}_meta.json")) as fh:
        args.meta = json.load(fh)

    if args.routes:
        routes = [r.strip() for r in args.routes.split(",") if r.strip()]
        name = "routes"
    else:
        routes = test_routes() if args.set == "test" else la_extra_routes()
        name = args.set
    stem = (f"fpi_{args.tag}_{name}_d{args.draws}"
            + (f"_m{args.min_gain:g}" if args.min_gain else ""))
    os.makedirs(RESULTS, exist_ok=True)
    partial = os.path.join(RESULTS, stem + ".partial.jsonl")
    done = {}
    if os.path.exists(partial):
        with open(partial) as fh:
            for line in fh:
                row = json.loads(line)
                done[row["instance"]] = row
    todo = [r for r in routes if r not in done]
    order = {"long": 0, "medium": 1, "short": 2}
    todo.sort(key=lambda i: (order.get(route_class(i), 3), i))
    print(f"[fpi-eval] {stem}: {len(routes)} routes, {len(done)} done, "
          f"{len(todo)} to run on {args.jobs} workers", flush=True)
    t0 = time.time()
    if todo:
        with Pool(args.jobs, initializer=_init, initargs=(args,)) as pool, \
                open(partial, "a") as fh:
            for row in pool.imap_unordered(_one, todo):
                done[row["instance"]] = row
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if len(done) % 25 == 0:
                    print(f"  {len(done)}/{len(routes)}  [{(time.time() - t0) / 60:.0f} min]",
                          flush=True)
    rows = [done[r] for r in routes if r in done]
    out = os.path.join(RESULTS, stem + ".json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    report(rows)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
