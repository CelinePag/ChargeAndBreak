"""
run_hybrid.py — the student drives; the LA decides only where a rest is at stake
==============================================================================
On mixed-charger routes the students now charge where the LA does, and most
of what still separates them from it is a handful of runs that take one more
daily rest (5-10 % of runs, 60-73 % of the gap).  In those runs the decision
that set it up -- rest at this stop or drive on -- was a close call for the
student: its best rest and best no-rest action were 1-24 min apart in
predicted cost (ML/scratch/diag_extra_rest.py).

So the student hands exactly those decisions to the LA:

  margin   call the LA when the student's best REST and best NO-REST actions
           (both judged feasible by its feasibility head) are within `margin`
           minutes of each other
  final    also call it at every stop with a choice once the nominal driving
           left to the destination is below `final` hours (the last shift,
           where an overflow of minutes forces the extra rest)

When the LA is called, its decision is executed exactly as in an LA run:
select_best_action with the teacher's configuration (run_la_mixed.LA_CONFIG),
the winner's nominal re-solve as the vehicle's durations.  If every action is
infeasible over its horizon the student's own action is kept.  The student
part runs behind its usual shield (guard 0.99 + spread room).

    python ML/code/run_hybrid.py --dry                       # trigger counts, no LA
    python ML/code/run_hybrid.py --margin 30 --slice 0/4     # one share of a config
    python ML/code/run_hybrid.py --margin 30 --final 10

Rows go to ML/results/hybrid[_val]_<config>.jsonl, one per (route, model);
runs already there are skipped, so shares and restarts are safe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mixed_eval as me                                           # noqa: E402
from features import Precomp, action_key, charger_curve           # noqa: E402
from policy_core import durations, load_policy                    # noqa: E402
from run_la_mixed import LA_CONFIG                                # noqa: E402
from src.simulation.BEHDV import BEHDV                            # noqa: E402

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
DEFAULT_MODELS = ",".join(f"tmlp_T144_physmix_split_list_s{s}" for s in range(3))
RESTS = ("_r1", "_r2")
INFEASIBLE = 1e5          # the LA scores an infeasible action with a huge penalty


def rest_margin(cand):
    """Hours between the student's best rest and best no-rest action, both
    unmasked by its feasibility head; None if either kind is missing."""
    ok = [c for c in cand if c[3] < 1e5]
    r = [c[3] for c in ok if c[2].endswith(RESTS)]
    n = [c[3] for c in ok if not c[2].endswith(RESTS)]
    return abs(min(r) - min(n)) if r and n else None


def ask_la(fd, stop, veh, cv, seed, workers, time_limit):
    from src.simulation.Simulation import select_best_action
    best, scores, nom = select_best_action(
        full_data=fd, stop=stop, state=veh,
        n_scenarios=LA_CONFIG["n_scenarios"],
        horizon_hours=LA_CONFIG["horizon_hours"], cv=cv,
        scenario_seed=seed, time_limit=time_limit, verbose=False,
        n_workers=workers, solve_mode=LA_CONFIG["solve_mode"],
        charge_only=False, criterion=LA_CONFIG["criterion"],
        include_best=False, include_worst=False, prev_nom_sol=None,
        log_fh=None, tracker=None, ext_shift_used=veh.ext_shift_used,
        prune_quantile=LA_CONFIG["prune_quantile"],
        tiebreak_min=LA_CONFIG["tiebreak_min"])
    if all(s[1] >= INFEASIBLE / 2 for s in scores):
        return None
    return best, nom


def drive(fd, D, E, cv, pol, name, tag, margin_h, final_h, dry, workers, time_limit):
    """One route.  Returns the result row (and, dry, the per-stop margins)."""
    pre = Precomp(fd)
    veh = BEHDV(fd)
    N = int(fd["N"])
    T0 = float(fd.get("T_START", 8.0))
    cum = np.r_[0.0, np.cumsum([float(fd["D"].get(i, 0.0)) for i in range(N)])]
    curves, tbar = fd.get("TbarK"), fd["Tbar"]
    calls, la_s, fallback, margins, keys, t_arr = [], 0.0, 0, [], [], []
    while veh.stop < N:
        s = veh.stop
        if curves:
            fd["Tbar"] = charger_curve(fd, s if s in pre.K else int(pre.next_cs[s]))
        cand = pol.candidates(fd, pre, s, veh, cv, k=64)
        act, tc, key, _ = cand[0]
        m = rest_margin(cand)
        left = cum[N] - cum[s]
        margins.append((m, left, len(cand)))
        fire = ((margin_h is not None and m is not None and m <= margin_h)
                or (final_h is not None and left <= final_h and len(cand) > 1))
        plan = None
        if fire and not dry:
            fd["Tbar"] = tbar                         # the LA sees the route as in its own runs
            t0 = time.time()
            got = ask_la(fd, s, veh, cv, zlib.crc32(f"{name}|{tag}|{s}".encode()),
                         workers, time_limit)
            la_s += time.time() - t0
            calls.append(s)
            if got is None:
                fallback += 1
                if curves:
                    fd["Tbar"] = charger_curve(fd, s if s in pre.K else int(pre.next_cs[s]))
            else:
                act, plan = got
                key = action_key(act.get("y", 0), act.get("break_type"),
                                 act.get("rest_type"))
        keys.append(key)
        t_arr.append(veh.t_arr)
        sol = plan if plan is not None else dict(
            feasible=True, sol=[dict(i=0, **durations(fd, s, act, tc))])
        veh.advance(action=act, D_next=float(D[s]), E_next=float(E[s]), milp_sol=sol)
        if veh.is_halted:
            break
    fd["Tbar"] = tbar
    done = (not veh.is_halted) and veh.stop >= N
    # rests from the dwell, as mixed_eval.la_row counts the LA's: what the
    # vehicle did, whoever chose it
    t_arr.append(veh.t_arr)
    dwell = [t_arr[k + 1] - float(D[k]) - t_arr[k] for k in range(len(keys))]
    rests = sum(1 for d in dwell if d >= me.REST_DWELL_H)
    return dict(instance=name, method=tag, completed=done,
                duration_h=(veh.t_arr - T0) if done else None,
                tw=len(veh.tw_misses), rests=rests, violation=veh.halt_reason,
                decisions=len(keys), n_calls=len(calls), la_s=round(la_s, 1),
                call_stops=calls, fallback=fallback), margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="pmix,mix")
    ap.add_argument("--set", default="test", choices=["test", "val"])
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--margin", type=float, default=None, help="minutes")
    ap.add_argument("--final", type=float, default=None, help="hours of driving left")
    ap.add_argument("--dry", action="store_true",
                    help="no LA: drive the student, report how often each margin fires")
    ap.add_argument("--slice", default="0/1")
    ap.add_argument("--workers", type=int, default=LA_CONFIG["n_workers"])
    ap.add_argument("--time-limit", type=int, default=LA_CONFIG["time_limit"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    me.ROUTE_SET = args.set

    margin_h = None if args.margin is None else args.margin / 60.0
    cfg = (f"m{args.margin:g}" if args.margin is not None else "") + \
          (f"f{args.final:g}" if args.final is not None else "")
    if not cfg and not args.dry:
        ap.error("give --margin and/or --final (or --dry)")
    tail = "" if args.set == "test" else f"_{args.set}"
    store = os.path.join(RESULTS, f"hybrid{tail}_{cfg or 'dry'}.jsonl")
    done = set()
    if os.path.exists(store) and not args.dry:
        with open(store) as fh:
            done = {(r["variant"], r["instance"], r["method"])
                    for r in map(json.loads, filter(str.strip, fh))}

    jobs = [(v, n, p, t) for v, n, p in me.instances(args.variants.split(","), uniform=False)
            for t in args.models.split(",")]
    i, n = (int(x) for x in args.slice.split("/"))
    jobs = [j for j in jobs[i::n] if (j[0], j[1], j[3]) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[hybrid] {cfg or 'dry'} on {args.set}: {len(jobs)} runs (slice {args.slice}) "
          f"-> {store}", flush=True)

    pols, all_m = {}, []
    for k, (v, name, path, tag) in enumerate(jobs):
        pol = pols.setdefault(tag, load_policy("torch", tag, guard_q=0.99, spread_room=True))
        fd, D, E, cv = me.load(path)
        fd["title"] = name
        if LA_CONFIG.get("la_energy_quantile"):
            fd["la_energy_quantile"] = float(LA_CONFIG["la_energy_quantile"])
            fd["la_energy_cv"] = cv
        t0 = time.time()
        row, margins = drive(fd, D, E, cv, pol, name, tag, margin_h, args.final,
                             args.dry, args.workers, args.time_limit)
        row["variant"], row["config"] = v, cfg or "dry"
        all_m += [(v, name, tag, *x) for x in margins]
        if not args.dry:
            with open(store, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        print(f"  {k+1}/{len(jobs)} {v} {name} {tag[-2:]}: "
              f"{'done' if row['completed'] else row['violation']} "
              f"{row['duration_h'] or 0:.1f} h, {row['rests']} rests, "
              f"{row['n_calls']} LA calls ({row['la_s']:.0f} s), "
              f"{time.time()-t0:.0f} s", flush=True)

    if args.dry:
        runs = len(jobs) or 1
        print("\ntrigger counts along the student's own trajectories (calls per route):")
        for mm in (5, 10, 15, 30, 60, 120):
            c = sum(1 for x in all_m if x[3] is not None and x[3] <= mm / 60)
            print(f"  margin {mm:4d} min: {c / runs:6.2f}")
        for fh_ in (5, 10):
            c = sum(1 for x in all_m if x[4] <= fh_ and x[5] > 1)
            print(f"  final {fh_:4d} h  : {c / runs:6.2f}")


if __name__ == "__main__":
    main()
