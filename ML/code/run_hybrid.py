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
from features import Precomp, action_key, charger_curve, state_features  # noqa: E402
from policy_core import durations, load_policy                    # noqa: E402
from run_la_mixed import LA_CONFIG                                # noqa: E402
from src.simulation.BEHDV import BEHDV                            # noqa: E402
from src.simulation.Simulation import enumerate_actions           # noqa: E402

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


class Gate:
    """The learned solver-call gate (gate_train.py) at driving time.

    Computes exactly the signals of gate_data.py for the state at hand: the
    student's own margins and feasibility, its disagreement with the other arm
    and with the oracle-trained trees (both run on the same state, same
    shield), and the state features at nominal travel times (as the dataset).
    `score` is the gate's predicted value of calling the LA (minutes, or a
    probability for the classifier); the trigger fires at the validation
    threshold of the chosen budget."""

    STUDENT = {"gbt": "gbt_F95_base_s0", "torch": "tmlp_F95_split_list_s0"}

    def __init__(self, arm, budget, nominal=True, pay_mseq=False, model=None):
        import lightgbm as lgb
        model = model or f"gate_{arm}"
        self.model = model
        self.meta = json.load(open(os.path.join(me.ML, "models", f"{model}_meta.json")))
        self.booster = lgb.Booster(model_file=os.path.join(me.ML, "models", f"{model}.txt"))
        self.thr = float(self.meta["thresholds"][f"{budget:g}"]) if budget is not None else None
        self.names = self.meta["features"]
        other = "torch" if arm == "gbt" else "gbt"
        self.other = load_policy(other, self.STUDENT[other], guard_q=0.99, spread_room=True)
        self.oracle = load_policy("gbt", "gbt_F95_oracle_s0", guard_q=0.99, spread_room=True)
        for p in (self.other, self.oracle):
            if nominal:
                p.feature_q = None
        self.scores = []

    @staticmethod
    def _pick(p, fd, pre, s, veh, cv):
        _l, _r, c, f, _fl = p._score(fd, pre, s, veh, cv)
        return p._legal_keys[int(np.argmin(c + 1e6 * (f < p.feas_thr)))]

    def score(self, fd, pre, s, veh, cv, cost, feas, keys, j):
        cand = np.flatnonzero(feas >= 0.5)
        if len(cand) == 0:
            cand = np.arange(len(cost))
        c = np.sort(cost[cand])
        rest = np.array([k.endswith(RESTS) for k in keys])
        r, n = cost[cand][rest[cand]], cost[cand][~rest[cand]]
        z = -(cost[cand] - c[0]) / 0.25
        pz = np.exp(z) / np.exp(z).sum()
        top2 = cand[np.argsort(cost[cand])[:2]]
        ko = self._pick(self.other, fd, pre, s, veh, cv)
        kr = self._pick(self.oracle, fd, pre, s, veh, cv)
        gap = lambda k: (cost[keys.index(k)] - cost[j]) if k in keys else 0.0
        sig = dict(margin=c[1] - c[0] if len(c) > 1 else 9.0,
                   rest_margin=abs(r.min() - n.min()) if len(r) and len(n) else 9.0,
                   entropy=float(-(pz * np.log(pz + 1e-12)).sum()),
                   pick_feas=feas[j], min_feas_top2=feas[top2].min(), n_cand=len(cand),
                   n_legal=len(enumerate_actions(s, veh, fd, charge_only=False)),
                   pick_rest=float(keys[j].endswith(RESTS)), pick_break=float("_b" in keys[j]),
                   pick_charge=float(keys[j].startswith("y1")),
                   other_arm=float(ko != keys[j]), other_gap=gap(ko),
                   oracle_dis=float(kr != keys[j]), oracle_gap=gap(kr))
        if len(sig) < len(self.names):
            sf, _ = state_features(fd, pre, s, veh, cv, None)       # nominal, as the dataset
            sig.update(sf)
        x = np.array([[float(sig[nm]) for nm in self.names]])
        v = float(self.booster.predict(x)[0])
        self.scores.append(v)
        return v


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


def drive(fd, D, E, cv, pol, name, tag, margin_h, final_h, dry, workers, time_limit,
          gate=None, rand_p=None, check=None):
    """One route.  Returns the result row (and, dry, the per-stop margins).

    Triggers (any one fires the LA): the rest/no-rest margin (margin_h), the
    last `final_h` hours of driving, the learned gate at its budget threshold
    (`gate`), or a random call with probability `rand_p` (the control at an
    equal call budget, seeded by route and stop).  Gate and random fire only
    where the student has at least two candidates, as the gate was trained."""
    pre = Precomp(fd)
    veh = BEHDV(fd)
    N = int(fd["N"])
    T0 = float(fd.get("T_START", 8.0))
    cum = np.r_[0.0, np.cumsum([float(fd["D"].get(i, 0.0)) for i in range(N)])]
    curves, tbar = fd.get("TbarK"), fd["Tbar"]
    calls, la_s, fallback, margins, keys, t_arr = [], 0.0, 0, [], [], []
    vetoed, check_s = [], 0.0
    while veh.stop < N:
        s = veh.stop
        if curves:
            fd["Tbar"] = charger_curve(fd, s if s in pre.K else int(pre.next_cs[s]))
        # policy_core.candidates()[0], unrolled to keep the cost/feas vectors
        legal, rows, cost, feas, flags = pol._score(fd, pre, s, veh, cv)
        lkeys = list(pol._legal_keys)
        score = cost + 1e6 * (feas < pol.feas_thr)
        order = np.argsort(score, kind="stable")
        j = int(order[0])
        act, key = legal[j], lkeys[j]
        tc = pol._charge_hours(fd, s, veh, act, rows[j:j + 1], flags)
        cand = [(legal[i], None, lkeys[i], float(score[i])) for i in order]
        m = rest_margin(cand)
        left = cum[N] - cum[s]
        n_ok = int((score < 1e5).sum())
        g = (gate.score(fd, pre, s, veh, cv, cost, feas, lkeys, j)
             if gate is not None and n_ok >= 2 else None)
        u = (zlib.crc32(f"{name}|{tag}|{s}|rand".encode()) / 2**32
             if rand_p is not None and n_ok >= 2 else None)
        margins.append((m, left, len(cand), g, u))
        fire = ((margin_h is not None and m is not None and m <= margin_h)
                or (final_h is not None and left <= final_h and len(cand) > 1)
                or (g is not None and gate.thr is not None and g >= gate.thr)
                or (u is not None and u < rand_p))
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
                la_act, la_plan = got
                la_key = action_key(la_act.get("y", 0), la_act.get("break_type"),
                                    la_act.get("rest_type"))
                accept = True
                if check is not None and la_key != key:
                    # does the LA's action still win once the STUDENT drives on?
                    from gate_rollout import call_value
                    k_fut, z = check
                    sol0 = (la_plan or {}).get("sol") or [{}]
                    t1 = time.time()
                    mean, se, _fl, _fs = call_value(
                        fd, pre, veh, cv, pol, la_act, float(sol0[0].get("tauc", 0.0) or 0.0),
                        act, tc, k_fut, zlib.crc32(f"{name}|{tag}|{s}|check".encode()))
                    check_s += time.time() - t1
                    accept = mean - z * se > 0
                    if not accept:
                        vetoed.append(s)
                if accept:
                    act, plan, key = la_act, la_plan, la_key
        keys.append(key)
        t_arr.append(veh.t_arr)
        sol = plan if plan is not None else dict(
            feasible=True, sol=[dict(i=0, **durations(fd, s, act, tc, pol.pay_mseq))])
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
                call_stops=calls, fallback=fallback, vetoed=vetoed,
                check_s=round(check_s, 1)), margins


def base_routes(split):
    """Base-case routes with a stored LA run (the LA dataset's), by seed."""
    seeds = range(20, 22) if split == "val" else range(22, 26)
    d = np.load(os.path.join(me.ML, "data", "dataset.npz"), allow_pickle=True)
    names = sorted(str(x) for x in d["instances"])
    return [("uniform", n, os.path.join(me._ROOT, "instances", n + ".json")) for n in names
            if int(n.rsplit("_", 1)[1]) in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="mixed", choices=["mixed", "base"],
                    help="mixed: the mixed-charger routes (--variants); base: base-case "
                         "routes with an LA run, seeds 20-21 (--set val) or 22-25 (test)")
    ap.add_argument("--variants", default="pmix,mix")
    ap.add_argument("--set", default="test", choices=["test", "val"])
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="student tags, comma-separated; 'kind:tag' or a bare torch tag")
    ap.add_argument("--margin", type=float, default=None, help="minutes")
    ap.add_argument("--final", type=float, default=None, help="hours of driving left")
    ap.add_argument("--gate", default=None, choices=["gbt", "torch"],
                    help="learned gate of this arm (gate_train.py); the student must be its arm's")
    ap.add_argument("--budget", type=float, default=None,
                    help="gate budget in %% of decisions (a threshold stored by gate_train.py)")
    ap.add_argument("--gate-model", default=None,
                    help="gate file name (default gate_<arm>; gate_<arm>_ro = rollout labels)")
    ap.add_argument("--check", type=int, default=None,
                    help="K futures: keep the LA's action only if it beats the student's own "
                         "with the student driving on (gate_rollout.call_value); base routes")
    ap.add_argument("--check-z", type=float, default=1.0)
    ap.add_argument("--random", type=float, default=None,
                    help="control: call the LA at random on this %% of decisions")
    ap.add_argument("--mseq", action="store_true",
                    help="students pay the charge-then-rest overhead, as the LA does")
    ap.add_argument("--nominal-features", action="store_true",
                    help="student inputs at nominal travel times, as in training")
    ap.add_argument("--dry", action="store_true",
                    help="no LA: drive the student, report how often each trigger fires")
    ap.add_argument("--slice", default="0/1")
    ap.add_argument("--workers", type=int, default=LA_CONFIG["n_workers"])
    ap.add_argument("--time-limit", type=int, default=LA_CONFIG["time_limit"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    me.ROUTE_SET = args.set

    margin_h = None if args.margin is None else args.margin / 60.0
    rand_p = None if args.random is None else args.random / 100.0
    cfg = ((f"m{args.margin:g}" if args.margin is not None else "")
           + (f"f{args.final:g}" if args.final is not None else "")
           + (f"g{args.gate}{args.budget:g}" if args.gate and args.budget is not None else "")
           + ("ro" if args.gate_model and args.gate_model.endswith("_ro") else "")
           + (f"r{args.random:g}" if args.random is not None else ""))
    if not cfg and not args.dry:
        ap.error("give --margin, --final, --gate with --budget, or --random (or --dry)")
    if args.check is not None:
        if args.routes != "base":
            ap.error("--check rolls out without per-charger curves: base routes only")
        cfg += f"c{args.check}z{args.check_z:g}"
    if cfg:
        cfg += ("_ms" if args.mseq else "") + ("_N" if args.nominal_features else "")
    tail = "" if args.set == "test" else f"_{args.set}"
    head = "hybrid_base" if args.routes == "base" else "hybrid"
    store = os.path.join(RESULTS, f"{head}{tail}_{cfg or 'dry'}.jsonl")
    done = set()
    if os.path.exists(store) and not args.dry:
        with open(store) as fh:
            done = {(r["variant"], r["instance"], r["method"])
                    for r in map(json.loads, filter(str.strip, fh))}

    routes = (base_routes(args.set) if args.routes == "base"
              else me.instances(args.variants.split(","), uniform=False))
    jobs = [(v, n, p, t) for v, n, p in routes for t in args.models.split(",")]
    i, n = (int(x) for x in args.slice.split("/"))
    jobs = [j for j in jobs[i::n] if (j[0], j[1], j[3]) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[hybrid] {cfg or 'dry'} on {args.routes}/{args.set}: {len(jobs)} runs "
          f"(slice {args.slice}) -> {store}", flush=True)

    gate = None
    if args.gate:
        gate = Gate(args.gate, args.budget, nominal=args.nominal_features, model=args.gate_model)
    check = None if args.check is None else (args.check, args.check_z)
    pols, all_m = {}, []
    for k, (v, name, path, spec) in enumerate(jobs):
        kind, _, tag = spec.rpartition(":")
        kind = kind or "torch"
        if gate is not None and tag != Gate.STUDENT[args.gate]:
            raise SystemExit(f"the {args.gate} gate was trained for {Gate.STUDENT[args.gate]}, "
                             f"not {tag}")
        if spec not in pols:
            pol = load_policy(kind, tag, guard_q=0.99, spread_room=True)
            if args.nominal_features:
                pol.feature_q = None
            pol.pay_mseq = args.mseq
            pols[spec] = pol
        pol = pols[spec]
        fd, D, E, cv = me.load(path)
        fd["title"] = name
        if LA_CONFIG.get("la_energy_quantile"):
            fd["la_energy_quantile"] = float(LA_CONFIG["la_energy_quantile"])
            fd["la_energy_cv"] = cv
        t0 = time.time()
        row, margins = drive(fd, D, E, cv, pol, name, tag, margin_h, args.final,
                             args.dry, args.workers, args.time_limit, gate, rand_p, check)
        row["variant"], row["config"] = v, cfg or "dry"
        all_m += [(v, name, tag, *x) for x in margins]
        if not args.dry:
            with open(store, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        print(f"  {k+1}/{len(jobs)} {v} {name} {tag[-2:]}: "
              f"{'done' if row['completed'] else row['violation']} "
              f"{row['duration_h'] or 0:.1f} h, {row['rests']} rests, "
              f"{row['n_calls']} LA calls ({row['la_s']:.0f} s), "
              f"{len(row['vetoed'])} vetoed ({row['check_s']:.0f} s), "
              f"{time.time()-t0:.0f} s", flush=True)

    if args.dry:
        runs = len(jobs) or 1
        dec = len(all_m) or 1
        print(f"\ntrigger counts along the student's own trajectories "
              f"({dec / runs:.0f} decisions per route): calls per route (% of decisions)")
        for mm in (5, 10, 15, 30, 60, 120):
            c = sum(1 for x in all_m if x[3] is not None and x[3] <= mm / 60)
            print(f"  margin {mm:4d} min: {c / runs:6.2f}  ({c / dec * 100:4.1f} %)")
        for fh_ in (5, 10):
            c = sum(1 for x in all_m if x[4] <= fh_ and x[5] > 1)
            print(f"  final {fh_:4d} h  : {c / runs:6.2f}  ({c / dec * 100:4.1f} %)")
        if gate is not None:
            for b, thr in gate.meta["thresholds"].items():
                c = sum(1 for x in all_m if x[6] is not None and x[6] >= thr)
                print(f"  gate {b:>3s} % budget: {c / runs:6.2f}  ({c / dec * 100:4.1f} %)")
        if rand_p is not None:
            c = sum(1 for x in all_m if x[7] is not None and x[7] < rand_p)
            print(f"  random {args.random:g} %: {c / runs:6.2f}  ({c / dec * 100:4.1f} %)")


if __name__ == "__main__":
    main()
