"""
run_rollout.py — drive routes with the rollout policy, paired with the student it wraps
=========================================================================================
    python ML/code/run_rollout.py --set smoke --guard-q 0.99 --spread-room --jobs 7
    python ML/code/run_rollout.py --routes RlongCfewTmedium_23 --K 4 --top-k 2

--set smoke : the test routes (seeds 22-25) on which the LA or the plain student
              ends with more daily rests than the hindsight oracle, plus
              --controls routes on which all three agree (long routes first)
--set test  : every test route
--routes    : an explicit comma-separated list

On every route the plain student is driven too (0.3 s), so each comparison is
paired on the same realisation; the LA and the oracle come from their stored
runs.  Rests are counted as the vehicle EXECUTED them: from the actions for the
learned policies (they execute what they choose), from the trajectory's dwell
for the LA, whose stored actions are what the look-ahead selected and can
differ from what ran (METHOD.md §3), and from rho1 + rho2 for the oracle.

Output: ML/results/rollout_<tag>_<variant>_<name>.json -- never solutions/,
which the reporting pipeline globs by method name.  Finished routes are
appended to a .partial.jsonl as they land, so an interrupted run resumes.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one LightGBM thread per worker

import argparse                                                     # noqa: E402
import glob                                                         # noqa: E402
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

from src.instance_gen.instance_io import load_instance_json         # noqa: E402

from paper_link import gaps, oracle_facts                           # noqa: E402
from policy_core import load_policy, run_student                    # noqa: E402
from rollout_policy import RolloutPolicy                            # noqa: E402

SOL = os.path.join(_ROOT, "solutions", "basecase")
INST = os.path.join(_ROOT, "instances")
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
TEST_SEEDS = range(22, 26)
REST_DWELL_H = 8.9          # a dwell this long is a daily rest (9 h minimum)


def route_class(inst):
    for c in ("short", "medium", "long"):
        if inst.startswith("R" + c):
            return c
    return "?"


def test_routes():
    """The student's test batch, exactly as evaluate.py defines it."""
    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    insts = sorted({str(x) for x in d["instances"]})
    return [i for i in insts if int(i.rsplit("_", 1)[1]) in TEST_SEEDS]


def load(inst):
    fd, D, E, cv = load_instance_json(os.path.join(INST, inst + ".json"))
    fd["_horizon_h"] = 24.0
    return fd, D, E, cv


def n_rests(actions):
    return sum(1 for k in actions if "_r" in k)


def oracle_rests(inst):
    p = os.path.join(SOL, f"oracle_{inst}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        sol = json.load(fh).get("sol") or []
    return sum(int(round(s.get("rho1", 0) + s.get("rho2", 0))) for s in sol) or None


def la_run(inst):
    fs = sorted(glob.glob(os.path.join(SOL, f"{inst}_LA_MIPTAIL_*.json")))
    if not fs:
        return None
    with open(fs[-1]) as fh:
        s = json.load(fh)
    m = s.get("metrics", {})
    dur = s.get("duration_h")
    if dur is None:
        return dict(completed=False)
    ta = [x["t_arr"] for x in s["sim_trajectory"]]
    td = s["td_list"]
    rests = sum(1 for k in range(min(len(td), len(ta)))
                if td[k] - ta[k] >= REST_DWELL_H)
    tw = int(m.get("tw_n_misses", 0))
    return dict(completed=True, duration=dur, tw=tw, rests=rests,
                gap_pen=gaps(dur, tw, s.get("sim_arrival_h"), oracle_facts(inst))[1])


def summarise(r, orc, T0):
    d = r["duration_h"]
    g = gaps(d, r["tw_misses"], (T0 + d) if d is not None else None, orc)
    return dict(completed=r["route_completed"], duration=d, tw=r["tw_misses"],
                rests=n_rests(r["actions"]), gap_pen=g[1],
                halt=r["halt_reason"], actions=r["actions"])


# -- worker ------------------------------------------------------------------
_POL = None
_ARGS = None


def _init(args):
    global _POL, _ARGS
    _ARGS = args
    _POL = load_policy(args.kind, args.tag, guard_q=args.guard_q,
                       spread_room=args.spread_room)


def _one(inst):
    fd, D, E, cv = load(inst)
    T0 = float(fd.get("T_START", 8.0))
    orc = oracle_facts(inst)
    out = dict(instance=inst, route=route_class(inst),
               oracle_rests=oracle_rests(inst), LA=la_run(inst))
    out["student"] = summarise(run_student(fd, D, E, _POL, cv=cv), orc, T0)
    rp = RolloutPolicy(_POL, n_scen=_ARGS.K, top_k=_ARGS.top_k,
                       margin_h=_ARGS.margin_h, z=_ARGS.z,
                       min_gain_h=_ARGS.min_gain_h, k_max=_ARGS.k_max,
                       seed=_ARGS.seed, route=inst)
    w = time.perf_counter()
    ro = summarise(run_student(fd, D, E, rp, cv=cv), orc, T0)
    ro.update(decisions=rp.n_decisions, rolled=rp.n_rolled,
              changed=rp.n_changed, rollouts=rp.n_rollouts,
              failed=rp.n_failed, wall_s=round(time.perf_counter() - w, 1))
    out["rollout"] = ro
    out["log"] = rp.log
    return out


# -- route selection -----------------------------------------------------------
def smoke_routes(args):
    """Test routes where the LA or the plain student rests more than the
    oracle, then `controls` routes where all three agree, long routes first."""
    _init(args)
    extra, agree = [], []
    for inst in test_routes():
        orr, la = oracle_rests(inst), la_run(inst)
        if orr is None or la is None or not la["completed"]:
            continue
        fd, D, E, cv = load(inst)
        r = run_student(fd, D, E, _POL, cv=cv)
        s_r = n_rests(r["actions"]) if r["route_completed"] else None
        if la["rests"] > orr or (s_r is not None and s_r > orr):
            extra.append(inst)
        elif s_r == orr == la["rests"]:
            agree.append(inst)
    order = {"long": 0, "medium": 1, "short": 2}
    agree.sort(key=lambda i: (order.get(route_class(i), 3), i))
    print(f"[smoke] {len(extra)} routes with an extra rest (LA or student), "
          f"+ {min(args.controls, len(agree))} controls", flush=True)
    return extra + agree[: args.controls]


# -- report ---------------------------------------------------------------------
def report(rows):
    def f(x, nd=1):
        return "-" if x is None else f"{x:.{nd}f}"
    print(f"\n{'route':24s} {'rests o/LA/st/ro':>17s}  {'gap_pen % LA / st / ro':>24s}"
          f"  {'rolled':>6s} {'chg':>4s} {'min':>6s}")
    for r in sorted(rows, key=lambda r: r["instance"]):
        la, s, o = r["LA"] or {}, r["student"], r["rollout"]
        rs = (f"{r['oracle_rests']}/{la.get('rests', '-')}/"
              f"{s['rests'] if s['completed'] else 'x'}/"
              f"{o['rests'] if o['completed'] else 'x'}")
        gp = (f"{f(la.get('gap_pen'))} / {f(s['gap_pen'])} / {f(o['gap_pen'])}")
        print(f"{r['instance']:24s} {rs:>17s}  {gp:>24s}  {o['rolled']:6d} "
              f"{o['changed']:4d} {o['wall_s'] / 60:6.1f}")

    def extra(key):
        ok = [r for r in rows if (r[key] or {}).get("completed")
              and r["oracle_rests"] is not None]
        return sum(1 for r in ok if r[key]["rests"] > r["oracle_rests"]), len(ok)

    print()
    for key in ("LA", "student", "rollout"):
        e, n = extra(key)
        g = [r[key]["gap_pen"] for r in rows
             if (r[key] or {}).get("completed") and r[key].get("gap_pen") is not None]
        fails = sum(1 for r in rows if r[key] and not r[key].get("completed"))
        print(f"{key:8s} extra-rest routes {e}/{n}   infeasible {fails}   "
              f"gap_pen median {st.median(g):.2f} %  mean {st.mean(g):.2f} %"
              if g else f"{key:8s} (no completed runs)")
    paired = [(r["rollout"]["duration"] + 0.5 * r["rollout"]["tw"])
              - (r["student"]["duration"] + 0.5 * r["student"]["tw"])
              for r in rows if r["rollout"]["completed"] and r["student"]["completed"]]
    if paired:
        print(f"rollout - student, paired penalised duration: median "
              f"{st.median(paired):+.2f} h, mean {st.mean(paired):+.2f} h, "
              f"better on {sum(p < -1e-6 for p in paired)}, worse on "
              f"{sum(p > 1e-6 for p in paired)} of {len(paired)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gbt", choices=["gbt", "nn", "clf"])
    ap.add_argument("--tag", default="gbt_F95_base_s1")
    ap.add_argument("--guard-q", type=float, default=None)
    ap.add_argument("--spread-room", action="store_true")
    ap.add_argument("--set", default=None, choices=["smoke", "test"])
    ap.add_argument("--routes", default=None, help="comma-separated instances")
    ap.add_argument("--controls", type=int, default=8)
    ap.add_argument("--K", type=int, default=16, help="scenarios per rollout")
    ap.add_argument("--top-k", type=int, default=3, help="candidates rolled out")
    ap.add_argument("--margin-h", type=float, default=None,
                    help="roll out only when the student's two best scores "
                         "are within this many hours (default: always)")
    ap.add_argument("--z", type=float, default=2.0,
                    help="overrule the student only if paired mean + z*se < "
                         "-min-gain-h (0 = plain argmin, which chased noise)")
    ap.add_argument("--min-gain-h", type=float, default=0.0)
    ap.add_argument("--k-max", type=int, default=None,
                    help="double the scenarios for an unproven challenger "
                         "up to this many (default: no doubling)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.routes:
        routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    elif args.set == "smoke":
        routes = smoke_routes(args)
    elif args.set == "test":
        routes = test_routes()
    else:
        ap.error("give --set or --routes")

    variant = (f"g{round(100 * (args.guard_q or 0.95))}"
               f"{'sr' if args.spread_room else ''}")
    name = args.name or args.set or "routes"
    trig = "all" if args.margin_h is None else f"m{args.margin_h:g}"
    gate = (f"_z{args.z:g}" + (f"g{args.min_gain_h:g}" if args.min_gain_h else "")
            + (f"_kmax{args.k_max}" if args.k_max else ""))
    stem = (f"rollout_{args.tag}_{variant}_{name}_K{args.K}_k{args.top_k}_{trig}"
            f"{gate}_s{args.seed}")
    os.makedirs(RESULTS, exist_ok=True)
    partial = os.path.join(RESULTS, stem + ".partial.jsonl")
    done = {}
    if os.path.exists(partial):
        with open(partial) as fh:
            for line in fh:
                row = json.loads(line)
                done[row["instance"]] = row
    todo = [r for r in routes if r not in done]
    # long routes first: they take longest, so the pool drains evenly
    order = {"long": 0, "medium": 1, "short": 2}
    todo.sort(key=lambda i: (order.get(route_class(i), 3), i))
    print(f"[rollout] {stem}: {len(routes)} routes, {len(done)} already done, "
          f"{len(todo)} to run on {args.jobs} workers", flush=True)

    t0 = time.time()
    if todo:
        with Pool(args.jobs, initializer=_init, initargs=(args,)) as pool, \
                open(partial, "a") as fh:
            for row in pool.imap_unordered(_one, todo):
                done[row["instance"]] = row
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                ro = row["rollout"]
                print(f"  {len(done)}/{len(routes)} {row['instance']}: rests "
                      f"oracle {row['oracle_rests']} / student "
                      f"{row['student']['rests']} / rollout {ro['rests']}, "
                      f"{ro['changed']} of {ro['rolled']} rolled decisions changed, "
                      f"{ro['wall_s'] / 60:.1f} min  [{(time.time() - t0) / 60:.0f} min]",
                      flush=True)

    rows = [done[r] for r in routes if r in done]
    out = os.path.join(RESULTS, stem + ".json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    report(rows)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
