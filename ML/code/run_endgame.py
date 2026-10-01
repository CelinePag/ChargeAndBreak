"""
run_endgame.py — drive routes with the end-game policy, paired with the student it wraps
========================================================================================
    python ML/code/run_endgame.py --set smoke --guard-q 0.99 --spread-room --jobs 4
    python ML/code/run_endgame.py --set la-extra --guard-q 0.99 --spread-room
    python ML/code/run_endgame.py --routes RlongCfewTmedium_23 --zone-h 28

--set smoke    : as in run_rollout.py -- the test routes on which the LA or the
                 plain student rests more than the oracle, plus --controls
--set la-extra : every base-case route on which the LA rests more than the
                 oracle (seeds 1-25 -- most of them are TRAINING routes of the
                 student; the effect measured is paired, student vs student +
                 end game, but say so when quoting it)
--set test     : every test route (seeds 22-25)

Route selection, rest counting and the gap definition are run_rollout.py's.
Output: ML/results/endgame_<tag>_<variant>_<name>_z<zone>_t<tail>_L<limit>.json,
with a .partial.jsonl that lets an interrupted run resume.
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

from endgame_policy import EndgamePolicy                            # noqa: E402
from paper_link import oracle_facts                                 # noqa: E402
from policy_core import load_policy, run_student                    # noqa: E402
from run_rollout import (DATA, RESULTS, la_run, load, oracle_rests, # noqa: E402
                         route_class, smoke_routes, summarise,
                         test_routes)

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
    ep = EndgamePolicy(_POL, zone_h=_ARGS.zone_h, tail=_ARGS.tail,
                       time_limit=_ARGS.time_limit, retry=_ARGS.retry,
                       route=inst)
    w = time.perf_counter()
    eg = summarise(run_student(fd, D, E, ep, cv=cv), orc, T0)
    eg.update(decisions=ep.n_decisions, solves=ep.n_solves,
              retries=ep.n_retries, fallback=ep.n_fallback,
              solve_s=round(ep.seconds, 1),
              zone_stop=ep.zone_stop, wall_s=round(time.perf_counter() - w, 1))
    out["endgame"] = eg
    out["log"] = ep.log
    return out


def la_extra_routes():
    """Every base-case route (all seeds) on which the LA rests more than the oracle."""
    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    out = []
    for inst in sorted({str(x) for x in d["instances"]}):
        orr, la = oracle_rests(inst), la_run(inst)
        if orr is not None and la and la.get("completed") and la["rests"] > orr:
            out.append(inst)
    print(f"[la-extra] {len(out)} routes on which the LA rests more than the oracle",
          flush=True)
    return out


def report(rows):
    def f(x):
        return "-" if x is None else f"{x:.1f}"
    print(f"\n{'route':24s} {'rests o/LA/st/eg':>17s}  {'gap_pen % LA / st / eg':>24s}"
          f"  {'solves':>6s} {'fb':>3s} {'min':>6s}")
    for r in sorted(rows, key=lambda r: r["instance"]):
        la, s, e = r["LA"] or {}, r["student"], r["endgame"]
        rs = (f"{r['oracle_rests']}/{la.get('rests', '-')}/"
              f"{s['rests'] if s['completed'] else 'x'}/"
              f"{e['rests'] if e['completed'] else 'x'}")
        print(f"{r['instance']:24s} {rs:>17s}  "
              f"{f(la.get('gap_pen')) + ' / ' + f(s['gap_pen']) + ' / ' + f(e['gap_pen']):>24s}"
              f"  {e['solves']:6d} {e['fallback']:3d} {e['wall_s'] / 60:6.1f}")
    print()
    for key in ("LA", "student", "endgame"):
        ok = [r for r in rows if (r[key] or {}).get("completed")
              and r["oracle_rests"] is not None]
        extra = sum(1 for r in ok if r[key]["rests"] > r["oracle_rests"])
        fails = sum(1 for r in rows if r[key] and not r[key].get("completed"))
        g = [r[key]["gap_pen"] for r in ok if r[key].get("gap_pen") is not None]
        if g:
            print(f"{key:8s} extra-rest routes {extra}/{len(ok)}   infeasible {fails}   "
                  f"gap_pen median {st.median(g):.2f} %  mean {st.mean(g):.2f} %  "
                  f"TW misses {sum(r[key]['tw'] for r in ok)}")
    paired = [(r["endgame"]["duration"] + 0.5 * r["endgame"]["tw"])
              - (r["student"]["duration"] + 0.5 * r["student"]["tw"])
              for r in rows if r["endgame"]["completed"] and r["student"]["completed"]]
    if paired:
        print(f"endgame - student, paired penalised duration: median "
              f"{st.median(paired):+.2f} h, mean {st.mean(paired):+.2f} h, "
              f"better on {sum(p < -1e-6 for p in paired)}, worse on "
              f"{sum(p > 1e-6 for p in paired)} of {len(paired)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gbt", choices=["gbt", "nn", "clf"])
    ap.add_argument("--tag", default="gbt_F95_base_s1")
    ap.add_argument("--guard-q", type=float, default=None)
    ap.add_argument("--spread-room", action="store_true")
    ap.add_argument("--set", default=None, choices=["smoke", "test", "la-extra"])
    ap.add_argument("--routes", default=None, help="comma-separated instances")
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--zone-h", type=float, default=28.0,
                    help="hand over to the MILP once this much nominal "
                         "driving is left")
    ap.add_argument("--tail", type=float, default=1.0,
                    help="factor on the legs after the next one")
    ap.add_argument("--time-limit", type=int, default=30,
                    help="seconds per MILP solve (an incumbent at the limit is used)")
    ap.add_argument("--retry", type=int, default=4,
                    help="x time for an unproven solve that plans more rests "
                         "than the last accepted plan")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.routes:
        routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    elif args.set == "smoke":
        routes = smoke_routes(args)
    elif args.set == "la-extra":
        routes = la_extra_routes()
    elif args.set == "test":
        routes = test_routes()
    else:
        ap.error("give --set or --routes")

    variant = (f"g{round(100 * (args.guard_q or 0.95))}"
               f"{'sr' if args.spread_room else ''}")
    name = args.name or args.set or "routes"
    stem = (f"endgame_{args.tag}_{variant}_{name}_z{args.zone_h:g}"
            f"_t{args.tail:g}_L{args.time_limit}_r{args.retry}")
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
    print(f"[endgame] {stem}: {len(routes)} routes, {len(done)} already done, "
          f"{len(todo)} to run on {args.jobs} workers", flush=True)

    t0 = time.time()
    if todo:
        with Pool(args.jobs, initializer=_init, initargs=(args,)) as pool, \
                open(partial, "a") as fh:
            for row in pool.imap_unordered(_one, todo):
                done[row["instance"]] = row
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                e = row["endgame"]
                print(f"  {len(done)}/{len(routes)} {row['instance']}: rests oracle "
                      f"{row['oracle_rests']} / student {row['student']['rests']} / "
                      f"endgame {e['rests'] if e['completed'] else 'x'}, "
                      f"{e['solves']} solves ({e['fallback']} fallback), "
                      f"{e['wall_s'] / 60:.1f} min  [{(time.time() - t0) / 60:.0f} min]",
                      flush=True)

    rows = [done[r] for r in routes if r in done]
    out = os.path.join(RESULTS, stem + ".json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    report(rows)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
