"""
fpi_collect.py — label the student's own decisions by simulating where they lead
================================================================================
    python ML/code/fpi_collect.py --guard-q 0.99 --spread-room --name r1
    python ML/code/fpi_collect.py --guard-q 0.99 --spread-room --routes RlongCfewTmedium_23 --name pilot

For every training route (seeds 1-21: 1-19 to fit G, 20-21 to stop its
boosting early -- the split of dataset.py) and each of --rollins draws of its
travel times, the student drives the route.  At a random --p-stop share of the
stops where it has at least two candidate actions, each of its --top-k best
actions is tried and the student drives every branch to the destination under
one shared draw of the remaining legs (fpi_policy.py explains why).  Every
draw comes from the instance's own travel-time distribution with a seed built
from (route, roll-in, stop), so a run is reproducible and order-free; the
realised travel times of the instance are never used, so the routes the
student is later evaluated on stay unseen in the same sense as before.

One file per route: ML/data/fpi_<name>/<route>.npz, one row per branch:
  X      G's input (the student's row + fpi_policy.EXTRA)
  old    the student's cost-head prediction for that action (regret, h)
  pi     True for the student's own choice (the reference branch)
  key    action key          grp   decision id within the file
  stop   stop index          rollin, t   roll-in and time at the stop
  C      the branch's route cost: arrival + BETA per window miss, or FAIL_H
  rests  daily rests the branch took over the whole route
plus the roll-ins' own outcomes.  Finished routes are skipped on a rerun.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one LightGBM thread per worker

import argparse                                                     # noqa: E402
import json                                                         # noqa: E402
import sys                                                          # noqa: E402
import time                                                         # noqa: E402
from multiprocessing import Pool                                    # noqa: E402

import numpy as np                                                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.simulation.BEHDV import BEHDV                              # noqa: E402
from src.simulation.scenarios import generate_scenarios            # noqa: E402

from features import Precomp, action_key                            # noqa: E402
from fpi_policy import EXTRA, crc, drive_fleet, g_matrix, ranked    # noqa: E402
from policy_core import load_policy                                 # noqa: E402
from rollout_policy import clone, step                              # noqa: E402
from run_rollout import DATA, load, route_class                     # noqa: E402

_BASE = None
_A = None


def _init(args):
    global _BASE, _A
    _A = args
    _BASE = load_policy("gbt", args.tag, guard_q=args.guard_q,
                        spread_room=args.spread_room)


def _rests(veh):
    return sum(1 for d in veh.durations if float(d.get("taur", 0) or 0) > 0)


def collect(inst):
    """All branch rows of one route (see the module docstring)."""
    fd, _d_real, _e_real, cv = load(inst)       # only the distribution is used
    pre = Precomp(fd)
    n_stops = int(fd["N"])
    b, a = _BASE, _A
    pick = np.random.default_rng(crc(f"{inst}|pick|{a.seed}"))
    rec = {k: [] for k in ("X", "old", "pi", "key", "grp", "stop", "rollin", "t")}
    cost_out, rests_out, rollins = [], [], []
    pend_v, pend_s = [], []

    def flush():
        if pend_v:
            cost_out.extend(drive_fleet(b, fd, pre, cv, pend_v, pend_s).tolist())
            rests_out.extend(_rests(v) for v in pend_v)
            pend_v.clear()
            pend_s.clear()

    g = 0
    for r in range(a.rollins):
        w = generate_scenarios(fd, 0, n_stops, 1, cv,
                               seed=crc(f"{inst}|in|{r}|{a.seed}"))[0]
        veh = BEHDV(fd)
        while not veh.is_halted and veh.stop < n_stops:
            s = veh.stop
            legal, rows, cost, flags, j0, elig = ranked(b, fd, pre, s, veh, cv, a.top_k)
            tc0 = b._charge_hours(fd, s, veh, legal[j0], rows[j0:j0 + 1], flags)
            if len(elig) >= 2 and pick.random() < a.p_stop:
                sc = generate_scenarios(fd, s, n_stops, 1, cv,
                                        seed=crc(f"{inst}|br|{r}|{s}|{a.seed}"))[0]
                gm = g_matrix(fd, pre, s, veh, rows, legal, elig)
                for q, j in enumerate(elig):
                    tc = tc0 if j == j0 else b._charge_hours(
                        fd, s, veh, legal[j], rows[j:j + 1], flags)
                    v = clone(veh)
                    step(v, fd, legal[j], tc, sc["D"][s], sc["E"][s])
                    pend_v.append(v)
                    pend_s.append(sc)
                    act = legal[j]
                    rec["X"].append(gm[q])
                    rec["old"].append(float(cost[j]))
                    rec["pi"].append(j == j0)
                    rec["key"].append(action_key(act.get("y", 0), act.get("break_type"),
                                                 act.get("rest_type")))
                    rec["grp"].append(g)
                    rec["stop"].append(s)
                    rec["rollin"].append(r)
                    rec["t"].append(float(veh.t_arr))
                g += 1
                if len(pend_v) >= a.fleet:
                    flush()
            step(veh, fd, legal[j0], tc0, w["D"][s], w["E"][s])
        flush()
        rollins.append(dict(completed=not veh.is_halted, arrival=float(veh.t_arr),
                            rests=_rests(veh), tw=len(veh.tw_misses)))
    return rec, cost_out, rests_out, rollins


def _one(inst):
    t0 = time.perf_counter()
    rec, cost, rests, rollins = collect(inst)
    names = list(_BASE.state_names) + list(_BASE.action_names) + list(EXTRA)
    n = len(rec["X"])
    out = os.path.join(_A.out_dir, inst + ".npz")
    np.savez_compressed(
        out + ".tmp.npz",
        X=np.array(rec["X"], dtype=np.float32).reshape(n, len(names)),
        old=np.array(rec["old"], dtype=np.float32),
        pi=np.array(rec["pi"], dtype=bool),
        key=np.array(rec["key"]),
        grp=np.array(rec["grp"], dtype=np.int32),
        stop=np.array(rec["stop"], dtype=np.int16),
        rollin=np.array(rec["rollin"], dtype=np.int8),
        t=np.array(rec["t"], dtype=np.float32),
        C=np.array(cost, dtype=np.float64),
        rests=np.array(rests, dtype=np.int16),
        names=np.array(names),
        instance=inst, seed=int(inst.rsplit("_", 1)[1]), cls=route_class(inst),
        rollins=json.dumps(rollins), wall_s=time.perf_counter() - t0)
    os.replace(out + ".tmp.npz", out)             # a crash never leaves half a file
    return dict(instance=inst, decisions=int(len(set(rec["grp"]))), rows=n,
                wall=time.perf_counter() - t0,
                fails=int(sum(c >= 999.0 for c in cost)))


def training_routes(seeds, classes):
    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    insts = sorted({str(x) for x in d["instances"]})
    return [i for i in insts
            if int(i.rsplit("_", 1)[1]) in seeds and route_class(i) in classes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gbt_F95_base_s1")
    ap.add_argument("--guard-q", type=float, default=None)
    ap.add_argument("--spread-room", action="store_true")
    ap.add_argument("--seeds", default="1-21", help="instance seeds, e.g. 1-21")
    ap.add_argument("--classes", default="long,medium,short")
    ap.add_argument("--routes", default=None, help="comma-separated instances")
    ap.add_argument("--rollins", type=int, default=2)
    ap.add_argument("--p-stop", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--fleet", type=int, default=300,
                    help="branches driven together (memory vs batching)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    if args.routes:
        routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    else:
        lo, hi = (int(x) for x in args.seeds.split("-"))
        routes = training_routes(set(range(lo, hi + 1)), set(args.classes.split(",")))
    args.out_dir = os.path.join(DATA, f"fpi_{args.name}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "_args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=1)
    todo = [r for r in routes
            if not os.path.exists(os.path.join(args.out_dir, r + ".npz"))]
    order = {"long": 0, "medium": 1, "short": 2}
    todo.sort(key=lambda i: (order.get(route_class(i), 3), i))
    print(f"[fpi-collect] {args.out_dir}: {len(routes)} routes, "
          f"{len(routes) - len(todo)} done, {len(todo)} to run on {args.jobs} workers",
          flush=True)
    t0 = time.time()
    n_rows = 0
    with Pool(args.jobs, initializer=_init, initargs=(args,)) as pool:
        for k, res in enumerate(pool.imap_unordered(_one, todo), 1):
            n_rows += res["rows"]
            print(f"  {k}/{len(todo)} {res['instance']}: {res['decisions']} decisions, "
                  f"{res['rows']} rows, {res['fails']} failed branches, "
                  f"{res['wall'] / 60:.1f} min  [{(time.time() - t0) / 60:.0f} min, "
                  f"{n_rows} rows]", flush=True)
    print(f"[fpi-collect] done in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
