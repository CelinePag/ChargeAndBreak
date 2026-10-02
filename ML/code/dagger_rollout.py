"""
dagger_rollout.py — DAgger step A: the students drive, we keep where they went
=============================================================================
Behaviour cloning only ever shows a student the states the TEACHER created.
Driving on its own, one small difference puts the truck somewhere the teacher
never was, and nobody has shown the student what to do there.  DAgger asks the
teacher about the states the student actually visits.  Our teacher is a solver,
so -- unlike a human expert -- it can be asked about ANY state.

This step only drives.  Each listed model (shield as deployed: guard 0.99 and
the spread-room check by default) drives the same training routes through
policy_core.run_student, exactly as evaluate.py does.  At every stop the
vehicle is snapshotted with BEHDV.to_checkpoint() BEFORE the student decides;
afterwards `--stops` of the eligible stops are kept per route:

  eligible = stop > 0, at least two legal actions (a forced stop teaches
             nothing), and the student's state already differs from the
             teacher's own run on that route (before that, the state IS a
             teacher state and its label is already in the dataset)

Sampling among eligible stops is uniform on purpose: labelling only where the
student went wrong would teach it that rare states are common.

`--check-stops k` also keeps up to k stops from BEFORE the divergence, flagged
`check`.  Their state equals the teacher's, so the label step can compare its
fresh answer with the teacher's logged one -- the check that the queried LA is
the same teacher (dagger_report.py --check).  Check rows never enter training.

Every model drives the SAME routes (one shared pool): each trainer then gets
the same extra data, so the comparison between arms stays controlled.

The state features are recorded too, computed the way extract.py computes
them (the route's own Tbar, guard None).  dagger_label.py recomputes them from
the restored checkpoint and refuses a query whose features differ -- the proof
that the restored state is the visited one.

    python ML/code/dagger_rollout.py --label probe --routes base --split stop \\
        --models torch:tmlp_F95_split_list_s0,gbt:gbt_F95_base_s0 --stops 3
    python ML/code/dagger_rollout.py --label r1 --routes pmix --split fit \\
        --models torch:tmlp_F95_split_list_s0,gbt:gbt_F95_base_s0 --stops 10
"""
from __future__ import annotations

import argparse
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

from dagger_io import (ROUTE_SETS, family_seed, jsonable, label_dir,  # noqa: E402
                       query_name, routes, stable_seed, teacher_logged_stops,
                       teacher_states)
from dataset import FIT_SEEDS, STOP_SEEDS                         # noqa: E402
from extract import GUARD_Q                                       # noqa: E402
from features import Precomp, state_features                      # noqa: E402
from policy_core import load_policy, run_student                  # noqa: E402

SPLITS = {"fit": FIT_SEEDS, "stop": STOP_SEEDS}   # never test
# "same state as the teacher": the stored trajectory is ROUNDED (t to 4 dp, e
# to 2 dp -- see extract.G1_TOL), so the tolerance is the print precision
T_TOL, E_TOL = 1e-4, 6e-3                         # h, kWh


class Recorder:
    """Wraps a student policy: snapshots the state, then lets it decide."""

    def __init__(self, inner, fd):
        self.inner = inner
        self.pre = Precomp(fd)
        self.tbar_route = fd["Tbar"]       # run_student swaps it per charger
        self.steps = []

    def decide(self, fd, pre, stop, state, cv):
        ck = state.to_checkpoint()
        cur, fd["Tbar"] = fd["Tbar"], self.tbar_route
        sf, _ = state_features(fd, self.pre, stop, state, cv, GUARD_Q)
        fd["Tbar"] = cur
        out = self.inner.decide(fd, pre, stop, state, cv)
        self.steps.append(dict(stop=int(stop), ck=ck, key=out[2], tauc=float(out[1]),
                               legal=list(self.inner._legal_keys),
                               t_arr=float(state.t_arr), e_arr=float(state.e_arr),
                               state_feats={k: float(v) for k, v in sf.items()}))
        return out


def divergence(steps, teacher):
    """First stop whose state differs from the teacher's (1 without a teacher)."""
    if teacher is None:
        return 1
    for s in steps:
        t = teacher.get(s["stop"])
        if (t is None or abs(s["t_arr"] - t[0]) > T_TOL
                or abs(s["e_arr"] - t[1]) > E_TOL):
            return max(1, s["stop"])
    return 10 ** 9                        # never left the teacher's trajectory


def pick_routes(route_set, split, per_family, label, seed, limit):
    fams = routes(route_set, SPLITS[split])
    rng = np.random.default_rng(stable_seed("routes", label, route_set, split, seed))
    out = []
    for fam in sorted(fams):
        cand = fams[fam]
        for j in rng.choice(len(cand), size=min(per_family, len(cand)), replace=False):
            out.append(cand[int(j)])
    return out[:limit] if limit else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="r1, r2, ... or probe")
    ap.add_argument("--routes", default="base", choices=list(ROUTE_SETS))
    ap.add_argument("--split", default="fit", choices=list(SPLITS),
                    help="fit = training data; stop = a probe (measuring only)")
    ap.add_argument("--models", required=True,
                    help="comma list of kind:tag, e.g. torch:tmlp_F95_split_list_s0")
    ap.add_argument("--per-family", type=int, default=1)
    ap.add_argument("--stops", type=int, default=10, help="queries per route and model")
    ap.add_argument("--check-stops", type=int, default=0,
                    help="pre-divergence stops to keep as teacher checks")
    ap.add_argument("--guard-q", type=float, default=0.99)
    ap.add_argument("--no-spread-room", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="routes (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = label_dir(args.label, "queries")
    os.makedirs(out_dir, exist_ok=True)
    picked = pick_routes(args.routes, args.split, args.per_family, args.label,
                         args.seed, args.limit)
    models = [m.split(":", 1) for m in args.models.split(",") if m]
    print(f"[rollout] label={args.label} routes={args.routes}/{args.split}: "
          f"{len(picked)} routes x {len(models)} models, {args.stops} queries each")

    totals = dict(files=0, queries=0, checks=0, eligible=0, skipped=0)
    for kind, tag in models:
        pol = load_policy(kind, tag, guard_q=args.guard_q,
                          spread_room=not args.no_spread_room)
        t0 = time.time()
        for inst, path in picked:
            out = os.path.join(out_dir, query_name(tag, inst) + ".json")
            if os.path.exists(out) and not args.overwrite:
                totals["skipped"] += 1
                continue
            fd, D_real, E_real, cv = load_instance_json(path)
            fd["_horizon_h"] = 24.0
            rec = Recorder(pol, fd)
            res = run_student(fd, D_real, E_real, rec, cv=cv)

            div = divergence(rec.steps, teacher_states(args.routes, inst))
            usable = [s for s in rec.steps if s["stop"] > 0 and len(s["legal"]) >= 2]
            elig = [s for s in usable if s["stop"] >= div]
            logged = teacher_logged_stops(args.routes, inst) if args.check_stops else set()
            pre_div = [s for s in usable if s["stop"] < div and s["stop"] in logged]
            rng = np.random.default_rng(stable_seed("stops", args.label, tag, inst, args.seed))
            take = sorted(rng.choice(len(elig), size=min(args.stops, len(elig)),
                                     replace=False).tolist()) if elig else []
            chk = sorted(rng.choice(len(pre_div), size=min(args.check_stops, len(pre_div)),
                                    replace=False).tolist()) if pre_div else []
            queries = ([dict(elig[j], check=False) for j in take]
                       + [dict(pre_div[j], check=True) for j in chk])
            fam, seed = family_seed(inst)
            rec_out = dict(
                label=args.label, route_set=args.routes, split=args.split,
                physics=ROUTE_SETS[args.routes]["physics"],
                model=dict(kind=kind, tag=tag, guard_q=args.guard_q,
                           spread_room=not args.no_spread_room),
                instance=inst, path=os.path.relpath(path, _ROOT), family=fam,
                seed=seed, cv=float(cv),
                result=dict(completed=bool(res["route_completed"]),
                            duration_h=res["duration_h"],
                            halt_reason=res["halt_reason"],
                            n_violations=res["n_violations"],
                            tw_misses=res["tw_misses"]),
                n_steps=len(rec.steps), diverged_at=int(min(div, 10 ** 6)),
                n_eligible=len(elig), queries=queries)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(rec_out, fh, default=jsonable)
            totals["files"] += 1
            totals["queries"] += len(take)
            totals["checks"] += len(chk)
            totals["eligible"] += len(elig)
        print(f"  {tag}: {time.time() - t0:.0f}s", flush=True)

    print(f"[rollout] wrote {totals['files']} files ({totals['skipped']} already there): "
          f"{totals['queries']} queries + {totals['checks']} checks "
          f"from {totals['eligible']} eligible stops -> {out_dir}")
    print("next: dagger_label.py --label", args.label, "(anaconda python: needs gurobipy)")


if __name__ == "__main__":
    main()
