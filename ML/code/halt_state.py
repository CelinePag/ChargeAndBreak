"""
halt_state.py — where, in the week's rule budget, does an infeasible run fail?
==============================================================================
An evaluation row records only THAT a run became infeasible and which rule
ended it.  This replays each infeasible run (policies and realisations are
deterministic, so the replay is exact -- checked against the stored halting
stop) and reads the simulator's state at the decision that broke the rule.

The question it answers: are the failures on long routes concentrated in the
regimes that short and medium routes almost never reach?  Two weekly budgets
change the rules mid-route, and only long routes routinely exhaust them:

    ext_shift_used   extended driving days used (EU 561/2006 Art. 6(1): at most
                     2 a week).  While one is left the daily driving limit is
                     10 h; once both are spent it is 9 h.
    rho2_used        reduced daily rests used (at most 3 between weekly rests).
                     A regular rest must begin within 13 h of the shift start,
                     a reduced one within 15 h, so once the three are spent the
                     spread limit tightens from 15 h to 13 h.

    python ML/code/halt_state.py --kind gbt --tag gbt_F95_base_SM_s0 \
        --eval eval_gbt_F95_base_SM_s0_g95_longall.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.instance_gen.instance_io import load_instance_json      # noqa: E402
from src.simulation.BEHDV import BEHDV                           # noqa: E402
from src.simulation.supervisor import _action_min_dwell          # noqa: E402

from features import Precomp, state_features                     # noqa: E402
from policy_core import durations, load_policy                   # noqa: E402

INST = os.path.join(_ROOT, "instances")
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def replay(fd, D_real, E_real, pol, cv):
    """run_student, keeping the vehicle so its histories can be read.

    Also returns the time account of the LAST decision.  The legality check
    (supervisor._spread_with_dwell_fails) admits a non-rest action when
    h + o(a) + D_wc <= 15 h, where o(a) counts service, queue and the minimum
    break -- but NOT the charge the policy then chooses, nor the stop overhead
    M_stop that any activity at a charger costs.  The simulator spends
    dwell = departure - arrival, which includes both, so the spread after the
    leg is h + dwell + D_real.
    """
    pre = Precomp(fd)
    veh = BEHDV(fd)
    N = int(fd["N"])
    keys, last = [], {}
    while veh.stop < N:
        stop = veh.stop
        _sf, flags = state_features(fd, pre, stop, veh, cv, pol.guard_q)
        action, tauc, key = pol.decide(fd, pre, stop, veh, cv)
        keys.append(key)
        o_a = _action_min_dwell(fd, stop, action)
        t_arr, sd0 = float(veh.t_arr), float(veh.sd)
        mock = dict(feasible=True,
                    sol=[dict(i=0, **durations(fd, stop, action, tauc))])
        veh.advance(action=action, D_next=float(D_real[stop]),
                    E_next=float(E_real[stop]), milp_sol=mock)
        last = dict(h=flags["h_state"], o_a=o_a, D_wc=flags["D_next_wc"],
                    D_real=float(D_real[stop]), tauc=tauc,
                    dwell=float(veh.td_list[-1]) - t_arr,
                    # fixed stop overhead: M_stop at a charger, M_lay at a
                    # layby, M_seq for a charge sequenced before a rest
                    overhead=sum(float(veh.durations[-1].get(k, 0.0))
                                 for k in ("mstop", "mlay", "mseq")),
                    Tspr2=flags["Tspr2"], sd=sd0,
                    rest=action.get("rest_type") in ("r1", "r2"),
                    ferry=stop in {int(k) for k in (fd.get("ferry") or {})})
        if veh.is_halted:
            break
    return veh, keys, last


def cause(rule, a):
    """Which part of the account broke the rule.

    uncounted-dwell  the check passed, but with the dwell actually spent
                     (charge + stop overhead) the guarded drive no longer fits
    drive-tail       the plan fitted at the guarded drive time; the realised
                     drive was longer (the guard is a quantile, not a bound) --
                     on the spread, the shift-driving or the 4.5 h
                     consecutive-driving limit
    ferry            the breaking decision is a sea crossing: one legal action
                     (the forced crossing), so the mistake was made earlier
    energy           the battery fell below its floor on the leg (stranding)
    """
    if a.get("ferry"):
        return "ferry"
    if rule == "hos_spread" and not a["rest"]:
        if a["h"] + a["dwell"] + a["D_wc"] > a["Tspr2"] + 1e-9:
            return "uncounted-dwell"
        return "drive-tail"
    if rule in ("hos_sd", "hos_cd"):
        return "drive-tail" if a["D_real"] > a["D_wc"] + 1e-9 else "other"
    if rule == "stranding":
        return "energy"
    return "other"


def runs_from_eval(eval_file):
    """(all rows, [(instance, path, stored halt stop, stored rule)]) of an
    evaluate.py result."""
    with open(os.path.join(RESULTS, eval_file)) as fh:
        rows = json.load(fh)
    return rows, [(r["instance"], os.path.join(INST, r["instance"] + ".json"),
                   r.get("halted_at"), r.get("halt_reason"))
                  for r in rows if not r.get("route_completed")]


def runs_from_ood(axis, tag, guard_q=0.95, spread_room=False):
    """The same, for one model on one axis of ood_eval.py (which stores the
    rule but not the stop)."""
    from ood_eval import AXES, variant_tail
    store = f"ood_test{variant_tail(guard_q, spread_room)}.json"
    with open(os.path.join(RESULTS, store)) as fh:
        rows = [r for r in json.load(fh) if r["axis"] == axis and r["tag"] == tag]
    d = os.path.join(_ROOT, AXES[axis][0])
    return rows, [(r["instance"], os.path.join(d, r["instance"] + ".json"),
                   None, r.get("violation"))
                  for r in rows if not r.get("completed")]


def diagnose(kind, tag, runs, guard_q=0.95, spread_room=False):
    pol = load_policy(kind, tag, guard_q=guard_q, spread_room=spread_room)
    out = []
    for inst, path, stored_stop, stored_rule in runs:
        fd, D_real, E_real, cv = load_instance_json(path)
        fd["_horizon_h"] = 24.0
        veh, keys, last = replay(fd, D_real, E_real, pol, cv)
        v0 = veh.violations[0] if veh.violations else {}
        # the state the policy SAW at the breaking decision is the one before
        # the last append (the halting stop is appended, then the run ends)
        k = -2 if len(veh.ext_shift_used_history) >= 2 else -1
        rests = sum(1 for a in keys[:-1] if "_r" in a)
        out.append(dict(
            instance=inst, halted_at=veh.halted_at, stored_halted_at=stored_stop,
            reproduced=(veh.halt_reason == stored_rule
                        and stored_stop in (None, veh.halted_at)),
            rule=v0.get("type"), detail=v0.get("detail"),
            action=keys[-1] if keys else None,
            shift=rests + 1, n_stops=int(fd["N"]),
            route_frac=round(veh.halted_at / int(fd["N"]), 3),
            ext_used=int(veh.ext_shift_used_history[k]),
            rho2_used=int(veh.rho2_used_history[k]),
            sd=round(float(veh.sd_history[k]), 3),
            h=round(float(veh.h_history[k]), 3),
            # the time account of the breaking decision (see replay)
            charge_h=round(last["tauc"], 3), overhead_h=round(last["overhead"], 3),
            uncounted_h=round(last["dwell"] - last["o_a"], 3),
            room_h=round(last["Tspr2"] - last["h"] - last["o_a"] - last["D_wc"], 3),
            drive_wc=round(last["D_wc"], 3), drive_real=round(last["D_real"], 3),
            cause=cause(v0.get("type"), last)))
    return out


def summary(rows, diag, label=""):
    n = len(rows)
    print(f"\n{label}: {len(diag)}/{n} infeasible "
          f"({sum(d['reproduced'] for d in diag)} reproduced exactly)")
    c = collections.Counter(d["rule"] for d in diag)
    print("   rule:", dict(c))
    print("   shift at failure:", dict(sorted(collections.Counter(
        d["shift"] for d in diag).items())))
    print("   route fraction done at failure: median "
          f"{np.median([d['route_frac'] for d in diag]):.2f}" if diag else "")
    ex = sum(1 for d in diag if d["ext_used"] >= 2)
    rh = sum(1 for d in diag if d["rho2_used"] >= 3)
    either = sum(1 for d in diag if d["ext_used"] >= 2 or d["rho2_used"] >= 3)
    print(f"   at failure: extended-day budget spent {ex}/{len(diag)}, "
          f"reduced-rest budget spent {rh}/{len(diag)}, either {either}/{len(diag)}")
    print("   cause:", dict(collections.Counter(d["cause"] for d in diag)))
    for d in diag:
        print(f"     {d['instance']:24s} stop {d['halted_at']:3d}/{d['n_stops']:3d} "
              f"shift {d['shift']}  ext {d['ext_used']}/2  red.rest {d['rho2_used']}/3  "
              f"{d['action']:7s} {d['cause']:15s} room {d['room_h']:5.2f}h  "
              f"uncounted {d['uncounted_h']:4.2f}h (charge {d['charge_h']:4.2f} + "
              f"overhead {d['overhead_h']:4.2f})  drive {d['drive_real']:4.2f}h "
              f"(guard {d['drive_wc']:4.2f}h)  {d['detail']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["gbt", "nn", "clf"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval", default=None, help="evaluation file in ML/results")
    ap.add_argument("--ood-axis", default=None,
                    help="instead of --eval: an axis of ood_eval.py (kw150, ...)")
    ap.add_argument("--guard-q", type=float, default=0.95)
    ap.add_argument("--spread-room", action="store_true",
                    help="replay with policy_core.spread_room on (the runs must "
                         "have been produced with it)")
    ap.add_argument("--out", default=None, help="write the diagnosis as JSON")
    args = ap.parse_args()
    if args.ood_axis:
        rows, runs = runs_from_ood(args.ood_axis, args.tag, args.guard_q,
                                   args.spread_room)
        where = f"OOD axis {args.ood_axis}"
    else:
        rows, runs = runs_from_eval(args.eval)
        where = args.eval
    diag = diagnose(args.kind, args.tag, runs, args.guard_q, args.spread_room)
    summary(rows, diag, label=f"{args.tag} on {where}")
    if args.out:
        with open(os.path.join(RESULTS, args.out), "w") as fh:
            json.dump(diag, fh, indent=1)
        print(f"\n[saved] {os.path.join(RESULTS, args.out)}")


if __name__ == "__main__":
    main()
