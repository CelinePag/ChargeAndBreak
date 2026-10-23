"""
gate_rollout.py — is a call to the LA worth it, given that the STUDENT drives on?
================================================================================
The one-step gate label (gate_data.py) scores the LA's action with the LA's own
horizon cost, i.e. as if the LA kept driving.  In the closed loop it does not:
the student takes over again, and on a few validation routes the LA's early
rest was then followed by a second, forced rest (2026-10-21, handoff_trace.py).

Here a call is valued the way it will be used.  At a decision, both the LA's
action and the student's own action are applied, then the STUDENT drives to
the destination, under the same K sampled futures for both (common random
numbers; the LA's own scenario generator, never the realised times):

    value of the call = mean over futures of
                        [cost after the student's action - cost after the LA's]

in hours of route cost (arrival + 0.5 h per missed window); > 0 = calling
helps.  A rollout that breaks a rule costs the median finished cost of the
pair + FAIL_EXTRA_H, so one failure cannot swamp the mean.  This is
AggreVaTe's / LOLS' cost-to-go of the policy that will actually continue.

Two uses, one function (call_value):
  - labels: at every LA-visited decision of seeds 1-21 where the student
    (fold model on the training routes, deployed model on 20-21) would act
    differently from the LA's choice, the state is restored from the LA's
    stored trajectory (gate G1 below) and both actions are rolled out
    -> ML/data/gate_<arm>_rollout.npz, aligned with gate_<arm>.npz rows;
  - at driving time: run_hybrid.py --check keeps the LA's action only if this
    value is positive by more than z standard errors.

    python ML/code/gate_rollout.py --arm gbt [--k 32] [--jobs 6] [--limit 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.instance_gen.instance_io import load_instance_json       # noqa: E402
from src.simulation.BEHDV import BEHDV                            # noqa: E402
from src.simulation.scenarios import generate_scenarios           # noqa: E402

from dataset import DATA, load                                    # noqa: E402
from extract import reconstruct_spread                            # noqa: E402
from features import Precomp, action_key, charger_curve, key_to_action  # noqa: E402
from fpi_policy import drive_fleet                                # noqa: E402
from gate_data import ARMS, FOLDS, fold_of                        # noqa: E402
from policy_core import load_policy                               # noqa: E402
from rollout_policy import FAIL_H, clone, step                    # noqa: E402

SOL = os.path.join(_ROOT, "solutions", "basecase")
INST = os.path.join(_ROOT, "instances")
FAIL_EXTRA_H = 24.0


# ── the shared valuation ─────────────────────────────────────────────────────

def call_value(fd, pre, veh, cv, pol, la_act, la_tauc, s_act, s_tauc, k, seed):
    """Paired value (h) of taking la_act instead of s_act at veh's stop, the
    student `pol` driving on.  Returns (mean, se, fails_la, fails_student)."""
    s0, N = int(veh.stop), int(fd["N"])
    scen = generate_scenarios(fd, s0, N, n_scenarios=k, cv=cv, seed=seed)
    vehs, sc = [], []
    tbar = fd["Tbar"]
    if fd.get("TbarK"):            # per-charger curves: this stop's, as run_student
        fd["Tbar"] = charger_curve(fd, s0 if s0 in pre.K else int(pre.next_cs[s0]))
    for act, tc in ((la_act, la_tauc), (s_act, s_tauc)):
        for x in scen:
            v = clone(veh)
            step(v, fd, act, tc, x["D"][s0], x["E"][s0], pol.pay_mseq)
            vehs.append(v)
            sc.append(x)
    cost = drive_fleet(pol, fd, pre, cv, vehs, sc)
    fd["Tbar"] = tbar
    ok = cost < FAIL_H / 2
    ref = float(np.median(cost[ok])) if ok.any() else 0.0
    cost = np.where(ok, cost, ref + FAIL_EXTRA_H)
    d = cost[k:] - cost[:k]                       # student minus LA: > 0 = the call helps
    return (float(d.mean()), float(d.std(ddof=1) / np.sqrt(k)) if k > 1 else float("inf"),
            int((~ok[:k]).sum()), int((~ok[k:]).sum()))


def student_pick(pol, fd, pre, veh, cv):
    """policy_core.decide, keeping the key: (action, tauc, key)."""
    a, tc, key = pol.decide(fd, pre, veh.stop, veh, cv)
    return a, tc, key


# ── restoring a stored LA state ──────────────────────────────────────────────

def restore(fd, tr, h):
    """A BEHDV standing at trajectory entry `tr` with spread h (no history)."""
    v = BEHDV(fd)
    one = dict(stop_history=[int(tr["stop"])], t_arr_history=[float(tr["t_arr"])],
               e_arr_history=[float(tr["e_arr"])], cd_history=[float(tr["cd"])],
               sd_history=[float(tr["sd"])], sw_history=[float(tr["sw"])],
               h_history=[float(h)], phi_history=[int(tr.get("phi", 0))],
               rho2_used_history=[int(tr.get("rho2_used", 0))],
               ext_shift_used_history=[int(tr.get("ext_shift_used", 0))],
               sw_week_history=[0.0], actions=[], durations=[], td_list=[],
               D_actual_list=[])
    v.load_checkpoint(one)
    return v


def la_solution(inst):
    fs = sorted(glob.glob(os.path.join(SOL, f"{inst}_LA_MIPTAIL_*.json")))
    return json.load(open(fs[-1])) if fs else None


# ── labels ───────────────────────────────────────────────────────────────────

_POL = {}


def _policy(kind, tag):
    if tag not in _POL:
        p = load_policy(kind, tag, guard_q=0.99, spread_room=True)
        p.feature_q, p.pay_mseq = None, True          # as every run since 2026-10-20
        _POL[tag] = p
    return _POL[tag]


def route_labels(job):
    inst, arm, decs, k = job        # decs: [(gate_row, stop, la_key, la_tauc, t_feat, e_feat)]
    out, rep = [], dict(instance=inst, status="ok", dev_t=0.0, dev_e=0.0, n=0, rolled=0, s=0.0)
    t0 = time.time()
    try:
        sol = la_solution(inst)
        traj = sol.get("sim_trajectory") or []
        spread = reconstruct_spread(sol, traj)
        by_stop = {int(t["stop"]): (i, t) for i, t in enumerate(traj)}
        fd, _D, _E, cv = load_instance_json(os.path.join(INST, inst + ".json"))
        fd["_horizon_h"] = 24.0
        pre = Precomp(fd)
        T0 = float(fd.get("T_START", 8.0))
        seed_r = int(inst.rsplit("_", 1)[1])
        fold = fold_of(seed_r)
        fmt, full = ARMS[arm]
        tag = full if fold is None else fmt.format(k=fold)
        pol = _policy(arm, tag)
        for row, stop, la_key, la_tauc, t_feat, e_feat in decs:
            hit = by_stop.get(stop)
            if hit is None:
                continue
            i, tr = hit
            veh = restore(fd, tr, spread[i] if i < len(spread) else 0.0)
            # G1: the restored state is the one the training rows describe
            rep["dev_t"] = max(rep["dev_t"], abs((veh.t_arr - T0) - t_feat))
            rep["dev_e"] = max(rep["dev_e"], abs(veh.e_arr - e_feat))
            s_act, s_tc, s_key = student_pick(pol, fd, pre, veh, cv)
            rep["n"] += 1
            if s_key == la_key:
                out.append((row, 0.0, 0.0, 0, 0, 1, s_key))
                continue
            la_act = key_to_action(la_key)
            mean, se, fl, fs = call_value(fd, pre, veh, cv, pol, la_act, la_tauc, s_act, s_tc, k,
                                          zlib.crc32(f"{inst}|{stop}|gate".encode()))
            out.append((row, mean, se, fl, fs, 0, s_key))
            rep["rolled"] += 1
    except Exception as exc:                      # one bad route must not kill the batch
        rep["status"] = f"error:{type(exc).__name__}:{exc}"
    rep["s"] = time.time() - t0
    return out, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gbt", choices=list(ARMS))
    ap.add_argument("--k", type=int, default=32, help="sampled futures per action")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="routes (smoke test)")
    ap.add_argument("--max-seed", type=int, default=21, help="label seeds 1..this (21 = fit + stop)")
    args = ap.parse_args()

    g = np.load(os.path.join(DATA, f"gate_{args.arm}.npz"), allow_pickle=True)
    d = load()
    vocab = [str(x) for x in d["action_vocab"]]
    names = [str(x) for x in d["feature_names"]]
    it, ie = names.index("t_elapsed"), names.index("soc_kwh")
    chosen = np.flatnonzero(d["chosen"] == 1)
    la = {(int(d["instance_ix"][r]), int(d["stop"][r])): r for r in chosen}
    insts = [str(x) for x in g["instances"]]
    jobs = {}
    for row in np.flatnonzero(g["seed"] <= args.max_seed):
        key = (int(g["instance_ix"][row]), int(g["stop"][row]))
        r = la.get(key)
        if r is None:
            continue
        jobs.setdefault(insts[key[0]], []).append(
            (int(row), key[1], vocab[d["action_ix"][r]], float(d["tauc"][r]),
             float(d["X"][r, it]), float(d["X"][r, ie])))
    routes = sorted(jobs)
    if args.limit:
        routes = routes[:: max(1, len(routes) // args.limit)][: args.limit]
    print(f"[rollout] {args.arm}: {sum(len(jobs[r]) for r in routes)} decisions on {len(routes)} "
          f"routes, K={args.k} futures per action, {args.jobs} jobs", flush=True)

    t0 = time.time()
    res, reps = [], []
    with Pool(args.jobs) as pool:
        for i, (out, rep) in enumerate(pool.imap_unordered(
                route_labels, [(r, args.arm, jobs[r], args.k) for r in routes])):
            res += out
            reps.append(rep)
            if (i + 1) % 20 == 0 or i + 1 == len(routes):
                print(f"  {i+1}/{len(routes)} routes  {time.time()-t0:.0f}s", flush=True)
    bad = [r for r in reps if r["status"] != "ok"]
    print(f"[gates] restored state vs training features: max |dt| "
          f"{max(r['dev_t'] for r in reps):.4f} h, max |de| {max(r['dev_e'] for r in reps):.3f} kWh; "
          f"{len(bad)} routes with errors" + (f": {bad[0]['instance']} {bad[0]['status']}" if bad else ""))
    if not res:
        return
    a = np.array([r[:6] for r in res], dtype=np.float64)
    v, same = a[:, 1], a[:, 5] > 0
    print(f"[labels] {len(a)} decisions, student = LA on {same.mean()*100:.1f} %; rolled out {(~same).sum()}")
    if (~same).any():
        vr, ser = v[~same], a[~same, 2]
        print(f"  value of the call where they differ: mean {vr.mean()*60:+.1f} min, "
              f"median {np.median(vr)*60:+.1f}, helps (> 2 se) {np.mean(vr > 2 * ser)*100:.0f} %, "
              f"hurts (< -2 se) {np.mean(vr < -2 * ser)*100:.0f} %, "
              f"rollout failures LA / student {int(a[:, 3].sum())} / {int(a[:, 4].sum())}")
    out = os.path.join(DATA, f"gate_{args.arm}_rollout{'_smoke' if args.limit else ''}.npz")
    np.savez_compressed(out, row=a[:, 0].astype(np.int64), value=v, se=a[:, 2],
                        fail_la=a[:, 3].astype(np.int16), fail_student=a[:, 4].astype(np.int16),
                        same=same, student_key=np.array([r[6] for r in res]), k=args.k)
    print(f"[saved] {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
