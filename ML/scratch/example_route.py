"""
example_route.py — one test route, every policy, side by side
==============================================================
Picks the base-case TEST route whose results are closest to every method's
own average (LA, Greedy, trees and PyTorch students seed 0, each with the
decisive-state rollout; gap to the oracle with the window penalty, the
paper's definition) and shows what each policy does on it: duration, gap,
rests (where, which type), charging stops and time, missed windows, and the
decisions where the rollout overruled the student.  The students and
rollouts are re-driven here on the stored realisation (and checked against
the stored test runs); the LA, Greedy and oracle are read from their stored
runs.  Prints only.

    python ML/scratch/example_route.py              # choose and show
    python ML/scratch/example_route.py --inst RmediumCmanyTmedium_23
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from paper_link import baseline_gaps, gaps, oracle_facts             # noqa: E402
from rl_env import final_student, load_route, realised, run_batch    # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(HERE, "..", "results")
SOL = os.path.join(ROOT, "solutions", "basecase")
ARMS = [("trees", "gbt", "gbt_F95_base_s0"), ("PyTorch", "torch", "tmlp_F95_split_list_s0")]
T0 = 8.0


def stored_gaps():
    """{inst: {method: gap_pen}} on the test routes, seed-0 students."""
    out = {}
    for lab, _kind, tag in ARMS:
        for line in open(os.path.join(RES, f"rl_search_dsr_test_{tag}.jsonl")):
            r = json.loads(line)
            orc = oracle_facts(r["instance"])
            if orc is None:
                continue
            s, d = r["res"][0][0], r["res"][1][0]
            g = out.setdefault(r["instance"], {})
            g[lab] = gaps(s["duration"], s["misses"], s["duration"] + T0, orc)[1]
            g[lab + " + rollout"] = gaps(d["duration"], d["misses"], d["duration"] + T0, orc)[1]
            bl = baseline_gaps(r["instance"], orc)
            g["LA"] = bl.get("LA_MIPTAIL", (None, None))[1]
            g["Greedy"] = bl.get("GREEDY", (None, None))[1]
    return {k: v for k, v in out.items() if all(x is not None for x in v.values()) and len(v) == 6}


def choose(G):
    meths = list(next(iter(G.values())))
    mu = {m: np.mean([g[m] for g in G.values()]) for m in meths}
    sd = {m: np.std([g[m] for g in G.values()]) for m in meths}
    score = {i: np.mean([((g[m] - mu[m]) / sd[m]) ** 2 for m in meths]) for i, g in G.items()}
    best = sorted(score, key=score.get)[:5]
    print(f"{len(G)} test routes with every method; averages: "
          + ", ".join(f"{m} {mu[m]:.2f} %" for m in meths))
    for i in best:
        print(f"  {i:28s} distance {score[i]:.3f}: "
              + ", ".join(f"{m} {G[i][m]:.2f}" for m in meths))
    return best[0], mu


def stop_kind(fd, s):
    if s in set(fd["K"]):
        return "charger"
    if s in set(fd["C"]):
        return "customer"
    if s in set(fd.get("L", [])):
        return "layby"
    return "origin" if s == 0 else "?"


def events_from(fd, t_arr, e_arr, td, durs, dest_t):
    """[(stop, kind, h into the route, dwell h, charge h, kWh charged, break h, rest h)]."""
    from src.simulation.BEHDV import _energy_after_charging
    ev = []
    for k in range(len(td)):
        dw = float(td[k]) - float(t_arr[k])
        if dw < 1e-6:
            continue
        d = durs[k]
        tc = float(d.get("tauc", 0.0))
        kwh = (_energy_after_charging(float(e_arr[k]), tc, fd, stop=k) - float(e_arr[k])) if tc > 0 else 0.0
        ev.append((k, stop_kind(fd, k), float(t_arr[k]) - T0, dw, tc, kwh,
                   float(d.get("taub", 0.0)), float(d.get("taur", 0.0))))
    return ev


def summary(name, ev, duration, misses, gap):
    rests = [e for e in ev if e[7] > 1.0]
    ch = [e for e in ev if e[4] > 0]
    brk = [e for e in ev if e[6] > 1e-6 or (e[4] >= 0.75 - 1e-6 and e[7] <= 1.0)]
    rest_txt = "; ".join(f"{'11 h' if e[7] > 10 else '9 h'} at stop {e[0]} ({e[1]}, hour {e[2]:.1f})"
                         for e in rests)
    print(f"{name:20s} {duration:6.2f} h  gap {gap:5.2f} %  missed windows {misses}  "
          f"rests {len(rests)}: {rest_txt}")
    print(f"{'':20s} charging: {len(ch)} stops, {sum(e[4] for e in ch):.2f} h, "
          f"{sum(e[5] for e in ch):.0f} kWh; breaks or charges >= 45 min: {len(brk)}; "
          f"dwell outside rests {sum(e[3] for e in ev if e[7] <= 1.0):.2f} h")


def show(inst, mu):
    fd, D, E, cv = load_route(inst)
    orc = oracle_facts(inst)
    N = int(fd["N"])
    print(f"\n=== {inst}: {N} stops, {len(fd['K'])} chargers, {len(fd['C'])} customers, "
          f"nominal driving {sum(float(fd['D'].get(i, 0)) for i in range(N)):.1f} h ===")
    # oracle
    o = json.load(open(os.path.join(SOL, f"oracle_{inst}.json")))
    sol = sorted(o["sol"], key=lambda s: s["i"])
    ta = [s["ta"] for s in sol]
    ev = []
    for s in sol:
        dw = s["td"] - s["ta"] if s.get("td") is not None else 0.0
        if dw > 1e-6:
            ev.append((s["i"], stop_kind(fd, s["i"]), s["ta"] - T0, dw, s["tauc"], s["ed"] - s["ea"],
                       s["taub"], s["taur"]))
    summary("oracle (hindsight)", ev, ta[-1] - T0, int(sum(round(s.get("delta", 0)) for s in sol)), 0.0)
    # stored LA and Greedy
    for lab, m in (("LA", "LA_MIPTAIL"), ("Greedy", "GREEDY")):
        f = sorted(glob.glob(os.path.join(SOL, f"{inst}_{m}_*.json")))[-1]
        s = json.load(open(f))
        tr = s["sim_trajectory"]
        ev = events_from(fd, [x["t_arr"] for x in tr], [x["e_arr"] for x in tr], s["td_list"],
                         s["durations_list"], None)
        mis = int(s["metrics"].get("tw_n_misses", 0))
        summary(lab, ev, s["duration_h"], mis, gaps(s["duration_h"], mis, s["duration_h"] + T0, orc)[1])
    # students and rollouts, re-driven on the stored realisation
    from rl_policy import RolloutChooser
    from features import action_key
    for lab, kind, tag in ARMS:
        base = final_student(kind, tag)
        rec = []
        ch = RolloutChooser(base, futures=32, race=8, record=rec)
        ch.route, ch.cv = inst, cv
        res, vehs = run_batch(base, fd, cv, [(None, realised(D, E)), (ch, realised(D, E))],
                              return_vehicles=True)
        stored = {json.loads(l)["instance"]: json.loads(l)
                  for l in open(os.path.join(RES, f"rl_search_dsr_test_{tag}.jsonl"))}[inst]
        for nm, r, v, k in ((lab, res[0], vehs[0], 0), (lab + " + rollout", res[1], vehs[1], 1)):
            same = abs(r["duration"] - stored["res"][k][0]["duration"]) < 1e-6
            ev = events_from(fd, v.t_arr_history, v.e_arr_history, v.td_list, v.durations, None)
            summary(nm + ("" if same else " (!)"), ev, r["duration"], r["misses"],
                    gaps(r["duration"], r["misses"], r["duration"] + T0, orc)[1])
        sw = [x for x in rec if int(x["choice"]) != 0]
        print(f"{'':20s} the rollout checked {len(rec)} decisions and overruled the student at {len(sw)}:")
        for x in sw:
            c = int(x["choice"])
            print(f"{'':22s} stop {x['stop']:3d} ({stop_kind(fd, x['stop'])}, {x['kind']}): "
                  f"{x['keys'][0]} -> {x['keys'][c]}, measured {x['d_mean'][c]:+.2f} h "
                  f"(± {x['d_se'][c]:.2f}, {int(x['n_fut'][c])} futures)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default=None)
    a = ap.parse_args()
    G = stored_gaps()
    inst, mu = choose(G)
    show(a.inst or inst, mu)


if __name__ == "__main__":
    main()
