"""
window_misses.py — why the students miss twice as many time windows as the LA
==============================================================================
On the 66 base-case VALIDATION routes (seeds 20-21), stored realisations:
  - every customer visit of the LA, Greedy (stored runs), the oracle (stored
    schedule), and the PyTorch / trees students and the round-1 RL student
    (re-driven): hit, early or late, and by how much;
  - by window class (tight 0.5-1 h half-width, medium 1-3 h, large 3-6 h);
  - whether the LA and the oracle WAIT: breaks longer than 45 min or rests
    longer than their minimum (9 / 11 h), which no student can do (its break
    durations are the minima, policy_core.durations).
Prints only.

    python ML/scratch/window_misses.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from rl_env import final_student, load_route, realised, run_batch, split_routes  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SOL = os.path.join(ROOT, "solutions", "basecase")
STUDENTS = [("PyTorch", "torch", "tmlp_F95_split_list_s0"),
            ("trees", "gbt", "gbt_F95_base_s0"),
            ("PyTorch RL round 1", "torch", "tmlp_F95_split_list_s0_exit1")]
T0 = 8.0


def wclass(inst):
    for c in ("Ttight", "Tmedium", "Tlarge", "Tnone"):
        if c in inst:
            return c[1:]
    return "?"


def main():
    routes = split_routes("stop")
    visits = defaultdict(list)          # policy -> [(class, half-width, status, amount)]
    waits = defaultdict(lambda: dict(brk_long=0, rest_long=0, wait=0, stops=0, wait_h=0.0))
    pols = {lab: final_student(k, t) for lab, k, t in STUDENTS}
    for inst in routes:
        fd, D, E, cv = load_route(inst)
        C = sorted(int(c) for c in fd["C"])
        wa, wf = fd.get("Wha", {}), fd.get("Whf", {})

        def win(c):
            a, b = wa.get(c, wa.get(str(c))), wf.get(c, wf.get(str(c)))
            return (None, None) if a is None or b is None else (float(a), float(b))

        def add(lab, arrivals):
            for c in C:
                a, b = win(c)
                if a is None or c not in arrivals:
                    continue
                t = arrivals[c]
                st, amt = ("early", a - t) if t < a - 1e-3 else ("late", t - b) if t > b + 1e-3 else ("hit", 0.0)
                visits[lab].append((wclass(inst), (b - a) / 2, st, amt))

        # stored LA and Greedy: arrival times from the trajectory
        for lab, m in (("LA", "LA_MIPTAIL"), ("Greedy", "GREEDY")):
            fs = sorted(glob.glob(os.path.join(SOL, f"{inst}_{m}_*.json")))
            if not fs:
                continue
            s = json.load(open(fs[-1]))
            tr = s["sim_trajectory"]
            add(lab, {int(x["stop"]): float(x["t_arr"]) for x in tr})
            if lab == "LA":
                for d in s["durations_list"]:
                    w = waits["LA"]
                    w["stops"] += 1
                    w["brk_long"] += int(float(d.get("taub", 0)) > 0.77)
                    w["rest_long"] += int(float(d.get("taur", 0)) > 1 and abs(float(d["taur"]) - 9) > 0.02
                                          and abs(float(d["taur"]) - 11) > 0.02)
        # oracle
        p = os.path.join(SOL, f"oracle_{inst}.json")
        if os.path.exists(p):
            o = json.load(open(p))
            if o.get("feasible"):
                add("oracle", {int(x["i"]): float(x["ta"]) for x in o["sol"]})
                for x in o["sol"]:
                    w = waits["oracle"]
                    w["stops"] += 1
                    w["brk_long"] += int(float(x.get("taub") or 0) > 0.77)
                    tr_ = float(x.get("taur") or 0)
                    w["rest_long"] += int(tr_ > 1 and abs(tr_ - 9) > 0.02 and abs(tr_ - 11) > 0.02)
                    w["wait"] += int(float(x.get("wait") or 0) > 1e-3)
                    w["wait_h"] += float(x.get("wait") or 0)
        # students, re-driven
        for lab, pol in pols.items():
            _res, vehs = run_batch(pol, fd, cv, [(None, realised(D, E))], return_vehicles=True)
            v = vehs[0]
            add(lab, {int(s): float(t) for s, t in zip(v.stop_history, v.t_arr_history)})

    order = ["oracle", "LA", "PyTorch", "trees", "PyTorch RL round 1", "Greedy"]
    print(f"{len(routes)} validation routes; customer visits with a window per policy\n")
    print(f"{'policy':20s} {'visits':>6s} {'missed':>7s} {'early':>6s} {'late':>6s} "
          f"{'early: median / share < 45 min':>32s} {'late: median / share < 45 min':>31s}")
    for lab in order:
        v = visits.get(lab, [])
        if not v:
            continue
        e = [x[3] for x in v if x[2] == "early"]
        l_ = [x[3] for x in v if x[2] == "late"]
        fmt = lambda a: (f"{np.median(a):5.2f} h / {100 * np.mean(np.array(a) < 0.75):3.0f} %"
                         if a else "      -      ")
        print(f"{lab:20s} {len(v):6d} {len(e) + len(l_):7d} {len(e):6d} {len(l_):6d} "
              f"{fmt(e):>32s} {fmt(l_):>31s}")
    print("\nmissed windows by window class (misses / visits):")
    for lab in order:
        v = visits.get(lab, [])
        if not v:
            continue
        parts = []
        for c in ("tight", "medium", "large"):
            sub = [x for x in v if x[0] == c]
            if sub:
                parts.append(f"{c} {sum(x[2] != 'hit' for x in sub)}/{len(sub)}")
        print(f"  {lab:20s} " + ", ".join(parts))
    print("\nwaiting by the planners (stops where a break is longer than 45 min, or a rest "
          "longer than its minimum):")
    for lab in ("LA", "oracle"):
        w = waits[lab]
        print(f"  {lab:7s} {w['brk_long']} long breaks, {w['rest_long']} long rests over "
              f"{w['stops']} stops" + (f"; explicit wait at {w['wait']} stops ({w['wait_h']:.1f} h)"
                                       if lab == "oracle" else ""))


if __name__ == "__main__":
    main()
