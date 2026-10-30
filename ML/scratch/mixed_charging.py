"""
mixed_charging.py — where each method charges on the mixed-power test routes
===========================================================================
Share of the charged energy taken at 150 / 350 / 700 / 1000 kW chargers, on
the mixed-charger test routes that the LA has run (pmix + mix), for: the
hindsight oracle and the LA (stored runs), the greedy heuristic (re-driven,
mixed_eval.run_greedy_ml), and two PyTorch students alone and with the frozen
decisive-state rollout (re-driven on the stored realisation with rl_env,
checked against the stored cluster test runs):

    A  tmlp_T144_phys_split_list_s0     base case + the 8 one-at-a-time truck types
    B  tmlp_T144_physmix_split_list_s0  A's data + the 238 mixed training routes

Writes ML/results/mixed_charging.json ({method: {kW: kWh}}, plus per-route
checks) and prints the shares.

    python ML/scratch/mixed_charging.py --workers 4
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "mixed_charging.json")
STUDENTS = [("A", "tmlp_T144_phys_split_list_s0", "dsr_mtest_tmlp_T144_phys_split_list_s0"),
            ("B", "tmlp_T144_physmix_split_list_s0", "dsr_mtest_tmlp_T144_physmix_split_list_s0")]
_G = {}


def _init():
    from rl_env import final_student
    _G["pol"] = {lab: final_student("torch", tag) for lab, tag, _ in STUDENTS}


def charged(veh, E):
    st, ea = veh.stop_history, veh.e_arr_history
    return {st[k]: ea[k + 1] + float(E[st[k]]) - ea[k] for k in range(len(st) - 1)}


def one(job):
    import mixed_eval as me
    from features import charger_kw
    from rl_env import load_route, realised, run_batch
    from rl_policy import RolloutChooser
    v, name, path = job
    fd, D, E, cv = load_route(name, path=path)
    kw = {k: int(round(charger_kw(fd, k))) for k in fd["K"]}
    E_by = collections.defaultdict(collections.Counter)
    la = me.la_row(v, name)
    tr = la["trajectory"]
    for k in range(len(tr) - 1):
        st = int(tr[k]["stop"])
        c = float(tr[k + 1]["e_arr"]) + float(E[st]) - float(tr[k]["e_arr"])
        if st in kw and c > 1e-3:
            E_by["LA"][kw[st]] += c
    with open(me.oracle_path(v, name)) as fh:
        o = json.load(fh)
    for s in o.get("sol") or []:
        if s["i"] in kw and s["ed"] - s["ea"] > 1e-3:
            E_by["oracle"][kw[s["i"]]] += s["ed"] - s["ea"]
    fd2, D2, E2, cv2 = me.load(path)
    for st, c in charged(me.run_greedy_ml(fd2, D2, E2, cv2, return_vehicle=True), E2).items():
        if st in kw and c > 1e-3:
            E_by["Greedy"][kw[st]] += c
    dur = {}
    for lab, _tag, _run in STUDENTS:
        base = _G["pol"][lab]
        ch = RolloutChooser(base, futures=32, race=8)
        ch.route, ch.cv = name, cv
        res, vehs = run_batch(base, fd, cv, [(None, realised(D, E)), (ch, realised(D, E))],
                              return_vehicles=True)
        for nm, r, veh in ((lab, res[0], vehs[0]), (lab + "+rollout", res[1], vehs[1])):
            dur[nm] = r["duration"]
            for st, c in charged(veh, E).items():
                if st in kw and c > 1e-3:
                    E_by[nm][kw[st]] += c
    return dict(variant=v, instance=name, E={m: dict(c) for m, c in E_by.items()}, duration=dur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import mixed_eval as me
    me.ROUTE_SET = "test"
    jobs = []
    for v in ("pmix", "mix"):
        keep = set(me.la_routes(v))
        jobs += [j for j in me.instances([v], uniform=False) if j[1] in keep]
    stored = {}
    for lab, _tag, run in STUDENTS:
        for l in open(os.path.join(RES, f"rl_search_{run}.jsonl")):
            r = json.loads(l)
            stored[(lab, r["instance"])] = r["res"][0][0]["duration"]
            stored[(lab + "+rollout", r["instance"])] = r["res"][1][0]["duration"]
    print(f"{len(jobs)} mixed test routes with an LA run ({a.workers} workers)", flush=True)
    tot = collections.defaultdict(collections.Counter)
    mism = 0
    rows = []
    with Pool(a.workers, initializer=_init) as pool:
        for n, r in enumerate(pool.imap_unordered(one, jobs), 1):
            rows.append(r)
            for m, c in r["E"].items():
                for k, e in c.items():
                    tot[m][int(k)] += e
            for m, d in r["duration"].items():
                s = stored.get((m, r["instance"]))
                mism += int(s is not None and abs(s - d) > 1e-6)
            if n % 8 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}", flush=True)
    out = dict(routes=len(rows), mismatches_vs_cluster=mism,
               energy={m: {str(k): float(e) for k, e in sorted(c.items())} for m, c in tot.items()})
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"mismatches vs the stored cluster runs: {mism}")
    order = ["oracle", "LA", "Greedy", "A", "A+rollout", "B", "B+rollout"]
    print(f"{'method':12s}" + "".join(f"{k:>8d} kW" for k in (150, 350, 700, 1000)))
    for m in order:
        c = tot[m]
        s = sum(c.values())
        print(f"{m:12s}" + "".join(f"{100 * c.get(k, 0) / s:10.0f} %" for k in (150, 350, 700, 1000)))


if __name__ == "__main__":
    main()
