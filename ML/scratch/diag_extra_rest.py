"""Where do the students' extra daily rests on mixed routes come from? (2026-10-05)
Re-drives the chosen model (pool + pmix + mix, seeds 0-2) on every mixed test
run where it rested more often than the LA, and prints both rest schedules:
stop, clock time, driving left to the destination (nominal), and for the
student the 2nd-best action's cost gap at each rest decision."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import mixed_eval as me                                          # noqa: E402
from features import Precomp, charger_curve                       # noqa: E402
from policy_core import load_policy, durations                   # noqa: E402
from src.simulation.BEHDV import BEHDV                           # noqa: E402

LAB = "torch T, + pmix + mix routes"
TAGS = {LAB: "tmlp_T144_physmix_split_list_s0",
        f"{LAB} (s1)": "tmlp_T144_physmix_split_list_s1",
        f"{LAB} (s2)": "tmlp_T144_physmix_split_list_s2"}


def drive(fd, D, E, pol, cv):
    pre = Precomp(fd)
    veh = BEHDV(fd)
    N = int(fd["N"])
    cum = np.r_[0.0, np.cumsum([float(fd["D"].get(i, 0.0)) for i in range(N)])]
    curves, tbar = fd.get("TbarK"), fd["Tbar"]
    log = []
    while veh.stop < N:
        s = veh.stop
        if curves:
            fd["Tbar"] = charger_curve(fd, s if s in pre.K else int(pre.next_cs[s]))
        cand = pol.candidates(fd, pre, s, veh, cv, k=12)
        act, tc, key, score = cand[0]
        rest = key.endswith(("_r1", "_r2"))
        # best alternative of the other kind (rest vs no rest)
        alt = [c for c in cand[1:] if c[2].endswith(("_r1", "_r2")) != rest]
        gap = (alt[0][3] - score) if alt else np.nan
        log.append(dict(stop=s, t=veh.t_arr, left=cum[N] - cum[s], sd=veh.sd,
                        h=getattr(veh, "h", 0.0), key=key, rest=rest, gap=gap))
        veh.advance(action=act, D_next=float(D[s]), E_next=float(E[s]),
                    milp_sol=dict(feasible=True,
                                  sol=[dict(i=0, **durations(fd, s, act, tc))]))
        if veh.is_halted:
            break
    fd["Tbar"] = tbar
    return log, veh.t_arr - float(fd.get("T_START", 8.0))


rows = me.read_rows(me.store_path(0.99, True))
pols = {}
for v in ("pmix", "mix"):
    paths = {n: p for _v, n, p in me.instances([v], uniform=False)}
    for r in rows:
        if r["variant"] != v or r["method"] not in TAGS or not r["completed"]:
            continue
        la = me.la_row(v, r["instance"])
        if not la or not la["completed"] or r["rests"] <= la["rests"]:
            continue
        tag = TAGS[r["method"]]
        pol = pols.setdefault(tag, load_policy("torch", tag, guard_q=0.99, spread_room=True))
        fd, D, E, cv = me.load(paths[r["instance"]])
        N = int(fd["N"])
        cum = np.r_[0.0, np.cumsum([float(fd["D"].get(i, 0.0)) for i in range(N)])]
        log, dur = drive(fd, D, E, pol, cv)
        tr, td = la["trajectory"], []
        import json
        with open(me.la_solution(v, r["instance"])) as fh:
            td = json.load(fh).get("td_list") or []
        la_rests = [(int(tr[k]["stop"]), float(tr[k]["t_arr"]), cum[N] - cum[int(tr[k]["stop"])])
                    for k in range(min(len(tr), len(td)))
                    if float(td[k]) - float(tr[k]["t_arr"]) >= me.REST_DWELL_H]
        st_rests = [(x["stop"], x["t"], x["left"], x["gap"]) for x in log if x["rest"]]
        print(f"\n{v} {r['instance']} [{tag[-2:]}]  student {dur:.1f} h, {len(st_rests)} rests;"
              f"  LA {la['duration_h']:.1f} h, {len(la_rests)} rests;  route {cum[N]:.1f} h driving")
        print("   LA rests     : " + ", ".join(f"stop {s} t={t:.1f} left={l:.1f}h" for s, t, l in la_rests))
        print("   student rests: " + ", ".join(f"stop {s} t={t:.1f} left={l:.1f}h (alt +{g*60:.0f} min)"
                                               for s, t, l, g in st_rests))
        last = log[-1]
        print(f"   student after last rest: {cum[N] - cum[st_rests[-1][0]]:.1f} h of driving to go")
