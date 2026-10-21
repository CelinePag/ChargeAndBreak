"""Why does a gate call create an extra daily rest? (2026-10-21)
Replays one validation route twice on the laptop -- the student alone, and the
student with the trees gate at the 10 % budget (real LA calls, same scenario
seeds as run_hybrid) -- and prints, stop by stop, who decided, the action,
and the driving-time counters.  Anaconda's site-packages are APPENDED to
sys.path for gurobipy (as smoke_gate_la.py); nothing is installed."""
import os
import sys
import zlib

sys.path.append(r"C:\Users\celinep\AppData\Local\anaconda3\Lib\site-packages")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

import numpy as np                                                 # noqa: E402
import mixed_eval as me                                            # noqa: E402
import run_hybrid as rh                                            # noqa: E402
from features import Precomp, action_key                           # noqa: E402
from policy_core import durations, load_policy                     # noqa: E402
from src.simulation.BEHDV import BEHDV                             # noqa: E402

TAG = "gbt_F95_base_s0"


def run(name, gate):
    fd, D, E, cv = me.load(os.path.join(me._ROOT, "instances", name + ".json"))
    fd["title"] = name
    fd["la_energy_quantile"], fd["la_energy_cv"] = float(rh.LA_CONFIG["la_energy_quantile"]), cv
    pol = load_policy("gbt", TAG, guard_q=0.99, spread_room=True)
    pol.feature_q, pol.pay_mseq = None, True
    pre, veh, N = Precomp(fd), BEHDV(fd), int(fd["N"])
    trace = []
    while veh.stop < N:
        s = veh.stop
        legal, rows, cost, feas, flags = pol._score(fd, pre, s, veh, cv)
        keys = list(pol._legal_keys)
        sc = cost + 1e6 * (feas < pol.feas_thr)
        j = int(np.argmin(sc))
        act, key = legal[j], keys[j]
        tc = pol._charge_hours(fd, s, veh, act, rows[j:j + 1], flags)
        who, plan, g = "student", None, None
        if gate is not None and int((sc < 1e5).sum()) >= 2:
            g = gate.score(fd, pre, s, veh, cv, cost, feas, keys, j)
            if g >= gate.thr:
                got = rh.ask_la(fd, s, veh, cv, zlib.crc32(f"{name}|{TAG}|{s}".encode()),
                                4, rh.LA_CONFIG["time_limit"])
                if got is not None:
                    act, plan = got
                    la_key = action_key(act.get("y", 0), act.get("break_type"), act.get("rest_type"))
                    who = f"LA (student wanted {key})"
                    key = la_key
        trace.append((s, veh.t_arr, veh.cd, veh.sd, veh.h, key, who, g))
        sol = plan if plan is not None else dict(
            feasible=True, sol=[dict(i=0, **durations(fd, s, act, tc, True))])
        veh.advance(action=act, D_next=float(D[s]), E_next=float(E[s]), milp_sol=sol)
        if veh.is_halted:
            break
    return trace, veh.t_arr - float(fd.get("T_START", 8.0))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "RshortCmanyTnone_20"
    a, da = run(name, None)
    b, db = run(name, rh.Gate("gbt", 10, nominal=True))
    print(f"{name}: student alone {da:.2f} h, with gate {db:.2f} h")
    print(f"{'stop':>4s} | {'alone':22s} | {'with gate':60s} | t_arr  cd    sd    h")
    for x, y in zip(a, b):
        mark = " <==" if x[5] != y[5] or y[6] != "student" else ""
        print(f"{x[0]:4d} | {x[5]:8s} t{x[1]:6.2f} | {y[5]:8s} {y[6]:42s} | "
              f"{y[1]:6.2f} {y[2]:5.2f} {y[3]:5.2f} {y[4]:5.2f}{mark}")
