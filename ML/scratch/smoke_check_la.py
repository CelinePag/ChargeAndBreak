"""One-off (2026-10-21): the trees gate at 10 % on one validation route, with
real LA calls, without and with the rollout check (run_hybrid --check 32).
Anaconda's site-packages appended for gurobipy, as smoke_gate_la.py."""
import os, sys, time
sys.path.append(r"C:\Users\celinep\AppData\Local\anaconda3\Lib\site-packages")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import mixed_eval as me
import run_hybrid as rh
from policy_core import load_policy

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "RshortCmanyTnone_20"
    for check in (None, (32, 1.0)):
        pol = load_policy("gbt", "gbt_F95_base_s0", guard_q=0.99, spread_room=True)
        pol.feature_q, pol.pay_mseq = None, True
        gate = rh.Gate("gbt", 10, nominal=True)
        fd, D, E, cv = me.load(os.path.join(me._ROOT, "instances", name + ".json"))
        fd["title"] = name
        fd["la_energy_quantile"], fd["la_energy_cv"] = float(rh.LA_CONFIG["la_energy_quantile"]), cv
        t0 = time.time()
        row, _ = rh.drive(fd, D, E, cv, pol, name, "gbt_F95_base_s0", None, None, False,
                          4, rh.LA_CONFIG["time_limit"], gate, None, check)
        print(f"check={check}: {row['duration_h']:.2f} h, {row['rests']} rests, calls at {row['call_stops']}, "
              f"vetoed {row['vetoed']} ({row['check_s']:.0f} s of rollouts), {time.time()-t0:.0f} s")
