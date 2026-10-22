"""One-off (2026-10-22): the route that broke in gate-val2 (rollout gate 10 % +
check, hos_sd after an LA call), replayed with real LA calls without and with
run_hybrid's --shield-la.  Anaconda site-packages appended for gurobipy."""
import os, sys, time
sys.path.append(r"C:\Users\celinep\AppData\Local\anaconda3\Lib\site-packages")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import mixed_eval as me
import run_hybrid as rh
from policy_core import load_policy

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "RlongCfewTmedium_21"
    for shield in (False, True):
        pol = load_policy("gbt", "gbt_F95_base_s0", guard_q=0.99, spread_room=True)
        pol.feature_q, pol.pay_mseq = None, True
        gate = rh.Gate("gbt", 10, nominal=True, model="gate_gbt_ro")
        fd, D, E, cv = me.load(os.path.join(me._ROOT, "instances", name + ".json"))
        fd["title"] = name
        fd["la_energy_quantile"], fd["la_energy_cv"] = float(rh.LA_CONFIG["la_energy_quantile"]), cv
        t0 = time.time()
        row, _ = rh.drive(fd, D, E, cv, pol, name, "gbt_F95_base_s0", None, None, False,
                          8, rh.LA_CONFIG["time_limit"], gate, None, (32, 1.0), shield)
        print(f"shield_la={shield}: completed={row['completed']} violation={row['violation']} "
              f"duration={row['duration_h']} rests={row['rests']} calls={row['call_stops']} "
              f"vetoed={row['vetoed']} unsafe={row['unsafe']} raised={row['raised']} {time.time()-t0:.0f} s",
              flush=True)
