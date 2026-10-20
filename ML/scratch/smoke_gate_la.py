"""One-off smoke test (2026-10-20): the gate hybrid end to end WITH real LA
calls on one short validation route, on the laptop.  The .venv has torch and
lightgbm but no gurobipy; Anaconda (the venv's base interpreter, same ABI) has
gurobipy and pyomo.  Its site-packages is APPENDED to sys.path, so every
package the venv has still wins; nothing is installed or changed.  The HPC
stage gate-val runs the real thing in one environment."""
import os
import sys
import time

sys.path.append(r"C:\Users\celinep\AppData\Local\anaconda3\Lib\site-packages")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

import mixed_eval as me                                            # noqa: E402
import run_hybrid as rh                                            # noqa: E402
from policy_core import load_policy                                # noqa: E402

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "RshortCfewTnone_20"
    path = os.path.join(me._ROOT, "instances", name + ".json")
    pol = load_policy("gbt", "gbt_F95_base_s0", guard_q=0.99, spread_room=True)
    pol.feature_q, pol.pay_mseq = None, True
    gate = rh.Gate("gbt", 5, nominal=True)
    fd, D, E, cv = me.load(path)
    fd["title"] = name
    fd["la_energy_quantile"], fd["la_energy_cv"] = float(rh.LA_CONFIG["la_energy_quantile"]), cv
    t0 = time.time()
    row, _ = rh.drive(fd, D, E, cv, pol, name, "gbt_F95_base_s0", None, None, False,
                      4, rh.LA_CONFIG["time_limit"], gate, None)
    print(row)
    print(f"{time.time() - t0:.0f} s")
