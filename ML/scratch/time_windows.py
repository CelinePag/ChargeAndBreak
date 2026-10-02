"""Solve time of the LA's window MIP: uniform route vs its pmix version."""
import os
import sys
import time

ROOT = r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak"
sys.path.insert(0, ROOT)
import logging
logging.getLogger("pyomo").setLevel(logging.ERROR)
from src.instance_gen.instance_io import load_instance_json
from src.methods.MILP import solve_horizon

pairs = [("uniform", os.path.join(ROOT, "instances", "RshortCfewTnone_22.json")),
         ("pmix", os.path.join(ROOT, "ML", "instances_mixed", "pmix", "RshortCfewTnone_22__pmix.json"))]
data = {k: load_instance_json(p) for k, p in pairs}
for start in (2, 10, 20):
    for k in ("uniform", "pmix"):
        fd, D, E, cv = data[k]
        init = dict(ta=8.5 + 0.3 * start, ea=0.85 * fd["Ecap"] - 10 * start, cd=0.5, sd=0.5 + 0.2 * start,
                    sw=0.5 + 0.2 * start, phi=0, h=0.5 + 0.2 * start)
        end = min(start + 42, fd["N"])
        t0 = time.time()
        r = solve_horizon(full_data=fd, start_stop=start, end_stop=end, init_state=init,
                          fixed_action=dict(y=0, break_type=None, rest_type=None),
                          tee=False, time_limit=300, relax=False)
        print(f"start {start:2d} {k:8s} {time.time() - t0:6.1f}s  obj {r['obj']:.3f}  feasible {r['feasible']}", flush=True)
