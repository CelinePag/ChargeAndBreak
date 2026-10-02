"""Validate the native per-charger support in src against the ML patch results."""
import glob
import json
import os
import sys

import numpy as np

ROOT = r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ML", "code"))
import logging
logging.getLogger("pyomo").setLevel(logging.ERROR)

from src.instance_gen.instance_io import load_instance_json
from src.methods.oracle import oracle_solve
from src.methods.MILP import solve_horizon
from src.simulation.BEHDV import charging_curve_at


def t_of(e, curve, Ebar):
    rs = sorted(Ebar)
    return float(np.interp(e, [Ebar[r] for r in rs], [curve[r] for r in rs]))


def kw(fd, i):
    c = charging_curve_at(fd, i)
    return fd["Ebar"][1] / c[1]


# 1. native oracle vs the stored (patched) oracle on short pmix routes
print("1. native oracle vs stored ML-patched oracle")
for n in ["RshortCfewTnone_22", "RshortCfewTtight_23", "RshortCmanyTnone_24", "RshortCmanyTtight_25"]:
    name = f"{n}__pmix"
    fd, D, E, cv = load_instance_json(os.path.join(ROOT, "ML", "instances_mixed", "pmix", name + ".json"))
    o = oracle_solve(fd, D, time_limit=300, tee=False, verbose=False)
    ref = json.load(open(os.path.join(ROOT, "ML", "solutions_mixed", "pmix", f"oracle_{name}.json")))
    worst = 0.0
    for s in o["sol"]:
        i = s["i"]
        if i in fd["K"] and s.get("tauc", 0) > 1e-4:
            own = t_of(s["ed"], charging_curve_at(fd, i), fd["Ebar"]) - t_of(s["ea"], charging_curve_at(fd, i), fd["Ebar"])
            worst = max(worst, abs(own - s["tauc"]))
    print(f"   {name}: native {o['obj']:.4f} (gap {o['gap']:.4f})  stored {ref['obj']:.4f} (gap {ref['gap']:.4f})"
          f"  rel diff {abs(o['obj'] - ref['obj']) / ref['obj']:.2e}  max|tauc - own curve| {worst:.1e} h")

# 2. LA window: forced charge at a 150 kW station follows that station's curve
print("2. rolling-horizon window, forced charge at a slow and a fast station")
fd, D, E, cv = load_instance_json(os.path.join(ROOT, "ML", "instances_mixed", "pmix", "RmediumCfewTnone_22__pmix.json"))
for target in (150.0, 1000.0):
    st = next(k for k in sorted(fd["K"]) if abs(kw(fd, k) - target) < 1 and k > 5)
    init = dict(ta=8.0 + 2.0, ea=fd["Emin"] + 60.0, cd=1.0, sd=2.0, sw=2.5, phi=0, h=3.0)
    end = min(st + 30, fd["N"])
    sol = solve_horizon(full_data=fd, start_stop=st, end_stop=end, init_state=init,
                        fixed_action=dict(y=1, break_type=None, rest_type=None),
                        tee=False, time_limit=60, relax=False)
    s0 = sol["sol"][0] if isinstance(sol, dict) else sol[0]
    own = t_of(s0["ed"], charging_curve_at(fd, st), fd["Ebar"]) - t_of(s0["ea"], charging_curve_at(fd, st), fd["Ebar"])
    route = t_of(s0["ed"], fd["Tbar"], fd["Ebar"]) - t_of(s0["ea"], fd["Tbar"], fd["Ebar"])
    print(f"   station {st} ({kw(fd, st):.0f} kW): tauc {s0['tauc']:.4f} h  own curve {own:.4f} h"
          f"  route-wide (slowest) curve {route:.4f} h  charged {s0['ed'] - s0['ea']:.1f} kWh")
