"""Write LP files for full-route and window models of a few instances.

usage: python lp_snapshot.py <out_dir>
Run once before and once after the per-charger change; the files must be
identical for every instance that has no per-charger curves.
"""
import os
import sys

ROOT = r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak"
sys.path.insert(0, ROOT)
import logging
logging.getLogger("pyomo").setLevel(logging.ERROR)

from src.instance_gen.instance_io import load_instance_json
from src.methods.MILP import build_model, build_horizon_model, make_subproblem_data

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
CASES = [
    ("base_short", os.path.join(ROOT, "instances", "RshortCfewTnone_22.json")),
    ("base_medium_tight", os.path.join(ROOT, "instances", "RmediumCmanyTtight_23.json")),
    ("kw150_medium", os.path.join(ROOT, "instances_sens", "charger_power_150", "RmediumCfewTnone_22__kw150.json")),
    ("kwh300_short", os.path.join(ROOT, "instances_sens", "battery_300", "RshortCmanyTtight_24__kwh300.json")),
]
for tag, path in CASES:
    fd, D, E, cv = load_instance_json(path)
    m = build_model(fd)
    m.write(os.path.join(out, f"{tag}_full.lp"), io_options={"symbolic_solver_labels": True})
    for start in (3, 20):
        end = min(start + 30, fd["N"])
        init = dict(ta=8.0 + start, ea=0.6 * fd["Ecap"], cd=1.0, sd=2.0, sw=2.5, phi=0, h=3.0)
        sub = make_subproblem_data(fd, start, end, init)
        hm = build_horizon_model(sub, init, fixed_action=dict(y=1, break_type=None, rest_type=None))
        hm.write(os.path.join(out, f"{tag}_win{start}.lp"), io_options={"symbolic_solver_labels": True})
    print(tag, "ok", flush=True)
