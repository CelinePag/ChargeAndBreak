"""After the src change: Greedy (native per-charger path, no swap) and the
students must reproduce every stored mixed-route row; Greedy must still
match the stored base-case runs."""
import glob
import json
import os
import sys

ROOT = r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ML", "code"))
from mixed_eval import instances, load, read_rows, run_greedy_ml, store_path
from policy_core import load_policy, run_student

rows = {(r["method"], r["instance"]): r for r in read_rows(store_path(0.99, True))}
same = diff = 0
for v, name, path in instances():
    fd, D, E, cv = load(path)
    g = run_greedy_ml(fd, D, E, cv)
    ref = rows[("Greedy", name)]
    ok = (g["duration_h"] == ref["duration_h"] and g["tw_misses"] == ref["tw"]
          and g["route_completed"] == ref["completed"])
    same += ok; diff += (not ok)
    if not ok:
        print("GREEDY DIFF", name, g["duration_h"], ref["duration_h"])
print(f"greedy on uniform + mixed routes: {same} identical, {diff} different")

pol = load_policy("gbt", "gbt_P102_phys_s1", guard_q=0.99, spread_room=True)
same = diff = 0
for v, name, path in instances(("pmix", "mix")):
    fd, D, E, cv = load(path)
    r = run_student(fd, D, E, pol, cv=cv)
    ref = rows[("trees, all physics + power", name)]
    ok = r["duration_h"] == ref["duration_h"] and r["tw_misses"] == ref["tw"]
    same += ok; diff += (not ok)
print(f"power-aware student on pmix + mix: {same} identical, {diff} different")

for n in ["RshortCfewTnone_22", "RmediumCmanyTtight_25", "RlongCfewTnone_22"]:
    fd, D, E, cv = load(os.path.join(ROOT, "instances", n + ".json"))
    g = run_greedy_ml(fd, D, E, cv)
    s = json.load(open(sorted(glob.glob(os.path.join(ROOT, "solutions", "basecase", f"{n}_GREEDY_*.json")))[-1]))
    print("stored base greedy", n, g["duration_h"] == s["duration_h"])
