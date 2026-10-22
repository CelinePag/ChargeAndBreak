"""In-memory checks (persist=False, logs to the scratchpad): nothing is written to solutions/.

1. rules="milp" reproduces the stored CORR runs exactly.
2. rules="bay": per instance, route duration at 350 / 700 / 1000 kW; count increases.
"""
import contextlib, glob, io, json, os, sys, tempfile
import src.paths as P
from src.instance_gen.instance_io import load_instance_json
from src.methods.greedy_corrected import run_greedy_corrected
from src.methods.greedy import run_greedy

# keep throwaway logs out of the repo
_tmp = tempfile.mkdtemp(prefix="greedyc_test_")
P.logs = (lambda orig: (lambda *a: _tmp if a and a[0] == "_internal" else orig(*a)))(P.logs)

KW = dict(guard_quantile=0.95, safety_buffer=0.1, verbose=False, persist=False,
          prune_quantile=0.95, oracle_tee=False)

def run(path, fn, **extra):
    fd, D, E, cv = load_instance_json(path)
    fd["title"] = os.path.splitext(os.path.basename(path))[0]
    with contextlib.redirect_stdout(io.StringIO()):
        r = fn(fd, D, E, cv=cv, **KW, **extra)
    return None if r["metrics"]["run_infeasible"] else r["total_time"] - 8.0

n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
# 1. regression of the milp rule set against stored CORR runs
ok = tot = 0
for f in sorted(glob.glob("instances_sens/charger_power_1000/*.json"))[:n]:
    stem = os.path.splitext(os.path.basename(f))[0]
    st = sorted(P.glob_solutions(f"{stem}_GREEDY_CORR_2*.json"))
    if not st:
        continue
    sd = json.load(open(st[-1])).get("duration_h")
    d = run(f, run_greedy_corrected, rules="milp")
    if sd is not None and d is not None:
        tot += 1; ok += abs(sd - d) < 1e-9
print(f"milp rules reproduce stored CORR runs: {ok}/{tot}")

# 2. monotonicity in power, bay vs submitted greedy
stems = sorted(os.path.splitext(os.path.basename(f))[0]
               for f in glob.glob("instances/*.json"))[:: max(1, 900 // n)][:n]
res = {"bay": [], "old": []}
for stem in stems:
    paths = {350: f"instances/{stem}.json",
             700: f"instances_sens/charger_power_700/{stem}__kw700.json",
             1000: f"instances_sens/charger_power_1000/{stem}__kw1000.json"}
    if not all(os.path.exists(p) for p in paths.values()):
        continue
    for tag, fn, ex in (("bay", run_greedy_corrected, dict(rules="bay")),
                        ("old", run_greedy, {})):
        res[tag].append({kw: run(p, fn, **ex) for kw, p in paths.items()})
for tag, rows in res.items():
    worse = {(a, b): 0 for a, b in ((350, 700), (700, 1000))}
    n_ok = {(a, b): 0 for a, b in worse}
    for r in rows:
        for a, b in worse:
            if r[a] is not None and r[b] is not None:
                n_ok[(a, b)] += 1
                worse[(a, b)] += r[b] > r[a] + 1e-6
    print(tag, {f"{a}->{b}": f"{worse[(a,b)]}/{n_ok[(a,b)]} worse" for a, b in worse})
