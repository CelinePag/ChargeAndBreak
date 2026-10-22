"""In-memory: bayref on the 26 v2 non-monotone pairs, and bayref == bay at 350 kW."""
import contextlib, glob, io, json, os, tempfile
import src.paths as P
from src.instance_gen.instance_io import load_instance_json
from src.methods.greedy_corrected import run_greedy_corrected
_tmp = tempfile.mkdtemp(prefix="greedyc_test_")
P.logs = (lambda orig: (lambda *a: _tmp if a and a[0] == "_internal" else orig(*a)))(P.logs)
KW = dict(guard_quantile=0.95, safety_buffer=0.1, verbose=False, persist=False,
          prune_quantile=0.95, oracle_tee=False)
def path(stem, tag):
    if tag in (None, "None"):
        return f"instances/{stem}.json"
    hits = glob.glob(f"instances_sens/*/{stem}__{tag}.json")
    return hits[0]
def run(p, rules):
    fd, D, E, cv = load_instance_json(p)
    fd["title"] = os.path.splitext(os.path.basename(p))[0]
    with contextlib.redirect_stdout(io.StringIO()):
        r = run_greedy_corrected(fd, D, E, cv=cv, rules=rules, **KW)
    return None if r["metrics"]["run_infeasible"] else r["total_time"] - 8.0
cases = json.load(open("ML/scratch/greedyc2_nonmono.json"))
worse = 0
for stem, lo, hi in cases:
    a, b = run(path(stem, lo), "bayref"), run(path(stem, hi), "bayref")
    w = (a is not None and b is not None and b > a + 1e-9)
    worse += w
    print(f"{stem:<24} {str(lo):<14}->{hi:<14} bayref {a:.3f} -> {b:.3f}  {'WORSE' if w else 'ok'}")
print("bayref worse:", worse, "/", len(cases))
# bayref == bay at 350 kW (base) on a sample of stored CORR2 runs
same = tot = 0
for f in sorted(glob.glob("instances/RmediumC*_*.json"))[:20]:
    stem = os.path.splitext(os.path.basename(f))[0]
    st = sorted(P.glob_solutions(f"{stem}_GREEDY_CORR2_2*.json"))
    if not st: continue
    sd = json.load(open(st[-1])).get("duration_h")
    d = run(f, "bayref")
    if sd is not None and d is not None:
        tot += 1; same += abs(sd - d) < 1e-9
print(f"bayref reproduces CORR2 at 350 kW: {same}/{tot}")
