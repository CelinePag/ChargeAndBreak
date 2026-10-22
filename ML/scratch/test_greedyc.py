"""In-memory comparison of greedy vs greedy_corrected (persist=False: nothing written)."""
import glob, json, sys, io, contextlib, os
from collections import Counter
from src.instance_gen.instance_io import load_instance_json
from src.methods.greedy import run_greedy
from src.methods.greedy_corrected import run_greedy_corrected
from src import paths as P

pat, n = sys.argv[1], int(sys.argv[2])
diesel = len(sys.argv) > 3 and sys.argv[3] == "diesel"
files = sorted(glob.glob(pat))[:n]
match = 0; tot = 0; d_old = []; d_new = []; viol = Counter(); seqs = Counter()
for f in files:
    fd, D, E, cv = load_instance_json(f)
    stem = os.path.splitext(os.path.basename(f))[0]
    fd['title'] = stem
    if diesel:
        from src.simulation.runner_dispatch import _apply_diesel_mode
        fd, D, E = _apply_diesel_mode(fd, D, E)
    kw = dict(cv=cv, guard_quantile=0.95, safety_buffer=0.1, verbose=False,
              persist=False, prune_quantile=0.95, oracle_tee=False)
    with contextlib.redirect_stdout(io.StringIO()):
        a = run_greedy(fd, D, E, **kw)
        b = run_greedy_corrected(fd, D, E, **kw)
    ta = a['total_time'] - 8 if not a['metrics']['run_infeasible'] else None
    tb = b['total_time'] - 8 if not b['metrics']['run_infeasible'] else None
    stored = [p for p in P.glob_solutions(f"{stem}_GREEDY_2*.json")]
    sd = None
    if stored:
        js = json.load(open(sorted(stored)[-1]))
        sd = js.get('duration_h')
    if sd is not None and ta is not None:
        tot += 1; match += abs(sd - ta) < 1e-6
    for k, v in (b['metrics'].get('violations_by_type') or {}).items(): viol['new:'+k] += 1
    for k, v in (a['metrics'].get('violations_by_type') or {}).items(): viol['old:'+k] += 1
    for du, ac in zip(b['durations_list'], b['actions']):
        if du.get('sigma'):
            seqs['rest' if ac.get('rest_type') else 'break'] += 1
    if ta and tb: d_old.append(ta); d_new.append(tb)
print(f"files {len(files)}  old-greedy reproduces stored duration: {match}/{tot}")
if d_old:
    import statistics as st
    rel = [100*(b/a-1) for a, b in zip(d_old, d_new)]
    print(f"paired feasible {len(rel)}: mean change {st.mean(rel):+.2f}%  min {min(rel):+.2f}  max {max(rel):+.2f}  unchanged {sum(abs(x)<1e-9 for x in rel)}")
print('sequential stops (new):', dict(seqs)); print('violations:', dict(viol))
