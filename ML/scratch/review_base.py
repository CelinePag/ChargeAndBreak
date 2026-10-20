"""Independent re-check of the base-case headline (external review, 2026-10-20).
Reads only ML/results eval files + solutions/basecase oracle/LA files."""
import json, glob, os, sys, collections
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "ML", "results")
SOL = os.path.join(ROOT, "solutions", "basecase")
BETA = 0.5

def load(tag):
    return {r["instance"]: r for r in json.load(open(os.path.join(RES, f"eval_{tag}_test.json")))}

def oracle(inst):
    p = os.path.join(SOL, f"oracle_{inst}.json")
    if not os.path.exists(p): return None
    return json.load(open(p))

def report(tags, name):
    runs = [load(t) for t in tags]
    insts = sorted(set.intersection(*[set(r) for r in runs]))
    d_dur, d_pen, tw_s, tw_la, n_inc = [], [], 0.0, 0, 0
    rel_dur, rel_pen = [], []
    for i in insts:
        rs = [r[i] for r in runs]
        if not all(x["route_completed"] for x in rs) or rs[0].get("LA") is None:
            n_inc += 1; continue
        la, la_tw = rs[0]["LA"], rs[0]["LA_tw"]
        s_dur = np.mean([x["duration_h"] for x in rs])
        s_tw = np.mean([x["tw_misses"] for x in rs])
        d_dur.append(s_dur - la); d_pen.append(s_dur + BETA*s_tw - la - BETA*la_tw)
        rel_dur.append(100*(s_dur-la)/la); rel_pen.append(100*(s_dur + BETA*s_tw - la - BETA*la_tw)/(la+BETA*la_tw))
        tw_s += s_tw; tw_la += la_tw
    def ci(x):
        x = np.asarray(x); m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x))
        return f"mean {m:+.3f} se {se:.3f} 95%CI [{m-1.96*se:+.3f},{m+1.96*se:+.3f}] median {np.median(x):+.3f}"
    print(f"== {name}: n={len(d_dur)} paired routes (incomplete {n_inc})")
    print("  rel duration %     ", ci(rel_dur))
    print("  rel objective %    ", ci(rel_pen), "(duration + 0.5h*TW miss)")
    print("  abs duration h     ", ci(d_dur))
    print("  abs objective h    ", ci(d_pen))
    print(f"  TW misses: student (seed-mean) {tw_s:.1f}  LA {tw_la}")
    x = np.asarray(rel_dur)
    print(f"  wins/ties/losses vs LA (|d|<0.05%): {np.sum(x<-0.05)}/{np.sum(abs(x)<=0.05)}/{np.sum(x>0.05)}; top5 losses {np.sort(x)[-5:].round(2)}")
    # bootstrap over routes
    rng = np.random.default_rng(0); xs = np.asarray(rel_pen)
    b = [rng.choice(xs, len(xs)).mean() for _ in range(5000)]
    print(f"  bootstrap 95% CI rel objective: [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]")
    return insts

for nm, tags in [("torch split_list g99sr", [f"tmlp_F95_split_list_s{s}_g99sr" for s in range(3)]),
                 ("gbt F95 g99sr", [f"gbt_F95_base_s{s}_g99sr" for s in range(3)])]:
    try: report(tags, nm)
    except FileNotFoundError as e: print("missing", e)

# test-set coverage
fam = collections.Counter()
allinst = [os.path.basename(p)[7:-5] for p in glob.glob(os.path.join(SOL, "oracle_*.json"))]
test = [i for i in allinst if 22 <= int(i.rsplit("_",1)[1]) <= 25]
ev = load("tmlp_F95_split_list_s0_g99sr")
missing = sorted(set(test) - set(ev))
print(f"\noracle files at test seeds: {len(test)}; evaluated: {len(ev)}; missing {len(missing)}")
for i in missing:
    la = sorted(glob.glob(os.path.join(SOL, f"{i}_LA_MIPTAIL_*.json")))
    s = json.load(open(la[-1])) if la else None
    print("  ", i, "LA file" if la else "NO LA", (s.get("metrics",{}).get("run_infeasible"), s.get("duration_h")) if s else "")
# oracle quality on test
gs = []; opt = 0
for i in ev:
    o = oracle(i); gs.append(o.get("gap") or 0); opt += bool(o.get("optimal"))
gs = np.array(gs)
print(f"oracle MIP gap on evaluated test routes: median {np.median(gs)*100:.2f}% max {gs.max()*100:.2f}%  >1%: {(gs>0.01).sum()}  proven optimal {opt}")
