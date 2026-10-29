"""Numbers for the CPAIOR draft 5 (rollout / expert iteration), from stored runs.

Part A (no simulation): per test run, missed windows on the stored
realisations, changed decisions, wall time per route by length, broken runs;
r2 + rollout vs original + rollout (seed 0), paired, manuscript gap with penalty.
Part B (--decisive): how often the decisive-state rule fires for the PyTorch
student (seed 0) on the stored realisations of the 125 test routes.
"""
import json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
sys.path.insert(0, CODE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
RES = os.path.join(HERE, "..", "results")


def load(name):
    return [json.loads(l) for l in open(os.path.join(RES, f"rl_search_{name}.jsonl")) if l.strip()]


def ln(inst):
    return "short" if inst.startswith("Rshort") else "medium" if inst.startswith("Rmedium") else "long"


def part_a():
    groups = dict(
        torch=[f"dsr_test_tmlp_F95_split_list_s{s}" for s in range(3)],
        trees=[f"dsr_test_gbt_F95_base_s{s}" for s in range(3)],
        exit2=["exit_test_r2"],
        mixed=[f"dsr_mtest_tmlp_T144_physmix_split_list_s{s}" for s in range(3)],
        mixed_eq=["dsr_mtest_tmlp_T144_phys_split_list_s0"],
    )
    for g, runs in groups.items():
        miss = defaultdict(list); changed = []; secs = defaultdict(list); dec = []; broken = 0
        rests = defaultdict(list)
        for run in runs:
            recs = load(run)
            names = recs[0]["variants"]
            tot = defaultdict(int)
            for r in recs:
                for v, nm in enumerate(names):
                    x = r["res"][v][0]
                    broken += int(x["failed"])
                    tot[nm] += x["misses"]
                    rests[nm].append(x["rests"])
                changed.append(r["res"][1][0]["changed"])
                dec.append(r["res"][1][0]["decisions"])
                secs[ln(r["instance"])].append(r["seconds"])
            for nm, t in tot.items():
                miss[nm].append(t)
        print(f"== {g}: {len(recs)} routes x {len(runs)} run(s), broken runs {broken}")
        for nm, t in miss.items():
            print(f"   missed windows {nm:40s} mean over seeds {np.mean(t):6.1f}  ({t})")
        print(f"   rollout: changed decisions per route {np.mean(changed):.2f} of {np.mean(dec):.1f} "
              f"({100*np.sum(changed)/np.sum(dec):.1f} %)")
        for k in ("short", "medium", "long"):
            if secs[k]:
                print(f"   seconds per route {k:6s} median {np.median(secs[k]):6.0f}  mean {np.mean(secs[k]):6.0f}")


def part_a_pair():
    """r2 + rollout minus original + rollout (seed 0), base test, gap with penalty."""
    from paper_link import gaps, oracle_facts
    T0 = 8.0
    a = {r["instance"]: r for r in load("exit_test_r2")}
    b = {r["instance"]: r for r in load("dsr_test_tmlp_F95_split_list_s0")}
    d, dd = [], []
    for inst in sorted(set(a) & set(b)):
        orc = oracle_facts(inst)
        if orc is None:
            continue
        x, y = a[inst]["res"][1][0], b[inst]["res"][1][0]
        if x["failed"] or y["failed"]:
            continue
        gx = gaps(x["duration"], x["misses"], x["duration"] + T0, orc)
        gy = gaps(y["duration"], y["misses"], y["duration"] + T0, orc)
        d.append(gx[1] - gy[1]); dd.append(gx[0] - gy[0])
    d = np.array(d); dd = np.array(dd)
    se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
    print(f"== r2+rollout minus orig+rollout (s0), n={len(d)}: pen {d.mean():+.2f} ± {se(d):.2f} pp, "
          f"duration {dd.mean():+.2f} ± {se(dd):.2f} pp, better/worse {(d < -1e-9).sum()}/{(d > 1e-9).sum()}")


def part_b():
    from rl_env import final_student, load_route, realised, run_batch, split_routes
    from rl_api import decisive
    from features import Precomp
    base = final_student("torch", "tmlp_F95_split_list_s0")
    tot = defaultdict(lambda: [0, 0, 0])            # length -> [decisions, decisive, routes]
    for inst in split_routes("test"):
        fd, D, E, cv = load_route(inst)
        cnt = [0, 0]

        def hook(i, veh, legal, rows, score, flags, j):
            cnt[0] += 1
            kind, _ = decisive(legal, score, 2.0, 0.25)
            cnt[1] += int(kind is not None)
        run_batch(base, fd, cv, [(None, realised(D, E))], pre=Precomp(fd), on_decision=hook)
        t = tot[ln(inst)]
        t[0] += cnt[0]; t[1] += cnt[1]; t[2] += 1
    alld = sum(t[0] for t in tot.values()); alls = sum(t[1] for t in tot.values())
    for k, t in tot.items():
        print(f"   {k:6s}: {t[1]/t[2]:.1f} decisive of {t[0]/t[2]:.1f} decisions per route ({100*t[1]/t[0]:.1f} %)")
    print(f"   all: {100*alls/alld:.1f} % of decisions decisive")


if __name__ == "__main__":
    if "--decisive" in sys.argv:
        part_b()
    else:
        part_a()
        part_a_pair()
