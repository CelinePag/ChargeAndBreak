"""
rl_compare.py — paired comparison of the policies of one rl_search run, on fresh draws
======================================================================================
For an rl_search run whose variants share the same travel-time draws (the
run's student = variant 0, the others e.g. {"net": "torch:<another student>"}),
report each other variant minus variant 0 on the FRESH draws (the stored
realisation is skipped if present): mean route cost difference per route
(draws averaged within a route first) +- one standard error over routes, in
hours and in % of variant 0's cost, routes better / worse, missed windows
and daily rests per route, overall and by route length.  Draws where either
policy broke a rule are counted and left out.  Writes
results/rl_compare_<run>.json.

    python ML/code/rl_compare.py exit_test_fresh_r2
    python ML/code/rl_compare.py win_val_torch --ref 1   # window rollout minus the others
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main(run, ref=0):
    recs = [json.loads(l) for l in open(os.path.join(RESULTS, f"rl_search_{run}.jsonl"))
            if l.strip()]
    names = recs[0]["variants"]
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    broken = 0
    for r in recs:
        lo = 1 if r.get("realised_first") else 0
        ln = ("short" if r["instance"].startswith("Rshort") else
              "medium" if r["instance"].startswith("Rmedium") else "long")
        base = r["res"][ref][lo:]
        for v in range(len(names)):
            if v == ref:
                continue
            other = r["res"][v][lo:]
            ok = [i for i in range(len(base)) if not (base[i]["failed"] or other[i]["failed"])]
            broken += len(base) - len(ok)
            if not ok:
                continue
            d = [base[i]["cost"] - other[i]["cost"] for i in ok]
            p = [100 * (base[i]["cost"] - other[i]["cost"]) / other[i]["cost"] for i in ok]
            m = [base[i]["misses"] - other[i]["misses"] for i in ok]
            s = [base[i]["rests"] - other[i]["rests"] for i in ok]
            for k in ("all", ln):
                o = out[names[v]][k]
                o["h"].append(np.mean(d))
                o["pct"].append(np.mean(p))
                o["miss"].append(np.mean(m))
                o["rests"].append(np.mean(s))
    nd = len(recs[0]["res"][0]) - (1 if recs[0].get("realised_first") else 0)
    print(f"{run}: {len(recs)} routes x {nd} fresh draws; {names[ref]} minus each other policy, "
          f"paired; {broken} draws with a broken run left out")
    js = {}
    for vn, byk in out.items():
        print(f"\n{names[ref]}  minus  {vn}:")
        for k in ("all", "short", "medium", "long"):
            o = byk.get(k)
            if not o:
                continue
            h, pc = np.array(o["h"]), np.array(o["pct"])
            se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            print(f"  {k:6s} n={len(h):3d}: {h.mean():+.3f} ± {se(h):.3f} h per route "
                  f"({pc.mean():+.2f} ± {se(pc):.2f} %), better / worse {int((h < -1e-9).sum())} / "
                  f"{int((h > 1e-9).sum())}, missed windows {np.mean(o['miss']):+.2f}, "
                  f"rests {np.mean(o['rests']):+.3f} per route")
            js.setdefault(vn, {})[k] = dict(n=len(h), h=[float(h.mean()), se(h)],
                                            pct=[float(pc.mean()), se(pc)],
                                            better=int((h < -1e-9).sum()), worse=int((h > 1e-9).sum()),
                                            misses=float(np.mean(o["miss"])),
                                            rests=float(np.mean(o["rests"])))
    with open(os.path.join(RESULTS, f"rl_compare_{run}.json"), "w") as fh:
        json.dump(dict(run=run, base=names[ref], n_draws=nd, broken=broken, vs=js), fh, indent=1)


if __name__ == "__main__":
    # rl_compare.py <run> [--ref k]: k = index of the reference variant (default 0)
    k = int(sys.argv[sys.argv.index("--ref") + 1]) if "--ref" in sys.argv else 0
    main(sys.argv[1], ref=k)
