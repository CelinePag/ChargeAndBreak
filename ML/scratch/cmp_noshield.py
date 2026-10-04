"""Shield on vs off, test split:  python ML/scratch/cmp_noshield.py
Per model x variant: runs completed of 375, which rule broke, gap vs LA on
the routes completed (per seed, then paired on routes all three seeds
completed), and how often the unshielded model picked an action the 0.99
guard would have blocked."""
import json
import os
from collections import Counter

import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = {"torch": "tmlp_F95_split_list_s{}", "trees": "gbt_F95_base_s{}"}
VARIANTS = {"g99sr": "shield (headline)", "g99srN": "shield, nominal features",
            "nsN": "NO shield"}


def load(tag, v):
    p = os.path.join(R, f"eval_{tag}_{v}_test.json")
    return json.load(open(p)) if os.path.exists(p) else None


for m, pat in MODELS.items():
    print(f"=== {m}")
    for v, name in VARIANTS.items():
        runs = [load(pat.format(s), v) for s in range(3)]
        if any(r is None for r in runs):
            print(f"  {name:26s} missing"); continue
        allr = [r for rr in runs for r in rr]
        done = [r for r in allr if r["route_completed"]]
        halts = Counter(r["halt_reason"] for r in allr if not r["route_completed"])
        pct = np.array([100 * (r["duration_h"] / r["LA"] - 1) for r in done
                        if r.get("LA_completed") and r.get("LA")])
        nd = sum(r["decisions"] for r in allr)
        uns = sum(r.get("n_unsafe", 0) for r in allr)
        unsafe_routes = sum(1 for r in allr if r.get("n_unsafe", 0))
        # paired on routes completed by all three seeds
        by = {}
        for rr in runs:
            for r in rr:
                by.setdefault(r["instance"], []).append(r)
        paired = [100 * (np.mean([r["duration_h"] for r in v3]) / v3[0]["LA"] - 1)
                  for v3 in by.values()
                  if all(r["route_completed"] for r in v3) and v3[0].get("LA")]
        pa = np.array(paired)
        print(f"  {name:26s} done {len(done):3d}/{len(allr)}  "
              f"gap vs LA {pct.mean():+5.2f}% (med {np.median(pct):+5.2f})  "
              f"seed-avg on {len(pa)} routes {pa.mean():+5.2f}±{pa.std(ddof=1)/np.sqrt(len(pa)):.2f}  "
              f"TW {sum(r['tw_misses'] for r in allr)}")
        if halts:
            print(f"  {'':26s} halts: " + ", ".join(f"{k}={n}" for k, n in halts.most_common()))
        if v == "nsN":
            print(f"  {'':26s} picks the 0.99 guard would block: {uns}/{nd} "
                  f"({100*uns/nd:.2f}%), on {unsafe_routes} runs; "
                  f"short charges {sum(r.get('n_short_charge', 0) for r in allr)}")
    print()
