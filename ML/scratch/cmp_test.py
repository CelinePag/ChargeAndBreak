"""Test-split comparison of seed groups:  python cmp_test.py g99sr
Per seed: completion, violations, gap vs LA.  Per group: the route-wise mean
over seeds, paired against the LA and against the trees group."""
import json
import os
import sys

import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
GUARD = sys.argv[1] if len(sys.argv) > 1 else "g99sr"
GROUPS = {"trees F95": [f"gbt_F95_base_s{i}" for i in range(3)],
          "sklearn MLP F95": [f"mlp_F95_base_s{i}" for i in range(3)],
          "torch split+list": [f"tmlp_F95_split_list_s{i}" for i in range(3)]}


def load(tag):
    p = os.path.join(R, f"eval_{tag}_{GUARD}_test.json")
    return {r["instance"]: r for r in json.load(open(p))} if os.path.exists(p) else None


def stats(x):
    return f"{x.mean():+6.2f} ± {x.std(ddof=1) / np.sqrt(len(x)):.2f} (median {np.median(x):+.2f}, n={len(x)})"


print(f"TEST / {GUARD}\n")
print(f"{'model':26s} {'done':>4s} {'viol':>4s} {'mean%LA':>8s} {'med':>6s} {'>5%':>4s} {'rest+':>6s} {'TW':>4s}  halts")
dur = {}
for g, tags in GROUPS.items():
    for t in tags:
        d = load(t)
        if d is None:
            print(f"{t:26s} missing"); continue
        both = [r for r in d.values() if r["route_completed"] and r["LA_completed"] and r["LA"]]
        pct = np.array([100 * (r["duration_h"] / r["LA"] - 1) for r in both])
        halts = sorted({r["halt_reason"] for r in d.values() if not r["route_completed"]})
        print(f"{t:26s} {sum(r['route_completed'] for r in d.values()):4d} "
              f"{sum(r['n_violations'] for r in d.values()):4d} {pct.mean():+8.2f} {np.median(pct):+6.2f} "
              f"{(pct > 5).sum():4d} {np.mean([r['n_rests'] - r['LA_rests'] for r in both]):+6.3f} "
              f"{sum(r['tw_misses'] for r in d.values()):4d}  {','.join(halts)}")
        for k, r in d.items():
            dur.setdefault(g, {}).setdefault(k, []).append(r["duration_h"] if r["route_completed"] else np.nan)
        dur.setdefault("LA", {}).update({k: [r["LA"]] for k, r in d.items() if r["LA_completed"] and r["LA"]})
    print()

# route-wise mean over the 3 seeds (routes every seed completed)
mean = {g: {k: np.mean(v) for k, v in m.items() if len(v) == 3 or g == "LA"} for g, m in dur.items()}
mean = {g: {k: v for k, v in m.items() if np.isfinite(v)} for g, m in mean.items()}
print("seed-averaged, paired (%):")
for g in GROUPS:
    if g not in mean:
        continue
    ks = [k for k in mean[g] if k in mean["LA"]]
    print(f"  {g:18s} vs LA     {stats(np.array([100 * (mean[g][k] / mean['LA'][k] - 1) for k in ks]))}")
    if g != "trees F95":
        ks = [k for k in mean[g] if k in mean["trees F95"]]
        print(f"  {g:18s} vs trees  {stats(np.array([100 * (mean[g][k] / mean['trees F95'][k] - 1) for k in ks]))}")
