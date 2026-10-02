"""Direction B vs pilot data, on VALIDATION routes only.

1. mixed-power validation routes (16, seeds 20-21): per method, the paired
   change mixed - uniform on the same route, and the paired difference on the
   mixed routes vs the base-only trees.
2. uniform base-case validation routes (66, stop split): paired vs the LA.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import mixed_eval as me                                         # noqa: E402

me.ROUTE_SET = "val"
rows = me.read_rows(me.store_path(0.99, True))
orc = {}
G = {}
for r in rows:
    k = (r["variant"], r["instance"])
    if k not in orc:
        orc[k] = me.read_oracle(*k)
    g = me._gap(r, orc[k]) if r["completed"] else None
    if g is not None and orc[k] and (orc[k]["gap"] or 0) <= 0.01:
        G[(r["method"], r["variant"], r["instance"].split("__")[0])] = (g, r["rests"], orc[k]["rests"])

methods = ["Greedy"] + [lab for _k, _t, lab in me.MODELS]
bases = sorted({b for (_m, v, b) in G if v == "pmix"})
ref = "trees, base only"


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


print(f"1. MIXED-POWER VALIDATION ROUTES ({len(bases)} with certified oracles)\n")
print(f"{'method':38s} {'uniform':>8s} {'mixed':>8s} {'change (pp)':>15s} "
      f"{'mixed vs base trees':>20s} {'+rest':>6s}")
for m in methods:
    ch, vs, u, x, extra = [], [], [], [], 0
    for b in bases:
        a, c = G.get((m, "uniform", b)), G.get((m, "pmix", b))
        if a and c:
            ch.append(c[0] - a[0])
            u.append(a[0])
            x.append(c[0])
            extra += int(c[1] > c[2])
        t = G.get((ref, "pmix", b))
        if c and t:
            vs.append(c[0] - t[0])
    if not ch:
        continue
    print(f"{m:38s} {np.mean(u):+8.2f} {np.mean(x):+8.2f} {np.mean(ch):+7.2f} ± {se(ch):4.2f} "
          f"{np.mean(vs):+11.2f} ± {se(vs):4.2f} {extra:6d}")

print("\n2. UNIFORM BASE-CASE VALIDATION ROUTES (stop split, g99sr), paired vs the LA\n")
R = os.path.join(HERE, "..", "results")
tags = ["gbt_F95_base_s0", "tmlp_F95_split_list_s0", "gbt_F95_phys_s1",
        "tmlp_F95_phys_split_list_s0", "tmlp_T144_phys_split_list_s0",
        "tmlp_T144_phys_charger_s0", "tmlp_T144_phys_chargerG_s0",
        "gbt_P102_physpmix_s1", "gbt_P102_physpmix_w5_s1",
        "tmlp_T144_physpmix_split_list_s0", "tmlp_T144_physpmix_charger_s0"]
print(f"{'model':36s} {'done':>4s} {'viol':>4s} {'vs LA %':>14s} {'median':>7s} {'+rest':>6s}")
for t in tags:
    p = os.path.join(R, f"eval_{t}_g99sr_stop.json")
    if not os.path.exists(p):
        print(f"{t:36s} missing")
        continue
    d = json.load(open(p))
    both = [r for r in d if r["route_completed"] and r["LA_completed"] and r["LA"]]
    pct = np.array([100 * (r["duration_h"] / r["LA"] - 1) for r in both])
    print(f"{t:36s} {sum(r['route_completed'] for r in d):4d} {sum(r['n_violations'] for r in d):4d} "
          f"{pct.mean():+7.2f} ± {se(pct):4.2f} {np.median(pct):+7.2f} "
          f"{np.mean([r['n_rests'] - r['LA_rests'] for r in both]):+6.3f}")
