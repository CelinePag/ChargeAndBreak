"""Direction B on the TEST routes (looked at once, 2026-10-03).

1. mixed test routes (pmix / dmix / mix, seeds 22-25): per method, paired
   change mixed - uniform on the same route, and the mixed gap vs base trees.
   ChargerNet + pilot is also shown averaged over its 3 seeds.
2. the 125 base-case test routes: paired vs the LA.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import mixed_eval as me                                         # noqa: E402

rows = me.read_rows(me.store_path(0.99, True))
orc, G = {}, {}
for r in rows:
    k = (r["variant"], r["instance"])
    if k not in orc:
        orc[k] = me.read_oracle(*k)
    g = me._gap(r, orc[k]) if r["completed"] else None
    if g is not None and orc[k] and (orc[k]["gap"] or 0) <= 0.01:
        G[(r["method"], r["variant"], r["instance"].split("__")[0])] = (g, r["rests"], orc[k]["rests"])

SEEDS = ["ChargerNet, + pilot", "ChargerNet, + pilot (s1)", "ChargerNet, + pilot (s2)"]
for v in ("pmix", "dmix", "mix"):            # seed-averaged ChargerNet + pilot
    for b in {b for (_m, vv, b) in G if vv == v}:
        for vv in ("uniform", v):
            got = [G[(m, vv, b)] for m in SEEDS if (m, vv, b) in G]
            if len(got) == 3:
                G[("ChargerNet + pilot, 3 seeds", vv, b)] = (
                    np.mean([x[0] for x in got]), np.mean([x[1] for x in got]), got[0][2])

methods = (["Greedy"] + [lab for _k, _t, lab in me.MODELS if lab not in SEEDS[1:]]
           + ["ChargerNet + pilot, 3 seeds"])
ref = "trees, base only"


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


for v in ("pmix", "dmix", "mix"):
    bases = sorted({b for (_m, vv, b) in G if vv == v})
    print(f"\n{v.upper()} TEST ROUTES ({len(bases)} with certified oracles)")
    print(f"{'method':38s} {'uniform':>8s} {'mixed':>8s} {'change (pp)':>15s} "
          f"{'mixed vs base trees':>20s} {'+rest':>6s}")
    for m in methods:
        ch, vs, u, x, extra = [], [], [], [], 0
        for b in bases:
            a, c = G.get((m, "uniform", b)), G.get((m, v, b))
            if a and c:
                ch.append(c[0] - a[0]); u.append(a[0]); x.append(c[0])
                extra += int(c[1] > c[2] + 0.5)
            t = G.get((ref, v, b))
            if c and t:
                vs.append(c[0] - t[0])
        if len(ch) < 2:
            continue
        print(f"{m:38s} {np.mean(u):+8.2f} {np.mean(x):+8.2f} {np.mean(ch):+7.2f} ± {se(ch):4.2f} "
              f"{np.mean(vs):+11.2f} ± {se(vs):4.2f} {extra:6d}")

print("\nBASE-CASE TEST ROUTES (125, g99sr), paired vs the LA")
R = os.path.join(HERE, "..", "results")
groups = {"trees F95, base (3 seeds)": [f"gbt_F95_base_s{i}" for i in range(3)],
          "torch split+list, base (3 seeds)": [f"tmlp_F95_split_list_s{i}" for i in range(3)],
          "trees F95, all physics": ["gbt_F95_phys_s1"],
          "torch, all physics (B0)": ["tmlp_F95_phys_split_list_s0"],
          "torch T inputs, all physics": ["tmlp_T144_phys_split_list_s0"],
          "ChargerNet, all physics": ["tmlp_T144_phys_charger_s0"],
          "trees + power, + pilot": ["gbt_P102_physpmix_s1"],
          "trees + power, + pilot x5": ["gbt_P102_physpmix_w5_s1"],
          "torch T inputs, + pilot": ["tmlp_T144_physpmix_split_list_s0"],
          "ChargerNet + pilot (3 seeds)": [f"tmlp_T144_physpmix_charger_s{i}" for i in range(3)]}
print(f"{'model':36s} {'viol':>5s} {'vs LA % (route mean over seeds)':>32s} {'median':>7s} {'+rest/route':>12s}")
for name, tags in groups.items():
    per, viol, rest = {}, 0, []
    for t in tags:
        p = os.path.join(R, f"eval_{t}_g99sr_test.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            viol += r["n_violations"]
            if r["route_completed"] and r["LA_completed"] and r["LA"]:
                per.setdefault(r["instance"], []).append(100 * (r["duration_h"] / r["LA"] - 1))
                rest.append(r["n_rests"] - r["LA_rests"])
    pct = np.array([np.mean(v) for v in per.values() if len(v) == len(tags)])
    if not len(pct):
        print(f"{name:36s} missing")
        continue
    print(f"{name:36s} {viol:5d} {pct.mean():+16.2f} ± {se(pct):4.2f} (n={len(pct)}) "
          f"{np.median(pct):+7.2f} {np.mean(rest):+12.3f}")
