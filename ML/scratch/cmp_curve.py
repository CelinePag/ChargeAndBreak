"""Mixed-route data vs DAgger, test routes (looked at once, 2026-10-03).

Per configuration, the route-wise mean over its seeds, then paired over routes:
cost of mixing (mixed - uniform, same route) and the mixed gap vs the
no-mixed-data model.  Plus the base-case test routes vs the LA.
NOTE: the "89 routes" models were trained before the last LA shares finished
(extraction at 10:16, LA done 13:27-13:57): 89 of the 121 usable routes.
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

CONFIGS = {   # name: (LA work, method labels = seeds)
    "torch, no mixed data": ("0", ["torch split+list, T inputs, all phys"]),
    "torch, + DAgger": ("1,781 calls", ["torch T, + DAgger", "torch T, + DAgger (s1)", "torch T, + DAgger (s2)"]),
    "torch, + pilot (47 routes)": ("2,279 dec.", ["torch T inputs, + pilot", "torch T, + pilot (s1)", "torch T, + pilot (s2)"]),
    "torch, + 89 mixed routes": ("6,451 dec.", ["torch T, + all mixed data", "torch T, + all mixed data (s1)",
                                                "torch T, + all mixed data (s2)"]),
    "torch, + 121 mixed routes": ("8,246 dec.", ["torch T, + 121 mixed routes", "torch T, + 121 mixed routes (s1)",
                                                 "torch T, + 121 mixed routes (s2)"]),
    "torch, + DAgger x2": ("3,557 calls", ["torch T, + DAgger x2", "torch T, + DAgger x2 (s1)",
                                           "torch T, + DAgger x2 (s2)"]),
    # chosen on the validation routes 2026-10-04 (mixed-only and x20 lost there)
    "torch, + 121 mixed x5": ("8,246 dec.", ["torch T, + 121 mixed x5", "torch T, + 121 mixed x5 (s1)",
                                             "torch T, + 121 mixed x5 (s2)"]),
    "trees, no mixed data": ("0", ["trees, all physics + power"]),
    "trees, + DAgger": ("1,781 calls", ["trees + power, + DAgger"]),
    "trees, + pilot (47 routes)": ("2,279 dec.", ["trees + power, + pilot"]),
    "trees, + 89 mixed routes": ("6,451 dec.", ["trees + power, + all mixed data"]),
    "trees, + 121 mixed routes": ("8,246 dec.", ["trees + power, + 121 mixed routes"]),
}

# the LA itself on every mixed test route it has run (uniform: the stored teacher run)
for b in sorted({b for (_m, vv, b) in G if vv == "pmix"}):
    for vv, nm in (("uniform", b), ("pmix", f"{b}__pmix")):
        la = me.la_row(vv, nm)
        o = me.read_oracle(vv, nm)
        if la and la["completed"] and o and (o["gap"] or 0) <= 0.01:
            G[("LA", vv, b)] = (me._gap(la, o), la["rests"], o["rests"])
CONFIGS["LA (the teacher)"] = ("—", ["LA"])
REF = {"torch": "torch, no mixed data", "trees": "trees, no mixed data",
       "LA (the teacher)": "LA (the teacher)"}


def mean_over_seeds(labels, v, b):
    got = [G[(m, v, b)] for m in labels if (m, v, b) in G]
    if len(got) != len(labels):
        return None
    return np.mean([x[0] for x in got]), np.mean([x[1] > x[2] + 0.5 for x in got])


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


for v in ("pmix", "mix", "dmix"):
    bases = sorted({b for (_m, vv, b) in G if vv == v})
    print(f"\n{v.upper()} TEST ROUTES ({len(bases)})   cost of mixing = {v} - uniform, same route")
    print(f"{'configuration':30s} {'LA work':>12s} {'uniform':>8s} {'mixed':>8s} "
          f"{'cost of mixing':>16s} {'mixed vs no data':>18s} {'+rest':>6s}")
    for name, (work, labels) in CONFIGS.items():
        ref = CONFIGS[REF.get(name, REF.get(name.split(",")[0], name))][1]
        ch, vs, u, x, extra = [], [], [], [], 0.0
        for b in bases:
            a, c = mean_over_seeds(labels, "uniform", b), mean_over_seeds(labels, v, b)
            if a and c:
                ch.append(c[0] - a[0]); u.append(a[0]); x.append(c[0]); extra += c[1]
            r0 = mean_over_seeds(ref, v, b)
            if c and r0:
                vs.append(c[0] - r0[0])
        if len(ch) < 2:
            print(f"{name:30s} (missing)")
            continue
        print(f"{name:30s} {work:>12s} {np.mean(u):+8.2f} {np.mean(x):+8.2f} "
              f"{np.mean(ch):+8.2f} ± {se(ch):4.2f} {np.mean(vs):+10.2f} ± {se(vs):4.2f} {extra:6.1f}")

print("\nVS THE LA on the mixed-power test routes it has run: cost of mixing, method minus LA, paired")
la_b = sorted({b for (m, vv, b) in G if m == "LA" and vv == "pmix" and ("LA", "uniform", b) in G})
for name, (work, labels) in CONFIGS.items():
    if name.startswith("LA"):
        continue
    d = []
    for b in la_b:
        a, c = mean_over_seeds(labels, "uniform", b), mean_over_seeds(labels, "pmix", b)
        if a and c:
            d.append((c[0] - a[0]) - (G[("LA", "pmix", b)][0] - G[("LA", "uniform", b)][0]))
    if len(d) > 1:
        print(f"  {name:30s} {np.mean(d):+6.2f} ± {se(d):4.2f} pp  (n={len(d)})")

print("\nBASE-CASE TEST ROUTES (125, g99sr), paired vs the LA, route mean over seeds")
R = os.path.join(HERE, "..", "results")
TAGS = {"torch, no mixed data": ["tmlp_T144_phys_split_list_s0"],
        "torch, + DAgger": [f"tmlp_T144_physdg_split_list_s{i}" for i in range(3)],
        "torch, + pilot (47 routes)": [f"tmlp_T144_physpmix_split_list_s{i}" for i in range(3)],
        "torch, + 89 mixed routes": [f"tmlp_T144_physpmixall_split_list_s{i}" for i in range(3)],
        "torch, + 121 mixed routes": [f"tmlp_T144_physpmix121_split_list_s{i}" for i in range(3)],
        "torch, + DAgger x2": [f"tmlp_T144_physdg2_split_list_s{i}" for i in range(3)],
        "trees, + 121 mixed routes": ["gbt_P102_physpmix121_s1"],
        "trees, + DAgger": ["gbt_P102_physdg_s1"],
        "trees, + pilot (47 routes)": ["gbt_P102_physpmix_s1"],
        "trees, + 89 mixed routes": ["gbt_P102_physpmixall_s1"]}
for name, tags in TAGS.items():
    per, viol = {}, 0
    for t in tags:
        p = os.path.join(R, f"eval_{t}_g99sr_test.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            viol += r["n_violations"]
            if r["route_completed"] and r["LA_completed"] and r["LA"]:
                per.setdefault(r["instance"], []).append(100 * (r["duration_h"] / r["LA"] - 1))
    pct = np.array([np.mean(v) for v in per.values() if len(v) == len(tags)])
    print(f"{name:30s} viol {viol:3d}   vs LA {pct.mean():+6.2f} ± {se(pct):4.2f} %  (n={len(pct)})")
