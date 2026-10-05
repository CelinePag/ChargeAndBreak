"""Power+spacing mixed training routes (cluster stage mix-train, 2026-10-05).
    python ML/scratch/cmp_mix.py val     # choose here
    python ML/scratch/cmp_mix.py test    # then read the chosen one here
Per configuration (route-wise mean over its seeds), on the power-mixed (pmix)
and power+spacing (mix) routes: gap to a certified oracle on the uniform twin
and on the mixed route, cost of mixing, paired difference on the mixed routes
against the pool + 121 pmix routes, and (test) paired against the LA."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import mixed_eval as me                                         # noqa: E402

me.ROUTE_SET = sys.argv[1] if len(sys.argv) > 1 else "val"
rows = me.read_rows(me.store_path(0.99, True))
orc, G, FAIL = {}, {}, {}
for r in rows:
    k = (r["variant"], r["instance"])
    if k not in orc:
        orc[k] = me.read_oracle(*k)
    if not r["completed"]:
        FAIL[r["method"]] = FAIL.get(r["method"], 0) + 1
        continue
    g = me._gap(r, orc[k])
    if g is not None and orc[k] and (orc[k]["gap"] or 0) <= 0.01:
        G[(r["method"], r["variant"], r["instance"].split("__")[0])] = g


def seeds(label, n=3):
    return [label] + [f"{label} (s{s})" for s in range(1, n)]


CONFIGS = {   # name: (mixed LA data, method labels = seeds, base-case tags)
    "no mixed data": ("0", ["torch split+list, T inputs, all phys"],
                      ["tmlp_T144_phys_split_list_s0"]),
    "pool + 121 pmix": ("8,246", seeds("torch T, + 121 mixed routes"),
                        [f"tmlp_T144_physpmix121_split_list_s{i}" for i in range(3)]),
    "pool + 121 pmix x5": ("8,246", seeds("torch T, + 121 mixed x5"),
                           [f"tmlp_T144_physpmix121w5_split_list_s{i}" for i in range(3)]),
    "pool + DAgger x2 (pmix)": ("3,557", seeds("torch T, + DAgger x2"),
                                [f"tmlp_T144_physdg2_split_list_s{i}" for i in range(3)]),
    "pool + pmix + mix": ("16,186", seeds("torch T, + pmix + mix routes"),
                          [f"tmlp_T144_physmix_split_list_s{i}" for i in range(3)]),
    "pmix + mix only": ("16,186", seeds("torch T, pmix + mix only"),
                        [f"tmlp_T144_mixonly_split_list_s{i}" for i in range(3)]),
    "pool + pmix + mix x5": ("16,186", seeds("torch T, + pmix + mix x5"),
                             [f"tmlp_T144_physmixw5_split_list_s{i}" for i in range(3)]),
}
REF = "pool + 121 pmix"
NEW = ("pool + pmix + mix", "pmix + mix only", "pool + pmix + mix x5")

# the LA's own runs (test routes only)
if me.ROUTE_SET == "test":
    for v in ("pmix", "mix"):
        for b in sorted({b for (_m, vv, b) in G if vv == v}):
            for vv, nm in (("uniform", b), (v, f"{b}__{v}")):
                la = me.la_row(vv, nm)
                o = me.read_oracle(vv, nm)
                if la and la["completed"] and o and (o["gap"] or 0) <= 0.01:
                    G[("LA", vv, b)] = me._gap(la, o)
    CONFIGS["LA"] = ("-", ["LA"], [])


def mean(labels, v, b):
    x = [G.get((m, v, b)) for m in labels]
    return None if any(y is None for y in x) else float(np.mean(x))


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def fmt(x):
    return f"{np.mean(x):+6.2f} ± {se(x):4.2f}" if len(x) > 1 else f"{'—':>13s}"


for v in ("pmix", "mix"):
    bases = sorted({b for (_m, vv, b) in G if vv == v})
    print(f"\n{me.ROUTE_SET.upper()} · {v} routes ({len(bases)}), gap to the oracle (%)")
    print(f"{'configuration':26s} {'LA dec.':>7s} {'uniform':>8s} {'mixed':>8s} "
          f"{'cost of mixing':>15s} {'mixed vs ' + REF:>24s} {'vs LA (cost)':>15s} {'fails':>5s}")
    for name, (work, labels, _t) in CONFIGS.items():
        u, x, c, vs, la = [], [], [], [], []
        for b in bases:
            a, m = mean(labels, "uniform", b), mean(labels, v, b)
            if a is None or m is None:
                continue
            u.append(a); x.append(m); c.append(m - a)
            r = mean(CONFIGS[REF][1], v, b)
            if r is not None and name != REF:
                vs.append(m - r)
            if ("LA", v, b) in G and ("LA", "uniform", b) in G and name != "LA":
                la.append((m - a) - (G[("LA", v, b)] - G[("LA", "uniform", b)]))
        if len(x) < 2:
            print(f"{name:26s} (missing)")
            continue
        nf = sum(FAIL.get(m, 0) for m in labels)
        print(f"{name:26s} {work:>7s} {np.mean(u):+8.2f} {np.mean(x):+8.2f} "
              f"{fmt(c):>15s} {fmt(vs):>24s} {fmt(la):>15s} {nf:5d}")

if me.ROUTE_SET == "val":
    print("\nCHOICE among the new configurations: mean mixed-route gap over pmix and mix val routes")
    for name in NEW:
        labels = CONFIGS[name][1]
        x = [mean(labels, v, b) for v in ("pmix", "mix")
             for b in sorted({b for (_m, vv, b) in G if vv == v})]
        x = [y for y in x if y is not None]
        print(f"  {name:26s} {np.mean(x):+6.2f} % over {len(x)} routes")
else:
    print("\nBASE-CASE TEST ROUTES (125, g99sr), paired vs the LA, route mean over seeds")
    R = os.path.join(HERE, "..", "results")
    for name, (_w, _l, tags) in CONFIGS.items():
        if not tags:
            continue
        per, viol = {}, 0
        for t in tags:
            p = os.path.join(R, f"eval_{t}_g99sr_test.json")
            if not os.path.exists(p):
                continue
            for r in json.load(open(p)):
                viol += r["n_violations"]
                if r["route_completed"] and r["LA_completed"] and r["LA"]:
                    per.setdefault(r["instance"], []).append(100 * (r["duration_h"] / r["LA"] - 1))
        pct = [np.mean(v) for v in per.values() if len(v) == len(tags)]
        print(f"  {name:26s} viol {viol:3d}   vs LA {fmt(pct)} %  (n={len(pct)})")
