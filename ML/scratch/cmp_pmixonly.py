"""Is the uniform pool diluting the mixed data?  python cmp_pmixonly.py [val|test]
Cost of mixing (pmix - uniform, same route, gap to a certified oracle) per
configuration, route-wise mean over the seeds it has, paired against the pool
with the same mixed data."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import mixed_eval as me                                         # noqa: E402

me.ROUTE_SET = sys.argv[1] if len(sys.argv) > 1 else "val"
rows = me.read_rows(me.store_path(0.99, True))
orc, G = {}, {}
for r in rows:
    k = (r["variant"], r["instance"])
    if k not in orc:
        orc[k] = me.read_oracle(*k)
    g = me._gap(r, orc[k]) if r["completed"] else None
    if g is not None and orc[k] and (orc[k]["gap"] or 0) <= 0.01:
        G[(r["method"], r["variant"], r["instance"].split("__")[0])] = g
fails = {}
for r in rows:
    if not r["completed"]:
        fails[r["method"]] = fails.get(r["method"], 0) + 1


def seeds(label, n=3):
    return [label] + [f"{label} (s{s})" for s in range(1, n)]


CONFIGS = {
    "no mixed data": ["torch split+list, T inputs, all phys"],
    "pool + 121 mixed routes": seeds("torch T, + 121 mixed routes"),
    "pool + 121 mixed x5": seeds("torch T, + 121 mixed x5"),
    "pool + 121 mixed x20": seeds("torch T, + 121 mixed x20"),
    "mixed only (106 fit + 15 stop)": seeds("torch T, mixed only"),
    "pool + DAgger x2": seeds("torch T, + DAgger x2"),
    "mixed only + DAgger x2": seeds("torch T, mixed only + DAgger x2"),
}
REF = "pool + 121 mixed routes"


def per_route(labels, v):
    """{base route: mean gap over the seeds that have it}, seeds present."""
    have = [m for m in labels if any(k[0] == m for k in G)]
    out = {}
    for b in {k[2] for k in G if k[1] == v}:
        x = [G[(m, v, b)] for m in have if (m, v, b) in G]
        if have and len(x) == len(have):
            out[b] = np.mean(x)
    return out, len(have)


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


print(f"{me.ROUTE_SET.upper()} routes, cost of mixing = pmix - uniform (pp), "
      f"certified oracles only")
print(f"{'configuration':32s} {'seeds':>5s} {'n':>3s} {'uniform':>8s} {'pmix':>7s} "
      f"{'cost of mixing':>16s} {'vs ' + REF:>30s} {'fails':>5s}")
cost, mixed, unif = {}, {}, {}
for name, labels in CONFIGS.items():
    u, ns = per_route(labels, "uniform")
    x, _ = per_route(labels, "pmix")
    bs = sorted(set(u) & set(x))
    if not bs:
        print(f"{name:32s}  (not driven yet)")
        continue
    cost[name] = {b: x[b] - u[b] for b in bs}
    mixed[name], unif[name] = x, u
    c = np.array(list(cost[name].values()))
    vs = ""
    if name != REF and REF in cost:
        d = np.array([cost[name][b] - cost[REF][b] for b in bs if b in cost[REF]])
        vs = f"{d.mean():+.2f} ± {se(d):.2f} (n={len(d)})"
    nf = sum(fails.get(m, 0) for m in labels)
    print(f"{name:32s} {ns:5d} {len(bs):3d} {np.mean([u[b] for b in bs]):+8.2f} "
          f"{np.mean([x[b] for b in bs]):+7.2f} {c.mean():+8.2f} ± {se(c):.2f}   {vs:>30s} {nf:5d}")

# a model trained without uniform routes makes "cost of mixing" meaningless
# (its uniform gap is what changes), so compare each side directly
print(f"\npaired vs '{REF}' (pp of gap to the oracle), same routes")
print(f"{'configuration':32s} {'on mixed routes':>20s} {'on their uniform twins':>24s}")
for name in mixed:
    if name == REF:
        continue
    out = []
    for side in (mixed, unif):
        d = np.array([side[name][b] - side[REF][b] for b in side[name] if b in side[REF]])
        out.append(f"{d.mean():+.2f} ± {se(d):.2f}")
    print(f"{name:32s} {out[0]:>20s} {out[1]:>24s}")
