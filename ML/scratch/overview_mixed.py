"""Numbers for the results overview table, mixed-charger test routes (2026-10-20).
For each configuration: gap to the certified hindsight optimum on the mixed
route (route mean over its seeds), and the paired difference to the LA on the
same routes, for pmix (power varies) and mix (power and spacing vary); dmix
(spacing varies) has no LA runs, so its row is the change vs the uniform twin."""
import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "paper"))
with contextlib.redirect_stdout(io.StringIO()):
    import fig_results as fr                                     # noqa: E402
    me, G, D, R = fr.mixed_gaps()


def seeds(label, n=3):
    return [label] + [f"{label} (s{s})" for s in range(1, n)]


CFG = [
    ("Greedy rule", ["Greedy"]),
    ("Trees, 350 kW only", ["trees, base only"]),
    ("Trees, all uniform powers", ["trees, all physics"]),
    ("Trees, all + 47 mixed", ["trees + power, + pilot"]),
    ("Trees, all + 121 mixed", ["trees + power, + 121 mixed routes"]),
    ("Trees, all + DAgger", ["trees + power, + DAgger"]),
    ("PyTorch, 350 kW only", ["torch split+list, base only"]),
    ("PyTorch, all uniform powers", ["torch split+list, all physics"]),
    ("PyTorch T, all uniform powers", ["torch split+list, T inputs, all phys"]),
    ("ChargerNet, all uniform powers", ["ChargerNet, all physics"]),
    ("ChargerNet, + 47 mixed", ["ChargerNet, + pilot", "ChargerNet, + pilot (s1)", "ChargerNet, + pilot (s2)"]),
    ("PyTorch T, + 47 mixed", ["torch T inputs, + pilot", "torch T, + pilot (s1)", "torch T, + pilot (s2)"]),
    ("PyTorch T, + 121 mixed", seeds("torch T, + 121 mixed routes")),
    ("PyTorch T, + 121 mixed x5", seeds("torch T, + 121 mixed x5")),
    ("PyTorch T, mixed only", seeds("torch T, mixed only")),
    ("PyTorch T, + DAgger 1 round", seeds("torch T, + DAgger")),
    ("PyTorch T, + DAgger 2 rounds", seeds("torch T, + DAgger x2")),
    ("PyTorch T, + 121 pmix + 117 mix", seeds("torch T, + pmix + mix routes")),
    ("PyTorch T, pmix + mix only", seeds("torch T, pmix + mix only")),
    ("PyTorch T, + pmix + mix x5", seeds("torch T, + pmix + mix x5")),
    ("LA", ["LA"]),
]


def stat(labels, v):
    g, d = [], []
    for b in sorted({b for (_m, vv, b) in G if vv == v}):
        x = fr.route_mean(G, labels, v, b)
        if x is None:
            continue
        g.append(x)
        if ("LA", v, b) in G and labels != ["LA"]:
            d.append(x - G[("LA", v, b)])
    return g, d


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


print(f"{'configuration':34s} {'pmix gap':>9s} {'vs LA':>14s} {'mix gap':>9s} {'vs LA':>14s} {'dmix-unif':>10s}")
for name, labs in CFG:
    out = []
    for v in ("pmix", "mix"):
        g, d = stat(labs, v)
        out.append(f"{np.mean(g):9.1f}" if g else f"{'—':>9s}")
        out.append(f"{np.mean(d):+7.1f} ± {se(d):3.1f}" if len(d) > 1 else f"{'—':>14s}")
    # spacing-mixed: change vs the uniform twin (no LA there)
    ch = []
    for b in sorted({b for (_m, vv, b) in G if vv == "dmix"}):
        a, c = fr.route_mean(G, labs, "uniform", b), fr.route_mean(G, labs, "dmix", b)
        if a is not None and c is not None:
            ch.append(c - a)
    out.append(f"{np.mean(ch):+10.1f}" if ch else f"{'—':>10s}")
    print(f"{name:34s} " + " ".join(out))
