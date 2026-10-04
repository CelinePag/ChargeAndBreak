"""Numbers the CPAIOR draft states that no table prints directly."""
import contextlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
with contextlib.redirect_stdout(io.StringIO()):
    import cmp_curve as c                                       # noqa: E402
import mixed_eval as me                                         # noqa: E402
from features import charger_kw                                 # noqa: E402


def paired(a, b, v="pmix"):
    A, B = c.CONFIGS[a][1], c.CONFIGS[b][1]
    d = []
    for k in sorted({k for (_m, vv, k) in c.G if vv == v}):
        xa, xb = c.mean_over_seeds(A, v, k), c.mean_over_seeds(B, v, k)
        ua, ub = c.mean_over_seeds(A, "uniform", k), c.mean_over_seeds(B, "uniform", k)
        if xa and xb and ua and ub:
            d.append((xa[0] - ua[0]) - (xb[0] - ub[0]))
    d = np.array(d)
    return f"{d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(len(d)):.2f} (n={len(d)})"


print("DAgger minus 121 routes:", paired("torch, + DAgger", "torch, + 121 mixed routes"))
print("DAgger minus pilot 47:  ", paired("torch, + DAgger", "torch, + pilot (47 routes)"))

# the LA's energy by charger power on the mixed test routes
tot = {}
for name in me.la_routes("pmix"):
    path = os.path.join(me.INST, "pmix", name + ".json")
    fd, D, E, _cv = me.load(path)
    with open(me.la_solution("pmix", name)) as fh:
        s = json.load(fh)
    tr = {int(x["stop"]): float(x["e_arr"]) for x in s["sim_trajectory"]}
    for k in sorted(int(j) for j in fd["K"]):
        if k in tr and k + 1 in tr:
            charged = tr[k + 1] + float(E[k]) - tr[k]
            if charged > 1e-6:
                p = round(charger_kw(fd, k))
                tot[p] = tot.get(p, 0.0) + charged
all_e = sum(tot.values())
print("LA energy share by charger power on", len(me.la_routes("pmix")), "routes:",
      {p: f"{100 * e / all_e:.0f}%" for p, e in sorted(tot.items())},
      "| at 700+1000 kW:", f"{100 * (tot.get(700, 0) + tot.get(1000, 0)) / all_e:.0f}%")
