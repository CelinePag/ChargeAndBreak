"""Does the student rank vehicle / network designs the way the LA does?

Per sensitivity axis (battery, charger power), paired over the same underlying
test routes: mean route duration per design value for the LA (stored runs) and
for the students in results/phys_g99sr.jsonl."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from phys_eval import AXES, baselines, instances  # noqa: E402

STORE = os.path.join(os.path.dirname(__file__), "..", "results", "phys_g99sr.jsonl")
MODELS = ("gbt_F95_phys_s1", "gbt_F95_base_s1")
rows = [json.loads(l) for l in open(STORE)]


def base_of(name):
    return name.split("__")[0]


def collect(axis):
    """{method: {base route: duration h}}.  Stored LA rows carry only the gap
    to that design's oracle, so the LA duration is recovered through the
    oracle implied by a student row on the same route: O = dur / (1 + gap/100).
    Routes with any window miss are dropped (the gap there is penalised)."""
    names = [n for n, _ in instances(axis)]
    out = {}
    for m in MODELS:
        out[m] = {base_of(r["instance"]): r for r in rows
                  if r["tag"] == m and r["axis"] == axis and r["completed"] and r["tw"] == 0}
    la = {}
    for n, r in baselines(axis, names)["LA"].items():
        k, s = base_of(n), out[MODELS[1]].get(base_of(n))
        if r.get("completed") and r.get("gap") is not None and r["tw"] == 0 and s:
            orc = s["duration_h"] / (1 + s["gap"] / 100)
            la[k] = orc * (1 + r["gap"] / 100)
    res = {"LA": la}
    for m in MODELS:
        res[m] = {k: r["duration_h"] for k, r in out[m].items()}
    return res


for title, axis_vals in (("battery kWh", [("kwh300", 300), ("base", 500), ("kwh700", 700), ("kwh900", 900)]),
                         ("charger kW", [("kw150", 150), ("base", 350), ("kw700", 700), ("kw1000", 1000)])):
    data = {v: collect(a) for a, v in axis_vals}
    keys = set.intersection(*[set(data[v][m]) for v in data for m in ("LA",) + MODELS])
    keys = sorted(keys)
    print(f"\n== {title}: {len(keys)} routes with every design x method")
    print(f"{'value':>6s} " + " ".join(f"{m[:18]:>20s}" for m in ("LA",) + MODELS))
    means = {}
    for v in data:
        means[v] = [np.mean([data[v][m][k] for k in keys]) for m in ("LA",) + MODELS]
        print(f"{v:6d} " + " ".join(f"{x:20.2f}" for x in means[v]))
    ref = 500 if "battery" in title else 350
    print("  paired change vs base design (h, mean +- se):")
    for v in data:
        if v == ref:
            continue
        cells = []
        for m in ("LA",) + MODELS:
            dlt = np.array([data[v][m][k] - data[ref][m][k] for k in keys])
            cells.append(f"{dlt.mean():+7.2f} +- {dlt.std(ddof=1) / np.sqrt(len(dlt)):.2f}")
        print(f"{v:6d} " + " ".join(f"{c:>20s}" for c in cells))
    for m_i, m in enumerate(MODELS, start=1):
        order_la = sorted(data, key=lambda v: means[v][0])
        order_m = sorted(data, key=lambda v: means[v][m_i])
        print(f"  ranking LA {order_la}  {m} {order_m}  {'SAME' if order_la == order_m else 'DIFFERENT'}")
