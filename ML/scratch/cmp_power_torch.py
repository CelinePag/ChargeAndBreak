"""Charger power seen vs unseen: trees and PyTorch side by side (2026-10-20).
    python ML/scratch/cmp_power_torch.py
Gap above the LA (pp, paired on routes both completed) per charger power,
for models trained on 350 kW only / all powers / all but the tested one,
plus the plain gap to the hindsight optimum and broken runs.  PyTorch
base-only is the headline model: each route averaged over its 3 seeds."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import phys_eval as pe                                            # noqa: E402

rows = pe.read_store(pe.store_path(0.99, True))
AX = [("base", "350 kW (base case)"), ("kw150", "150 kW"), ("kw700", "700 kW"),
      ("kw1000", "1000 kW")]
MODELS = {   # column: {axis: [tags averaged per route]}
    "trees, 350 kW only": lambda a: ["gbt_F95_base_s1"],
    "trees, all powers": lambda a: ["gbt_F95_phys_s1"],
    "trees, all but tested": lambda a: [f"gbt_F95_phys_LO{a}_s1"],
    "PyTorch, 350 kW only (3 seeds)": lambda a: [f"tmlp_F95_split_list_s{s}" for s in range(3)],
    "PyTorch, 350 kW only (seed 0)": lambda a: ["tmlp_F95_split_list_s0"],
    "PyTorch, all powers": lambda a: ["tmlp_F95_phys_split_list_s0"],
    "PyTorch, all but tested": lambda a: [f"tmlp_F95_phys_LO{a}_split_list_s0"],
}

print(f"{'model':32s}" + "".join(f"{lab:>22s}" for _a, lab in AX))
summ = {}
for axis, _lab in AX:
    sub = [r for r in rows if r["axis"] == axis]
    names = sorted({r["instance"] for r in sub})
    summ[axis] = (pe.baselines(axis, names), sub)

for col, tags_of in list(MODELS.items()) + [("Greedy", None), ("LA itself", None)]:
    cells = []
    for axis, _lab in AX:
        base, sub = summ[axis]
        if col in ("Greedy", "LA itself"):
            runs = base["Greedy" if col == "Greedy" else "LA"]
        else:
            tags = tags_of(axis)
            per = {}
            for r in sub:
                if r["tag"] in tags:
                    per.setdefault(r["instance"], []).append(r)
            runs = {}
            for k, rs in per.items():
                if len(rs) == len(tags):
                    ok = all(r["completed"] and r["gap"] is not None for r in rs)
                    runs[k] = dict(completed=ok,
                                   gap=float(np.mean([r["gap"] for r in rs])) if ok else None,
                                   n_fail=sum(not r["completed"] for r in rs))
        if not runs:
            cells.append(f"{'—':>22s}")
            continue
        if col == "LA itself":
            g = [r["gap"] for r in runs.values() if r["completed"] and r["gap"] is not None]
            cells.append(f"{'gap ' + format(np.mean(g), '.1f') + ' %':>22s}")
            continue
        p = pe._paired(runs, base["LA"])
        fails = sum(r.get("n_fail", 0) for r in runs.values())
        cells.append(f"{p['mean']:+6.1f} ± {p['se']:.1f} (n={p['n']})" + (f" F{fails}" if fails else "").rjust(0)
                     if p else f"{'—':>22s}")
        cells[-1] = f"{cells[-1]:>22s}"
    print(f"{col:32s}" + "".join(cells))
print("\nvalues: model gap minus LA gap, pp, paired; F<n> = runs that broke a rule")
