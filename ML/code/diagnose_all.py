"""
diagnose_all.py — the cause of every infeasible run, in every experiment
========================================================================
Runs `halt_state.diagnose` over each result that has infeasible runs and
stores one record per failure, so the report can state how many failures each
cause explains instead of quoting a console.

    route length   models trained on short+medium, on all 239 long routes
    physics        the base-case models on each shifted axis (ood_eval.py)

each as trained (g95), with the spread-room check (g95sr) and with the check
and a 0.99 drive guard (g99sr), where the runs exist.  Causes (halt_state.cause):

    uncounted-dwell   the legality check passed, but the dwell actually spent
                      (charge + stop overhead) no longer fits the 15 h spread
    drive-tail        the plan fitted at the guarded drive time; the realised
                      drive was longer
    ferry             the breaking decision is a sea crossing, which has one
                      legal action: the mistake (no rest before boarding) was
                      made stops earlier, beyond a one-step check
    other             anything else

    python ML/code/diagnose_all.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from halt_state import diagnose, runs_from_eval, runs_from_ood   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
STORE = os.path.join(RESULTS, "halt_diagnosis.json")
KIND = {"gbt": "gbt", "clf": "clf", "mlp": "nn"}
LENGTH_SETS = {"gbt": ("F95", "R88"), "clf": ("F77", "R70"), "mlp": ("F95", "R88")}
VARIANTS = {"g95": (0.95, False), "g95sr": (0.95, True), "g99sr": (0.99, True)}


def length_cases():
    for arm, sets in LENGTH_SETS.items():
        for fs in sets:
            for sd in range(3):
                tag = f"{arm}_{fs}_base_SM_s{sd}"
                for v in VARIANTS:
                    f = f"eval_{tag}_{v}_longall.json"
                    if os.path.exists(os.path.join(RESULTS, f)):
                        yield dict(experiment="route length", axis="long, all 239",
                                   variant=v, arm=arm, tag=tag), f


def ood_cases():
    from ood_eval import AXES, variant_tail
    for v, (q, sr) in VARIANTS.items():
        p = os.path.join(RESULTS, f"ood_test{variant_tail(q, sr)}.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            rows = json.load(fh)
        for axis in AXES:
            for tag in sorted({r["tag"] for r in rows if r["axis"] == axis
                               and r["method"] not in ("LA", "Greedy")}):
                arm = tag.split("_")[0]
                yield dict(experiment="physics", axis=axis, variant=v, arm=arm,
                           tag=tag), (axis, q, sr)


def main():
    out = []
    for meta, f in length_cases():
        rows, runs = runs_from_eval(f)
        if runs:
            q, sr = VARIANTS[meta["variant"]]
            for d in diagnose(KIND[meta["arm"]], meta["tag"], runs, q, sr):
                out.append({**meta, **d})
        print(f"{meta['tag']:24s} {meta['variant']:6s} long: {len(runs)} infeasible",
              flush=True)
    for meta, (axis, q, sr) in ood_cases():
        rows, runs = runs_from_ood(axis, meta["tag"], q, sr)
        if runs:
            for d in diagnose(KIND[meta["arm"]], meta["tag"], runs, q, sr):
                out.append({**meta, **d})
        print(f"{meta['tag']:24s} {meta['variant']:6s} {axis}: {len(runs)} infeasible",
              flush=True)
    with open(STORE, "w") as fh:
        json.dump(out, fh, indent=1)

    print("\nfailures by cause  (reproduced = replay matched the stored run)")
    agg = collections.OrderedDict()
    for d in out:
        agg.setdefault((d["experiment"], d["variant"], d["arm"], d["axis"]), []).append(d)
    for (ex, v, arm, axis), ds in agg.items():
        c = collections.Counter(d["cause"] for d in ds)
        print(f"   {ex:13s} {v:6s} {arm:4s} {axis:14s} {len(ds):3d}  "
              f"{dict(c)}  reproduced {sum(d['reproduced'] for d in ds)}/{len(ds)}")
    print(f"\nsaved: {STORE}")


if __name__ == "__main__":
    main()
