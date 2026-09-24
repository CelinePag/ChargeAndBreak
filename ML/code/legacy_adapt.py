"""
legacy_adapt.py — fold the restored 2026-08 model into the common results table
==============================================================================
`ML/legacy/code/rollout.py` drives the simulator through the `external_policy`
hook and writes one solution JSON per route under `ML/solutions/`.  This turns
those into the same `eval_*.json` shape every other arm produces, so the
legacy model can appear in `fig_gap.py` and `report.py` alongside the rest.

The JSONs are read rather than the rollout's own printed lines: it prints
`res.get("duration_h", -1)`, but `run_simulation_precomputed` returns the
arrival as `total_time` and has no `duration_h` key, so every printed duration
is -1.  That is a symptom of the repo moving on since the legacy code was
written -- the written solutions carry the real numbers.

It is a FORMAT adapter, not a re-run: the durations and window misses are the
legacy rollout's own.  What it adds is the baseline columns (LA / Greedy /
oracle for the same instance), read from `solutions/basecase` exactly as
`evaluate.py` does, so the row is paired the same way.

Reminder, repeated here because the number invites a false comparison: the
legacy model uses 141 features, 772 teacher runs and its own forcing and clamp
code.  Its row is "what we had before", not a controlled contrast -- see
`ML/legacy/README.md`.  The controlled test of the same framing is `clf_*`.

    python ML/code/legacy_adapt.py
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.instance_gen.instance_io import load_instance_json      # noqa: E402

from configs import LEGACY                                       # noqa: E402
from evaluate import baselines                                   # noqa: E402

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
SOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "solutions"))
INST = os.path.join(_ROOT, "instances")


def main():
    import glob
    files = sorted(glob.glob(os.path.join(SOLS, "*_STUDENT_*.json")))
    if not files:
        raise SystemExit(f"no STUDENT solutions in {SOLS} — run the legacy "
                         f"pipeline first")
    # one row per instance, newest run wins
    latest = {}
    for f in files:
        inst = os.path.basename(f).split("_STUDENT_")[0]
        latest[inst] = f

    rows = []
    for inst, f in sorted(latest.items()):
        with open(f) as fh:
            s = json.load(fh)
        m = s.get("metrics", {})
        dur = s.get("duration_h")
        completed = dur is not None and not m.get("run_infeasible")
        fd, _D, _E, _cv = load_instance_json(os.path.join(INST, inst + ".json"))
        r = dict(instance=inst, family=inst.rsplit("_", 1)[0],
                 duration_h=dur if completed else None,
                 sim_arrival_h=s.get("sim_arrival_h"),
                 route_completed=bool(completed),
                 halt_reason=(m.get("violations") or [{}])[0].get("type")
                 if not completed else None,
                 tw_misses=int(m.get("tw_n_misses", 0)),
                 n_customers=len(fd["C"]), n_stops=int(fd["N"]),
                 decisions=0, ms_per_decision=0.0,
                 n_rests=None, n_charges=None, action_mix={})
        r.update(baselines(inst))
        rows.append(r)

    if not rows:
        raise SystemExit("no per-route lines found in the legacy log")
    out = os.path.join(RESULTS, LEGACY.eval_name)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    comp = [r for r in rows if r["route_completed"]]
    print(f"wrote {out}")
    print(f"  {len(rows)} routes, {len(comp)} completed, "
          f"{len(rows)-len(comp)} infeasible, "
          f"{sum(r['tw_misses'] for r in rows)} window misses")


if __name__ == "__main__":
    main()
