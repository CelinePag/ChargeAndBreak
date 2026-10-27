"""
rl_check_mixed.py — rl_env.run_batch must reproduce the mixed student exactly
==============================================================================
On every 4th mixed-charger test route (pmix + mix), the mixed student
(tmlp_T144_physmix_split_list_s0, final protocol) driven by run_batch on the
stored realisation must match results/hybrid_alone_ms_N.jsonl, which
run_hybrid.py --alone wrote under the same protocol.  The student alone was
already evaluated on these routes; nothing about the rollout is looked at.
Prints only.

    python ML/scratch/rl_check_mixed.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from rl_env import final_student, load_route, realised, route_set, run_batch  # noqa: E402

TAG = "tmlp_T144_physmix_split_list_s0"


def main():
    ref = {}
    with open(os.path.join(HERE, "..", "results", "hybrid_alone_ms_N.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r["method"] == TAG:
                ref[(r["variant"], r["instance"])] = r
    base = final_student("torch", TAG)
    jobs = route_set("mixed", "test", "pmix,mix")
    print(f"{len(jobs)} mixed test routes; {len(ref)} reference rows")
    n = bad = 0
    for v, name, path in jobs[::4]:
        e = ref.get((v, name))
        if e is None:
            print("no reference row:", v, name)
            continue
        fd, D, E, cv = load_route(name, path=path)
        r = run_batch(base, fd, cv, [(None, realised(D, E))])[0]
        n += 1
        same = (r["failed"] == (not e["completed"])
                and (r["duration"] is None or abs(r["duration"] - e["duration_h"]) < 1e-6)
                and r["misses"] == e["tw"])
        bad += int(not same)
        if not same:
            print(f"MISMATCH {v} {name}: env {r['duration']} / {r['misses']}  "
                  f"stored {e['duration_h']} / {e['tw']}")
    print(f"{n - bad}/{n} identical")


if __name__ == "__main__":
    main()
