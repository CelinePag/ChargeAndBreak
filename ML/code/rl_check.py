"""
rl_check.py — rl_env.run_batch must reproduce the evaluated students exactly
============================================================================
On the stored realisations of the validation routes (route seeds 20-21), the
student driven by run_batch (chooser None) must give the same duration and
window misses as policy_core.run_student did in evaluate.py under the final
protocol (results/eval_<tag>_g99srNm_stop.json).  Also times the batch on
fresh draws.  Prints only; writes nothing.

    python ML/code/rl_check.py --kind gbt --tag gbt_F95_base_s0 [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from rl_env import (final_student, fresh_draws, load_route, realised,
                    run_batch)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gbt")
    ap.add_argument("--tag", default="gbt_F95_base_s0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--draws", type=int, default=8)
    a = ap.parse_args()

    ref = {r["instance"]: r for r in json.load(open(os.path.join(
        RESULTS, f"eval_{a.tag}_g99srNm_stop.json")))}
    insts = sorted(ref)[: a.limit or None]
    base = final_student(a.kind, a.tag)
    bad, t_real, t_fresh, n_steps = 0, 0.0, 0.0, 0
    for inst in insts:
        fd, D, E, cv = load_route(inst)
        t0 = time.perf_counter()
        r = run_batch(base, fd, cv, [(None, realised(D, E))])[0]
        t_real += time.perf_counter() - t0
        e = ref[inst]
        same = (r["failed"] == (not e["route_completed"])
                and (r["duration"] is None or abs(r["duration"] - e["duration_h"]) < 1e-6)
                and r["misses"] == e["tw_misses"])
        bad += int(not same)
        if not same:
            print(f"MISMATCH {inst}: env {r['duration']} / {r['misses']}  "
                  f"eval {e['duration_h']} / {e['tw_misses']}")
        t0 = time.perf_counter()
        rs = run_batch(base, fd, cv, [(None, d) for d in fresh_draws(fd, cv, a.draws, f"{inst}|check")])
        t_fresh += time.perf_counter() - t0
        n_steps += sum(x["decisions"] for x in rs)
    print(f"{a.tag}: {len(insts) - bad}/{len(insts)} identical to the evaluation")
    print(f"realised: {t_real / len(insts):.2f} s per route (one vehicle)")
    print(f"fresh draws, batches of {a.draws}: {t_fresh / len(insts):.2f} s per route, "
          f"{1000 * t_fresh / max(n_steps, 1):.2f} ms per vehicle-step")


if __name__ == "__main__":
    main()
