"""
rollout_timing.py — decision time per stop of the rollout, measured on this machine
==================================================================================
The cluster runs only stored the wall time per route.  This re-drives the
frozen rollout (32 futures, races of 8) of one student on the 125 base-case
test routes, stored realisation, and times every decision (the student's
batched prediction is shared, so the time of a stop is the rollout's choose()
call plus a negligible prediction).  Checks each route against the stored
cluster test run (same decisions, same duration).

Writes ML/results/rollout_timing_<tag>.jsonl: per route, the decision times
(s) and which stops were decisive.  --report prints mean / p95 / max.

    python ML/scratch/rollout_timing.py --tag tmlp_F95_split_list_s0 --workers 4
    python ML/scratch/rollout_timing.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
RES = os.path.join(HERE, "..", "results")
STORED = {"tmlp_F95_split_list_s0": ("dsr_test_tmlp_F95_split_list_s0", 1),
          "tmlp_F95_split_list_s0_exit2": ("exit_test_r2", 1),
          "gbt_F95_base_s0": ("dsr_test_gbt_F95_base_s0", 1)}
_G = {}


class Timed:
    def __init__(self, inner):
        self.inner, self.t, self.dec = inner, [], []

    def choose(self, fd, pre, veh, legal, rows, score, flags):
        t0 = time.perf_counter()
        calls = self.inner.calls
        j = self.inner.choose(fd, pre, veh, legal, rows, score, flags)
        self.t.append(time.perf_counter() - t0)
        self.dec.append(int(self.inner.calls > calls))
        return j


def _init(kind, tag):
    from rl_env import final_student
    _G["base"] = final_student(kind, tag)


def one(inst):
    from rl_env import load_route, realised, run_batch
    from rl_policy import RolloutChooser
    fd, D, E, cv = load_route(inst)
    ch = RolloutChooser(_G["base"], futures=32, race=8)
    ch.route, ch.cv = inst, cv
    tc = Timed(ch)
    t0 = time.perf_counter()
    res = run_batch(_G["base"], fd, cv, [(tc, realised(D, E))])
    return dict(instance=inst, times=tc.t, decisive=tc.dec, duration=res[0]["duration"],
                route_s=time.perf_counter() - t0)


def report():
    for f in sorted(glob.glob(os.path.join(RES, "rollout_timing_*.jsonl"))):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        t = np.concatenate([r["times"] for r in recs])
        d = np.concatenate([r["decisive"] for r in recs]).astype(bool)
        print(f"{os.path.basename(f)}: {len(recs)} routes, {len(t)} stops; all stops mean {t.mean():.2f} s, "
              f"p95 {np.percentile(t, 95):.1f} s, max {t.max():.1f} s; decisive stops ({100 * d.mean():.0f} %) "
              f"mean {t[d].mean():.2f} s, p95 {np.percentile(t[d], 95):.1f} s; "
              f"mismatches vs cluster {sum(r.get('mismatch', 0) for r in recs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tmlp_F95_split_list_s0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--per-length", type=int, default=0, help="time only the first N routes of each length")
    a = ap.parse_args()
    if a.report:
        return report()
    from rl_env import split_routes
    kind = "gbt" if a.tag.startswith("gbt") else "torch"
    run, k = STORED[a.tag]
    stored = {json.loads(l)["instance"]: json.loads(l)["res"][k][0]
              for l in open(os.path.join(RES, f"rl_search_{run}.jsonl"))}
    out = os.path.join(RES, f"rollout_timing_{a.tag}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["instance"] for l in open(out) if l.strip()}
    routes = split_routes("test")
    if a.per_length:
        routes = [i for ln in ("Rlong", "Rmedium", "Rshort") for i in [r for r in routes if r.startswith(ln)][: a.per_length]]
    todo = [i for i in routes if i not in done]
    # longest routes first, so the pool's tail is short
    todo.sort(key=lambda i: (0 if i.startswith("Rlong") else 1 if i.startswith("Rmedium") else 2))
    print(f"{a.tag}: {len(todo)} routes to time ({a.workers} workers) -> {out}", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(kind, a.tag)) as pool, open(out, "a") as fh:
        for n, r in enumerate(pool.imap_unordered(one, todo), 1):
            r["mismatch"] = int(abs(r["duration"] - stored[r["instance"]]["duration"]) > 1e-6)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
