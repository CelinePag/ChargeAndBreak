"""
rl_exit_collect.py — expert-iteration labels: the rollout, at the student's own states
======================================================================================
Step 2 of the RL work (ML/RL_THEORY.md §13): learn the decisions the
decisive-state rollout makes, so the student takes them in milliseconds.

The current student drives each training route under `draws` fresh travel-
time draws (never the stored realisation).  At EVERY decisive stop
(rl_api.decisive) the rollout of rl_policy.RolloutChooser scores the
student's candidates -- each executed, the student driving on, 32 shared
futures raced in rounds of 8, exactly the tested rollout policy -- and the
student's own pick is executed (follow="student"), so the labels are taken
on the states the student visits, as DAgger prescribes.  Saved per route:
ML/data/rl_exit_<name>/<instance>.npz, the rl_api record format (rows,
score, keys, d_mean / d_se with the pick first at 0, n_fut, the rollout's
choice), which rl_torch_ft.py reads.  Resumable.

    python ML/code/rl_exit_collect.py --kind torch --tag tmlp_F95_split_list_s0 \
        --split fit --draws 2 --name r1 --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:                    # `src` on machines without the editable install
    sys.path.insert(0, _ROOT)

from rl_env import final_student, fresh_draws, load_route, route_set, run_batch  # noqa: E402
from rl_policy import RolloutChooser                                 # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_G = {}


def _init(kind, tag, cfg):
    _G.update(base=final_student(kind, tag), cfg=cfg)


def one(job):
    variant, inst, path = job
    c = _G["cfg"]
    out_f = os.path.join(c["dir"], inst + ".npz")
    if os.path.exists(out_f):
        return inst, -1, 0.0
    t0 = time.perf_counter()
    fd, _D, _E, cv = load_route(inst, path=path)
    draws = fresh_draws(fd, cv, c["draws"], f"{inst}|exit|{c['seed']}")
    chs, recs = [], []
    for k in range(len(draws)):
        rec = []
        ch = RolloutChooser(_G["base"], futures=c["futures"], race=c["race"],
                            seed=1000 * c["seed"] + k, record=rec, follow="student")
        ch.route, ch.cv = inst, cv
        chs.append(ch)
        recs.append(rec)
    res = run_batch(_G["base"], fd, cv, list(zip(chs, draws)))
    out = []
    for k, rec in enumerate(recs):
        for r in rec:
            r["draw"] = k
            out.append(r)
    tmp = out_f + ".tmp.npz"
    np.savez_compressed(tmp, instance=inst, variant=variant,
                        n_decisions=np.array([r["decisions"] for r in res]),
                        recs=np.array(out, dtype=object))
    os.replace(tmp, out_f)                   # never leave a half-written file
    return inst, len(out), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="torch")
    ap.add_argument("--tag", default="tmlp_F95_split_list_s0")
    ap.add_argument("--routes", default="base", choices=["base", "mixed"])
    ap.add_argument("--variants", default="pmix,mix")
    ap.add_argument("--split", default="fit", choices=["fit"],
                    help="training routes only: labels are training data")
    ap.add_argument("--lengths", default="short,medium,long")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--futures", type=int, default=32)
    ap.add_argument("--race", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.routes == "mixed":
        raise SystemExit("mixed training routes: not wired yet (mixed_eval has no fit split)")
    out_dir = os.path.join(DATA, f"rl_exit_{a.name}")
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(dir=out_dir, draws=a.draws, futures=a.futures, race=a.race, seed=a.seed)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(dict(kind=a.kind, tag=a.tag, routes=a.routes, split=a.split,
                       **{k: v for k, v in cfg.items() if k != "dir"}), fh, indent=1)
    lens = tuple("R" + x for x in a.lengths.split(","))
    jobs = [j for j in route_set(a.routes, a.split, a.variants) if j[1].startswith(lens)]
    if a.limit:
        jobs = jobs[: a.limit]
    todo = [j for j in jobs if not os.path.exists(os.path.join(out_dir, j[1] + ".npz"))]
    print(f"[rl_exit_collect] {a.tag} on {a.routes}/{a.split}: {len(jobs)} routes, "
          f"{len(todo)} to do, {a.draws} draws each -> {out_dir}", flush=True)
    t0, n_states = time.time(), 0
    with Pool(a.workers, initializer=_init, initargs=(a.kind, a.tag, cfg)) as pool:
        for n, (inst, ns, _sec) in enumerate(pool.imap_unordered(one, todo), 1):
            n_states += max(ns, 0)
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} routes, {n_states} labelled states, {el / 60:.1f} min "
                      f"(~{el / n * (len(todo) - n) / 60:.0f} min left)", flush=True)


if __name__ == "__main__":
    main()
