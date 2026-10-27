"""
rl_api.py — step 2 of ML/RL_THEORY.md: targeted, many-rollout labels for approximate policy iteration
=====================================================================================================
FPI round 1 (fpi_collect.py) labelled ~70k decisions with ONE shared draw per
branch: label sd 2.1 h against effects of tenths of an hour, on 3 % of rows
that mattered, and its correction learned nothing (R^2 0.02).  This collector
spends the same simulation budget where it can pay:

  roll-in   the current policy (round 1: the student) drives a training route
            under one fresh draw; every decision is snapshotted;
  states    only DECISIVE decisions are kept: a daily rest is contested (the
            best rest within `contest` h of the best non-rest candidate) or the
            two best candidates are within `margin` h; at most `per_route`
            per route, sampled reproducibly, plus `n_random` other decisions
            as a calibration set;
  labels    at each kept state, the `topk` best candidates by the policy's own
            score (always including the best rest and the best non-rest) are
            each executed and the policy drives on to the destination, under
            `futures` shared fresh draws (common random numbers): cost matrix
            [candidate, future]; a rule break counts FAIL_H.

Saved per route: ML/data/rl_api_<name>/<instance>.npz with, per state, the
candidates' feature rows, keys, policy scores, the policy's own pick, the mean
and se of each candidate's cost minus the pick's cost, plus the route-end walk
features (fpi_policy.walk_to_end) used by the correction.  Resumable.

    python ML/code/rl_api.py --kind gbt --tag gbt_F95_base_s0 --split fit \
        --name r1_gbt --futures 32 --per-route 8 --workers 4
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

from features import Precomp, action_key                             # noqa: E402
from fpi_policy import walk_to_end                                   # noqa: E402
from policy_core import _norm                                        # noqa: E402
from rl_env import (ForceFirst, crc, final_student, fresh_draws,      # noqa: E402
                    load_route, rests_needed, run_batch, split_routes)
from rollout_policy import clone                                     # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FAIL_H = 24.0      # added to the pick's mean cost for a candidate run that breaks a rule


def _is_rest(a):
    return _norm(a.get("rest_type")) in ("r1", "r2")


def decisive(legal, score, contest, margin):
    """(kind, candidate order) or (None, order): 'rest' if a rest is contested,
    'close' if the two best candidates are within `margin` hours."""
    ok = np.where(score < 1e5)[0]
    order = ok[np.argsort(score[ok], kind="stable")]
    if len(order) < 2:
        return None, order
    r = [j for j in order if _is_rest(legal[j])]
    n = [j for j in order if not _is_rest(legal[j])]
    if r and n and score[r[0]] - score[n[0]] < contest:
        return "rest", order
    if score[order[1]] - score[order[0]] < margin:
        return "close", order
    return None, order


def candidates(legal, order, topk):
    """The `topk` best, forced to include the best rest and best non-rest."""
    pick = list(order[:topk])
    for want in (_is_rest, lambda a: not _is_rest(a)):
        if not any(want(legal[j]) for j in pick):
            extra = [j for j in order if want(legal[j])]
            if extra:
                pick[-1] = extra[0]
    return [int(j) for j in dict.fromkeys(pick)]


def walk_feats(fd, pre, veh, a):
    """Route-end features after action a: (total rests from here incl. this
    one, final-shift slack) by rl_env.rests_needed (counts the 10 h days),
    then the same by the old drive-only walk (fpi_policy.walk_to_end)."""
    rest = _is_rest(a)
    brk = _norm(a.get("break_type"))
    cd = 0.0 if (rest or brk in ("b45", "b30")) else float(veh.cd)
    sd = 0.0 if rest else float(veh.sd)
    drive_left = float(pre.totD - pre.cumD[veh.stop])
    ext_left = int(fd.get("ext_bar", 2)) - int(veh.ext_shift_used)
    if rest and float(veh.sd) > float(fd["Tdrv_sh1"]) + 1e-3:
        ext_left -= 1                      # this rest closes an extended day
    n, slack = rests_needed(fd, drive_left, sd, ext_left)
    nr, slack_w = walk_to_end(fd, pre, veh.stop, cd, sd)
    return n + int(rest), slack, nr + int(rest), slack_w


_G = {}


def _init(kind, tag, cfg):
    _G.update(base=final_student(kind, tag), cfg=cfg)


def one(inst):
    base, c = _G["base"], _G["cfg"]
    out_f = os.path.join(c["dir"], inst + ".npz")
    if os.path.exists(out_f):
        return inst, -1, 0.0
    t0 = time.perf_counter()
    fd, _D, _E, cv = load_route(inst)
    pre = Precomp(fd)

    # -- roll-in: snapshot every decision ----------------------------------
    snaps = []

    def hook(i, veh, legal, rows, score, flags, j):
        kind, order = decisive(legal, score, c["contest"], c["margin"])
        snaps.append((kind, clone(veh), legal, rows.copy(), score.copy(), int(j), order))

    w_in = fresh_draws(fd, cv, 1, f"{inst}|api|{c['seed']}|in")[0]
    run_batch(base, fd, cv, [(None, w_in)], pre=pre, on_decision=hook)

    rng = np.random.default_rng(crc(f"{inst}|api|{c['seed']}|pick"))
    dec = [k for k, s in enumerate(snaps) if s[0] is not None]
    oth = [k for k, s in enumerate(snaps) if s[0] is None and len(s[6]) >= 2]
    keep = list(rng.permutation(dec)[: c["per_route"]]) if dec else []
    if oth and c["n_random"]:
        keep += list(rng.choice(oth, size=min(c["n_random"], len(oth)), replace=False))

    recs = []
    for k in sorted(int(x) for x in keep):
        kind, veh, legal, rows, score, j0, order = snaps[k]
        cand = candidates(legal, order, c["topk"])
        if j0 not in cand:
            cand = [j0] + cand[:-1]
        cand = [j0] + [j for j in cand if j != j0]          # the policy's pick first
        keys = [action_key(legal[j].get("y", 0), legal[j].get("break_type"),
                           legal[j].get("rest_type")) for j in cand]
        fs = c["seed"] if c.get("futures_seed") is None else c["futures_seed"]
        tag = (f"{inst}|api|{c['seed']}|{veh.stop}" if fs == c["seed"]
               else f"{inst}|api|{c['seed']}|{veh.stop}|f{fs}")
        fut = fresh_draws(fd, cv, c["futures"], tag)
        jobs, start = [], []
        for key in keys:
            for w in fut:
                jobs.append((ForceFirst(key), w))
                start.append(clone(veh))
        res = run_batch(base, fd, cv, jobs, pre=pre, start=start)
        M = len(fut)
        cost = np.array([[np.nan if r["failed"] else r["cost"] for r in res[a * M:(a + 1) * M]]
                         for a in range(len(keys))])
        rests = np.array([[r["rests"] for r in res[a * M:(a + 1) * M]] for a in range(len(keys))])
        fail = np.isnan(cost)
        ref = np.nanmean(cost[0]) if (~fail[0]).any() else np.nanmean(cost)
        cost = np.where(fail, ref + FAIL_H, cost)
        d = cost - cost[0]                                   # paired vs the pick
        wf = np.array([walk_feats(fd, pre, veh, legal[j]) for j in cand])
        recs.append(dict(
            stop=int(veh.stop), kind=kind, keys=keys,
            rows=rows[cand].astype(np.float32), score=score[cand].astype(np.float32),
            d_mean=d.mean(axis=1), d_se=d.std(axis=1, ddof=1) / np.sqrt(M),
            cost_mean=cost.mean(axis=1), rests_mean=rests.mean(axis=1),
            fails=fail.sum(axis=1), walk=wf.astype(np.float32),
            cost=cost.astype(np.float32), rests_all=rests.astype(np.int8)))

    np.savez_compressed(out_f, instance=inst, n_decisions=len(snaps),
                        recs=np.array(recs, dtype=object))   # load with allow_pickle=True
    return inst, len(recs), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gbt")
    ap.add_argument("--tag", default="gbt_F95_base_s0")
    ap.add_argument("--split", default="fit", choices=["fit", "stop"])
    ap.add_argument("--lengths", default="short,medium,long")
    ap.add_argument("--name", required=True)
    ap.add_argument("--futures", type=int, default=32)
    ap.add_argument("--per-route", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=1)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--contest", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--futures-seed", type=int, default=None,
                    help="draw the futures from another seed (same roll-in, so the "
                         "same states): a replication of an earlier run's labels")
    ap.add_argument("--routes-from", default=None,
                    help="label only the routes already in rl_api_<name>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_dir = os.path.join(DATA, f"rl_api_{a.name}")
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(dir=out_dir, futures=a.futures, per_route=a.per_route, n_random=a.n_random,
               topk=a.topk, contest=a.contest, margin=a.margin, seed=a.seed,
               futures_seed=a.futures_seed)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(dict(kind=a.kind, tag=a.tag, split=a.split, **{k: v for k, v in cfg.items()
                                                                  if k != "dir"}), fh, indent=1)
    lens = tuple("R" + x for x in a.lengths.split(","))
    routes = [r for r in split_routes(a.split) if r.startswith(lens)]
    if a.routes_from:
        have = {os.path.basename(f)[:-4] for f in
                os.listdir(os.path.join(DATA, f"rl_api_{a.routes_from}")) if f.endswith(".npz")}
        routes = [r for r in routes if r in have]
    if a.limit:
        routes = routes[: a.limit]
    todo = [r for r in routes if not os.path.exists(os.path.join(out_dir, r + ".npz"))]
    print(f"[rl_api] {a.tag} on {a.split}: {len(routes)} routes, {len(todo)} to do -> {out_dir}",
          flush=True)
    t0 = time.time()
    n_states = 0
    with Pool(a.workers, initializer=_init, initargs=(a.kind, a.tag, cfg)) as pool:
        for n, (inst, ns, sec) in enumerate(pool.imap_unordered(one, todo), 1):
            n_states += max(ns, 0)
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} routes, {n_states} states, {el / 60:.1f} min "
                      f"(~{el / n * (len(todo) - n) / 60:.0f} min left)", flush=True)


if __name__ == "__main__":
    main()
