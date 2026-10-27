"""
rl_search.py — step 1 of ML/RL_THEORY.md: direct policy search with common random numbers
==========================================================================================
The policy family is the student plus a few interpretable correction terms,

    score'(s, a) = score_student(s, a) + sum_t theta_t * g_t(s, a),

argmin over the student's own shielded candidates (so the shield, the charge
clamp and the charge head are untouched).  Every variant drives every route
under the SAME fresh travel-time draws as the plain student (rl_env.run_batch
holds them in one lockstep batch), so a paired difference is due to the
decisions alone, and the objective is deterministic given the draws (PEGASUS,
Ng & Jordan 2000).

Correction terms (all in hours):
  rest   1{a is a daily rest}.  Tests the FPI diagnostic: the LA-trained head
         overprices "rest now" by ~1.2 h on long routes.
  win    a_cu1_late + a_cu1_early: the action's predicted window miss at the
         next customer (the student's own action features).
  nr     total daily rests to the destination after a (this one + a nominal
         drive-only walk, fpi_policy.walk_to_end), minus the fewest over the
         candidates: a whole-route view the 24 h horizon lacks.
  slack  max(0, m - driving left in the final shift after a), same walk.
nr and slack only change rest vs no-rest comparisons, so they are evaluated
only where a rest is CONTESTED: the best rest within `contest` hours (default
2) of the best non-rest candidate.  Elsewhere the student is left alone.

Output: one JSON line per route in ML/results/rl_search_<name>.jsonl, every
variant x draw summarised (rl_env._summary).  Resumable: routes already in
the file are skipped.  rl_report.py reads it.

    python ML/code/rl_search.py --kind gbt --tag gbt_F95_base_s0 --split fit \
        --draws 2 --grid v1 --name v1_gbt --workers 4
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

from fpi_policy import walk_to_end                                   # noqa: E402
from policy_core import _norm                                        # noqa: E402
from rl_env import (final_student, fresh_draws, load_route, realised,  # noqa: E402
                    rests_needed, route_set, run_batch)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
KEEP = ("failed", "cost", "dwell", "duration", "misses", "rests", "r1", "charges",
        "changed", "overrun", "decisions")


class Corrected:
    """The student plus theta * g; see the module docstring."""

    def __init__(self, base, theta: dict, contest=2.0):
        self.th = {k: float(v) for k, v in theta.items() if k not in ("m", "far")}
        # "fm" (free-rest slack margin) stays in self.th: _free_rest reads it
        self.m = float(theta.get("m", 0.5))
        self.far = float(theta.get("far", 13.5))
        self.contest = float(contest)
        names = list(base.state_names) + list(base.action_names)
        self.ix = {n: names.index(n) for n in
                   ("a_r1", "a_r2", "a_cu1_late", "a_cu1_early", "drive_left")}

    def _free_rest(self, fd, pre, veh, legal, score, rest, adj):
        """The truncation-bias fix (rl_api labels, 2026-10-26): where the LA's
        24 h horizon holds no mandatory rest, its cost head prices "rest now"
        as a whole extra rest (~+10 h) even when the route needs the same
        number of rests either way.  If resting now does not raise the total
        (rl_env.rests_needed, counting the 10 h days) and leaves at least `fm`
        hours of driving slack in the final shift, move the rests' scores down
        so the best rest costs `free` + `kappa` x (driving allowance this
        shift gives up) hours more than the best non-rest.  kappa prices an
        hour of slack: with kappa = 0 the policy rests at the first count-
        neutral stop, which wastes slack (smoke test, 2026-10-26)."""
        ok = score < 1e5
        r_ok, n_ok = ok & (rest > 0), ok & (rest == 0)
        if not (r_ok.any() and n_ok.any()):
            return adj
        jr = int(np.where(r_ok)[0][np.argmin(score[r_ok])])
        jn = int(np.where(n_ok)[0][np.argmin(score[n_ok])])
        drive_left = float(pre.totD - pre.cumD[veh.stop])
        ext_left = int(fd.get("ext_bar", 2)) - int(veh.ext_shift_used)
        ext_r = ext_left - int(float(veh.sd) > float(fd["Tdrv_sh1"]) + 1e-3)
        n_n, _ = rests_needed(fd, drive_left, float(veh.sd), ext_left)
        n_r, slack_r = rests_needed(fd, drive_left, 0.0, ext_r)
        if 1 + n_r <= n_n and slack_r >= self.th.get("fm", 1.0):
            sd_lim = (float(fd.get("Tdrv_sh2", 10.0)) if ext_left > 0
                      else float(fd["Tdrv_sh1"]))
            lost = max(0.0, sd_lim - float(veh.sd))
            shift = score[jr] - (score[jn] + self.th["free"]
                                 + self.th.get("kappa", 0.0) * lost)
            if shift > 0:
                adj = adj - shift * rest
        return adj

    def choose(self, fd, pre, veh, legal, rows, score, flags):
        rest = rows[:, self.ix["a_r1"]] + rows[:, self.ix["a_r2"]]
        adj = score.astype(float).copy()
        if self.th.get("rest"):
            adj += self.th["rest"] * rest
        if self.th.get("restfar") or self.th.get("restnear"):
            # nominal driving still ahead of this stop (a state column, the
            # same on every row): "far" = at least `far` hours left
            far = float(rows[0, self.ix["drive_left"]]) >= self.far
            adj += (self.th.get("restfar", 0.0) if far
                    else self.th.get("restnear", 0.0)) * rest
        if "free" in self.th:
            adj = self._free_rest(fd, pre, veh, legal, score, rest, adj)
        if self.th.get("win"):
            adj += self.th["win"] * (rows[:, self.ix["a_cu1_late"]]
                                     + rows[:, self.ix["a_cu1_early"]])
        if (self.th.get("nr") or self.th.get("slack")) and rest.any() and (~rest.astype(bool)).any():
            ok = score < 1e5
            r_ok, n_ok = ok & (rest > 0), ok & (rest == 0)
            if r_ok.any() and n_ok.any() and score[r_ok].min() - score[n_ok].min() < self.contest:
                tot = np.empty(len(legal))
                sl = np.empty(len(legal))
                for j, a in enumerate(legal):
                    is_r = rest[j] > 0
                    brk = _norm(a.get("break_type"))
                    cd = 0.0 if (is_r or brk in ("b45", "b30")) else float(veh.cd)
                    sd = 0.0 if is_r else float(veh.sd)
                    nr, slack = walk_to_end(fd, pre, veh.stop, cd, sd)
                    tot[j] = nr + int(is_r)
                    sl[j] = slack
                if self.th.get("nr"):
                    adj += self.th["nr"] * (tot - tot[ok].min())
                if self.th.get("slack"):
                    adj += self.th["slack"] * np.maximum(0.0, self.m - sl)
        return int(np.argmin(adj))


GRIDS = {
    "v1": ([dict(rest=x) for x in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)]
           + [dict(win=x) for x in (0.1, 0.25, 0.5, 1.0)]
           + [dict(nr=x) for x in (0.5, 1.0, 2.0, 5.0)]
           + [dict(slack=t, m=m) for m, t in ((0.3, 1.0), (0.5, 1.0), (0.5, 3.0), (1.0, 1.0))]),
    # after the 8-route smoke test: nr never changed a decision (a drive-only
    # walk rarely counts a different number of rests for "rest now" and
    # "rest a few stops later"), so it is dropped; the rest offset is swept
    # finer and split by how much driving is left (far: >= 13.5 h)
    "v2": ([dict(rest=x) for x in (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5)]
           + [dict(restfar=x) for x in (-1.0, -0.5, -0.25)]
           + [dict(restnear=x) for x in (-0.5, -0.25, 0.25)]
           + [dict(win=x) for x in (0.1, 0.25, 0.5)]
           + [dict(slack=t, m=m) for m, t in ((0.5, 1.0), (1.0, 3.0))]),
    # the truncation-bias fix found in the first rl_api labels: a rest that
    # does not raise the route's rest count is not an extra 10 h
    # (v1 of this grid had no kappa: free >= 0 never fired, free = 0 rested
    # at the first count-neutral stop; kappa grid to be set from the labels)
    "free": [dict(free=f, kappa=k, fm=1.0) for f in (-0.25, 0.0) for k in (0.25, 0.5, 1.0)],
}


def vname(th):
    return ("+".join(f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}:{v}"
                     for k, v in th.items()) if th else "student")


_G = {}


class NetChooser:
    """Another student's cost model over the SAME shielded candidates (rows
    are the same feature columns): argmin of its cost, with the base's
    feasibility veto (score >= 1e5).  The charge duration still comes from
    the base's head, so this compares cost models only -- what rl_torch_ft
    changes."""

    def __init__(self, spec):
        kind, tag = spec.split(":", 1)
        from policy_core import load_policy
        self.pol = load_policy(kind, tag)

    def choose(self, fd, pre, veh, legal, rows, score, flags):
        c, _f = self.pol._predict(rows)
        return int(np.argmin(np.asarray(c, dtype=float) + 1e6 * (score >= 1e5)))


def make_chooser(base, th, contest):
    """A theta dict -> chooser: {"api": tag, "min_gain": h} is a policy-
    iteration chooser (rl_policy.ApiChooser); {"net": "kind:tag"} another
    cost model (NetChooser); anything else is Corrected."""
    if "api" in th:
        from rl_policy import ApiChooser
        return ApiChooser(th["api"], min_gain=th.get("min_gain", 0.0))
    if "dsr" in th:
        from rl_policy import RolloutChooser
        return RolloutChooser(base, futures=int(th["dsr"]), z=th.get("z", 0.0),
                              min_gain=th.get("min_gain", 0.0), contest=contest,
                              race=int(th.get("race", 0)))
    if "net" in th:
        return NetChooser(th["net"])
    return Corrected(base, th, contest)


def _init(kind, tag, grid, draws, contest, seed, realised_too):
    base = final_student(kind, tag)
    _G.update(base=base, grid=grid, draws=draws, contest=contest, seed=seed,
              realised_too=realised_too,
              choosers=[None] + [make_chooser(base, th, contest) for th in grid])


def one(job):
    variant, inst, path = job
    t0 = time.perf_counter()
    fd, D, E, cv = load_route(inst, path=path)
    ds = fresh_draws(fd, cv, _G["draws"], f"{inst}|rl1|{_G['seed']}")
    if _G["realised_too"]:
        ds = [realised(D, E)] + ds
    for ch in _G["choosers"]:
        if hasattr(ch, "route"):            # RolloutChooser: per-route settings
            ch.route, ch.cv = inst, cv
    jobs = [(ch, d) for ch in _G["choosers"] for d in ds]
    res = run_batch(_G["base"], fd, cv, jobs)
    k = len(ds)
    out = [[{f: r[f] for f in KEEP} for r in res[v * k:(v + 1) * k]]
           for v in range(len(_G["choosers"]))]
    return dict(instance=inst, variant=variant, n_draws=k,
                realised_first=bool(_G["realised_too"]),
                variants=["student"] + [vname(th) for th in _G["grid"]],
                res=out, seconds=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gbt")
    ap.add_argument("--tag", default="gbt_F95_base_s0")
    ap.add_argument("--split", default="fit", choices=["fit", "stop", "test"])
    ap.add_argument("--routes", default="base", choices=["base", "mixed"],
                    help="base-case routes, or the mixed-charger routes of mixed_eval "
                         "(--variants; --split test or stop = its validation set)")
    ap.add_argument("--variants", default="pmix,mix")
    ap.add_argument("--lengths", default="short,medium,long")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--realised", action="store_true",
                    help="also drive the stored realisation (validation / test)")
    ap.add_argument("--grid", default="v1")
    ap.add_argument("--theta", default=None,
                    help="JSON list of theta dicts instead of a named grid")
    ap.add_argument("--contest", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.split == "test":
        print("[rl_search] NOTE: test split -- run once, for the frozen choice only")

    grid = json.loads(a.theta) if a.theta else GRIDS[a.grid]
    lens = tuple("R" + x for x in a.lengths.split(","))
    routes = [j for j in route_set(a.routes, a.split, a.variants) if j[1].startswith(lens)]
    if a.limit:
        routes = routes[: a.limit]
    out = os.path.join(RESULTS, f"rl_search_{a.name}.jsonl")
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            done = {json.loads(l)["instance"] for l in fh if l.strip()}
    todo = [j for j in routes if j[1] not in done]
    print(f"[rl_search] {a.tag} on {a.routes}/{a.split}: {len(routes)} routes, {len(todo)} "
          f"to do, {len(grid) + 1} variants x {a.draws + int(a.realised)} draws -> {out}",
          flush=True)
    meta = dict(kind=a.kind, tag=a.tag, split=a.split, routes=a.routes, variants=a.variants,
                draws=a.draws, realised=a.realised, grid=grid, contest=a.contest, seed=a.seed)
    with open(out + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    t0 = time.time()
    with Pool(a.workers, initializer=_init,
              initargs=(a.kind, a.tag, grid, a.draws, a.contest, a.seed, a.realised)) as pool, \
            open(out, "a") as fh:
        for n, rec in enumerate(pool.imap_unordered(one, todo), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / 60:.1f} min  "
                      f"(~{el / n * (len(todo) - n) / 60:.0f} min left)", flush=True)


if __name__ == "__main__":
    main()
