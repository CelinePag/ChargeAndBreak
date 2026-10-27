"""
rl_policy.py — the improved policies of the RL work, as rl_env choosers
=======================================================================
ApiChooser: the student corrected by a policy-iteration model G
(rl_api_fit.py).  At a DECISIVE state (rl_api.decisive: a rest contested, or
the two best candidates close; the same rule the labels were collected
under) it scores the same candidates as the collector (rl_api.candidates,
the student's pick first) by s_j + G(x_j) and moves off the pick only if that
is lower by more than `min_gain` hours.  Elsewhere it is the student.  It
keeps no state, so one instance serves every vehicle of a batch.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:                    # `src` on machines without the editable install
    sys.path.insert(0, _ROOT)

import lightgbm as lgb                                               # noqa: E402

from features import Precomp                                         # noqa: E402
from rl_api import candidates, decisive, walk_feats                  # noqa: E402
from rl_api_fit import g_inputs                                      # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


class RolloutChooser:
    """Decisive-state rollout: policy improvement over the student at run time,
    no model, no solver (ML/RL_THEORY.md §7.2, acted on directly).

    At a decisive state (rl_api.decisive) the candidates of rl_api.candidates
    (the student's pick first) are each executed and the student drives on to
    the destination under `futures` shared fresh draws; the challenger with
    the lowest mean cost difference d to the pick is taken if
    d + z * se < -min_gain.  Elsewhere it is the student.

    Why this and not rollout_policy.py (which made routes worse, +7.6 %):
    that one rolled out EVERY decision with 8-16 scenarios, top-3 by score;
    here only the ~5-10 decisive states of a route, 32 futures, the best rest
    always tested.  rl_api labels made this way replicate on independent
    futures (corr 0.95 near the pick; acting on run A's best challenger is
    worth -0.18 ± 0.09 h per state measured on run B, 2026-10-26).

    Set .cv and .route (a name for seeding) before each route; .calls and
    .switches count rollouts and changed decisions."""

    def __init__(self, base, futures=32, topk=4, contest=2.0, margin=0.25,
                 z=0.0, min_gain=0.0, seed=0, race=0, race_z=2.0,
                 record=None, follow="expert"):
        self.base, self.futures, self.topk = base, int(futures), int(topk)
        self.contest, self.margin = float(contest), float(margin)
        self.z, self.min_gain, self.seed = float(z), float(min_gain), int(seed)
        # race > 0: futures in rounds of `race`; after each round, drop
        # challengers whose lower bound mean - race_z*se is above 0 (clearly
        # not better than the pick); stop early when none is left.  The final
        # decision rule is unchanged (mean + z*se < -min_gain over all the
        # futures a surviving challenger saw).  race = 0: all futures at once.
        self.race, self.race_z = int(race), float(race_z)
        # record: a list to append every evaluated decision to (expert
        # iteration labels, rl_exit_collect.py): the candidates' feature rows,
        # keys and student scores, and each one's measured mean cost minus the
        # pick's (d, the pick first with 0), its se and number of futures.
        # follow "student": execute the student's own pick, so the labels are
        # taken at the states the student visits (DAgger); "expert": the
        # rollout's choice (the rollout policy itself).
        self.record, self.follow = record, follow
        self.cv, self.route = 0.15, ""
        self.calls = self.switches = self.rollouts = 0

    def _costs(self, fd, pre, veh, keys, fut):
        from rl_env import ForceFirst, run_batch
        from rollout_policy import clone
        jobs, start = [], []
        for key in keys:
            for w in fut:
                jobs.append((ForceFirst(key), w))
                start.append(clone(veh))
        res = run_batch(self.base, fd, self.cv, jobs, pre=pre, start=start)
        self.rollouts += len(jobs)
        M = len(fut)
        return np.array([[np.nan if r["failed"] else r["cost"] for r in res[a * M:(a + 1) * M]]
                         for a in range(len(keys))])

    def choose(self, fd, pre, veh, legal, rows, score, flags):
        from features import action_key
        from rl_env import fresh_draws
        kind, order = decisive(legal, score, self.contest, self.margin)
        j0 = int(np.argmin(score))
        if kind is None:
            return j0
        cand = candidates(legal, order, self.topk)
        cand = [j0] + [j for j in cand if j != j0]
        if len(cand) < 2:
            return j0
        keys = [action_key(legal[j].get("y", 0), legal[j].get("break_type"),
                           legal[j].get("rest_type")) for j in cand]
        # seeded by route, stop and the state itself, so two vehicles that
        # reach the same stop in different states get different futures
        fut = fresh_draws(fd, self.cv, self.futures,
                          f"{self.route}|dsr|{self.seed}|{veh.stop}|{veh.t_arr:.4f}|{veh.e_arr:.2f}")
        self.calls += 1
        step = self.race if self.race > 0 else len(fut)
        alive = list(range(1, len(keys)))           # challengers still in the race
        d_all = {a: [] for a in alive}
        for b0 in range(0, len(fut), step):
            ks = [0] + alive
            cost = self._costs(fd, pre, veh, [keys[a] for a in ks], fut[b0:b0 + step])
            fail = np.isnan(cost)
            ref = np.nanmean(cost[0]) if (~fail[0]).any() else np.nanmean(cost)
            cost = np.where(fail, ref + 24.0, cost)
            for r, a in enumerate(alive, 1):
                d_all[a] += list(cost[r] - cost[0])
            if b0 + step >= len(fut):
                break
            keep = []
            for a in alive:
                x = np.array(d_all[a])
                lo = x.mean() - self.race_z * (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.inf)
                if lo <= 0:
                    keep.append(a)
            alive = keep
            if not alive:
                break
        best, best_val = None, None
        for a in alive:
            x = np.array(d_all[a])
            val = x.mean() + self.z * x.std(ddof=1) / np.sqrt(len(x))
            if best_val is None or val < best_val:
                best, best_val = a, val
        choice = cand[best] if (best is not None and best_val < -self.min_gain) else j0
        if self.record is not None:
            n = [0] + [len(d_all[a]) for a in range(1, len(keys))]
            dm = [0.0] + [float(np.mean(d_all[a])) for a in range(1, len(keys))]
            ds = [0.0] + [float(np.std(d_all[a], ddof=1) / np.sqrt(len(d_all[a])))
                          if len(d_all[a]) > 1 else 0.0 for a in range(1, len(keys))]
            self.record.append(dict(
                stop=int(veh.stop), kind=kind, keys=keys,
                rows=np.asarray(rows[cand], dtype=np.float32),
                score=np.asarray(score[cand], dtype=np.float32),
                d_mean=np.array(dm, dtype=np.float32), d_se=np.array(ds, dtype=np.float32),
                n_fut=np.array(n, dtype=np.int16), choice=cand.index(choice)))
        if choice != j0:
            self.switches += 1
        return j0 if self.follow == "student" else choice


class ApiChooser:
    def __init__(self, tag, min_gain=0.0, models_dir=MODELS):
        with open(os.path.join(models_dir, f"{tag}_meta.json")) as fh:
            self.meta = json.load(fh)
        self.Gs = lgb.Booster(model_file=os.path.join(models_dir, f"{tag}_Gs.txt"))
        self.Gr = lgb.Booster(model_file=os.path.join(models_dir, f"{tag}_Gr.txt"))
        self.rest_h = float(self.meta["rest_h"])
        self.tag = tag
        self.min_gain = float(min_gain)
        self.contest = float(self.meta["contest"])
        self.margin = float(self.meta["margin"])
        self.topk = int(self.meta["topk"])

    def choose(self, fd, pre, veh, legal, rows, score, flags):
        kind, order = decisive(legal, score, self.contest, self.margin)
        j0 = int(np.argmin(score))
        if kind is None:
            return j0
        cand = candidates(legal, order, self.topk)
        cand = [j0] + [j for j in cand if j != j0]
        if len(cand) < 2:
            return j0
        walk = np.array([walk_feats(fd, pre, veh, legal[j]) for j in cand], dtype=float)
        x = g_inputs(rows[cand], score[cand].astype(float), walk)
        # corrected advantage of each challenger over the pick (the pick: 0)
        adj = ((score[cand[1:]] - score[j0]) + self.Gs.predict(x)
               + self.rest_h * self.Gr.predict(x))
        i = int(np.argmin(adj))
        if adj[i] < -self.min_gain:
            return cand[1 + i]
        return j0
