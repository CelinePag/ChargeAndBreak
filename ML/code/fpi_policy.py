"""
fpi_policy.py — one round of fitted policy iteration on top of the tree student
===============================================================================
Why.  The student learns the LA's estimate of what each action costs, so at
best it matches the LA.  Beating it needs better estimates than the LA has.
This module gets them from the student itself:

  1. drive a route with the student (the "roll-in"), under travel times drawn
     from the instance's own distribution -- never its realised times;
  2. at a stop, try each of the student's k best actions and let the student
     drive every branch to the destination under ONE shared draw of the
     remaining travel times (common random numbers), so the branches differ
     only by the action;
  3. record each branch's route cost (arrival + BETA per window miss) minus
     the cost of the branch that took the student's own choice.

fpi_collect.py does that for thousands of stops, fpi_train.py fits a boosted
correction G to (the measured advantage) minus (the advantage the student's
cost head predicted), and FPIPolicy below picks, among the student's k best
actions, the one with the lowest corrected score.  G can only move the student
where the simulated outcomes disagree with its cost head consistently across
many stops: with no evidence it stays near zero and the student is unchanged.

This is approximate policy iteration (Lagoudakis & Parr 2003; with tree
ensembles, Ernst, Geurts & Wehenkel 2005), and the setting of Chang et al.
2015, where training on the learner's own simulated outcomes is what lets it
beat its teacher.  One round targets the same improvement as rollout_policy.py
-- which could not prove any at single stops from 16 draws -- but pools the
evidence over thousands of stops instead.

G sees two features the student's heads do not have, from a nominal walk to
the destination with the same accounting as features._walk_forward: the daily
rests the route still needs after the action, and the daily driving left in
the final shift.  The LA never looks past 24 h, so its labels could not teach
a model to use them; these labels can.  The LA's extra-rest routes overflow
their final shift by minutes, which is what the second one measures.

Speed.  Every branch is the student driving to the end, ~3 ms per stop.  All
vehicles advance exactly one stop per decision, so drive_fleet() moves a whole
batch of branches in lockstep and scores all their rows in ONE LightGBM call
per stop: identical decisions (rows are predicted independently), a fraction
of the per-call overhead.  Nothing here writes outside ML/.
"""
from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb                                              # noqa: E402

from features import action_key                                     # noqa: E402
from policy_core import MODELS, _norm                               # noqa: E402
from rollout_policy import BETA, FAIL_H, step                       # noqa: E402

EXTRA = ("rt_rests_left", "rt_last_slack")
CLIP_H = 24.0      # a branch that breaks a rule (FAIL_H) counts as +-24 h


def crc(text: str) -> int:
    """A scenario seed from a string: reproducible, independent of run order."""
    return zlib.crc32(text.encode())


def walk_to_end(fd, pre, stop, cd, sd):
    """features._walk_forward's nominal accounting, run to the destination
    from continuous-driving clock `cd` and daily-driving clock `sd`:
    (daily rests still needed, daily driving left in the final shift)."""
    t_cd, t_sd = float(fd["Tdrv_cons"]), float(fd["Tdrv_sh1"])
    nr = 0
    for i in range(stop, pre.N):
        d = float(pre.D[i])
        if cd + d > t_cd + 1e-9:
            cd = 0.0
        if sd + d > t_sd + 1e-9:
            nr += 1
            cd = sd = 0.0
        cd += d
        sd += d
    return nr, t_sd - sd


def g_matrix(fd, pre, stop, state, rows, legal, idx):
    """G's input for the candidate rows `idx`: the student's own row followed
    by the two route-end features after that action (a break resets the
    continuous-driving clock, a rest both clocks -- as in action_features)."""
    out = np.empty((len(idx), rows.shape[1] + len(EXTRA)), dtype=np.float32)
    for r, j in enumerate(idx):
        brk = _norm(legal[j].get("break_type"))
        rest = _norm(legal[j].get("rest_type")) in ("r1", "r2")
        cd = 0.0 if (rest or brk in ("b45", "b30")) else float(state.cd)
        sd = 0.0 if rest else float(state.sd)
        out[r, :rows.shape[1]] = rows[j]
        out[r, rows.shape[1]:] = walk_to_end(fd, pre, stop, cd, sd)
    return out


def ranked(base, fd, pre, stop, state, cv, k):
    """The student's view of this stop, as decide() takes it:
    (legal, rows, cost, flags, j0, elig) with j0 the student's own choice and
    elig its k best actions that the feasibility head accepts (j0 first)."""
    legal, rows, cost, feas, flags = base._score(fd, pre, stop, state, cv)
    score = cost + 1e6 * (feas < base.feas_thr)
    order = np.argsort(score, kind="stable")
    elig = [int(j) for j in order[:k] if score[j] < 1e5]
    return legal, rows, cost, flags, int(order[0]), elig


def _tauc_batch(base, rows):
    if not len(rows):
        return []
    if getattr(base, "kind", "") == "gbt":
        return [float(x) for x in base.tauc.predict(np.vstack(rows))]
    return [float(base._predict_tauc(r)) for r in rows]


def drive_fleet(base, fd, pre, cv, vehs, scens):
    """Drive every vehicle to the destination with the student, vehicle i
    under scenario scens[i], all of them one stop per tick with one batched
    prediction per tick.  Makes exactly the decisions StudentPolicy.decide
    would.  Returns each vehicle's route cost: arrival + BETA per window miss,
    or FAIL_H if it broke a rule."""
    n_stops = int(fd["N"])
    live = [i for i, v in enumerate(vehs) if not v.is_halted and v.stop < n_stops]
    while live:
        built = [base._rows(fd, pre, vehs[i].stop, vehs[i], cv) for i in live]
        cost, feas = base._predict(np.vstack([b[1] for b in built]))
        picks, o = [], 0
        for i, (legal, rows, flags) in zip(live, built):
            n = len(legal)
            j = int(np.argmin(cost[o:o + n] + 1e6 * (feas[o:o + n] < base.feas_thr)))
            o += n
            picks.append((i, legal[j], rows[j:j + 1], flags))
        charging = [k for k, p in enumerate(picks) if int(p[1].get("y", 0)) == 1]
        raw = dict(zip(charging, _tauc_batch(base, [picks[k][2] for k in charging])))
        for k, (i, act, row, flags) in enumerate(picks):
            v = vehs[i]
            s = v.stop
            tc = base._charge_hours(fd, s, v, act, row, flags, raw=raw.get(k))
            step(v, fd, act, tc, scens[i]["D"][s], scens[i]["E"][s])
        live = [i for i in live if not vehs[i].is_halted and vehs[i].stop < n_stops]
    return np.array([FAIL_H if v.is_halted else v.t_arr + BETA * len(v.tw_misses)
                     for v in vehs])


class FPIPolicy:
    """The student, corrected by G among its own k best actions.  A drop-in
    for policy_core.run_student.  `min_gain` (hours): leave the student's
    choice only when the corrected score is lower by more than this."""

    def __init__(self, base, tag, min_gain=0.0, models_dir=MODELS):
        with open(os.path.join(models_dir, f"{tag}_meta.json")) as fh:
            self.meta = json.load(fh)
        if self.meta["base"] != base.tag:
            raise ValueError(f"{tag} corrects {self.meta['base']}, not {base.tag}")
        self.G = lgb.Booster(model_file=os.path.join(models_dir, f"{tag}_delta.txt"))
        full = list(base.state_names) + list(base.action_names) + list(EXTRA)
        self.cols = [full.index(n) for n in self.meta["features"]]
        self.base = base
        self.tag = tag
        self.k = int(self.meta["top_k"])
        self.min_gain = float(min_gain)
        self.reset()

    def reset(self):
        """Zero the counters (one policy object can drive many routes)."""
        self.n_decisions = 0
        self.n_scored = 0        # decisions with at least two candidates
        self.n_changed = 0       # ... on which G overruled the student

    def decide(self, fd, pre, stop, state, cv):
        self.n_decisions += 1
        b = self.base
        legal, rows, cost, flags, j, elig = ranked(b, fd, pre, stop, state, cv, self.k)
        if len(elig) >= 2:
            self.n_scored += 1
            gm = g_matrix(fd, pre, stop, state, rows, legal, elig)
            adj = cost[elig] + self.G.predict(gm[:, self.cols])
            i = int(np.argmin(adj))
            if i != 0 and adj[i] < adj[0] - self.min_gain:
                j = elig[i]
                self.n_changed += 1
        act = legal[j]
        tc = b._charge_hours(fd, stop, state, act, rows[j:j + 1], flags)
        return act, tc, action_key(act.get("y", 0), act.get("break_type"),
                                   act.get("rest_type"))
