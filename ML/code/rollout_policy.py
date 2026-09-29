"""
rollout_policy.py — the student, improved by rolling each option out to the end of the route
============================================================================================
The look-ahead (LA) scores an action by solving a 24 h horizon MILP under 25
sampled scenarios, each of which knows its own future.  Two blind spots
follow.  It is OPTIMISTIC: inside a scenario the MILP plans with perfect
information, so it undervalues slack.  And it is TRUNCATED: 24 h is one duty
cycle, so it cannot see a daily rest that only becomes unavoidable two shifts
later.  On the base case, every LA run on a long route that ends more than 8 %
above the hindsight oracle has taken one daily rest more than the oracle, and
the students, trained on the LA, copy those errors.

A student decides in ~2 ms, so the WHOLE remaining route can be simulated many
times per decision.  At a decision stop this policy

  1. takes the base student's k best actions (StudentPolicy.candidates);
  2. draws K travel-time scenarios for the remaining legs with the LA's own
     generator and settings (scenarios.generate_scenarios) -- never the
     realised times, which policy_core.run_student keeps to itself;
  3. for every (action, scenario): clones the vehicle, applies the action, and
     lets the base student drive to the destination under that scenario;
  4. overrules the student's own choice only when an alternative is better by
     a margin the scenarios can vouch for: paired mean difference + z * its
     standard error < -min_gain (route cost = arrival + BETA per window miss,
     a rollout that breaks a rule counting FAIL_H).  A challenger that looks
     better but not significantly so gets more scenarios -- the batch doubles
     up to k_max -- before the student is kept.

Every action faces the same scenarios (common random numbers), so the
comparison is paired.  This is Bertsekas' rollout: with EXACT evaluation it is
never worse than the base policy it simulates.  With sampled evaluation it can
be much worse, and was (smoke test, 2026-09-29): the student driving on takes
an extra daily rest in some scenarios, a ~10 h jump, so with 8 scenarios the
plain argmin (z = 0) chased that noise -- e.g. an early 45-min break "saving"
1.8 h that 64 scenarios put at +0.56 h -- and every route ended worse than the
student.  Hence the gate.  Nothing here is trained -- it wraps any
StudentPolicy -- and nothing is written outside ML/ (run_rollout.py saves).

Scenarios are seeded by (route, stop, seed), so a run is reproducible and does
not depend on the order in which routes are processed.
"""
from __future__ import annotations

import copy
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.simulation.scenarios import generate_scenarios            # noqa: E402

from policy_core import StudentPolicy, durations                    # noqa: E402

BETA = 0.5        # h per window miss: the manuscript's objective (settings.BETA_TW)
FAIL_H = 1000.0   # cost of a rollout that breaks a rule; dominates any route


def clone(veh):
    """An independent copy of a BEHDV for a what-if drive.

    advance() only ever appends to the history lists and adds keys to
    tw_misses, and halt() only sets scalars, so a copy of every list and dict
    is a full copy of the state.  The instance dict (_fd) is shared, never
    copied: ~0.03 ms instead of a deepcopy's ~2 ms."""
    v = copy.copy(veh)
    for name, val in vars(veh).items():
        if name != "_fd" and isinstance(val, (list, dict)):
            setattr(v, name, copy.copy(val))
    return v


def step(veh, fd, action, tauc, D, E):
    """Execute one decision, exactly as policy_core.run_student does."""
    mock = dict(feasible=True,
                sol=[dict(i=0, **durations(fd, veh.stop, action, tauc))])
    veh.advance(action=action, D_next=float(D), E_next=float(E), milp_sol=mock)


def finish(veh, fd, pre, base, cv, scen):
    """Drive `veh` to the destination with `base`, the legs taking the times of
    scenario `scen`.  Returns arrival + BETA * window misses, or FAIL_H."""
    N = int(fd["N"])
    while not veh.is_halted and veh.stop < N:
        s = veh.stop
        a, tc, _ = base.decide(fd, pre, s, veh, cv)
        step(veh, fd, a, tc, scen["D"][s], scen["E"][s])
    if veh.is_halted:
        return FAIL_H
    return veh.t_arr + BETA * len(veh.tw_misses)


class RolloutPolicy:
    """Rollout on top of a StudentPolicy; a drop-in for policy_core.run_student.

    Which decisions are rolled out:
      margin_h = None -> every decision with more than one candidate
      margin_h = x    -> only those where the student's two best scores (its
                         own predicted regrets) are within x hours
    When to overrule the student (see the module docstring):
      z, min_gain_h   -> paired mean + z * se < -min_gain_h;  z = 0 and
                         min_gain_h = 0 is the plain argmin of the means
      k_max           -> a promising but unproven challenger doubles the
                         scenarios until k_max (default: no doubling)
    """

    def __init__(self, base: StudentPolicy, n_scen=16, top_k=3, margin_h=None,
                 z=2.0, min_gain_h=0.0, k_max=None, seed=0, route=""):
        self.base = base
        self.n_scen = int(n_scen)
        self.top_k = int(top_k)
        self.margin_h = margin_h
        self.z = float(z)
        self.min_gain_h = float(min_gain_h)
        self.k_max = int(k_max) if k_max else self.n_scen
        self.seed = int(seed)
        self.route = route
        self.n_decisions = 0
        self.n_rolled = 0        # decisions that were rolled out
        self.n_changed = 0       # ... on which the rollout overruled the student
        self.n_rollouts = 0
        self.n_failed = 0        # rollouts that broke a rule
        self.seconds = 0.0       # time spent rolling out
        self.log: list = []      # one entry per rolled-out decision

    def _seed(self, stop, batch=0):
        # batch 0 keeps the seed of the first version, so a z = 0 run
        # reproduces the plain-rollout smoke test exactly
        tail = f"|{batch}" if batch else ""
        return zlib.crc32(f"{self.route}|{stop}|{self.seed}{tail}".encode())

    def _costs(self, fd, pre, stop, state, cv, cands, scen):
        cost = np.empty((len(cands), len(scen)))
        for i, (act, tc, _key, _score) in enumerate(cands):
            for j, s in enumerate(scen):
                v = clone(state)
                step(v, fd, act, tc, s["D"][stop], s["E"][stop])
                cost[i, j] = (FAIL_H if v.is_halted
                              else finish(v, fd, pre, self.base, cv, s))
        return cost

    def decide(self, fd, pre, stop, state, cv):
        """Return (action dict, tauc hours, action key), like StudentPolicy."""
        self.n_decisions += 1
        cands = self.base.candidates(fd, pre, stop, state, cv, k=self.top_k)
        if len(cands) < 2 or (self.margin_h is not None
                              and cands[1][3] - cands[0][3] > self.margin_h):
            return cands[0][:3]

        t0 = time.perf_counter()
        N = int(fd["N"])
        cost = self._costs(fd, pre, stop, state, cv, cands,
                           generate_scenarios(fd, stop, N,
                                              n_scenarios=self.n_scen, cv=cv,
                                              seed=self._seed(stop)))
        batch = 0
        while True:
            d = cost[1:] - cost[0]                  # paired, vs the student
            k = d.shape[1]
            mean = d.mean(axis=1)
            se = (d.std(axis=1, ddof=1) / np.sqrt(k)) if k > 1 else np.full(len(d), np.inf)
            i = int(np.argmin(mean))
            if mean[i] + self.z * se[i] < -self.min_gain_h:
                best = i + 1
                break
            if mean[i] >= -self.min_gain_h or k >= self.k_max:
                best = 0
                break
            batch += 1                              # promising, not proven yet
            more = generate_scenarios(fd, stop, N, n_scenarios=min(k, self.k_max - k),
                                      cv=cv, seed=self._seed(stop, batch))
            cost = np.hstack([cost, self._costs(fd, pre, stop, state, cv, cands, more)])

        self.n_rolled += 1
        self.n_changed += int(best != 0)
        self.n_rollouts += cost.size
        self.n_failed += int((cost >= FAIL_H).sum())
        self.seconds += time.perf_counter() - t0
        self.log.append(dict(
            stop=int(stop), t=round(float(state.t_arr), 3),
            keys=[c[2] for c in cands],
            score=[round(c[3], 3) for c in cands],
            mean=[round(float(m), 3) for m in cost.mean(axis=1)],
            diff=[round(float(m), 3) for m in mean],
            se=[round(float(s), 3) for s in se],
            k=int(cost.shape[1]),
            fails=[int(n) for n in (cost >= FAIL_H).sum(axis=1)],
            chosen=best))
        return cands[best][:3]
