"""
policy_core.py — everything the GBT and neural students SHARE
=============================================================
The two arms of this project differ in exactly one thing: the function that
maps a (state, action) row to a predicted cost.  Everything else -- which
actions exist, which are mandatory, how the argmin is taken, how the charge
duration is clamped, and how the route is driven -- lives here and is used by
both, so a comparison between them measures the regressor and nothing else.

  ML/code/policy_core.py   <- this file: the decision rule + simulator loop
  ML/code/gbt_policy.py    <- LightGBM boosters supply the predictions
  ML/code/nn_policy.py     <- an sklearn MLP supplies the predictions

Subclasses implement two methods:

    _predict(rows)      -> (cost[n], feas_prob[n])
    _predict_tauc(row)  -> charge hours for the chosen action

Three things are deliberately NOT learned by either arm
-------------------------------------------------------
* LEGALITY.  enumerate_actions decides what exists at this stop (no charging
  where there is no charger, no b30 without a split in progress, no r2 once
  the budget is spent).  Reimplementing that would eventually diverge from the
  simulator and produce a policy that looks good and is illegal.
* FORCING.  compute_flags / action_passes decide what is mandatory.  These are
  the same calls greedy and the look-ahead's own pruner make.
* THE CHARGE CLAMP.  The predicted charge duration is clipped into
  [enough to reach the next charging opportunity, charge to full].  Both ends
  are physics, computable in microseconds.  The model chooses within a safe
  interval; it can be wrong, but not dangerous.

This module owns its simulator loop rather than registering a method in
src/simulation/runner_dispatch.py, for two reasons: everything this project
writes stays under ML/, and the reporting pipeline discovers runs by globbing
solutions/<bucket>/ by method name -- a stray student run in the main tree
would silently enter the manuscript's tables.  The loop mirrors
greedy.run_greedy, which is the same shape: decide, build durations, advance,
check for a halt.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.simulation.BEHDV import (BEHDV, _charging_time_needed,     # noqa: E402
                                  _energy_after_charging)
from src.simulation.Simulation import enumerate_actions             # noqa: E402
from src.simulation.supervisor import action_passes                 # noqa: E402

from features import (Precomp, action_features, action_key,         # noqa: E402
                      state_features)

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def charge_needed_to_reach(fd, e_arr, flags):
    """Charge hours so the worst-case run to the next CS keeps SoC >= Emin."""
    deficit = float(flags["e_needed"]) - (e_arr - float(fd["Emin"]))
    if deficit <= 0:
        return 0.0
    lo, hi = 0.0, _charging_time_needed(e_arr, fd)
    target = e_arr + deficit
    for _ in range(40):                       # bisect the PWL charging curve
        mid = 0.5 * (lo + hi)
        if _energy_after_charging(e_arr, mid, fd) < target:
            lo = mid
        else:
            hi = mid
    return hi


class StudentPolicy:
    """The decision rule.  Subclasses only supply the predictions."""

    def __init__(self, feas_thr=0.5, guard_q=None):
        self.feas_thr = feas_thr
        self.guard_q = guard_q
        self.state_names: list = []
        self.action_names: list = []
        self.tag = "?"
        self.kind = "?"
        self.n_forced = 0      # decisions where forcing left exactly one action
        self.n_empty = 0       # decisions where forcing left nothing
        self.n_clamped = 0     # charge durations the physics clamp moved

    # -- to be implemented per arm -------------------------------------------
    def _predict(self, rows):
        raise NotImplementedError

    def _predict_tauc(self, row):
        raise NotImplementedError

    # -- identical for every arm ---------------------------------------------
    def decide(self, fd, pre, stop, state, cv):
        """Return (action dict, tauc hours, action key)."""
        acts = enumerate_actions(stop, state, fd, charge_only=False)
        sf, flags = state_features(fd, pre, stop, state, cv, self.guard_q)

        legal = [a for a in acts if action_passes(fd, stop, state, a, flags)]
        if not legal:
            self.n_empty += 1
            legal = acts                  # keep moving; BEHDV records the breach
        elif len(legal) == 1:
            self.n_forced += 1

        sv = [sf[n] for n in self.state_names]
        rows = np.array(
            [sv + [action_features(fd, pre, stop, state, a, sf)[n]
                   for n in self.action_names] for a in legal],
            dtype=np.float32)

        # An arm that scores the STATE once and reads off per-action values
        # (the classifier) needs to know which action each row is; a scoring
        # arm ignores this.  Stashing it here keeps `decide` identical for
        # every arm, which is the point of this class.
        self._legal_keys = [action_key(a.get("y", 0), a.get("break_type"),
                                       a.get("rest_type")) for a in legal]
        cost, feas = self._predict(rows)
        j = int(np.argmin(cost + 1e6 * (feas < self.feas_thr)))
        act = legal[j]

        tc = 0.0
        if int(act.get("y", 0)) == 1:
            raw = float(self._predict_tauc(rows[j:j + 1]))
            lo = charge_needed_to_reach(fd, state.e_arr, flags)
            hi = _charging_time_needed(state.e_arr, fd)
            tc = min(max(raw, lo), max(hi, lo))
            if abs(tc - raw) > 1e-6:
                self.n_clamped += 1
        return act, tc, action_key(act.get("y", 0), act.get("break_type"),
                                   act.get("rest_type"))


def durations(fd, stop, action, tauc):
    """Execution durations, same parallel-charging model as greedy/the MILP."""
    is_CS = stop in set(fd["K"])
    y = int(action.get("y", 0))
    brk = action.get("break_type")
    rst = action.get("rest_type")
    brk = None if str(brk).lower() in ("0", "none", "-", "") else str(brk).lower()
    rst = None if str(rst).lower() in ("0", "none", "-", "") else str(rst).lower()
    bmin = {"b45": fd["Tb45"], "b15": fd["Tb15"], "b30": fd["Tb30"]}.get(brk, 0.0)
    rmin = fd["Tr1"] if rst == "r1" else fd["Tr2"] if rst == "r2" else 0.0
    tauq = float(fd["Q"].get(stop, 0.0)) * y if is_CS else 0.0
    if is_CS and y:
        taub = max(0.0, bmin - tauc)      # break absorbed by the charge
    else:
        tauc = 0.0
        taub = bmin
    return dict(taub=taub, tauc=tauc, taur=rmin, tauq=tauq,
                b45=int(brk == "b45"), b15=int(brk == "b15"),
                b30=int(brk == "b30"), rho1=int(rst == "r1"),
                rho2=int(rst == "r2"), y=y)


def run_student(fd, D_real, E_real, policy: StudentPolicy, cv=0.15):
    """Drive one route with a learned policy.  Mirrors greedy.run_greedy."""
    pre = Precomp(fd)
    veh = BEHDV(fd)
    N = int(fd["N"])
    T0 = float(fd.get("T_START", 8.0))
    dec_times, acts_taken = [], []

    while veh.stop < N:
        stop = veh.stop
        t0 = time.perf_counter()
        action, tauc, key = policy.decide(fd, pre, stop, veh, cv)
        dec_times.append(time.perf_counter() - t0)
        acts_taken.append(key)

        mock = dict(feasible=True,
                    sol=[dict(i=0, **durations(fd, stop, action, tauc))])
        veh.advance(action=action, D_next=float(D_real[stop]),
                    E_next=float(E_real[stop]), milp_sol=mock)
        if veh.is_halted:
            break

    completed = (not veh.is_halted) and veh.stop >= N
    return dict(
        duration_h=(veh.t_arr - T0) if completed else None,
        partial_duration_h=veh.t_arr - T0,
        route_completed=bool(completed),
        halted_at=veh.halted_at,
        halt_reason=veh.halt_reason,
        n_violations=len(veh.violations),
        violations_by_type={t: sum(1 for v in veh.violations if v["type"] == t)
                            for t in {v["type"] for v in veh.violations}},
        tw_misses=len(veh.tw_misses),
        n_customers=len(fd["C"]),
        n_stops=N,
        decisions=len(dec_times),
        ms_per_decision=1000.0 * float(np.mean(dec_times)) if dec_times else 0.0,
        actions=acts_taken,
    )


def load_policy(kind: str, tag: str, feas_thr=0.5, guard_q=None):
    """Factory: 'gbt' (trees), 'nn' (MLP regression), 'clf' (MLP classifier)."""
    if kind == "gbt":
        from gbt_policy import GBTPolicy
        return GBTPolicy(tag=tag, feas_thr=feas_thr, guard_q=guard_q)
    if kind == "nn":
        from nn_policy import NNPolicy
        return NNPolicy(tag=tag, feas_thr=feas_thr, guard_q=guard_q)
    if kind == "clf":
        from clf_policy import ClfPolicy
        return ClfPolicy(tag=tag, feas_thr=feas_thr, guard_q=guard_q)
    raise ValueError(f"unknown policy kind: {kind!r}")
