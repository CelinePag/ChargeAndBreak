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
from src.simulation.supervisor import (_action_min_dwell,           # noqa: E402
                                       action_passes)

from features import (Precomp, action_features, action_key,         # noqa: E402
                      charger_curve, state_features)

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


def _norm(v):
    return None if str(v).lower() in ("0", "none", "-", "") else str(v).lower()


def stop_overhead(fd, stop, action):
    """Fixed time an action costs at this stop, as BEHDV's departure time
    charges it: M_stop at a charger whenever anything happens there, M_lay at
    a layby for a break or rest."""
    active = (_norm(action.get("break_type")) in ("b45", "b15", "b30")
              or _norm(action.get("rest_type")) in ("r1", "r2"))
    if stop in set(fd["K"]):
        if int(action.get("y", 0)) or active:
            return float(fd.get("M_stop", {}).get(stop, 0.0))
        return 0.0
    if stop in set(fd.get("L", [])) and active:
        return float(fd.get("M_lay", {}).get(stop, 0.0))
    return 0.0


def spread_room(fd, stop, action, flags):
    """Hours of charging the 15 h spread still allows (None after a rest).

    The shared legality check (supervisor._spread_with_dwell_fails) admits a
    non-rest action when h + o(a) + D_wc <= 15 h, with o(a) = service + queue
    + minimum break.  It leaves out two things the simulator then spends: the
    charge, which the MILP models itself and so the check never needed, and
    the stop overhead.  A learned policy picks its charge AFTER that check, so
    for it the room has to be computed here.  A rest resets the spread, so it
    has no such limit.
    """
    if _norm(action.get("rest_type")) in ("r1", "r2"):
        return None
    return (float(flags["Tspr2"]) - float(flags["h_state"])
            - _action_min_dwell(fd, stop, action)
            - stop_overhead(fd, stop, action) - float(flags["D_next_wc"]))


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
        # Opt-in (evaluate.py --spread-room): complete the spread check with
        # the charge and stop overhead it leaves out -- see spread_room().
        # Off by default, so every result produced without it is unchanged.
        self.spread_room = False
        self.n_spread_dropped = 0   # actions removed: even the minimum charge
        self.n_spread_cut = 0       # charges shortened to fit the spread

    # -- to be implemented per arm -------------------------------------------
    def _predict(self, rows):
        raise NotImplementedError

    def _predict_tauc(self, row):
        raise NotImplementedError

    # -- identical for every arm ---------------------------------------------
    def _score(self, fd, pre, stop, state, cv):
        """The legal actions at this stop and the model's view of each:
        (legal, rows, cost, feas, flags).  decide() and candidates() both go
        through here, so they see the same legality, forcing and spread-room
        filtering, and bump the counters once per call."""
        legal, rows, flags = self._rows(fd, pre, stop, state, cv)
        cost, feas = self._predict(rows)
        return legal, rows, cost, feas, flags

    def _rows(self, fd, pre, stop, state, cv):
        """_score without the prediction: (legal, rows, flags).  Split out so a
        batch of vehicles can share one _predict call (fpi_policy.drive_fleet);
        rows are predicted independently, so batching changes no value."""
        acts = enumerate_actions(stop, state, fd, charge_only=False)
        sf, flags = state_features(fd, pre, stop, state, cv, self.guard_q)

        legal = [a for a in acts if action_passes(fd, stop, state, a, flags)]
        if self.spread_room and legal:
            # drop non-rest actions whose dwell cannot fit the spread even with
            # the least charge the next leg needs (none, for y=0); if that
            # would leave nothing, keep the unfiltered set
            need = charge_needed_to_reach(fd, state.e_arr, flags)
            fits = [a for a in legal
                    if (r := spread_room(fd, stop, a, flags)) is None
                    or r >= (need if int(a.get("y", 0)) else 0.0) - 1e-9]
            self.n_spread_dropped += len(legal) - len(fits)
            if fits:
                legal = fits
        if not legal:
            self.n_empty += 1
            legal = acts                  # keep moving; BEHDV records the breach
        elif len(legal) == 1:
            self.n_forced += 1

        sv = [sf[n] for n in self.state_names]
        # one action_features dict per action (it used to be rebuilt once per
        # column: same values, ~18x the work)
        rows = np.array(
            [sv + [af[n] for n in self.action_names]
             for af in (action_features(fd, pre, stop, state, a, sf)
                        for a in legal)],
            dtype=np.float32)

        # An arm that scores the STATE once and reads off per-action values
        # (the classifier) needs to know which action each row is; a scoring
        # arm ignores this.  Stashing it here keeps `decide` identical for
        # every arm, which is the point of this class.
        self._legal_keys = [action_key(a.get("y", 0), a.get("break_type"),
                                       a.get("rest_type")) for a in legal]
        return legal, rows, flags

    def _charge_hours(self, fd, stop, state, act, row, flags, raw=None):
        """Charge duration for `act` (0 if it does not charge): the tauc head,
        clamped into [reach the next charger, charge to full] and, with the
        spread-room check on, into what the 15 h spread still allows.  `raw`
        is the tauc head's output when the caller already has it (a batch)."""
        tc = 0.0
        if int(act.get("y", 0)) == 1:
            raw = float(self._predict_tauc(row) if raw is None else raw)
            lo = charge_needed_to_reach(fd, state.e_arr, flags)
            hi = _charging_time_needed(state.e_arr, fd)
            if self.spread_room:
                room = spread_room(fd, stop, act, flags)
                if room is not None and room < hi:
                    hi = room               # never below lo: max(hi, lo) below
                    if raw > room:
                        self.n_spread_cut += 1
            tc = min(max(raw, lo), max(hi, lo))
            if abs(tc - raw) > 1e-6:
                self.n_clamped += 1
        return tc

    def decide(self, fd, pre, stop, state, cv):
        """Return (action dict, tauc hours, action key)."""
        legal, rows, cost, feas, flags = self._score(fd, pre, stop, state, cv)
        j = int(np.argmin(cost + 1e6 * (feas < self.feas_thr)))
        act = legal[j]
        tc = self._charge_hours(fd, stop, state, act, rows[j:j + 1], flags)
        return act, tc, action_key(act.get("y", 0), act.get("break_type"),
                                   act.get("rest_type"))

    def candidates(self, fd, pre, stop, state, cv, k=3):
        """The k best actions by the model's own score, best first, as
        [(action, tauc, key, score), ...].  candidates()[0] is exactly what
        decide() returns (a stable sort keeps argmin's tie-break); the rollout
        policy (rollout_policy.py) weighs the others against it."""
        legal, rows, cost, feas, flags = self._score(fd, pre, stop, state, cv)
        score = cost + 1e6 * (feas < self.feas_thr)
        out = []
        for j in np.argsort(score, kind="stable")[:k]:
            j = int(j)
            act = legal[j]
            tc = self._charge_hours(fd, stop, state, act, rows[j:j + 1], flags)
            out.append((act, tc, action_key(act.get("y", 0),
                                            act.get("break_type"),
                                            act.get("rest_type")),
                        float(score[j])))
        return out


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

    curves = fd.get("TbarK")          # a route whose chargers differ
    tbar_route = fd["Tbar"]
    while veh.stop < N:
        stop = veh.stop
        if curves:
            # the simulator and the policy's charge clamp read fd["Tbar"]:
            # make it the curve of the charger here (or of the next one,
            # which is what a layby's charge features describe)
            fd["Tbar"] = charger_curve(fd, stop if stop in pre.K
                                       else int(pre.next_cs[stop]))
        t0 = time.perf_counter()
        out = policy.decide(fd, pre, stop, veh, cv)
        dec_times.append(time.perf_counter() - t0)
        action, tauc, key = out[:3]
        acts_taken.append(key)

        # A policy that plans with the MILP (endgame_policy.py) hands back the
        # plan itself as a 4th element, and the vehicle executes it the way
        # the LA's nominal re-solve is executed -- durations, sequencing and
        # all.  Every other policy returns three and gets the mock below.
        plan = out[3] if len(out) > 3 else None
        mock = plan or dict(feasible=True,
                            sol=[dict(i=0, **durations(fd, stop, action, tauc))])
        veh.advance(action=action, D_next=float(D_real[stop]),
                    E_next=float(E_real[stop]), milp_sol=mock)
        if veh.is_halted:
            break

    fd["Tbar"] = tbar_route
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


def load_policy(kind: str, tag: str, feas_thr=0.5, guard_q=None,
                spread_room=False):
    """Factory: 'gbt' (trees), 'nn' (MLP regression), 'clf' (MLP classifier),
    'torch' (PyTorch networks, torch_train.py)."""
    if kind == "gbt":
        from gbt_policy import GBTPolicy
        pol = GBTPolicy(tag=tag, feas_thr=feas_thr, guard_q=guard_q)
    elif kind == "torch":
        from torch_policy import TorchPolicy
        pol = TorchPolicy(tag=tag, feas_thr=feas_thr, guard_q=guard_q)
    elif kind == "nn":
        from nn_policy import NNPolicy
        pol = NNPolicy(tag=tag, feas_thr=feas_thr, guard_q=guard_q)
    elif kind == "clf":
        from clf_policy import ClfPolicy
        pol = ClfPolicy(tag=tag, feas_thr=feas_thr, guard_q=guard_q)
    else:
        raise ValueError(f"unknown policy kind: {kind!r}")
    pol.spread_room = bool(spread_room)
    return pol
