"""
endgame_policy.py — the student, handing over to the MILP for the last shifts of the route
=========================================================================================
Why an end game.  In the base case, every LA run on a long route that ends more
than 8 % above the hindsight oracle has taken one daily rest more than the
oracle, and 22 of the 60 extra-rest runs have less than 2 h of driving left
after their last rest -- the route overflowed its final shift by minutes.  The
students copy it.  On RlongCfewTmedium_23 the MILP started from the student's
own state at the beginning of shift 3 still finds the oracle's rest count;
one shift later it no longer can.  The decision that matters is taken about
three shifts before the end, beyond the LA's 24 h horizon, and it is a packing
decision -- driving, charging and breaks into 13/15 h spreads over several
shifts -- which is what the MILP is for and what one-step rollout
(rollout_policy.py) could not fix.

So: the student drives until the nominal driving left falls below `zone_h`;
from there, at every stop, the journal's horizon MILP (MILP.solve_horizon) is
solved to the DESTINATION from the vehicle's current state and its first
decision is executed, durations and all, exactly as the LA executes its
nominal re-solve.  The model the MILP sees:

  * the NEXT leg's time at the box corner (x XI_MAX), so whatever it plans now
    is legal on arrival at the next stop -- every stop admits a break or a
    rest, so one leg of margin is enough for the driving clocks (the student's
    guard is the 0.99 quantile, ~ the same corner);
  * ENERGY at the fastest speed on every leg up to the next charger, because
    only a charger can make good a shortfall: with nominal energy there, the
    plan charged just enough, a few fast legs later no plan existed at all
    and the truck stranded between chargers (RlongCfewTmedium_23, first
    version).  The student's shield checks the same span (e_needed);
  * every later leg at its nominal time and energy (x `tail`), because it is
    re-planned before it is driven; a blanket margin on 100 legs is far wider
    than their sum varies (~1.5 %) and makes the plan give the rest away.

Where the MILP takes over matters as much.  Taking over early (40 h of
driving left, ~140 stops) the MILP could not find the oracle's rest count
even in 300 s, committed to an evenly spaced plan with one rest more, and
locked it in; the student's drive-as-far-as-you-can keeps that option open.
From ~28 h left (~95 stops) it solves to optimality in 3-30 s and plans the
oracle's count when the times allow it.

A solve that finds no plan hands the stop back to the student.  Solves are
warm-started from the previous plan, shifted by one stop, as the LA does.  A
solve that stops at the time limit with MORE rests in total than the last
accepted plan gets `retry` x the time before its first decision is executed:
on RlongCfewTnone_25 such an incumbent (one rest more, +11 h) was executed as
a premature rest, one stop before the solver was back to the better plan, and
the route ended with the extra rest the student had avoided.
Nothing is trained here and nothing is written outside ML/.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.methods.MILP import solve_horizon                          # noqa: E402
from src.settings import XI_MAX                                     # noqa: E402
from src.simulation.scenarios import generate_scenarios            # noqa: E402

from features import action_key                                     # noqa: E402
from policy_core import StudentPolicy                               # noqa: E402


def _leg(d, leg):
    """Route dicts come back from JSON with string keys; replays use ints."""
    return float(d[leg] if leg in d else d[str(leg)])


class EndgamePolicy:
    """Student first, MILP to the destination once `zone_h` hours of nominal
    driving are left.  A drop-in for policy_core.run_student: in the end game
    decide() also returns the plan, which the loop hands to the vehicle."""

    def __init__(self, base: StudentPolicy, zone_h=28.0, tail=1.0,
                 time_limit=30, retry=4, route=""):
        self.base = base
        self.zone_h = float(zone_h)
        self.tail = float(tail)
        self.time_limit = int(time_limit)
        self.retry = int(retry)
        self.route = route
        self._prev = None            # (stop, plan) of the last solve
        self._prev_total = None      # total rests (done + planned) of that plan
        self.n_decisions = 0
        self.n_solves = 0
        self.n_retries = 0           # time-limited solves given more time
        self.n_fallback = 0          # solves that found no plan -> student
        self.seconds = 0.0
        self.zone_stop = None        # first stop handled by the MILP
        self.log: list = []

    def _override(self, fd, pre, stop, N):
        D = {leg: _leg(fd["D"], leg) * self.tail for leg in range(stop, N)}
        E = {leg: _leg(fd["E"], leg) * self.tail for leg in range(stop, N)}
        D[stop] = _leg(fd["D"], stop) * XI_MAX
        nxt = min(int(pre.next_cs[stop]), N)        # first charger after stop
        fast = generate_scenarios(fd, stop, nxt, n_scenarios=0,
                                  include_best=True)[0]["E"]
        E.update({int(leg): float(v) for leg, v in fast.items()})
        return D, E

    def decide(self, fd, pre, stop, state, cv):
        self.n_decisions += 1
        N = int(fd["N"])
        if float(pre.totD - pre.cumD[stop]) > self.zone_h:
            return self.base.decide(fd, pre, stop, state, cv)
        if self.zone_stop is None:
            self.zone_stop = int(stop)

        D, E = self._override(fd, pre, stop, N)
        warm = None
        if self._prev is not None and self._prev[0] == stop - 1:
            warm = [dict(s, i=s["i"] - 1) for s in self._prev[1][1:]
                    if s["i"] - 1 >= 0]
        done = sum(1 for d in state.durations if float(d.get("taur", 0) or 0) > 0)

        def solve(limit):
            return solve_horizon(
                full_data=fd, start_stop=stop, end_stop=N,
                init_state=state.as_init_state(),
                D_override=D, E_override=E,
                rho2_remaining=int(fd.get("rho_bar", 3)) - int(state.rho2_used),
                ext_remaining=int(fd.get("ext_bar", 2)) - int(state.ext_shift_used),
                time_limit=limit, relax=False, warm_start=warm)

        def total(res):
            return done + int(sum(round(s.get("rho1", 0) + s.get("rho2", 0))
                                  for s in res["sol"]))

        t0 = time.perf_counter()
        r = solve(self.time_limit)
        retried = False
        if (r.get("feasible") and r.get("status") != "optimal"
                and self._prev_total is not None and total(r) > self._prev_total):
            retried = True                    # unproven, and a rest worse
            self.n_retries += 1
            r2 = solve(self.time_limit * self.retry)
            if r2.get("feasible") and (total(r2) < total(r)
                                       or r2.get("status") == "optimal"):
                r = r2
        dt = time.perf_counter() - t0
        self.n_solves += 1
        self.seconds += dt
        if not r.get("feasible"):
            self.n_fallback += 1
            self._prev = None
            self._prev_total = None
            self.log.append(dict(stop=int(stop), status=r.get("status"),
                                 wall=round(dt, 2), fallback=True))
            return self.base.decide(fd, pre, stop, state, cv)

        sol = r["sol"]
        self._prev = (stop, sol)
        self._prev_total = total(r)
        fa = r["first_action"]
        action = dict(y=int(fa["y"]), break_type=fa["break_type"],
                      rest_type=fa["rest_type"])
        self.log.append(dict(
            stop=int(stop), status=r.get("status"), wall=round(dt, 2),
            rests_planned=self._prev_total - done, retried=retried,
            obj=round(float(r["obj"]), 3),
            key=action_key(action["y"], action["break_type"],
                           action["rest_type"])))
        return (action, float(fa["tauc"]),
                action_key(action["y"], action["break_type"],
                           action["rest_type"]), r)
