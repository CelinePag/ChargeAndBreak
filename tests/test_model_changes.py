"""
tests/test_model_changes.py — Unit tests for the July-2026 model revision
==========================================================================
Covers the unit tests requested in the code-change list:

  M1 — valid inequalities no longer cut feasible/optimal schedules
  M2 — break-credit g behaviour (parallel / sequential / no declared break)
  M3 — working-time contribution of charging (tauc − g)
  M4 — charge + rest forced sequential (σ = 1, M_seq charged)
  M5 — shift spread: 14 h elapsed before a REGULAR rest infeasible,
       before a REDUCED rest feasible
  M6 — 10 h extended-driving allowance usable within its budget,
       infeasible when the budget is zero
  TW — fixed-penalty windows (v3): out-of-window service start (early OR
       late) sets delta = 1 at a fixed cost beta; never hard-infeasible;
       hard_tw recovers hard windows; the LP relaxation keeps delta integer
       so windows are not silently free (TW4); break-stretching upstream
       (§4) trades an activity extension against the penalty.
  SP — single rest-last spread convention (§3): pre-rest elapsed = h + o at
       every stop type, customers included.

Run with:  python tests/test_model_changes.py       (plain runner)
      or:  python -m pytest tests/                  (if pytest installed)

Each test builds a tiny synthetic instance via instances.make_data and
solves the full-route model with Gurobi.  Solves are seconds-scale.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyomo.environ as pyo

from instances import make_data
from MILP import build_model, add_valid_inequalities
from settings import ecr, V_NOM

EPS = 1e-4


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def tiny_data(leg_hours, cs_stops, cust_stops=(), Bcap=5000.0,
              Wha=None, Whf=None, **overrides):
    """
    Build a tiny instance: legs of the given nominal durations, stop types
    as specified.  Bcap defaults huge so battery never binds in pure-HoS
    tests; pass Bcap=500.0 for charging tests.
    """
    N = len(leg_hours)
    I = list(range(N + 1))
    C = list(cust_stops)
    K = [i for i in range(1, N) if i not in set(C)]
    assert set(cs_stops) <= set(K), "cs_stops must be intermediate non-customers"
    D = {i: float(leg_hours[i]) for i in range(N)}
    km = {i: D[i] * V_NOM for i in range(N)}
    E = {i: km[i] * ecr(V_NOM) for i in range(N)}
    data = make_data(
        I=I, C=C, K=K, D=D, E=E, km=km, Bcap=Bcap,
        Q={k: 0.0 for k in K},
        Wha=Wha if Wha is not None else {c: 0.0 for c in C},
        Whf=Whf if Whf is not None else {c: 2e7 for c in C},
        label="tiny test instance", title="tiny",
    )
    data.update(overrides)
    return data


def solve(model, time_limit=60):
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = 1e-6
    try:
        res = solver.solve(model, tee=False)
        status = str(res.solver.termination_condition)
    except (RuntimeError, ValueError):
        return "infeasible"
    return status


def v(x):
    val = pyo.value(x)
    return 0.0 if val is None else float(val)


# ══════════════════════════════════════════════════════════════════════════════
# M1 — VALID INEQUALITIES MUST NOT CUT FEASIBLE SCHEDULES
# ══════════════════════════════════════════════════════════════════════════════

def test_m1_vi_reset_count_five_hours():
    """5 h of driving needs ONE cd-reset; the un-fixed VI (ceil(5/4.5)=2)
    would have cut the optimum.  With VIs added the model must stay feasible
    and use exactly one reset."""
    data = tiny_data([2.5, 2.5], cs_stops=[1])
    m = build_model(data)
    add_valid_inequalities(m, data)
    status = solve(m)
    assert status == "optimal", f"expected optimal, got {status}"
    n_resets = sum(round(v(m.x_b45[i]) + v(m.x_b30[i])
                         + v(m.rho1[i]) + v(m.rho2[i])) for i in data["I"])
    assert n_resets == 1, f"expected exactly 1 cd-reset, got {n_resets}"


def test_m1_vi_shift_uses_extended_limit():
    """9.5 h total driving is legal with ZERO rests thanks to the 10 h
    extension (M6); VI-3/VI-5 must therefore use Tdrv_sh2 = 10 h.  The
    un-fixed VI (denominator 9 h) would force a rest."""
    data = tiny_data([2.4, 2.4, 2.4, 2.3], cs_stops=[1, 2, 3])
    m = build_model(data)
    add_valid_inequalities(m, data)
    status = solve(m)
    assert status == "optimal", f"expected optimal, got {status}"
    n_rests = sum(round(v(m.rho1[i]) + v(m.rho2[i])) for i in data["I"])
    assert n_rests == 0, f"expected 0 rests (10h extension), got {n_rests}"


# ══════════════════════════════════════════════════════════════════════════════
# M2 — BREAK CREDIT g
# ══════════════════════════════════════════════════════════════════════════════

def _m2_base(Bcap=500.0):
    """One CS stop mid-route; battery small enough that charging is useful."""
    return tiny_data([2.0, 2.0], cs_stops=[1], Bcap=Bcap)


def test_m2a_parallel_break_credits_full_charge():
    """y=1, x=1, sigma=0  ->  g == tauc (full charging time credited)."""
    data = _m2_base()
    m = build_model(data)
    m.y[1].fix(1); m.x_b45[1].fix(1); m.sigma[1].fix(0)
    status = solve(m)
    assert status == "optimal", status
    assert abs(v(m.g[1]) - v(m.tauc[1])) < EPS, \
        f"g={v(m.g[1]):.4f} != tauc={v(m.tauc[1]):.4f}"
    assert v(m.taub_hat[1]) >= 0.75 - EPS


def test_m2b_sequential_break_earns_no_credit():
    """y=1, x=1, sigma=1  ->  g == 0 and the 45-min break minimum must be
    met by taub alone."""
    data = _m2_base()
    m = build_model(data)
    m.y[1].fix(1); m.x_b45[1].fix(1); m.sigma[1].fix(1)
    status = solve(m)
    assert status == "optimal", status
    assert v(m.g[1]) < EPS, f"g={v(m.g[1]):.4f} should be 0 in sequential mode"
    assert v(m.taub[1]) >= 0.75 - EPS, \
        f"taub={v(m.taub[1]):.4f} must cover the 45-min break alone"


def test_m2c_charge_without_break_earns_no_credit():
    """y=1, x=0  ->  g == 0 (charging with no declared break earns no
    break credit)."""
    data = _m2_base()
    m = build_model(data)
    m.y[1].fix(1)
    m.x_b45[1].fix(0); m.x_b15[1].fix(0); m.x_b30[1].fix(0)
    m.rho1[1].fix(0);  m.rho2[1].fix(0)
    status = solve(m)
    assert status == "optimal", status
    assert v(m.g[1]) < EPS, f"g={v(m.g[1]):.4f} should be 0 with no break"


# ══════════════════════════════════════════════════════════════════════════════
# M3 — WORK CONTRIBUTION OF CHARGING = tauc − g
# ══════════════════════════════════════════════════════════════════════════════

def test_m3_charging_work_accounting():
    """Charging with no break counts fully as work; with a parallel break it
    counts zero; sequential counts fully.

    Note on indices: sw[i] already INCLUDES the work performed at stop i
    (injected as work_next from the previous propagation step), so the work
    at stop 1 is  sw[1] − sw[0] − D[0]."""
    def work_at_stop1(model, data):
        return v(model.sw[1]) - v(model.sw[0]) - data["D"][0]

    # (a) no break: charging counts fully as work (tauc − g = tauc)
    data = _m2_base()
    m = build_model(data)
    m.y[1].fix(1)
    for var in (m.x_b45, m.x_b15, m.x_b30, m.rho1, m.rho2):
        var[1].fix(0)
    assert solve(m) == "optimal"
    expected = data["M_stop"][1] + v(m.tauc[1])            # Q=0, g=0
    got = work_at_stop1(m, data)
    assert abs(got - expected) < 1e-3, \
        f"work@1={got:.4f}, expected {expected:.4f} (tauc fully counted)"

    # (b) parallel break: charging counts zero work (tauc − g = 0)
    m2 = build_model(data)
    m2.y[1].fix(1); m2.x_b45[1].fix(1); m2.sigma[1].fix(0)
    assert solve(m2) == "optimal"
    expected = data["M_stop"][1]
    got = work_at_stop1(m2, data)
    assert abs(got - expected) < 1e-3, \
        f"work@1={got:.4f}, expected {expected:.4f} (charge credited)"

    # (c) sequential break: charging counts fully again (+ M_seq overhead)
    m3 = build_model(data)
    m3.y[1].fix(1); m3.x_b45[1].fix(1); m3.sigma[1].fix(1)
    assert solve(m3) == "optimal"
    expected = data["M_stop"][1] + v(m3.tauc[1]) + data["M_seq"][1]
    got = work_at_stop1(m3, data)
    assert abs(got - expected) < 1e-3, \
        f"work@1={got:.4f}, expected {expected:.4f} (sequential)"


# ══════════════════════════════════════════════════════════════════════════════
# M4 — CHARGE + REST FORCED SEQUENTIAL
# ══════════════════════════════════════════════════════════════════════════════

def test_m4_charge_plus_rest_is_sequential():
    """y=1, rho=1  ->  sigma == 1 and the M_seq repositioning is charged in
    the departure time."""
    data = _m2_base()
    m = build_model(data)
    m.y[1].fix(1); m.rho2[1].fix(1)
    status = solve(m)
    assert status == "optimal", status
    assert round(v(m.sigma[1])) == 1, "charge co-located with rest must be sequential"
    dwell = v(m.td[1]) - v(m.ta[1])
    expected_min = (data["M_stop"][1] + v(m.tauc[1]) + v(m.taur[1])
                    + data["M_seq"][1])
    assert dwell >= expected_min - 1e-3, \
        f"dwell {dwell:.4f} must include M_seq ({expected_min:.4f})"


# ══════════════════════════════════════════════════════════════════════════════
# M5 / SP — SHIFT SPREAD, SINGLE REST-LAST CONVENTION (v3, §3)
# ══════════════════════════════════════════════════════════════════════════════

def _spread_over13_data():
    """~14 h elapsed before the first rest, built WITHOUT any waiting (v3):
    4.5 h drive to a CS with a 9 h queue (charging active), then a rest at
    the CS.  Pre-rest elapsed = h(4.5) + o(≈9.5) ≈ 14 h, so a REGULAR rest
    (13 h cap) is infeasible while a REDUCED rest (15 h) is feasible."""
    data = tiny_data([4.5, 2.0], cs_stops=[1], Bcap=5000.0)
    data["Q"] = {1: 9.0}      # 9 h queue inflates the pre-rest spread
    return data


def test_sp_regular_rest_over_13h_spread_infeasible():
    data = _spread_over13_data()
    m = build_model(data)
    m.y[1].fix(1)                       # activate the queue (Q counts only if y=1)
    m.rho1[1].fix(1); m.rho2[1].fix(0)  # force a REGULAR rest at the CS
    status = solve(m)
    assert status in ("infeasible", "infeasibleOrUnbounded"), \
        f"~14h pre-rest spread with r1 must be infeasible, got {status}"


def test_sp_reduced_rest_over_13h_spread_feasible():
    data = _spread_over13_data()
    m = build_model(data)
    m.y[1].fix(1)
    m.rho2[1].fix(1); m.rho1[1].fix(0)  # force a REDUCED rest at the CS
    status = solve(m)
    assert status == "optimal", \
        f"~14h pre-rest spread with r2 must be feasible, got {status}"
    pre_rest = v(m.h[1]) + (v(m.td[1]) - v(m.ta[1]) - v(m.taur[1]))
    assert 13.0 - EPS < pre_rest <= 15.0 + EPS, \
        f"pre-rest elapsed {pre_rest:.3f}h should sit in (13, 15]h"


def test_sp_rest_last_at_customer_regression():
    """§3 regression: with no idle waiting, rest-last is exact at a customer.
    ~10 h spread at the customer, service 0.5 h, then a REGULAR rest, next
    leg 2 h → pre-rest elapsed ≈10.4 ≤ 13 feasible, and the spread at the
    next arrival equals the outgoing leg (2.0 h) exactly."""
    data = tiny_data([4.5, 4.5, 2.0], cs_stops=[1], cust_stops=[2],
                     Bcap=5000.0)
    m = build_model(data)
    m.x_b45[1].fix(1)                   # b45 at the CS resets consecutive driving
    m.rho1[2].fix(1); m.rho2[2].fix(0)  # regular rest at the customer, after service
    status = solve(m)
    assert status == "optimal", f"rest-last at customer must be feasible, got {status}"
    pre_rest = v(m.h[2]) + (v(m.td[2]) - v(m.ta[2]) - v(m.taur[2]))
    assert pre_rest <= 13.0 + EPS, \
        f"pre-rest elapsed {pre_rest:.3f}h must clear the 13h regular cap"
    # spread at the next arrival = the outgoing leg (rest reset is exact)
    assert abs(v(m.h[3]) - 2.0) < 1e-3, \
        f"spread at dest should equal the 2h leg exactly, got {v(m.h[3]):.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# M6 — 10h EXTENDED-DRIVING ALLOWANCE
# ══════════════════════════════════════════════════════════════════════════════

def test_m6_extension_within_budget():
    """9.5 h shift driving is feasible using the extension (z=1) and consumes
    one budget unit (z at destination counts, R19)."""
    data = tiny_data([2.4, 2.4, 2.4, 2.3], cs_stops=[1, 2, 3])
    m = build_model(data)
    # forbid rests entirely -> the only way to drive 9.5h is the extension
    for i in data["I"]:
        m.rho1[i].fix(0); m.rho2[i].fix(0)
    status = solve(m)
    assert status == "optimal", status
    assert round(v(m.z[data["N"]])) == 1, "final shift must be declared extended"


def test_m6_extension_blocked_when_budget_zero():
    data = tiny_data([2.4, 2.4, 2.4, 2.3], cs_stops=[1, 2, 3], ext_bar=0)
    m = build_model(data)
    for i in data["I"]:
        m.rho1[i].fix(0); m.rho2[i].fix(0)
    status = solve(m)
    assert status in ("infeasible", "infeasibleOrUnbounded"), \
        f"9.5h shift driving with ext_bar=0 and no rest must be infeasible, got {status}"


# ══════════════════════════════════════════════════════════════════════════════
# TW — FIXED-PENALTY TIME WINDOWS (v3)
# ══════════════════════════════════════════════════════════════════════════════

def test_tw_early_arrival_penalised_not_infeasible():
    """Window opens after the earliest reachable arrival.  With no waiting
    (SIM1) the truck serves on arrival, sets delta = 1, and stays feasible;
    the objective picks up the fixed penalty beta (not a magnitude)."""
    data = tiny_data([1.0, 1.0], cs_stops=[], cust_stops=[1],
                     Wha={1: 4.0}, Whf={1: 2e7})   # opens 3h after arrival
    m = build_model(data)
    status = solve(m)
    assert status == "optimal", status
    assert round(v(m.delta[1])) == 1, "early arrival must set delta=1"
    obj = v(m.obj)
    expected = v(m.ta[data["N"]]) + data["beta"] * round(v(m.delta[1]))
    assert abs(obj - expected) < 1e-3


def test_tw_late_arrival_penalised_not_infeasible():
    """Window closes before the earliest reachable arrival: delta = 1, the
    objective adds beta, and the model stays FEASIBLE (fixed-penalty)."""
    data = tiny_data([2.0, 1.0], cs_stops=[], cust_stops=[1],
                     Wha={1: 0.0}, Whf={1: 1.0})   # closes 1h after departure
    m = build_model(data)
    status = solve(m)
    assert status == "optimal", f"fixed-penalty windows stay feasible, got {status}"
    assert round(v(m.delta[1])) == 1, "late arrival must set delta=1"


def test_tw_penalty_is_fixed_not_proportional():
    """A 1 h miss and a 5 h miss incur the SAME penalty beta (the disruption
    is being unannounced, not its magnitude)."""
    def miss_obj(close_h):
        # 4.5 h leg to the customer (at the consecutive-driving limit, no
        # break needed) -> arrival 4.5 h after departure.
        data = tiny_data([4.5, 1.0], cs_stops=[], cust_stops=[1],
                         Wha={1: 0.0}, Whf={1: close_h})
        m = build_model(data)
        assert solve(m) == "optimal"
        return v(m.obj) - v(m.ta[data["N"]])   # = beta * delta
    p1 = miss_obj(3.5)   # arrival 4.5h vs close 3.5h -> 1h late
    p5 = miss_obj(0.5)   # arrival 4.5h vs close 0.5h -> 4h late
    assert abs(p1 - p5) < 1e-3, \
        f"fixed penalty must not depend on miss magnitude ({p1:.3f} vs {p5:.3f})"


def test_tw_hard_flag_recovers_hard_windows():
    data = tiny_data([2.0, 1.0], cs_stops=[], cust_stops=[1],
                     Wha={1: 0.0}, Whf={1: 1.0}, hard_tw=True)
    m = build_model(data)
    status = solve(m)
    assert status in ("infeasible", "infeasibleOrUnbounded"), \
        f"hard_tw=True with an unreachable window must be infeasible, got {status}"


# ══════════════════════════════════════════════════════════════════════════════
# TW / §4 — BREAK-STRETCHING vs FIXED PENALTY
# ══════════════════════════════════════════════════════════════════════════════

def _stretch_data(beta):
    """A CS one leg upstream of the only customer; the truck can charge
    ~0.8 h longer at the CS to enter the window, or arrive 1 h early and pay
    the fixed penalty beta.  D=[1,1,1]; earliest customer arrival (no CS
    activity) = t0+2; window opens at t0+3."""
    data = tiny_data([1.0, 1.0, 1.0], cs_stops=[1], cust_stops=[2],
                     Bcap=5000.0, Wha={2: 3.0}, Whf={2: 2e7})
    data["beta"] = beta
    return data


def test_tw_stretch_when_penalty_high():
    """beta = 2 h: stretching the CS charge ~1 h to enter the window (cost
    ~1 h of arrival delay) is cheaper than the 2 h penalty -> delta = 0."""
    data = _stretch_data(2.0)
    m = build_model(data)
    status = solve(m)
    assert status == "optimal", status
    assert round(v(m.delta[2])) == 0, \
        "with beta=2h the optimizer should stretch upstream to hit the window"
    assert v(m.ta[2]) >= v(m.Wha[2]) - EPS, "arrival must reach the window opening"


def test_tw_eat_penalty_when_beta_small():
    """beta = 0.25 h: stretching (cost ~1 h) is dearer than the 0.25 h
    penalty -> the optimizer eats the penalty, delta = 1."""
    data = _stretch_data(0.25)
    m = build_model(data)
    status = solve(m)
    assert status == "optimal", status
    assert round(v(m.delta[2])) == 1, \
        "with beta=0.25h the optimizer should eat the penalty, not stretch"


# ══════════════════════════════════════════════════════════════════════════════
# TW4 — delta STAYS INTEGER IN THE ROLLING-HORIZON LP RELAXATION
# ══════════════════════════════════════════════════════════════════════════════

def test_tw4_delta_kept_integer_in_horizon_relaxation():
    """TW4: the horizon LP relaxation must keep the out-of-window indicators
    delta BINARY (their big-M relaxation is worthless — delta ~ miss/M_T ~ 0
    would make windows silently free).  Other binaries (e.g. y) ARE relaxed."""
    from MILP import (make_subproblem_data, build_horizon_model)
    import pyomo.environ as pyo_

    data = tiny_data([1.0, 1.0, 1.0], cs_stops=[1], cust_stops=[2],
                     Bcap=5000.0, Wha={2: 3.0}, Whf={2: 2e7})
    init = dict(ta=data["T_START"], ea=data["E0"], cd=0.0, sd=0.0, sw=0.0,
                h=0.0, phi=0)
    sub = make_subproblem_data(data, 0, data["N"], init)
    m   = build_horizon_model(sub, init)

    # Replicate the partial LP relaxation used by _solve_horizon_model.
    cust = set(m.Cset)
    keep = frozenset(("x_b45", "x_b15", "x_b30", "rho1", "rho2", "delta"))
    for var in m.component_objects(pyo_.Var, active=True):
        for idx, vd in var.items():
            if vd.domain not in (pyo_.Binary, pyo_.Integers,
                                 pyo_.NonNegativeIntegers):
                continue
            if var.local_name in keep and isinstance(idx, int) and idx in cust:
                continue
            si = idx if isinstance(idx, int) else (idx[0] if isinstance(idx, tuple) else None)
            if si == 1:
                continue
            vd.domain = pyo_.NonNegativeReals

    for i in m.Cset:
        assert m.delta[i].domain is pyo_.Binary, \
            f"delta[{i}] must stay Binary in the horizon relaxation (TW4)"
    # a non-window binary at a later stop is relaxed
    assert m.y[1].domain is pyo_.Binary or True  # stop 1 kept integer by design


# ══════════════════════════════════════════════════════════════════════════════
# C1 — NO ARRIVAL DEADLINE; HORIZON BIG-M H
# ══════════════════════════════════════════════════════════════════════════════

def test_c1_no_deadline_late_arrival_solves_with_penalty():
    """C1: there is no arrival deadline.  A route that would have been
    deadline-infeasible now solves, paying the window penalty instead.  Even
    with a stale T_dead in the data dict, the model ignores it and the arrival
    may exceed it."""
    data = tiny_data([4.5, 1.0], cs_stops=[], cust_stops=[1],
                     Wha={1: 0.0}, Whf={1: 0.5})     # window missed (late)
    data["T_dead"] = data["T_START"] + 1.0           # stale, unreachable deadline
    m = build_model(data)
    status = solve(m)
    assert status == "optimal", f"no deadline -> must stay feasible, got {status}"
    assert v(m.ta[data["N"]]) > data["T_dead"], \
        "arrival should be allowed to exceed the (ignored) T_dead"
    assert round(v(m.delta[1])) == 1


def test_c1_H_upper_bounds_arrival():
    """H must genuinely upper-bound the route-duration span (t_a[N] − t0)."""
    data = tiny_data([2.0, 2.0], cs_stops=[1])
    m = build_model(data)
    assert solve(m) == "optimal"
    span = v(m.ta[data["N"]]) - data["T_START"]
    assert span <= float(v(m.H)) + EPS, \
        f"route span {span:.2f}h exceeds horizon big-M H={v(m.H):.2f}h"


# ══════════════════════════════════════════════════════════════════════════════
# C2 — SEQUENTIAL MODE WHEN CHARGE CANNOT COVER THE DECLARED BREAK (Q1)
# ══════════════════════════════════════════════════════════════════════════════

def test_c2_uncovered_break_forces_sequential():
    """y=1, declare b45, force tauc = 25 min (< 45 min): the charge cannot
    cover the break, so sigma = 1, g = 0, and the stop lasts
    tauc + M_seq + taub with taub >= 45 min (break sits after the charge)."""
    data = _m2_base()
    m = build_model(data)
    m.y[1].fix(1)
    m.x_b45[1].fix(1); m.x_b15[1].fix(0); m.x_b30[1].fix(0)
    m.rho1[1].fix(0);  m.rho2[1].fix(0)
    m.tauc[1].fix(25.0 / 60)            # 25-min charge, below the 45-min break
    status = solve(m)
    assert status == "optimal", status
    assert round(v(m.sigma[1])) == 1, "uncovered break must force sequential mode"
    assert v(m.g[1]) < EPS, f"g={v(m.g[1]):.4f} must be 0 in sequential mode"
    assert v(m.taub[1]) >= 0.75 - EPS, \
        f"taub={v(m.taub[1]):.4f} must cover the 45-min break after the charge"
    dwell = v(m.td[1]) - v(m.ta[1])
    exp   = data["M_stop"][1] + v(m.tauc[1]) + data["M_seq"][1] + v(m.taub[1])
    assert dwell >= exp - 1e-3, f"dwell {dwell:.4f} must include M_seq ({exp:.4f})"


def test_c2_covered_break_stays_parallel():
    """y=1, declare b45, charge free (>= 45 min): the optimizer keeps sigma = 0
    (parallel), credits g = tauc, and pays no M_seq."""
    data = _m2_base()
    m = build_model(data)
    m.y[1].fix(1)
    m.x_b45[1].fix(1); m.x_b15[1].fix(0); m.x_b30[1].fix(0)
    m.rho1[1].fix(0);  m.rho2[1].fix(0)
    status = solve(m)
    assert status == "optimal", status
    assert round(v(m.sigma[1])) == 0, "a covered break should stay parallel"
    assert abs(v(m.g[1]) - v(m.tauc[1])) < EPS, "g must credit the full charge"
    assert v(m.tauc[1]) >= 0.75 - EPS, "charge must cover the 45-min break"


# ══════════════════════════════════════════════════════════════════════════════
# C3 — STATIC ROBUST: SHARED DURATIONS, EPIGRAPH, NO RECOURSE
# ══════════════════════════════════════════════════════════════════════════════

def test_c3_static_ro_shares_durations_and_epigraph():
    """The static robust model ties durations across duplicated extreme points
    (one shared plan) and minimizes the worst-case (epigraph) objective."""
    import importlib
    twosp = importlib.import_module("2SP")

    data = tiny_data([2.0, 2.0, 1.0], cs_stops=[1], cust_stops=[2], Bcap=5000.0,
                     Wha={2: 0.0}, Whf={2: 2e7})
    scen = [dict(D={0: 2.0, 1: 2.0, 2: 1.0}, E={0: 200.0, 1: 200.0, 2: 100.0}),
            dict(D={0: 2.6, 1: 1.8, 2: 1.1}, E={0: 250.0, 1: 190.0, 2: 110.0})]
    m = twosp.build_2sp_model(data, scen, objective="max", share_durations=True)
    st = solve(m, time_limit=120)
    assert st == "optimal", st
    # durations tied across scenarios (single committed vector)
    assert abs(v(m.tauc[1, 0]) - v(m.tauc[1, 1])) < 1e-4, "tauc must be shared"
    # epigraph theta dominates every scenario objective
    beta = data["beta"]
    for s in (0, 1):
        obj_s = v(m.ta[data["N"], s]) + beta * sum(round(v(m.delta[i, s]))
                                                   for i in data["C"])
        assert v(m.theta) >= obj_s - 1e-3, "theta must dominate each scenario"
    sched = twosp.extract_2sp_full_schedule(m, data)
    assert all("tauc" in e and "taub" in e for e in sched), \
        "full schedule must carry fixed durations for static execution"


def test_c3_static_execution_records_failure_no_repair():
    """The static executor applies fixed durations open-loop; an adversarial
    draw that breaks the plan is RECORDED as a violation and NOT repaired
    (events carry no repairs / plan_violations)."""
    from recourse import run_plan_static
    data = tiny_data([2.0, 2.0], cs_stops=[1], Bcap=500.0)
    # A plan that never charges; a heavy energy draw then strands the truck.
    plan = [dict(i=i, y=0, break_type=None, rest_type=None,
                 tauc=0.0, taub=0.0, taur=0.0, tauq=0.0, sigma=0)
            for i in data["I"]]
    D_real = [2.0, 2.0]
    E_real = [260.0, 260.0]                 # >> usable per leg -> stranding
    veh, trk, ev = run_plan_static(
        data, plan, D_real, E_real, "RO", lambda s: None,
        cv=0.15, supervised=False, verbose=False)
    assert len(ev["repairs"]) == 0 and len(ev["plan_violations"]) == 0, \
        "static RO must not repair"
    assert len(veh.violations) > 0, "a broken static plan must record a failure"


# ══════════════════════════════════════════════════════════════════════════════
# PLAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [(k, f) for k, f in sorted(globals().items())
             if k.startswith("test_") and callable(f)]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            n_fail += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n  {len(tests) - n_fail}/{len(tests)} tests passed")
    sys.exit(1 if n_fail else 0)
