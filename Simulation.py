"""
simulation.py — Look-Ahead Stochastic Simulation
=================================================
Rolling-horizon policy for the battery-electric truck scheduling problem.

Algorithm (per stop)
--------------------
1.  Enumerate all *feasible* actions at the current stop
    (charge/no-charge × break-type × rest-type).
2.  Determine the horizon end-stop: the last stop whose cumulative
    *nominal* travel time from here is ≤ `horizon_hours`.
3.  For each candidate action A:
        for each scenario n in 1..N_scenarios:
            draw D_scen ← D_nom × U(1−δ, 1+δ)  per leg
            solve MILP2(start=current_stop, end=end_stop,
                        init_state, fixed_action=A, D_scen)
            objective_n ← arrival time at end_stop  (or PENALTY)
        score(A) ← mean(objective_1, …, objective_N)
4.  Choose A* = argmin score.
5.  Execute A*:
    a. Compute exact departure time td from init_state + A* durations
       (taken from the MILP2 solution of scenario 0, the "nominal").
    b. Draw ACTUAL travel time D_actual ← D_nom × U(1−δ, 1+δ).
    c. Advance vehicle state to the next stop.
6.  Repeat from the next stop.

State convention
----------------
VehicleState.sw is the shift working time at arrival at the current
stop, INCLUDING work done at that stop (service for customers; queue +
charging-counted-as-work for CS).  This matches the model's sw[i]
semantics: "working time just before the break at stop i".

For a CS stop, charging time is unknown before the action is chosen,
so sw is initialised as  sw_prev_departure + D_actual + Q_nom*q_mult
(charging will be added internally by MILP2's sw propagation for the
*next* stop).  This is a documented approximation.

Usage
-----
    from MILP import instance_realistic
    from simulation import run_simulation

    data = instance_realistic()
    results = run_simulation(data, n_scenarios=8, horizon_hours=12,
                             delta=0.20, seed=42)
"""

from __future__ import annotations

import math
import time
import random
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from MILP  import _time_bounds   # noqa – also validates MILP import
from MILP2 import solve_horizon, INFEASIBLE_PENALTY
from concurrent.futures import ProcessPoolExecutor, as_completed
import os as _os


# ── Parallel worker (must be a top-level function for pickling) ───────────
def _solve_one_scenario(args):
    """
    Worker called by ProcessPoolExecutor for a single (action, scenario) pair.

    args tuple
    ----------
    (full_data, start_stop, end_stop, init_state,
     action, scenario, rho2_rem, time_limit, relax)

    scenario is a dict with keys 'D' and 'E' (from generate_scenarios).
    """
    full_data, start_stop, end_stop, init_state, action, scenario, rho2_rem, time_limit, relax = args
    return solve_horizon(
        full_data      = full_data,
        start_stop     = start_stop,
        end_stop       = end_stop,
        init_state     = init_state,
        fixed_action   = action,
        D_override     = scenario["D"],
        E_override     = scenario.get("E"),
        rho2_remaining = rho2_rem,
        tee            = False,
        time_limit     = time_limit,
        relax          = relax,
        warm_start     = scenario.get("warm_start"),   # injected per scenario
    )



# ══════════════════════════════════════════════════════════════════════════
# VEHICLE STATE
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class VehicleState:
    """
    Complete state of the vehicle at arrival at `stop`.

    All times in hours (absolute, from route start).
    """
    stop      : int    # global stop index
    t_arr     : float  # arrival time (h, absolute)
    e_arr     : float  # SOC at arrival (kWh)
    cd        : float  # consecutive driving time accumulated (h)
    sd        : float  # shift driving time accumulated (h)
    sw        : float  # shift working time before break (h)  — see module docstring
    phi       : int    # split-break flag: 1 if b15 taken since last reset
    rho2_used : int    # number of reduced rests taken so far (week)

    def as_init_state(self) -> dict:
        """Convert to the dict format expected by MILP2."""
        return dict(ta=self.t_arr, ea=self.e_arr,
                    cd=self.cd,    sd=self.sd,
                    sw=self.sw,    phi=self.phi)

    def __repr__(self):
        return (f"VehicleState(stop={self.stop}, t={self.t_arr:.2f}h, "
                f"soc={self.e_arr:.0f}kWh, cd={self.cd:.2f}h, "
                f"sd={self.sd:.2f}h, sw={self.sw:.2f}h, "
                f"phi={self.phi}, rho2_used={self.rho2_used})")


# ══════════════════════════════════════════════════════════════════════════
# HORIZON END-STOP
# ══════════════════════════════════════════════════════════════════════════

def find_horizon_end_stop(full_data, start_stop, t_now, horizon_hours,
                          state=None):
    """
    Return the last stop reachable within `horizon_hours` of wall-clock
    time from `start_stop`, correctly accounting for mandatory HoS rests.

    A driver cannot drive for 24h straight.  Within a 24h window, the
    maximum shift working time is 13h and each rest takes at least 9h
    (reduced rest r2) — so only roughly 13h of productive time is
    available per 22h cycle.  Ignoring rests causes the horizon to be
    grossly overestimated for long windows, forcing MILP2 to solve the
    entire route as a sub-problem at every stop.

    Algorithm
    ---------
    Forward simulation along nominal leg durations:
      1. Accumulate sd (shift driving) and sw (shift working).
      2. When a rest becomes mandatory (sd ≥ Tdrv_sh1 or sw ≥ Twrk_sh),
         add the minimum rest duration (Tr2 if budget allows, else Tr1)
         to the wall-clock counter and reset sd/sw to 0.
      3. Stop when wall-clock ≥ horizon_hours.

    Service times at customer stops are included in sw but not in sd.
    Charging and queue times are ignored (conservative: horizon slightly
    shorter than necessary, which makes sub-problems smaller — better).

    Parameters
    ----------
    full_data      : dict from MILP._make_data
    start_stop     : int
    t_now          : float — current absolute time (unused; kept for API compat)
    horizon_hours  : float — wall-clock look-ahead window (hours)
    state          : VehicleState or None — carries current sd, sw, rho2_used.
                     If None, accumulators start from 0.

    Returns
    -------
    end_stop : int  (start_stop < end_stop ≤ N)
    n_rests  : int  — number of mandatory rests inserted (for verbose output)
    """
    N         = full_data["N"]
    C_set     = set(full_data["C"])
    Tdrv_sh1  = full_data["Tdrv_sh1"]   # 9h
    Twrk_sh   = full_data["Twrk_sh"]    # 13h
    Tr1       = full_data["Tr1"]         # 11h
    Tr2       = full_data["Tr2"]         # 9h

    # Initialise accumulators from current vehicle state if provided
    if state is not None:
        sd        = state.sd
        sw        = state.sw
        rho2_used = state.rho2_used
    else:
        sd = sw = 0.0
        rho2_used = 0

    wall     = 0.0           # wall-clock time consumed so far
    end_stop = start_stop + 1
    n_rests  = 0

    for j in range(start_stop, N):
        d_nom = full_data["D"].get(j, 0.0)

        # Work at the next stop (service at customer, 0 elsewhere)
        next_stop = j + 1
        work_at_next = full_data["S"].get(next_stop, 0.0) if next_stop in C_set else 0.0

        # Would driving leg j violate a shift limit?
        # Check BEFORE adding leg — if we would exceed the limit we must
        # rest first.  Use the tighter of the two limits.
        new_sd = sd + d_nom
        new_sw = sw + d_nom + work_at_next

        if new_sd > Tdrv_sh1 or new_sw > Twrk_sh:
            # Insert a mandatory rest
            rest_dur  = Tr2 if rho2_used < 3 else Tr1
            wall     += rest_dur
            n_rests  += 1
            rho2_used = rho2_used + 1 if rest_dur == Tr2 else rho2_used
            sd = sw = 0.0          # reset after rest
            new_sd = d_nom
            new_sw = d_nom + work_at_next

        wall += d_nom
        if wall > horizon_hours:
            break

        sd       = new_sd
        sw       = new_sw
        end_stop = j + 1

    return min(end_stop, N), n_rests


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATION
# ══════════════════════════════════════════════════════════════════════════

# ── ECR curve: fitted to the supplied graph data (20–70 km/h) ────────────
# Model: ECR(v) = _ECR_A/v + _ECR_B + _ECR_C·v²
# where  _ECR_A/v captures low-speed accessory/stop-and-go losses,
#        _ECR_C·v² captures aerodynamic drag.
# Fitted via scipy.curve_fit to data: v=20→1.38, 30→0.95, 40→0.68,
# 50→0.54, 60→0.55, 70→0.59 kWh/km.
# Minimum near v*=61 km/h, ECR*≈0.55 kWh/km; extrapolates smoothly beyond.
_ECR_A =  33.055   # kWh·km/h
_ECR_B =  -0.257   # kWh/km
_ECR_C =   7.2e-5  # kWh·h²/km³


def _ecr(v_kmh):
    """Energy Consumption Rate (kWh/km) at speed v (km/h)."""
    v = max(float(v_kmh), 5.0)
    return _ECR_A / v + _ECR_B + _ECR_C * v**2


def generate_scenarios(full_data, start_stop, end_stop,
                       n_scenarios, delta=0.20, seed=None,
                       correlation=0.0, zone_size=8,
                       include_best=False, include_worst=False,
                       c_base_frac=0.3):   # kept for API compat, unused
    """
    Generate stochastic travel-time + energy-consumption scenarios.

    Travel-time noise model
    -----------------------
    Each leg draws:
        log_mult[i] = α × ε_zone[z(i)] + (1−α) × ε_leg[i]
        D_scen[i]   = D_nom[i] × exp(log_mult[i])

    where α = correlation, z(i) = i // zone_size.
    σ is chosen so that exp(±3σ) ≈ 1 ± δ  (log-normal ≈ uniform for small δ).
    correlation=0 → independent per leg; correlation=1 → all legs in a zone
    share the same draw (same factor model as the literature).

    Energy coupling
    ---------------
    When full_data["km"] is available (physical leg distances), scenario
    energy is derived from the scenario travel speed:

        v_scen[i]  = km[i] / D_scen[i]
        E_scen[i]  = km[i] × [c_base + c_aero × v_scen[i]²]

    where c_base and c_aero are calibrated to E_nom at v_nom.
    This correctly captures: slow leg → less aero drag → lower kWh/km;
    fast leg → more aero drag → higher kWh/km.
    When km is unavailable, E_scen = E_nom (no coupling).

    Deterministic extremes
    ----------------------
    include_best  : prepend a scenario where every leg has mult = 1−δ
    include_worst : append  a scenario where every leg has mult = 1+δ
    These are useful for robustness checks but excluded from mean/std.

    Parameters
    ----------
    correlation  : float in [0,1] — spatial correlation between nearby legs
    zone_size    : int — number of consecutive legs per correlation zone
    include_best : bool — prepend best-case (all minimum travel times)
    include_worst: bool — append  worst-case (all maximum travel times)
    c_base_frac  : float — share of nominal consumption that is speed-independent

    Returns
    -------
    list of scenario dicts, each with keys 'D' and 'E':
        scenario["D"] : {leg_index: duration_hours}
        scenario["E"] : {leg_index: energy_kWh}
        scenario["is_best"]  : bool
        scenario["is_worst"] : bool
    """
    rng  = np.random.default_rng(seed)
    legs = list(range(start_stop, end_stop))
    n_legs = len(legs)
    if n_legs == 0:
        return []

    # Zone assignment for correlation model
    n_zones = max((l - start_stop) // zone_size for l in legs) + 1
    sigma   = np.log(1 + delta) / 3.0   # log-normal σ: exp(3σ) ≈ 1+δ

    # Speed-energy coupling: calibrate c_base and c_aero
    km_dict  = full_data.get("km", None)
    e_dict   = full_data["E"]

    def _E_scen(i, D_si):
        """
        Scenario energy for leg i given scenario duration D_si (h).
        Uses the fitted ECR(v) curve: E = km[i] * ECR(km[i]/D_si).
        Fast scenarios (small D_si) → higher speed → more aero drag.
        Slow scenarios (large D_si) → lower speed → less aero drag.
        Falls back to nominal E when km data is unavailable.
        """
        if km_dict is None:
            return e_dict.get(i, 0.0)
        km_i = km_dict.get(i, 0.0)
        if km_i <= 0 or D_si <= 0:
            return e_dict.get(i, 0.0)
        v_scen = km_i / D_si          # km/h
        return max(km_i * _ecr(v_scen), 0.0)

    def _make_scenario(mults, is_best=False, is_worst=False):
        D_scen = {}; E_scen = {}
        for k, i in enumerate(legs):
            D_nom   = full_data["D"].get(i, 0.0)
            D_si    = max(D_nom * mults[k], 1e-4)
            D_scen[i] = D_si
            E_scen[i] = _E_scen(i, D_si)
        return {"D": D_scen, "E": E_scen,
                "is_best": is_best, "is_worst": is_worst}

    scenarios = []

    # Deterministic best case
    if include_best:
        mults = [(1.0 - delta)] * n_legs
        scenarios.append(_make_scenario(mults, is_best=True))

    # Stochastic scenarios
    for _ in range(n_scenarios):
        eps_zone = rng.standard_normal(n_zones)
        eps_leg  = rng.standard_normal(n_legs)
        mults = []
        for k, i in enumerate(legs):
            z   = (i - start_stop) // zone_size
            lm  = (      correlation  * sigma * eps_zone[z]
                   + (1 - correlation) * sigma * eps_leg[k])
            mults.append(np.exp(lm))
        scenarios.append(_make_scenario(mults))

    # Deterministic worst case
    if include_worst:
        mults = [(1.0 + delta)] * n_legs
        scenarios.append(_make_scenario(mults, is_worst=True))

    return scenarios


# ══════════════════════════════════════════════════════════════════════════
# ACTION ENUMERATION
# ══════════════════════════════════════════════════════════════════════════

def enumerate_actions(stop_global, state: VehicleState, full_data,
                      charge_only=False):
    """
    Return the list of structurally feasible action dicts at `stop_global`.

    charge_only mode
    ----------------
    When charge_only=True, only the charge/no-charge decision is fixed.
    Break and rest decisions are left as None (free for MILP2 to optimise).
    This reduces the action space from ~10 to 2 at CS stops and 1 at
    customer stops, dramatically speeding up the look-ahead.

    Full mode (charge_only=False, default)
    ---------------------------------------
    All combinations of (y, break_type, rest_type) are enumerated subject
    to structural feasibility: b30 only when phi=1, r2 only when
    rho2_used<3, break and rest mutually exclusive.
    """
    K_set = set(full_data["K"])
    N     = full_data["N"]
    is_CS = (stop_global in K_set)

    def _skip_y1():
        return (state.e_arr > 0.98 * full_data["Ecap"] or
                stop_global >= N)

    if charge_only:
        if is_CS:
            actions = [dict(y=0, break_type=None, rest_type=None)]
            if not _skip_y1():
                actions.append(dict(y=1, break_type=None, rest_type=None))
        else:
            actions = [dict(y=0, break_type=None, rest_type=None)]
        return actions

    # Full enumeration
    break_types = [None, "b45", "b15"]
    if state.phi == 1:
        break_types.append("b30")

    rest_types = [None, "r1"]
    if state.rho2_used < 3:
        rest_types.append("r2")

    actions = []
    for brk in break_types:
        for rst in rest_types:
            if brk is not None and rst is not None:
                continue
            if is_CS:
                for y_val in (0, 1):
                    if y_val == 1 and _skip_y1():
                        continue
                    actions.append(dict(y=y_val,
                                        break_type=brk,
                                        rest_type=rst))
            else:
                actions.append(dict(y=0, break_type=brk, rest_type=rst))

    return actions


# ══════════════════════════════════════════════════════════════════════════
# EVALUATE A SINGLE ACTION ACROSS SCENARIOS
# ══════════════════════════════════════════════════════════════════════════

def evaluate_action(full_data, start_stop, end_stop, state: VehicleState,
                    action, scenarios, time_limit=20, tee=False,
                    n_workers=1, relax=True, criterion="mean"):
    """
    Solve MILP2 for each scenario with `action` fixed at `start_stop`.

    Parameters
    ----------
    scenarios  : list of scenario dicts from generate_scenarios,
                 each with keys "D", "E", "is_best", "is_worst".
    criterion  : str — how to score an action across scenarios:
        "mean"  : expected arrival time (default) — minimise E[obj]
        "worst" : minimax — minimise max(obj) — robust/conservative
        "best"  : optimistic — minimise min(feasible obj)

    Returns
    -------
    score     : float  — criterion value (mean / worst / best)
    std_obj   : float  — std dev of feasible scenario objectives
    n_feasible: int
    first_sol : dict or None — MIP solution for the best-D scenario
    objs      : list[float] — raw per-scenario objectives
    """
    rho2_rem = 3 - state.rho2_used
    init_st  = state.as_init_state()

    # Build one argument tuple per scenario (all picklable plain types)
    arg_list = [
        (full_data, start_stop, end_stop, init_st,
         action, scenario, rho2_rem, time_limit, relax)
        for scenario in scenarios
    ]

    if n_workers > 1:
        # Submit all scenarios to the process pool; collect in submission order
        results_ordered = [None] * len(arg_list)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_solve_one_scenario, a): idx
                       for idx, a in enumerate(arg_list)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results_ordered[idx] = fut.result()
                except Exception as exc:
                    # Worker crashed: treat as infeasible
                    results_ordered[idx] = {"feasible": False,
                                            "obj": INFEASIBLE_PENALTY}
        res_list = results_ordered
    else:
        res_list = [_solve_one_scenario(a) for a in arg_list]

    objs = [r["obj"] for r in res_list]

    # Nominal MIP re-solve on the first feasible scenario
    first_lp  = next((r for r in res_list if r.get("feasible")), None)
    nom_scen  = next((scenarios[i] for i, r in enumerate(res_list)
                      if r.get("feasible")), None)
    if relax and first_lp is not None and nom_scen is not None:
        first_sol = solve_horizon(
            full_data      = full_data,
            start_stop     = start_stop,
            end_stop       = end_stop,
            init_state     = init_st,
            fixed_action   = action,
            D_override     = nom_scen["D"],
            E_override     = nom_scen.get("E"),
            rho2_remaining = rho2_rem,
            tee            = False,
            time_limit     = time_limit * 4,
            relax          = False,
            warm_start     = nom_scen.get("warm_start"),  # free-horizon solution
        )
    else:
        first_sol = first_lp

    n_feasible = sum(1 for o in objs if o < INFEASIBLE_PENALTY / 2)
    if n_feasible == 0:
        return INFEASIBLE_PENALTY, 0.0, 0, None, objs

    feasible_objs = [o for o in objs if o < INFEASIBLE_PENALTY / 2]
    std_obj = float(np.std(feasible_objs)) if n_feasible > 0 else 0.0

    # Criterion scoring
    # All three use ALL objs (infeasible = PENALTY) except "best"
    # which ignores infeasible scenarios (optimistic view).
    if criterion == "worst":
        # Minimax: prefer the action that minimises the worst-case outcome.
        # Infeasible scenarios contribute INFEASIBLE_PENALTY → an action
        # with any infeasible scenario will have worst-case = PENALTY.
        score = float(max(objs))
    elif criterion == "best":
        # Optimistic: use the best feasible scenario only.
        score = float(min(feasible_objs))
    else:
        # mean (default): expected value across all scenarios.
        # Infeasible scenarios are included as PENALTY so partially-feasible
        # actions are penalised correctly.
        score = float(np.mean(objs))

    return score, std_obj, n_feasible, first_sol, objs


# ══════════════════════════════════════════════════════════════════════════
# LOOK-AHEAD: SELECT BEST ACTION
# ══════════════════════════════════════════════════════════════════════════

def _prune_actions(actions, stop_global, state, full_data, delta):
    """
    Remove structurally dominated actions before evaluating scenarios.

    Rule 1 — Mandatory charge:
      If current SOC minus worst-case energy to next CS < Emin,
      the truck MUST charge here. Drop all y=0 options.

    Rule 2 — Mandatory break (consecutive driving):
      If cd + worst-case D_next > Tdrv_cons, a break that resets cd
      is mandatory. Drop actions with no break AND no rest.

    Rule 3 — Mandatory rest (shift driving):
      If sd + worst-case D_next > Tdrv_sh, a rest is mandatory.
      Drop actions with no rest.

    Rule 4 — Mandatory rest (shift working):
      If sw + worst-case D_next + dwell_next > Twrk_sh, a rest
      is mandatory. Drop actions with no rest.

    "Worst-case" uses D_nom * (1 + delta) for the next leg.
    """
    N     = full_data["N"]
    K_set = set(full_data["K"])
    C_set = set(full_data["C"])

    if stop_global >= N - 1:
        return actions   # last stop — no pruning needed

    # Worst-case next leg travel time
    D_next_wc = full_data["D"].get(stop_global, 0.0) * (1.0 + delta)

    # ── Rule 1: must charge? ───────────────────────────────────────────
    must_charge = False
    if stop_global in K_set:
        # Energy needed to reach next CS from this stop's departure
        # Use e_next (pre-computed in _make_data)
        e_next = 0.0
        cur = stop_global
        while cur < N:
            e_next += full_data["E"].get(cur, 0.0) * (1.0 + delta)
            cur += 1
            if cur in K_set or cur == N:
                break
        if state.e_arr - e_next < full_data["Emin"] + 1e-3:
            must_charge = True

    # ── Rule 2: must reset cd (break or rest)? ───────────────────────
    must_reset_cd = (state.cd + D_next_wc > full_data["Tdrv_cons"] - 1e-3)

    # ── Rules 3 & 4: must rest? ──────────────────────────────────────
    dwell_next = (full_data["S"].get(stop_global + 1, 0.0)
                  if stop_global + 1 in C_set else 0.0)
    must_rest = (
        state.sd + D_next_wc > full_data["Tdrv_sh1"] - 1e-3 or
        state.sw + D_next_wc + dwell_next > full_data["Twrk_sh"] - 1e-3
    )

    pruned = []
    n_pruned = 0
    for a in actions:
        y   = a.get("y", 0)
        brk = a.get("break_type")
        rst = a.get("rest_type")

        skip = False
        if must_charge and y == 0 and stop_global in K_set:
            skip = True
        if must_reset_cd and brk is None and rst is None:
            skip = True
        if must_rest and rst is None:
            skip = True

        if skip:
            n_pruned += 1
        else:
            pruned.append(a)

    # Safety: always keep at least one action
    if not pruned:
        return actions
    return pruned, n_pruned


def select_best_action(full_data, stop_global, state: VehicleState,
                       n_scenarios=10, horizon_hours=12, delta=0.20,
                       scenario_seed=None, time_limit=20, tee=False,
                       verbose=True, n_workers=1, relax=True,
                       charge_only=False, criterion="mean",
                       correlation=0.0, zone_size=8,
                       include_best=False, include_worst=False,
                       c_base_frac=0.3,
                       prev_nom_sol=None,
                       log_fh=None):
    """
    Evaluate all feasible actions at `stop_global` using the look-ahead.

    Parameters
    ----------
    charge_only   : only fix y; break/rest free for MILP2.
    criterion     : "mean" | "worst" | "best".
    prev_nom_sol  : nominal MIP sol from previous stop (warm-start seed).
    log_fh        : open file handle for log output.
    """
    def _p(msg):
        """Print to stdout (if verbose) and to log file."""
        if verbose: print(msg)
        if log_fh:
            try:
                print(msg, file=log_fh)
            except Exception:
                pass   # never crash on logging

    # ── Enumerate + prune actions ─────────────────────────────────────
    raw_actions = enumerate_actions(stop_global, state, full_data,
                                    charge_only=charge_only)
    prune_result = _prune_actions(raw_actions, stop_global, state,
                                  full_data, delta)
    if isinstance(prune_result, tuple):
        actions, n_pruned = prune_result
    else:
        actions, n_pruned = prune_result, 0

    end_stop, n_rests = find_horizon_end_stop(full_data, stop_global,
                                              state.t_arr, horizon_hours,
                                              state=state)

    stop_type  = ("CS"   if stop_global in set(full_data["K"])
                  else "CUST" if stop_global in set(full_data["C"])
                  else "ORIG")
    worker_str = f"  {n_workers}w" if n_workers > 1 else ""
    rest_str   = f" +{n_rests}rest" if n_rests else ""
    mode_str   = f"[{criterion}"
    if charge_only: mode_str += ",co"
    mode_str  += "]"
    nom_travel = sum(full_data["D"].get(j, 0) for j in range(stop_global, end_stop))
    prune_str  = f"  pruned={n_pruned}" if n_pruned else ""
    warm_str   = "  ws=prev" if prev_nom_sol else ""

    _p(f"\n[LA] stop {stop_global} ({stop_type})"
       f"  t={state.t_arr:.3f}h"
       f"  soc={state.e_arr:.0f}kWh"
       f"  cd={state.cd:.2f}h  sd={state.sd:.2f}h  sw={state.sw:.2f}h"
       f"  phi={state.phi}  r2={state.rho2_used}")
    _p(f"     horizon [{stop_global}->{end_stop}]"
       f"  travel={nom_travel:.2f}h{rest_str}"
       f"  {len(actions)} actions x {n_scenarios} scen"
       f"{worker_str}  {mode_str}{prune_str}{warm_str}")

    scenarios = generate_scenarios(
        full_data, stop_global, end_stop,
        n_scenarios=n_scenarios, delta=delta, seed=scenario_seed,
        correlation=correlation, zone_size=zone_size,
        include_best=include_best, include_worst=include_worst,
        c_base_frac=c_base_frac)

    # ── Warm-start (A): tail of previous nominal solution ─────────────
    tail_warm = None
    if prev_nom_sol and len(prev_nom_sol) > 1:
        tail_warm = []
        for s in prev_nom_sol[1:]:
            s2 = dict(s); s2["i"] = s["i"] - 1
            if s2["i"] >= 0:
                tail_warm.append(s2)

    # ── Warm-start (B): free-horizon solve (no action fixed) ──────────
    free_sol = None
    if prev_nom_sol is not None:
        _t0_free = time.perf_counter()
        _free = solve_horizon(
            full_data      = full_data,
            start_stop     = stop_global,
            end_stop       = end_stop,
            init_state     = state.as_init_state(),
            fixed_action   = None,
            rho2_remaining = 3 - state.rho2_used,
            tee            = False,
            time_limit     = max(time_limit, 15),
            relax          = relax,
            warm_start     = tail_warm,
        )
        _t_free = time.perf_counter() - _t0_free
        si = _free.get("solve_info", {})
        _ws_flag = "ws=yes" if si.get("had_warm") else "ws=no"
        if _free["feasible"]:
            free_sol = _free["sol"]
            _p(f"     [FREE] obj={_free['obj']:.3f}h"
               f"  {_free['status']}"
               f"  {_t_free:.1f}s"
               f"  {_ws_flag}"
               f"  {si.get('n_vars','?')}v/{si.get('n_cons','?')}c")
        else:
            _p(f"     [FREE] infeasible  {_t_free:.1f}s  {_ws_flag}")

    # Attach free solution as warm-start to every scenario
    if free_sol is not None:
        for scen in scenarios:
            scen["warm_start"] = free_sol

    scores      = []
    best_score  = math.inf
    best_action = actions[0]
    nominal_sol = None

    for action in actions:
        t0 = time.perf_counter()
        score, std_obj, n_feas, first_sol, raw_objs = evaluate_action(
            full_data, stop_global, end_stop, state,
            action, scenarios, time_limit=time_limit, tee=tee,
            n_workers=n_workers, relax=relax, criterion=criterion)
        elapsed = time.perf_counter() - t0

        scores.append((action, score, std_obj, n_feas, raw_objs))

        brk = action.get("break_type") or "-"
        rst = action.get("rest_type")  or "-"
        y   = action.get("y", "-")
        # warm-start status from first feasible solve
        ws_tag = ""
        if first_sol and first_sol.get("solve_info"):
            si = first_sol["solve_info"]
            ws_tag = f"  ws={'Y' if si.get('had_warm') else 'N'}  {si.get('wall_s',0):.1f}s"
        _p(f"     y={y}  brk={brk:3}  rst={rst:2}"
           f"  {criterion}={score:.3f}h  std={std_obj:.3f}h"
           f"  ok={n_feas}/{len(scenarios)}"
           f"{ws_tag}  ({elapsed:.1f}s)")

        if score < best_score:
            best_score  = score
            best_action = action
            nominal_sol = first_sol

    scores.sort(key=lambda x: x[1])

    ba = best_action
    _p(f"  -> CHOSEN y={ba.get('y','-')}"
       f"  brk={ba.get('break_type') or '-'}"
       f"  rst={ba.get('rest_type') or '-'}"
       f"  ({criterion}={best_score:.3f}h)")

    return best_action, scores, nominal_sol



def _energy_after_charging(ea, tauc, full_data):
    """
    Given arrival SOC `ea` (kWh) and charging duration `tauc` (h),
    return departure SOC `ed` (kWh) by evaluating the PWL charging curve.

    The PWL maps SoC → cumulative-charge-time (Ebar[r] → Tbar[r]).
    We invert: given ea find Ta, add tauc to get Td, invert again for ed.
    This is always valid regardless of whether tauc came from LP or MIP.
    """
    Ebar = full_data["Ebar"]   # {r: kWh}
    Tbar = full_data["Tbar"]   # {r: h}
    Ecap = full_data["Ecap"]
    Emin = full_data["Emin"]

    rs = sorted(Ebar)          # sorted breakpoint indices
    Es = [Ebar[r] for r in rs]
    Ts = [Tbar[r] for r in rs]

    def energy_to_time(e):
        """Interpolate: given SoC e, return cumulative charge time T(e)."""
        e = max(Es[0], min(Es[-1], e))
        for k in range(len(Es) - 1):
            if Es[k] <= e <= Es[k + 1]:
                if Es[k + 1] == Es[k]:
                    return Ts[k]
                frac = (e - Es[k]) / (Es[k + 1] - Es[k])
                return Ts[k] + frac * (Ts[k + 1] - Ts[k])
        return Ts[-1]

    def time_to_energy(t):
        """Interpolate: given cumulative charge time T, return SoC e(T)."""
        t = max(Ts[0], min(Ts[-1], t))
        for k in range(len(Ts) - 1):
            if Ts[k] <= t <= Ts[k + 1]:
                if Ts[k + 1] == Ts[k]:
                    return Es[k]
                frac = (t - Ts[k]) / (Ts[k + 1] - Ts[k])
                return Es[k] + frac * (Es[k + 1] - Es[k])
        return Es[-1]

    Ta = energy_to_time(ea)
    Td = Ta + tauc
    ed = time_to_energy(Td)
    return max(Emin, min(Ecap, ed))

# ══════════════════════════════════════════════════════════════════════════
# STATE TRANSITION
# ══════════════════════════════════════════════════════════════════════════

def advance_state(full_data, state: VehicleState, action,
                  milp_sol, actual_D_override=None,
                  delta=0.20, rng=None):
    """
    Apply `action` at state.stop and advance to the next stop.

    Execution durations (taub, tauc, taur, tauq) are taken from
    `milp_sol` — the MILP2 result for the chosen action on the nominal
    (or first feasible) scenario.  This ensures the driver executes the
    MILP-optimal duration, not just the minimum.

    Actual travel time to the next stop is drawn from U(1−δ, 1+δ) × D_nom,
    unless `actual_D_override` is provided (useful for replay / testing).

    State transition formulas
    -------------------------
    The formulas below mirror the model's HoS accumulator propagation
    (MILP.py, constraints cd_prop / sd_prop / sw_prop) but evaluated
    using the ACTUAL travel time D_actual.

    cd reset  ← b45 or b30 or rest  (ri indicator)
    sd reset  ← rest only           (rho indicator)
    sw reset  ← rest only
    phi       ← 1 if b15 and no reset; 0 if reset; unchanged otherwise

    Note on sw (exact MILP match via split responsibility)
    -------------------------------------------------------
    state.sw is defined as accumulated shift working time at arrival,
    EXCLUDING work at the current stop (service S, queue Q*y, tauc).
    MILP2 adds work at stop 0 internally via its init_sw constraint
    (Q*y[0] + u[0] for CS; S[0] for customers), so the result is exact
    and independent of the action being evaluated.

    advance_state propagates:
      sw_dep  = 0 if rest, else state.sw
      sw_new  = sw_dep + man_time + D_actual
    (no work_at_next — MILP2 handles it at the next call).

    Returns
    -------
    VehicleState at the next stop
    """
    stop   = state.stop
    N      = full_data["N"]
    C_set  = set(full_data["C"])
    K_set  = set(full_data["K"])

    if rng is None:
        rng = np.random.default_rng()

    # ── Extract durations and actual decisions from MILP solution ─────────
    # When charge_only=True, action has break_type=None/rest_type=None, but
    # milp_sol contains the MILP2-optimal break/rest decisions.  We must read
    # brk/rst from the solution, not from the action, otherwise the executed
    # trajectory has no breaks and the driver immediately violates HoS.
    if milp_sol is not None and milp_sol.get("sol"):
        s0 = milp_sol["sol"][0]
        taub_exec = s0["taub"]
        taur_exec = s0["taur"]
        tauc_exec = s0["tauc"]
        tauq_exec = s0["tauq"]
        # Read actual break/rest type from MILP solution (critical for charge_only)
        brk = ("b45" if s0["b45"] else
               "b15" if s0["b15"] else
               "b30" if s0["b30"] else None)
        rst = ("r1"  if s0["rho1"] else
               "r2"  if s0["rho2"] else None)
        y   = int(s0.get("y", action.get("y", 0)))
    else:
        # Fallback: minimum required durations from the action
        brk        = action.get("break_type")
        rst        = action.get("rest_type")
        taub_exec  = (full_data["Tb45"] if brk == "b45" else
                      full_data["Tb15"] if brk == "b15" else
                      full_data["Tb30"] if brk == "b30" else 0.0)
        taur_exec  = (full_data["Tr1"]  if rst == "r1"  else
                      full_data["Tr2"]  if rst == "r2"  else 0.0)
        tauc_exec  = 0.0
        tauq_exec  = 0.0
        y   = int(action.get("y", 0))

    # ── Departure time from current stop ─────────────────────────────────
    is_CS   = (stop in K_set)
    is_cust = (stop in C_set)
    S_stop  = full_data["S"].get(stop, 0.0)
    man_time = full_data["M"].get(stop, 5.0 / 60) if (brk or rst) else 0.0

    if is_CS:
        td = state.t_arr + tauq_exec + tauc_exec + taub_exec + taur_exec + man_time
    elif is_cust:
        td = state.t_arr + S_stop + taub_exec + taur_exec + man_time
    else:
        td = state.t_arr   # origin or other non-stop

    # ── Draw actual travel time to next stop ─────────────────────────────
    if stop >= N:
        # Already at destination
        return state

    if actual_D_override is not None:
        D_actual = actual_D_override
    else:
        D_nom    = full_data["D"].get(stop, 0.0)
        mult     = rng.uniform(1.0 - delta, 1.0 + delta)
        D_actual = max(D_nom * mult, 1e-4)

    next_stop = stop + 1
    t_arr_new = td + D_actual

    # ── Energy update ─────────────────────────────────────────────────────
    # Always compute ed from the PWL charging curve given (ea, tauc_exec).
    # This is correct regardless of whether milp_sol came from LP relaxation
    # or MIP — LP solutions have fractional ed values that can be wrong.
    if y and is_CS and tauc_exec > 0:
        e_dep_new = _energy_after_charging(state.e_arr, tauc_exec, full_data)
    else:
        e_dep_new = state.e_arr
    E_leg     = full_data["E"].get(stop, 0.0)
    e_arr_new = max(e_dep_new - E_leg, full_data["Emin"])

    # ── HoS accumulators ─────────────────────────────────────────────────
    ri  = (brk in ("b45", "b30")) or (rst in ("r1", "r2"))   # cd reset
    rho = rst in ("r1", "r2")                                  # sd/sw reset

    cd_dep = 0.0 if ri  else state.cd
    sd_dep = 0.0 if rho else state.sd
    # ── sw update ─────────────────────────────────────────────────────────
    # state.sw = accumulated shift working time at arrival, EXCLUDING work
    # at the current stop.  MILP2 adds work at stop 0 via its init_sw
    # constraint (Q*y[0] + u[0] for CS; S[0] for customers), so init_sw is
    # exact and action-independent.
    #
    # In advance_state, we now know the action and can add work at the current
    # stop exactly, then propagate without any approximation.
    #
    #   work_at_current:
    #     CS       → Q*y (queue, if charging)
    #              + tauc * (no break/rest)   (charge counts as work unless break)
    #     customer → S  (service always counts as work)
    #     other    → 0
    #
    #   sw_k = state.sw + work_at_current   ← full sw at stop k (MILP convention)
    #   sw_dep = 0 if rest else sw_k        ← reset on rest
    #   sw_new = sw_dep + man_time + D_actual ← carry to next stop (pre-work)
    if is_CS:
        work_at_current = (tauq_exec * y
                           + (tauc_exec if (y and not brk and not rst) else 0.0))
    elif is_cust:
        work_at_current = S_stop
    else:
        work_at_current = 0.0

    sw_k   = state.sw + work_at_current    # exact sw[k] matching MILP convention
    sw_dep = 0.0 if rho else sw_k          # reset on rest

    # Update accumulators with actual travel time
    cd_new = cd_dep + D_actual
    sd_new = sd_dep + D_actual

    # sw at next stop — exact match to MILP sw_prop constraint:
    #   sw[i+1] = sw[i] - l4[i]            (reset to 0 if rest, else keep)
    #           + Man[i]*(x[i]+rho[i])      (manoeuver at current stop)
    #           + D_actual[i]               (driving leg)
    #           + work_at_next[i+1]         (work done AT next stop before break)
    #
    # work_at_next:
    #   customer → service time S (always counts as work)
    #   CS       → queue Q*y  (work only if charging)
    #            + tauc        (counts as work UNLESS a break/rest is declared;
    #                           the MILP's u[i] term captures this — we can't
    #                           know next stop's action yet, so we conservatively
    #                           include the full tauc here; MILP2 receives this
    #                           sw as an upper bound and will enforce the reset)
    #   The queue and charge at the CURRENT CS stop are already counted in
    #   sw_dep (they happened before the break); we only add next-stop work.
    # sw_new: carry forward — manoeuver (post-break, work) + driving.
    # Work at next stop is NOT added; MILP2's init_sw adds it.
    sw_new = sw_dep + man_time + D_actual

    # ── phi update ───────────────────────────────────────────────────────
    if ri or rho:
        phi_new = 0   # any driving reset clears the split-break flag
    elif brk == "b15":
        phi_new = 1
    else:
        phi_new = state.phi   # unchanged

    # ── rho2_used ────────────────────────────────────────────────────────
    rho2_used_new = state.rho2_used + (1 if rst == "r2" else 0)

    new_state = VehicleState(
        stop      = next_stop,
        t_arr     = t_arr_new,
        e_arr     = e_arr_new,
        cd        = cd_new,
        sd        = sd_new,
        sw        = sw_new,
        phi       = phi_new,
        rho2_used = rho2_used_new,
    )
    return new_state, td, D_actual


# ══════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════

def run_simulation(full_data,
                   n_scenarios   = 10,
                   horizon_hours = 12.0,
                   delta         = 0.20,
                   seed          = 42,
                   time_limit    = 20,
                   tee           = False,
                   verbose       = True,
                   n_workers     = None,
                   relax         = True,
                   charge_only   = False,
                   criterion     = "mean",
                   correlation   = 0.0,
                   zone_size     = 8,
                   include_best  = False,
                   include_worst = False,
                   c_base_frac   = 0.3,
                   run_id        = None):
    """
    n_workers    : parallel processes. None → auto (min(cpu_count, n_scenarios)).
    charge_only  : only fix charge decision; breaks/rests free for MILP2.
    criterion    : "mean" | "worst" | "best" — action selection criterion.
    correlation  : spatial correlation between nearby legs [0,1].
    zone_size    : legs per correlation zone.
    include_best : add best-case scenario to each stop's scenario set.
    include_worst: add worst-case scenario to each stop's scenario set.
    c_base_frac  : speed-independent fraction for energy-speed coupling.
    run_id       : str or None — base name for output files (auto-generated when None).
    """
    if n_workers is None:
        n_workers = min(_os.cpu_count() or 1, n_scenarios)
    """
    Run the full look-ahead simulation from stop 0 to stop N.

    Parameters
    ----------
    full_data     : dict from MILP._make_data
    n_scenarios   : int   — scenarios per action per stop
    horizon_hours : float — look-ahead window in hours
    delta         : float — travel-time noise half-width (e.g. 0.20 = ±20 %)
    seed          : int   — master RNG seed
    time_limit    : float — solver time limit per MILP2 call (seconds)
    tee           : bool  — print solver output
    verbose       : bool  — print simulation log

    Returns
    -------
    results : dict with keys
        'states'       : list of VehicleState (one per stop, in order)
        'actions'      : list of action dicts chosen at each stop
        'scores_log'   : list of score lists (one per stop)
        'total_time'   : float — simulated arrival time at destination (h)
        'wall_clock'   : float — real computation time (s)
    """
    import datetime as _dt, json as _json, os as _os2
    master_rng   = np.random.default_rng(seed)
    N            = full_data["N"]
    prev_nom_sol = None   # warm-start: nominal MIP solution from previous stop

    # ── Output folders + log file ─────────────────────────────────────────
    for _d in ("logs", "figures", "solutions"):
        _os2.makedirs(_d, exist_ok=True)
    _ts    = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _title = full_data.get("title", "run")
    _rid   = run_id or f"{_title}_S{n_scenarios}_H{horizon_hours:.0f}_{_ts}"
    _logpath = _os2.path.join("logs",      f"{_rid}.txt")
    _figpath = _os2.path.join("figures",   f"{_rid}.png")
    _solpath = _os2.path.join("solutions", f"{_rid}.json")
    _log = open(_logpath, "w", buffering=1)

    def _lprint(*args):
        line = " ".join(str(a) for a in args)
        if verbose: print(line)
        print(line, file=_log)
    # ─────────────────────────────────────────────────────────────────────

    # Deterministic seed per stop (reproducible but varied)
    def stop_seed(s):
        return int(master_rng.integers(0, 2**31))

    # Initial state: departure at 08:00 (T_START stored in instance data)
    T_START = full_data.get("T_START", 8.0)
    state = VehicleState(
        stop      = 0,
        t_arr     = T_START,
        e_arr     = full_data["E0"],
        cd        = 0.0,
        sd        = 0.0,
        sw        = 0.0,
        phi       = 0,
        rho2_used = 0,
    )

    states       = [state]
    actions      = []
    scores_log   = []
    td_list      = []          # departure time from each stop (h)
    D_actual_list = []         # actual travel time on each leg (h)
    durations_list = []        # per-stop activity durations dict
    wall_start   = time.perf_counter()

    relax_str = "LP-relax" if relax else "MIP"
    crit_str  = f"  [{criterion}"
    if charge_only: crit_str += ", charge-only"
    if correlation > 0: crit_str += f", rho={correlation:.2f}"
    if include_best or include_worst:
        crit_str += f", +{'B' if include_best else ''}{'W' if include_worst else ''}"
    crit_str += "]"
    _lprint(f"\n{'='*65}")
    _lprint(f"  SIMULATION START   ({_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    _lprint(f"  Instance : {_title}   run_id={_rid}")
    _lprint(f"  Route    : {N} stops  departure=08:00")
    _lprint(f"  Settings : N_scen={n_scenarios}  H={horizon_hours}h  d={delta:.0%}  "
            f"workers={n_workers}  {relax_str}{crit_str}")
    _lprint(f"  Log      : {_logpath}")
    _lprint(f"{'='*65}")

    for stop in range(N):
        if stop == 0:
            # Route origin: no action (no break/charge at departure)
            action     = dict(y=0, break_type=None, rest_type=None)
            nom_sol    = None
            score_list = [(action, 0.0, 0.0, 0, [])]
            prev_nom_sol = None
        else:
            action, score_list, nom_sol = select_best_action(
                full_data     = full_data,
                stop_global   = stop,
                state         = state,
                n_scenarios   = n_scenarios,
                horizon_hours = horizon_hours,
                delta         = delta,
                scenario_seed = stop_seed(stop),
                time_limit    = time_limit,
                tee           = tee,
                verbose       = verbose,
                n_workers     = n_workers,
                relax         = relax,
                charge_only   = charge_only,
                criterion     = criterion,
                correlation   = correlation,
                zone_size     = zone_size,
                include_best  = include_best,
                include_worst = include_worst,
                c_base_frac   = c_base_frac,
                prev_nom_sol  = prev_nom_sol,
                log_fh        = _log,
            )

        # ── Forced-rest safety net ────────────────────────────────────────
        # If all actions are infeasible it almost always means the vehicle
        # is in a HoS-infeasible state (sd or cd already exceeds the limit).
        # Silently passing through would produce a completely invalid
        # trajectory.  Instead, force the minimum corrective action:
        # a reduced rest (r2) if the budget allows, else a full rest (r1).
        # This can only happen when the look-ahead horizon was too short to
        # plan a rest preemptively.
        all_penalty = all(s[1] >= INFEASIBLE_PENALTY / 2 for s in score_list)
        if all_penalty and stop > 0:
            K_stop = stop in set(full_data["K"])
            C_stop = stop in set(full_data["C"])
            rst_type = "r2" if state.rho2_used < 3 else "r1"
            forced_action = dict(
                y          = 1 if K_stop else 0,
                break_type = None,
                rest_type  = rst_type,
            )
            if verbose:
                reason = (f"sd={state.sd:.2f}h>{full_data['Tdrv_sh1']}h"
                          if state.sd > full_data["Tdrv_sh1"] + 1e-3
                          else f"cd={state.cd:.2f}h>{full_data['Tdrv_cons']}h"
                          if state.cd > full_data["Tdrv_cons"] + 1e-3
                          else "all actions infeasible")
                print(f"  ⚠ FORCED REST ({rst_type}) at stop {stop}: "
                      f"{reason}")
            # Solve a quick MIP to get actual durations (especially tauc) for
            # the forced action so advance_state uses the correct charge amount.
            end_fr, _ = find_horizon_end_stop(full_data, stop, state.t_arr,
                                              2.0, state=state)
            forced_sol = solve_horizon(
                full_data      = full_data,
                start_stop     = stop,
                end_stop       = end_fr,
                init_state     = state.as_init_state(),
                fixed_action   = forced_action,
                rho2_remaining = 3 - state.rho2_used,
                tee            = False,
                time_limit     = 30,
                relax          = False,   # need integer solution for durations
            )
            action  = forced_action
            nom_sol = forced_sol if forced_sol["feasible"] else None
            score_list = [(forced_action, INFEASIBLE_PENALTY, 0.0, 0, [])]

        actions.append(action)
        scores_log.append(score_list)
        # Carry nominal MIP solution forward as warm-start for next stop
        prev_nom_sol = nom_sol["sol"] if (nom_sol and nom_sol.get("sol")) else None

        if stop < N:
            state, td_stop, D_act = advance_state(
                full_data = full_data,
                state     = state,
                action    = action,
                milp_sol  = nom_sol,
                delta     = delta,
                rng       = master_rng,
            )
            states.append(state)
            td_list.append(td_stop)
            D_actual_list.append(D_act)

            # Collect durations at this stop from nominal MILP2 solution
            if nom_sol is not None and nom_sol.get("sol"):
                s0 = nom_sol["sol"][0]
                dur = dict(taub=s0["taub"], taur=s0["taur"],
                           tauc=s0["tauc"], tauq=s0["tauq"])
            else:
                brk_type = action.get("break_type")
                rst_type = action.get("rest_type")
                dur = dict(
                    taub = (full_data["Tb45"] if brk_type == "b45" else
                            full_data["Tb15"] if brk_type == "b15" else
                            full_data["Tb30"] if brk_type == "b30" else 0.0),
                    taur = (full_data["Tr1"]  if rst_type == "r1" else
                            full_data["Tr2"]  if rst_type == "r2" else 0.0),
                    tauc = 0.0, tauq = 0.0,
                )
            durations_list.append(dur)

        if verbose and stop > 0:
            print(f"     → arrived stop {state.stop} at t={state.t_arr:.3f}h  "
                  f"soc={state.e_arr:.1f}kWh")

    wall_elapsed = time.perf_counter() - wall_start

    _lprint(f"\n{'='*65}")
    _lprint(f"  SIMULATION COMPLETE")
    T_START = full_data.get("T_START", 8.0)
    _arr = states[-1].t_arr
    _lprint(f"  Arrival (absolute) : {_arr:.3f} h  "
            f"({int(_arr):02d}:{int((_arr%1)*60):02d})")
    _lprint(f"  Travel duration    : {_arr - T_START:.3f} h")
    _lprint(f"  Wall-clock time    : {wall_elapsed:.1f} s")
    _lprint(f"{'='*65}\n")

    # ── Oracle: solve full MILP with realised D values ──────────────────
    oracle = oracle_solve(full_data, D_actual_list,
                          sim_results=dict(states=states, actions=actions,
                                           durations_list=durations_list,
                                           total_time=states[-1].t_arr),
                          verbose=verbose,
                          tee=True,
                          log_fh=_log)

    # ── Save JSON solution ────────────────────────────────────────────────
    def _ser(obj):
        if isinstance(obj, (int, float, bool, str, type(None))): return obj
        if isinstance(obj, dict):  return {str(k): _ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_ser(v) for v in obj]
        return str(obj)

    _sol_json = dict(
        run_id         = _rid,
        instance       = _title,
        n_scenarios    = n_scenarios,
        horizon_hours  = horizon_hours,
        delta          = delta,
        criterion      = criterion,
        charge_only    = charge_only,
        seed           = seed,
        departure_h    = full_data.get("T_START", 8.0),
        sim_arrival_h  = states[-1].t_arr,
        wall_clock_s   = wall_elapsed,
        oracle         = _ser(oracle),
        sim_trajectory = [
            dict(stop=st.stop,
                 t_arr=round(st.t_arr, 4),
                 e_arr=round(st.e_arr, 2),
                 cd=round(st.cd, 4),
                 sd=round(st.sd, 4),
                 sw=round(st.sw, 4),
                 phi=st.phi,
                 rho2_used=st.rho2_used)
            for st in states],
        actions = [_ser(a) for a in actions],
    )
    with open(_solpath, "w") as _fj:
        _json.dump(_sol_json, _fj, indent=2)
    _lprint(f"  Solution JSON : {_solpath}")
    _log.close()

    return dict(
        states         = states,
        actions        = actions,
        scores_log     = scores_log,
        td_list        = td_list,
        D_actual_list  = D_actual_list,
        durations_list = durations_list,
        total_time     = states[-1].t_arr,
        wall_clock     = wall_elapsed,
        oracle         = oracle,
        log_path       = _logpath,
        fig_path       = _figpath,
        sol_path       = _solpath,
        run_id         = _rid,
    )



# ══════════════════════════════════════════════════════════════════════════
# ORACLE (HINDSIGHT OPTIMAL)
# ══════════════════════════════════════════════════════════════════════════

def check_simulation_feasibility(results, full_data, tol=1e-3):
    """
    Check that the simulated trajectory satisfies HoS and energy constraints.
    Returns (ok: bool, issues: list[str]).
    """
    states  = results["states"]
    actions = results["actions"]
    durs    = results.get("durations_list", [])
    K_set   = set(full_data["K"])
    C_set   = set(full_data["C"])

    Tdrv_cons = full_data["Tdrv_cons"]
    Tdrv_sh1  = full_data["Tdrv_sh1"]
    Twrk_sh   = full_data["Twrk_sh"]
    Emin      = full_data["Emin"]
    Ecap      = full_data["Ecap"]
    issues    = []

    for i, state in enumerate(states):
        s   = state.stop
        act = actions[i] if i < len(actions) else {}
        dur = durs[i]    if i < len(durs)    else {}
        y   = int(act.get("y", 0))
        brk = act.get("break_type")
        rst = act.get("rest_type")
        is_CS   = s in K_set
        is_cust = s in C_set

        tauq = dur.get("tauq", 0.0)
        tauc = dur.get("tauc", 0.0)
        if is_CS:
            work = tauq * y + (tauc if (y and not brk and not rst) else 0.0)
        elif is_cust:
            work = full_data["S"].get(s, 0.0)
        else:
            work = 0.0
        sw_k = state.sw + work

        if state.cd > Tdrv_cons + tol:
            issues.append(f"stop {s:>2}: cd={state.cd:.3f}h > {Tdrv_cons}h (consec driving)")
        if state.sd > Tdrv_sh1 + tol:
            issues.append(f"stop {s:>2}: sd={state.sd:.3f}h > {Tdrv_sh1}h (shift driving)")
        if sw_k > Twrk_sh + tol:
            issues.append(f"stop {s:>2}: sw={sw_k:.3f}h > {Twrk_sh}h (shift working)")
        if state.e_arr < Emin - tol:
            issues.append(f"stop {s:>2}: soc={state.e_arr:.1f}kWh < Emin={Emin}kWh")

        # Also check energy-out: departure energy minus next leg must stay >= Emin.
        # advance_state clips e_arr_new to Emin, so this violation is invisible
        # from state.e_arr alone.
        if i < len(actions) and s < full_data["N"]:
            E_leg   = full_data["E"].get(s, 0.0)
            e_dep_c = state.e_arr    # no charge if not CS or not charging
            if is_CS and y:
                dur_i  = durs[i] if i < len(durs) else {}
                tauc_i = dur_i.get("tauc", 0.0)
                if tauc_i > 0:
                    e_dep_c = _energy_after_charging(state.e_arr, tauc_i, full_data)
            if e_dep_c - E_leg < Emin - tol:
                issues.append(
                    f"stop {s:>2}: energy violation — "
                    f"ed={e_dep_c:.1f} − E[{s}]={E_leg:.1f} = "
                    f"{e_dep_c-E_leg:.1f} < Emin={Emin}kWh (clipped in sim)")

    return len(issues) == 0, issues

def _warmstart_oracle(model, full_data, sim_results):
    """
    Set variable values on the oracle model using the simulation trajectory
    as a warm-start hint.  HiGHS will use these as an initial incumbent,
    tightening the MIP bound immediately and often cutting search time
    significantly.

    The simulation gives us exact values for:
      - Binary decisions: y[i], x_b45/b15/b30[i], rho1/rho2[i]
      - Times: ta[i], td[i]
      - Energy: ea[i], ed[i]
      - HoS accumulators: cd[i], sd[i], sw[i]
      - Durations: tauc[i], taub[i], taur[i]

    We set these as variable initial values (not constraints), so the solver
    is free to improve upon them.
    """
    import pyomo.environ as pyo

    states    = sim_results["states"]
    actions   = sim_results["actions"]
    durs      = sim_results.get("durations_list", [])
    K_set     = set(full_data["K"])
    C_set     = set(full_data["C"])
    N         = full_data["N"]

    # Build per-stop lookup from simulation
    sim_by_stop = {}
    for idx, state in enumerate(states):
        s   = state.stop
        act = actions[idx] if idx < len(actions) else {}
        dur = durs[idx]    if idx < len(durs)    else {}
        y   = int(act.get("y", 0))
        brk = act.get("break_type")
        rst = act.get("rest_type")
        is_CS = s in K_set

        tauc = dur.get("tauc", 0.0)
        taub = dur.get("taub", 0.0)
        taur = dur.get("taur", 0.0)
        tauq = dur.get("tauq", 0.0)

        # Approximate departure energy from PWL
        if is_CS and y and tauc > 0:
            ed_val = _energy_after_charging(state.e_arr, tauc, full_data)
        else:
            ed_val = state.e_arr

        sim_by_stop[s] = dict(
            ta=state.t_arr, ea=state.e_arr, ed=ed_val,
            cd=state.cd, sd=state.sd, sw=state.sw,
            y=y, b45=int(brk=="b45"), b15=int(brk=="b15"),
            b30=int(brk=="b30"), rho1=int(rst=="r1"), rho2=int(rst=="r2"),
            tauc=tauc, taub=taub, taur=taur,
        )

    # Inject into model variables — Pyomo uses .set_value() for warm-start hints
    for i in model.I:
        if i not in sim_by_stop:
            continue
        s = sim_by_stop[i]
        try:
            model.ta[i].set_value(s["ta"])
            model.ea[i].set_value(max(s["ea"], full_data["Emin"]))
            model.ed[i].set_value(max(s["ed"], full_data["Emin"]))
            model.cd[i].set_value(s["cd"])
            model.sd[i].set_value(s["sd"])
            model.sw[i].set_value(s["sw"])
            model.x_b45[i].set_value(s["b45"])
            model.x_b15[i].set_value(s["b15"])
            model.x_b30[i].set_value(s["b30"])
            model.rho1[i].set_value(s["rho1"])
            model.rho2[i].set_value(s["rho2"])
        except Exception:
            pass  # skip stops that are outside the model's index set

    for i in model.Kset:
        if i not in sim_by_stop:
            continue
        s = sim_by_stop[i]
        try:
            model.y[i].set_value(s["y"])
            model.tauc[i].set_value(s["tauc"])
        except Exception:
            pass

    for i in model.I:
        if i not in sim_by_stop:
            continue
        s = sim_by_stop[i]
        try:
            model.taub[i].set_value(s["taub"])
            model.taur[i].set_value(s["taur"])
        except Exception:
            pass


def oracle_solve(full_data, D_actual_list, sim_results=None,
                 time_limit=6*3600, tee=True, verbose=True, log_fh=None):
    """
    Solve the full deterministic MILP with the travel times that actually
    occurred during the simulation (perfect hindsight).

    Parameters
    ----------
    full_data     : dict from MILP._make_data
    D_actual_list : list of floats, length N
    sim_results   : dict returned by run_simulation — if provided, the
                    simulation trajectory is used as a warm-start incumbent.
                    This lets HiGHS start with a known feasible solution and
                    immediately focus on improving it, which is critical when
                    the instance is too large to prove optimality in the time
                    limit.
    time_limit    : int — solver wall-clock limit in seconds (default 6 h).
    tee           : print solver log
    verbose       : print summary

    Returns
    -------
    dict with keys:
        'feasible' : bool — True if any feasible solution was found
        'optimal'  : bool — True only if proven optimal
        'obj'      : float — best arrival time found, or inf
        'gap'      : float — MIP optimality gap at termination (0 if optimal)
        'sol'      : list of per-stop dicts
        'status'   : str
        'D_actual' : dict
    """
    from MILP import build_model, extract_solution
    import copy

    N = full_data["N"]
    assert len(D_actual_list) == N, (
        f"D_actual_list has {len(D_actual_list)} entries but route has {N} legs")

    D_actual_dict   = {i: D_actual_list[i] for i in range(N)}
    oracle_data     = dict(full_data)
    oracle_data["D"] = D_actual_dict

    from MILP import _time_bounds
    lb_t, ub_t = _time_bounds(
        oracle_data["I"], oracle_data["C"], oracle_data["K"],
        D_actual_dict, oracle_data["S"], oracle_data["Q"],
        oracle_data["Tbar"], oracle_data["T_hor"],
        t0=oracle_data.get("T_START", 8.0))
    oracle_data["lb_t"] = lb_t
    oracle_data["ub_t"] = ub_t

    ws_str = " + sim warm-start" if sim_results is not None else ""
    h = time_limit // 3600; m = (time_limit % 3600) // 60
    tl_str = f"{h}h{m:02d}m" if h else f"{time_limit}s"
    def _op(msg):
        if verbose: print(msg)
        if log_fh: print(msg, file=log_fh)
    _op(f"\n{'='*65}")
    _op(f"  ORACLE SOLVE  (hindsight-optimal{ws_str})")
    _op(f"  time_limit={tl_str}")
    _op(f"{'='*65}")

    model = build_model(oracle_data)

    # Warm-start: inject simulation trajectory as initial incumbent
    if sim_results is not None:
        _warmstart_oracle(model, full_data, sim_results)
        _op(f"  Warm-start: sim arrival {sim_results['total_time']:.3f}h "
               f"injected as incumbent")

    from MILP2 import _solve_quiet
    import pyomo.environ as pyo
    solver = pyo.SolverFactory("appsi_highs")
    solver.options["mip_rel_gap"]  = 0.005
    solver.options["time_limit"]   = time_limit
    solver.options["presolve"]     = "on"
    # Tell HiGHS to use the provided variable values as a starting solution
    solver.options["mip_heuristic_effort"] = 0.2  # more aggressive heuristics early

    if tee:
        import contextlib as _cl, io as _sio
        _buf = _sio.StringIO()
        with _cl.redirect_stdout(_buf):
            try:
                res    = solver.solve(model, tee=True, load_solution=False)
                status = str(res.solver.termination_condition)
                if status not in ("infeasible",):
                    model.solutions.load_from(res)
            except Exception:
                try:
                    res    = solver.solve(model, tee=True)
                    status = str(res.solver.termination_condition)
                except RuntimeError:
                    status = "infeasible"; res = None
        _out = _buf.getvalue()
        if verbose: print(_out)
        if log_fh:
            print("\n[ORACLE SOLVER OUTPUT]", file=log_fh)
            print(_out, file=log_fh)
    else:
        try:
            res    = _solve_quiet(solver, model, False)
            status = str(res.solver.termination_condition)
        except RuntimeError:
            status = "infeasible"; res = None
    _op(f"  Status : {status}")

    # Accept any status that produced a loaded feasible solution:
    # "optimal" (proven), "feasible" (gap not closed), or "maxTimeLimit"
    # (time ran out but HiGHS found and loaded an incumbent).
    # The WARNING "Loading a feasible but suboptimal solution" is normal
    # for maxTimeLimit when an incumbent exists — it is NOT an error.
    has_solution = False
    obj_val = float("inf")
    gap_val = float("inf")

    if status in ("optimal", "feasible", "maxTimeLimit"):
        try:
            obj_val = pyo.value(model.obj)
            has_solution = obj_val is not None and obj_val < 1e8
        except Exception:
            has_solution = False

    if not has_solution:
        if verbose:
            _op("  No feasible solution found within time limit.")
        return dict(feasible=False, optimal=False, obj=float("inf"),
                    gap=float("inf"), sol=[], status=status,
                    D_actual=D_actual_dict)

    is_optimal = (status == "optimal")
    sol = extract_solution(model, oracle_data)

    # Extract MIP gap if available
    try:
        gap_val = res.solver.termination_condition_message
        # Parse gap from HiGHS message if present
        import re
        m = re.search(r"gap[^0-9]*([0-9.e+-]+)%", str(gap_val), re.I)
        gap_val = float(m.group(1)) / 100 if m else (0.0 if is_optimal else float("nan"))
    except Exception:
        gap_val = 0.0 if is_optimal else float("nan")

    if verbose:
        opt_str = " (optimal)" if is_optimal else f" (gap ≈ {gap_val:.1%})" if not np.isnan(gap_val) else ""
        _op(f"  Oracle arrival : {obj_val:.3f} h{opt_str}")

    return dict(feasible=True, optimal=is_optimal, obj=obj_val,
                gap=gap_val, sol=sol, status=status, D_actual=D_actual_dict)

# ══════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════

def print_simulation_log(results, full_data):
    """Pretty-print the simulation action log."""
    N     = full_data["N"]
    C_set = set(full_data["C"])
    K_set = set(full_data["K"])

    hdr = (f"  {'stop':>4}  {'type':>5}  {'t_arr':>7}  {'soc':>6}  "
           f"{'cd':>5}  {'sd':>5}  {'sw':>5}  "
           f"{'y':>2}  {'brk':>4}  {'rst':>4}  action")
    print(f"\n{hdr}\n  {'─'*95}")

    for i, (state, action) in enumerate(
            zip(results["states"], results["actions"])):
        stop = state.stop
        typ  = ("ORIG" if stop == 0 else
                "DEST" if stop == N else
                "CUST" if stop in C_set else "CS")
        brk  = action.get("break_type") or "—"
        rst  = action.get("rest_type")  or "—"
        y    = action.get("y", 0)
        acts = []
        if y:           acts.append("CHARGE")
        if brk != "—":  acts.append(f"BRK-{brk.upper()}")
        if rst != "—":  acts.append(f"REST-{rst.upper()}")

        print(f"  {stop:>4}  {typ:>5}  "
              f"{state.t_arr:>7.3f}  {state.e_arr:>6.1f}  "
              f"{state.cd:>5.2f}  {state.sd:>5.2f}  {state.sw:>5.2f}  "
              f"{y:>2}  {brk:>4}  {rst:>4}  "
              f"{', '.join(acts) or '—'}")



# ══════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_COL = dict(
    drive   = "#2C6FAC",
    service = "#27AE60",
    queue   = "#C0392B",
    charge  = "#E67E22",
    brk     = "#F1C40F",
    rest    = "#8E44AD",
)
_EPS = 1e-3


def _bar(ax, start, dur, y, h, color, label=None, fsize=7, tc="white"):
    if dur < _EPS:
        return
    ax.barh(y, dur, left=start, height=h, color=color,
            edgecolor="white", linewidth=0.3)
    if dur > 0.1 and label:
        ax.text(start + dur / 2, y, label, ha="center", va="center",
                fontsize=fsize, color=tc, fontweight="bold", clip_on=True)


def _shade_bands(ax, t0, t1):
    """Alternate night/day/evening shading over the timeline."""
    bands = [(0, 6, "#D6EAF8"), (6, 20, "#FEF9E7"), (20, 24, "#E8DAEF")]
    t = 0
    while t < t1:
        day = int(t) // 24
        for h0, h1, col in bands:
            s = max(day * 24 + h0, t0)
            e = min(day * 24 + h1, t1)
            if e > s:
                ax.axvspan(s, e, color=col, alpha=0.25, zorder=0, lw=0)
        t += 24


def plot_simulation_results(results, full_data, title="simulation", save=True):
    """
    Three-panel plot of the simulation run:
      1. Gantt — activity timeline (drive, service, queue, charge, break, rest)
      2. SOC   — battery state of charge
      3. HoS   — consecutive-driving, shift-driving, shift-working accumulators

    Parameters
    ----------
    results  : dict returned by run_simulation
    full_data: dict from MILP._make_data
    title    : string used in suptitle and filename
    save     : if True, saves PNG to current directory
    """
    import os, time as _t

    states        = results["states"]
    actions       = results["actions"]
    td_list       = results["td_list"]          # departure from stop i
    D_actual_list = results["D_actual_list"]    # actual leg i travel time
    durations_list= results["durations_list"]   # taub/taur/tauc/tauq at stop i

    N     = full_data["N"]
    C_set = set(full_data["C"])
    K_set = set(full_data["K"])

    # Convenience: index by stop number
    # states[i] = arrival at stop i  (len = N+1)
    # actions[i], td_list[i], durations_list[i] correspond to stop i  (len = N)

    tend = states[-1].t_arr

    oracle      = results.get("oracle", {})
    oracle_sol  = oracle.get("sol", [])
    oracle_obj  = oracle.get("obj", None)
    oracle_feas = oracle.get("feasible", False)

    # x-axis extent: longest of simulation and oracle timelines
    tend_all = tend
    if oracle_feas and oracle_obj:
        tend_all = max(tend, oracle_obj)

    fig, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 2.5, 2, 2, 2.5]})

    gap_str = ""
    if oracle_feas and oracle_obj:
        gap = tend - oracle_obj
        gap_str = f"  |  oracle {oracle_obj:.2f}h  |  gap {gap:+.2f}h ({100*gap/oracle_obj:.1f}%)"
    fig.suptitle(f"Simulation — {title}  (arrival {tend:.2f}h){gap_str}",
                 fontsize=11, fontweight="bold")

    # ── Panel 1: Gantt ────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_title("Simulation — actual realization", fontsize=10)
    _shade_bands(ax1, 0, tend)
    Y, H = 0.5, 0.40

    for i in range(N):
        st   = states[i]
        ta_i = st.t_arr
        act  = actions[i]
        dur  = durations_list[i] if i < len(durations_list) else {}
        td_i = td_list[i]        if i < len(td_list)        else ta_i

        is_K = (i in K_set)
        is_C = (i in C_set)
        y_val = int(act.get("y", 0))

        brk = act.get("break_type")
        rst = act.get("rest_type")

        t = ta_i

        # Service (customers)
        if is_C:
            svc = full_data["S"].get(i, 0.0)
            _bar(ax1, t, svc, Y, H, _COL["service"], f"C{i}", fsize=7); t += svc

        # Queue (CS, charged only)
        if is_K and y_val:
            tauq = dur.get("tauq", 0.0)
            _bar(ax1, t, tauq, Y, H, _COL["queue"], "Q", fsize=7); t += tauq

        # Charge
        if is_K and y_val:
            tauc = dur.get("tauc", 0.0)
            ea_v = st.e_arr
            ed_v = states[i + 1].e_arr + full_data["E"].get(i, 0.0)  # approx
            _bar(ax1, t, tauc, Y, H, _COL["charge"],
                 f"CHG\n{ea_v:.0f}→{ed_v:.0f}", fsize=6); t += tauc

        # Break
        taub = dur.get("taub", 0.0)
        if brk and taub > _EPS:
            lbl = {"b45": "B45", "b15": "B15", "b30": "B30"}.get(brk, brk)
            _bar(ax1, t, taub, Y, H, _COL["brk"], lbl, fsize=7, tc="#333")
            t += taub

        # Rest
        taur = dur.get("taur", 0.0)
        if rst and taur > _EPS:
            lbl = "RST-r1" if rst == "r1" else "RST-r2"
            _bar(ax1, t, taur, Y, H, _COL["rest"], lbl, fsize=7); t += taur

        # Driving to next stop
        if i < N and i < len(D_actual_list):
            D_act = D_actual_list[i]
            _bar(ax1, td_i, D_act, Y, H, _COL["drive"],
                 f"→{i+1}", fsize=6)

        # Stop label
        typ = "●" if is_C else ("▲" if is_K else "O")
        ax1.text(ta_i, Y + H / 2 + 0.06, f"{typ}{i}",
                 ha="left", va="bottom", fontsize=6,
                 color="#444", rotation=45, clip_on=True)

    # Shade look-ahead windows: from decision stop to its mean horizon end
    LA_PENALTY = 1e9 / 2
    for stp, sc_list in enumerate(results["scores_log"]):
        if stp == 0 or not sc_list:
            continue
        best_entry = sc_list[0]          # best = lowest mean_obj, already sorted
        mean_horizon = best_entry[1]
        if mean_horizon < LA_PENALTY:
            t_la_start = states[stp].t_arr
            ax1.axvspan(t_la_start, mean_horizon, alpha=0.055,
                        color="navy", zorder=0, lw=0)

    ax1.set_yticks([])
    ax1.set_xlim(-0.1, tend_all * 1.04)
    patches = [mpatches.Patch(color=v, label=k.replace("_", " ").title())
               for k, v in _COL.items()]
    patches += [mpatches.Patch(color="#D6EAF8", alpha=0.6, label="night 0–6h"),
                mpatches.Patch(color="#FEF9E7", alpha=0.6, label="day 6–20h"),
                mpatches.Patch(color="#E8DAEF", alpha=0.6, label="eve 20–24h")]
    ax1.legend(handles=patches, loc="upper left", fontsize=7, ncol=5)

    # ── Panel 2: Oracle Gantt ────────────────────────────────────────────
    ax_or = axes[1]
    ax_or.set_title("Oracle — hindsight-optimal schedule (same realised travel times)",
                    fontsize=10)
    _shade_bands(ax_or, 0, tend_all)
    Yo, Ho = 0.5, 0.40

    if oracle_feas and oracle_sol:
        orsol = {s["i"]: s for s in oracle_sol}
        for i in range(N):
            s_or = orsol.get(i, {})
            if not s_or:
                continue
            ta_or = s_or.get("ta", 0.0)
            td_or = s_or.get("td", ta_or)
            t     = ta_or
            is_K  = i in K_set
            is_C  = i in C_set
            y_or  = s_or.get("y", 0)

            if is_C:
                svc = full_data["S"].get(i, 0.0)
                _bar(ax_or, t, svc, Yo, Ho, _COL["service"], f"C{i}", fsize=7)
                t += svc
            if is_K and y_or:
                tauq_or = s_or.get("tauq", 0.0)
                _bar(ax_or, t, tauq_or, Yo, Ho, _COL["queue"], "Q", fsize=7)
                t += tauq_or
                tauc_or = s_or.get("tauc", 0.0)
                ea_or = s_or.get("ea", 0.0); ed_or = s_or.get("ed", 0.0)
                _bar(ax_or, t, tauc_or, Yo, Ho, _COL["charge"],
                     f"CHG\n{ea_or:.0f}→{ed_or:.0f}", fsize=6)
                t += tauc_or
            if s_or.get("b45"):
                _bar(ax_or, t, s_or.get("taub", 0), Yo, Ho, _COL["brk"], "B45", fsize=7, tc="#333")
                t += s_or.get("taub", 0)
            elif s_or.get("b15"):
                _bar(ax_or, t, s_or.get("taub", 0), Yo, Ho, _COL["brk"], "B15", fsize=7, tc="#333")
                t += s_or.get("taub", 0)
            elif s_or.get("b30"):
                _bar(ax_or, t, s_or.get("taub", 0), Yo, Ho, _COL["brk"], "B30", fsize=7, tc="#333")
                t += s_or.get("taub", 0)
            if s_or.get("rho1") or s_or.get("rho2"):
                lbl = "RST-r1" if s_or.get("rho1") else "RST-r2"
                _bar(ax_or, t, s_or.get("taur", 0), Yo, Ho, _COL["rest"], lbl, fsize=7)
                t += s_or.get("taur", 0)

            # Driving bar: use actual realised D (same as simulation)
            if i < len(D_actual_list):
                _bar(ax_or, td_or, D_actual_list[i], Yo, Ho, _COL["drive"],
                     f"→{i+1}", fsize=6)

            typ = "●" if is_C else ("▲" if is_K else "O")
            ax_or.text(ta_or, Yo + Ho/2 + 0.06, f"{typ}{i}",
                       ha="left", va="bottom", fontsize=6,
                       color="#444", rotation=45, clip_on=True)

        # Highlight arrival difference
        ax_or.axvline(oracle_obj, color="green",  lw=2, ls="-",  alpha=0.9,
                      label=f"oracle arrival {oracle_obj:.2f}h")
        ax_or.axvline(tend,       color="crimson", lw=2, ls="--", alpha=0.9,
                      label=f"simulation arrival {tend:.2f}h")
    else:
        ax_or.text(0.5, 0.5, "Oracle infeasible or not run",
                   ha="center", va="center", transform=ax_or.transAxes,
                   fontsize=12, color="grey")

    ax_or.set_yticks([])
    ax_or.legend(fontsize=8, loc="upper right")

    # ── Panel 3: SOC ─────────────────────────────────────────────────────
    ax2 = axes[2]
    ax2.set_title("Battery state of charge (at arrival)", fontsize=10)
    _shade_bands(ax2, 0, tend)

    t_pts = [s.t_arr for s in states]
    e_pts = [s.e_arr for s in states]

    # Insert charge-event jumps for smoother SOC curve
    t_full, e_full = [], []
    for i, (t, e) in enumerate(zip(t_pts, e_pts)):
        t_full.append(t); e_full.append(e)
        if i < N and int(actions[i].get("y", 0)):
            dur = durations_list[i] if i < len(durations_list) else {}
            td_i = td_list[i] if i < len(td_list) else t
            tauq = dur.get("tauq", 0.0)
            tauc = dur.get("tauc", 0.0)
            t_cs = t + tauq
            t_ce = t_cs + tauc
            e_dep = e_pts[i + 1] + full_data["E"].get(i, 0.0)  # before driving
            t_full.append(t_cs); e_full.append(e)
            t_full.append(t_ce); e_full.append(e_dep)
            t_full.append(td_i); e_full.append(e_dep)

    ax2.plot(t_full, e_full, color=_COL["drive"], lw=2, label="SOC", zorder=2)
    ax2.fill_between(t_full, e_full, alpha=0.10, color=_COL["drive"])
    ax2.axhline(full_data["Emin"], color="red", ls=":", lw=1.2,
                label=f"E_min={full_data['Emin']} kWh")
    ax2.axhline(full_data["Ecap"], color="gray", ls=":", lw=1.2,
                label=f"E_cap={full_data['Ecap']} kWh")
    ax2.set_ylabel("kWh")
    ax2.set_ylim(0, full_data["Ecap"] * 1.15)
    ax2.legend(fontsize=8, loc="upper right")

    # ── Panel 3: HoS counters ─────────────────────────────────────────────
    ax3 = axes[3]
    ax3.set_title("HoS accumulators at arrival", fontsize=10)
    _shade_bands(ax3, 0, tend)

    cd_vals = [s.cd for s in states]
    sd_vals = [s.sd for s in states]
    sw_vals = [s.sw for s in states]
    ta_vals = [s.t_arr for s in states]

    ax3.plot(ta_vals, cd_vals, "o-", color="#E74C3C", lw=1.5, ms=4,
             label="Consec. driving")
    ax3.plot(ta_vals, sd_vals, "s-", color="#3498DB", lw=1.5, ms=4,
             label="Shift driving")
    ax3.plot(ta_vals, sw_vals, "^-", color="#1ABC9C", lw=1.5, ms=4,
             label="Shift working")

    ax3.axhline(full_data["Tdrv_cons"], color="#E74C3C", ls=":", lw=1.2,
                alpha=0.7, label=f"max consec {full_data['Tdrv_cons']}h")
    ax3.axhline(full_data["Tdrv_sh1"], color="#3498DB", ls=":", lw=1.2,
                alpha=0.7, label=f"max shift drv {full_data['Tdrv_sh1']}h")
    ax3.axhline(full_data["Twrk_sh"], color="#1ABC9C", ls=":", lw=1.2,
                alpha=0.7, label=f"max shift wk {full_data['Twrk_sh']}h")

    # Mark break / rest events
    for i, act in enumerate(actions):
        brk = act.get("break_type")
        rst = act.get("rest_type")
        if brk or rst:
            t_ev = states[i].t_arr
            color = _COL["rest"] if rst else _COL["brk"]
            ax3.axvline(t_ev, color=color, lw=1.2, alpha=0.55, ls="--")

    ax3.set_xlabel("Time (h)")
    ax3.set_ylabel("Hours")
    ax3.legend(fontsize=7, ncol=3, loc="upper left")

    # ── Panel 4: Look-ahead decision quality ────────────────────────────────
    ax4 = axes[4]
    ax4.set_title("Look-ahead: scenario objectives by decision stop  "
                  "(●=chosen ±σ,  ×=2nd-best,  dots=raw scenarios)", fontsize=10)

    PENALTY = INFEASIBLE_PENALTY / 2

    def _action_label(act):
        rst = act.get("rest_type")
        brk = act.get("break_type")
        y   = int(act.get("y", 0))
        if rst:  return f"REST-{rst}"
        if brk:  return f"BRK-{brk}"
        if y:    return "CHARGE"
        return "pass"

    def _action_color(act):
        rst = act.get("rest_type")
        brk = act.get("break_type")
        y   = int(act.get("y", 0))
        if rst == "r1":  return "#8E44AD"
        if rst == "r2":  return "#6C3483"
        if brk == "b45": return "#E67E22"
        if brk == "b30": return "#D68910"
        if brk == "b15": return "#F1C40F"
        if y:            return "#E74C3C"
        return "#2C6FAC"

    rng_jit = np.random.default_rng(0)
    _shade_bands(ax4, 0, tend)

    # Collect lines to connect chosen-action means
    line_x, line_y = [], []

    for stp, sc_list in enumerate(results["scores_log"]):
        if stp == 0 or not sc_list:
            continue

        t_x     = states[stp].t_arr
        best    = sc_list[0]              # (action, mean, std, n_feas, raw_objs)
        b_act, b_mean, b_std, b_feas, b_raw = best
        b_col   = _action_color(b_act)

        # Raw feasible scenario dots for chosen action
        feas_raw = [o for o in b_raw if o < PENALTY]
        if feas_raw:
            n   = len(feas_raw)
            jit = rng_jit.uniform(-0.12, 0.12, n)
            ax4.scatter(t_x + jit, feas_raw,
                        color=b_col, alpha=0.30, s=18, zorder=3, lw=0)

        # Chosen action mean ± std
        if b_mean < PENALTY:
            ax4.errorbar(t_x, b_mean, yerr=b_std,
                         fmt="o", color=b_col, ms=9,
                         elinewidth=1.8, capsize=5, zorder=6,
                         label=_action_label(b_act) if stp == 1 else "")
            line_x.append(t_x)
            line_y.append(b_mean)

        # Second-best reference (×)
        if len(sc_list) > 1:
            _, s_mean, _, _, _ = sc_list[1]
            if s_mean < PENALTY:
                ax4.scatter(t_x, s_mean, color="grey", marker="x",
                            s=55, lw=2, zorder=5, alpha=0.65)

    # Confidence band: fill between min and max scenario objective across stops
    if line_x:
        ax4.plot(line_x, line_y, color="dimgrey", lw=1.2, ls="--",
                 alpha=0.6, zorder=4, label="chosen action mean")

    # Actual final arrival + oracle reference
    ax4.axhline(states[-1].t_arr, color="crimson", ls="-", lw=1.8,
                label=f"simulation arrival {states[-1].t_arr:.2f}h", zorder=7)
    if oracle_feas and oracle_obj:
        ax4.axhline(oracle_obj, color="green", ls="-", lw=1.8,
                    label=f"oracle (hindsight optimal) {oracle_obj:.2f}h", zorder=7)
        ax4.fill_between([0, tend_all], oracle_obj, states[-1].t_arr,
                         alpha=0.08, color="red", label="suboptimality gap")

    ax4.set_ylabel("Horizon arrival time (h)")
    ax4.legend(fontsize=7, loc="upper left", ncol=3)

    plt.tight_layout()

    if oracle_feas and oracle_obj:
        gap = tend - oracle_obj
        feas_ok2, feas_iss2 = check_simulation_feasibility(results, full_data)
        feas_tag  = "✓ feasible" if feas_ok2 else f"✗ INFEASIBLE ({len(feas_iss2)} HoS violations)"
        ora_opt   = oracle.get("optimal", False)
        ora_gap   = oracle.get("gap", float("nan"))
        ora_tag   = "optimal" if ora_opt else f"feasible (gap≈{ora_gap:.1%})" if not np.isnan(ora_gap) else "feasible"
        print(f"\n  ┌────────────────────────────────────────────────────┐")
        print(f"  │  Simulation arrival :   {tend:>8.3f} h                    │")
        print(f"  │  Simulation status  :   {feas_tag:<32}│")
        print(f"  │  Oracle  arrival    :   {oracle_obj:>8.3f} h  [{ora_tag}]  │")
        print(f"  │  Gap (sim − oracle) :   {gap:>+8.3f} h ({100*gap/oracle_obj:.1f}%)              │")
        if not feas_ok2:
            print(f"  │  ⚠  Gap meaningless — trajectory violates HoS.     │")
            print(f"  │     Increase horizon (H ≥ 6h recommended).          │")
        print(f"  └────────────────────────────────────────────────────┘")

    if save:
        import os as _oss
        _oss.makedirs("figures", exist_ok=True)
        if isinstance(results, dict) and results.get("fig_path"):
            fname = results["fig_path"]
        else:
            fname = _oss.path.join("figures", f"simulation_{title}_{int(_t.time())}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {fname}")

    plt.show()
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import random as _rnd

    from instances import ALL_INSTANCES as INSTANCES

    _rnd.seed(5)
    name           = sys.argv[1] if len(sys.argv) > 1 else "break_forced"
    n_scenarios    = int(sys.argv[2])   if len(sys.argv) > 2 else 5
    horizon_hours  = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    delta          = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
    n_workers      = int(sys.argv[5])   if len(sys.argv) > 5 else None
    relax          = (sys.argv[6].lower() not in ("0","false","mip"))                      if len(sys.argv) > 6 else True
    criterion      = sys.argv[7]        if len(sys.argv) > 7 else "mean"
    charge_only    = (sys.argv[8].lower() in ("1","true","co"))                      if len(sys.argv) > 8 else False
    correlation    = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    # Usage: python simulation.py <inst> <N_scen> <H> <δ> [workers] [relax] [criterion] [charge_only] [correlation]
    # criterion: mean (default) | worst | best
    # charge_only: 0 (default) | 1
    # correlation: 0.0 (default, independent) .. 1.0 (fully correlated per zone)

    if name not in INSTANCES:
        print(f"Unknown instance '{name}'. Choose: {list(INSTANCES)}")
        sys.exit(1)

    data    = INSTANCES[name]()
    results = run_simulation(
        data,
        n_scenarios   = n_scenarios,
        horizon_hours = horizon_hours,
        delta         = delta,
        seed          = 42,
        time_limit    = 30,
        verbose       = True,
        n_workers     = n_workers,
        relax         = relax,
        criterion     = criterion,
        charge_only   = charge_only,
        correlation   = correlation,
    )

    print_simulation_log(results, data)
    print(f"\n  Total simulated duration : {results['total_time']:.3f} h")
    print(f"  Computation time         : {results['wall_clock']:.1f} s")

    plot_simulation_results(results, data,
                            title=f"{name}_n{n_scenarios}_H{horizon_hours:.0f}_d{delta:.0f}",
                            save=True)
    print(f"  Log      : {results.get('log_path','')}")
    print(f"  Solution : {results.get('sol_path','')}")
    print(f"  Figure   : {results.get('fig_path','')}")