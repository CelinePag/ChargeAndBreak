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
     action, scenario, rho2_rem, time_limit, solve_mode)

    solve_mode : "lp" | "mip"  — never "both" (that is handled upstream).
    scenario is a dict with keys 'D' and 'E' (from generate_scenarios).
    """
    full_data, start_stop, end_stop, init_state, action, scenario, rho2_rem, time_limit, solve_mode = args
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
        relax          = (solve_mode != "mip"),
        warm_start     = scenario.get("warm_start"),
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

def find_horizon_end_stop(full_data, start_stop, horizon_hours, state=None):
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
    """Energy Consumption Rate (kWh/km) at speed v (km/h).
    Clamped to [5, 120] km/h — extrapolation beyond 120 km/h is physically
    unrealistic for a BET truck and can give spuriously large energies in
    extreme fast scenarios.
    """
    v = max(5.0, min(float(v_kmh), 120.0))
    return _ECR_A / v + _ECR_B + _ECR_C * v**2


def generate_scenarios(full_data, start_stop, end_stop,
                       n_scenarios, delta=0.20, seed=None,
                       correlation=0.0, zone_size=8,
                       include_best=False, include_worst=False):
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
    # Log-normal noise: D_scen = D_nom * exp(lm) where lm ~ N(0, sigma²).
    # We want exp(±3σ) ≈ 1±δ in a symmetric sense in log-space.
    # Setting σ = ln(1+δ)/3 gives exp(+3σ) = 1+δ exactly but
    # exp(-3σ) = 1/(1+δ) ≠ 1-δ (asymmetric tails).
    #
    # Symmetric choice: set σ so that ±3σ corresponds to ±δ in log-scale,
    # i.e. σ = δ/3.  Then the distribution has median 1 (unbiased) and
    # the 3σ tail sits at exp(±δ) ≈ 1±δ for small δ.
    # For δ=0.20: exp(0.20)=1.221 vs 1+δ=1.200 — close enough.
    sigma = delta / 3.0

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
                    n_workers=1, solve_mode="lp", criterion="mean"):
    """
    Solve MILP2 for each scenario with `action` fixed at `start_stop`.
    Returns scoring statistics only — does NOT solve a nominal MIP.

    The single nominal MIP re-solve (for extracting tauc/taub/taur to give
    to advance_state) is done ONCE in select_best_action after the winner
    is chosen, using nominal travel times. Doing it here for every action
    wasted (n_actions - 1) MIP solves per stop.

    Parameters
    ----------
    solve_mode : "lp" | "mip"
        "lp"  — LP relaxation (fast, used for scoring in look-ahead).
        "mip" — full integer MIP (slower, more accurate objective values).
        "both" is handled upstream; this function only ever sees "lp" or "mip".
    criterion  : "mean" | "worst" | "best"

    Returns
    -------
    score      : float  — criterion value across scenarios
    std_obj    : float  — std dev of feasible scenario objectives
    n_feasible : int
    first_feas : dict or None — first feasible solve result (for feasibility
                 signalling only — tauc/taub/taur here are NOT used for
                 execution; the nominal MIP re-solve in select_best_action
                 does that for the winner)
    objs       : list[float] — raw per-scenario objectives
    """
    rho2_rem = 3 - state.rho2_used
    init_st  = state.as_init_state()

    def _run_batch(mode):
        """Run all scenario sub-problems for this action under `mode`."""
        arg_list = [
            (full_data, start_stop, end_stop, init_st,
             action, scenario, rho2_rem, time_limit, mode)
            for scenario in scenarios
        ]
        if n_workers > 1:
            results_ordered = [None] * len(arg_list)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_solve_one_scenario, a): idx
                           for idx, a in enumerate(arg_list)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        results_ordered[idx] = fut.result()
                    except Exception:
                        results_ordered[idx] = {"feasible": False,
                                                "obj": INFEASIBLE_PENALTY}
            return results_ordered
        else:
            return [_solve_one_scenario(a) for a in arg_list]

    res_list   = _run_batch(solve_mode)
    objs       = [r["obj"] for r in res_list]
    first_feas = next((r for r in res_list if r.get("feasible")), None)

    n_feasible = sum(1 for o in objs if o < INFEASIBLE_PENALTY / 2)
    if n_feasible == 0:
        return INFEASIBLE_PENALTY, 0.0, 0, None, objs

    feasible_objs = [o for o in objs if o < INFEASIBLE_PENALTY / 2]
    std_obj = float(np.std(feasible_objs)) if n_feasible > 0 else 0.0

    if criterion == "worst":
        score = float(max(objs))
    elif criterion == "best":
        score = float(min(feasible_objs))
    else:
        score = float(np.mean(objs))

    return score, std_obj, n_feasible, first_feas, objs


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
        return actions, 0   # last stop — no pruning needed

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

        # Hard structural cuts (always correct, zero solver cost)
        # b15 when phi=1: phi tracker forces phi[i+1] >= 2 → always infeasible
        if brk == "b15" and state.phi == 1:
            skip = True
        # Battery essentially full: charging adds time with negligible gain.
        # Exception: if a break/rest is being taken anyway, keep y=1 (free).
        if (y == 1 and stop_global in K_set
                and state.e_arr > full_data["Ecap"] - 1.0
                and brk is None and rst is None):
            skip = True

        if not skip:
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
        return actions, 0   # all actions are needed; pruning was over-aggressive
    return pruned, n_pruned


def select_best_action(full_data, stop_global, state: VehicleState,
                       n_scenarios=10, horizon_hours=12, delta=0.20,
                       scenario_seed=None, time_limit=20, tee=False,
                       verbose=True, n_workers=1, solve_mode="lp",
                       charge_only=False, criterion="mean",
                       correlation=0.0, zone_size=8,
                       include_best=False, include_worst=False,
                       prev_nom_sol=None,
                       log_fh=None):
    """
    Evaluate all feasible actions at `stop_global` using the look-ahead.

    Parameters
    ----------
    solve_mode    : "lp" | "mip" | "both"
        "lp"   — LP relaxation for all scenario sub-problems (fast).
        "mip"  — full MIP for all scenario sub-problems (accurate, slow).
        "both" — run LP pass first, then MIP pass; log whether the two
                 methods agree on the chosen action; execute MIP decision.
    charge_only   : only fix y; break/rest free for MILP2.
    criterion     : "mean" | "worst" | "best".
    prev_nom_sol  : nominal MIP sol from previous stop (warm-start seed).
    log_fh        : open file handle for log output.

    Returns
    -------
    best_action  : dict
    scores       : list of (action, score, std, n_feas, raw_objs) — for plotting
    nominal_sol  : dict — MIP solution for the chosen action (for advance_state)
    """
    # ── Stop-level wall-clock timer ───────────────────────────────────────
    t_stop_start = time.perf_counter()

    def _p(msg):
        if verbose: print(msg)
        if log_fh:
            try: print(msg, file=log_fh)
            except Exception: pass

    # ── Enumerate + prune ─────────────────────────────────────────────────
    raw_actions = enumerate_actions(stop_global, state, full_data,
                                    charge_only=charge_only)
    prune_result = _prune_actions(raw_actions, stop_global, state,
                                  full_data, delta)
    if isinstance(prune_result, tuple):
        actions, n_pruned = prune_result
    else:
        actions, n_pruned = prune_result, 0

    end_stop, n_rests = find_horizon_end_stop(full_data, stop_global,
                                              horizon_hours, state=state)

    stop_type  = ("CS"   if stop_global in set(full_data["K"])
                  else "CUST" if stop_global in set(full_data["C"])
                  else "ORIG")
    worker_str = f"  {n_workers}w" if n_workers > 1 else ""
    rest_str   = f" +{n_rests}rest" if n_rests else ""
    mode_str   = f"[{criterion},{solve_mode}"
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
        include_best=include_best, include_worst=include_worst)

    # ── Warm-start A: tail of previous nominal solution ───────────────────
    tail_warm = None
    if prev_nom_sol and len(prev_nom_sol) > 1:
        tail_warm = []
        for s in prev_nom_sol[1:]:
            s2 = dict(s); s2["i"] = s["i"] - 1
            if s2["i"] >= 0:
                tail_warm.append(s2)

    # ── Warm-start B: FREE solve on nominal travel times ─────────────────
    # Relax matches the scenario batch mode:
    #   LP   → FREE as LP  (fast; fractional solution is fine for LP scenarios)
    #   MIP  → FREE as MIP (integer decisions; HiGHS can use it directly as a
    #          feasible incumbent for each scenario MIP without repair,
    #          typically cutting 30-50% off each sub-problem's B&B)
    #   both → FREE as MIP (to warm-start the MIP pass)
    free_relax = (solve_mode == "lp")

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
            relax          = free_relax,
            warm_start     = tail_warm,
        )
        _t_free = time.perf_counter() - _t0_free
        si = _free.get("solve_info", {})
        _ws_flag = "ws=yes" if si.get("had_warm") else "ws=no"
        mode_tag = "LP" if free_relax else "MIP"
        if _free["feasible"]:
            free_sol = _free["sol"]
            _p(f"     [FREE-{mode_tag}] obj={_free['obj']:.3f}h"
               f"  {_free['status']}"
               f"  {_t_free:.1f}s  {_ws_flag}"
               f"  {si.get('n_vars','?')}v/{si.get('n_cons','?')}c")
        else:
            _p(f"     [FREE-{mode_tag}] infeasible  {_t_free:.1f}s  {_ws_flag}")

    if free_sol is not None:
        for scen in scenarios:
            scen["warm_start"] = free_sol

    # ── Score all actions ─────────────────────────────────────────────────
    # scores_detail: (action, score, std, n_feas, raw_objs, first_sol)
    def _score_all_actions(mode, header=""):
        """Run evaluate_action for every action under `mode`; return detail list."""
        detail = []
        for action in actions:
            t0 = time.perf_counter()
            score, std_obj, n_feas, first_feas, raw_objs = evaluate_action(
                full_data, stop_global, end_stop, state,
                action, scenarios, time_limit=time_limit, tee=tee,
                n_workers=n_workers, solve_mode=mode, criterion=criterion)
            elapsed = time.perf_counter() - t0

            detail.append((action, score, std_obj, n_feas, first_feas, raw_objs))

            brk = action.get("break_type") or "-"
            rst = action.get("rest_type")  or "-"
            y   = action.get("y", "-")
            _p(f"  {header}  y={y}  brk={brk:3}  rst={rst:2}"
               f"  {criterion}={score:.3f}h  std={std_obj:.3f}h"
               f"  ok={n_feas}/{len(scenarios)}"
               f"  ({elapsed:.1f}s)")
        return detail

    if solve_mode == "both":
        _p(f"     --- LP pass ---")
        lp_detail = _score_all_actions("lp", header="[LP] ")
        _p(f"     --- MIP pass ---")
        mip_detail = _score_all_actions("mip", header="[MIP]")
        # Decision is based on MIP scores
        scored_detail = mip_detail
    else:
        scored_detail = _score_all_actions(solve_mode)

    # ── Tie-breaking: prefer charging when it costs negligibly more ───────
    #
    # INTENT: if the optimal action already involves a stop (break or rest),
    # and the y=1 version of that SAME stop type costs less than TIEBREAK_ABS
    # extra, prefer charging — the vehicle is docked anyway.
    #
    # WRONG approaches avoided here:
    #   • Relative threshold (e.g. 1% of 60h = 36 min): too loose on long trips.
    #   • Cross-type comparison (y=0-pass vs y=1-pass): the vehicle is NOT
    #     stopping anyway; charging adds pure dwell time with no other benefit.
    #
    # We therefore:
    #   1. Use an ABSOLUTE threshold (5 min = 0.083h).
    #   2. Only compare y=0 vs y=1 within the SAME (break_type, rest_type) pair.
    TIEBREAK_ABS = 5.0 / 60.0   # 5 minutes in hours

    best_raw_score = min(s[1] for s in scored_detail)
    winner         = min(scored_detail, key=lambda s: s[1])
    tiebreak_applied = False

    # Check if a y=1 version of the winner's (break, rest) pair costs
    # less than TIEBREAK_ABS more than the winner's score.
    w_brk = winner[0].get("break_type")
    w_rst = winner[0].get("rest_type")
    w_y   = winner[0].get("y", 0)

    if w_y == 0:
        # Look for the y=1 version of the same stop type
        matching_charge = [
            s for s in scored_detail
            if s[0].get("y", 0) == 1
            and s[0].get("break_type") == w_brk
            and s[0].get("rest_type")  == w_rst
            and s[1] < INFEASIBLE_PENALTY / 2
        ]
        if matching_charge:
            best_charge = min(matching_charge, key=lambda s: s[1])
            if best_charge[1] <= best_raw_score + TIEBREAK_ABS:
                winner           = best_charge
                tiebreak_applied = True

    best_action  = winner[0]
    best_score   = winner[1]

    # ── Single nominal MIP re-solve for the CHOSEN action only ───────────
    # The scenario batch above (LP or MIP) was used for decision-making.
    # advance_state needs actual integer-valued tauc/taub/taur.  We get
    # those by solving one MIP with:
    #   • the chosen action fixed at local stop 0
    #   • NOMINAL travel times (full_data["D"], no D_override) — they
    #     represent the expected case, are deterministic and reproducible,
    #     and are independent of whichever scenario happened to be first
    #     feasible in the batch.
    # In MIP mode the batch already produced integer solutions, so we
    # reuse the first feasible result from the winner's batch directly.
    _t0_nom = time.perf_counter()
    if solve_mode in ("lp",):
        nominal_sol = solve_horizon(
            full_data      = full_data,
            start_stop     = stop_global,
            end_stop       = end_stop,
            init_state     = state.as_init_state(),
            fixed_action   = best_action,
            D_override     = None,          # nominal travel times
            E_override     = None,
            rho2_remaining = 3 - state.rho2_used,
            tee            = False,
            time_limit     = time_limit * 4,
            relax          = False,
            warm_start     = free_sol,      # best available warm-start
        )
    else:
        # MIP / both: winner[4] is first_feas from the MIP batch
        # (already integer-valued tauc/taub/taur)
        nominal_sol = winner[4] if winner[4] else None
    _t_nom = time.perf_counter() - _t0_nom
    nom_feas = nominal_sol is not None and nominal_sol.get("feasible", False)
    _p(f"     [NOM-MIP] y={best_action.get('y',0)}"
       f"  brk={best_action.get('break_type') or '-'}"
       f"  rst={best_action.get('rest_type') or '-'}"
       f"  nominal-D  {'ok' if nom_feas else 'INFEASIBLE'}"
       f"  {_t_nom:.1f}s")

    # ── "both" comparison log ─────────────────────────────────────────────
    if solve_mode == "both":
        lp_winner_idx  = min(range(len(lp_detail)),  key=lambda i: lp_detail[i][1])
        mip_winner_idx = min(range(len(mip_detail)), key=lambda i: mip_detail[i][1])
        lp_ba  = lp_detail[lp_winner_idx][0]
        mip_ba = mip_detail[mip_winner_idx][0]

        def _act_str(a):
            return (f"y={a.get('y',0)}"
                    f"  brk={a.get('break_type') or '-'}"
                    f"  rst={a.get('rest_type')  or '-'}")

        if (lp_ba.get("y")          == mip_ba.get("y") and
            lp_ba.get("break_type") == mip_ba.get("break_type") and
            lp_ba.get("rest_type")  == mip_ba.get("rest_type")):
            _p(f"     [CMP] LP and MIP agree ✓  →  {_act_str(mip_ba)}")
        else:
            _p(f"     [CMP] LP chose  {_act_str(lp_ba)}")
            _p(f"     [CMP] MIP chose {_act_str(mip_ba)}  ← executed")
            _p(f"     [CMP] ✗ DIFFER")

    # ── Final selection log ───────────────────────────────────────────────
    ba = best_action
    t_stop_total = time.perf_counter() - t_stop_start
    tb_str = "  [charge-tiebreak]" if tiebreak_applied else ""
    _p(f"  -> CHOSEN y={ba.get('y','-')}"
       f"  brk={ba.get('break_type') or '-'}"
       f"  rst={ba.get('rest_type') or '-'}"
       f"  ({criterion}={best_score:.3f}h)"
       f"{tb_str}"
       f"  stop_wall={t_stop_total:.1f}s")

    # Trim to 5-tuple for scores_log (plot-compatible format):
    # (action, score, std, n_feas, raw_objs)
    scores = [(s[0], s[1], s[2], s[3], s[5]) for s in scored_detail]
    scores.sort(key=lambda x: x[1])
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
    # Manoeuver time: 5 min applies whenever z_man = 1 in the MILP.
    # z_man[i] >= y[i] at CS stops AND z_man[i] >= xsum (breaks/rests).
    # So man_time applies when: (charging at CS) OR (any break/rest anywhere).
    man_time = full_data["M"].get(stop, 5.0 / 60) if ((is_CS and y) or brk or rst) else 0.0

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
    E_leg         = full_data["E"].get(stop, 0.0)
    e_arr_new_raw = e_dep_new - E_leg
    if e_arr_new_raw < full_data["Emin"] - 1e-3:
        warnings.warn(
            f"[advance_state] Energy violation leg {stop}→{stop+1}: "
            f"ed={e_dep_new:.2f} − E={E_leg:.2f} = {e_arr_new_raw:.2f} kWh "
            f"< Emin={full_data['Emin']:.2f} kWh. "
            f"Clipping to Emin — check scenario feasibility.",
            stacklevel=2,
        )
    e_arr_new = max(e_arr_new_raw, full_data["Emin"])

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
                   solve_mode    = "lp",
                   charge_only   = False,
                   criterion     = "mean",
                   correlation   = 0.0,
                   zone_size     = 8,
                   include_best  = False,
                   include_worst = False,
                   run_id        = None):
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
    n_workers     : parallel processes. None → auto (min(cpu_count, n_scenarios)).
    solve_mode    : "lp" | "mip" | "both"
        "lp"   — LP relaxation for scenario scoring (fast, default).
        "mip"  — full MIP for scenario scoring (slower, more accurate).
        "both" — run both passes per stop; log LP vs MIP agreement; execute MIP.
    charge_only   : only fix charge decision; breaks/rests free for MILP2.
    criterion     : "mean" | "worst" | "best" — action selection criterion.
    correlation   : spatial correlation between nearby legs [0,1].
    zone_size     : legs per correlation zone.
    include_best  : add best-case scenario to each stop's scenario set.
    include_worst : add worst-case scenario to each stop's scenario set.
    run_id        : str or None — base name for output files (auto-generated when None).

    Returns
    -------
    results : dict with keys
        'states'       : list of VehicleState (one per stop, in order)
        'actions'      : list of action dicts chosen at each stop
        'scores_log'   : list of score lists (one per stop)
        'total_time'   : float — simulated arrival time at destination (h)
        'wall_clock'   : float — real computation time (s)
    """
    if n_workers is None:
        n_workers = min(_os.cpu_count() or 1, n_scenarios)
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

    relax_str = {"lp": "LP-relax", "mip": "MIP", "both": "LP+MIP"}.get(solve_mode, solve_mode)
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
                solve_mode    = solve_mode,
                charge_only   = charge_only,
                criterion     = criterion,
                correlation   = correlation,
                zone_size     = zone_size,
                include_best  = include_best,
                include_worst = include_worst,
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
            end_fr, _ = find_horizon_end_stop(full_data, stop, 2.0, state=state)
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
                                           td_list=td_list,
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
    Inject the simulation trajectory as a complete MIP warm-start incumbent.

    HiGHS requires EVERY variable to have a consistent value before it
    accepts a user solution (Src "X" in the B&B log).  A partial
    assignment is silently discarded — HiGHS tests the solution against all
    constraints and rejects it if any are violated.

    Variables initialised
    ---------------------
    Continuous : ta, td, ea, ed, tauc, taub, taur, taub_hat, u, z_man,
                 cd, sd, sw, l1, l2, l4, lam_a, lam_d
    Binary     : y, x_b45/b15/b30, rho1/rho2, mu_a/mu_d, phi
    """
    import warnings as _ws

    states    = sim_results["states"]
    actions   = sim_results["actions"]
    durs      = sim_results.get("durations_list", [])
    td_list   = sim_results.get("td_list", [])
    K_set     = set(full_data["K"])
    C_set     = set(full_data["C"])
    Ebar      = full_data["Ebar"]
    Tbar      = full_data["Tbar"]
    R         = full_data["R"]
    Emin      = full_data["Emin"]

    def _pwl_weights(e_kWh):
        """Convex PWL weights for energy value e_kWh on the charging curve."""
        e = max(Ebar[R[0]], min(float(e_kWh), Ebar[R[-1]]))
        lam = {r: 0.0 for r in R}
        for j in range(len(R) - 1):
            r_lo, r_hi = R[j], R[j + 1]
            e_lo, e_hi = Ebar[r_lo], Ebar[r_hi]
            if e_lo <= e <= e_hi + 1e-9:
                span    = max(e_hi - e_lo, 1e-9)
                lam[r_hi] = (e - e_lo) / span
                lam[r_lo] = 1.0 - lam[r_hi]
                return lam, r_hi   # mu[r_hi] = 1 activates this segment
        lam[R[-1]] = 1.0
        return lam, R[-1]

    # Build per-stop lookup
    sim_by_stop = {}
    phi_track   = 0   # phi[i] tracks split-break state through sequence

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

        b45  = int(brk == "b45");  b15 = int(brk == "b15");  b30 = int(brk == "b30")
        rho1 = int(rst == "r1");   rho2 = int(rst == "r2")
        xsum = b45 + b15 + b30 + rho1 + rho2
        ri   = b45 + b30 + rho1 + rho2   # consecutive-driving reset
        rho  = rho1 + rho2                # shift reset

        z_man_val  = float(bool(y or xsum))
        taub_hat_v = taub + tauc if is_CS else taub
        u_val      = tauc if (is_CS and y and not xsum) else 0.0

        # phi at this stop = phi carried from previous stop
        phi_now = phi_track
        if ri or b45:   phi_track = 0
        elif b15:       phi_track = 1
        # else unchanged

        # departure energy
        if is_CS and y and tauc > 0:
            ed_val = _energy_after_charging(state.e_arr, tauc, full_data)
        else:
            ed_val = state.e_arr

        # departure time
        if idx < len(td_list):
            td_val = float(td_list[idx])
        elif s == 0:
            td_val = state.t_arr
        elif is_CS:
            td_val = (state.t_arr + tauq * y + tauc + taub + taur
                      + full_data["M"].get(s, 0.0) * z_man_val)
        elif s in C_set:
            td_val = (state.t_arr + full_data["S"].get(s, 0.0) + taub + taur
                      + full_data["M"].get(s, 0.0) * z_man_val)
        else:
            td_val = state.t_arr

        lam_a_vals, mu_a_seg = _pwl_weights(state.e_arr)
        lam_d_vals, mu_d_seg = _pwl_weights(ed_val)

        l1_val = float(state.cd) if ri  else 0.0
        l2_val = float(state.sd) if rho else 0.0
        l4_val = float(state.sw) if rho else 0.0

        sim_by_stop[s] = dict(
            ta=state.t_arr, td=td_val,
            ea=state.e_arr, ed=ed_val,
            cd=state.cd, sd=state.sd, sw=state.sw, phi=phi_now,
            y=y, b45=b45, b15=b15, b30=b30, rho1=rho1, rho2=rho2,
            tauc=tauc, taub=taub, taur=taur, taub_hat=taub_hat_v,
            u=u_val, z_man=z_man_val, l1=l1_val, l2=l2_val, l4=l4_val,
            lam_a=lam_a_vals, mu_a_seg=mu_a_seg,
            lam_d=lam_d_vals, mu_d_seg=mu_d_seg,
        )

    # ── Inject ───────────────────────────────────────────────────────────
    with _ws.catch_warnings():
        _ws.simplefilter("ignore")   # suppress Pyomo W1001/W1002

        for i in model.I:
            sv = sim_by_stop.get(i)
            if sv is None:
                continue
            try:
                model.ta[i].set_value(sv["ta"])
                model.td[i].set_value(sv["td"])
                model.ea[i].set_value(max(sv["ea"], Emin))
                model.ed[i].set_value(max(sv["ed"], Emin))
                model.cd[i].set_value(sv["cd"])
                model.sd[i].set_value(sv["sd"])
                model.sw[i].set_value(sv["sw"])
                model.phi[i].set_value(sv["phi"])
                model.taub[i].set_value(sv["taub"])
                model.taur[i].set_value(sv["taur"])
                model.taub_hat[i].set_value(sv["taub_hat"])
                model.x_b45[i].set_value(sv["b45"])
                model.x_b15[i].set_value(sv["b15"])
                model.x_b30[i].set_value(sv["b30"])
                model.rho1[i].set_value(sv["rho1"])
                model.rho2[i].set_value(sv["rho2"])
                model.z_man[i].set_value(sv["z_man"])
                model.l1[i].set_value(sv["l1"])
                model.l2[i].set_value(sv["l2"])
                model.l4[i].set_value(sv["l4"])
            except Exception:
                pass

        for i in model.Kset:
            sv = sim_by_stop.get(i)
            if sv is None:
                continue
            try:
                model.y[i].set_value(sv["y"])
                model.tauc[i].set_value(sv["tauc"])
                model.u[i].set_value(sv["u"])
                for r in model.Rset:
                    model.lam_a[i, r].set_value(sv["lam_a"].get(r, 0.0))
                    model.lam_d[i, r].set_value(sv["lam_d"].get(r, 0.0))
                mu_a_seg = sv["mu_a_seg"]
                mu_d_seg = sv["mu_d_seg"]
                for r in model.RsegS:
                    model.mu_a[i, r].set_value(1 if r == mu_a_seg else 0)
                    model.mu_d[i, r].set_value(1 if r == mu_d_seg else 0)
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
                res    = solver.solve(model, tee=True, warmstart=True,
                                      load_solution=False)
                status = str(res.solver.termination_condition)
                if status not in ("infeasible",):
                    model.solutions.load_from(res)
            except Exception:
                try:
                    res    = solver.solve(model, tee=True, warmstart=True)
                    status = str(res.solver.termination_condition)
                except RuntimeError:
                    status = "infeasible"; res = None
        _out = _buf.getvalue()
        if verbose: print(_out)
        if log_fh:
            print("\n[ORACLE SOLVER OUTPUT]", file=log_fh)
            print(_out, file=log_fh)
    else:
        import io as _sio2, contextlib as _cl2
        _sink = _sio2.StringIO()
        try:
            with _cl2.redirect_stdout(_sink), _cl2.redirect_stderr(_sink):
                res = solver.solve(model, tee=False, warmstart=True)
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
    print(f"\n  === SIMULATION TRAJECTORY ===")
    print(f"{hdr}\n  {'─'*95}")

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


def print_oracle_log(oracle, full_data):
    """Pretty-print the oracle (hindsight-optimal) solution schedule."""
    if not oracle.get("feasible") or not oracle.get("sol"):
        print("  Oracle: no feasible solution available.")
        return

    N     = full_data["N"]
    C_set = set(full_data["C"])
    K_set = set(full_data["K"])
    sol   = oracle["sol"]

    opt_str = " (optimal)" if oracle.get("optimal") else (
              f" (gap ≈ {oracle['gap']:.1%})"
              if not (oracle.get("gap") != oracle.get("gap"))   # not NaN
              else " (feasible)")

    hdr = (f"  {'stop':>4}  {'type':>5}  {'t_arr':>7}  {'soc':>6}  "
           f"{'cd':>5}  {'sd':>5}  {'sw':>5}  "
           f"{'y':>2}  {'brk':>4}  {'rst':>4}  action")
    print(f"\n  === ORACLE SCHEDULE  (arrival {oracle['obj']:.3f}h{opt_str}) ===")
    print(f"{hdr}\n  {'─'*95}")

    for s in sol:
        stop = s["i"]
        typ  = ("ORIG" if stop == 0 else
                "DEST" if stop == N else
                "CUST" if stop in C_set else "CS")
        brk  = ("b45" if s["b45"] else
                "b15" if s["b15"] else
                "b30" if s["b30"] else "—")
        rst  = ("r1" if s["rho1"] else
                "r2" if s["rho2"] else "—")
        y    = s.get("y", 0)
        acts = []
        if y:           acts.append("CHARGE")
        if brk != "—":  acts.append(f"BRK-{brk.upper()}")
        if rst != "—":  acts.append(f"REST-{rst.upper()}")

        print(f"  {stop:>4}  {typ:>5}  "
              f"{s['ta']:>7.3f}  {s['ea']:>6.1f}  "
              f"{s['cd']:>5.2f}  {s['sd']:>5.2f}  {s['sw']:>5.2f}  "
              f"{y:>2}  {brk:>4}  {rst:>4}  "
              f"{', '.join(acts) or '—'}")



# ══════════════════════════════════════════════════════════════════════════
# VISUALISATION  →  see plots.py
# ══════════════════════════════════════════════════════════════════════════
from plots import plot_simulation_results    # noqa: F401 — re-exported for callers

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
    # solve_mode: 0/lp → "lp" (default),  1/mip → "mip",  2/both → "both"
    _sm_raw        = sys.argv[6].lower() if len(sys.argv) > 6 else "0"
    solve_mode     = {"0": "lp",  "lp":  "lp",
                      "1": "mip", "mip": "mip",
                      "2": "both","both":"both"}.get(_sm_raw, "lp")
    criterion      = sys.argv[7]        if len(sys.argv) > 7 else "mean"
    charge_only    = (sys.argv[8].lower() in ("1","true","co")) \
                                         if len(sys.argv) > 8 else False
    correlation    = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    # Usage: python simulation.py <inst> <N_scen> <H> <δ> [workers] [solve_mode] [criterion] [charge_only] [correlation]
    # solve_mode: 0/lp (default) | 1/mip | 2/both
    # criterion:  mean (default) | worst | best

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
        time_limit    = 300,
        verbose       = True,
        n_workers     = n_workers,
        solve_mode    = solve_mode,
        criterion     = criterion,
        charge_only   = charge_only,
        correlation   = correlation,
    )

    print_simulation_log(results, data)
    print_oracle_log(results.get("oracle", {}), data)
    print(f"\n  Total simulated duration : {results['total_time']:.3f} h")
    print(f"  Computation time         : {results['wall_clock']:.1f} s")

    plot_simulation_results(results, data,
                            title=f"{name}_n{n_scenarios}_H{horizon_hours:.0f}_d{delta:.0f}",
                            save=True)
    print(f"  Log      : {results.get('log_path','')}")
    print(f"  Solution : {results.get('sol_path','')}")
    print(f"  Figure   : {results.get('fig_path','')}")