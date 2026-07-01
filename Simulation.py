"""
Simulation.py — Rolling-horizon look-ahead simulation
======================================================
Implements the main simulation loop and all supporting look-ahead machinery.

Module responsibilities
-----------------------
  enumerate_actions(stop, state, full_data, charge_only)
      Produce all structurally feasible action dicts at a given stop.
      Lives here (not in BEHDV) because action enumeration depends on
      policy flags (charge_only) and pruning rules — decision-layer concerns.

  _prune_actions(actions, stop, state, full_data, delta, charge_only)
      Drop dominated or structurally infeasible actions before evaluation.

  find_horizon_end_stop(full_data, start_stop, horizon_hours, state)
      Compute the end of the look-ahead window, extended to cover mandatory
      rest events.

  evaluate_action(full_data, start_stop, end_stop, state, action, scenarios)
      Score one action across all scenarios via MILP.solve_horizon sub-problems.

  select_best_action(full_data, stop, state, n_scenarios, ...)
      Enumerate → prune → generate scenarios → score → tie-break → re-solve
      nominal MIP.  Returns the best action, score list, and nominal solution.

  run_simulation(full_data, n_scenarios, horizon_hours, ...)
      Main loop: initialise BEHDV and ScenarioTracker, iterate over stops,
      call select_best_action, advance vehicle, record realisations, then
      delegate the epilogue to runner.finalize_run.

Other modules
-------------
  BEHDV.py      — vehicle state and advance()
  MILP.py       — solve_horizon (rolling-horizon sub-problem solver)
  scenarios.py  — generate_scenarios, ScenarioTracker
  runner.py     — finalize_run (oracle, JSON save, tables, feasibility check)
  instances.py  — ALL_INSTANCES, make_data
  oracle.py     — oracle_solve, print_* (called via runner.finalize_run)
  plots.py      — plot_simulation_results (called by the CLI entry point)

Import chain
------------
  Simulation.py → BEHDV, MILP, scenarios, runner, plots, instances
  No circular imports.
"""

from __future__ import annotations

import datetime as _dt
import os as _os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from BEHDV     import BEHDV
from MILP      import solve_horizon, INFEASIBLE_PENALTY
from scenarios import generate_scenarios, ScenarioTracker
from settings  import LOWER_PCT, V_NOM, ecr, sample_travel_time
from runner    import finalize_run
from plots     import plot_simulation_results   # re-exported for callers



# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _solve_one_scenario(args):
    """
    Top-level worker function (must be picklable for ProcessPoolExecutor).

    Unpacks args and calls MILP.solve_horizon.  A single flat tuple is used
    rather than kwargs because multiprocessing requires picklable callables
    with picklable arguments, and named arguments add no serialisation benefit.

    args tuple: (full_data, start_stop, end_stop, init_state,
                 action, scenario, rho2_rem, time_limit, solve_mode)
    """
    (full_data, start_stop, end_stop, init_state,
     action, scenario, rho2_rem, time_limit, solve_mode) = args
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


# ══════════════════════════════════════════════════════════════════════════════
# HORIZON END-STOP
# ══════════════════════════════════════════════════════════════════════════════

def find_horizon_end_stop(full_data: dict, start_stop: int,
                          horizon_hours: float,
                          state=None) -> tuple[int, int]:
    """
    Return (end_stop, n_mandatory_rests) for a look-ahead window of
    ``horizon_hours`` hours starting at ``start_stop``.

    The horizon is extended automatically to cover any mandatory rest event
    (consecutive driving or shift driving limit) that falls within the nominal
    travel window.  This ensures the sub-problem always contains at least one
    complete shift cycle, so the MILP is never forced to leave an
    unconstrained tail.

    Parameters
    ----------
    full_data     : dict from instances.make_data()
    start_stop    : global stop index at which the window begins
    horizon_hours : nominal look-ahead length (hours of driving)
    state         : BEHDV or None — when provided, accumulated cd/sd from
                    the current state are used to estimate where the next
                    mandatory rest falls; otherwise assumed 0.

    Returns
    -------
    end_stop      : int — global index of the last stop in the window
    n_rests       : int — number of mandatory rest events estimated in window
    """
    N         = full_data["N"]
    D_nom     = full_data["D"]
    Tdrv_cons = full_data["Tdrv_cons"]   # 4.5h — consecutive driving limit
    Tdrv_sh1  = full_data["Tdrv_sh1"]    # 9.0h — shift driving limit
    Tr1       = full_data["Tr1"]          # 11.0h — daily rest duration

    cd_acc = state.cd if state is not None else 0.0
    sd_acc = state.sd if state is not None else 0.0

    t_remaining = horizon_hours
    n_rests     = 0
    stop        = start_stop

    while stop < N and t_remaining > 0:
        d = D_nom.get(stop, 0.0)
        if d <= 0.0:
            stop += 1
            continue

        cd_acc += d
        sd_acc += d

        if cd_acc > Tdrv_cons + 1e-6:
            # Break required: extend horizon by Tb45
            t_remaining -= full_data["Tb45"]
            cd_acc = d

        if sd_acc > Tdrv_sh1 + 1e-6:
            # Rest required: extend horizon by Tr1
            t_remaining -= Tr1
            n_rests += 1
            sd_acc = 0.0
            cd_acc = 0.0

        t_remaining -= d
        stop += 1

    C_set = set(full_data["C"])
    while stop in C_set and stop < full_data["N"]:
        stop += 1
    return min(stop, N), n_rests


# ══════════════════════════════════════════════════════════════════════════════
# ACTION ENUMERATION + PRUNING
# ══════════════════════════════════════════════════════════════════════════════

def enumerate_actions(stop: int, state, full_data: dict, delta_rng: float = 0.0,
                      charge_only: bool = False) -> list[dict]:
    """
    Return all structurally feasible action dicts at ``stop``.

    Action dict keys: y (0/1), break_type (None/"b45"/"b15"/"b30"),
                      rest_type (None/"r1"/"r2").

    charge_only=True  (recommended for large instances)
    ────────────────
    Only the charge binary y is enumerated; break_type and rest_type are set
    to None, leaving MILP.solve_horizon free to choose the optimal break/rest
    over the entire horizon.  Gives 2 actions at CS stops, 1 elsewhere.
    After the best y is chosen, a nominal MIP re-solve determines the actual
    break/rest to execute.

    charge_only=False  (full enumeration, default)
    ─────────────────
    All (y, break_type, rest_type) combinations are enumerated subject to:
      - break and rest are mutually exclusive at the same stop
      - b30 requires phi=1 (prior b15 in current shift)
      - r2 requires rho2_used < 3 (reduced-rest budget not exhausted)
    Here None means "no break/rest" (an explicit decision), not "free choice".
    """
    K_set     = set(full_data["K"])
    is_CS     = stop in K_set
    batt_full = state.e_arr > 0.98 * full_data["Ecap"] or stop >= full_data["N"]

    if charge_only:
        actions = [dict(y=0, break_type=None, rest_type=None)]
        if is_CS and not batt_full:
            actions.append(dict(y=1, break_type=None, rest_type=None))
        return actions

    break_opts = ["0", "b45", "b15"] if is_CS else ["0"]
    if state.phi == 1 and is_CS:
        break_opts.append("b30")
    rest_opts = ["0", "r1"]
    if state.rho2_used < 3:
        rest_opts.append("r2")

    actions = []
    for brk in break_opts:
        for rst in rest_opts:
            if brk != "0" and rst != "0":
                continue   # break and rest are mutually exclusive
            if not is_CS:
                y_vals = [0]
            else:
                y_vals = [0, 1]
            for y in y_vals:
                actions.append(dict(y=y, break_type=brk, rest_type=rst))
    return actions


def _prune_actions(actions: list, stop: int, state, full_data: dict,
                   delta: float, charge_only: bool = False, horizon: int = 48) -> tuple[list, int]:
    """
    Drop structurally dominated or infeasible actions before evaluation.

    Pruning rules
    -------------
    must_charge  : energy to next CS (worst-case) falls below Emin + buffer
                   → drop y=0 actions at CS stops
    must_reset_cd: cd + worst-case next leg exceeds Tdrv_cons
                   → drop actions with no break or rest
    must_rest    : sd or sw + worst-case next leg exceeds their limits
                   → drop actions with no rest
    batt_full    : drop pure-charge actions when SOC is already ≥ Ecap−1 kWh
    b15+phi=1    : b15 is invalid when phi=1 (would require phi=2)

    In charge_only mode, HoS rules are skipped because break_type=None means
    the sub-problem is free to insert breaks/rests as needed.

    Falls back to the full action list if pruning would leave nothing.

    Returns
    -------
    (pruned_actions, n_dropped)
    """
    N     = full_data["N"]
    K_set = set(full_data["K"])

    if stop >= N:
        return actions, 0

    D_next_wc = full_data["D"].get(stop, 0.0) * (1.0 + delta)

    must_charge = False
    must_reset_cd = False
    must_rest = False

    e_needed, cur = 0.0, stop
    while cur < N:
        d_nom = full_data["D"].get(cur, 0.0)
        L_km  = full_data.get("km", {}).get(cur, d_nom * V_NOM)
        d_min = max(d_nom * (1.0 - delta), 1e-9)
        v_wc  = L_km / d_min   # fastest speed → most energy (consistent with RO)
        e_needed += L_km * ecr(v_wc)
        cur += 1
        if cur in K_set or cur == N:
            break

    if state.e_arr - e_needed < full_data["Emin"] and stop in K_set:
        must_charge = True   # must charge to reach next CS, so no point enumerating y=0

    if state.cd + D_next_wc > full_data["Tdrv_cons"]:
        must_reset_cd = True

    if state.sd + D_next_wc > full_data["Tdrv_sh2"] and state.ext_shift_used < 2:
        must_rest = True
    elif state.sd + D_next_wc > full_data["Tdrv_sh1"] and state.ext_shift_used >= 2:
        must_rest = True
    elif state.sw + D_next_wc > full_data["Twrk_sh"]:
        must_rest = True

    pruned = []


    print()
    print(f"     [PRUNE] stop={stop}  must_charge={must_charge} (e_arr={state.e_arr:.1f}, e_needed={e_needed:.1f}) ")
    print(f"             must_reset_cd={must_reset_cd} (cd={state.cd:.2f}, D_next_wc={D_next_wc:.2f})  ")
    print(f"             must_rest={must_rest}   (sd={state.sd:.2f}, sw={state.sw:.2f}, D_next_wc={D_next_wc:.2f}, ext_shift_used={state.ext_shift_used})  ")


    for a in actions:
        y, brk, rst = a["y"], a["break_type"], a["rest_type"]
        if not charge_only and brk == "b15" and state.phi == 1:
            continue   # b15 invalid: phi would need to be 2
        if y and stop in K_set and state.e_arr > full_data["Ecap"] - 1.0 \
                and brk in (None, "0") and rst in (None, "0"):
            continue   # pure charge on full battery: queue cost, negligible gain
        if must_charge   and not y and stop in K_set: continue
        if must_reset_cd and brk in (None, "0") and rst in (None, "0"): continue
        if must_rest     and rst in (None, "0"):                       continue
        if horizon <= 48 and rst == "r1" and state.rho2_used < 2: continue  # will always choose reduced rest anyway
        pruned.append(a)

    if not pruned:
        return actions, 0   # safety: never prune everything
    return pruned, len(actions) - len(pruned)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE A SINGLE ACTION
# ══════════════════════════════════════════════════════════════════════════════


def _sol_activity_summary(sol: list, skip_stop0: bool = False) -> dict:
    """
    Summarise activity counts and durations across a horizon solution.

    Works correctly for both LP-relaxed and MIP solutions.  For LP solutions
    the binary variables (b45, rho1, …) are continuous in [0,1], so we detect
    activities using duration thresholds rather than rounded binaries:
      - a break  is present when taub > 1 min
      - a rest   is present when taur > 30 min
      - a charge is present when tauc > 1 min  (y may be fractional in LP)

    This avoids the common LP artefact where the solver spreads rest/break
    fractionally across many stops, each rounding to 0.

    Parameters
    ----------
    sol        : list of per-stop dicts from extract_horizon_solution
    skip_stop0 : if True, exclude local stop 0 from the counts/totals.
                 Use this when reporting horizon-wide activity to avoid
                 double-counting the forced decision at the current stop.

    Keys in the returned dict
    -------------------------
    stop0_tauc  : tauc at local stop 0 (h)  — always included regardless of skip_stop0
    stop0_taub  : taub at local stop 0 (h)
    stop0_taur  : taur at local stop 0 (h)
    stop0_tauq  : tauq at local stop 0 (h)
    stop0_ed    : departure SOC at local stop 0 (kWh)
    n_charge    : number of CS stops with tauc > 1 min  (excl. stop 0 when skip_stop0)
    n_break     : number of stops with taub > 1 min
    n_rest      : number of stops with taur > 30 min
    n_cb        : charge+break simultaneously
    n_cr        : charge+rest simultaneously
    tauc_total  : total charging time (h) in the horizon (excl. stop 0 when skip_stop0)
    taub_total  : total break time (h)
    taur_total  : total rest time (h)
    ea_end      : arrival SOC at the last horizon stop (kWh)
    """
    if not sol:
        return dict(stop0_tauc=0, stop0_taub=0, stop0_taur=0, stop0_tauq=0,
                    stop0_ed=0, n_charge=0, n_break=0, n_rest=0,
                    n_cb=0, n_cr=0, tauc_total=0, taub_total=0, taur_total=0,
                    ea_end=0)

    s0 = sol[0]
    stop0_tauc = float(s0.get("tauc", 0.0))
    stop0_taub = float(s0.get("taub", 0.0))
    stop0_taur = float(s0.get("taur", 0.0))
    stop0_tauq = float(s0.get("tauq", 0.0))
    stop0_ed   = float(s0.get("ed",   s0.get("ea", 0.0)))
    ea_end     = float(sol[-1].get("ea", 0.0))
    # At CS stops taub_hat = taub + tauc (charging covers part of the break).
    # Report taub_hat so the displayed value matches the actual break constraint.
    stop0_taub_hat = stop0_taub + (stop0_tauc if s0.get("is_K") else 0.0)

    # Thresholds to detect activity in LP-relaxed solutions
    _TAUC_MIN = 1.0  / 60   # 1 min — minimum meaningful charge
    _TAUB_MIN = 1.0  / 60   # 1 min — minimum meaningful break
    _TAUR_MIN = 30.0 / 60   # 30 min — avoids counting LP fractions as rests

    n_charge = n_break = n_rest = n_cb = n_cr = 0
    tauc_total = taub_total = taur_total = 0.0

    stops_to_scan = sol[1:] if skip_stop0 else sol
    for s in stops_to_scan:
        tauc = float(s.get("tauc", 0.0))
        taub = float(s.get("taub", 0.0))
        taur = float(s.get("taur", 0.0))
        has_charge = tauc > _TAUC_MIN
        has_break  = taub > _TAUB_MIN
        has_rest   = taur > _TAUR_MIN

        if has_charge:               n_charge += 1
        if has_break:                n_break  += 1
        if has_rest:                 n_rest   += 1
        if has_charge and has_break: n_cb     += 1
        if has_charge and has_rest:  n_cr     += 1

        tauc_total += tauc
        taub_total += taub
        taur_total += taur

    return dict(
        stop0_tauc = stop0_tauc,
        stop0_taub = stop0_taub,
        stop0_taur = stop0_taur,
        stop0_tauq = stop0_tauq,
        stop0_ed   = stop0_ed,
        n_charge   = n_charge,
        n_break    = n_break,
        n_rest     = n_rest,
        n_cb       = n_cb,
        n_cr       = n_cr,
        tauc_total = tauc_total,
        taub_total = taub_total,
        taur_total = taur_total,
        ea_end     = ea_end,
    )


def _scenario_stats(results: list, full_data: dict) -> dict:
    """
    Aggregate activity statistics across a list of scenario solve results.

    Each element of ``results`` is the dict returned by solve_horizon for one
    scenario.  Only feasible results (obj < INFEASIBLE_PENALTY / 2) contribute
    to the averages.

    Returns a dict with:
      n_feas           : int   — number of feasible scenarios
      mean_tauc        : float — mean total charging time per scenario (h)
      mean_taub        : float — mean total break time per scenario (h)
      mean_taur        : float — mean total rest time per scenario (h)
      mean_n_charge    : float — mean number of CS stops charged
      mean_n_break     : float — mean number of break stops
      mean_n_rest      : float — mean number of rest stops
      mean_n_cb        : float — mean number of charge+break stops
      mean_n_cr        : float — mean number of charge+rest stops
      mean_ea_end      : float — mean SOC at horizon end (kWh)
      ws_frac          : float — fraction of scenarios that had a warm-start
    """
    from MILP import INFEASIBLE_PENALTY as _PEN
    feas = [r for r in results
            if r is not None and r.get("feasible")
            and r.get("obj", _PEN) < _PEN / 2
            and r.get("sol")]
    n_feas = len(feas)
    if n_feas == 0:
        return dict(n_feas=0, mean_tauc=0, mean_taub=0, mean_taur=0,
                    mean_n_charge=0, mean_n_break=0, mean_n_rest=0,
                    mean_n_cb=0, mean_n_cr=0, mean_ea_end=0, ws_frac=0.0,
                    stop0_tauc=0.0, stop0_taub=0.0, stop0_taur=0.0,
                    stop0_tauq=0.0, stop0_ed=0.0)

    # skip_stop0=True: the fixed action at stop 0 is the same for all scenarios
    # for a given action evaluation, so we exclude it from the horizon-wide
    # averages.  The per-action log already shows stop-0 quantities separately.
    summaries = [_sol_activity_summary(r["sol"], skip_stop0=True) for r in feas]

    def _mean(key):
        return sum(s[key] for s in summaries) / n_feas

    # Also collect stop-0 quantities (identical across scenarios for a fixed action)
    s0_summaries = [_sol_activity_summary(r["sol"], skip_stop0=False) for r in feas[:1]]
    stop0 = s0_summaries[0] if s0_summaries else {}

    return dict(
        n_feas        = n_feas,
        mean_tauc     = _mean("tauc_total"),
        mean_taub     = _mean("taub_total"),
        mean_taur     = _mean("taur_total"),
        mean_n_charge = _mean("n_charge"),
        mean_n_break  = _mean("n_break"),
        mean_n_rest   = _mean("n_rest"),
        mean_n_cb     = _mean("n_cb"),
        mean_n_cr     = _mean("n_cr"),
        mean_ea_end   = _mean("ea_end"),
        stop0_tauc    = stop0.get("stop0_tauc", 0.0),
        stop0_taub    = stop0.get("stop0_taub", 0.0),
        stop0_taur    = stop0.get("stop0_taur", 0.0),
        stop0_tauq    = stop0.get("stop0_tauq", 0.0),
        stop0_ed      = stop0.get("stop0_ed",   0.0),
    )

def evaluate_action(full_data, start_stop, end_stop, state,
                    action, scenarios, time_limit=20,
                    n_workers=1, solve_mode="lp",
                    criterion="mean",
                    executor=None) -> tuple:
    """
    Score ``action`` across ``scenarios`` using MILP.solve_horizon sub-problems.

    Each scenario sub-problem fixes ``action`` at local stop 0 and solves the
    rolling-horizon model under the scenario's travel times and energies.
    The objective is arrival time at the horizon end.

    Parameters
    ----------
    full_data   : dict from instances.make_data()
    start_stop  : global index of the current decision stop
    end_stop    : global index of the last horizon stop
    state       : BEHDV — current vehicle state
    action      : dict — the candidate action to score
    scenarios   : list of scenario dicts from generate_scenarios
    time_limit  : int  — per sub-problem solver time limit (s)
    n_workers   : int  — number of parallel workers (1 = serial)
    solve_mode  : "lp" | "mip" | "both"
    criterion   : "mean" | "worst" | "best" — aggregation over scenarios
    executor    : ProcessPoolExecutor or None
        When provided, reuses this existing pool rather than spawning a new
        one per call.  select_best_action creates one pool for the full stop
        and passes it here, eliminating repeated spawn overhead (critical on
        Windows where multiprocessing uses 'spawn' instead of 'fork').

    On std being identical across actions at a given stop
    ------------------------------------------------------
    All actions share the same scenario draws.  Different actions at stop 0
    shift every scenario's objective by the same fixed dwell, so
    std({f_k + c}) == std({f_k}).  The std will differ across actions only
    when some scenarios are infeasible for certain actions (ok < N/N), or
    when the battery is so low that the charging pattern downstream changes
    differently per scenario.

    Returns
    -------
    (score, std, n_feasible, first_feas_result, raw_objs)
    """
    rho2_rem = 3 - state.rho2_used
    init_st  = state.as_init_state()

    arg_list = [
        (full_data, start_stop, end_stop, init_st,
         action, scen, rho2_rem, time_limit, solve_mode)
        for scen in scenarios
    ]

    if n_workers > 1:
        ordered = [None] * len(arg_list)

        def _dispatch(pool):
            futs = {pool.submit(_solve_one_scenario, a): i
                    for i, a in enumerate(arg_list)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:    ordered[i] = fut.result()
                except: ordered[i] = {"feasible": False, "obj": INFEASIBLE_PENALTY}

        if executor is not None:
            _dispatch(executor)          # reuse caller's pool
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                _dispatch(pool)          # fallback: own short-lived pool
        results = ordered
    else:
        results = [_solve_one_scenario(a) for a in arg_list]

    objs       = [r["obj"] for r in results]
    first_feas = next((r for r in results if r.get("feasible")), None)
    n_feas     = sum(1 for o in objs if o < INFEASIBLE_PENALTY / 2)

    if n_feas == 0:
        return INFEASIBLE_PENALTY, 0.0, 0, None, objs, results

    feas_objs = [o for o in objs if o < INFEASIBLE_PENALTY / 2]
    std       = float(np.std(feas_objs))

    if criterion == "worst":  score = float(max(objs))
    elif criterion == "best": score = float(min(feas_objs))
    else:                     score = float(np.mean(objs))   # default: mean

    return score, std, n_feas, first_feas, objs, results


# ══════════════════════════════════════════════════════════════════════════════
# SELECT BEST ACTION
# ══════════════════════════════════════════════════════════════════════════════

def select_best_action(full_data, stop: int, state,
                       n_scenarios=10, horizon_hours=12, delta=0.20,
                       scenario_seed=None, time_limit=20,
                       verbose=True, n_workers=1, solve_mode="lp",
                       charge_only=False, criterion="mean",
                       include_best=False, include_worst=False,
                       prev_nom_sol=None, log_fh=None,
                       tracker: ScenarioTracker = None,
                       precomputed_scenarios=None,
                       ext_shift_used: int = 0) -> tuple:
    """
    Enumerate, prune, score, and select the best action at ``stop``.

    Steps
    -----
    1. Enumerate all feasible actions → prune dominated ones.
    2. Determine horizon end stop (extended to cover mandatory HoS events).
    3. Generate ``n_scenarios`` travel-time / energy scenarios.
    4. Optionally warm-start scenarios from the previous nominal solution.
    5. Score each action under each scenario via MILP.solve_horizon.
    6. Apply tie-breaking: prefer y=1 if cost difference ≤ 5 min.
    7. Re-solve the winner with a nominal MIP (no scenario perturbation) to
       get integer-valued tauc/taub/taur for vehicle.advance().

    Parameters
    ----------
    tracker : ScenarioTracker or None
        When provided, records the generated scenarios at this stop.

    Returns
    -------
    (best_action, scores_list, nominal_sol)
    scores_list : [(action, score, std, n_feas, raw_objs), …] sorted by score
    nominal_sol : integer MIP result dict for the winning action on nominal D
    """
    t0 = time.perf_counter()

    def _p(msg):
        if verbose: print(msg)
        if log_fh:
            try: print(msg, file=log_fh)
            except Exception: pass

    # ── Extended shift driving exception (EU Reg 561/2006, Art. 6(2)) ─────────
    # Tdrv_sh2 (10h) is allowed instead of the normal Tdrv_sh1 (9h) at most
    # twice per week.  When the budget is not yet exhausted, pass 10h as the
    # effective shift-driving limit to every MILP/LP sub-problem built in this
    # call.  A shallow copy of full_data is sufficient since only a scalar changes.
    _tdrv_sh2 = full_data.get("Tdrv_sh2", full_data["Tdrv_sh1"])
    if ext_shift_used < 2 and _tdrv_sh2 > full_data["Tdrv_sh1"]:
        # M_sd is the big-M for the shift-driving linearisation (l2 = sd·ρ).
        # It must be ≥ max(sd), i.e. ≥ Tdrv_sh1.  If we raise Tdrv_sh1 to 10h
        # but leave M_sd=9h, the big-M constraints make sd>9h infeasible even
        # though sd_ub allows 10h.
        full_data = dict(full_data, Tdrv_sh1=_tdrv_sh2, M_sd=_tdrv_sh2)

    # ── 1. Enumerate + prune ─────────────────────────────────────────────────
    actions, n_pruned = _prune_actions(
        enumerate_actions(stop, state, full_data, charge_only=charge_only),
        stop, state, full_data, delta, charge_only=charge_only, horizon = horizon_hours)

    end_stop, n_rests = find_horizon_end_stop(
        full_data, stop, horizon_hours, state=state)

    K_set = set(full_data["K"]); C_set = set(full_data["C"])
    stype    = "CS" if stop in K_set else "CUST" if stop in C_set else "ORIG"
    travel_h = sum(full_data["D"].get(j, 0) for j in range(stop, end_stop))
    mode_tag = solve_mode + (",co" if charge_only else "")

    _p(f"\n[LA] stop {stop} ({stype})"
       f"  t={state.t_arr:.3f}h  soc={state.e_arr:.0f}kWh"
       f"  cd={state.cd:.2f}h  sd={state.sd:.2f}h  sw={state.sw:.2f}h"
       f"  phi={state.phi}  r2={state.rho2_used}  ext_sh={ext_shift_used}/2"
       f"  sd_lim={full_data['Tdrv_sh1']:.0f}h")
    _p(f"     horizon [{stop}->{end_stop}]"
       f"  travel={travel_h:.2f}h{f' +{n_rests}rest' if n_rests else ''}"
       f"  {len(actions)} actions × {n_scenarios} scen"
       f"  {n_workers}w  [{criterion},{mode_tag}]"
       f"{f'  pruned={n_pruned}' if n_pruned else ''}"
       f"{'  ws=prev' if prev_nom_sol else ''}")

    # ── 2. Generate scenarios (or use precomputed pool) ───────────────────────
    if precomputed_scenarios is not None:
        scenarios = precomputed_scenarios[:n_scenarios]
    else:
        scenarios = generate_scenarios(
            full_data, stop, end_stop,
            n_scenarios=n_scenarios, delta=delta, seed=scenario_seed,
            include_best=include_best, include_worst=include_worst)

    if tracker is not None:
        tracker.record_scenarios(stop, scenarios)

    # ── 3. Warm-start: re-index tail of previous nominal solution ─────────────
    tail_warm = None
    if prev_nom_sol and len(prev_nom_sol) > 1:
        tail_warm = [dict(s, i=s["i"] - 1) for s in prev_nom_sol[1:]
                     if s["i"] - 1 >= 0]

    # ── 4. FREE solve (LP on nominal D) to warm-start all scenario solves ─────
    free_relax = (solve_mode == "lp")
    free_sol   = None
    if prev_nom_sol is not None:
        t_free = time.perf_counter()
        _fr = solve_horizon(
            full_data      = full_data,
            start_stop     = stop,
            end_stop       = end_stop,
            init_state     = state.as_init_state(),
            fixed_action   = None,
            rho2_remaining = 3 - state.rho2_used,
            tee            = False,
            time_limit     = max(time_limit, 15),
            relax          = free_relax,
            warm_start     = tail_warm,
        )


        if not _fr["feasible"]:
            # ── DEBUG: re-solve with tee=True to dump the LP and get HiGHS IIS ──
            _p(f"     [FREE-] INFEASIBLE — re-solving with tee=True to dump LP")
            solve_horizon(
                full_data      = full_data,
                start_stop     = stop,
                end_stop       = end_stop,
                init_state     = state.as_init_state(),
                fixed_action   = None,
                rho2_remaining = 3 - state.rho2_used,
                tee            = True,          # ← dumps debug_stop23.lp
                time_limit     = 30,
                relax          = free_relax,
            )
            raise RuntimeError("DEBUG STOP — check debug_stop23.lp and stdout above")

        si  = _fr.get("solve_info", {})
        tag = "LP" if free_relax else "MIP"
        _fr_obj = f"obj={_fr['obj']:.3f}h" if _fr["feasible"] else "infeasible"
        _p(f"     [FREE-{tag}]  {_fr_obj}"
           f"  {time.perf_counter()-t_free:.1f}s"
           f"  {'ws=yes' if si.get('had_warm') else 'ws=no'}"
           f"  {si.get('n_vars','?')}v/{si.get('n_cons','?')}c")
        if _fr["feasible"]:
            free_sol = _fr["sol"]

    if free_sol:
        for scen in scenarios:
            scen["warm_start"] = free_sol

    # ── 5. Score all actions ──────────────────────────────────────────────────
    # Create one pool for the entire stop so all evaluate_action calls across
    # all actions share the same worker processes.  On Windows (spawn-based
    # multiprocessing) pool startup takes 1-3 seconds; creating it once per
    # stop instead of once per action×pass saves n_actions × pool_startup_cost.
    def _score_pass(mode, label="", executor=None):
        detail = []
        for act in actions:
            ta = time.perf_counter()
            score, std, n_feas, first_feas, objs, scen_results = evaluate_action(
                full_data, stop, end_stop, state, act, scenarios,
                time_limit=time_limit, n_workers=n_workers,
                solve_mode=mode, criterion=criterion,
                executor=executor)
            detail.append((act, score, std, n_feas, first_feas, objs, scen_results))
            _ss   = _scenario_stats(scen_results, full_data)
            n_ws  = len(scenarios) if any(sc.get("warm_start") for sc in scenarios) else 0
            _s0_tauc = _ss["stop0_tauc"]   # h — avg charge time at THIS stop
            _s0_taub = _ss["stop0_taub"]   # h — avg break time at THIS stop
            _p(f"  {label}y={act['y']}"
               f"  brk={act['break_type'] or '-':3}"
               f"  rst={act['rest_type'] or '-':2}"
               f"  {score:.3f}h ({std:.3f}h)"
               f"  ok={n_feas}/{len(scenarios)}"
               f"  ws={n_ws}/{len(scenarios)}"
               f"  tauc={_s0_tauc*60:.0f}m"
               f"  taub={_s0_taub*60:.0f}m"
               f"  ({time.perf_counter()-ta:.1f}s)")
        return detail

    _pool_ctx = (ProcessPoolExecutor(max_workers=n_workers)
                 if n_workers > 1 else None)
    try:
        if solve_mode == "both":
            _p("     --- LP pass ---")
            lp_det = _score_pass("lp",  "[LP] ", executor=_pool_ctx)
            _p("     --- MIP pass ---")
            scored = _score_pass("mip", "[MIP]", executor=_pool_ctx)
        else:
            scored = _score_pass(solve_mode, executor=_pool_ctx)
    finally:
        if _pool_ctx is not None:
            _pool_ctx.shutdown(wait=True)

    # ── Diagnostic: check whether per-scenario objectives are constant shifts ──
    # For each pair (action_0, action_k), compute the per-scenario delta
    # obj_k[s] - obj_0[s] for s=0..N_scen-1.  If all deltas are constant
    # (max-min < 1e-4h), the action only adds a fixed dwell at stop 0.
    # Non-constant deltas indicate the action changes downstream decisions.
    if verbose and len(scored) > 1:
        import numpy as _np_diag
        objs0 = _np_diag.array(scored[0][5], dtype=float)
        max_nonconstant = 0.0
        for s in scored[1:]:
            delta = _np_diag.array(s[5], dtype=float) - objs0
            feas_mask = (objs0 < INFEASIBLE_PENALTY / 2) & \
                        (_np_diag.array(s[5]) < INFEASIBLE_PENALTY / 2)
            if feas_mask.sum() > 1:
                spread = float(delta[feas_mask].max() - delta[feas_mask].min())
                max_nonconstant = max(max_nonconstant, spread)
        _p(f"     [DIAG] max per-scenario delta spread across actions: "
           f"{max_nonconstant*60:.2f} min"
           f"  ({'constant shift' if max_nonconstant < 1/60 else 'non-constant -- downstream decisions differ'})")

    # ── 6. Tie-breaking: prefer y=1 within same (brk, rst) if ≤5 min extra and prefer brk taken if y=1 anyway ───
    TIEBREAK = 5.0 / 60.0
    winner   = min(scored, key=lambda s: s[1])
    tb_flag  = False
    if winner[0]["y"] == 0 and (winner[0].get("break_type") not in (None, "0")
                                 or winner[0].get("rest_type") not in (None, "0")):
        w_brk, w_rst = winner[0]["break_type"], winner[0]["rest_type"]
        peers = [s for s in scored
                 if s[0]["y"] == 1
                 and s[0]["break_type"] == w_brk
                 and s[0]["rest_type"]  == w_rst
                 and s[1] < INFEASIBLE_PENALTY / 2]
        if peers:
            best_peer = min(peers, key=lambda s: s[1])
            if best_peer[1] <= winner[1] + TIEBREAK:
                winner  = best_peer
                tb_flag = True
    if winner[0]["y"] == 1 and winner[0].get("break_type") in (None, "0"):
        w_brk, w_rst = winner[0]["break_type"], winner[0]["rest_type"]
        peers = [s for s in scored
                 if s[0]["y"] == 1
                 and s[0]["break_type"] in ["b15", "b30", "b45"]
                 and s[1] < INFEASIBLE_PENALTY / 2]
        if peers:
            best_peer = min(peers, key=lambda s: s[1])
            if best_peer[1] <= winner[1] + TIEBREAK:
                winner  = best_peer
                tb_flag = True


    best_action = winner[0]
    best_score  = winner[1]

    # ── 7. Nominal MIP re-solve for the winner ────────────────────────────────
    # Produces integer-valued tauc/taub/taur for vehicle.advance() to execute.
    # Uses nominal D (no D_override) for a deterministic, reproducible result.
    #
    # Skip when y=0 and no break/rest: tauc=taub=taur=0 by definition, so
    # BEHDV.advance() needs no MILP solution — the fallback heuristic gives
    # identical results and saves the full NOM-MIP solve time.
    #
    # Fallback chain when y=1 or a break/rest is taken and the MIP fails:
    #   1. Retry without fixed_action (free MIP on nominal D).
    #   2. Use first feasible scenario LP solution (still gives valid tauc).
    #   3. None — BEHDV falls back to minimum-duration heuristic.
    t_nom    = time.perf_counter()
    _y       = best_action.get("y", 0)
    _brk     = best_action.get("break_type")
    _rst     = best_action.get("rest_type")
    _needs_milp = bool(_y or _brk in ["b15", "b30", "b45"] or _rst in ["r1", "r2"])   # any dwell at current stop

    if not _needs_milp:
        # No activity at stop 0: synthesise a trivial nom_sol so the rest of
        # the code (horizon plan, CHOSEN line) can read stop-0 quantities.
        nom_sol = dict(feasible=True, obj=best_score, sol=[dict(
            i=0, ta=state.t_arr, td=state.t_arr,
            ea=state.e_arr, ed=state.e_arr,
            cd=state.cd, sd=state.sd, sw=state.sw,
            tauc=0.0, tauq=0.0, taub=0.0, taur=0.0,
            y=0, b45=0, b15=0, b30=0, rho1=0, rho2=0,
            is_C=(stop in set(full_data["C"])),
            is_K=(stop in set(full_data["K"])),
            D_nom=full_data["D"].get(stop, 0.0),
        )])
        _p(f"     [NOM-MIP] y=0  brk=-  rst=-  skipped (no dwell)")
    elif solve_mode == "lp":
        nom_sol = solve_horizon(
            full_data      = full_data,
            start_stop     = stop,
            end_stop       = end_stop,
            init_state     = state.as_init_state(),
            fixed_action   = best_action,
            rho2_remaining = 3 - state.rho2_used,
            tee            = False,
            time_limit     = time_limit * 4,
            relax          = False,
            warm_start     = free_sol,
        )
        if not nom_sol.get("feasible"):
            _p(f"     [NOM-MIP] fixed-action infeasible, retrying free MIP...")
            nom_sol = solve_horizon(
                full_data      = full_data,
                start_stop     = stop,
                end_stop       = end_stop,
                init_state     = state.as_init_state(),
                fixed_action   = None,
                rho2_remaining = 3 - state.rho2_used,
                tee            = False,
                time_limit     = time_limit * 4,
                relax          = False,
                warm_start     = free_sol,
            )
        if not nom_sol.get("feasible"):
            first_feas = winner[4]
            if first_feas and first_feas.get("feasible"):
                _p(f"     [NOM-MIP] free MIP also infeasible, using first_feas LP sol")
                nom_sol = first_feas
    else:
        nom_sol = solve_horizon(
            full_data      = full_data,
            start_stop     = stop,
            end_stop       = end_stop,
            init_state     = state.as_init_state(),
            fixed_action   = best_action,
            rho2_remaining = 3 - state.rho2_used,
            tee            = False,
            time_limit     = time_limit * 4,
            relax          = False,
            warm_start     = free_sol,
        )
    nom_ok = nom_sol is not None and nom_sol.get("feasible", False)
    if nom_ok and nom_sol.get("sol") and _needs_milp:
        _nh = _sol_activity_summary(nom_sol["sol"], skip_stop0=True)
        _n_brk_only = _nh["n_break"]  - _nh["n_cb"]  - _nh["n_cr"]
        _n_rst_only = _nh["n_rest"]   - _nh["n_cr"]
        _n_chg_only = _nh["n_charge"] - _nh["n_cb"]  - _nh["n_cr"]
        _nom_sigma  = int(nom_sol["sol"][0].get("sigma", 0)) if nom_sol.get("sol") else 0
        _nom_mode   = f"  {'SEQ' if _nom_sigma else 'CONC'}" if _y and stop in set(full_data["K"]) else ""
        _nom_plan   = (f"  hor_stops:"
                       f" brk={_n_brk_only}"
                       f" rst={_n_rst_only}"
                       f" chg={_n_chg_only}"
                       f" c+b={_nh['n_cb']}"
                       f" c+r={_nh['n_cr']}")
        _p(f"     [NOM-MIP] y={_y}"
           f"  brk={_brk or '-'}"
           f"  rst={_rst or '-'}"
           f"{_nom_mode}"
           f"  ok{_nom_plan}"
           f"  {time.perf_counter()-t_nom:.1f}s")
    elif _needs_milp:
        _p(f"     [NOM-MIP] y={_y}"
           f"  brk={_brk or '-'}"
           f"  rst={_rst or '-'}"
           f"  INFEASIBLE"
           f"  {time.perf_counter()-t_nom:.1f}s")

    # ── Post-hoc break injection ──────────────────────────────────────────────
    # The LP underestimates tauc (picks a small value), so a cheaper small-break
    # or no-break action wins the LP scoring.  The nominal MIP may then pick a
    # larger tauc that would have made a concurrent b45 (or b30) break free.
    # Inject the upgrade here, covering brk=0 and brk=b15/b30 cases.
    if (nom_ok
            and nom_sol.get("sol")
            and stop in set(full_data["K"])
            and best_action.get("y") == 1
            and best_action.get("break_type") in (None, "0", "b15", "b30")
            and best_action.get("rest_type")  in (None, "0")):
        _s0_patch = nom_sol["sol"][0]
        _tc_patch  = float(_s0_patch.get("tauc", 0.0))
        _sig_patch = int(_s0_patch.get("sigma", 0))
        if _sig_patch == 0 and _tc_patch >= full_data["Tb45"] - 1e-6:
            _s0_patch["b45"] = 1
            best_action = dict(best_action, break_type="b45")
            _brk = "b45"
            _p(f"     [POST-HOC] tauc={_tc_patch*60:.0f}m >= 45m → b45 injected (free, concurrent)")
        elif _sig_patch == 0 and _tc_patch >= full_data["Tb30"] - 1e-6 and best_action.get("break_type") in (None, "0", "b15"):
            _s0_patch["b30"] = 1
            best_action = dict(best_action, break_type="b30")
            _brk = "b30"
            _p(f"     [POST-HOC] tauc={_tc_patch*60:.0f}m >= 30m → b30 injected (free, concurrent)")

    # ── LP vs MIP comparison log ──────────────────────────────────────────────
    if solve_mode == "both":
        lp_w  = min(lp_det,  key=lambda s: s[1])[0]
        mip_w = best_action
        def _astr(a):
            return (f"y={a['y']} brk={a['break_type'] or '-'} "
                    f"rst={a['rest_type'] or '-'}")
        match = (lp_w["y"] == mip_w["y"]
                 and lp_w["break_type"] == mip_w["break_type"]
                 and lp_w["rest_type"]  == mip_w["rest_type"])
        if match:
            _p(f"     [CMP] LP=MIP OK  {_astr(mip_w)}")
        else:
            _p(f"     [CMP] LP  {_astr(lp_w)}")
            _p(f"     [CMP] MIP {_astr(mip_w)}  <- executed")
            _p(f"     [CMP] DIFFER")

    # ── CHOSEN line ───────────────────────────────────────────────────────────
    # Compute ta and td at the current stop from the nominal solution.
    # ta = arrival time (known: state.t_arr).
    # td = departure time = ta + tauq + tauc + taub + taur + manoeuver.
    # When no nom_sol, derive td from minimum-duration action values.
    ta_cur = state.t_arr
    if nom_ok and nom_sol.get("sol"):
        _ch      = _sol_activity_summary(nom_sol["sol"], skip_stop0=False)
        _tauc    = _ch["stop0_tauc"]
        _taub    = _ch["stop0_taub_hat"] if "stop0_taub_hat" in _ch else _ch["stop0_taub"]
        _taur    = _ch["stop0_taur"]
        _tauq    = _ch["stop0_tauq"]
        _sigma   = int(nom_sol["sol"][0].get("sigma", 0))
        td_cur   = nom_sol["sol"][0].get("td", ta_cur + _tauq + _tauc + _taub + _taur)
        _soc_gain = _ch["stop0_ed"] - state.e_arr
        _chg_str  = f"  tauc={_tauc*60:.0f}m (+{_soc_gain:.0f}kWh)" if _y else "  tauc=0m"
        _brk_str  = f"  taub={_taub*60:.0f}m" if _taub > 1/60 else "  taub=0m"
    else:
        _tauc  = 0.0
        _sigma = 0
        _taub  = (full_data["Tb45"] if _brk == "b45" else
                  full_data["Tb15"] if _brk == "b15" else
                  full_data["Tb30"] if _brk == "b30" else 0.0)
        _taur  = (full_data["Tr1"]  if _rst == "r1"  else
                  full_data["Tr2"]  if _rst == "r2"  else 0.0)
        _tauq  = full_data["Q"].get(stop, 0.0) * _y if stop in set(full_data["K"]) else 0.0
        td_cur = ta_cur + _tauq + _tauc + _taub + _taur
        _chg_str = "  tauc=0m"
        _brk_str = f"  taub={_taub*60:.0f}m" if _taub > 1/60 else "  taub=0m"

    _mode_str = ""
    if _y and stop in set(full_data["K"]):
        _mode_str = f"  [{'SEQ' if _sigma else 'CONC'}]"

    _p(f"  -> CHOSEN y={_y}"
       f"  brk={_brk or '-'}"
       f"  rst={_rst or '-'}"
       f"{_chg_str}{_brk_str}"
       f"{_mode_str}"
       f"  ta={ta_cur:.3f}h  td={td_cur:.3f}h"
       f"  ({criterion}={best_score:.3f}h)"
       f"{'  [tiebreak]' if tb_flag else ''}"
       f"  {time.perf_counter()-t0:.1f}s")

    scores = [(s[0], s[1], s[2], s[3], s[5]) for s in scored]  # (act,score,std,n_feas,objs)
    scores.sort(key=lambda x: x[1])
    return best_action, scores, nom_sol


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(full_data: dict,
                   n_scenarios: int    = 10,
                   horizon_hours: float = 12.0,
                   delta: float        = 0.20,
                   seed: int           = 42,
                   time_limit: int     = 20,
                   verbose: bool       = True,
                   n_workers           = None,
                   solve_mode: str     = "lp",
                   charge_only: bool   = False,
                   criterion: str      = "mean",
                   include_best: bool  = False,
                   include_worst: bool = False,
                   run_id: str         = None) -> dict:
    """
    Run the rolling-horizon look-ahead simulation from stop 0 to stop N.

    At each stop, select_best_action evaluates all candidate actions across
    generated scenarios and returns the best one.  The BEHDV vehicle then
    advances with the winning action and a stochastic travel time draw.
    Realisations are recorded in the ScenarioTracker.

    After the route is complete, runner.finalize_run handles the oracle solve,
    JSON save, schedule printing, and feasibility check.

    Parameters
    ----------
    full_data      : dict from instances.make_data()
    n_scenarios    : number of scenarios per decision stop
    horizon_hours  : nominal look-ahead window length (hours)
    delta          : travel-time uncertainty half-width (e.g. 0.20 = ±20%)
    seed           : master RNG seed for reproducibility
    time_limit     : per-scenario sub-problem solver time limit (s)
    verbose        : print per-stop decisions and oracle output
    n_workers      : parallel workers for scenario evaluation (None = auto)
    solve_mode     : "lp" | "mip" | "both"
    charge_only    : True → only enumerate charge decision (y=0/1)
    criterion      : "mean" | "worst" | "best" — scenario aggregation
    include_best   : append a best-case (1−δ) deterministic scenario
    include_worst  : append a worst-case (1+δ) deterministic scenario
    run_id         : base name for output files (auto-generated if None)

    Returns
    -------
    dict — canonical results dict from runner.finalize_run, plus:
        scores_log       : list — per-stop score lists from select_best_action
        scenario_tracker : ScenarioTracker
    """
    if n_workers is None:
        n_workers = min(_os.cpu_count() or 1, n_scenarios)

    N          = full_data["N"]
    rng        = np.random.default_rng(seed)
    vehicle    = BEHDV(full_data)
    tracker    = ScenarioTracker(full_data)
    scores_log = []
    prev_sol   = None

    # ── Output directories and file paths ─────────────────────────────────────
    for d in ("logs", "figures", "solutions"):
        _os.makedirs(d, exist_ok=True)
    ts      = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    title   = full_data.get("title", "run")
    rid     = run_id or f"{title}_S{n_scenarios}_H{horizon_hours:.0f}_{ts}"
    paths   = dict(
        log = _os.path.join("logs",      f"{rid}.txt"),
        fig = _os.path.join("figures",   f"{rid}.png"),
        sol = _os.path.join("solutions", f"{rid}.json"),
        scn = _os.path.join("logs",      f"{rid}_scenarios.json"),
    )
    log = open(paths["log"], "w", buffering=1, encoding="utf-8")

    def _lp(msg):
        if verbose: print(msg)
        print(msg, file=log)

    mode_label = {"lp": "LP-relax", "mip": "MIP", "both": "LP+MIP"}.get(
        solve_mode, solve_mode)
    _lp(f"\n{'='*65}")
    _lp(f"  SIMULATION START   ({_dt.datetime.now():%Y-%m-%d %H:%M:%S})")
    _lp(f"  Instance : {title}   run_id={rid}")
    _lp(f"  Route    : {N} stops  departure={full_data.get('T_START',8):.0f}:00")
    _lp(f"  Settings : N_scen={n_scenarios}  H={horizon_hours}h  d={delta:.0%}"
        f"  workers={n_workers}  {mode_label}  [{criterion}]")
    _lp(f"  Log      : {paths['log']}")
    _lp(f"{'='*65}")

    wall_start = time.perf_counter()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for stop in range(N):
        if stop == 0:
            # Origin: no decision needed (no activity allowed at stop 0)
            action     = dict(y=0, break_type=None, rest_type=None)
            nom_sol    = None
            score_list = [(action, 0.0, 0.0, 0, [])]
        else:
            action, score_list, nom_sol = select_best_action(
                full_data      = full_data,
                stop           = stop,
                state          = vehicle,
                n_scenarios    = n_scenarios,
                horizon_hours  = horizon_hours,
                delta          = delta,
                scenario_seed  = int(rng.integers(0, 2**31)),
                time_limit     = time_limit,
                verbose        = verbose,
                n_workers      = n_workers,
                solve_mode     = solve_mode,
                charge_only    = charge_only,
                criterion      = criterion,
                include_best   = include_best,
                include_worst  = include_worst,
                prev_nom_sol   = prev_sol,
                log_fh         = log,
                tracker        = tracker,
                ext_shift_used = vehicle.ext_shift_used,
            )

        # ── Forced-rest safety net ────────────────────────────────────────────
        # If ALL scored actions are infeasible, the horizon was too short to
        # plan a mandatory rest.  Insert a minimum corrective rest.
        if stop > 0 and all(s[1] >= INFEASIBLE_PENALTY / 2 for s in score_list):
            rst_type = "r2" if vehicle.rho2_used < 3 else "r1"
            action   = dict(y=1 if stop in set(full_data["K"]) else 0,
                            break_type=None, rest_type=rst_type)
            _lp(f"  [!] FORCED REST ({rst_type}) at stop {stop}")
            end_fr, _ = find_horizon_end_stop(full_data, stop, 2.0, state=vehicle)
            forced    = solve_horizon(
                full_data      = full_data,
                start_stop     = stop,
                end_stop       = end_fr,
                init_state     = vehicle.as_init_state(),
                fixed_action   = action,
                rho2_remaining = 3 - vehicle.rho2_used,
                tee            = False,
                time_limit     = 30,
                relax          = False,
            )
            nom_sol    = forced if forced["feasible"] else None
            score_list = [(action, INFEASIBLE_PENALTY, 0.0, 0, [])]

        scores_log.append(score_list)
        prev_sol = nom_sol["sol"] if nom_sol and nom_sol.get("sol") else None

        # Draw actual travel time and energy from the uncertainty distribution.
        # This is the simulation realisation — NOT used in any decision above.
        d_nom = full_data["D"].get(stop, 0.0)
        D_next = sample_travel_time(d_nom, rng, lower_pct=delta, upper_pct=delta)
        km_leg = full_data.get("km", {}).get(stop, d_nom * V_NOM)
        v_act  = km_leg / D_next if D_next > 0 else V_NOM
        E_next = km_leg * ecr(v_act)

        vehicle.advance(action=action, D_next=D_next, E_next=E_next, milp_sol=nom_sol)
        tracker.record_realisation(stop, D_next, E_actual=E_next)

        if verbose and stop > 0:
            print(f"     -> arrived stop {vehicle.stop} after Driving {D_next:.2f}h and consuming {E_next:.1f}kWh"
                  f"  t={vehicle.t_arr:.3f}h  soc={vehicle.e_arr:.1f}kWh")

    wall = time.perf_counter() - wall_start
    arr  = vehicle.t_arr
    T0   = full_data.get("T_START", 8.0)
    _lp(f"\n{'='*65}")
    _lp(f"  SIMULATION COMPLETE")
    _lp(f"  Arrival  : {arr:.3f} h  ({int(arr):02d}:{int((arr%1)*60):02d})")
    _lp(f"  Duration : {arr - T0:.3f} h")
    _lp(f"  Wall     : {wall:.1f} s")
    _lp(f"{'='*65}\n")

    # ── Delegate epilogue to runner ───────────────────────────────────────────
    results = finalize_run(
        vehicle     = vehicle,
        full_data   = full_data,
        tracker     = tracker,
        run_id      = rid,
        paths       = paths,
        timing      = dict(wall_clock=wall, T_START=T0),
        log_fh      = log,
        verbose     = verbose,
        oracle_tee  = True,
        scores_log  = scores_log,
        method_meta = dict(
            method        = "simulation",
            n_scenarios   = n_scenarios,
            horizon_hours = horizon_hours,
            delta         = delta,
            criterion     = criterion,
            solve_mode    = solve_mode,
            charge_only   = charge_only,
            seed          = seed,
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTED-REALISATION SIMULATION (LA and RO)
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation_precomputed(
    full_data: dict,
    D_real: list,
    E_real: list,
    scenarios_by_stop: list      = None,
    n_scenarios: int             = 10,
    horizon_hours: float         = 12.0,
    delta: float                 = 0.20,
    time_limit: int              = 20,
    verbose: bool                = True,
    n_workers                    = None,
    solve_mode: str              = "lp",
    charge_only: bool            = False,
    criterion: str               = "mean",
    include_best: bool           = False,
    include_worst: bool          = False,
    run_id: str                  = None,
    oracle_tee: bool             = False,
) -> dict:
    """
    Run the rolling-horizon simulation using a precomputed uncertainty
    realisation and scenario pool from instance_io.py.

    Identical to run_simulation() except:
      - Travel times / energies come from D_real / E_real (no live RNG draw).
      - Scenarios come from scenarios_by_stop[stop][:n_scenarios] (LA mode).
        If scenarios_by_stop is None, the sub-problem uses delta=0 (RO mode).

    Parameters
    ----------
    full_data          : dict from instance_io.load_instance_json()
    D_real             : list[float] -- N realised travel times (h)
    E_real             : list[float] -- N realised energies (kWh)
    scenarios_by_stop  : list[list[dict]] or None
                         scenarios_by_stop[i] = scenario list at stop i.
                         None => RO mode: sub-problem solved at delta=0.
    n_scenarios        : how many scenarios to use per stop (first n taken)
    horizon_hours      : look-ahead window length (h)
    delta              : uncertainty half-width passed to sub-problem (LA only)
    time_limit         : per-scenario MILP time limit (s)
    verbose            : print per-stop decisions
    n_workers          : parallel workers (None = auto)
    solve_mode         : "lp" | "mip" | "both"
    charge_only        : enumerate charge decision only
    criterion          : "mean" | "worst" | "best"
    include_best       : append best-case scenario
    include_worst      : append worst-case scenario
    run_id             : override auto-generated run_id
    oracle_tee         : show HiGHS output in oracle solve

    Returns
    -------
    dict -- canonical results dict (same schema as run_simulation)
    """
    if n_workers is None:
        n_workers = min(_os.cpu_count() or 1, n_scenarios)

    N       = full_data["N"]
    vehicle = BEHDV(full_data)
    tracker = ScenarioTracker(full_data)
    scores_log = []
    prev_sol   = None

    assert len(D_real) == N, f"D_real length {len(D_real)} != N={N}"
    assert len(E_real) == N, f"E_real length {len(E_real)} != N={N}"
    if scenarios_by_stop is not None:
        assert len(scenarios_by_stop) == N, (
            f"scenarios_by_stop length {len(scenarios_by_stop)} != N={N}")

    # ── Output dirs and paths ─────────────────────────────────────────────────
    for d in ("logs", "figures", "solutions"):
        _os.makedirs(d, exist_ok=True)
    ts    = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    title = full_data.get("title", "run")
    alg   = "LA" if scenarios_by_stop is not None else "RO"
    rid   = run_id or f"{title}_{alg}_S{n_scenarios}_H{horizon_hours:.0f}_{ts}"
    paths = dict(
        log = _os.path.join("logs",      f"{rid}.txt"),
        fig = _os.path.join("figures",   f"{rid}.png"),
        sol = _os.path.join("solutions", f"{rid}.json"),
        scn = _os.path.join("logs",      f"{rid}_scenarios.json"),
    )
    log = open(paths["log"], "w", buffering=1, encoding="utf-8")

    def _lp(msg):
        if verbose: print(msg)
        print(msg, file=log)

    mode_label = {"lp": "LP-relax", "mip": "MIP", "both": "LP+MIP"}.get(
        solve_mode, solve_mode)
    T0 = full_data.get("T_START", 8.0)
    _lp(f"\n{'='*65}")
    _lp(f"  {alg} SIMULATION START  ({_dt.datetime.now():%Y-%m-%d %H:%M:%S})")
    _lp(f"  Instance : {title}   run_id={rid}")
    _lp(f"  Route    : {N} stops  departure={T0:.0f}:00")
    _lp(f"  Settings : N_scen={n_scenarios}  H={horizon_hours}h"
        f"  d={delta:.0%}  workers={n_workers}  {mode_label}  [{criterion}]")
    _lp(f"  Source   : precomputed realisations (D_real/E_real fixed)")
    _lp(f"{'='*65}")

    wall_start = time.perf_counter()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for stop in range(N):
        if stop == 0:
            action     = dict(y=0, break_type=None, rest_type=None)
            nom_sol    = None
            score_list = [(action, 0.0, 0.0, 0, [])]
        else:
            # Scenarios: use precomputed pool (LA) or empty list (RO, delta=0)
            if scenarios_by_stop is not None:
                stop_scens = scenarios_by_stop[stop][:n_scenarios]
            else:
                # RO: generate a single nominal scenario (delta=0)
                end_ro, _ = find_horizon_end_stop(
                    full_data, stop, horizon_hours, state=vehicle)
                stop_scens = generate_scenarios(
                    full_data   = full_data,
                    start_stop  = stop,
                    end_stop    = end_ro,
                    n_scenarios = n_scenarios,
                    delta       = 0.0,
                    seed        = None,
                )

            action, score_list, nom_sol = select_best_action(
                full_data             = full_data,
                stop                  = stop,
                state                 = vehicle,
                n_scenarios           = n_scenarios,
                horizon_hours         = horizon_hours,
                delta                 = delta if scenarios_by_stop is not None else 0.0,
                scenario_seed         = None,
                time_limit            = time_limit,
                verbose               = verbose,
                n_workers             = n_workers,
                solve_mode            = solve_mode,
                charge_only           = charge_only,
                criterion             = criterion,
                include_best          = include_best,
                include_worst         = include_worst,
                prev_nom_sol          = prev_sol,
                log_fh                = log,
                tracker               = tracker,
                precomputed_scenarios = stop_scens,
                ext_shift_used        = vehicle.ext_shift_used,
            )

        # ── Forced-rest safety net ────────────────────────────────────────────
        if stop > 0 and all(s[1] >= INFEASIBLE_PENALTY / 2 for s in score_list):
            rst_type = "r2" if vehicle.rho2_used < 3 else "r1"
            action   = dict(y=1 if stop in set(full_data["K"]) else 0,
                            break_type=None, rest_type=rst_type)
            _lp(f"  [!] FORCED REST ({rst_type}) at stop {stop}")
            end_fr, _ = find_horizon_end_stop(full_data, stop, 2.0, state=vehicle)
            forced    = solve_horizon(
                full_data      = full_data,
                start_stop     = stop,
                end_stop       = end_fr,
                init_state     = vehicle.as_init_state(),
                fixed_action   = action,
                rho2_remaining = 3 - vehicle.rho2_used,
                tee            = False,
                time_limit     = 30,
                relax          = False,
            )
            nom_sol    = forced if forced["feasible"] else None
            score_list = [(action, INFEASIBLE_PENALTY, 0.0, 0, [])]

        scores_log.append(score_list)
        prev_sol = nom_sol["sol"] if nom_sol and nom_sol.get("sol") else None

        # Use PRECOMPUTED travel time and energy (not drawn from RNG)
        D_next = float(D_real[stop])
        E_next = float(E_real[stop])

        vehicle.advance(action=action, D_next=D_next, E_next=E_next,
                        milp_sol=nom_sol)
        tracker.record_realisation(stop, D_next, E_actual=E_next)

        if verbose and stop > 0:
            print(f"     -> arrived stop {vehicle.stop}"
                  f"  D={D_next:.2f}h  E={E_next:.1f}kWh"
                  f"  t={vehicle.t_arr:.3f}h  soc={vehicle.e_arr:.1f}kWh")

    wall = time.perf_counter() - wall_start
    arr  = vehicle.t_arr
    _lp(f"\n{'='*65}")
    _lp(f"  {alg} SIMULATION COMPLETE")
    _lp(f"  Arrival  : {arr:.3f} h  ({int(arr):02d}:{int((arr%1)*60):02d})")
    _lp(f"  Duration : {arr - T0:.3f} h")
    _lp(f"  Wall     : {wall:.1f} s")
    _lp(f"{'='*65}\n")

    results = finalize_run(
        vehicle     = vehicle,
        full_data   = full_data,
        tracker     = tracker,
        run_id      = rid,
        paths       = paths,
        timing      = dict(wall_clock=wall, T_START=T0),
        log_fh      = log,
        verbose     = verbose,
        oracle_tee  = oracle_tee,
        scores_log  = scores_log,
        method_meta = dict(
            method        = alg,
            n_scenarios   = n_scenarios,
            horizon_hours = horizon_hours,
            delta         = delta if alg == "LA" else 0.0,
            criterion     = criterion,
            solve_mode    = solve_mode,
            charge_only   = charge_only,
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import random as _rnd
    from instances import ALL_INSTANCES

    _rnd.seed(5)
    name          = sys.argv[1]               if len(sys.argv) > 1 else "break_forced"
    n_scenarios   = int(sys.argv[2])          if len(sys.argv) > 2 else 5
    horizon_hours = float(sys.argv[3])        if len(sys.argv) > 3 else 8.0
    delta         = float(sys.argv[4])        if len(sys.argv) > 4 else 0.15
    n_workers     = int(sys.argv[5])          if len(sys.argv) > 5 else None
    solve_mode    = {"0":"lp","lp":"lp","1":"mip","mip":"mip",
                     "2":"both","both":"both"}.get(
                    sys.argv[6].lower() if len(sys.argv) > 6 else "0", "lp")
    criterion     = sys.argv[7]               if len(sys.argv) > 7 else "mean"
    charge_only   = (sys.argv[8].lower() in ("1","true","co")
                     if len(sys.argv) > 8 else False)
    correlation   = float(sys.argv[9])        if len(sys.argv) > 9 else 0.0
    # Usage: python Simulation.py <inst> [N_scen H δ workers mode criterion co corr]

    if name not in ALL_INSTANCES:
        print(f"Unknown: '{name}'.  Available: {list(ALL_INSTANCES)}")
        sys.exit(1)

    for t in range(10):
        data    = ALL_INSTANCES[name]()
        results = run_simulation(
            data,
            n_scenarios   = n_scenarios,
            horizon_hours = horizon_hours,
            delta         = delta,
            seed          = 42*t,
            time_limit    = 300,
            verbose       = True,
            n_workers     = n_workers,
            solve_mode    = solve_mode,
            criterion     = criterion,
            charge_only   = charge_only,
        )

        plot_simulation_results(
            results, data,
            title = f"{name}_n{n_scenarios}_H{int(horizon_hours)}_d{int(delta*100)}",
            save  = True,
            show  = False,
        )
        print(f"\n  Log      : {results['log_path']}")
        print(f"  Solution : {results['sol_path']}")
        print(f"  Figure   : {results['fig_path']}")
        print(f"  Scenarios: {results['scn_path']}")