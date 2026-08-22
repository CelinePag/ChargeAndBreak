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

  _prune_actions(actions, stop, state, full_data, cv, charge_only)
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

from src.simulation.BEHDV      import BEHDV
from src.methods.MILP       import solve_horizon, INFEASIBLE_PENALTY
from src.simulation.scenarios  import generate_scenarios, ScenarioTracker
from src.settings   import (TRAVEL_TIME_CV_TARGET, GUARD_QUANTILE, V_NOM, ecr,
                        sample_travel_time)
from src.simulation.supervisor import compute_flags, action_passes, supervise_action
from src.simulation.runner     import finalize_run
from src.plot.plots      import plot_simulation_results   # re-exported for callers
from src import paths as _paths

# RH3: named constant for the infeasible-scenario penalty used in scoring
T_PEN = INFEASIBLE_PENALTY



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
                 action, scenario, rho2_rem, ext_rem, time_limit, solve_mode)
    """
    (full_data, start_stop, end_stop, init_state,
     action, scenario, rho2_rem, ext_rem, time_limit, solve_mode) = args
    return solve_horizon(
        full_data      = full_data,
        start_stop     = start_stop,
        end_stop       = end_stop,
        init_state     = init_state,
        fixed_action   = action,
        D_override     = scenario["D"],
        E_override     = scenario.get("E"),
        rho2_remaining = rho2_rem,
        ext_remaining  = ext_rem,
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
    ``horizon_hours`` hours starting at ``start_stop``  (RH6 — this is the
    horizon end-stop algorithm referenced in paper §5.2; appendix pseudocode).

    Algorithm
    ---------
    Walk the route forward from `start_stop`, consuming a time budget
    initialised to `horizon_hours`:
      1. accumulate each leg's nominal duration into the consecutive-driving
         (cd) and shift-driving (sd) counters, seeded from the current
         vehicle state when provided;
      2. whenever cd would exceed 4.5 h, deduct a 45-min break from the
         budget and reset cd (a mandatory break fits inside the window);
      3. whenever sd would exceed the shift limit, deduct a full daily rest
         (11 h) from the budget, count it, and reset cd and sd;
      4. stop when the budget is exhausted or the destination is reached;
      5. finally, extend past any trailing customer stops so the window
         never ends immediately before a hard service commitment.

    This guarantees the sub-problem always contains at least one complete
    shift cycle, so the MILP is never forced to leave an unconstrained tail.

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

    A vehicle with no charging (full_data["no_charging"], the §8.4 diesel
    transform) enumerates y=0 only, in both modes below.

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
    # A vehicle that cannot take energy has no charge decision to make: the K
    # stops remain break-eligible, but y is fixed at 0 rather than enumerated
    # and scored.  Set by runner_dispatch._apply_diesel_mode (§8.4).
    is_CS     = stop in K_set and not full_data.get("no_charging")
    batt_full = state.e_arr > 0.98 * full_data["Ecap"] or stop >= full_data["N"]

    # Sea crossing: the vehicle is aboard for a known duration, so there is
    # exactly one action and no decision to make (mirrors the x_b45 = 1 /
    # taub = T_cross fixing in the MILP).  BEHDV substitutes the crossing
    # duration for the 45-minute minimum when it executes this action.
    if stop in {int(k) for k in (full_data.get("ferry") or {})}:
        return [dict(y=0, break_type="b45", rest_type=None)]

    if charge_only:
        actions = [dict(y=0, break_type=None, rest_type=None)]
        if is_CS and not batt_full:
            actions.append(dict(y=1, break_type=None, rest_type=None))
        return actions

    # 8.3 no-split axis: without the Art. 7 split the 45' block is the only
    # legal break, so b15/b30 leave the action set entirely.
    allow_split = bool(full_data.get("allow_split", True))
    break_opts = ["0", "b45"] + (["b15"] if allow_split else [])
    if allow_split and state.phi == 1:# and is_CS:
        break_opts.append("b30")
    rest_opts = ["0", "r1"]
    if state.rho2_used < int(full_data.get("rho_bar", 3)):
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
                   cv: float, charge_only: bool = False, horizon: int = 48,
                   prune_quantile: float | None = GUARD_QUANTILE,
                   verbose: bool = False) -> tuple[list, int]:
    """
    Drop structurally dominated or infeasible actions before evaluation (RH2).

    Feasibility guard (prune_quantile — settings.GUARD_QUANTILE default):
      None : guard DISABLED — no flag-based pruning at all; actions that
             would be infeasible on the next leg are left in and exposed by
             the per-scenario scores (an infeasible sub-problem earns
             INFEASIBLE_PENALTY, so it loses on its own).
      q<1  : the one-step checks (must_charge / must_reset_cd / must_rest,
             via supervisor.compute_flags — the same function the greedy
             rule and the opt-in S1 supervisor use) run at the xi q-quantile;
             residual risk alpha = 1 - q is reported.
      1.0  : checks at the full support corners [XI_MIN, XI_MAX] — exact:
             removes no action that is feasible under all realizations.

    Structural rules (always applied — action validity, not safety):
      batt_full    : drop pure-charge actions when SOC is already ≥ Ecap−1 kWh
      b15+phi=1    : b15 is invalid when phi=1 (would require phi=2)
      r1 dominance : with a short horizon and reduced-rest budget left,
                     r2 dominates r1

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

    flags = (compute_flags(full_data, stop, state, cv, prune_quantile)
             if prune_quantile is not None else None)

    if verbose and flags is not None:
        print(f"     [PRUNE] stop={stop}  must_charge={flags['must_charge']} "
              f"(e_arr={state.e_arr:.1f}, e_needed={flags['e_needed']:.1f})  "
              f"must_reset_cd={flags['must_reset_cd']} (cd={state.cd:.2f})  "
              f"must_rest={flags['must_rest']} (sd={state.sd:.2f}, "
              f"h={getattr(state, 'h', 0.0):.2f}, "
              f"D_wc={flags['D_next_wc']:.2f})")

    pruned = []
    for a in actions:
        y, brk, rst = a["y"], a["break_type"], a["rest_type"]
        if not charge_only and brk == "b15" and state.phi == 1:
            continue   # b15 invalid: phi would need to be 2
        if y and stop in K_set and state.e_arr > full_data["Ecap"] - 1.0 \
                and brk in (None, "0") and rst in (None, "0"):
            continue   # pure charge on full battery: queue cost, negligible gain
        if flags is not None:
            # normalise "0" placeholders to None for the shared check
            a_norm = dict(y=y,
                          break_type=None if brk in (None, "0") else brk,
                          rest_type=None if rst in (None, "0") else rst)
            if not charge_only and not action_passes(full_data, stop, state,
                                                     a_norm, flags):
                continue
            if flags["must_charge"] and not y and stop in K_set:
                continue   # applies in charge_only mode too
        if horizon <= 48 and rst == "r1" and state.rho2_used < 2:
            continue   # will always choose reduced rest anyway
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
    from src.methods.MILP import INFEASIBLE_PENALTY as _PEN
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
    rho2_rem = int(full_data.get("rho_bar", 3)) - state.rho2_used
    ext_rem  = int(full_data.get("ext_bar", 2)) - getattr(state,
                                                          "ext_shift_used", 0)
    init_st  = state.as_init_state()

    arg_list = [
        (full_data, start_stop, end_stop, init_st,
         action, scen, rho2_rem, ext_rem, time_limit, solve_mode)
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

    # RH3 — scenario aggregation: mean (default, infeasible penalised at
    # T_PEN), worst, best, or CVaR_0.8 (mean of the worst 20% of scenarios;
    # trades mean duration against tail risk).
    if criterion == "worst":
        score = float(max(objs))
    elif criterion == "best":
        score = float(min(feas_objs))
    elif criterion.startswith("cvar"):
        try:
            alpha = float(criterion.split("_", 1)[1]) if "_" in criterion else 0.8
        except ValueError:
            alpha = 0.8
        srt   = sorted(objs)                       # penalties included in tail
        k     = max(1, int(round((1.0 - alpha) * len(srt))))
        score = float(np.mean(srt[-k:]))
    else:
        score = float(np.mean(objs))               # default: mean

    return score, std, n_feas, first_feas, objs, results


# ══════════════════════════════════════════════════════════════════════════════
# SELECT BEST ACTION
# ══════════════════════════════════════════════════════════════════════════════

def select_best_action(full_data, stop: int, state,
                       n_scenarios=10, horizon_hours=12, cv=TRAVEL_TIME_CV_TARGET,
                       scenario_seed=None, time_limit=20,
                       verbose=True, n_workers=1, solve_mode="lp",
                       charge_only=False, criterion="mean",
                       include_best=False, include_worst=False,
                       prev_nom_sol=None, log_fh=None,
                       tracker: ScenarioTracker = None,
                       precomputed_scenarios=None,
                       ext_shift_used: int = 0,
                       prune_quantile: float | None = GUARD_QUANTILE,
                       tiebreak_min: float = 5.0,
                       cmp_log: list = None,
                       stats_out: dict = None) -> tuple:
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

    # ── Extended shift driving exception (EU Reg 561/2006, Art. 6(1)) ─────────
    # M6: the 10 h allowance is now modelled EXPLICITLY inside the MILP via
    # the z / q_ext budget mechanism (R16)–(R19); the old shallow-copy hack
    # that raised Tdrv_sh1 for every sub-problem is gone.  Here we only
    # compute the remaining budget and pass it to every sub-problem solve.
    ext_rem = int(full_data.get("ext_bar", 2)) - ext_shift_used

    # ── 1. Enumerate + prune ─────────────────────────────────────────────────
    actions, n_pruned = _prune_actions(
        enumerate_actions(stop, state, full_data, charge_only=charge_only),
        stop, state, full_data, cv, charge_only=charge_only,
        horizon=horizon_hours, prune_quantile=prune_quantile, verbose=verbose)

    end_stop, n_rests = find_horizon_end_stop(
        full_data, stop, horizon_hours, state=state)

    K_set = set(full_data["K"]); C_set = set(full_data["C"])
    L_set = set(full_data.get("L", []))
    stype    = ("CS" if stop in K_set else "CUST" if stop in C_set else
                "LAYBY" if stop in L_set else "ORIG")
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
            n_scenarios=n_scenarios, cv=cv, seed=scenario_seed,
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
            rho2_remaining = int(full_data.get("rho_bar", 3)) - state.rho2_used,
                ext_remaining  = ext_rem,
            tee            = False,
            time_limit     = max(time_limit, 15),
            relax          = free_relax,
            warm_start     = tail_warm,
        )


        if not _fr["feasible"]:
            # The free solve is ONLY a warm start for the scenario solves below
            # (its sol is handed to each scenario as `warm_start`), so its
            # infeasibility costs a cold start and nothing else: the decision
            # still comes from the per-action scenario scores, and evaluate_action
            # already prices infeasible scenarios at INFEASIBLE_PENALTY.
            #
            # This used to re-solve with tee=True and then `raise RuntimeError`.
            # That killed the process mid-route, so the run wrote no solution
            # JSON at all and was recorded as INCOMPLETE — excluded from the
            # gaps, from n_infe, from every denominator.  It fired on 10.6% of
            # long-route base runs, and those runs were then re-run until they
            # happened to succeed, which biased LA's reported reliability upward
            # exactly where LA is weakest.  Feasibility is BEHDV's call at
            # execution time, not this warm start's.
            _p(f"     [FREE-] INFEASIBLE — no warm start for this stop "
               f"(scenario scores decide; execution feasibility is BEHDV's call)")

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

    # ── 6. Tie-breaking: prefer y=1 within same (brk, rst) if ≤ tiebreak_min extra and prefer brk taken if y=1 anyway ───
    # The threshold is a POLICY parameter, not a constant: it fires on ~8% of
    # decisions and systematically buys opportunistic charge/break dwell, which
    # is a much larger effect than the scenario count.  tiebreak_min=0 leaves
    # only exact ties and gives the pure argmin policy.
    TIEBREAK = float(tiebreak_min) / 60.0
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
            rho2_remaining = int(full_data.get("rho_bar", 3)) - state.rho2_used,
                ext_remaining  = ext_rem,
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
                rho2_remaining = int(full_data.get("rho_bar", 3)) - state.rho2_used,
                ext_remaining  = ext_rem,
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
            rho2_remaining = int(full_data.get("rho_bar", 3)) - state.rho2_used,
                ext_remaining  = ext_rem,
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
        elif (_sig_patch == 0 and full_data.get("allow_split", True)
                and _tc_patch >= full_data["Tb30"] - 1e-6
                and best_action.get("break_type") in (None, "0", "b15")):
            _s0_patch["b30"] = 1
            best_action = dict(best_action, break_type="b30")
            _brk = "b30"
            _p(f"     [POST-HOC] tauc={_tc_patch*60:.0f}m >= 30m → b30 injected (free, concurrent)")

    # ── RH4: LP vs MIP action-selection agreement ─────────────────────────────
    # With solve_mode="both", record per-stop whether the LP-scored and
    # MIP-scored winners agree, plus the MIP-score delta induced by executing
    # the LP choice.  Aggregated by run_simulation* into the agreement-rate
    # summary (paper §5.2, Table lp-vs-milp).
    if solve_mode == "both":
        lp_w  = min(lp_det,  key=lambda s: s[1])[0]
        mip_w = best_action
        def _astr(a):
            return (f"y={a['y']} brk={a['break_type'] or '-'} "
                    f"rst={a['rest_type'] or '-'}")
        match = (lp_w["y"] == mip_w["y"]
                 and lp_w["break_type"] == mip_w["break_type"]
                 and lp_w["rest_type"]  == mip_w["rest_type"])
        _lp_choice_mip_score = next(
            (s[1] for s in scored
             if s[0]["y"] == lp_w["y"]
             and s[0]["break_type"] == lp_w["break_type"]
             and s[0]["rest_type"]  == lp_w["rest_type"]), None)
        if cmp_log is not None:
            cmp_log.append(dict(
                stop=stop, agree=bool(match),
                lp_action=_astr(lp_w), mip_action=_astr(mip_w),
                mip_score_of_mip_choice=float(best_score),
                mip_score_of_lp_choice=(float(_lp_choice_mip_score)
                                        if _lp_choice_mip_score is not None
                                        else None)))
        if match:
            _p(f"     [CMP] LP=MIP OK  {_astr(mip_w)}")
        else:
            _dstr = (f"  (MIP-score delta "
                     f"{(_lp_choice_mip_score - best_score)*60:.1f} min)"
                     if _lp_choice_mip_score is not None else "")
            _p(f"     [CMP] LP  {_astr(lp_w)}")
            _p(f"     [CMP] MIP {_astr(mip_w)}  <- executed")
            _p(f"     [CMP] DIFFER{_dstr}")

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

    # ── Solver-effort accounting for this decision ────────────────────────────
    # Wall-clock decision time is NOT the compute cost of a configuration: the
    # |Xi| scenario sub-problems run in one parallel wave (n_workers =
    # min(cpu_count, n_scenarios)), so the clock measures the slowest
    # sub-problem and the CPU contention around it, not the work done.  Summing
    # the per-sub-problem solve times gives a cost that scales with |Xi| and
    # with the horizon the way the reader expects.
    #
    # The time-limit hit rate matters just as much: every sub-problem is capped
    # (time_limit s, four times that for the nominal re-solve) and since the
    # 2026-08-21 change a capped solve is ACCEPTED at its incumbent rather than
    # discarded.  That is the right trade — an unconverged answer beats a false
    # infeasibility — but it means a configuration too big to solve no longer
    # announces itself in either the clock or the feasibility count.  This
    # counter is the only place it shows up, so report it alongside any gap
    # measured on a long horizon.
    if stats_out is not None:
        _cpu, _n_sub, _n_cap = 0.0, 0, 0
        for _s in scored:
            for _r in (_s[6] or []):
                _si = (_r or {}).get("solve_info") or {}
                _w  = _si.get("wall_s")
                if _w is None:
                    continue
                _cpu += float(_w)
                _n_sub += 1
                if _si.get("status") == "maxTimeLimit":
                    _n_cap += 1
        _si = (nom_sol or {}).get("solve_info") or {}
        if _si.get("wall_s") is not None:
            _cpu += float(_si["wall_s"])
            _n_sub += 1
            if _si.get("status") == "maxTimeLimit":
                _n_cap += 1
        stats_out.update(solve_cpu_s=_cpu, n_subproblems=_n_sub,
                         n_subproblem_capped=_n_cap)

    scores = [(s[0], s[1], s[2], s[3], s[5]) for s in scored]  # (act,score,std,n_feas,objs)
    scores.sort(key=lambda x: x[1])
    return best_action, scores, nom_sol


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(full_data: dict,
                   n_scenarios: int    = 10,
                   horizon_hours: float = 12.0,
                   cv: float           = TRAVEL_TIME_CV_TARGET,
                   seed: int           = 42,
                   time_limit: int     = 20,
                   verbose: bool       = True,
                   n_workers           = None,
                   solve_mode: str     = "lp",
                   charge_only: bool   = False,
                   criterion: str      = "mean",
                   include_best: bool  = False,
                   include_worst: bool = False,
                   run_id: str         = None,
                   supervised: bool    = False,
                   prune_quantile: float | None = GUARD_QUANTILE,
                   tiebreak_min: float = 5.0) -> dict:
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
    cv             : CV of the travel-time multiplier (e.g. 0.15)
    seed           : master RNG seed for reproducibility
    time_limit     : per-scenario sub-problem solver time limit (s)
    verbose        : print per-stop decisions and oracle output
    n_workers      : parallel workers for scenario evaluation (None = auto)
    solve_mode     : "lp" | "mip" | "both"
    charge_only    : True → only enumerate charge decision (y=0/1)
    criterion      : "mean" | "worst" | "best" — scenario aggregation
    include_best   : append the deterministic fast corner (xi = XI_MIN)
    include_worst  : append the deterministic slow corner (xi = XI_MAX)
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
    # S1/S2/RH4 event records
    events     = dict(interventions=[], decision_times=[], cmp_log=[],
                      repairs=[], plan_violations=[],
                      solve_cpu_s=[], n_subproblems=[],
                      n_subproblem_capped=[])

    # ── Output directories and file paths ─────────────────────────────────────
    _paths.ensure_dirs()
    ts      = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    title   = full_data.get("title", "run")
    rid     = run_id or f"{title}_S{n_scenarios}_H{horizon_hours:.0f}_{ts}"
    paths   = dict(
        log = _paths.log_out(f"{rid}.txt"),
        fig = _paths.figure_out(f"{rid}.png"),
        sol = _paths.solution_out(f"{rid}.json"),
        scn = _paths.log_out(f"{rid}_scenarios.json"),
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
    _lp(f"  Settings : N_scen={n_scenarios}  H={horizon_hours}h  cv={cv:.2f}"
        f"  workers={n_workers}  {mode_label}  [{criterion}]")
    _lp(f"  Log      : {paths['log']}")
    _lp(f"{'='*65}")

    wall_start = time.perf_counter()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for stop in range(N):
        t_dec = time.perf_counter()
        _eff = {}
        if stop == 0:
            # Origin: no decision needed (no activity allowed at stop 0)
            action     = dict(y=0, break_type=None, rest_type=None)
            nom_sol    = None
            score_list = [(action, 0.0, 0.0, 0, [])]
        else:
            action, score_list, nom_sol = select_best_action(
                stats_out      = _eff,
                full_data      = full_data,
                stop           = stop,
                state          = vehicle,
                n_scenarios    = n_scenarios,
                horizon_hours  = horizon_hours,
                cv             = cv,
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
                prune_quantile = prune_quantile,
                tiebreak_min   = tiebreak_min,
                cmp_log        = events["cmp_log"],
            )

        # ── No feasible action: the run ends here ─────────────────────────────
        # Every scored action came back infeasible over the look-ahead horizon,
        # so the policy has nothing legal to play.  The run is infeasible and
        # STOPS at this stop (2026-08-22).
        #
        # This replaces a forced-rest safety net, which inserted a minimum
        # corrective rest and carried on.  The net rescued 457 of the 614
        # stored runs that tripped it, and that is the problem: it converted a
        # policy failure into a completed route with a duration, so a
        # configuration too myopic to plan its own rest was scored as if it had.
        # What the net measured was the horizon's blind spot, not the policy's.
        if stop > 0 and all(s[1] >= INFEASIBLE_PENALTY / 2 for s in score_list):
            _lp(f"  [!] NO FEASIBLE ACTION at stop {stop} — run halted")
            vehicle.violations.append(dict(
                type="no_feasible_action", stop=stop, amount=0.0,
                detail=(f"every scored action was infeasible over the "
                        f"{horizon_hours:g} h look-ahead at stop {stop}")))
            vehicle.halt(stop, "no_feasible_action")
            break

        # ── S1: safety supervisor (identical layer for every policy) ──────────
        if supervised and stop > 0:
            action, itv = supervise_action(full_data, stop, vehicle, action,
                                           cv=cv, quantile=prune_quantile)
            if itv is not None:
                events["interventions"].append(itv)
                _lp(f"  [SUPERVISOR] stop {stop}: {itv['fixes']} "
                    f"({', '.join(itv['checks'])})")
                nom_sol = None   # planned durations no longer match

        events["decision_times"].append(time.perf_counter() - t_dec)
        for _k in ("solve_cpu_s", "n_subproblems", "n_subproblem_capped"):
            if _k in _eff:
                events[_k].append(_eff[_k])

        scores_log.append(score_list)
        prev_sol = nom_sol["sol"] if nom_sol and nom_sol.get("sol") else None

        # Draw actual travel time and energy from the uncertainty distribution.
        # This is the simulation realisation — NOT used in any decision above.
        # RH2: the pruning/supervisor use the SAME cv as this draw.
        d_nom = full_data["D"].get(stop, 0.0)
        D_next = sample_travel_time(d_nom, rng, cv=cv)
        km_leg = full_data.get("km", {}).get(stop, d_nom * V_NOM)
        v_act  = km_leg / D_next if D_next > 0 else V_NOM
        E_next = km_leg * ecr(v_act)

        vehicle.advance(action=action, D_next=D_next, E_next=E_next, milp_sol=nom_sol)
        tracker.record_realisation(stop, D_next, E_actual=E_next)

        if verbose and stop > 0:
            print(f"     -> arrived stop {vehicle.stop} after Driving {D_next:.2f}h and consuming {E_next:.1f}kWh"
                  f"  t={vehicle.t_arr:.3f}h  soc={vehicle.e_arr:.1f}kWh")

        if vehicle.is_halted:
            _lp(f"  [!] {vehicle.halt_reason.upper()} at stop "
                f"{vehicle.halted_at} — run halted, route not completed")
            break

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
        events      = events,
        method_meta = dict(
            method        = "simulation",
            n_scenarios   = n_scenarios,
            horizon_hours = horizon_hours,
            cv            = cv,
            criterion     = criterion,
            solve_mode    = solve_mode,
            charge_only   = charge_only,
            supervised    = supervised,
            prune_quantile= prune_quantile,
            tiebreak_min  = tiebreak_min,
            # Solver budget and parallelism: neither was recorded before, so
            # stored runs cannot be checked for comparability after the fact —
            # which is exactly how the LA ladder became unreadable.
            time_limit    = time_limit,
            n_workers     = n_workers,
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
    n_scenarios: int             = 10,
    horizon_hours: float         = 12.0,
    cv: float                    = TRAVEL_TIME_CV_TARGET,
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
    supervised: bool             = False,
    prune_quantile: float | None = GUARD_QUANTILE,
    tiebreak_min: float          = 5.0,
    resume: bool                 = False,
    external_policy              = None,
    alg_label: str               = "LA",
) -> dict:
    """
    Run the rolling-horizon look-ahead (LA) simulation using a precomputed
    uncertainty realisation from instance_io.py.

    Identical to run_simulation() except travel times / energies come from
    D_real / E_real (no live RNG draw).  Scenarios are NOT precomputed or
    stored anywhere: at each decision stop, select_best_action() draws
    n_scenarios fresh scenarios over [stop, horizon_end) via
    scenarios.generate_scenarios() (same mechanism RO.py uses), so LA runs
    are not tied to a fixed scenario pool across repeated runs.

    Parameters
    ----------
    full_data          : dict from instance_io.load_instance_json()
    D_real             : list[float] -- N realised travel times (h)
    E_real             : list[float] -- N realised energies (kWh)
    n_scenarios        : how many scenarios to draw per stop
    horizon_hours      : look-ahead window length (h)
    cv                 : CV of the multiplier passed to sub-problem (LA only)
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
    external_policy    : optional callable (full_data, stop, vehicle) ->
                         (action, score_list, nom_sol).  When given, it
                         replaces select_best_action at every stop>0 and the
                         run is labelled alg_label; execution, metrics and
                         saving are byte-identical to an LA run (used by
                         ML/code/rollout.py to evaluate learned policies).
    alg_label          : method name recorded in run_id / metadata

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
    events     = dict(interventions=[], decision_times=[], cmp_log=[],
                      repairs=[], plan_violations=[],
                      solve_cpu_s=[], n_subproblems=[],
                      n_subproblem_capped=[])

    assert len(D_real) == N, f"D_real length {len(D_real)} != N={N}"
    assert len(E_real) == N, f"E_real length {len(E_real)} != N={N}"

    # ── Output dirs and paths ─────────────────────────────────────────────────
    _paths.ensure_dirs()
    ts    = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    title = full_data.get("title", "run")
    alg   = alg_label
    rid   = run_id or f"{title}_{alg}_S{n_scenarios}_H{horizon_hours:.0f}_{ts}"
    paths = dict(
        log = _paths.log_out(f"{rid}.txt"),
        fig = _paths.figure_out(f"{rid}.png"),
        sol = _paths.solution_out(f"{rid}.json"),
        scn = _paths.log_out(f"{rid}_scenarios.json"),
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
        f"  cv={cv:.2f}  workers={n_workers}  {mode_label}  [{criterion}]")
    _lp(f"  Source   : precomputed realisations (D_real/E_real fixed)")
    _lp(f"{'='*65}")

    wall_start = time.perf_counter()

    # ── Resume support: checkpoint keyed by instance + params ─────────────────
    # A stop-by-stop LA run is expensive; if it crashed, --resume continues from
    # the last completed stop instead of restarting.  The checkpoint holds the
    # full vehicle trajectory (fixed D_real/E_real make it exactly restorable);
    # only stops not yet decided are re-solved.  Deleted on clean completion.
    import json as _json
    _ckpt_key = "_".join(str(p) for p in (
        title, alg, f"ns{n_scenarios}", f"h{horizon_hours:g}", criterion,
        solve_mode, f"co{int(charge_only)}", f"sup{int(supervised)}",
        f"pq{prune_quantile}")).replace("/", "-").replace(" ", "")
    _ckpt_dir  = _paths.solutions(".checkpoints")
    _ckpt_path = _os.path.join(_ckpt_dir, _ckpt_key + ".json")

    def _write_ckpt():
        try:
            _os.makedirs(_ckpt_dir, exist_ok=True)
            _tmp = _ckpt_path + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as _fh:
                _json.dump(dict(vehicle=vehicle.to_checkpoint(),
                                scores_log=scores_log, events=events), _fh,
                           default=lambda o: o.item() if hasattr(o, "item")
                           else str(o))
            _os.replace(_tmp, _ckpt_path)          # atomic
        except Exception as _e:                    # never let I/O kill the run
            _lp(f"  [RESUME] checkpoint write failed ({_e})")

    start_stop = 0
    if resume and _os.path.isfile(_ckpt_path):
        try:
            with open(_ckpt_path, encoding="utf-8") as _fh:
                _ck = _json.load(_fh)
            vehicle.load_checkpoint(_ck["vehicle"])
            scores_log = _ck.get("scores_log", scores_log)
            for _k, _v in (_ck.get("events") or {}).items():
                events[_k] = _v
            start_stop = int(_ck["vehicle"]["n_done"])
            for _j in range(min(start_stop, N)):    # rebuild tracker realisations
                tracker.record_realisation(_j, float(D_real[_j]),
                                           E_actual=float(E_real[_j]))
            _lp(f"  [RESUME] restored checkpoint: {start_stop}/{N} stops done, "
                f"continuing from stop {start_stop}")
        except Exception as _e:
            _lp(f"  [RESUME] checkpoint load failed ({_e}); starting fresh")
            start_stop = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    for stop in range(start_stop, N):
        t_dec = time.perf_counter()
        _eff = {}
        if stop == 0:
            action     = dict(y=0, break_type=None, rest_type=None)
            nom_sol    = None
            score_list = [(action, 0.0, 0.0, 0, [])]
        elif external_policy is not None:
            # Learned/externally supplied policy: one function call replaces
            # the whole scenario-lookahead machinery.  decision_times below
            # therefore measures the policy's true online latency.
            action, score_list, nom_sol = external_policy(full_data, stop, vehicle)
        else:
            # Scenarios are generated live inside select_best_action (no
            # precomputed pool): it draws n_scenarios over [stop, horizon_end)
            # via generate_scenarios() when precomputed_scenarios is left None.
            action, score_list, nom_sol = select_best_action(
                stats_out      = _eff,
                full_data             = full_data,
                stop                  = stop,
                state                 = vehicle,
                n_scenarios           = n_scenarios,
                horizon_hours         = horizon_hours,
                cv                    = cv,
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
                ext_shift_used        = vehicle.ext_shift_used,
                prune_quantile        = prune_quantile,
                tiebreak_min          = tiebreak_min,
                cmp_log               = events["cmp_log"],
            )

        # ── No feasible action: the run ends here ─────────────────────────────
        # See the identical block in run_simulation for why the forced-rest
        # safety net was removed: it turned a policy failure into a completed
        # route with a duration, and what it measured was the horizon's blind
        # spot rather than the policy's quality.
        if stop > 0 and all(s[1] >= INFEASIBLE_PENALTY / 2 for s in score_list):
            _lp(f"  [!] NO FEASIBLE ACTION at stop {stop} — run halted")
            vehicle.violations.append(dict(
                type="no_feasible_action", stop=stop, amount=0.0,
                detail=(f"every scored action was infeasible over the "
                        f"{horizon_hours:g} h look-ahead at stop {stop}")))
            vehicle.halt(stop, "no_feasible_action")
            break

        # ── S1: safety supervisor (identical layer for every policy) ──────────
        if supervised and stop > 0:
            action, itv = supervise_action(full_data, stop, vehicle, action,
                                           cv=cv, quantile=prune_quantile)
            if itv is not None:
                events["interventions"].append(itv)
                _lp(f"  [SUPERVISOR] stop {stop}: {itv['fixes']} "
                    f"({', '.join(itv['checks'])})")
                nom_sol = None   # planned durations no longer match

        events["decision_times"].append(time.perf_counter() - t_dec)
        for _k in ("solve_cpu_s", "n_subproblems", "n_subproblem_capped"):
            if _k in _eff:
                events[_k].append(_eff[_k])

        scores_log.append(score_list)
        prev_sol = nom_sol["sol"] if nom_sol and nom_sol.get("sol") else None

        # Use PRECOMPUTED travel time and energy (not drawn from RNG)
        D_next = float(D_real[stop])
        E_next = float(E_real[stop])

        vehicle.advance(action=action, D_next=D_next, E_next=E_next,
                        milp_sol=nom_sol)
        tracker.record_realisation(stop, D_next, E_actual=E_next)
        if resume:
            _write_ckpt()          # after each completed stop, for restart

        if verbose and stop > 0:
            print(f"     -> arrived stop {vehicle.stop}"
                  f"  D={D_next:.2f}h  E={E_next:.1f}kWh"
                  f"  t={vehicle.t_arr:.3f}h  soc={vehicle.e_arr:.1f}kWh")

        if vehicle.is_halted:
            _lp(f"  [!] {vehicle.halt_reason.upper()} at stop "
                f"{vehicle.halted_at} — run halted, route not completed")
            break

    # run completed cleanly: drop the checkpoint so a rerun starts fresh
    if resume:
        try:
            if _os.path.isfile(_ckpt_path):
                _os.remove(_ckpt_path)
        except OSError:
            pass

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
        events      = events,
        method_meta = dict(
            method        = alg,
            n_scenarios   = n_scenarios,
            horizon_hours = horizon_hours,
            cv            = cv if alg == "LA" else 0.0,
            criterion     = criterion,
            solve_mode    = solve_mode,
            charge_only   = charge_only,
            supervised    = supervised,
            prune_quantile= prune_quantile,
            tiebreak_min  = tiebreak_min,
            # Solver budget and parallelism: neither was recorded before, so
            # stored runs cannot be checked for comparability after the fact —
            # which is exactly how the LA ladder became unreadable.
            time_limit    = time_limit,
            n_workers     = n_workers,
            # Travels on full_data (it is consumed in MILP._build_sub_data);
            # surfaced here so a guarded run is self-describing.
            la_energy_quantile = full_data.get("la_energy_quantile"),
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import random as _rnd
    from src.instance_gen.instances import ALL_INSTANCES

    _rnd.seed(5)
    name          = sys.argv[1]               if len(sys.argv) > 1 else "break_forced"
    n_scenarios   = int(sys.argv[2])          if len(sys.argv) > 2 else 5
    horizon_hours = float(sys.argv[3])        if len(sys.argv) > 3 else 8.0
    cv            = float(sys.argv[4])        if len(sys.argv) > 4 else TRAVEL_TIME_CV_TARGET
    n_workers     = int(sys.argv[5])          if len(sys.argv) > 5 else None
    solve_mode    = {"0":"lp","lp":"lp","1":"mip","mip":"mip",
                     "2":"both","both":"both"}.get(
                    sys.argv[6].lower() if len(sys.argv) > 6 else "0", "lp")
    criterion     = sys.argv[7]               if len(sys.argv) > 7 else "mean"
    charge_only   = (sys.argv[8].lower() in ("1","true","co")
                     if len(sys.argv) > 8 else False)
    correlation   = float(sys.argv[9])        if len(sys.argv) > 9 else 0.0
    # Usage: python -m src.simulation.Simulation <inst> [N_scen H cv workers mode criterion co corr]

    if name not in ALL_INSTANCES:
        print(f"Unknown: '{name}'.  Available: {list(ALL_INSTANCES)}")
        sys.exit(1)

    for t in range(10):
        data    = ALL_INSTANCES[name]()
        results = run_simulation(
            data,
            n_scenarios   = n_scenarios,
            horizon_hours = horizon_hours,
            cv            = cv,
            seed          = 42*t,
            time_limit    = 300,
            verbose       = True,
            n_workers     = n_workers,
            solve_mode    = solve_mode,
            criterion     = criterion,
            charge_only   = charge_only,
        )

        print(f"\n  Log      : {results['log_path']}")
        print(f"  Solution : {results['sol_path']}")
        print(f"  Figure   : plot later with `python -m src.plot.plots {results['run_id']}`")
        print(f"  Scenarios: {results['scn_path']}")