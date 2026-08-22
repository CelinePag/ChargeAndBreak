"""
recourse.py — SP1/RO1: fixed-structure online duration recourse
================================================================
Shared execution path for the offline plans (two-stage stochastic plan and
adjustable robust plan).  Replaces the old open-loop execution of
scenario-averaged durations, which was not a well-defined policy (averaged
durations satisfy no scenario's constraints in general).

Execution loop (SP1 pseudocode from the code-change list)
---------------------------------------------------------
offline:  solve 2-SP / RO  ->  first-stage binaries per stop (the "plan")
online, at each stop i with realized state sigma_i:
    LP_i = deterministic model over [i, N] with
           - binaries fixed to the plan for stops >= i
           - initial conditions from the realized state
           - nominal travel times for legs >= i
    if LP_i feasible:
        execute durations (tauc, taub, taur) at stop i only
        (TW5: the remaining out-of-window indicators delta stay INTEGER, so
        the "LP" is a tiny MIP — still sub-second)
    else:
        REPAIR: MILP over [i, N] where binary activities may only be ADDED
                (never removed), objective  BIGP * (#added) + ta[N] + beta*sum(delta)
                (SP-recourse ordering: min #added activities, then
                ta[N] + beta·Σ delta);
        if feasible: execute stop-i action, adopt the repaired structure for
                     the remaining stops, log the repair event;
        else: log a plan violation and hand control to the safety supervisor.

The plan's STRUCTURE is committed offline, its durations adapt online, and
the repair frequency is itself a reported robustness metric (S2).

Note: with the structural binaries fixed the remaining sub-problem is tiny
(only the PWL segment indicators and mode flags stay integer), so each
per-stop re-optimisation solves in well under a second.

Import chain
------------
  recourse.py -> BEHDV, MILP, scenarios, supervisor
  twosp.py / RO.py -> recourse
"""

from __future__ import annotations

import time

from src.simulation.BEHDV      import BEHDV
from src.methods.MILP       import solve_horizon
from src.simulation.scenarios  import ScenarioTracker
from src.settings   import TRAVEL_TIME_CV_TARGET, GUARD_QUANTILE
from src.simulation.supervisor import supervise_action


def _plan_entry(entry: dict) -> dict:
    """Normalise one plan stop into {y, break_type, rest_type}."""
    return dict(
        y          = int(entry.get("y", 0) or 0),
        break_type = entry.get("break_type"),
        rest_type  = entry.get("rest_type"),
    )


def _sol_to_plan_updates(sol: list, start_stop: int) -> dict:
    """Map a horizon solution (local indices) back to global plan entries."""
    upd = {}
    for s in sol:
        g = start_stop + s["i"]
        brk = ("b45" if s.get("b45") else "b15" if s.get("b15") else
               "b30" if s.get("b30") else None)
        rst = ("r1" if s.get("rho1") else "r2" if s.get("rho2") else None)
        upd[g] = dict(y=int(s.get("y", 0)), break_type=brk, rest_type=rst)
    return upd


def run_plan_with_recourse(full_data: dict,
                           plan: list,
                           D_real: list,
                           E_real: list,
                           method_name: str,
                           log_fn,
                           cv: float = TRAVEL_TIME_CV_TARGET,
                           supervised: bool = False,
                           prune_quantile: float | None = GUARD_QUANTILE,
                           time_limit: int = 60,
                           verbose: bool = True) -> tuple:
    """
    Execute an offline plan with online duration recourse (SP1/RO1).

    Parameters
    ----------
    full_data      : instance dict
    plan           : list of per-stop dicts with keys y, break_type, rest_type
                     (global stop order, index 0..N; entries at 0 and N ignored)
    D_real, E_real : realized travel times / energies per leg
    method_name    : "2SP" | "RO" — used in logs and event records
    log_fn         : callable(str) for logging
    cv             : CV of the travel-time multiplier (for the safety supervisor)
    supervised     : apply the S1 safety supervisor (default False = raw
                     mode: violations recorded, never prevented)
    prune_quantile : supervisor worst-case quantile (RH2)
    time_limit     : per-stop solver time limit (s)

    Returns
    -------
    (vehicle, tracker, events) where events is a dict with keys:
        repairs        : list of repair-event dicts
        plan_violations: list of stops where even repair failed
        interventions  : list of supervisor-intervention dicts
        decision_times : list of per-stop wall times (s)
    """
    N        = full_data["N"]
    rho_bar  = int(full_data.get("rho_bar", 3))
    ext_bar  = int(full_data.get("ext_bar", 2))
    vehicle  = BEHDV(full_data)
    tracker  = ScenarioTracker(full_data)

    plan_by_stop = {i: _plan_entry(plan[i]) for i in range(min(len(plan), N + 1))}

    events = dict(repairs=[], plan_violations=[], interventions=[],
                  decision_times=[])

    for stop in range(N):
        t0 = time.perf_counter()

        if stop == 0:
            action  = dict(y=0, break_type=None, rest_type=None)
            nom_sol = None
        else:
            entry  = plan_by_stop.get(stop, dict(y=0, break_type=None,
                                                 rest_type=None))
            action = dict(entry)

            # fixed_plan in LOCAL indices over [stop, N]
            fixed_plan = {j: plan_by_stop.get(stop + j)
                          for j in range(N - stop)
                          if plan_by_stop.get(stop + j) is not None}

            common = dict(
                full_data      = full_data,
                start_stop     = stop,
                end_stop       = N,
                init_state     = vehicle.as_init_state(),
                rho2_remaining = rho_bar - vehicle.rho2_used,
                ext_remaining  = ext_bar - vehicle.ext_shift_used,
                tee            = False,
                time_limit     = time_limit,
                relax          = False,   # structure fixed -> tiny MIP
            )

            # ── 1. Duration-only recourse (structure fixed) ────────────────
            nom_sol = solve_horizon(fixed_plan=fixed_plan, plan_mode="fix",
                                    **common)

            if not nom_sol.get("feasible"):
                # ── 2. Repair: binaries may be ADDED but never removed ─────
                rep = solve_horizon(fixed_plan=fixed_plan, plan_mode="repair",
                                    **common)
                if rep.get("feasible"):
                    upd = _sol_to_plan_updates(rep["sol"], stop)
                    n_added = 0
                    added_detail = []
                    for g, e_new in upd.items():
                        e_old = plan_by_stop.get(g, dict(y=0, break_type=None,
                                                         rest_type=None))
                        for key in ("y", "break_type", "rest_type"):
                            if e_new.get(key) and not e_old.get(key):
                                n_added += 1
                                added_detail.append(f"{key}@{g}={e_new[key]}")
                        plan_by_stop[g] = e_new
                    events["repairs"].append(dict(
                        stop=stop, n_added=n_added, added=added_detail))
                    log_fn(f"  [{method_name}-REPAIR] stop {stop}: "
                           f"{n_added} activity(ies) added "
                           f"({', '.join(added_detail) or 'durations only'})")
                    nom_sol = rep
                    action  = _plan_entry(plan_by_stop.get(stop, {}))
                else:
                    # ── 3. Plan violation: hand over to the supervisor ─────
                    events["plan_violations"].append(dict(stop=stop))
                    log_fn(f"  [{method_name}-VIOLATION] stop {stop}: "
                           f"recourse and repair both infeasible")
                    nom_sol = None

            if supervised:
                action, itv = supervise_action(
                    full_data, stop, vehicle, action,
                    cv=cv, quantile=prune_quantile)
                if itv is not None:
                    events["interventions"].append(itv)
                    log_fn(f"  [SUPERVISOR] stop {stop}: {itv['fixes']} "
                           f"({', '.join(itv['checks'])})")
                    # the plan's durations no longer match the overridden
                    # action — fall back to minimum-duration execution
                    nom_sol = None
                    plan_by_stop[stop] = _plan_entry(action)

        events["decision_times"].append(time.perf_counter() - t0)

        D_act = float(D_real[stop])
        E_act = float(E_real[stop])

        if verbose:
            brk = action.get("break_type") or "---"
            rst = action.get("rest_type")  or "---"
            log_fn(f"  stop {stop:>3}  t={vehicle.t_arr:.3f}h "
                   f"soc={vehicle.e_arr:.0f}kWh  cd={vehicle.cd:.2f} "
                   f"sd={vehicle.sd:.2f} h={vehicle.h:.2f}  "
                   f"-> y={action.get('y', 0)} brk={brk} rst={rst}")

        vehicle.advance(action=action, D_next=D_act, E_next=E_act,
                        milp_sol=nom_sol)
        tracker.record_realisation(stop, D_act, E_actual=E_act)

        # A violation ends the run at this stop (BEHDV halt semantics).
        if vehicle.is_halted:
            break

    return vehicle, tracker, events


def run_plan_static(full_data: dict,
                    plan: list,
                    D_real: list,
                    E_real: list,
                    method_name: str,
                    log_fn,
                    cv: float = TRAVEL_TIME_CV_TARGET,
                    supervised: bool = False,
                    prune_quantile: float | None = GUARD_QUANTILE,
                    verbose: bool = True) -> tuple:
    """
    C3 — Execute a STATIC plan with NO online recourse (the robust plan).

    Both the binary structure AND the activity durations are committed
    offline.  At each stop the pre-computed durations are applied as-is: there
    is no duration re-optimisation and no add-only repair MILP.  If a realised
    draw breaks the fixed plan (an HoS breach or a stranding), BEHDV records
    the violation and the run is a ROBUST-PLAN FAILURE — it is reported, not
    repaired.  (Under the conservative box worst case the plan is feasible
    for every realization inside the box, and the multiplier support
    [XI_MIN, XI_MAX] is hard, so failures should not occur by construction.)

    The shared one-step safety supervisor (§5.1) applies only when
    supervised=True (non-default) — it is the identical guard used by every
    policy, not plan recourse; when it overrides the action, the pre-computed
    durations no longer match, so execution falls back to the
    minimum-duration heuristic for that stop.

    ``plan`` entries must carry the fixed durations (from
    twosp.extract_2sp_full_schedule): keys y, break_type, rest_type, tauc,
    taub, taur, tauq, sigma.

    Returns
    -------
    (vehicle, tracker, events) — events has empty repairs/plan_violations
    (there is no repair step); the robust-plan failure signal is the set of
    BEHDV violations surfaced by runner.finalize_run.
    """
    N       = full_data["N"]
    vehicle = BEHDV(full_data)
    tracker = ScenarioTracker(full_data)
    plan_by_stop = {p["i"]: p for p in plan if "i" in p}

    events = dict(repairs=[], plan_violations=[], interventions=[],
                  decision_times=[])

    for stop in range(N):
        t0 = time.perf_counter()

        if stop == 0:
            action  = dict(y=0, break_type=None, rest_type=None)
            nom_sol = None
        else:
            entry  = plan_by_stop.get(stop, dict(y=0, break_type=None,
                                                 rest_type=None))
            action = dict(y=int(entry.get("y", 0) or 0),
                          break_type=entry.get("break_type"),
                          rest_type=entry.get("rest_type"))

            # Build a fixed-duration "solution" so BEHDV applies the committed
            # durations verbatim (no re-optimisation).
            brk, rst = action["break_type"], action["rest_type"]
            nom_sol = dict(feasible=True, sol=[dict(
                i=0,
                taub=float(entry.get("taub", 0.0)),
                tauc=float(entry.get("tauc", 0.0)),
                taur=float(entry.get("taur", 0.0)),
                tauq=float(entry.get("tauq", 0.0)),
                sigma=int(entry.get("sigma", 0)),
                y=action["y"],
                b45=int(brk == "b45"), b15=int(brk == "b15"),
                b30=int(brk == "b30"),
                rho1=int(rst == "r1"), rho2=int(rst == "r2"),
                is_C=(stop in set(full_data["C"])),
                is_K=(stop in set(full_data["K"])),
            )])

            if supervised:
                action, itv = supervise_action(
                    full_data, stop, vehicle, action,
                    cv=cv, quantile=prune_quantile)
                if itv is not None:
                    events["interventions"].append(itv)
                    log_fn(f"  [SUPERVISOR] stop {stop}: {itv['fixes']} "
                           f"({', '.join(itv['checks'])})")
                    # overridden action -> committed durations no longer match
                    nom_sol = None

        events["decision_times"].append(time.perf_counter() - t0)

        D_act = float(D_real[stop])
        E_act = float(E_real[stop])

        if verbose:
            brk = action.get("break_type") or "---"
            rst = action.get("rest_type")  or "---"
            log_fn(f"  stop {stop:>3}  t={vehicle.t_arr:.3f}h "
                   f"soc={vehicle.e_arr:.0f}kWh  cd={vehicle.cd:.2f} "
                   f"sd={vehicle.sd:.2f} h={vehicle.h:.2f}  "
                   f"-> y={action.get('y', 0)} brk={brk} rst={rst}")

        vehicle.advance(action=action, D_next=D_act, E_next=E_act,
                        milp_sol=nom_sol)
        tracker.record_realisation(stop, D_act, E_actual=E_act)

        # A violation ends the run at this stop (BEHDV halt semantics).
        if vehicle.is_halted:
            break

    return vehicle, tracker, events
