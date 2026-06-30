"""
RO.py — Robust Optimisation counterpart of the BET scheduling MILP
===================================================================
Classic box-uncertainty robust counterpart: worst-case travel time and
energy are assumed on every leg simultaneously.

Mathematical formulation
------------------------
Uncertainty model (box set):
    D̃_i ∈ [D_i·(1−δ), D_i·(1+δ)]
    Ẽ_i ∈ {E_i^max}   (derived from worst-case speed)

Worst-case travel time on leg i (used in time propagation and HoS):
    D_wc[i] = D_nom[i] · (1 + δ)

Worst-case energy on leg i (used in SOC propagation):
    E_i^max = L_i · ECR( L_i / (D_i · (1−δ)) )
where L_i = km[i] is the leg distance (km) and ECR is the energy
consumption rate function.  Shorter travel time → higher speed → more
energy consumed.

The robust model replaces (D_nom, E_nom) with (D_wc, E_max) everywhere:

    min  t^a_N
    s.t. all original MILP constraints, with:
           • D_wc[i]  in time-propagation  (ta[i+1] = td[i] + D_wc[i])
           • D_wc[i]  in HoS accumulators  (cd, sd, sw propagation)
           • E_max[i] in SOC constraints

No dual variables, no budget parameter Γ.  This is the fully conservative
counterpart of the deterministic MILP: every leg is simultaneously at its
worst-case travel time and energy.

Simulation step
---------------
After solving, the schedule (y, breaks, rests, tauc, taub, taur at each
stop) is extracted and fed into the BEHDV simulator using D_real/E_real
as realised travel times.  The simulator respects the precomputed schedule
decisions rather than re-optimising.

Integration with the framework
-------------------------------
  from RO import run_ro
  results = run_ro(full_data, D_real, E_real, delta=0.20, verbose=True)

  Or via runner_dispatch:
    python runner_dispatch.py instances/RmediumCfew_7.json RO
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from typing import Optional

import pyomo.environ as pyo

from BEHDV     import BEHDV
from MILP      import (
    _declare_common_params, _declare_common_vars,
    _add_pwl_charging_constraints, _add_break_rest_constraints,
    _add_hos_accumulator_constraints, _add_v_sigma_constraints,
    _add_soc_constraints, _add_time_constraints,
    _solve_quiet, extract_solution, INFEASIBLE_PENALTY,
)
from scenarios import _ecr, ScenarioTracker
from settings  import V_NOM
from runner    import finalize_run
from plots     import plot_simulation_results


# ══════════════════════════════════════════════════════════════════════════════
# WORST-CASE PARAMETER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _worst_case_energies(full_data: dict, delta: float) -> dict:
    """
    Compute E_i^max = L_i · ECR( L_i / (D_i · (1−δ)) ) for each leg i.

    The worst case for energy is the fastest travel: D_nom · (1−δ).
    Higher speed → more energy (ECR is convex increasing in speed).
    """
    D_nom = full_data["D"]
    km    = full_data.get("km", {})
    N     = full_data["N"]

    E_max = {}
    for i in range(N):
        d_nom = D_nom.get(i, 0.0)
        L_km  = km.get(i, d_nom * V_NOM)
        d_min = max(d_nom * (1.0 - delta), 1e-6)
        v_max = L_km / d_min if d_min > 0 else 120.0
        E_max[i] = L_km * _ecr(v_max)
    return E_max


def _worst_case_times(full_data: dict, delta: float) -> dict:
    """
    Compute D_wc[i] = D_nom[i] · (1 + δ) for each leg i.

    The worst case for time is the slowest travel: every leg 100·δ% longer.
    """
    D_nom = full_data["D"]
    N     = full_data["N"]
    return {i: D_nom.get(i, 0.0) * (1.0 + delta) for i in range(N)}


# ══════════════════════════════════════════════════════════════════════════════
# ROBUST MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_robust_model(data: dict, delta: float) -> pyo.ConcreteModel:
    """
    Build the box-uncertainty robust counterpart of the full-route MILP.

    Both D and E in the data dict are replaced by their worst-case values
    before constructing the model, so all standard constraint blocks pick up
    the conservative parameters automatically.

    Parameters
    ----------
    data  : dict from instances.make_data()
    delta : uncertainty half-width δ

    Returns
    -------
    pyo.ConcreteModel — the robust Pyomo model
    """
    E_max = _worst_case_energies(data, delta)
    D_wc  = _worst_case_times(data, delta)

    ro_data = dict(data)
    ro_data["D"] = D_wc    # time propagation + HoS use worst-case travel time
    ro_data["E"] = E_max   # SOC constraints use worst-case energy

    m = pyo.ConcreteModel()

    N, C, K, R, Rseg, lb_t, ub_t = _declare_common_params(m, ro_data)
    _declare_common_vars(m)

    # Simple objective: minimise worst-case arrival time at destination
    m.obj = pyo.Objective(expr=m.ta[N], sense=pyo.minimize)

    # Initial conditions
    m.init_ta  = pyo.Constraint(expr=m.ta[0] == ro_data.get("T_START", 0.0))
    m.init_ea  = pyo.Constraint(expr=m.ea[0] == m.E0)
    m.init_cd  = pyo.Constraint(expr=m.cd[0] == 0)
    m.init_sd  = pyo.Constraint(expr=m.sd[0] == 0)
    m.init_sw  = pyo.Constraint(expr=m.sw[0] == 0)
    m.init_phi = pyo.Constraint(expr=m.phi[0] == 0)

    for v in [m.x_b45, m.x_b15, m.x_b30, m.rho1, m.rho2]:
        v[0].fix(0); v[N].fix(0)
    m.taub[0].fix(0); m.taur[0].fix(0)
    m.taub[N].fix(0); m.taur[N].fix(0)

    m.td_orig    = pyo.Constraint(expr=m.td[0] == m.ta[0])
    m.td_dest    = pyo.Constraint(expr=m.td[N] == m.ta[N])
    m.soc_nc_orig= pyo.Constraint(expr=m.ed[0] == m.ea[0])
    m.soc_nc_dest= pyo.Constraint(expr=m.ed[N] == m.ea[N])

    # All constraint blocks use worst-case D and E via ro_data
    _add_soc_constraints(m, N, ro_data)
    _add_time_constraints(m, N)            # uses m.D_nom = D_wc
    _add_v_sigma_constraints(m, m.M_big)
    _add_pwl_charging_constraints(m, K, R, Rseg)
    _add_break_rest_constraints(m, N, ro_data["I"], set(K), m.M_big, rho2_limit=3)
    _add_hos_accumulator_constraints(m, N, ro_data["I"], set(C), set(K),
                                     ro_data["S"], m.M_drv, m.M_sd, m.M_sw,
                                     m.TK)   # D_nom already worst-case; no D_wc override needed

    return m


# ══════════════════════════════════════════════════════════════════════════════
# SOLVE THE ROBUST MODEL
# ══════════════════════════════════════════════════════════════════════════════

def solve_robust(model: pyo.ConcreteModel,
                 time_limit: int = 2 * 3600,
                 mip_gap: float  = 0.005,
                 tee: bool       = True) -> tuple[dict, str]:
    """
    Solve the robust model with Gurobi.

    Returns
    -------
    info   : dict with keys: feasible, optimal, obj, gap, status
    status : str — Gurobi termination condition
    """
    solver = pyo.SolverFactory("gurobi")
    solver.options["MIPGap"]    = mip_gap
    solver.options["TimeLimit"] = time_limit

    try:
        res    = _solve_quiet(solver, model, tee=tee)
        status = str(res.solver.termination_condition)
    except RuntimeError:
        return dict(feasible=False, optimal=False,
                    obj=float("inf"), gap=float("inf"), status="infeasible"), "infeasible"

    feasible   = status in ("optimal", "feasible", "maxTimeLimit")
    is_optimal = status == "optimal"

    if not feasible:
        return dict(feasible=False, optimal=False,
                    obj=float("inf"), gap=float("inf"), status=status), status

    obj_val = pyo.value(model.obj)
    try:
        import re
        msg   = str(res.solver.termination_condition_message)
        m_gap = re.search(r"gap[^0-9]*([0-9.e+-]+)%", msg, re.I)
        gap   = float(m_gap.group(1)) / 100 if m_gap else (0.0 if is_optimal else float("nan"))
    except Exception:
        gap = 0.0 if is_optimal else float("nan")

    return dict(feasible=True, optimal=is_optimal,
                obj=obj_val, gap=gap, status=status), status


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULE EXTRACTOR (maps RO solution to per-stop action dicts)
# ══════════════════════════════════════════════════════════════════════════════

def _ro_sol_to_actions(sol: list) -> list[dict]:
    """
    Convert extract_solution() output to a list of action dicts compatible
    with BEHDV.advance() and finalize_run().
    """
    actions = []
    for s in sol:
        brk = ("b45" if s["b45"] else
               "b15" if s["b15"] else
               "b30" if s["b30"] else None)
        rst = ("r1" if s["rho1"] else
               "r2" if s["rho2"] else None)
        actions.append(dict(
            y          = s["y"],
            break_type = brk,
            rest_type  = rst,
            taub       = s["taub"],
            tauc       = s["tauc"],
            taur       = s["taur"],
            tauq       = s["tauq"],
            sigma      = s.get("sigma", 0),
        ))
    return actions


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION STEP: execute RO schedule on actual realisations
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_ro_schedule(full_data: dict, sol: list,
                          D_real: list, E_real: list,
                          verbose: bool, log_fn) -> tuple[BEHDV, ScenarioTracker, list]:
    """
    Execute the RO schedule on the precomputed uncertainty realisation.

    The schedule decisions (y, breaks, rests, tauc, taub, taur) are taken
    directly from the RO optimal solution.  Only the actual travel times
    (D_real) and energies (E_real) differ from the model's assumptions.
    """
    N       = full_data["N"]
    vehicle = BEHDV(full_data)
    tracker = ScenarioTracker(full_data)
    actions = _ro_sol_to_actions(sol)

    for stop in range(N):
        action = actions[stop] if stop < len(actions) else \
                 dict(y=0, break_type=None, rest_type=None,
                      taub=0.0, tauc=0.0, taur=0.0, tauq=0.0)

        mock_sol = dict(
            feasible = True,
            sol = [dict(
                i    = 0,
                taub = action["taub"], tauc = action["tauc"],
                taur = action["taur"], tauq = action["tauq"],
                y    = action["y"],
                sigma= action.get("sigma", 0),
                b45  = int(action["break_type"] == "b45"),
                b15  = int(action["break_type"] == "b15"),
                b30  = int(action["break_type"] == "b30"),
                rho1 = int(action["rest_type"]  == "r1"),
                rho2 = int(action["rest_type"]  == "r2"),
                is_C = stop in set(full_data["C"]),
                is_K = stop in set(full_data["K"]),
            )],
        )

        D_act = float(D_real[stop])
        E_act = float(E_real[stop])

        y   = action["y"]
        brk = action["break_type"] or "---"
        rst = action["rest_type"]  or "---"
        stop_type = ("CS"   if stop in set(full_data["K"]) else
                     "CUST" if stop in set(full_data["C"]) else
                     "ORIG" if stop == 0 else "INT")

        log_fn(f"\n  stop {stop:>3} ({stop_type})"
               f"  t={vehicle.t_arr:.3f}h  soc={vehicle.e_arr:.0f}kWh"
               f"  cd={vehicle.cd:.2f}  sd={vehicle.sd:.2f}  sw={vehicle.sw:.2f}")
        log_fn(f"     -> y={y}  brk={brk}  rst={rst}"
               f"  tauc={action['tauc']*60:.0f}m"
               f"  taub={action['taub']*60:.0f}m"
               f"  taur={action['taur']*60:.0f}m"
               f"  D_act={D_act:.3f}h  E_act={E_act:.1f}kWh")

        vehicle.advance(action=action, D_next=D_act, E_next=E_act,
                        milp_sol=mock_sol)
        tracker.record_realisation(stop, D_act, E_actual=E_act)

    return vehicle, tracker, []


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_ro(full_data: dict,
           D_real: list,
           E_real: list,
           delta: float      = 0.20,
           time_limit: int   = 2 * 3600,
           mip_gap: float    = 0.005,
           tee: bool         = True,
           verbose: bool     = True,
           run_id: str       = None,
           oracle_tee: bool  = True,
           **kwargs) -> dict:
    """
    Solve the box-uncertainty robust counterpart and simulate on D_real/E_real.

    Parameters
    ----------
    full_data   : instance dict (from instance_io.load_instance_json)
    D_real      : list[float] — precomputed realised travel times (h), length N
    E_real      : list[float] — precomputed realised energies (kWh), length N
    delta       : uncertainty half-width δ (default 0.20)
    time_limit  : solver wall-clock limit in seconds (default 2h)
    mip_gap     : MIP relative gap tolerance (default 0.5%)
    tee         : show Gurobi solver output (default True)
    verbose     : print per-stop trajectory to stdout
    run_id      : override auto-generated run_id
    oracle_tee  : show Gurobi output in oracle solve (default True)
    **kwargs    : ignored (absorbs legacy Gamma= calls from callers)

    Returns
    -------
    dict — canonical results dict (same schema as run_simulation / run_greedy)
    """
    t_wall_start = time.perf_counter()
    N            = full_data["N"]
    T_START      = full_data.get("T_START", 8.0)
    label        = full_data.get("label", "ro")
    title        = full_data.get("title", "inst")

    assert len(D_real) == N, f"D_real length {len(D_real)} != N={N}"
    assert len(E_real) == N, f"E_real length {len(E_real)} != N={N}"

    for d in ("logs", "figures", "solutions"):
        os.makedirs(d, exist_ok=True)
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{title}_RO_d{int(delta*100)}_{ts}"
    paths = dict(
        log = os.path.join("logs",      f"{run_id}.txt"),
        fig = os.path.join("figures",   f"{run_id}.png"),
        sol = os.path.join("solutions", f"{run_id}.json"),
        scn = os.path.join("logs",      f"{run_id}_scenarios.json"),
    )
    log = open(paths["log"], "w", encoding="utf-8")

    def _p(msg):
        if verbose: print(msg)
        try: print(msg, file=log)
        except Exception: pass

    _p("=" * 65)
    _p(f"  RO SOLVE START   ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    _p(f"  Instance : {label}   run_id={run_id}")
    _p(f"  Route    : {N} stops  departure={T_START:.0f}:00")
    _p(f"  Settings : δ={delta:.0%}  gap={mip_gap:.1%}  time_limit={time_limit}s")
    _p(f"  Model    : box uncertainty — worst case assumed on every leg")
    _p("=" * 65)

    _p(f"\n  Building robust model...")
    model = build_robust_model(full_data, delta=delta)

    _p(f"  Solving...")
    t_solve = time.perf_counter()
    info, status = solve_robust(model, time_limit=time_limit,
                                mip_gap=mip_gap, tee=tee)
    t_solve = time.perf_counter() - t_solve

    _p(f"  Status   : {status}  ({t_solve:.1f}s)")

    if not info["feasible"]:
        _p("  No feasible solution found — aborting.")
        log.close()
        return dict(feasible=False, status=status,
                    total_time=float("inf"), wall_clock=time.perf_counter()-t_wall_start)

    import math
    gap_str = (f"{info['gap']:.2%}" if not math.isnan(info.get("gap", float("nan")))
               else "n/a")
    _p(f"  RO objective : {info['obj']:.3f} h  (gap={gap_str},"
       f"  optimal={info['optimal']})")

    sol = extract_solution(model, full_data)
    _p(f"  RO schedule  : {sum(1 for s in sol if s['y'])} charges,"
       f"  {sum(1 for s in sol if s['b45'] or s['b15'] or s['b30'])} breaks,"
       f"  {sum(1 for s in sol if s['rho1'] or s['rho2'])} rests")

    _p(f"\n  Simulating RO schedule on precomputed realisation...")
    vehicle, tracker, scores_log = _simulate_ro_schedule(
        full_data, sol, D_real, E_real, verbose=verbose, log_fn=_p)

    wall_elapsed = time.perf_counter() - t_wall_start
    arr_h        = vehicle.t_arr
    _p(f"\n{'='*65}")
    _p(f"  RO SIMULATION COMPLETE")
    _p(f"  Arrival (absolute) : {arr_h:.3f} h  ({int(arr_h):02d}:{int((arr_h%1)*60):02d})")
    _p(f"  Travel duration    : {arr_h - T_START:.3f} h")
    _p(f"  Solve time         : {t_solve:.1f} s")
    _p(f"  Wall-clock         : {wall_elapsed:.1f} s")
    _p("=" * 65)

    results = finalize_run(
        vehicle     = vehicle,
        full_data   = full_data,
        tracker     = tracker,
        run_id      = run_id,
        paths       = paths,
        timing      = dict(wall_clock=wall_elapsed, T_START=T_START),
        log_fh      = log,
        verbose     = verbose,
        oracle_tee  = oracle_tee,
        scores_log  = scores_log,
        method_meta = dict(
            method     = "RO",
            delta      = delta,
            ro_obj     = info["obj"],
            ro_gap     = info.get("gap"),
            ro_optimal = info["optimal"],
            ro_status  = status,
            solve_time = t_solve,
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from instance_io import load_instance_json

    # Usage: python RO.py <json_file> [delta] [time_limit]
    json_file  = sys.argv[1] if len(sys.argv) > 1 else None
    delta      = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    time_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 2 * 3600

    if json_file is None:
        print("Usage: python RO.py <json_file> [delta] [time_limit_s]")
        sys.exit(1)

    full_data, D_real, E_real, _ = load_instance_json(json_file)

    results = run_ro(
        full_data,
        D_real     = D_real,
        E_real     = E_real,
        delta      = delta,
        time_limit = time_limit,
        tee        = True,
        verbose    = True,
        oracle_tee = True,
    )

    print(f"\n  RO arrival  : {results['total_time']:.3f} h")
    print(f"  Wall clock  : {results['wall_clock']:.1f} s")
    print(f"  Figure      : {results['fig_path']}")
    print(f"  Solution    : {results['sol_path']}")
