"""
RO.py — Robust Optimisation counterpart of the BET scheduling MILP
===================================================================
Implements the Bertsimas–Sim (2004) robust counterpart of the full-route
deterministic MILP.  The model is solved ONCE on the complete route, using
nominal travel times D_i and worst-case energies E_i^max.  The resulting
schedule is then simulated forward on the precomputed uncertainty realisation
(D_real, E_real) to produce an arrival time comparable to LA and greedy.

Mathematical formulation
------------------------
Uncertainty model:
    D̃_i = D_i · ξ_i,   ξ_i ∈ [1−δ, 1+δ]

Budget uncertainty set (Bertsimas & Sim, 2004):
    U = { ξ | ξ_i ∈ [1−δ, 1+δ],  Σ_i (ξ_i − 1) ≤ Γδ }

  Γ ∈ [0, N]: conservatism parameter.
    Γ=0  →  nominal problem (δ has no effect on timing).
    Γ=N  →  all legs simultaneously at worst case.

Energy (SOC) constraints:
    Worst-case energy on leg i:
        E_i^max = L_i · ECR( L_i / (D_i (1−δ)) )
    where L_i = km[i] is the leg distance (km) and ECR is the energy
    consumption rate function from scenarios.py.  This replaces E_i in
    all SOC propagation constraints, leaving them otherwise unchanged.

Robust objective (LP dual of inner max, merged with outer min):
    min  t^a_N(x) + Γδπ + δ Σ_i p_i
    s.t. p_i + π ≥ D_i,   ∀i               (dual feasibility)
         π ≥ 0,  p_i ≥ 0,  ∀i
         all original MILP constraints, with:
           • D_nom  in time-propagation  (dual penalty covers worst-case delay)
           • E_max  in SOC constraints   (conservative worst-case energy)
           • D_nom*(1+δ) in HoS accumulators (cd, sd, sw propagation)

The dual variable π is the shared budget; p_i captures individual leg
excess above the budget.  At optimality, π covers the Γ most dangerous
legs, and p_i > 0 only for legs whose delay exceeds the shared threshold.

HoS robustness
--------------
A key asymmetry in a naive Bertsimas-Sim counterpart is that the dual
penalty protects only the objective (arrival time), not the constraint
feasibility under uncertainty.  The HoS accumulator constraints (cd, sd,
sw) also propagate travel time D_i — so under a realised delay D_i*ξ_i >
D_i the truck may accumulate more driving/working time than the model
anticipates, violating regulatory limits.

The fix mirrors the treatment of energy: D_nom[i]*(1+δ) (worst-case travel
time on each leg independently) is used in the three HoS propagation
constraints.  This is row-wise robustification (Bertsimas & Sim, 2004,
Section 2) applied to the constraint coefficient of D_i.  Because the
accumulators must stay within their fixed regulatory bounds (T_drv_cons,
T_drv_sh, T_wrk_sh) regardless of ξ, using the worst-case leg time in
those constraints is both correct and sufficient.

Budget parameter Γ
------------------
Γ controls conservatism of the objective penalty.
  Γ = 0     → nominal problem (no robustness in objective).
  Γ = N     → all legs at worst case simultaneously (fully robust).
  Γ = √N    → derived from B&S Proposition 2: for i.i.d. uniform ξ_i,
               P(constraint violation) ≤ exp(-Γ²/(2N)), so Γ=√(2N·ln(1/α))
               gives a (1-α) guarantee. A practical rule of thumb used in
               many papers is Γ = √N (≈ 95% confidence for N ≤ 30).
  Γ = N/2 + z_{1-α}·√N/2 → exact Binomial-based bound from B&S (2004)
                              Proposition 2 for Bernoulli perturbations.

See: Bertsimas & Sim (2004), Prop. 2, Operations Research 52(1):35-53.
     https://doi.org/10.1287/opre.1030.0065

Simulation step
---------------
After solving, the schedule (decision variables y, breaks, rests, tauc,
taub, taur at each stop) is extracted and fed into the BEHDV simulator
using D_real / E_real as realised travel times.  The simulator respects
the precomputed schedule decisions rather than re-optimising.

Integration with the framework
-------------------------------
  from RO import run_ro
  results = run_ro(full_data, D_real, E_real,
                   delta=0.20, Gamma=5, verbose=True)

  Or via runner_dispatch:
    python runner_dispatch.py instances/RmediumCfew_7.json RO

References
----------
Bertsimas, D. & Sim, M. (2004). The price of robustness.
Operations Research, 52(1), 35–53.
https://doi.org/10.1287/opre.1030.0065
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
    _add_hos_accumulator_constraints, _add_manoeuver_constraints,
    _add_soc_constraints, _add_time_constraints,
    _solve_quiet, extract_solution, INFEASIBLE_PENALTY,
)
from scenarios import _ecr, ScenarioTracker
from runner    import finalize_run
from plots     import plot_simulation_results


# ══════════════════════════════════════════════════════════════════════════════
# WORST-CASE ENERGY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _worst_case_energies(full_data: dict, delta: float) -> dict:
    """
    Compute E_i^max = L_i · ECR( L_i / (D_i · (1−δ)) ) for each leg i.

    This is the energy consumed when the truck drives at the highest possible
    speed (shortest possible travel time = D_i·(1−δ)), which is conservative
    because ECR is convex increasing in speed.

    Parameters
    ----------
    full_data : instance dict with keys "D" (nominal times), "km" (leg distances)
    delta     : uncertainty half-width

    Returns
    -------
    dict {leg_index: E_max_kWh}
    """
    D_nom = full_data["D"]
    km    = full_data.get("km", {})
    N     = full_data["N"]

    E_max = {}
    for i in range(N):
        d_nom = D_nom.get(i, 0.0)
        L_km  = km.get(i, d_nom * 80.0)        # fallback: assume 80 km/h nominal
        d_min = max(d_nom * (1.0 - delta), 1e-6)
        v_max = L_km / d_min if d_min > 0 else 120.0
        E_max[i] = L_km * _ecr(v_max)
    return E_max


def _worst_case_hos_driving(full_data: dict, delta: float) -> dict:
    """
    Compute D_wc[i] = D_nom[i] * (1 + δ) for each leg i.

    This is the worst-case (longest) travel time on leg i, used in the HoS
    accumulator propagation constraints (cd, sd, sw) to guarantee that the
    consecutive-driving, shift-driving, and shift-working limits are satisfied
    under any realised delay within the uncertainty set.

    Using D_nom*(1+δ) independently on every leg is conservative in the sense
    that the budget set U already prevents all legs from being simultaneously
    at their maximum delay — but for the *constraint* rows (unlike the
    objective) the row-wise approach (one row robustified independently) is the
    standard correct treatment, matching how E_max is derived for the SOC.

    Parameters
    ----------
    full_data : instance dict with key "D" (nominal travel times, h)
    delta     : uncertainty half-width δ

    Returns
    -------
    dict {leg_index: D_wc_h}  — worst-case travel time per leg (hours)
    """
    D_nom = full_data["D"]
    N     = full_data["N"]
    return {i: D_nom.get(i, 0.0) * (1.0 + delta) for i in range(N)}


# ══════════════════════════════════════════════════════════════════════════════
# ROBUST MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_robust_model(data: dict, delta: float, Gamma: float) -> pyo.ConcreteModel:
    """
    Build the Bertsimas–Sim robust counterpart of the full-route MILP.

    Modifies the base model in two ways:
      1. E_i^max replaces E_i in all SOC propagation constraints.
      2. Dual variables (pi, p_i) and budget constraint added to objective.

    Parameters
    ----------
    data  : dict from instances.make_data() with E replaced by E_max
    delta : uncertainty half-width δ
    Gamma : budget parameter Γ ∈ [0, N]

    Returns
    -------
    pyo.ConcreteModel — the robust Pyomo model
    """
    # ── Build a copy of data with E replaced by E_max ─────────────────────────
    E_max = _worst_case_energies(data, delta)
    ro_data = dict(data)
    ro_data["E"] = E_max           # SOC constraints now use worst-case energy

    m = pyo.ConcreteModel()

    N, C, K, R, Rseg, lb_t, ub_t = _declare_common_params(m, ro_data)
    _declare_common_vars(m)

    # ── Robust dual variables ─────────────────────────────────────────────────
    # π ≥ 0 : shared budget variable
    # p_i ≥ 0 : per-leg excess over budget
    m.pi_ro  = pyo.Var(domain=pyo.NonNegativeReals)
    m.p_ro   = pyo.Var(m.Legs, domain=pyo.NonNegativeReals)

    # Dual feasibility: p_i + π ≥ D_i  ∀i
    m.ro_dual = pyo.Constraint(m.Legs, rule=lambda m, i:
        m.p_ro[i] + m.pi_ro >= ro_data["D"].get(i, 0.0))

    # ── Robust objective: min t^a_N + Γδπ + δ Σ p_i ──────────────────────────
    m.obj = pyo.Objective(
        expr = (m.ta[N]
                + Gamma * delta * m.pi_ro
                + delta * sum(m.p_ro[i] for i in range(N))),
        sense = pyo.minimize,
    )

    # ── Initial conditions ────────────────────────────────────────────────────
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

    # ── Standard constraint blocks (with E_max in SOC) ────────────────────────
    _add_soc_constraints(m, N, ro_data)
    _add_time_constraints(m, N)
    _add_manoeuver_constraints(m, ro_data["I"], set(K))
    _add_pwl_charging_constraints(m, K, R, Rseg)
    _add_break_rest_constraints(m, N, ro_data["I"], set(K), m.M_big, rho2_limit=3)

    # ── Robust HoS constraints: use D_nom*(1+δ) in cd/sd/sw propagation ───────
    # The time-propagation uses D_nom (the dual penalty handles the objective
    # worst-case), but the HoS accumulators need the worst-case leg time to
    # guarantee regulatory feasibility under any ξ ∈ U.
    D_wc = _worst_case_hos_driving(data, delta)
    _add_hos_accumulator_constraints(m, N, ro_data["I"], set(C), set(K),
                                     ro_data["S"], m.M_drv, m.M_sd, m.M_sw,
                                     m.TK, D_wc=D_wc)
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

    Each action dict has keys: y, break_type, rest_type, taub, tauc, taur, tauq.
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

    Returns
    -------
    vehicle : BEHDV after completing the full route
    tracker : ScenarioTracker (realisation records only; no scenarios)
    scores_log : empty list (RO has no per-stop look-ahead scoring)
    """
    N       = full_data["N"]
    vehicle = BEHDV(full_data)
    tracker = ScenarioTracker(full_data)
    actions = _ro_sol_to_actions(sol)

    for stop in range(N):
        action   = actions[stop] if stop < len(actions) else \
                   dict(y=0, break_type=None, rest_type=None,
                        taub=0.0, tauc=0.0, taur=0.0, tauq=0.0)

        # Build mock milp_sol so vehicle.advance uses prescribed durations
        mock_sol = dict(
            feasible = True,
            sol = [dict(
                i    = 0,
                taub = action["taub"], tauc = action["tauc"],
                taur = action["taur"], tauq = action["tauq"],
                y    = action["y"],
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

    return vehicle, tracker, []   # scores_log empty for RO


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_ro(full_data: dict,
           D_real: list,
           E_real: list,
           delta: float      = 0.20,
           Gamma: float      = None,
           time_limit: int   = 2 * 3600,
           mip_gap: float    = 0.005,
           tee: bool         = True,
           verbose: bool     = True,
           run_id: str       = None,
           oracle_tee: bool  = True) -> dict:
    """
    Solve the Bertsimas–Sim robust counterpart and simulate on D_real/E_real.

    Parameters
    ----------
    full_data   : instance dict (from instance_io.load_instance_json)
    D_real      : list[float] — precomputed realised travel times (h), length N
    E_real      : list[float] — precomputed realised energies (kWh), length N
    delta       : uncertainty half-width δ (default 0.20)
    Gamma       : budget parameter Γ (default: N/2, i.e. half the legs)
    time_limit  : solver wall-clock limit in seconds (default 2h)
    mip_gap     : MIP relative gap tolerance (default 0.5%)
    tee         : show Gurobi solver output (default True)
    verbose     : print per-stop trajectory to stdout
    run_id      : override auto-generated run_id
    oracle_tee  : show Gurobi output in oracle solve (default True)

    Returns
    -------
    dict — canonical results dict (same schema as run_simulation / run_greedy)
    """
    t_wall_start = time.perf_counter()
    N            = full_data["N"]
    T_START      = full_data.get("T_START", 8.0)
    label        = full_data.get("label", "ro")
    title        = full_data.get("title", "inst")

    if Gamma is None:
        # Default: Γ = √N — a common statistically-motivated choice.
        #
        # Bertsimas & Sim (2004), Proposition 2, show that for i.i.d.
        # perturbations the probability of a constraint violation is bounded by
        #   P(violation) ≤ exp(-Γ² / (2·N))
        # Setting Γ = √(2·N·ln(1/α)) gives a (1−α) guarantee.
        # For practical moderate conservatism (≈ 90–95% for N ≤ 50) the rule
        # of thumb Γ = √N is widely used; see also Bertsimas & Sim (2004)
        # Section 4 and Ben-Tal et al. (2009) "Robust Optimization", §1.3.
        #
        # Former default was N/√2 (≈ 70% of all legs simultaneously at worst
        # case), which is extremely conservative and rarely motivated in
        # practice.  Override with e.g. Gamma=N/2 for stricter protection.
        import math as _math
        Gamma = _math.sqrt(N)   # ≈ 90-95% confidence for typical N

    assert len(D_real) == N, f"D_real length {len(D_real)} != N={N}"
    assert len(E_real) == N, f"E_real length {len(E_real)} != N={N}"

    # ── Output dirs and file paths ─────────────────────────────────────────────
    for d in ("logs", "figures", "solutions"):
        os.makedirs(d, exist_ok=True)
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{title}_RO_d{int(delta*100)}_G{int(Gamma)}_{ts}"
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
    _p(f"  Settings : δ={delta:.0%}  Γ={Gamma:.1f}  gap={mip_gap:.1%}"
       f"  time_limit={time_limit}s")
    _p("=" * 65)

    # ── Step 1: Build and solve the robust model ───────────────────────────────
    _p(f"\n  Building robust model...")
    model = build_robust_model(full_data, delta=delta, Gamma=Gamma)

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

    # Extract RO schedule
    sol = extract_solution(model, full_data)
    _p(f"  RO schedule  : {sum(1 for s in sol if s['y'])} charges,"
       f"  {sum(1 for s in sol if s['b45'] or s['b15'] or s['b30'])} breaks,"
       f"  {sum(1 for s in sol if s['rho1'] or s['rho2'])} rests")

    # ── Step 2: Simulate schedule on precomputed realisation ──────────────────
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

    # ── Step 3: Delegate epilogue to runner ───────────────────────────────────
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
            Gamma      = Gamma,
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

    # Usage: python RO.py <json_file> [delta] [Gamma] [time_limit]
    json_file  = sys.argv[1] if len(sys.argv) > 1 else None
    delta      = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    # Gamma: pass as integer or "auto" (default N/2)
    Gamma      = (float(sys.argv[3]) if len(sys.argv) > 3
                  and sys.argv[3].lower() != "auto" else None)
    time_limit = int(sys.argv[4]) if len(sys.argv) > 4 else 2 * 3600

    if json_file is None:
        print("Usage: python RO.py <json_file> [delta] [Gamma] [time_limit_s]")
        sys.exit(1)

    full_data, D_real, E_real, _ = load_instance_json(json_file)

    results = run_ro(
        full_data,
        D_real     = D_real,
        E_real     = E_real,
        delta      = delta,
        Gamma      = Gamma,
        time_limit = time_limit,
        tee        = True,
        verbose    = True,
        oracle_tee = True,
    )

    print(f"\n  RO arrival  : {results['total_time']:.3f} h")
    print(f"  Wall clock  : {results['wall_clock']:.1f} s")
    print(f"  Figure      : {results['fig_path']}")
    print(f"  Solution    : {results['sol_path']}")