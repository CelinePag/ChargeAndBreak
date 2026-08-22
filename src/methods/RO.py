"""
RO.py — Conservative (box) robust plan, offline, no recourse
============================================================
The robust plan is the WORST-CASE counterpart of the deterministic MILP:
every leg is assumed to be at its worst case simultaneously (Soyster-style
box uncertainty, no budget).  The model, decision variables, and objective
are identical to the full-route MILP; only the leg parameters differ:

    D_i  =  D_nom_i · XI_MAX                       (slowest travel time)
    E_i  =  L_i · ECR(L_i / (D_nom_i · XI_MIN))    (fastest-speed energy)

The box [XI_MIN, XI_MAX]^N is the HARD support of the shifted-lognormal
multiplier (settings S5): XI_MIN = V_NOM/90 from the HGV speed limiter,
XI_MAX = V_NOM/50 from the leg-average congestion floor.  Each constraint
faces its own worst extreme: the time/HoS accumulators bind at the slow
extreme, the SOC constraints at the fast (energy-hungry) extreme.  This is
the classical constraint-wise robust counterpart — deliberately conservative
(no single realization attains both extremes at once), and feasible for
EVERY realization in the box by construction.

Execution (no recourse)
-----------------------
The solution is executed AS IS: both the binary structure (charge / break /
rest decisions) and the activity durations are committed offline and applied
verbatim at each stop (recourse.run_plan_static).  There is no online
re-optimization, no duration recourse, and no add-only repair step.  Because
the plan is sized for the worst case, any realization inside the box is
feasible; the price is conservatism (long planned durations), which is the
comparison against the MILP / LA / 2SP policies.

Integration
-----------
  from src.methods.RO import run_ro
  results = run_ro(full_data, D_real, E_real)

  Or via runner_dispatch:
    python -m src.simulation.runner_dispatch instances/RmediumCfewTmedium_7.json RO
"""

from __future__ import annotations

import datetime
import os
import sys
import time

import pyomo.environ as pyo

from src.methods.recourse  import run_plan_static
from src.simulation.scenarios import _ecr
from src.settings  import V_NOM, XI_MIN, XI_MAX, TRAVEL_TIME_CV_TARGET, GUARD_QUANTILE
from src.simulation.runner    import finalize_run
from src.plot.plots     import plot_simulation_results
from src.methods import twosp as _twosp
from src import paths as _paths


# ══════════════════════════════════════════════════════════════════════════════
# WORST-CASE (BOX) SCENARIO
# ══════════════════════════════════════════════════════════════════════════════

def _box_scenario(full_data: dict) -> dict:
    """
    Constraint-wise worst case of the support box [XI_MIN, XI_MAX]^N as a
    single scenario: slowest times D·XI_MAX AND fastest-speed energies
    E(L/(D·XI_MIN)) — the truck pinned at the 90 km/h limiter.  Each
    constraint of the deterministic model then holds at its own worst
    extreme, so the plan is feasible for every realization in the box.
    """
    N     = full_data["N"]
    D_nom = full_data["D"]
    km    = full_data.get("km", {})
    D_s, E_s = {}, {}
    for i in range(N):
        d_nom = D_nom.get(i, 0.0)
        L     = km.get(i, d_nom * V_NOM)
        D_s[i] = d_nom * XI_MAX
        d_min  = max(d_nom * XI_MIN, 1e-9)
        E_s[i] = L * _ecr(L / d_min)
    return dict(D=D_s, E=E_s)


def _plan_from_model(model, full_data: dict) -> list[dict]:
    """Extract the committed plan STRUCTURE + fixed DURATIONS (static plan)."""
    return _twosp.extract_2sp_full_schedule(model, full_data)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_ro(full_data: dict,
           D_real: list,
           E_real: list,
           cv: float         = TRAVEL_TIME_CV_TARGET,
           time_limit: int   = 2 * 3600,
           mip_gap: float    = 0.005,
           heuristics: float | None = 0.2,
           mip_focus: int | None    = None,
           tee: bool         = True,
           verbose: bool     = True,
           run_id: str       = None,
           oracle_tee: bool  = True,
           supervised: bool  = False,
           prune_quantile: float | None = GUARD_QUANTILE,
           **kwargs) -> dict:
    """
    Solve the conservative box robust plan (worst-case MILP counterpart) and
    execute it AS IS on D_real/E_real — no online recourse of any kind.

    Parameters
    ----------
    full_data     : instance dict (from instance_io.load_instance_json)
    D_real        : list[float] — precomputed realised travel times (h)
    E_real        : list[float] — precomputed realised energies (kWh)
    cv            : CV of the travel-time multiplier — the box corners are the
                    fixed support bounds [XI_MIN, XI_MAX] regardless of cv;
                    cv is only forwarded to the S1 supervisor guard
    supervised    : apply the shared S1 safety supervisor during execution
                    (the identical guard used by every policy — NOT recourse;
                    default False — a broken plan is recorded, not rescued)
    **kwargs      : ignored (absorbs legacy caller arguments, e.g. Gamma,
                    legacy_box, n_random_scen from the old budgeted variant)

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

    _paths.ensure_dirs()
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{title}_RO_box_{ts}"
    paths = dict(
        log = _paths.log_out(f"{run_id}.txt"),
        fig = _paths.figure_out(f"{run_id}.png"),
        sol = _paths.solution_out(f"{run_id}.json"),
        scn = _paths.log_out(f"{run_id}_scenarios.json"),
        gurobi = _paths.log_out(f"{run_id}_gurobi.log"),
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
    _p(f"  Model    : conservative box RO (all legs at worst case, "
       f"xi in [{XI_MIN:.3f}, {XI_MAX:.3f}]), MILP counterpart, no recourse")
    _p("=" * 65)

    # ── Step 1: single worst-case scenario (all legs at their extremes) ───────
    scen_set = [_box_scenario(full_data)]

    # ── Step 2: deterministic solve on the worst-case parameters ──────────────
    # With one scenario, the epigraph max + shared durations reduce EXACTLY to
    # the deterministic full-route MILP evaluated at the box worst case.
    t0 = time.perf_counter()
    model = _twosp.build_2sp_model(full_data, scen_set, objective="max",
                                   share_durations=True)
    info, status = _twosp.solve_2sp(model, time_limit=time_limit,
                                    mip_gap=mip_gap, tee=tee,
                                    heuristics=heuristics, mip_focus=mip_focus,
                                    log_file=paths["gurobi"])
    t_solve_total = time.perf_counter() - t0
    _p(f"  Solve status={status}  ({t_solve_total:.1f}s)")

    if not info["feasible"]:
        _p("  No feasible robust plan under the box worst case — aborting.")
        log.close()
        return dict(feasible=False, status=status,
                    total_time=float("inf"),
                    wall_clock=time.perf_counter() - t_wall_start)

    theta = info["obj"]
    plan  = _plan_from_model(model, full_data)   # structure + fixed durations
    _p(f"  theta={theta:.3f}h  "
       f"plan: {sum(1 for e in plan if e['y'])} chg / "
       f"{sum(1 for e in plan if e['break_type'])} brk / "
       f"{sum(1 for e in plan if e['rest_type'])} rst")
    _p(f"\n  RO objective (worst-case arrival) : {theta:.3f} h   "
       f"solve {t_solve_total:.1f}s")

    # ── Step 3: execute the FIXED plan AS IS (static, no recourse) ────────────
    _p(f"\n  Executing RO plan AS IS (static, no recourse)...")
    vehicle, tracker, events = run_plan_static(
        full_data      = full_data,
        plan           = plan,
        D_real         = D_real,
        E_real         = E_real,
        method_name    = "RO",
        log_fn         = _p,
        cv             = cv,
        supervised     = supervised,
        prune_quantile = prune_quantile,
        verbose        = verbose,
    )
    n_fail = len(getattr(vehicle, "violations", []))
    _p(f"  Robust-plan failures (raw): {n_fail} violation(s); "
       f"{len(events['interventions'])} supervisor intervention(s)")

    wall_elapsed = time.perf_counter() - t_wall_start
    arr_h        = vehicle.t_arr
    _p(f"\n{'='*65}")
    _p(f"  RO SIMULATION COMPLETE")
    _p(f"  Arrival (absolute) : {arr_h:.3f} h  "
       f"({int(arr_h):02d}:{int((arr_h%1)*60):02d})")
    _p(f"  Travel duration    : {arr_h - T_START:.3f} h")
    _p(f"  Solve time         : {t_solve_total:.1f} s")
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
        scores_log  = [],
        events      = events,
        method_meta = dict(
            method       = "RO",
            cv           = cv,
            gamma        = "box",
            n_scenarios  = 1,
            ro_obj       = theta,
            ro_status    = info.get("status"),
            ro_optimal   = info.get("optimal"),
            solve_time   = t_solve_total,
            gurobi_log   = paths["gurobi"],
            supervised   = supervised,
            prune_quantile = prune_quantile,
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.instance_gen.instance_io import load_instance_json

    # Usage: python -m src.methods.RO <json_file> [time_limit]
    json_file = sys.argv[1] if len(sys.argv) > 1 else None
    time_lim  = int(sys.argv[2]) if len(sys.argv) > 2 else 2 * 3600

    if json_file is None:
        print("Usage: python -m src.methods.RO <json_file> [time_limit_s]")
        sys.exit(1)

    full_data, D_real, E_real, cv_file = load_instance_json(json_file)

    results = run_ro(
        full_data,
        D_real     = D_real,
        E_real     = E_real,
        cv         = cv_file,
        time_limit = time_lim,
        tee        = True,
        verbose    = True,
        oracle_tee = True,
    )

    print(f"\n  RO arrival  : {results['total_time']:.3f} h")
    print(f"  Wall clock  : {results['wall_clock']:.1f} s")
    print(f"  Figure      : {results['fig_path']}")
    print(f"  Solution    : {results['sol_path']}")
