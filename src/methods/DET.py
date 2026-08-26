"""
DET.py — Deterministic plan, executed as is
===========================================
The NAIVE counterpart of the robust plan: the driver builds the schedule on
the NOMINAL travel times, as if no uncertainty existed, then drives it.

    D_i  =  D_nom_i           (nominal travel time — the MEAN of xi*D_nom,
                               since settings calibrates E[xi] = 1 exactly)
    E_i  =  E_nom_i           (the instance's nominal leg energy; legs with
                               E = 0 stay at 0, so the diesel mode survives)

This is the expected-value problem.  Everything else — the model, the
decision variables, the objective, and the execution path — is IDENTICAL to
RO.py; only the leg parameters differ.  RO pins every leg at its worst case,
DET pins every leg at its mean, and the two bracket the policies in between.

Execution (no recourse)
-----------------------
Same as RO: the plan is executed AS IS (recourse.run_plan_static).  Both the
binary structure (charge / break / rest) and the activity durations are
committed offline and applied verbatim at each stop.  Nothing re-optimises
online and nothing repairs a broken plan.

That is the whole point of this method.  The nominal plan has NO slack for a
slow leg, so when the realised draw runs long the schedule breaks, and the
violations BEHDV records — stranding, HoS breaches, missed time windows —
are the answer to "how wrong does a deterministic plan go?".  Failures here
are the RESULT, not a bug: unlike RO, DET carries no feasibility guarantee.

Integration
-----------
  from src.methods.DET import run_det
  results = run_det(full_data, D_real, E_real)

  Or via runner_dispatch:
    python -m src.simulation.runner_dispatch instances/RmediumCfewTmedium_7.json DET
"""

from __future__ import annotations

import datetime
import sys
import time

from src.methods.recourse  import run_plan_static
from src.settings  import TRAVEL_TIME_CV_TARGET, GUARD_QUANTILE
from src.simulation.runner    import finalize_run
from src.methods import twosp as _twosp
from src import paths as _paths


# ══════════════════════════════════════════════════════════════════════════════
# NOMINAL SCENARIO
# ══════════════════════════════════════════════════════════════════════════════

def _nominal_scenario(full_data: dict) -> dict:
    """
    The instance's nominal leg parameters as a single scenario — the mean of
    the travel-time distribution (settings.lognormal_params re-solves mu after
    the XI_MAX cap so that E[xi] = 1 exactly).

    Energy is taken STRAIGHT from the instance rather than rebuilt from the
    ECR curve: E_nom is already the nominal-speed consumption, and copying it
    keeps a diesel-mode instance (E = 0 on every leg) at zero instead of
    silently turning it back into an electric problem.
    """
    N     = full_data["N"]
    D_nom = full_data["D"]
    E_nom = full_data.get("E", {})
    D_s = {i: float(D_nom.get(i, 0.0)) for i in range(N)}
    E_s = {i: float(E_nom.get(i, 0.0)) for i in range(N)}
    return dict(D=D_s, E=E_s)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_det(full_data: dict,
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
    Solve the deterministic (nominal) MILP and execute it AS IS on
    D_real/E_real — no online recourse of any kind.

    Parameters
    ----------
    full_data     : instance dict (from instance_io.load_instance_json)
    D_real        : list[float] — precomputed realised travel times (h)
    E_real        : list[float] — precomputed realised energies (kWh)
    cv            : CV of the travel-time multiplier — the plan is built at
                    the mean regardless of cv; cv is only forwarded to the S1
                    supervisor guard
    supervised    : apply the shared S1 safety supervisor during execution
                    (default False — a broken plan is recorded, not rescued,
                    which is the measurement this method exists to make)
    **kwargs      : ignored (absorbs caller arguments shared with the other
                    methods, e.g. n_scenarios)

    Returns
    -------
    dict — canonical results dict (same schema as run_ro / run_simulation)
    """
    t_wall_start = time.perf_counter()
    N            = full_data["N"]
    T_START      = full_data.get("T_START", 8.0)
    label        = full_data.get("label", "det")
    title        = full_data.get("title", "inst")

    assert len(D_real) == N, f"D_real length {len(D_real)} != N={N}"
    assert len(E_real) == N, f"E_real length {len(E_real)} != N={N}"

    _paths.ensure_dirs()
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{title}_DET_{ts}"
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
    _p(f"  DET SOLVE START  ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    _p(f"  Instance : {label}   run_id={run_id}")
    _p(f"  Route    : {N} stops  departure={T_START:.0f}:00")
    _p(f"  Model    : deterministic MILP at NOMINAL travel times (xi = 1), "
       f"executed as is, no recourse")
    _p("=" * 65)

    # ── Step 1: single nominal scenario (every leg at its mean) ───────────────
    scen_set = [_nominal_scenario(full_data)]

    # ── Step 2: deterministic solve on the nominal parameters ────────────────
    # With one scenario, the epigraph max + shared durations reduce EXACTLY to
    # the deterministic full-route MILP — the same reduction RO.py relies on,
    # here evaluated at the mean instead of the box corner.
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
        _p("  No feasible plan at nominal travel times — aborting.")
        log.close()
        return dict(feasible=False, status=status,
                    total_time=float("inf"),
                    wall_clock=time.perf_counter() - t_wall_start)

    theta = info["obj"]
    plan  = _twosp.extract_2sp_full_schedule(model, full_data)
    _p(f"  theta={theta:.3f}h  "
       f"plan: {sum(1 for e in plan if e['y'])} chg / "
       f"{sum(1 for e in plan if e['break_type'])} brk / "
       f"{sum(1 for e in plan if e['rest_type'])} rst")
    _p(f"\n  DET objective (planned arrival at nominal) : {theta:.3f} h   "
       f"solve {t_solve_total:.1f}s")

    # ── Step 3: execute the FIXED plan AS IS (static, no recourse) ───────────
    _p(f"\n  Executing DET plan AS IS (static, no recourse)...")
    vehicle, tracker, events = run_plan_static(
        full_data      = full_data,
        plan           = plan,
        D_real         = D_real,
        E_real         = E_real,
        method_name    = "DET",
        log_fn         = _p,
        cv             = cv,
        supervised     = supervised,
        prune_quantile = prune_quantile,
        verbose        = verbose,
    )
    n_fail = len(getattr(vehicle, "violations", []))
    _p(f"  Nominal-plan failures (raw): {n_fail} violation(s); "
       f"{len(events['interventions'])} supervisor intervention(s)")

    wall_elapsed = time.perf_counter() - t_wall_start
    arr_h        = vehicle.t_arr
    _p(f"\n{'='*65}")
    _p(f"  DET SIMULATION COMPLETE")
    _p(f"  Planned arrival    : {theta:.3f} h   (what the driver expected)")
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
            method       = "DET",
            cv           = cv,
            n_scenarios  = 1,
            det_obj      = theta,
            det_status   = info.get("status"),
            det_optimal  = info.get("optimal"),
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

    # Usage: python -m src.methods.DET <json_file> [time_limit]
    json_file = sys.argv[1] if len(sys.argv) > 1 else None
    time_lim  = int(sys.argv[2]) if len(sys.argv) > 2 else 2 * 3600

    if json_file is None:
        print("Usage: python -m src.methods.DET <json_file> [time_limit_s]")
        sys.exit(1)

    full_data, D_real, E_real, cv_file = load_instance_json(json_file)

    results = run_det(
        full_data,
        D_real     = D_real,
        E_real     = E_real,
        cv         = cv_file,
        time_limit = time_lim,
        tee        = True,
        verbose    = True,
        oracle_tee = True,
    )

    print(f"\n  DET arrival : {results['total_time']:.3f} h")
    print(f"  Wall clock  : {results['wall_clock']:.1f} s")
    print(f"  Figure      : {results['fig_path']}")
    print(f"  Solution    : {results['sol_path']}")
