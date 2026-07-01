"""
runner_dispatch.py — Unified algorithm dispatcher
==================================================
Provides run_algorithm(), a single entry point that loads a precomputed
instance JSON file and runs one of four algorithms:

  "LA"     — Look-ahead rolling-horizon simulation (Simulation.py)
  "RO"     — Robust optimisation, solved once on full route (RO.py)
  "greedy" — Greedy benchmark heuristic (greedy.py)
  "2SP"    — Two-stage stochastic program, extensive form (2SP.py)

Each JSON file produced by instance_io.py contains exactly one instance
(geometry + realisation + scenarios).  All algorithms consume the same
D_real / E_real, ensuring fair comparison.

Usage (Python)
--------------
  from runner_dispatch import run_algorithm

  results = run_algorithm(
      json_file  = "instances/RmediumCfew_7.json",
      algorithm  = "LA",        # "LA" | "RO" | "greedy" | "2SP"
      n_scenarios= 10,          # LA / 2SP: number of scenarios to use
  )

Usage (CLI)
-----------
  python runner_dispatch.py <json_file> <algorithm> [options...]

  algorithm: LA | RO | greedy | 2SP

  Common options:
    --verbose / --quiet
    --oracle_tee
    --run_id STR

  LA options:
    --n_scenarios INT     scenarios per stop (default: 10)
    --horizon FLOAT       look-ahead horizon hours (default: 12.0)
    --time_limit INT      per-scenario solver time limit s (default: 300)
    --n_workers INT       parallel workers (default: 8)
    --solve_mode STR      lp | mip | both (default: lp)
    --charge_only         enumerate charge decision only
    --criterion STR       mean | worst | best (default: mean)

  RO options:
    --ro_time_limit INT   full-route MIP time limit s (default: 7200)
    --ro_mip_gap FLOAT    MIP gap tolerance (default: 0.005)

  Greedy options:
    --safety FLOAT        SOC safety buffer fraction (default: 0.10)
    --queue_thresh FLOAT  skip charging at CS with queue_time > this (h)
                          (default: adaptive, 80% of instance max queue)

  2SP options:
    --n_scenarios INT     extensive-form scenario count (default: 10)
    --twosp_time_limit INT solver time limit s (default: 7200)
    --twosp_mip_gap FLOAT MIP gap tolerance (default: 0.005)
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Optional


def _apply_diesel_mode(full_data: dict, D_real: list, E_real: list,
                       scenarios_by_stop) -> tuple:
    """
    Transform EV instance data into a diesel-vehicle equivalent.

    Diesel trucks have no battery, so all energy/charging constraints are
    removed.  Only Hours-of-Service (HoS) constraints remain active.

    The transform keeps the K set intact so that former CS stops still have
    a proper td departure equation (they become break-eligible intermediate
    stops with no charging overhead).  Zeroing energy consumption causes the
    SOC to remain constant at Ecap throughout; the PWL charging math then
    forces y[i]=0 at every CS stop (tauc=0 while chg_act2 requires tauc≥0.25·y).

    Returns
    -------
    (full_data, D_real, E_real, scenarios_by_stop) — all transformed copies.
    """
    data = copy.deepcopy(full_data)
    N    = data["N"]

    # No energy consumption: unlimited diesel fuel
    data["E"]  = {i: 0.0 for i in range(N)}
    data["E0"] = data["Ecap"]   # start at max capacity (stays constant throughout)

    # Remove CS overhead — stops become plain break-eligible waypoints
    data["Q"]      = {k: 0.0 for k in data["K"]}
    data["M_stop"] = {k: 0.0 for k in data["K"]}
    data["M_seq"]  = {k: 0.0 for k in data["K"]}

    # Distinct title so the oracle cache is separate from the EV version
    data["title"] = data.get("title", "inst") + "_diesel"
    data["label"] = data.get("label", "inst") + " (diesel)"

    # Zero realised energy per leg
    E_real_out = [0.0] * len(E_real)

    # Zero scenario energy forecasts
    scn_out = []
    for stop_scens in (scenarios_by_stop or []):
        zeroed = []
        for s in stop_scens:
            s2 = dict(s)
            if "E" in s2:
                s2["E"] = {k: 0.0 for k in s2["E"]}
            zeroed.append(s2)
        scn_out.append(zeroed)

    return data, D_real, E_real_out, scn_out


def run_algorithm(
    json_file: str,
    algorithm: str,
    # LA options
    n_scenarios: int       = 10,
    horizon_hours: float   = 12.0,
    time_limit: int        = 300,
    n_workers              = None,
    solve_mode: str        = "lp",
    charge_only: bool      = False,
    criterion: str         = "mean",
    include_best: bool     = False,
    include_worst: bool    = False,
    # RO options
    ro_time_limit: int     = 7200,
    ro_mip_gap: float      = 0.005,
    # greedy options
    safety_buffer: float             = 0.10,
    queue_threshold: Optional[float] = None,
    # 2SP options
    twosp_time_limit: int  = 7200,
    twosp_mip_gap: float   = 0.005,
    # common
    verbose: bool          = True,
    oracle_tee: bool       = False,
    run_id: Optional[str]  = None,
    m_man_h: Optional[float] = None,   # override stored M values (h); None = keep JSON value
    diesel_mode: bool      = False,    # treat vehicle as diesel (HoS only, no charging)
) -> dict:
    """
    Load a precomputed instance JSON file and run the specified algorithm.

    Parameters
    ----------
    json_file       : path to a file produced by instance_io.generate_instance_file
                      (one file = one instance = one seed)
    algorithm       : "LA" | "RO" | "greedy"

    LA parameters
    -------------
    n_scenarios     : how many scenarios per stop to use (first n of 500)
    horizon_hours   : look-ahead window length (h)
    time_limit      : per-scenario MILP time limit (s)
    n_workers       : parallel workers (None = auto)
    solve_mode      : "lp" | "mip" | "both"
    charge_only     : enumerate charge decision only
    criterion       : "mean" | "worst" | "best"
    include_best    : append best-case scenario
    include_worst   : append worst-case scenario

    RO parameters
    -------------
    ro_time_limit   : full-route MIP solver time limit (s)
    ro_mip_gap      : MIP relative gap tolerance

    Greedy parameters
    -----------------
    safety_buffer   : SOC safety buffer fraction above Emin (default 0.10)
    queue_threshold : skip charging at CS stops with queue_time > this (h),
                      unless mandatory.  None = adaptive 80% of max queue.

    Common parameters
    -----------------
    verbose         : print per-stop decisions
    oracle_tee      : show HiGHS output in oracle solve
    run_id          : override auto-generated run_id
    m_man_h         : override the manoeuver time (h) stored in the JSON file.
                      The JSON was generated once with instances.make_data(); if
                      you have since changed M_man_h there, existing JSON files
                      still carry the OLD value.  Pass m_man_h here to inject the
                      new value at load-time without regenerating every file.
                      None (default) keeps whatever the JSON contains.
    diesel_mode     : if True, run as a diesel vehicle — all charging/energy
                      constraints are removed and only HoS rules apply.  CS stops
                      become plain break-eligible waypoints (no queue, no overhead).
                      The oracle cache uses a separate key so EV and diesel results
                      are stored independently.

    Returns
    -------
    dict -- canonical results dict (same schema for all three algorithms)
    """
    alg = algorithm.upper().strip()
    if alg not in ("LA", "RO", "GREEDY", "2SP"):
        raise ValueError(
            f"algorithm must be 'LA', 'RO', 'greedy', or '2SP'; got '{algorithm}'"
        )

    # ── Load precomputed instance ──────────────────────────────────────────────
    from instance_io import load_instance_json

    full_data, D_real, E_real, scenarios_by_stop, delta = load_instance_json(
        json_file,
        max_scenarios = n_scenarios if alg == "LA" else None,
    )

    # ── Diesel mode: remove all charging/energy constraints ───────────────────
    if diesel_mode:
        full_data, D_real, E_real, scenarios_by_stop = _apply_diesel_mode(
            full_data, D_real, E_real, scenarios_by_stop
        )

    # ── Override manoeuver time if requested ───────────────────────────────────
    # The JSON stores M from when the instance was first generated.  If you have
    # changed M_man_h in instances.make_data() since then, existing JSON files
    # still carry the OLD value.  This block injects the new value at load-time
    # so you don't need to regenerate every file just to change one parameter.
    if m_man_h is not None:
        N_inst = full_data["N"]
        full_data["M"] = {i: float(m_man_h) for i in range(N_inst + 1)}

    # ── Dispatch ───────────────────────────────────────────────────────────────
    if alg == "2SP":
        # 2SP.py starts with a digit, so standard import fails; use importlib
        import importlib
        twosp = importlib.import_module("2SP")
        return twosp.run_2sp(
            full_data         = full_data,
            D_real            = D_real,
            E_real            = E_real,
            scenarios_by_stop = scenarios_by_stop,
            n_scenarios       = n_scenarios,
            delta             = delta,
            time_limit        = twosp_time_limit,
            mip_gap           = twosp_mip_gap,
            tee               = False,
            verbose           = verbose,
            run_id            = run_id,
            oracle_tee        = oracle_tee,
        )

    elif alg == "GREEDY":
        from greedy import run_greedy
        return run_greedy(
            full_data       = full_data,
            D_real          = D_real,
            E_real          = E_real,
            safety_buffer   = safety_buffer,
            queue_threshold = queue_threshold,
            verbose         = verbose,
            run_id          = run_id,
            oracle_tee      = oracle_tee,
        )

    elif alg == "RO":
        from RO import run_ro
        return run_ro(
            full_data  = full_data,
            D_real     = D_real,
            E_real     = E_real,
            delta      = delta,
            time_limit = ro_time_limit,
            mip_gap    = ro_mip_gap,
            tee        = False,
            verbose    = verbose,
            run_id     = run_id,
            oracle_tee = oracle_tee,
        )

    else:  # LA
        from Simulation import run_simulation_precomputed
        return run_simulation_precomputed(
            full_data          = full_data,
            D_real             = D_real,
            E_real             = E_real,
            scenarios_by_stop  = scenarios_by_stop,
            n_scenarios        = n_scenarios,
            horizon_hours      = horizon_hours,
            delta              = delta,
            time_limit         = time_limit,
            verbose            = verbose,
            n_workers          = n_workers,
            solve_mode         = solve_mode,
            charge_only        = charge_only,
            criterion          = criterion,
            include_best       = include_best,
            include_worst      = include_worst,
            run_id             = run_id,
            oracle_tee         = oracle_tee,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LA, RO, or greedy on a precomputed instance JSON."
    )
    parser.add_argument("json_file",  help="Path to precomputed instance JSON")
    parser.add_argument("algorithm",  choices=["LA", "la", "RO", "ro",
                                               "greedy", "GREEDY",
                                               "2SP", "2sp"],
                        help="Algorithm: LA | RO | greedy | 2SP")

    # Common
    parser.add_argument("--run_id",      type=str,   default=None)
    parser.add_argument("--quiet",       action="store_true", default=False)
    parser.add_argument("--oracle_tee",  action="store_true", default=True)

    # LA
    parser.add_argument("--n_scenarios", type=int,   default=10)
    parser.add_argument("--horizon",     type=float, default=12.0)
    parser.add_argument("--time_limit",  type=int,   default=300)
    parser.add_argument("--n_workers",   type=int,   default=8)
    parser.add_argument("--solve_mode",  type=str,   default="lp",
                        choices=["lp", "mip", "both"])
    parser.add_argument("--charge_only", action="store_true", default=False)
    parser.add_argument("--criterion",   type=str,   default="mean",
                        choices=["mean", "worst", "best"])

    # RO
    parser.add_argument("--ro_time_limit",type=int,   default=7200)
    parser.add_argument("--ro_mip_gap",   type=float, default=0.005)

    # 2SP
    parser.add_argument("--twosp_time_limit", type=int,   default=7200)
    parser.add_argument("--twosp_mip_gap",    type=float, default=0.005)

    # Greedy
    parser.add_argument("--safety",       type=float, default=0.10)
    parser.add_argument("--queue_thresh", type=float, default=None)
    parser.add_argument("--m_man",        type=float, default=None,
                        help="Override manoeuver time h stored in JSON "
                             "(e.g. 0.25 = 15 min, 10 = 10 h). "
                             "Without this flag the value saved in the JSON is used.")
    parser.add_argument("--diesel",       action="store_true", default=False,
                        help="Run as diesel vehicle: remove all charging/energy "
                             "constraints, keep only HoS rules.")

    args = parser.parse_args()

    results = run_algorithm(
        json_file        = args.json_file,
        algorithm        = args.algorithm,
        n_scenarios      = args.n_scenarios,
        horizon_hours    = args.horizon,
        time_limit       = args.time_limit,
        n_workers        = args.n_workers,
        solve_mode       = args.solve_mode,
        charge_only      = args.charge_only,
        criterion        = args.criterion,
        ro_time_limit    = args.ro_time_limit,
        ro_mip_gap       = args.ro_mip_gap,
        safety_buffer    = args.safety,
        queue_threshold  = args.queue_thresh,
        twosp_time_limit = args.twosp_time_limit,
        twosp_mip_gap    = args.twosp_mip_gap,
        verbose          = not args.quiet,
        oracle_tee       = args.oracle_tee,
        run_id           = args.run_id,
        m_man_h          = args.m_man,
        diesel_mode      = args.diesel,
    )

    print(f"\n  Algorithm  : {args.algorithm.upper()}")
    print(f"  Arrival    : {results['total_time']:.3f} h")
    print(f"  Wall clock : {results['wall_clock']:.1f} s")
    print(f"  Solution   : {results['sol_path']}")
    print(f"  Log        : {results['log_path']}")