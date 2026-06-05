"""
runner_dispatch.py — Unified algorithm dispatcher
==================================================
Provides run_algorithm(), a single entry point that loads a precomputed
instance JSON file and runs one of three algorithms:

  "LA"     — Look-ahead rolling-horizon simulation (Simulation.py)
  "RO"     — Robust / deterministic simulation (Simulation.py, delta=0)
  "greedy" — Greedy benchmark heuristic (greedy.py)

Each JSON file produced by instance_io.py contains exactly one instance
(geometry + realisation + scenarios).  All three algorithms consume the
same D_real / E_real, ensuring fair comparison.

Usage (Python)
--------------
  from runner_dispatch import run_algorithm

  results = run_algorithm(
      json_file  = "instances/RmediumCfew_7.json",
      algorithm  = "LA",        # "LA" | "RO" | "greedy"
      n_scenarios= 10,          # LA only: how many of the 500 to use
      delta      = 0.20,
  )

Usage (CLI)
-----------
  python runner_dispatch.py <json_file> <algorithm> [options...]

  algorithm: LA | RO | greedy

  Common options:
    --verbose / --quiet
    --oracle_tee
    --run_id STR

  LA / RO options:
    --n_scenarios INT     scenarios per stop (default: 10)
    --horizon FLOAT       look-ahead horizon hours (default: 12.0)
    --delta FLOAT         uncertainty half-width (default: 0.20)
    --time_limit INT      per-scenario solver time limit s (default: 20)
    --n_workers INT       parallel workers (default: auto)
    --solve_mode STR      lp | mip | both (default: lp)
    --charge_only         enumerate charge decision only
    --criterion STR       mean | worst | best (default: mean)

  Greedy options:
    --safety FLOAT        SOC safety buffer fraction (default: 0.10)
    --queue_thresh FLOAT  skip CS with queue_time > this (h) (default: 999)
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def run_algorithm(
    json_file: str,
    algorithm: str,
    # LA / RO options
    n_scenarios: int       = 10,
    horizon_hours: float   = 12.0,
    delta: float           = 0.20,
    time_limit: int        = 20,
    n_workers              = None,
    solve_mode: str        = "lp",
    charge_only: bool      = False,
    criterion: str         = "mean",
    include_best: bool     = False,
    include_worst: bool    = False,
    # greedy options
    safety_buffer: float   = 0.10,
    queue_threshold: float = 999.0,
    # common
    verbose: bool          = True,
    oracle_tee: bool       = False,
    run_id: Optional[str]  = None,
) -> dict:
    """
    Load a precomputed instance JSON file and run the specified algorithm.

    Parameters
    ----------
    json_file       : path to a file produced by instance_io.generate_instance_file
                      (one file = one instance = one seed)
    algorithm       : "LA" | "RO" | "greedy"

    LA / RO parameters
    ------------------
    n_scenarios     : how many scenarios per stop to use (first n of 500)
    horizon_hours   : look-ahead window length (h)
    delta           : uncertainty half-width passed to sub-problem (LA only;
                      RO ignores this and uses delta=0)
    time_limit      : per-scenario MILP time limit (s)
    n_workers       : parallel workers (None = auto)
    solve_mode      : "lp" | "mip" | "both"
    charge_only     : enumerate charge decision only
    criterion       : "mean" | "worst" | "best"
    include_best    : append best-case scenario
    include_worst   : append worst-case scenario

    Greedy parameters
    -----------------
    safety_buffer   : SOC safety buffer fraction above Emin (default 0.10)
    queue_threshold : skip CS stops with queue_time > this (h)

    Common parameters
    -----------------
    verbose         : print per-stop decisions
    oracle_tee      : show HiGHS output in oracle solve
    run_id          : override auto-generated run_id

    Returns
    -------
    dict -- canonical results dict (same schema for all three algorithms)
    """
    alg = algorithm.upper().strip()
    if alg not in ("LA", "RO", "GREEDY"):
        raise ValueError(
            f"algorithm must be 'LA', 'RO', or 'greedy'; got '{algorithm}'"
        )

    # ── Load precomputed instance ──────────────────────────────────────────────
    from instance_io import load_instance_json

    full_data, D_real, E_real, scenarios_by_stop = load_instance_json(
        json_file,
        max_scenarios = n_scenarios if alg == "LA" else None,
    )

    # ── Dispatch ───────────────────────────────────────────────────────────────
    if alg == "GREEDY":
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

    else:  # LA or RO
        from Simulation import run_simulation_precomputed
        return run_simulation_precomputed(
            full_data          = full_data,
            D_real             = D_real,
            E_real             = E_real,
            scenarios_by_stop  = scenarios_by_stop if alg == "LA" else None,
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
                                               "greedy", "GREEDY"],
                        help="Algorithm: LA | RO | greedy")

    # Common
    parser.add_argument("--run_id",      type=str,   default=None)
    parser.add_argument("--quiet",       action="store_true", default=False)
    parser.add_argument("--oracle_tee",  action="store_true", default=False)

    # LA / RO
    parser.add_argument("--n_scenarios", type=int,   default=10)
    parser.add_argument("--horizon",     type=float, default=12.0)
    parser.add_argument("--delta",       type=float, default=0.20)
    parser.add_argument("--time_limit",  type=int,   default=300)
    parser.add_argument("--n_workers",   type=int,   default=8)
    parser.add_argument("--solve_mode",  type=str,   default="lp",
                        choices=["lp", "mip", "both"])
    parser.add_argument("--charge_only", action="store_true", default=False)
    parser.add_argument("--criterion",   type=str,   default="mean",
                        choices=["mean", "worst", "best"])

    # Greedy
    parser.add_argument("--safety",       type=float, default=0.10)
    parser.add_argument("--queue_thresh", type=float, default=999.0)

    args = parser.parse_args()

    results = run_algorithm(
        json_file       = args.json_file,
        algorithm       = args.algorithm,
        n_scenarios     = args.n_scenarios,
        horizon_hours   = args.horizon,
        delta           = args.delta,
        time_limit      = args.time_limit,
        n_workers       = args.n_workers,
        solve_mode      = args.solve_mode,
        charge_only     = args.charge_only,
        criterion       = args.criterion,
        safety_buffer   = args.safety,
        queue_threshold = args.queue_thresh,
        verbose         = not args.quiet,
        oracle_tee      = args.oracle_tee,
        run_id          = args.run_id,
    )

    print(f"\n  Algorithm  : {args.algorithm.upper()}")
    print(f"  Arrival    : {results['total_time']:.3f} h")
    print(f"  Wall clock : {results['wall_clock']:.1f} s")
    print(f"  Solution   : {results['sol_path']}")
    print(f"  Log        : {results['log_path']}")