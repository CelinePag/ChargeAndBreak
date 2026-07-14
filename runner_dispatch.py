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
(geometry + uncertainty realisation).  All algorithms consume the same
D_real / E_real, ensuring fair comparison.  LA and 2SP draw their forward-
looking scenarios live at run time (scenarios.generate_scenarios), rather
than from a precomputed pool.

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

  Batch usage
  -----------
  Both positional arguments accept a comma-separated list, a glob pattern
  (or several, comma-separated), or the literal "all". Every (instance,
  algorithm) combination is run. Examples:

    # one instance, all four algorithms
    python runner_dispatch.py instances/RmediumCfew_1.json all

    # a few instances, two algorithms
    python runner_dispatch.py "instances/RmediumCfew_1.json,instances/RmediumCfew_2.json" LA,RO

    # every instance in instances/, greedy only
    python runner_dispatch.py all greedy

    # every instance, every algorithm, 4 at a time in parallel processes
    python runner_dispatch.py all all --jobs 4

  --run_id is ignored for batches (>1 combination); each run gets its own
  auto-generated run_id instead, so files never collide.

  --jobs INT   number of (instance, algorithm) combinations to run
               concurrently in separate processes (default: 1, sequential).
               This is independent of --n_workers, which parallelises
               *inside* a single LA run. Only relevant for batches.

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
import os
import sys
from typing import Optional


def _apply_diesel_mode(full_data: dict, D_real: list, E_real: list) -> tuple:
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
    (full_data, D_real, E_real) — all transformed copies.
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

    return data, D_real, E_real_out


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
    safety_buffer: float             = 0.0,
    queue_threshold: Optional[float] = None,   # deprecated, ignored
    # 2SP options
    twosp_time_limit: int  = 7200,
    twosp_mip_gap: float   = 0.005,
    # common
    verbose: bool          = True,
    oracle_tee: bool       = False,
    run_id: Optional[str]  = None,
    m_man_h: Optional[float] = None,   # override stored M values (h); None = keep JSON value
    diesel_mode: bool      = False,    # treat vehicle as diesel (HoS only, no charging)
    supervised: bool       = True,     # S1: safety supervisor on (False = raw mode)
    prune_quantile: float  = 1.0,      # RH2: worst-case quantile for guard/pruning
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

    full_data, D_real, E_real, delta = load_instance_json(json_file)

    # ── Diesel mode: remove all charging/energy constraints ───────────────────
    if diesel_mode:
        full_data, D_real, E_real = _apply_diesel_mode(full_data, D_real, E_real)

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
            n_scenarios       = n_scenarios,
            delta             = delta,
            time_limit        = twosp_time_limit,
            mip_gap           = twosp_mip_gap,
            tee               = False,
            verbose           = verbose,
            run_id            = run_id,
            oracle_tee        = oracle_tee,
            supervised        = supervised,
            prune_quantile    = prune_quantile,
        )

    elif alg == "GREEDY":
        from greedy import run_greedy
        return run_greedy(
            full_data       = full_data,
            D_real          = D_real,
            E_real          = E_real,
            delta           = delta,
            safety_buffer   = safety_buffer,
            verbose         = verbose,
            run_id          = run_id,
            oracle_tee      = oracle_tee,
            supervised      = supervised,
            prune_quantile  = prune_quantile,
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
            supervised = supervised,
            prune_quantile = prune_quantile,
        )

    else:  # LA
        from Simulation import run_simulation_precomputed
        return run_simulation_precomputed(
            full_data          = full_data,
            D_real             = D_real,
            E_real             = E_real,
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
            supervised         = supervised,
            prune_quantile     = prune_quantile,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Batch dispatch — run many (instance, algorithm) combinations in one command
# ══════════════════════════════════════════════════════════════════════════════

_ALL_ALGORITHMS = ["LA", "RO", "GREEDY", "2SP"]
_INSTANCES_DIR  = "instances"


def _expand_instances(spec: str) -> list:
    """
    Expand a comma-separated spec of instance JSON paths into a concrete list.

    Each comma-separated item may be:
      - the literal "all"       -> every *.json in instances/
      - a glob pattern          -> expanded with glob.glob
      - a literal file path     -> kept as-is (existence checked by caller)
    """
    import glob as _glob

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = []
    for part in parts:
        if part.lower() == "all":
            matches = sorted(_glob.glob(os.path.join(_INSTANCES_DIR, "*.json")))
            if not matches:
                raise SystemExit(f"'all' requested but no *.json found in "
                                  f"'{_INSTANCES_DIR}/'")
            out.extend(matches)
        elif any(ch in part for ch in "*?["):
            matches = sorted(_glob.glob(part))
            if not matches:
                raise SystemExit(f"no files matched pattern '{part}'")
            out.extend(matches)
        else:
            if not os.path.isfile(part):
                raise SystemExit(f"instance file not found: '{part}'")
            out.append(part)
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _expand_algorithms(spec: str) -> list:
    """Expand a comma-separated algorithm spec ("all" or LA/RO/greedy/2SP names)."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = []
    for part in parts:
        if part.lower() == "all":
            out.extend(_ALL_ALGORITHMS)
            continue
        alg = part.upper()
        if alg not in _ALL_ALGORITHMS:
            raise SystemExit(f"unknown algorithm '{part}' "
                              f"(expected LA, RO, greedy, 2SP, or all)")
        out.append(alg)
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def _run_one_job(job: dict) -> dict:
    """
    Worker for a single (json_file, algorithm) combination.

    Runs in-process for sequential batches, or in a separate process (via
    ProcessPoolExecutor) for parallel ones -- so it must not raise, and
    everything it touches must be picklable.
    """
    job = dict(job)
    json_file = job.pop("json_file")
    algorithm = job.pop("algorithm")
    run_id    = job.get("run_id")
    try:
        res = run_algorithm(json_file=json_file, algorithm=algorithm, **job)
        return dict(
            ok=True, run_id=run_id, json_file=json_file, algorithm=algorithm,
            total_time=res["total_time"], wall_clock=res["wall_clock"],
            sol_path=res["sol_path"],
        )
    except Exception as e:
        return dict(
            ok=False, run_id=run_id, json_file=json_file, algorithm=algorithm,
            error=f"{type(e).__name__}: {e}",
        )


def run_batch(json_files: list, algorithms: list, jobs: int = 1, **kwargs) -> list:
    """
    Run every (instance, algorithm) combination from the cartesian product of
    json_files x algorithms, either sequentially (jobs=1) or across `jobs`
    worker processes.

    Each combination gets its own auto-generated run_id (instance stem +
    algorithm + shared batch timestamp + index) so files never collide, even
    when running in parallel. Any 'run_id' key in kwargs is ignored.

    A failure in one combination does not abort the rest of the batch; it is
    recorded and reported at the end.

    Returns
    -------
    list of dicts, one per combination: {"ok", "run_id", "json_file",
    "algorithm", and either ("total_time", "wall_clock", "sol_path") on
    success or "error" on failure}.
    """
    import itertools
    import time as _time
    import datetime as _dt

    kwargs.pop("run_id", None)

    combos = list(itertools.product(json_files, algorithms))
    if not combos:
        raise ValueError("no (instance, algorithm) combinations to run")

    ts_base = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs_list = []
    for idx, (jf, alg) in enumerate(combos):
        stem = os.path.splitext(os.path.basename(jf))[0]
        run_id = f"{stem}_{alg}_{ts_base}_{idx:03d}"
        jobs_list.append(dict(kwargs, json_file=jf, algorithm=alg, run_id=run_id))

    print(f"\n  Batch: {len(jobs_list)} run(s)  "
          f"({len(json_files)} instance(s) x {len(algorithms)} algorithm(s))  "
          f"[jobs={jobs}]")
    for j in jobs_list:
        print(f"    - {j['json_file']:<40} [{j['algorithm']:<6}] run_id={j['run_id']}")

    t0 = _time.time()
    results = []
    if jobs <= 1:
        for j in jobs_list:
            results.append(_run_one_job(j))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(_run_one_job, j): j for j in jobs_list}
            for fut in as_completed(futures):
                results.append(fut.result())

    elapsed = _time.time() - t0
    ok   = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]

    print(f"\n  Batch done in {elapsed:.1f}s: {len(ok)} succeeded, {len(fail)} failed")
    for r in sorted(results, key=lambda r: r["run_id"]):
        if r["ok"]:
            print(f"    OK   {r['run_id']:<45} "
                  f"arrival={r['total_time']:.3f}h  wall={r['wall_clock']:.1f}s")
        else:
            print(f"    FAIL {r['run_id']:<45} {r['error']}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LA, RO, greedy, or 2SP on one or more precomputed "
                     "instance JSON files. Both positional arguments accept "
                     "a comma-separated list, glob pattern(s), or 'all'; "
                     "every (instance, algorithm) combination is run as a "
                     "batch when more than one results from expansion."
    )
    parser.add_argument("json_file",  help="Instance JSON path(s): a single "
                         "path, a comma-separated list, glob pattern(s), or "
                         "'all' for every file in instances/.")
    parser.add_argument("algorithm",  help="Algorithm(s): LA | RO | greedy | "
                         "2SP, comma-separated for multiple, or 'all' for "
                         "all four.")

    # Common
    parser.add_argument("--run_id",      type=str,   default=None,
                        help="Only honoured for a single (instance, "
                             "algorithm) run; ignored (auto-generated "
                             "per-run instead) for batches.")
    parser.add_argument("--jobs",        type=int,   default=1,
                        help="Batches only: number of (instance, algorithm) "
                             "combinations to run concurrently in separate "
                             "processes (default: 1 = sequential). "
                             "Independent of --n_workers, which parallelises "
                             "inside a single LA run.")
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
                        choices=["mean", "worst", "best", "cvar_0.8"],
                        help="RH3: scenario aggregation for LA scoring; "
                             "cvar_0.8 = mean of the worst 20 percent.")
    parser.add_argument("--raw",          action="store_true", default=True,
                        help="S1: disable the safety supervisor (raw mode) "
                             "to expose each method's intrinsic feasibility "
                             "risk; default is unsupervised.")
    parser.add_argument("--prune_quantile", type=float, default=1.0,
                        help="RH2: worst-case quantile used by the "
                             "supervisor/pruning guard (1.0 = full support; "
                             "below 1 for unbounded distributions).")

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

    json_files = _expand_instances(args.json_file)
    algorithms = _expand_algorithms(args.algorithm)

    common_kwargs = dict(
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
        m_man_h          = args.m_man,
        diesel_mode      = args.diesel,
        supervised       = not args.raw,
        prune_quantile   = args.prune_quantile,
    )

    if len(json_files) == 1 and len(algorithms) == 1:
        results = run_algorithm(
            json_file = json_files[0],
            algorithm = algorithms[0],
            run_id    = args.run_id,
            **common_kwargs,
        )
        print(f"\n  Algorithm  : {algorithms[0]}")
        print(f"  Arrival    : {results['total_time']:.3f} h")
        print(f"  Wall clock : {results['wall_clock']:.1f} s")
        print(f"  Solution   : {results['sol_path']}")
        print(f"  Log        : {results['log_path']}")
    else:
        if args.run_id:
            print(f"  [!] --run_id ('{args.run_id}') ignored for batch runs; "
                  f"auto-generating a run_id per combination.")
        run_batch(json_files, algorithms, jobs=args.jobs, **common_kwargs)