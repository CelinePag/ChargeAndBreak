"""
runner_dispatch.py — Unified algorithm dispatcher
==================================================
Provides run_algorithm(), a single entry point that loads a precomputed
instance JSON file and runs one of four algorithms:

  "LA"     — Look-ahead rolling-horizon simulation (Simulation.py)
  "RO"     — Robust optimisation (conservative box), full route (RO.py)
  "ROBU"   — Budgeted robust optimisation (Bertsimas–Sim, C&CG) (RObudget.py)
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
      algorithm  = "LA",        # "LA" | "RO" | "ROBU" | "greedy" | "2SP"
      n_scenarios= 10,          # LA / 2SP: number of scenarios to use
  )

Usage (CLI)
-----------
  python runner_dispatch.py <json_file> <algorithm> [options...]

  algorithm: LA | RO | ROBU | greedy | 2SP

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

  ROBU options (also uses --ro_time_limit / --ro_mip_gap per master solve):
    --robu_eps FLOAT      target violation probability for the classic
                          Bertsimas-Sim budget Gamma = 1 + z_{1-eps}*sqrt(N)
                          (default: 0.01)
    --robu_gamma INT      explicit budget override (default: from --robu_eps)
    --robu_max_iter INT   max C&CG iterations (default: 12)

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

from settings import GUARD_QUANTILE


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
    # ROBU options (shares ro_time_limit / ro_mip_gap for the master solves)
    robu_eps: float        = 0.01,
    robu_gamma: Optional[int] = None,
    robu_max_iter: int     = 12,
    robu_max_cuts: int     = 3,     # feasibility cuts appended per C&CG round
    robu_wall_limit: Optional[int] = None,   # total C&CG budget (s); None=2*master
    robu_seed_scenarios: bool = True,
    robu_warmstart: bool   = True,
    # greedy options
    safety_buffer: float             = 0.0,
    queue_threshold: Optional[float] = None,   # deprecated, ignored
    # 2SP options
    twosp_time_limit: int  = 7200,
    twosp_mip_gap: float   = 0.005,
    twosp_warmstart_seed: bool = False,
    # shared MILP solver tuning (RO / 2SP); None → Gurobi defaults
    milp_heuristics: Optional[float] = 0.2,
    milp_mip_focus: Optional[int]    = None,
    # ORACLE options (the hindsight MILP, run as its own algorithm)
    oracle_time_limit: int = 12 * 3600,
    oracle_mip_gap: float  = 0.005,
    oracle_warmstart: bool = True,   # seed the MIP with a quiet greedy run
    # common
    verbose: bool          = True,
    oracle_tee: bool       = False,
    run_id: Optional[str]  = None,
    m_man_h: Optional[float] = None,   # override stored M values (h); None = keep JSON value
    diesel_mode: bool      = False,    # treat vehicle as diesel (HoS only, no charging)
    supervised: bool       = False,    # S1: safety supervisor off by default (raw mode)
    prune_quantile: Optional[float] = GUARD_QUANTILE,  # RH2 guard level; None = disabled
    resume: bool           = False,    # LA only: resume a crashed run from its checkpoint
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
    if alg not in ("LA", "RO", "ROBU", "GREEDY", "2SP", "ORACLE"):
        raise ValueError(
            f"algorithm must be 'LA', 'RO', 'ROBU', 'greedy', '2SP', or "
            f"'ORACLE'; got '{algorithm}'"
        )

    # ── Load precomputed instance ──────────────────────────────────────────────
    from instance_io import load_instance_json

    full_data, D_real, E_real, cv = load_instance_json(json_file)

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
            cv                = cv,
            time_limit        = twosp_time_limit,
            mip_gap           = twosp_mip_gap,
            heuristics        = milp_heuristics,
            mip_focus         = milp_mip_focus,
            warmstart_seed    = twosp_warmstart_seed,
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
            cv              = cv,
            guard_quantile  = prune_quantile,
            safety_buffer   = safety_buffer,
            verbose         = verbose,
            run_id          = run_id,
            oracle_tee      = oracle_tee,
            supervised      = supervised,
            prune_quantile  = prune_quantile,
        )

    elif alg == "ROBU":
        from RObudget import run_robu
        return run_robu(
            full_data  = full_data,
            D_real     = D_real,
            E_real     = E_real,
            cv         = cv,
            time_limit = ro_time_limit,
            wall_limit = robu_wall_limit,
            mip_gap    = ro_mip_gap,
            eps        = robu_eps,
            gamma      = robu_gamma,
            max_iter   = robu_max_iter,
            max_cuts   = robu_max_cuts,
            seed_scenarios = robu_seed_scenarios,
            warmstart  = robu_warmstart,
            tee        = False,
            verbose    = verbose,
            run_id     = run_id,
            oracle_tee = oracle_tee,
            supervised = supervised,
            prune_quantile = prune_quantile,
        )

    elif alg == "RO":
        from RO import run_ro
        return run_ro(
            full_data  = full_data,
            D_real     = D_real,
            E_real     = E_real,
            cv         = cv,
            time_limit = ro_time_limit,
            mip_gap    = ro_mip_gap,
            heuristics = milp_heuristics,
            mip_focus  = milp_mip_focus,
            tee        = False,
            verbose    = verbose,
            run_id     = run_id,
            oracle_tee = oracle_tee,
            supervised = supervised,
            prune_quantile = prune_quantile,
        )

    elif alg == "ORACLE":
        # The hindsight oracle, run independently of any method.  It solves the
        # full-route MILP on the instance's realised travel times, writes the
        # shared cache solutions/oracle_<instance>.json (consumed on demand for
        # the gap to oracle) and a Gurobi bound log for plotting.
        import time as _time
        from oracle import oracle_solve, save_oracle_cache
        instance   = full_data.get("title", "unknown")
        os.makedirs("logs", exist_ok=True)
        gurobi_log = os.path.join("logs", f"oracle_{instance}_gurobi.log")
        # Always produce a summary .txt log: a single manual run (run_id=None)
        # falls back to an instance-named file so the user still gets a log
        # instead of only the per-instance Gurobi node table.
        txt_stem   = run_id if run_id else f"{instance}_ORACLE"
        txt_log    = os.path.join("logs", f"{txt_stem}.txt")
        # line-buffered so the header/summary is readable while the (possibly
        # multi-hour) solve is still running, not only after it closes.
        _lfh = open(txt_log, "w", encoding="utf-8", buffering=1)
        # Warm start: run the (deterministic, seconds-fast) greedy policy on
        # the same realised travel times and inject its schedule as the MIP
        # start.  This hands the solver a feasible incumbent immediately so
        # all effort goes into the dual bound.  Never let a greedy failure
        # (e.g. an infeasible-as-recorded run) kill an hours-long oracle
        # solve — fall back to a cold start.
        _ws_results = None
        if oracle_warmstart:
            try:
                from greedy import run_greedy
                _ws_results = run_greedy(
                    full_data, D_real, E_real,
                    verbose=False, oracle_tee=False, supervised=supervised,
                )
            except Exception as e:
                print(f"  Oracle warm-start greedy failed ({e}); "
                      f"solving cold.", file=_lfh)
                _ws_results = None
        t0 = _time.perf_counter()
        res = oracle_solve(
            full_data, D_real, sim_results=_ws_results,
            time_limit=oracle_time_limit, mip_gap=oracle_mip_gap,
            tee=oracle_tee, verbose=verbose, log_fh=_lfh, log_file=gurobi_log,
        )
        wall = _time.perf_counter() - t0
        # Persist the wall-clock next to the gap/stop_reason in the .txt log too.
        try:
            print(f"  Wall clock : {wall:.1f} s\n"
                  f"  stop_reason={res.get('stop_reason')}  "
                  f"gap={res.get('gap')}  best_bound={res.get('best_bound')}",
                  file=_lfh)
            _lfh.flush()
        except Exception:
            pass
        _lfh.close()
        res = dict(res)
        res["wall_clock"] = wall          # cache the wall-clock alongside gap etc.
        cache_path = save_oracle_cache(instance, res)
        if verbose:
            print(f"  Oracle   : stop_reason={res.get('stop_reason')}  "
                  f"gap={res.get('gap')}  obj={res.get('obj')}  "
                  f"cached -> {cache_path}")
        out = dict(res)
        out["total_time"] = res.get("obj", float("inf"))
        out["wall_clock"] = wall
        out["sol_path"]   = cache_path
        out["gurobi_log"] = gurobi_log
        out["log_path"]   = txt_log
        return out

    else:  # LA
        from Simulation import run_simulation_precomputed
        return run_simulation_precomputed(
            full_data          = full_data,
            D_real             = D_real,
            E_real             = E_real,
            n_scenarios        = n_scenarios,
            horizon_hours      = horizon_hours,
            cv                 = cv,
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
            resume             = resume,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Batch dispatch — run many (instance, algorithm) combinations in one command
# ══════════════════════════════════════════════════════════════════════════════

_ALL_ALGORITHMS = ["LA", "RO", "ROBU", "GREEDY", "2SP"]
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
        # ORACLE is a valid explicit algorithm but is excluded from "all"
        # (it is the hindsight reference, not a policy under comparison)
        if alg not in _ALL_ALGORITHMS and alg != "ORACLE":
            raise SystemExit(f"unknown algorithm '{part}' "
                              f"(expected LA, RO, ROBU, greedy, 2SP, ORACLE, "
                              f"or all)")
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


# ── skip-already-run detection ────────────────────────────────────────────────
# Per method, the result-defining parameters recorded at the top level of each
# solution JSON (method_meta is merged there by runner.finalize_run).  Two runs
# with the same instance, method, and these values are considered identical, so
# --skip-existing will not re-solve them.  NOTE: this matches on stored
# PARAMETERS only, not on the code version — after a modelling change, delete the
# stale solutions (and oracle_*.json caches) rather than relying on the skip.
_SIG_FIELDS = {
    "GREEDY": ["safety_buffer", "supervised", "prune_quantile"],
    "RO":     ["supervised", "prune_quantile"],
    "ROBU":   ["robu_eps", "gamma", "supervised", "prune_quantile"],
    "2SP":    ["n_scenarios", "supervised", "prune_quantile"],
    "LA":     ["n_scenarios", "horizon_hours", "criterion", "solve_mode",
               "charge_only", "supervised", "prune_quantile"],
}


def _requested_sig(alg: str, kw: dict) -> Optional[dict]:
    """Signature (method_meta field -> value) the requested run would record."""
    sup = bool(kw.get("supervised", False))
    pq  = kw.get("prune_quantile")
    if alg == "GREEDY":
        return dict(safety_buffer=kw.get("safety_buffer", 0.0),
                    supervised=sup, prune_quantile=pq)
    if alg == "RO":
        return dict(supervised=sup, prune_quantile=pq)
    if alg == "ROBU":
        sig = dict(robu_eps=kw.get("robu_eps", 0.01),
                   gamma=kw.get("robu_gamma"),
                   supervised=sup, prune_quantile=pq)
        if sig["gamma"] is None:          # gamma is eps-derived -> match on eps
            sig.pop("gamma")
        return sig
    if alg == "2SP":
        return dict(n_scenarios=kw.get("n_scenarios", 10),
                    supervised=sup, prune_quantile=pq)
    if alg == "LA":
        return dict(n_scenarios=kw.get("n_scenarios", 10),
                    horizon_hours=kw.get("horizon_hours", 12.0),
                    criterion=kw.get("criterion", "mean"),
                    solve_mode=kw.get("solve_mode", "lp"),
                    charge_only=bool(kw.get("charge_only", False)),
                    supervised=sup, prune_quantile=pq)
    return None


def _val_eq(a, b) -> bool:
    """Tolerant equality for signature values (None / bool / number / str)."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return str(a) == str(b)


def _find_matching_run(json_file: str, alg: str, kw: dict,
                       solutions_dir: str = "solutions") -> Optional[str]:
    """Return the path of a finished solution with matching params, else None."""
    import glob as _glob
    import json as _json
    req = _requested_sig(alg, kw)
    if req is None:
        return None
    stem        = os.path.splitext(os.path.basename(json_file))[0]
    want_diesel = bool(kw.get("diesel_mode", False))
    for path in _glob.glob(os.path.join(solutions_dir, f"{stem}_{alg}_*.json")):
        if os.path.basename(path).startswith("oracle_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                sol = _json.load(fh)
        except Exception:
            continue
        if sol.get("sim_arrival_h") is None:          # not a finished run
            continue
        if str(sol.get("instance", "")).endswith("_diesel") != want_diesel:
            continue
        if all(_val_eq(v, sol.get(k)) for k, v in req.items()):
            return path
    return None


def run_batch(json_files: list, algorithms: list, jobs: int = 1,
              skip_existing: bool = False, **kwargs) -> list:
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

    skipped = []
    if skip_existing:
        kept = []
        for jf, alg in combos:
            match = _find_matching_run(jf, alg, kwargs)
            if match:
                skipped.append((jf, alg, match))
            else:
                kept.append((jf, alg))
        combos = kept
        print(f"\n  Skipping {len(skipped)} already-run combination(s) "
              f"(matching instance + method + parameters):")
        for jf, alg, match in skipped[:12]:
            print(f"    - {os.path.basename(jf):<40} [{alg:<6}] "
                  f"-> {os.path.basename(match)}")
        if len(skipped) > 12:
            print(f"    ... and {len(skipped) - 12} more")
        if not combos:
            print("  Nothing left to run.")
            return []

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
    parser.add_argument("algorithm",  help="Algorithm(s): LA | RO | ROBU | "
                         "greedy | 2SP, comma-separated for multiple, or "
                         "'all' for all five.")

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
    parser.add_argument("--supervised",   action="store_true", default=False,
                        help="S1: enable the safety supervisor guard. "
                             "Default off (raw mode): each method's intrinsic "
                             "feasibility risk is exposed — an infeasible run "
                             "is recorded as-is, never rescued.")
    parser.add_argument("--prune_quantile", type=float, default=GUARD_QUANTILE,
                        help="RH2: probability level of the one-step "
                             "feasibility guard (greedy rule, LA pruning, "
                             "opt-in supervisor).  Default disabled: greedy "
                             "checks at nominal, LA does no flag-based "
                             "pruning.  0.95 = guard at the xi 95%% "
                             "quantile; 1.0 = full support corners.")

    # RO
    parser.add_argument("--ro_time_limit",type=int,   default=7200)
    parser.add_argument("--ro_mip_gap",   type=float, default=0.005)

    # ROBU (budgeted robust, Bertsimas-Sim C&CG; masters use the RO limits)
    parser.add_argument("--robu_eps",     type=float, default=0.01,
                        help="Target violation probability for the classic "
                             "B-S budget Gamma = 1 + z_(1-eps)*sqrt(N).")
    parser.add_argument("--robu_gamma",   type=int,   default=None,
                        help="Explicit budget Gamma override (default: "
                             "derived from --robu_eps).")
    parser.add_argument("--robu_max_iter",type=int,   default=12,
                        help="Max C&CG robustification-pessimization "
                             "iterations.")
    parser.add_argument("--robu_max_cuts",type=int,   default=3,
                        help="Feasibility cuts (violated budget scenarios) "
                             "appended to the master per C&CG round. More = "
                             "fewer iterations but a faster-growing, slower "
                             "master; fewer = leaner masters, more iterations. "
                             "Keep small (1-3) on long routes.")
    parser.add_argument("--robu_wall_limit", type=int, default=None,
                        help="Total C&CG wall-clock budget (s) across all "
                             "iterations; the loop stops before a master that "
                             "would exceed it and returns the best plan so far "
                             "(reported as 'unsolved', not infeasible). "
                             "Default: 2x --ro_time_limit.")
    parser.add_argument("--robu_no_seed", dest="robu_seed_scenarios",
                        action="store_false", default=True,
                        help="Disable priming the scenario set with worst-case "
                             "vertices (start from nominal only).")
    parser.add_argument("--robu_no_warmstart", dest="robu_warmstart",
                        action="store_false", default=True,
                        help="Disable feeding each master the previous plan as "
                             "a Gurobi MIP start.")

    # ORACLE (hindsight MILP, run as its own algorithm)
    parser.add_argument("--oracle_time_limit", type=int, default=12 * 3600,
                        help="ORACLE solver time limit s (default: 12h).")
    parser.add_argument("--oracle_mip_gap", type=float, default=0.005,
                        help="ORACLE MIP gap tolerance (default: 0.005). Raise "
                             "for long routes where proving the last %% is slow.")
    parser.add_argument("--no_oracle_warmstart", dest="oracle_warmstart",
                        action="store_false", default=True,
                        help="Disable the greedy warm start of the ORACLE MIP "
                             "(default: enabled).")

    # 2SP
    parser.add_argument("--twosp_time_limit", type=int,   default=7200)
    parser.add_argument("--twosp_mip_gap",    type=float, default=0.005)
    parser.add_argument("--twosp_warmstart_seed", action="store_true",
                        default=False,
                        help="2SP: solve the cheap nominal 1-scenario model "
                             "first and feed its plan to the full extensive "
                             "form as a Gurobi MIP start (helps hard long "
                             "routes; wasteful on fast instances).")
    # shared MILP tuning (RO / 2SP master solves)
    parser.add_argument("--milp_heuristics", type=float, default=0.2,
                        help="Gurobi 'Heuristics' fraction for the RO/2SP "
                             "solves (default 0.2, as the oracle uses). Higher "
                             "finds good incumbents sooner on long routes; "
                             "pass a negative value to leave Gurobi's default.")
    parser.add_argument("--milp_mip_focus", type=int, default=None,
                        choices=[0, 1, 2, 3],
                        help="Gurobi 'MIPFocus' for RO/2SP (1=find feasible "
                             "fast, 2=prove optimality, 3=improve bound). "
                             "Default: Gurobi's balanced default.")

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
    parser.add_argument("--skip-existing", dest="skip_existing",
                        action="store_true", default=False,
                        help="Skip any (instance, method) that already has a "
                             "finished solution with the SAME parameters, so a "
                             "batch can be re-run without redoing completed "
                             "work.  Matches stored parameters only, NOT the "
                             "code version: after a model change, delete stale "
                             "solutions instead of relying on this.")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="LA only: checkpoint after every stop and, on a "
                             "re-run with the same parameters, continue a "
                             "crashed run from its last completed stop instead "
                             "of restarting from zero.")

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
        robu_eps         = args.robu_eps,
        robu_gamma       = args.robu_gamma,
        robu_max_iter    = args.robu_max_iter,
        robu_max_cuts    = args.robu_max_cuts,
        robu_wall_limit     = args.robu_wall_limit,
        robu_seed_scenarios = args.robu_seed_scenarios,
        robu_warmstart      = args.robu_warmstart,
        safety_buffer    = args.safety,
        queue_threshold  = args.queue_thresh,
        oracle_time_limit = args.oracle_time_limit,
        oracle_mip_gap    = args.oracle_mip_gap,
        oracle_warmstart  = args.oracle_warmstart,
        twosp_time_limit = args.twosp_time_limit,
        twosp_mip_gap    = args.twosp_mip_gap,
        twosp_warmstart_seed = args.twosp_warmstart_seed,
        milp_heuristics  = (None if (args.milp_heuristics is not None
                                     and args.milp_heuristics < 0)
                            else args.milp_heuristics),
        milp_mip_focus   = args.milp_mip_focus,
        verbose          = not args.quiet,
        oracle_tee       = args.oracle_tee,
        m_man_h          = args.m_man,
        diesel_mode      = args.diesel,
        supervised       = args.supervised,
        prune_quantile   = args.prune_quantile,
        resume           = args.resume,
    )

    if len(json_files) == 1 and len(algorithms) == 1:
        _match = (_find_matching_run(json_files[0], algorithms[0], common_kwargs)
                  if args.skip_existing else None)
        if _match:
            print(f"\n  Skipping {algorithms[0]} on "
                  f"{os.path.basename(json_files[0])}: already run with the "
                  f"same parameters -> {os.path.basename(_match)}")
        else:
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
            if results.get("log_path"):
                print(f"  Log        : {results['log_path']}")
            if results.get("gurobi_log"):
                print(f"  Gurobi log : {results['gurobi_log']}")
    else:
        if args.run_id:
            print(f"  [!] --run_id ('{args.run_id}') ignored for batch runs; "
                  f"auto-generating a run_id per combination.")
        run_batch(json_files, algorithms, jobs=args.jobs,
                  skip_existing=args.skip_existing, **common_kwargs)