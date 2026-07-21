"""
runner.py — Shared run epilogue for simulation and greedy
==========================================================
Centralises the boilerplate that is identical between run_simulation
(Simulation.py) and run_greedy (greedy.py):

  finalize_run(vehicle, full_data, tracker, run_id, paths, timing, log_fh,
               verbose, oracle_tee, method_meta)

      1. Calls oracle_solve with the actual travel times.
      2. Saves the results JSON (trajectory + oracle + scenario summary).
      3. Saves the ScenarioTracker JSON.
      4. Closes the log file.
      5. Prints the simulation and oracle schedule tables.
      6. Runs check_simulation_feasibility and warns on violations.
      7. Returns the canonical results dict consumed by plots and reporting.

Why this exists
---------------
Before runner.py, both run_simulation and run_greedy contained ~60 lines of
identical code that:
  - Called oracle_solve
  - Serialised results to JSON
  - Saved logs
  - Built the return dict with the same 14 keys

Any bug fixed in one copy had to be manually replicated in the other.
runner.py eliminates that duplication.  Adding a third decision method
(e.g. MCTS, rule-based) only requires calling finalize_run.

Results dict schema
-------------------
The dict returned by finalize_run has the following keys.  Simulation.py and
greedy.py add their own method-specific keys on top.

  vehicle          : BEHDV    — vehicle object with full history
  states           : list     — vehicle.states (_Snapshot namedtuples)
  actions          : list     — list of action dicts (one per stop)
  scores_log       : list     — per-stop score lists (empty for greedy)
  td_list          : list     — departure times (h)
  D_actual_list    : list     — actual leg travel times (h)
  durations_list   : list     — {taub, tauc, taur, tauq} per stop
  total_time       : float    — absolute arrival time at destination (h)
  wall_clock       : float    — elapsed wall-clock time (s)
  oracle           : dict     — oracle_solve result
  scenario_tracker : ScenarioTracker
  log_path         : str      — path to the text log file
  fig_path         : str      — path where the figure will be saved
  sol_path         : str      — path to the results JSON
  scn_path         : str      — path to the scenario tracker JSON
  run_id           : str

Import chain
------------
  runner.py → oracle, BEHDV (via oracle), scenarios
  Simulation.py → runner
  greedy.py     → runner
"""

from __future__ import annotations

import json
import os
from typing import Optional

from oracle import oracle_solve, check_simulation_feasibility, \
                   check_directive_compliance, \
                   print_simulation_log, print_oracle_log
from plots import plot_simulation_results
from scenarios import ScenarioTracker


def finalize_run(
    vehicle,
    full_data: dict,
    tracker: ScenarioTracker,
    run_id: str,
    paths: dict,
    timing: dict,
    log_fh,
    verbose: bool        = True,
    oracle_tee: bool     = True,
    scores_log: list     = None,
    method_meta: dict    = None,
    events: dict         = None,
    plot: bool           = False,
) -> dict:
    """
    Shared epilogue executed at the end of every simulation run.

    Parameters
    ----------
    vehicle      : BEHDV — vehicle after completing the full route
    full_data    : dict  — route data from instances.make_data()
    tracker      : ScenarioTracker — scenario and realisation records
    run_id       : str  — unique run identifier used in file names
    paths        : dict with keys: log, fig, sol, scn
                   (absolute or relative file paths)
    timing       : dict with keys: wall_clock (float, seconds),
                   T_START (float, departure time in hours)
    log_fh       : open file handle for the text log (will be closed here)
    verbose      : bool — print oracle and simulation tables to stdout
    oracle_tee   : bool — pass tee=True to oracle_solve (shows HiGHS output)
    scores_log   : list — per-stop score lists; empty list for greedy
    method_meta  : dict — extra keys merged into the results JSON
                   (e.g. {"method": "greedy", "charge_frac": 0.4})
    events       : dict — S1/S2/RH4 event records from the run loop:
                   interventions, decision_times, cmp_log, repairs,
                   plan_violations (all optional)
    plot         : bool — render and save the five-panel figure at the end
                   of the run (default False).  Figures can always be
                   produced later from the saved solution JSON with
                   `python plots.py <run_id>`.

    Returns
    -------
    dict — canonical results dict (see module docstring for key listing)
    """
    arr    = vehicle.t_arr
    T0     = timing.get("T_START", full_data.get("T_START", 8.0))
    wall   = timing["wall_clock"]
    events = events or {}

    def _lp(msg: str):
        if verbose:
            print(msg)
        if log_fh and not log_fh.closed:
            try:
                print(msg, file=log_fh)
            except Exception:
                pass

    # ── 1. Oracle solve (with file cache) ────────────────────────────────────
    # The oracle depends only on the instance geometry + the realised travel
    # times (D_real).  Both are fixed per precomputed JSON file, so the result
    # is identical regardless of which algorithm ran.
    # Cache file: solutions/oracle_<title>.json
    _oracle_cache_path = os.path.join(
        "solutions", f"oracle_{full_data.get('title', 'unknown')}.json"
    )

    def _load_oracle_cache(path):
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as _fh:
                cached = json.load(_fh)
            def _rik(obj):
                if isinstance(obj, dict):
                    return {(int(k) if isinstance(k, str) and
                             k.lstrip("-").isdigit() else k): _rik(v)
                            for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_rik(v) for v in obj]
                return obj
            cached = _rik(cached)
            _lp(f"  Oracle   : loaded from cache  {path}")
            return cached
        except Exception as _e:
            _lp(f"  Oracle   : cache read failed ({_e}), re-solving")
            return None

    def _save_oracle_cache(path, result):
        def _ser(o):
            if isinstance(o, (int, float, bool, str, type(None))): return o
            if isinstance(o, dict): return {str(k): _ser(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [_ser(v) for v in o]
            return str(o)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as _fh:
                json.dump(_ser(result), _fh, indent=2)
            _lp(f"  Oracle   : result cached to  {path}")
        except Exception as _e:
            _lp(f"  Oracle   : cache save failed ({_e})")

    oracle = _load_oracle_cache(_oracle_cache_path)
    if oracle is None:
        oracle = oracle_solve(
            full_data,
            vehicle.D_actual_list,
            sim_results=dict(
                states         = vehicle.states,
                actions        = vehicle.actions,
                durations_list = vehicle.durations,
                td_list        = vehicle.td_list,
                total_time     = arr,
            ),
            verbose = verbose,
            tee     = oracle_tee,
            log_fh  = log_fh,
        )
        _save_oracle_cache(_oracle_cache_path, oracle)

    # ── 2. Save scenario tracker ──────────────────────────────────────────────
    scn_path = paths["scn"]
    tracker.save(scn_path)
    scn_summary = tracker.summary()
    coverage = scn_summary.get("coverage_fraction")
    _lp(f"  Scenarios: {scn_path}"
        + (f"  (coverage={coverage:.1%})" if coverage is not None else ""))

    # ── 2.5. S2/S3 metrics block ──────────────────────────────────────────────
    violations = list(getattr(vehicle, "violations", []))
    tw_misses  = dict(getattr(vehicle, "tw_misses", {}))
    dec_times  = list(events.get("decision_times", []))
    cmp_log    = list(events.get("cmp_log", []))

    # SIM2 — window hit rate and early/late miss split (replaces the v2
    # lateness-hours metric; the penalty is a fixed indicator, TW2)
    _n_cust    = len(full_data.get("C", []))
    _n_early   = sum(1 for m_ in tw_misses.values() if m_.get("type") == "early")
    _n_late    = sum(1 for m_ in tw_misses.values() if m_.get("type") == "late")
    _n_miss    = len(tw_misses)
    _hit_rate  = (1.0 - _n_miss / _n_cust) if _n_cust else None

    _by_type = {}
    for v in violations:
        _by_type[v["type"]] = _by_type.get(v["type"], 0) + 1

    # S3: ex-post Directive 2002/15/EC working-time compliance
    _pre_results = dict(
        states=vehicle.states, actions=vehicle.actions,
        durations_list=vehicle.durations,
        D_actual_list=vehicle.D_actual_list)
    try:
        compliance = check_directive_compliance(_pre_results, full_data)
    except Exception as _ce:
        compliance = dict(compliant=None, issues=[f"check failed: {_ce}"],
                          n_shifts=0, max_consec_work=0.0)

    # RH4: LP-vs-MIP agreement summary (only populated for solve_mode="both")
    if cmp_log:
        _agree = sum(1 for c in cmp_log if c.get("agree"))
        _deltas = [c["mip_score_of_lp_choice"] - c["mip_score_of_mip_choice"]
                   for c in cmp_log
                   if c.get("mip_score_of_lp_choice") is not None
                   and not c.get("agree")]
        cmp_summary = dict(
            n_stops=len(cmp_log),
            agreement_rate=_agree / len(cmp_log),
            mean_mip_delta_when_differ_h=(sum(_deltas) / len(_deltas)
                                          if _deltas else 0.0))
    else:
        cmp_summary = None

    metrics = dict(
        run_infeasible        = len(violations) > 0,
        n_violations          = len(violations),
        violations_by_type    = _by_type,
        violations            = violations,
        n_stranding           = _by_type.get("stranding", 0),
        n_hos_violations      = sum(v for k, v in _by_type.items()
                                    if k.startswith("hos")),
        tw_n_customers        = _n_cust,
        tw_n_misses           = _n_miss,
        tw_n_early            = _n_early,
        tw_n_late             = _n_late,
        tw_hit_rate           = _hit_rate,
        tw_misses_by_stop     = tw_misses,
        tw_penalty_h          = float(full_data.get("beta", 2.0)) * _n_miss,
        n_interventions       = len(events.get("interventions", [])),
        interventions         = events.get("interventions", []),
        n_repairs             = len(events.get("repairs", [])),
        repairs               = events.get("repairs", []),
        n_plan_violations     = len(events.get("plan_violations", [])),
        decision_time_mean_s  = (sum(dec_times) / len(dec_times)
                                 if dec_times else 0.0),
        decision_time_max_s   = max(dec_times) if dec_times else 0.0,
        offline_solve_time_s  = (method_meta or {}).get("solve_time"),
        directive_compliance  = compliance,
        lp_vs_mip             = cmp_summary,
    )

    _lp(f"  Metrics  : violations={len(violations)} "
        f"({', '.join(f'{k}:{v}' for k, v in _by_type.items()) or 'none'})"
        + (f"  tw_hit={metrics['tw_hit_rate']:.0%}"
           f" (early={_n_early}, late={_n_late})"
           if metrics['tw_hit_rate'] is not None else "  tw_hit=n/a")
        +
        f"  interventions={metrics['n_interventions']}"
        f"  repairs={metrics['n_repairs']}")
    if compliance.get("compliant") is False:
        _lp(f"  [!] Directive 2002/15/EC compliance issues "
            f"({len(compliance['issues'])}):")
        for iss in compliance["issues"][:5]:
            _lp(f"      {iss}")

    # ── 3. Save results JSON ──────────────────────────────────────────────────
    def _ser(o):
        """JSON-safe serialiser for numpy scalars, dicts, lists."""
        if isinstance(o, (int, float, bool, str, type(None))):
            return o
        if isinstance(o, dict):
            return {str(k): _ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_ser(v) for v in o]
        return str(o)

    payload = dict(
        run_id       = run_id,
        instance     = full_data.get("title", "unknown"),
        sim_arrival_h= arr,
        duration_h   = arr - T0,
        wall_clock_s = wall,
        oracle       = _ser(oracle),
        sim_trajectory = [
            dict(stop=s.stop, t_arr=round(s.t_arr, 4),
                 e_arr=round(s.e_arr, 2), cd=round(s.cd, 4),
                 sd=round(s.sd, 4), sw=round(s.sw, 4),
                 phi=s.phi, rho2_used=s.rho2_used,
                 ext_shift_used=getattr(s, "ext_shift_used", 0))
            for s in vehicle.states
        ],
        actions          = [_ser(a) for a in vehicle.actions],
        # Execution lists needed to re-render the figure later
        # (python plots.py <run_id>) without re-running the algorithm.
        td_list          = [round(float(t), 6) for t in vehicle.td_list],
        D_actual_list    = [round(float(d), 6) for d in vehicle.D_actual_list],
        durations_list   = [_ser(d) for d in vehicle.durations],
        scenario_summary = _ser(scn_summary),
        metrics          = _ser(metrics),
    )
    if method_meta:
        payload.update(_ser(method_meta))

    sol_path = paths["sol"]
    with open(sol_path, "w") as fj:
        json.dump(payload, fj, indent=2)
    _lp(f"  Solution : {sol_path}")

    # ── 4. Close log ──────────────────────────────────────────────────────────
    if log_fh and not log_fh.closed:
        log_fh.close()

    # ── 5. Build results dict ─────────────────────────────────────────────────
    results = dict(
        vehicle          = vehicle,
        states           = vehicle.states,
        actions          = vehicle.actions,
        scores_log       = scores_log or [],
        td_list          = vehicle.td_list,
        D_actual_list    = vehicle.D_actual_list,
        durations_list   = vehicle.durations,
        total_time       = arr,
        wall_clock       = wall,
        oracle           = oracle,
        metrics          = metrics,
        events           = events,
        scenario_tracker = tracker,
        log_path         = paths["log"],
        fig_path         = paths["fig"],
        sol_path         = sol_path,
        scn_path         = scn_path,
        run_id           = run_id,
    )

    # ── 6. Print schedule tables ──────────────────────────────────────────────
    print_simulation_log(results, full_data)
    print_oracle_log(oracle, full_data)

    # ── 7. Feasibility check ──────────────────────────────────────────────────
    feas_ok, issues = check_simulation_feasibility(results, full_data)
    if not feas_ok:
        print(f"\n  [!] TRAJECTORY INFEASIBLE -- {len(issues)} violation(s):")
        for iss in issues[:10]:
            print(f"     {iss}")

    # ── 8. Save figure (opt-in; plot later with `python plots.py <run_id>`) ──
    if plot:
        fig_path = paths["fig"]
        try:
            plot_simulation_results(
                results   = results,
                full_data = full_data,
                title     = run_id,
                save      = True,
                show      = False,
            )
            if verbose:
                print(f"  Figure   : {fig_path}")
        except Exception as _pe:
            if verbose:
                print(f"  Figure   : could not save ({_pe})")
    elif verbose:
        print(f"  Figure   : skipped — render later with "
              f"`python plots.py {run_id}`")

    return results