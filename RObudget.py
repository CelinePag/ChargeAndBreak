"""
RObudget.py — Budgeted (Bertsimas–Sim) robust plan via column-and-constraint
generation, offline, no recourse ("ROBU")
=============================================================================
Improved static robust counterpart of RO.py.  Instead of the Soyster box
(EVERY leg at its worst case simultaneously), the plan is protected against
the budget uncertainty set

    U_Gamma = { xi : xi_i = 1 + z_i^+ (XI_MAX - 1) - z_i^- (1 - XI_MIN),
                z^+, z^- in [0,1]^N,  sum z^+ <= Gamma,  sum z^- <= Gamma }

i.e. at most Gamma legs' worth of deviation mass is slow (time side) and at
most Gamma legs' worth is fast/energy-hungry (SOC side) in any single
realization (Bertsimas & Sim 2004, "The Price of Robustness").  The budget
uses the classic literature value

    Gamma = ceil( 1 + z_{1-eps} * sqrt(N) )        (capped at N)

which guarantees a constraint-violation probability of at most eps for a
constraint aggregating N independent symmetric deviations (Bertsimas & Sim
2004, Prop. 2 bound  P <= 1 - Phi((Gamma-1)/sqrt(n))).

Solution method — cutting-plane / column-and-constraint generation
------------------------------------------------------------------
U_Gamma has combinatorially many relevant vertices, so the robust counterpart
is solved iteratively (Zeng & Zhao 2013 master structure; Mutapcic & Boyd
2009 / Paetzold & Schoebel 2020 robustification–pessimization loop):

  1. ROBUSTIFICATION (master): min–max over the finite scenario set found so
     far — exactly build_2sp_model(objective="max", share_durations=True),
     the same model the box RO solves with one scenario.  Adding a scenario
     appends one state-chain copy + constraints (the "column-and-constraint"
     step); the plan binaries and durations stay shared (static plan).
  2. PESSIMIZATION (adversary): with the plan FIXED, all reset points are
     known, so the worst attack on each constraint family is a greedy vertex
     of U_Gamma — deviate the Gamma legs with the largest deviation inside
     that constraint's window.  Candidates are generated per continuous-
     driving window, per shift window, per customer prefix (time windows),
     per between-charges segment (SOC), plus global (weekly cap / arrival).
     Each candidate is certified EXACTLY against the true model semantics by
     re-solving a one-scenario copy of the model with all first-stage
     variables fixed to the plan (small MIP: only outcome variables free).
  3. Violated candidates are appended as new scenarios (feasibility cuts);
     if none violates, the worst candidate objective is compared against the
     master objective (optimality cut) and the loop stops when the adversary
     can no longer hurt the plan.

Execution is identical to RO.py: the final plan (structure + durations) is
executed AS IS via recourse.run_plan_static — no online recourse.

Integration
-----------
  from RObudget import run_robu
  results = run_robu(full_data, D_real, E_real)

  Or via runner_dispatch:
    python runner_dispatch.py instances/RmediumCfewTmedium_7.json ROBU
"""

from __future__ import annotations

import datetime
import importlib
import math
import os
import sys
import time
from statistics import NormalDist

import pyomo.environ as pyo

from recourse  import run_plan_static
from settings  import (ecr as _ecr, V_NOM, XI_MIN, XI_MAX,
                       TRAVEL_TIME_CV, GUARD_QUANTILE)
from runner    import finalize_run

_twosp = importlib.import_module("2SP")

# Default probabilistic guarantee target for the classic B–S budget
ROBU_EPS_DEFAULT      = 0.01
ROBU_MAX_ITER_DEFAULT = 12
ROBU_MAX_CUTS_PER_IT  = 3      # feasibility cuts appended per iteration
ROBU_EVAL_TIME_LIMIT  = 120    # s per fixed-plan certification solve


# ══════════════════════════════════════════════════════════════════════════════
# BUDGET AND SCENARIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def classic_gamma(n_legs: int, eps: float = ROBU_EPS_DEFAULT) -> int:
    """
    Classic Bertsimas–Sim protection level: Gamma = 1 + z_{1-eps}·sqrt(n),
    the smallest budget with constraint-violation probability <= eps under
    independent symmetric deviations (B–S 2004), capped at n (full box).
    """
    z = NormalDist().inv_cdf(1.0 - eps)
    return int(min(n_legs, math.ceil(1.0 + z * math.sqrt(n_legs))))


def _leg_tables(full_data: dict) -> dict:
    """
    Per-leg nominal values and deviation magnitudes.

      D_nom  : nominal travel time (h)
      D_hat  : upward time deviation  D_nom·(XI_MAX − 1)
      E_nom  : nominal energy (kWh)
      E_hat  : upward energy deviation E(XI_MIN) − E_nom  (fast = hungry)
      E_slow : energy at the slow extreme E(XI_MAX) (used on up-deviated legs)
      E_fast : energy at the fast extreme E(XI_MIN) (used on down-deviated legs)

    Energy per leg follows the same speed→consumption mapping as
    scenarios.generate_scenarios: E = L · ecr(L / D).
    """
    N     = full_data["N"]
    D_nom = full_data["D"]
    E_nom = full_data["E"]
    km    = full_data.get("km", {})

    tab = dict(D_nom={}, D_hat={}, E_nom={}, E_hat={}, E_slow={}, E_fast={})
    for i in range(N):
        d = D_nom.get(i, 0.0)
        L = km.get(i, d * V_NOM)
        e_nom = E_nom.get(i, 0.0)

        def _e(mult):
            d_m = d * mult
            if d_m <= 0 or L <= 0:
                return e_nom
            return L * _ecr(L / d_m)

        tab["D_nom"][i]  = d
        tab["D_hat"][i]  = d * (XI_MAX - 1.0)
        tab["E_nom"][i]  = e_nom
        tab["E_fast"][i] = _e(XI_MIN)
        tab["E_slow"][i] = _e(XI_MAX)
        tab["E_hat"][i]  = max(0.0, tab["E_fast"][i] - e_nom)
    return tab


def _nominal_scenario(full_data: dict) -> dict:
    return dict(D=dict(full_data["D"]), E=dict(full_data["E"]))


def _make_scenario(tab: dict, n_legs: int,
                   up_legs: set, down_legs: set) -> dict:
    """
    Vertex of U_Gamma as a scenario dict: up-deviated legs at XI_MAX (slow
    time, slow-speed energy), down-deviated legs at XI_MIN (fast time,
    fast-speed energy), all other legs nominal.
    """
    D_s, E_s = {}, {}
    for i in range(n_legs):
        if i in up_legs:
            D_s[i] = tab["D_nom"][i] * XI_MAX
            E_s[i] = tab["E_slow"][i]
        elif i in down_legs:
            D_s[i] = tab["D_nom"][i] * XI_MIN
            E_s[i] = tab["E_fast"][i]
        else:
            D_s[i] = tab["D_nom"][i]
            E_s[i] = tab["E_nom"][i]
    return dict(D=D_s, E=E_s)


def _scen_key(up_legs: set, down_legs: set) -> tuple:
    return (tuple(sorted(up_legs)), tuple(sorted(down_legs)))


def _seed_scenarios(full_data: dict, tab: dict, gamma: int) -> list:
    """Plan-independent worst-case vertices used to PRIME the scenario set.

    Starting the cutting-plane loop from the nominal scenario alone forces the
    first master to build a plan with no robustness margin, so the adversary
    then has to teach it every worst case one feasibility cut at a time — the
    dominant cost on medium/long routes, where each master solve is expensive.
    Priming with the globally slowest-Gamma legs (latest arrival / tightest
    HoS) and the hungriest-Gamma legs (deepest SOC draw) means the very first
    plan already anticipates the extreme time and energy realisations, so far
    fewer iterations are needed.

    Each returned (up, down) pair is a genuine vertex of U_Gamma, so seeding
    only makes the plan more robust — it can never make it wrong.  Kept to a
    handful so the first extensive-form master stays tractable on long routes.
    """
    all_legs = list(range(full_data["N"]))
    slow   = _top_legs(all_legs, tab["D_hat"], gamma)   # latest arrival / HoS
    hungry = _top_legs(all_legs, tab["E_hat"], gamma)   # deepest SOC draw
    seeds = []
    for up, down in ((slow, set()), (set(), hungry), (slow, hungry)):
        if up or down:
            seeds.append((up, down))
    return seeds


def _apply_warmstart(model, fs: dict):
    """Prime a freshly built master's first-stage variables with the previous
    iteration's plan, so Gurobi begins from a known-good incumbent (MIP start).

    Only the first-stage binaries and shared durations are set; the newly added
    scenario's second stage is left for the solver to fill (Gurobi accepts a
    partial start).  Best-effort — any index mismatch is swallowed so a warm
    start can never block a solve."""
    def _set(var, idx, val):
        try:
            var[idx].value = val
        except Exception:
            pass

    for k in ("y", "x_b45", "x_b15", "x_b30", "rho1", "rho2",
              "phi", "z", "q_ext"):
        var = getattr(model, k, None)
        if var is not None:
            for i, v in fs.get(k, {}).items():
                _set(var, i, v)
    for k in ("tauc", "taub", "taur", "sigma"):
        var = getattr(model, k, None)
        if var is not None:
            for i, v in fs.get(k, {}).items():
                _set(var, (i, 0), v)


# ══════════════════════════════════════════════════════════════════════════════
# PESSIMIZATION — CANDIDATE GENERATION (greedy vertices per constraint family)
# ══════════════════════════════════════════════════════════════════════════════

def _top_legs(legs: list, weight: dict, gamma: int) -> set:
    """The (at most) gamma legs with the largest deviation weight."""
    ranked = sorted(legs, key=lambda j: weight.get(j, 0.0), reverse=True)
    return {j for j in ranked[:gamma] if weight.get(j, 0.0) > 1e-12}


def _windows_from_resets(reset_stops: list, N: int) -> list:
    """
    Leg windows between consecutive reset stops: departing a reset stop the
    accumulator is zero, so the window ending at the next reset (or at the
    destination) contains legs [r, r'-1].
    """
    bounds = sorted(set([0] + list(reset_stops) + [N]))
    wins = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        legs = list(range(a, b))
        if legs:
            wins.append(legs)
    return wins


def generate_candidates(full_data: dict, plan: list, tab: dict,
                        gamma: int) -> list:
    """
    Plan-aware greedy attack candidates (vertices of U_Gamma), one per
    constraint family and window.  Returns a list of dicts:
        {tag, up (set), down (set)}
    Exactness note: for a FIXED plan every listed constraint is a sum of leg
    contributions over a KNOWN window, so the worst case over the budget
    polytope is attained by fully deviating the Gamma largest legs of that
    window (vertex of a polytope maximizing a nonnegative linear function).
    """
    N      = full_data["N"]
    C_set  = set(full_data["C"])
    cands  = []

    cd_resets   = [e["i"] for e in plan
                   if e.get("b45") or e.get("b30") or e.get("rho1") or e.get("rho2")]
    rest_resets = [e["i"] for e in plan if e.get("rho1") or e.get("rho2")]
    chg_stops   = [e["i"] for e in plan if e.get("y") and e.get("tauc", 0.0) > 1e-6]

    # Global (weekly working cap + worst-case arrival) — all legs
    cands.append(dict(tag="global",
                      up=_top_legs(list(range(N)), tab["D_hat"], gamma),
                      down=set()))

    # Continuous-driving windows (cd — resets at b45/b30/r1/r2)
    for legs in _windows_from_resets(cd_resets, N):
        cands.append(dict(tag=f"cd[{legs[0]}-{legs[-1]}]",
                          up=_top_legs(legs, tab["D_hat"], gamma),
                          down=set()))

    # Shift windows (sd/sw/spread — reset at daily rests r1/r2)
    for legs in _windows_from_resets(rest_resets, N):
        cands.append(dict(tag=f"shift[{legs[0]}-{legs[-1]}]",
                          up=_top_legs(legs, tab["D_hat"], gamma),
                          down=set()))

    # Customer time windows — late side: slow the prefix before each customer
    for c in sorted(C_set):
        legs = list(range(0, c))
        if legs:
            cands.append(dict(tag=f"tw[{c}]",
                              up=_top_legs(legs, tab["D_hat"], gamma),
                              down=set()))

    # SOC segments between consecutive charging stops — fast/energy-hungry legs
    for legs in _windows_from_resets(chg_stops, N):
        cands.append(dict(tag=f"soc[{legs[0]}-{legs[-1]}]",
                          up=set(),
                          down=_top_legs(legs, tab["E_hat"], gamma)))

    # Combined worst: slow prefix for arrival AND hungry legs for energy
    # (the two budgets are separate, so this is a valid vertex of U_Gamma)
    cands.append(dict(tag="combined",
                      up=_top_legs(list(range(N)), tab["D_hat"], gamma),
                      down=_top_legs(list(range(N)), tab["E_hat"], gamma)))

    # Deduplicate (identical vertex sets from overlapping windows)
    seen, uniq = set(), []
    for c in cands:
        key = _scen_key(c["up"], c["down"])
        if key in seen or (not c["up"] and not c["down"]):
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


# ══════════════════════════════════════════════════════════════════════════════
# PESSIMIZATION — EXACT CERTIFICATION OF ONE CANDIDATE (fixed-plan solve)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_first_stage(model, full_data: dict) -> dict:
    """First-stage values of the solved master (shared binaries + durations)."""
    K_set = set(full_data["K"])
    I     = list(full_data["I"])

    def _b(var, i):
        try:    return int(round(pyo.value(var[i])))
        except Exception: return 0

    fs = dict(
        y     = {i: _b(model.y, i) for i in K_set},
        x_b45 = {i: _b(model.x_b45, i) for i in I},
        x_b15 = {i: _b(model.x_b15, i) for i in I},
        x_b30 = {i: _b(model.x_b30, i) for i in I},
        rho1  = {i: _b(model.rho1, i) for i in I},
        rho2  = {i: _b(model.rho2, i) for i in I},
        phi   = {i: _b(model.phi, i) for i in I},
        z     = {i: _b(model.z, i) for i in I},
        q_ext = {i: _b(model.q_ext, i) for i in I},
        tauc  = {i: max(0.0, float(pyo.value(model.tauc[i, 0]))) for i in K_set},
        taub  = {i: max(0.0, float(pyo.value(model.taub[i, 0]))) for i in I},
        taur  = {i: max(0.0, float(pyo.value(model.taur[i, 0]))) for i in I},
        sigma = {i: int(round(pyo.value(model.sigma[i, 0]))) for i in K_set},
    )
    return fs


def _certify_candidate(full_data: dict, fs: dict, scenario: dict,
                       eval_time_limit: int = ROBU_EVAL_TIME_LIMIT,
                       mip_gap: float = 0.005) -> tuple:
    """
    Evaluate the committed plan under ONE scenario with the true model
    semantics: build a one-scenario copy of the model and FIX every
    first-stage variable to the plan.  Only outcome variables (states, delta,
    PWL segments, ...) remain free, so the solve is small and fast.

    Returns (feasible: bool, obj: float) — obj is ta_N + beta·sum(delta)
    under that scenario, +inf when the plan is broken by the scenario.
    """
    m = _twosp.build_2sp_model(full_data, [scenario], objective="max",
                               share_durations=True)
    K_set = set(full_data["K"])
    I     = list(full_data["I"])

    for i in K_set:
        m.y[i].fix(fs["y"][i])
        m.tauc[i, 0].fix(fs["tauc"][i])
        m.sigma[i, 0].fix(fs["sigma"][i])
    for i in I:
        m.x_b45[i].fix(fs["x_b45"][i])
        m.x_b15[i].fix(fs["x_b15"][i])
        m.x_b30[i].fix(fs["x_b30"][i])
        m.rho1[i].fix(fs["rho1"][i])
        m.rho2[i].fix(fs["rho2"][i])
        m.phi[i].fix(fs["phi"][i])
        m.z[i].fix(fs["z"][i])
        m.q_ext[i].fix(fs["q_ext"][i])
        m.taub[i, 0].fix(fs["taub"][i])
        m.taur[i, 0].fix(fs["taur"][i])

    try:
        info, _status = _twosp.solve_2sp(m, time_limit=eval_time_limit,
                                         mip_gap=mip_gap, tee=False)
    except (ValueError, RuntimeError):
        # Solver aborted with no incumbent within the eval time limit —
        # certification inconclusive: skip this candidate (do not cut).
        return True, -float("inf")
    if not info["feasible"]:
        return False, float("inf")
    return True, float(info["obj"])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_robu(full_data: dict,
             D_real: list,
             E_real: list,
             cv: float         = TRAVEL_TIME_CV,
             time_limit: int   = 2 * 3600,
             wall_limit: int | None = None,
             mip_gap: float    = 0.005,
             eps: float        = ROBU_EPS_DEFAULT,
             gamma: int | None = None,
             max_iter: int     = ROBU_MAX_ITER_DEFAULT,
             max_cuts: int     = ROBU_MAX_CUTS_PER_IT,
             seed_scenarios: bool = True,
             warmstart: bool   = True,
             tee: bool         = True,
             verbose: bool     = True,
             run_id: str       = None,
             oracle_tee: bool  = True,
             supervised: bool  = False,
             prune_quantile: float | None = GUARD_QUANTILE,
             **kwargs) -> dict:
    """
    Solve the budgeted (Bertsimas–Sim) robust plan by column-and-constraint
    generation and execute it AS IS on D_real/E_real — no online recourse.

    Parameters (beyond the run_ro-compatible ones)
    ----------------------------------------------
    eps      : target per-constraint violation probability for the classic
               budget Gamma = 1 + z_{1-eps}·sqrt(N)  (default 0.01)
    gamma    : explicit budget override; None → classic value from eps
    max_iter : maximum robustification–pessimization iterations
    max_cuts : violated scenarios appended per iteration (feasibility cuts)
    time_limit / mip_gap : per master solve (same role as in run_ro)
    wall_limit : total solve wall-clock budget (s) across ALL iterations; the
               loop stops before a master that would exceed it, returning the
               best plan so far (robu_converged=False → reported as "unsolved",
               not infeasible).  None → 2·time_limit.  Prevents the runaway
               multi-hour solves seen on medium/long routes.
    seed_scenarios : prime the scenario set with plan-independent worst-case
               vertices so fewer feasibility-cut iterations are needed
    warmstart : feed each master the previous iteration's plan as a Gurobi MIP
               start (consecutive masters differ only by appended scenarios)

    Returns
    -------
    dict — canonical results dict (same schema as run_ro / run_greedy)
    """
    t_wall_start = time.perf_counter()
    N            = full_data["N"]
    T_START      = full_data.get("T_START", 8.0)
    label        = full_data.get("label", "robu")
    title        = full_data.get("title", "inst")

    assert len(D_real) == N, f"D_real length {len(D_real)} != N={N}"
    assert len(E_real) == N, f"E_real length {len(E_real)} != N={N}"

    if gamma is None:
        gamma = classic_gamma(N, eps)
    gamma = int(min(max(1, gamma), N))
    if wall_limit is None:
        wall_limit = 2 * time_limit

    for d in ("logs", "figures", "solutions"):
        os.makedirs(d, exist_ok=True)
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{title}_ROBU_{ts}"
    paths = dict(
        log = os.path.join("logs",      f"{run_id}.txt"),
        fig = os.path.join("figures",   f"{run_id}.png"),
        sol = os.path.join("solutions", f"{run_id}.json"),
        scn = os.path.join("logs",      f"{run_id}_scenarios.json"),
    )
    log = open(paths["log"], "w", encoding="utf-8")

    def _p(msg):
        if verbose: print(msg)
        try: print(msg, file=log, flush=True)
        except Exception: pass

    _p("=" * 65)
    _p(f"  ROBU SOLVE START   ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    _p(f"  Instance : {label}   run_id={run_id}")
    _p(f"  Route    : {N} stops  departure={T_START:.0f}:00")
    _p(f"  Model    : budgeted RO (Bertsimas-Sim, Gamma={gamma} of {N} legs, "
       f"eps={eps:g}), C&CG cutting-plane, no recourse")
    _p("=" * 65)

    tab      = _leg_tables(full_data)
    scen_set = [_nominal_scenario(full_data)]
    scen_keys = {_scen_key(set(), set())}
    if seed_scenarios:
        for up, down in _seed_scenarios(full_data, tab, gamma):
            key = _scen_key(up, down)
            if key not in scen_keys:
                scen_set.append(_make_scenario(tab, N, up, down))
                scen_keys.add(key)
        _p(f"  Primed scenario set: {len(scen_set)} vertices "
           f"(nominal + {len(scen_set) - 1} worst-case).")
    _p(f"  Budget wall-clock limit: {wall_limit}s "
       f"(<= {time_limit}s per master solve).")

    theta = None
    plan  = None
    prev_fs = None
    converged   = False
    n_cuts_feas = 0
    n_cuts_opt  = 0
    t_solve_total = 0.0
    worst_certified = None

    for it in range(1, max_iter + 1):
        # ── Robustification: min-max over the scenario set found so far ──────
        t0 = time.perf_counter()
        elapsed = time.perf_counter() - t_wall_start
        remaining = wall_limit - elapsed
        # keep enough headroom for a certification pass + plan execution; stop
        # cleanly (with the best plan so far) rather than overrun the budget
        if plan is not None and remaining < ROBU_EVAL_TIME_LIMIT:
            _p(f"  [it {it}] wall-clock budget spent "
               f"({elapsed:.0f}s / {wall_limit}s) — stopping with the "
               f"iteration-{it - 1} plan (robu_converged=False).")
            break
        master_tl = int(max(ROBU_EVAL_TIME_LIMIT, min(time_limit, remaining)))
        model = _twosp.build_2sp_model(full_data, scen_set, objective="max",
                                       share_durations=True)
        do_warm = warmstart and prev_fs is not None
        if do_warm:
            _apply_warmstart(model, prev_fs)
        try:
            info, status = _twosp.solve_2sp(model, time_limit=master_tl,
                                            mip_gap=mip_gap, tee=tee,
                                            warmstart=do_warm, heuristics=0.2)
        except (ValueError, RuntimeError) as e:
            # Master hit the time limit with NO incumbent (Pyomo 'aborted').
            t_master = time.perf_counter() - t0
            t_solve_total += t_master
            _p(f"  [it {it}] master solve aborted with no incumbent "
               f"({type(e).__name__}) after {t_master:.1f}s.")
            if plan is not None:
                _p(f"      Falling back to the iteration-{it-1} plan "
                   f"(robu_converged=False).")
                break
            _p("  No master solution at all — aborting.")
            log.close()
            return dict(feasible=False, status="aborted",
                        total_time=float("inf"),
                        wall_clock=time.perf_counter() - t_wall_start)
        t_master = time.perf_counter() - t0
        t_solve_total += t_master

        if not info["feasible"]:
            _p(f"  [it {it}] master Status : infeasible ({t_master:.1f}s) — "
               f"no plan is feasible for the generated scenarios.")
            _p("  No feasible robust plan under the budget set — aborting.")
            log.close()
            return dict(feasible=False, status=status,
                        total_time=float("inf"),
                        wall_clock=time.perf_counter() - t_wall_start)

        theta = info["obj"]
        plan  = _twosp.extract_2sp_full_schedule(model, full_data)
        fs    = _extract_first_stage(model, full_data)
        prev_fs = fs                       # MIP start for the next master
        _p(f"  [it {it}] master: |scen|={len(scen_set)}  theta={theta:.3f}h  "
           f"status={status}  ({t_master:.1f}s)")

        # ── Pessimization: greedy vertex attacks, certified on the model ─────
        t0 = time.perf_counter()
        candidates = generate_candidates(full_data, plan, tab, gamma)
        violated   = []
        worst_obj, worst_cand = -float("inf"), None

        for cand in candidates:
            key = _scen_key(cand["up"], cand["down"])
            if key in scen_keys:
                continue
            scen = _make_scenario(tab, N, cand["up"], cand["down"])
            feas, obj = _certify_candidate(full_data, fs, scen,
                                           mip_gap=mip_gap)
            if not feas:
                violated.append((cand, scen))
                if len(violated) >= max_cuts:
                    break
            elif obj > worst_obj:
                worst_obj, worst_cand = obj, cand

        t_pes = time.perf_counter() - t0
        t_solve_total += t_pes

        if violated:
            tags = ", ".join(c["tag"] for c, _ in violated)
            _p(f"  [it {it}] adversary: {len(violated)} feasibility cut(s) "
               f"[{tags}]  ({t_pes:.1f}s)")
            for cand, scen in violated:
                scen_set.append(scen)
                scen_keys.add(_scen_key(cand["up"], cand["down"]))
            n_cuts_feas += len(violated)
            continue

        tol = max(0.05, mip_gap * abs(theta))
        if worst_cand is not None and worst_obj > theta + tol:
            _p(f"  [it {it}] adversary: optimality cut [{worst_cand['tag']}] "
               f"obj={worst_obj:.3f}h > theta={theta:.3f}h  ({t_pes:.1f}s)")
            scen_set.append(_make_scenario(tab, N,
                                           worst_cand["up"], worst_cand["down"]))
            scen_keys.add(_scen_key(worst_cand["up"], worst_cand["down"]))
            n_cuts_opt += 1
            continue

        worst_certified = max(theta, worst_obj if worst_cand else theta)
        _p(f"  [it {it}] adversary: no violation, worst candidate "
           f"obj={worst_obj:.3f}h <= theta+tol — CONVERGED  ({t_pes:.1f}s)")
        converged = True
        break

    if not converged:
        _p(f"  [!] Loop ended without adversary certification "
           f"(max_iter={max_iter} exhausted or master aborted); "
           f"executing the last available plan (robu_converged=False).")

    _p(f"\n  ROBU objective (worst-case arrival over U_Gamma) : "
       f"{theta:.3f} h   Gamma={gamma}  scenarios={len(scen_set)}  "
       f"cuts: {n_cuts_feas} feas / {n_cuts_opt} opt   "
       f"solve {t_solve_total:.1f}s")
    _p(f"  plan: {sum(1 for e in plan if e['y'])} chg / "
       f"{sum(1 for e in plan if e['break_type'])} brk / "
       f"{sum(1 for e in plan if e['rest_type'])} rst")

    # ── Execute the FIXED plan AS IS (static, no recourse) ────────────────────
    _p(f"\n  Executing ROBU plan AS IS (static, no recourse)...")
    vehicle, tracker, events = run_plan_static(
        full_data      = full_data,
        plan           = plan,
        D_real         = D_real,
        E_real         = E_real,
        method_name    = "ROBU",
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
    _p(f"  ROBU SIMULATION COMPLETE")
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
            method         = "ROBU",
            cv             = cv,
            gamma          = gamma,
            robu_eps       = eps,
            n_scenarios    = len(scen_set),
            robu_iterations= it,
            robu_converged = converged,
            robu_cuts_feas = n_cuts_feas,
            robu_cuts_opt  = n_cuts_opt,
            robu_seeded    = bool(seed_scenarios),
            robu_warmstart = bool(warmstart),
            robu_wall_limit= wall_limit,
            ro_obj         = theta,
            robu_certified_obj = worst_certified,
            ro_status      = info.get("status"),
            ro_optimal     = info.get("optimal"),
            solve_time     = t_solve_total,
            supervised     = supervised,
            prune_quantile = prune_quantile,
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from instance_io import load_instance_json

    # Usage: python RObudget.py <json_file> [time_limit_s] [gamma]
    json_file = sys.argv[1] if len(sys.argv) > 1 else None
    time_lim  = int(sys.argv[2]) if len(sys.argv) > 2 else 2 * 3600
    gamma_cli = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if json_file is None:
        print("Usage: python RObudget.py <json_file> [time_limit_s] [gamma]")
        sys.exit(1)

    full_data, D_real, E_real, cv_file = load_instance_json(json_file)

    results = run_robu(
        full_data,
        D_real     = D_real,
        E_real     = E_real,
        cv         = cv_file,
        time_limit = time_lim,
        gamma      = gamma_cli,
        tee        = True,
        verbose    = True,
        oracle_tee = True,
    )

    print(f"\n  ROBU arrival : {results['total_time']:.3f} h")
    print(f"  Wall clock   : {results['wall_clock']:.1f} s")
    print(f"  Figure       : {results['fig_path']}")
    print(f"  Solution     : {results['sol_path']}")
