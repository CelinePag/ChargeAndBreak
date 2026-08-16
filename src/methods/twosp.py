"""
twosp.py — Two-stage stochastic programming for BET scheduling ("2SP")
=============================================================
Implements the deterministic equivalent (extensive form) of a two-stage SP
over a sample of S travel-time / energy scenarios.

Mathematical formulation
------------------------
Uncertainty model:
    D̃_i = D_i · ξ_i,   E̊_i drawn from ECR(speed) model
    Scenarios ω_1..ω_S drawn live at solve time via scenarios.generate_scenarios().

Stage decomposition:
    First stage  (here-and-now, binary):
        y[i]          — whether to charge at CS stop i
        x_b45/b15/b30[i] — which break type to take at stop i (if any)
        rho1/rho2[i]  — which rest type to take at stop i (if any)
        phi[i]        — split-break credit tracker (derived from breaks)

    Second stage (wait-and-see, per scenario s):
        ta[i,s], td[i,s]        — arrival/departure times
        ea[i,s], ed[i,s]        — battery SOC
        cd[i,s], sd[i,s], sw[i,s] — HoS accumulators
        tauc[i,s], taub[i,s], taur[i,s] — activity durations
        sigma[i,s], v[i,s]      — sequential mode / activity indicator (CS)
        mu_a/mu_d[i,k,s]        — SOS2 helpers for PWL charging curve
        lam_a/lam_d[i,k,s]      — PWL weighting variables
        l1/l2/l4, u, p          — linearisation auxiliaries

Objective:
    min (1/S) Σ_s ( ta[N, s] + β Σ_i delta[i, s] )
    (expected arrival time + expected fixed window penalties; the
    out-of-window indicators delta are scenario-indexed outcomes, TW5)

Linking constraints:
    The second-stage break/rest durations are bounded below by the minimum
    required for the chosen first-stage type:
        taub[i,s] >= Tb45 · x_b45[i]   (etc.)
    HoS accumulator propagation uses scenario travel times D[i,s].
    Charging amount (tauc[i,s]) adapts per scenario via the PWL recourse.

Execution (SP1 — online duration recourse)
------------------------------------------
After solving, only the FIRST-STAGE BINARY decisions are retained.  During
simulation, at each stop the remaining binaries are held fixed and the
continuous activity durations are re-optimised over [i, N] with the realized
state and nominal remaining travel times; only the durations at the current
stop are executed.  If this fixed-structure problem is infeasible, a repair
MILP in which binary activities may be ADDED but never removed is solved
(heavily penalising each addition); if repair also fails, the run is recorded
as a plan violation and the safety supervisor takes over.  See recourse.py.
The plan's structure is committed offline, its durations adapt online, and
the repair frequency is itself a reported robustness metric (S2).

Scenarios
---------
Scenarios are generated live at solve time via scenarios.generate_scenarios()
(start_stop=0, end_stop=N, cv=cv) — the same mechanism LA uses — rather
than read from a precomputed pool.  Runs are therefore not tied to a fixed
scenario sample across repeated calls; pass `scenario_seed` for reproducibility.

Integration with the framework
-------------------------------
  from src.methods import twosp
  results = twosp.run_2sp(full_data, D_real, E_real, n_scenarios=10)

  Or via runner_dispatch.py:
    python -m src.simulation.runner_dispatch instances/RmediumCfew_7.json 2SP

References
----------
Birge, J.R. & Louveaux, F. (2011). Introduction to Stochastic Programming.
Springer.  Chapter 2: The Two-Stage Model.
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from typing import Optional

import pyomo.environ as pyo

from src.methods.MILP      import (
    _declare_common_params,
    _solve_quiet,
)
from src.simulation.scenarios import ScenarioTracker, generate_scenarios
from src.methods.recourse  import run_plan_with_recourse
from src.simulation.runner    import finalize_run
from src.settings  import GUARD_QUANTILE, BETA_TW
from src import paths as _paths


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — EXTENSIVE FORM MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_2sp_model(data: dict, scenarios: list[dict],
                    objective: str = "mean",
                    share_durations: bool = False) -> pyo.ConcreteModel:
    """
    Build the extensive-form 2SP Pyomo model.

    Incorporates the July-2026 model revision: g break-credit (M2), linear
    charging-work tauc−g (M3), sequential-mode rules (M4), shift spread (M5),
    10 h driving extension (M6), and the weekly working cap (M9) — all per
    scenario where second-stage.  v3 windows (TW1/TW2/TW5): no idle waiting,
    service starts at arrival; the out-of-window indicators delta^(k) are
    SCENARIO-INDEXED binaries (an outcome, not a first-stage decision) and
    each scenario's objective term is ta_N^(k) + beta·Σ delta^(k).

    Parameters
    ----------
    data      : full route data dict from instances.make_data()
    scenarios : list of S scenario dicts, each with keys "D" and "E"
                (global leg index → float), from generate_scenarios().
    objective : "mean" — expected objective over scenarios (2SP);
                "max"  — epigraph min–max over scenarios (the STATIC robust
                plan reuses this exact model with mean replaced by an
                epigraph max; with a single worst-case scenario this reduces
                to the deterministic MILP on the box worst case).
    share_durations : C3 — when True, the second-stage activity durations
                (tauc, taub, taur) are tied across all scenarios by
                non-anticipativity constraints, so the plan commits a SINGLE
                duration vector (in addition to the shared binaries).  This is
                the static robust counterpart (no online recourse); the state
                variables (ta, ea, cd, …) and the out-of-window indicators
                delta remain scenario-indexed outcomes.  Default False (2-SP,
                which adapts durations online — §5.5).

    Returns
    -------
    pyo.ConcreteModel — the unsolved extensive form model
    """
    n_scen  = len(scenarios)
    S_list  = list(range(n_scen))

    m = pyo.ConcreteModel()

    # ── Standard sets and params (reused from MILP.py) ────────────────────────
    N, C, K, R, Rseg, _lb, _ub = _declare_common_params(m, data)

    K_set   = set(K)
    C_set   = set(C)
    L_set   = set(data.get("L", []))
    I_list  = list(data["I"])
    S_svc   = data["S"]          # service times at customer stops (Python dict)
    M_lay   = data.get("M_lay", {}) or {}
    M_big   = data["M_big"]
    M_drv   = data["M_drv"]
    M_sd    = data["M_sd"]
    M_sw    = data["M_sw"]
    M_h     = data.get("M_h", data.get("Tspr2", 15.0))
    hard_tw = bool(data.get("hard_tw", False))
    rho_bar = int(data.get("rho_bar", 3))
    ext_bar = int(data.get("ext_bar", 2))

    # ── Scenario set and per-scenario travel times / energies ─────────────────
    m.Scen = pyo.Set(initialize=S_list, ordered=True)

    m.D_sc = pyo.Param(
        m.Legs, m.Scen,
        initialize={(i, s): scenarios[s]["D"].get(i, data["D"].get(i, 0.0))
                    for s in S_list for i in range(N)},
        default=0.0,
    )
    m.E_sc = pyo.Param(
        m.Legs, m.Scen,
        initialize={(i, s): scenarios[s]["E"].get(i, data["E"].get(i, 0.0))
                    for s in S_list for i in range(N)},
        default=0.0,
    )

    # ── First-stage variables (binary, shared across all scenarios) ───────────
    m.y      = pyo.Var(m.Kset, domain=pyo.Binary)
    m.x_b45  = pyo.Var(m.I, domain=pyo.Binary)
    m.x_b15  = pyo.Var(m.I, domain=pyo.Binary)
    m.x_b30  = pyo.Var(m.I, domain=pyo.Binary)
    m.rho1   = pyo.Var(m.I, domain=pyo.Binary)
    m.rho2   = pyo.Var(m.I, domain=pyo.Binary)
    m.phi    = pyo.Var(m.I, domain=pyo.Binary)
    # M6 — extension declaration is structural (first-stage)
    m.z      = pyo.Var(m.I, domain=pyo.Binary)
    m.q_ext  = pyo.Var(m.I, domain=pyo.Binary)

    # Fix no activities at origin and destination
    for _v in [m.x_b45, m.x_b15, m.x_b30, m.rho1, m.rho2]:
        _v[0].fix(0)
        _v[N].fix(0)

    # 8.3 no-split axis — see MILP._drop_split_break.  The split blocks are
    # first-stage here too, so fixing them out leaves the scenario sub-problems
    # untouched and phi collapses to 0 through (26)–(29).
    if not m._allow_split:
        for i in m.I:
            m.x_b15[i].fix(0)
            m.x_b30[i].fix(0)

    # ── Second-stage variables (per scenario, indexed by (stop, scenario)) ────
    m.ta       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.td       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.ea       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.ed       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.cd       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.sd       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.sw       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.tauc     = pyo.Var(m.Kset, m.Scen, domain=pyo.NonNegativeReals)
    m.taub     = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.taub_hat = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.taur     = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.lam_a    = pyo.Var(m.Kset, m.Rset, m.Scen, domain=pyo.NonNegativeReals)
    m.lam_d    = pyo.Var(m.Kset, m.Rset, m.Scen, domain=pyo.NonNegativeReals)
    # mu per scenario so charging curve segment adapts to each scenario SOC
    m.mu_a     = pyo.Var(m.Kset, m.RsegS, m.Scen, domain=pyo.Binary)
    m.mu_d     = pyo.Var(m.Kset, m.RsegS, m.Scen, domain=pyo.Binary)
    m.sigma    = pyo.Var(m.Kset, m.Scen, domain=pyo.Binary)
    m.v        = pyo.Var(m.Kset, m.Scen, domain=pyo.Binary)
    m.l1       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.l2       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.l4       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    # M2 — credited charging (replaces p and u)
    m.g        = pyo.Var(m.Kset, m.Scen, domain=pyo.NonNegativeReals)
    # M5 — shift spread
    m.h        = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    m.l5       = pyo.Var(m.I, m.Scen, domain=pyo.NonNegativeReals)
    # TW5 — out-of-window indicators: SCENARIO-INDEXED binaries (an outcome,
    # not a first-stage decision).  |C|·S extra binaries in the offline solve.
    m.delta    = pyo.Var(m.Cset, m.Scen, domain=pyo.Binary)

    # ── Objective (TW2): arrival + fixed penalty per missed window ────────────
    beta = float(data.get("beta", BETA_TW))

    def _scen_obj(s):
        return m.ta[N, s] + beta * sum(m.delta[i, s] for i in C)

    if objective == "max":
        # RO1: epigraph min–max over the (finite) scenario set — adjustable
        # robust plan by constraint duplication, no dualization needed.
        m.theta   = pyo.Var(domain=pyo.Reals)
        m.epigraph = pyo.Constraint(m.Scen, rule=lambda m, s:
            m.theta >= _scen_obj(s))
        m.obj = pyo.Objective(expr=m.theta, sense=pyo.minimize)
    else:
        m.obj = pyo.Objective(
            expr  = sum(_scen_obj(s) for s in S_list) / n_scen,
            sense = pyo.minimize,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FIRST-STAGE CONSTRAINTS (no scenario index)
    # ══════════════════════════════════════════════════════════════════════════

    m.init_phi = pyo.Constraint(expr=m.phi[0] == 0)

    # At most one activity (break or rest) per stop
    m.one_brk = pyo.Constraint(m.I, rule=lambda m, i:
        m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i] <= 1)

    # x_b30 requires a prior x_b15 (split-break logic)
    m.split_ord = pyo.Constraint(m.I, rule=lambda m, i: m.x_b30[i] <= m.phi[i])

    # Reduced-rest budget (M9: rho_bar = 3 between weekly rests)
    m.rst_lim = pyo.Constraint(
        expr=sum(m.rho2[i] for i in I_list) <= rho_bar)

    # M6 — extension flag propagation and weekly budget (first-stage)
    m.init_z = pyo.Constraint(expr=m.z[0] == 0)

    def _z_persist(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.z[i+1] >= m.z[i] - m.rho1[i] - m.rho2[i]
    m.z_persist = pyo.Constraint(m.I, rule=_z_persist)
    m.q_ext_lb  = pyo.Constraint(m.I, rule=lambda m, i:
        m.q_ext[i] >= m.z[i] + m.rho1[i] + m.rho2[i] - 1)
    m.ext_budget = pyo.Constraint(
        expr=sum(m.q_ext[i] for i in I_list) + m.z[N] <= ext_bar)

    # phi propagation — split-break credit tracker
    def _phi1(m, i):
        if i >= N: return pyo.Constraint.Skip
        return (m.phi[i+1] >= m.phi[i] + m.x_b15[i]
                - m.x_b30[i] - m.x_b45[i] - m.rho1[i] - m.rho2[i])
    def _phi2(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.phi[i+1] <= m.phi[i] + m.x_b15[i]
    def _phi3(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.phi[i+1] <= 1 - m.x_b30[i] - m.x_b45[i] - m.rho1[i] - m.rho2[i]
    m.phi1 = pyo.Constraint(m.I, rule=_phi1)
    m.phi2 = pyo.Constraint(m.I, rule=_phi2)
    m.phi3 = pyo.Constraint(m.I, rule=_phi3)

    # ══════════════════════════════════════════════════════════════════════════
    # SECOND-STAGE CONSTRAINTS (per scenario s)
    # ══════════════════════════════════════════════════════════════════════════

    T_START = data.get("T_START", 0.0)
    E0      = data["E0"]

    def _xsum_first(m, i):
        """All first-stage break/rest binaries at stop i."""
        return m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]

    def _xbrk_first(m, i):
        """First-stage break-only binaries at stop i (excludes rests)."""
        return m.x_b45[i] + m.x_b15[i] + m.x_b30[i]

    # ── Initial conditions per scenario ───────────────────────────────────────
    m.init_ta = pyo.Constraint(m.Scen, rule=lambda m, s: m.ta[0, s]  == T_START)
    m.init_ea = pyo.Constraint(m.Scen, rule=lambda m, s: m.ea[0, s]  == E0)
    m.init_cd = pyo.Constraint(m.Scen, rule=lambda m, s: m.cd[0, s]  == 0.0)
    m.init_sd = pyo.Constraint(m.Scen, rule=lambda m, s: m.sd[0, s]  == 0.0)
    m.init_sw = pyo.Constraint(m.Scen, rule=lambda m, s: m.sw[0, s]  == 0.0)
    m.init_h  = pyo.Constraint(m.Scen, rule=lambda m, s: m.h[0, s]   == 0.0)

    # Origin: no activity, no charging
    m.td_orig = pyo.Constraint(m.Scen, rule=lambda m, s: m.td[0, s] == m.ta[0, s])
    m.ed_orig = pyo.Constraint(m.Scen, rule=lambda m, s: m.ed[0, s] == m.ea[0, s])
    # Destination: no activity
    m.td_dest = pyo.Constraint(m.Scen, rule=lambda m, s: m.td[N, s] == m.ta[N, s])
    m.ed_dest = pyo.Constraint(m.Scen, rule=lambda m, s: m.ed[N, s] == m.ea[N, s])
    # No break/rest at origin or destination (durations forced to 0)
    m.taub0 = pyo.Constraint(m.Scen, rule=lambda m, s: m.taub[0, s] == 0.0)
    m.taubN = pyo.Constraint(m.Scen, rule=lambda m, s: m.taub[N, s] == 0.0)
    m.taur0 = pyo.Constraint(m.Scen, rule=lambda m, s: m.taur[0, s] == 0.0)
    m.taurN = pyo.Constraint(m.Scen, rule=lambda m, s: m.taur[N, s] == 0.0)

    # ── Time propagation ──────────────────────────────────────────────────────
    def _tp(m, i, s):
        if i >= N: return pyo.Constraint.Skip
        return m.ta[i+1, s] == m.td[i, s] + m.D_sc[i, s]
    m.time_prop = pyo.Constraint(m.I, m.Scen, rule=_tp)

    # Departure at customer stops (depart_C, TW1): service starts at arrival;
    # any break/rest is taken AFTER service — no waiting term
    C_S = [(i, s) for i in C for s in S_list]
    m.td_C = pyo.Constraint(C_S, rule=lambda m, i, s:
        m.td[i, s] == (m.ta[i, s] + m.S[i]
                       + m.taub[i, s] + m.taur[i, s]))

    # Departure at CS stops: overhead + queue + charging + break + rest
    m.td_K = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.td[i, s] == (m.ta[i, s]
                       + m.v[i, s] * m.Mstop[i]
                       + m.Q_nom[i] * m.y[i]
                       + m.tauc[i, s] + m.taub[i, s] + m.taur[i, s]
                       + m.sigma[i, s] * m.Mseq[i]))

    # Departure at layby stops (M8): parking overhead + break/rest only
    if L_set:
        L_S = [(i, s) for i in sorted(L_set) for s in S_list]
        m.td_L = pyo.Constraint(L_S, rule=lambda m, i, s:
            m.td[i, s] == (m.ta[i, s]
                           + M_lay.get(i, 0.0) * _xsum_first(m, i)
                           + m.taub[i, s] + m.taur[i, s]))

    # TW2/TW5 — fixed binary window indicator per scenario, eqs. (5)/(6).
    # The single horizon big-M H (C1) upper-bounds any feasible arrival span,
    # relieving both window sides when delta = 1.  There is no arrival
    # deadline.
    if hard_tw:
        m.tw_early = pyo.Constraint(C_S, rule=lambda m, i, s:
            m.ta[i, s] >= m.Wha[i])
        m.tw_close = pyo.Constraint(C_S, rule=lambda m, i, s:
            m.ta[i, s] <= m.Whf[i])
        m.delta_zero = pyo.Constraint(C_S, rule=lambda m, i, s:
            m.delta[i, s] == 0)
    else:
        m.tw_early = pyo.Constraint(C_S, rule=lambda m, i, s:
            m.ta[i, s] >= m.Wha[i] - m.H * m.delta[i, s])
        m.tw_late = pyo.Constraint(C_S, rule=lambda m, i, s:
            m.ta[i, s] <= m.Whf[i] + m.H * m.delta[i, s])

    # ── SOC propagation ───────────────────────────────────────────────────────
    def _soc_prop(m, i, s):
        if i >= N: return pyo.Constraint.Skip
        return m.ea[i+1, s] == m.ed[i, s] - m.E_sc[i, s]
    m.soc_prop   = pyo.Constraint(m.I, m.Scen, rule=_soc_prop)
    m.soc_nc_C   = pyo.Constraint(C_S, rule=lambda m, i, s: m.ed[i, s] == m.ea[i, s])
    # M8 — laybys cannot charge: departure SOC == arrival SOC (otherwise the
    # solver draws free energy at every layby and never plans a CS charge).
    if L_set:
        L_S_soc = [(i, s) for i in sorted(L_set) for s in S_list]
        m.soc_nc_L = pyo.Constraint(L_S_soc,
            rule=lambda m, i, s: m.ed[i, s] == m.ea[i, s])
    m.soc_mono_K = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s: m.ed[i, s] >= m.ea[i, s])
    m.soc_lb     = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s: m.ea[i, s] >= m.Emin)
    m.soc_ub     = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s: m.ed[i, s] <= m.Ecap)
    m.chg_act    = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.tauc[i, s] <= m.TK * m.y[i])
    m.chg_act2   = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.tauc[i, s] >= 0.25 * m.y[i])
    m.chg_nofree = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.ed[i, s] - m.ea[i, s] <= (m.Ecap - m.Emin) * m.y[i])

    # ── PWL charging (per scenario — mu as second-stage binary) ──────────────
    R_list  = sorted(R)
    K_max   = max(Rseg)
    mid_3d  = [(i, k, s) for i in K for k in Rseg[:-1] for s in S_list]

    m.pwl_ea = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.ea[i, s] == sum(m.lam_a[i, k, s] * m.Ebar[k] for k in R))
    m.pwl_ed = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.ed[i, s] == sum(m.lam_d[i, k, s] * m.Ebar[k] for k in R))
    m.pwl_tc = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.tauc[i, s] == (sum(m.lam_d[i, k, s] * m.Tbar[k] for k in R)
                        - sum(m.lam_a[i, k, s] * m.Tbar[k] for k in R)))
    m.pwl_ca = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        sum(m.lam_a[i, k, s] for k in R) == 1)
    m.pwl_cd_c = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        sum(m.lam_d[i, k, s] for k in R) == 1)
    m.pwl_sa = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        sum(m.mu_a[i, k, s] for k in Rseg) == 1)
    m.pwl_sd = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        sum(m.mu_d[i, k, s] for k in Rseg) == 1)
    m.sos2_lo_a  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.lam_a[i, R_list[0], s] <= m.mu_a[i, R_list[1], s])
    m.sos2_hi_a  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.lam_a[i, R_list[-1], s] <= m.mu_a[i, K_max, s])
    m.sos2_lo_d  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.lam_d[i, R_list[0], s] <= m.mu_d[i, R_list[1], s])
    m.sos2_hi_d  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.lam_d[i, R_list[-1], s] <= m.mu_d[i, K_max, s])
    m.sos2_mid_a = pyo.Constraint(mid_3d, rule=lambda m, i, k, s:
        m.lam_a[i, k, s] <= m.mu_a[i, k, s] + m.mu_a[i, k+1, s])
    m.sos2_mid_d = pyo.Constraint(mid_3d, rule=lambda m, i, k, s:
        m.lam_d[i, k, s] <= m.mu_d[i, k, s] + m.mu_d[i, k+1, s])

    # ── v and sigma (second-stage binaries, bounded by first-stage decisions) ─
    _xsum = _xsum_first   # alias

    m.v_lb_y  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.v[i, s] >= m.y[i])
    m.v_lb_xr = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.v[i, s] >= _xsum(m, i))
    m.v_ub    = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.v[i, s] <= m.y[i] + _xsum(m, i))

    # M4 (43)–(44): σ only with charging; charge+rest forced sequential
    m.sigma_ub_y  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.sigma[i, s] <= m.y[i])
    m.sigma_lb_r  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.sigma[i, s] >= m.y[i] + m.rho1[i] + m.rho2[i] - 1)
    m.sigma_ub_xr = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.sigma[i, s] <= _xsum(m, i))
    # (45) C2/Q1 — uncovered declared break forces sequential mode.  The break
    # minimum b_i uses the first-stage break binaries; τ_c is the per-scenario
    # charge, so σ is scenario-indexed (a short realized charge forces σ=1).
    m.sigma_lb_brk = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.sigma[i, s] >= (m.Tb45 * m.x_b45[i] + m.Tb15 * m.x_b15[i]
                          + m.Tb30 * m.x_b30[i] - m.tauc[i, s]) / m.Tb45
                         - (1 - m.y[i]))

    # ── M2 (R7)–(R12): g = charging credited toward the break requirement ─────
    # g = tauc only when a break is DECLARED and runs in parallel (σ=0);
    # the old conc_* coverage constraints and the p machinery are deleted.
    m.g_ub1 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.g[i, s] <= m.tauc[i, s])
    m.g_ub2 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.g[i, s] <= m.TK * _xbrk_first(m, i))
    m.g_ub3 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.g[i, s] <= m.TK * (1 - m.sigma[i, s]))
    m.g_lb  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.g[i, s] >= (m.tauc[i, s] - m.TK * (1 - _xbrk_first(m, i))
                      - m.TK * m.sigma[i, s]))

    # taub_hat = taub + g at CS stops, taub_hat = taub elsewhere
    non_K_pairs = [(i, s) for i in I_list if i not in K_set for s in S_list]
    m.qb_K    = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.taub_hat[i, s] == m.taub[i, s] + m.g[i, s])
    m.qb_nonK = pyo.Constraint(non_K_pairs, rule=lambda m, i, s:
        m.taub_hat[i, s] == m.taub[i, s])

    # ── Break / rest duration constraints (first↔second stage linking) ────────
    m.brk45  = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taub_hat[i, s] >= m.Tb45 * m.x_b45[i])
    m.brk15  = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taub_hat[i, s] >= m.Tb15 * m.x_b15[i])
    m.brk30  = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taub_hat[i, s] >= m.Tb30 * m.x_b30[i])
    # Named tight big-M: a break lies within one shift spread (15 h)
    m.brk_ub = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taub[i, s] <= m.Tspr2 * (m.x_b45[i] + m.x_b15[i] + m.x_b30[i]))
    m.rst1   = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taur[i, s] >= m.Tr1 * m.rho1[i])
    m.rst2   = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taur[i, s] >= m.Tr2 * m.rho2[i])
    # Named big-M (eq. 36): a rest is bounded by the horizon H
    m.rst_ub = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taur[i, s] <= m.H * (m.rho1[i] + m.rho2[i]))

    # ── HoS accumulators (cd, sd, sw) using scenario-specific travel times ────
    # (M3: the charging work contribution is the linear expression tauc − g;
    #  the auxiliary u variable and its three constraints are deleted.)
    def _ri(m, i):
        return m.x_b45[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]
    def _rho(m, i):
        return m.rho1[i] + m.rho2[i]

    # cd: consecutive driving (reset by b45, b30, r1, r2)
    m.l1u1    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l1[i, s] <= M_drv * _ri(m, i))
    m.l1u2    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l1[i, s] <= m.cd[i, s])
    m.l1lb    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l1[i, s] >= m.cd[i, s] - M_drv * (1 - _ri(m, i)))
    def _cd(m, i, s):
        if i >= N: return pyo.Constraint.Skip
        return m.cd[i+1, s] == m.cd[i, s] + m.D_sc[i, s] - m.l1[i, s]
    m.cd_prop = pyo.Constraint(m.I, m.Scen, rule=_cd)
    m.cd_ub   = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.cd[i, s] <= m.Tdrv_cons)

    # sd: shift driving (reset only by r1 or r2)
    m.l2u1    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l2[i, s] <= M_sd * _rho(m, i))
    m.l2u2    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l2[i, s] <= m.sd[i, s])
    m.l2lb    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l2[i, s] >= m.sd[i, s] - M_sd * (1 - _rho(m, i)))
    def _sd(m, i, s):
        if i >= N: return pyo.Constraint.Skip
        return m.sd[i+1, s] == m.sd[i, s] + m.D_sc[i, s] - m.l2[i, s]
    m.sd_prop = pyo.Constraint(m.I, m.Scen, rule=_sd)
    # M6 (R16): 9 h regular / 10 h when the shift is declared extended
    m.sd_ub   = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.sd[i, s] <= m.Tdrv_sh1 + (m.Tdrv_sh2 - m.Tdrv_sh1) * m.z[i])

    # sw: shift working (driving + service/charging overhead; reset by r1/r2)
    def _cs_work(m, j, s):
        return (m.v[j, s] * m.Mstop[j] + m.Q_nom[j] * m.y[j]
                + (m.tauc[j, s] - m.g[j, s]) + m.sigma[j, s] * m.Mseq[j])

    def _work_at(m, j, s):
        if j in K_set:
            return _cs_work(m, j, s)
        if j in C_set:
            return S_svc.get(j, 0.0)
        if j in L_set:
            return M_lay.get(j, 0.0) * _xsum(m, j)
        return 0.0

    def _sw(m, i, s):
        if i >= N: return pyo.Constraint.Skip
        return (m.sw[i+1, s] == m.sw[i, s] - m.l4[i, s] + m.D_sc[i, s]
                + _work_at(m, i + 1, s))

    m.l4u1    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l4[i, s] <= M_sw * _rho(m, i))
    m.l4u2    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l4[i, s] <= m.sw[i, s])
    m.l4lb    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l4[i, s] >= m.sw[i, s] - M_sw * (1 - _rho(m, i)))
    m.sw_prop = pyo.Constraint(m.I, m.Scen, rule=_sw)
    # M5: the 13 h cap on sw is REPLACED by the shift-spread constraints below

    # ── M5 (R22)–(R25): shift spread per scenario ─────────────────────────────
    def _o(m, i, s):
        return m.td[i, s] - m.ta[i, s] - m.taur[i, s]

    def _h_prop(m, i, s):
        if i >= N: return pyo.Constraint.Skip
        return m.h[i+1, s] == m.h[i, s] + _o(m, i, s) + m.D_sc[i, s] - m.l5[i, s]
    m.h_prop = pyo.Constraint(m.I, m.Scen, rule=_h_prop)

    m.l5u1 = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l5[i, s] <= M_h * _rho(m, i))
    m.l5u2 = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l5[i, s] <= m.h[i, s] + _o(m, i, s))
    m.l5lb = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l5[i, s] >= m.h[i, s] + _o(m, i, s) - M_h * (1 - _rho(m, i)))

    Tspr1 = float(data.get("Tspr1", 13.0))
    Tspr2 = float(data.get("Tspr2", 15.0))
    m.spread_prerest = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.h[i, s] + _o(m, i, s) <= (Tspr1 + (Tspr2 - Tspr1) * m.rho2[i]
                                    + Tspr2 * (1 - _rho(m, i))))
    m.spread_ub = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.h[i, s] <= Tspr2)
    # (h_term / eq. 70): terminal spread bounded by the regular-rest cap
    m.spread_term = pyo.Constraint(m.Scen, rule=lambda m, s:
        m.h[N, s] <= Tspr1)

    # ── M9 (R21): weekly working-time cap per scenario ────────────────────────
    # OUT OF PROBLEM SCOPE (2026-07-29): the weekly working-time cap is not
    # enforced in any model (daily provisions only — see MILP.py M9 note).
    # The simulator records a breach as a diagnostic (BEHDV.weekly_notes),
    # never as a run-infeasible violation.

    # ── C3: non-anticipativity — one shared duration vector (static robust) ───
    # Tie tauc/taub/taur across scenarios to the reference scenario 0; sigma,
    # g and v then follow (their constraints become identical across
    # duplicates).  The state variables and delta stay scenario-indexed.
    if share_durations and n_scen > 1:
        s0 = S_list[0]
        rest = S_list[1:]
        m.na_tauc = pyo.Constraint(m.Kset, rest, rule=lambda m, i, s:
            m.tauc[i, s] == m.tauc[i, s0])
        m.na_taub = pyo.Constraint(m.I, rest, rule=lambda m, i, s:
            m.taub[i, s] == m.taub[i, s0])
        m.na_taur = pyo.Constraint(m.I, rest, rule=lambda m, i, s:
            m.taur[i, s] == m.taur[i, s0])

    _fix_ferry_nodes_2sp(m, data, S_list)
    return m


def _fix_ferry_nodes_2sp(m, data: dict, S_list) -> None:
    """Force the mandatory break at sea-crossing (ferry) nodes.

    The extensive-form twin of MILP._fix_ferry_nodes.  The stage split puts
    the two fixings in different places:

        x_b45[F]     = 1          FIRST stage — the break is taken, not chosen
        taub[F, s]   = T_cross    SECOND stage — same duration in every
                                  scenario, because the crossing is a
                                  timetable fact and not an outcome of the
                                  travel-time realisation

    Without this the extensive form plans the route as if the ferry stops were
    ordinary laybys, and the crossing time BEHDV imposes at execution has to
    be patched in by the repair MILP — after the first-stage structure is
    already committed.
    """
    ferry = {int(k): float(v) for k, v in (data.get("ferry") or {}).items()}
    if not ferry:
        return
    N = data["N"]
    L_set = set(data.get("L", []))
    for f, t_cross in ferry.items():
        if f not in m.I:
            continue
        if f in (0, N):
            raise ValueError(f"ferry node {f} cannot be the origin or the "
                             f"destination (taub is fixed to 0 there)")
        if f not in L_set:
            raise ValueError(f"ferry node {f} must be a layby (in data['L'])")
        m.x_b45[f].fix(1)
        for s in S_list:
            m.taub[f, s].fix(t_cross)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SOLVER WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def solve_2sp(model: pyo.ConcreteModel,
              time_limit: int = 2 * 3600,
              mip_gap: float  = 0.005,
              tee: bool       = True,
              warmstart: bool = False,
              heuristics: float | None = None,
              mip_focus: int | None    = None,
              extra_options: dict | None = None,
              log_file: str | None = None) -> tuple[dict, str]:
    """Solve the 2SP extensive form with Gurobi.

    warmstart=True feeds the model's current variable values to Gurobi as a MIP
    start (used by the ROBU cutting-plane loop, where consecutive masters differ
    only by a few appended scenarios, so the previous plan is an excellent
    incumbent).

    Performance knobs for the hard long-route solves (all default None → Gurobi
    defaults, so behaviour is unchanged unless set):
      heuristics    fraction of runtime spent on incumbent heuristics (Gurobi
                    'Heuristics'); raising it to ~0.2 finds good feasible plans
                    sooner, which is what matters when the time limit — not
                    optimality — is binding.  The oracle already uses 0.2.
      mip_focus     Gurobi 'MIPFocus' (1 = find feasible solutions fast,
                    2 = prove optimality, 3 = improve the bound).
      extra_options any further Gurobi option → value overrides.
    """
    solver = pyo.SolverFactory("gurobi")
    solver.options["MIPGap"]    = mip_gap
    solver.options["TimeLimit"] = time_limit
    if heuristics is not None:
        solver.options["Heuristics"] = heuristics
    if mip_focus is not None:
        solver.options["MIPFocus"] = mip_focus
    if log_file is not None:
        # persist Gurobi's full branch-and-bound log (incumbent / best-bound
        # node table) so the bound evolution can be plotted from real runs
        solver.options["LogFile"] = log_file
    for k, v in (extra_options or {}).items():
        solver.options[k] = v

    try:
        res    = _solve_quiet(solver, model, tee=tee, warmstart=warmstart)
        status = str(res.solver.termination_condition)
    except RuntimeError:
        return dict(feasible=False, optimal=False,
                    obj=float("inf"), status="infeasible"), "infeasible"

    feasible   = status in ("optimal", "feasible", "maxTimeLimit")
    is_optimal = status == "optimal"

    if not feasible:
        return dict(feasible=False, optimal=False,
                    obj=float("inf"), status=status), status

    obj_val = pyo.value(model.obj)
    return dict(feasible=True, optimal=is_optimal,
                obj=obj_val, status=status), status


# ══════════════════════════════════════════════════════════════════════════════
# WARM-START HELPERS (shared by RO / 2SP / ROBU seeding)
# ══════════════════════════════════════════════════════════════════════════════

def extract_first_stage(model: pyo.ConcreteModel, data: dict) -> dict:
    """First-stage decisions of a solved (share_durations) model — the binaries
    plus the shared durations at scenario index 0 — as a plain dict, for reuse
    as a Gurobi MIP start on a related model with the same first stage."""
    K_set = set(data["K"])
    I     = list(data["I"])

    def _b(var, i):
        try:    return int(round(pyo.value(var[i])))
        except Exception: return 0

    def _f(var, idx):
        try:    return max(0.0, float(pyo.value(var[idx])))
        except Exception: return 0.0

    return dict(
        y     = {i: _b(model.y, i) for i in K_set},
        x_b45 = {i: _b(model.x_b45, i) for i in I},
        x_b15 = {i: _b(model.x_b15, i) for i in I},
        x_b30 = {i: _b(model.x_b30, i) for i in I},
        rho1  = {i: _b(model.rho1, i) for i in I},
        rho2  = {i: _b(model.rho2, i) for i in I},
        phi   = {i: _b(model.phi, i) for i in I},
        z     = {i: _b(model.z, i) for i in I},
        q_ext = {i: _b(model.q_ext, i) for i in I},
        tauc  = {i: _f(model.tauc, (i, 0)) for i in K_set},
        taub  = {i: _f(model.taub, (i, 0)) for i in I},
        taur  = {i: _f(model.taur, (i, 0)) for i in I},
        sigma = {i: _b(model.sigma, (i, 0)) for i in K_set},
    )


def apply_first_stage_warmstart(model: pyo.ConcreteModel, fs: dict):
    """Prime a freshly built model's first-stage variables from ``fs`` so Gurobi
    starts from that plan as a MIP start.  Best-effort: any index/attr mismatch
    is swallowed, so a warm start can never block a solve.  The added model's
    second stage is left for the solver (Gurobi accepts a partial start)."""
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
# PART 3 — COMMITTED SCHEDULE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_2sp_committed_schedule(model: pyo.ConcreteModel, data: dict) -> list[dict]:
    """
    SP1 — Extract the committed PLAN STRUCTURE from the solved 2SP model.

    Only the first-stage binary decisions (y, break type, rest type) are
    retained.  The old behaviour of executing scenario-AVERAGED durations was
    removed: averaged durations satisfy no scenario's constraints in general
    and do not constitute a well-defined policy.  Durations are re-optimised
    ONLINE at each stop by recourse.run_plan_with_recourse (duration-only LP
    with the realized state; restricted add-only repair MILP on
    infeasibility).

    Returns
    -------
    list of dicts, one per stop 0..N, with keys:
        i, y, break_type, rest_type, is_C, is_K
    """
    K_set  = set(data["K"])

    schedule = []
    for i in data["I"]:
        y   = round(pyo.value(model.y[i]))    if i in K_set else 0
        b45 = round(pyo.value(model.x_b45[i]))
        b15 = round(pyo.value(model.x_b15[i]))
        b30 = round(pyo.value(model.x_b30[i]))
        r1  = round(pyo.value(model.rho1[i]))
        r2  = round(pyo.value(model.rho2[i]))

        brk = ("b45" if b45 else "b15" if b15 else "b30" if b30 else None)
        rst = ("r1"  if r1  else "r2"  if r2  else None)

        schedule.append(dict(
            i          = i,
            y          = y,
            break_type = brk,
            rest_type  = rst,
            is_C       = i in set(data["C"]),
            is_K       = i in K_set,
        ))
    return schedule


def extract_2sp_full_schedule(model: pyo.ConcreteModel, data: dict,
                              scen: int = 0) -> list[dict]:
    """
    C3 — Extract the committed plan STRUCTURE together with its fixed activity
    DURATIONS from a solved static-robust model (share_durations=True), reading
    the durations from reference scenario ``scen`` (they are tied across all
    scenarios by non-anticipativity, so the choice is immaterial).

    Returns per-stop dicts carrying y / break_type / rest_type AND the fixed
    tauc / taub / taur / tauq / sigma, for open-loop execution with no online
    recourse.
    """
    K_set = set(data["K"])
    Q_dict = data.get("Q", {})

    schedule = []
    for i in data["I"]:
        y   = round(pyo.value(model.y[i]))    if i in K_set else 0
        b45 = round(pyo.value(model.x_b45[i]))
        b15 = round(pyo.value(model.x_b15[i]))
        b30 = round(pyo.value(model.x_b30[i]))
        r1  = round(pyo.value(model.rho1[i]))
        r2  = round(pyo.value(model.rho2[i]))
        brk = ("b45" if b45 else "b15" if b15 else "b30" if b30 else None)
        rst = ("r1"  if r1  else "r2"  if r2  else None)

        tauc  = float(pyo.value(model.tauc[i, scen])) if i in K_set else 0.0
        taub  = float(pyo.value(model.taub[i, scen]))
        taur  = float(pyo.value(model.taur[i, scen]))
        sigma = round(pyo.value(model.sigma[i, scen])) if i in K_set else 0
        tauq  = Q_dict.get(i, 0.0) * y if i in K_set else 0.0

        schedule.append(dict(
            i=i, y=y, break_type=brk, rest_type=rst,
            b45=b45, b15=b15, b30=b30, rho1=r1, rho2=r2,
            tauc=tauc, taub=taub, taur=taur, tauq=tauq, sigma=sigma,
            is_C=i in set(data["C"]), is_K=i in K_set,
        ))
    return schedule


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — ONLINE DURATION RECOURSE (SP1)
# ══════════════════════════════════════════════════════════════════════════════
# The old open-loop execution of scenario-averaged durations was removed
# (SP1): averaged durations satisfy no scenario's constraints in general and
# are not a well-defined policy.  Execution now goes through
# recourse.run_plan_with_recourse: the plan's binary STRUCTURE is committed
# offline, the continuous durations are re-optimised at every stop from the
# realized state (tiny fixed-structure MIP), and an add-only repair MILP
# handles trajectories that leave the region the structure was designed for.
# Repair frequency and plan violations are reported as robustness metrics (S2).


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_2sp(full_data: dict,
            D_real: list,
            E_real: list,
            n_scenarios: int  = 10,
            cv: float         = None,
            scenario_seed              = None,
            time_limit: int   = 2 * 3600,
            mip_gap: float    = 0.005,
            heuristics: float | None = 0.2,
            mip_focus: int | None    = None,
            warmstart_seed: bool = False,
            tee: bool         = True,
            verbose: bool     = True,
            run_id: str       = None,
            oracle_tee: bool  = True,
            supervised: bool  = False,
            prune_quantile: float | None = GUARD_QUANTILE) -> dict:
    """
    Solve the two-stage stochastic program, then execute the first-stage
    structure with online duration recourse (SP1) on D_real/E_real.

    Information structure (SP2 caveat): the two-stage model assumes all
    travel times are revealed after the first stage — an approximation of
    the true multi-stage information process; the rolling-horizon policy
    is the corresponding multi-stage heuristic.

    Scenario budget (SP3): use the same n_scenarios as the LA policy (or
    justify any difference) so the comparison is scenario-budget fair.

    Parameters
    ----------
    full_data         : instance dict (from instance_io.load_instance_json)
    D_real            : list[float] — precomputed realised travel times (h), length N
    E_real            : list[float] — precomputed realised energies (kWh), length N
    n_scenarios       : number of full-route scenarios to draw (default 10)
    cv                : CV of the travel-time multiplier used to draw scenarios
                        (None → settings.TRAVEL_TIME_CV_TARGET)
    scenario_seed     : seed for scenario generation (None = unseeded/random)
    time_limit        : solver wall-clock limit in seconds (default 2h)
    mip_gap           : MIP relative gap tolerance (default 0.5%)
    tee               : show Gurobi solver output (default True)
    verbose           : print per-stop trajectory to stdout
    run_id            : override auto-generated run identifier
    oracle_tee        : show Gurobi output in oracle solve
    supervised        : apply the S1 safety supervisor during execution
                        (default False — infeasible runs are recorded as-is)
    prune_quantile    : supervisor worst-case quantile (RH2)

    Returns
    -------
    dict — canonical results dict (same schema as run_simulation / run_ro)
    """
    t_wall_start = time.perf_counter()
    N       = full_data["N"]
    T_START = full_data.get("T_START", 8.0)
    label   = full_data.get("label", "2sp")
    title   = full_data.get("title", "inst")

    assert len(D_real) == N, f"D_real length {len(D_real)} != N={N}"
    assert len(E_real) == N, f"E_real length {len(E_real)} != N={N}"

    _paths.ensure_dirs()
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{title}_2SP_S{n_scenarios}_{ts}"
    paths = dict(
        log = _paths.logs(f"{run_id}.txt"),
        fig = _paths.figures(f"{run_id}.png"),
        sol = _paths.solutions(f"{run_id}.json"),
        scn = _paths.logs(f"{run_id}_scenarios.json"),
        gurobi = _paths.logs(f"{run_id}_gurobi.log"),
    )
    log = open(paths["log"], "w", encoding="utf-8")

    def _p(msg):
        if verbose: print(msg)
        try:    print(msg, file=log)
        except Exception: pass

    _p("=" * 65)
    _p(f"  2SP SOLVE START   ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    _p(f"  Instance   : {label}   run_id={run_id}")
    _p(f"  Route      : {N} stops  departure={T_START:.0f}:00")
    _p(f"  Settings   : S={n_scenarios}  cv={cv}  gap={mip_gap:.1%}"
       f"  time_limit={time_limit}s")
    _p("=" * 65)

    # ── Step 1: Draw full-route scenarios live ─────────────────────────────────
    scenarios = generate_scenarios(
        full_data     = full_data,
        start_stop    = 0,
        end_stop      = N,
        n_scenarios   = n_scenarios,
        cv            = cv,
        seed          = scenario_seed,
        include_best  = False,
        include_worst = False,
    )
    _p(f"\n  Drew {len(scenarios)} full-route scenarios (cv={cv})")

    # ── Step 2: Build and solve the extensive form ─────────────────────────────
    _p(f"\n  Building 2SP extensive form (S={len(scenarios)}, N={N})...")
    model = build_2sp_model(full_data, scenarios)

    # Optional warm start: solve the cheap single NOMINAL-scenario model first
    # (same first-stage structure, a fraction of the extensive form's size) and
    # feed its plan to the full model as a Gurobi MIP start.  A short cap keeps
    # the seed from eating the budget; it only ever adds a starting incumbent,
    # so it cannot worsen the result.  Off by default — worthwhile on the hard
    # long-route instances, wasteful on the ones that already solve fast.
    t_solve = time.perf_counter()
    if warmstart_seed:
        seed_cap = int(max(60, min(time_limit // 6, 600)))
        _p(f"  Warm-start seed: solving nominal 1-scenario model "
           f"(<= {seed_cap}s)...")
        nominal = [dict(D=dict(full_data["D"]), E=dict(full_data["E"]))]
        seed_model = build_2sp_model(full_data, nominal)
        s_info, s_status = solve_2sp(seed_model, time_limit=seed_cap,
                                     mip_gap=mip_gap, tee=False,
                                     heuristics=heuristics, mip_focus=1)
        if s_info["feasible"]:
            apply_first_stage_warmstart(model, extract_first_stage(seed_model,
                                                                   full_data))
            _p(f"  Warm-start seed ready ({s_status}); priming full solve.")
        else:
            _p(f"  Warm-start seed found no plan ({s_status}); "
               f"solving cold.")

    _p(f"  Solving...")
    info, status = solve_2sp(model, time_limit=time_limit, mip_gap=mip_gap,
                             tee=True, warmstart=warmstart_seed,
                             heuristics=heuristics, mip_focus=mip_focus,
                             log_file=paths["gurobi"])
    t_solve = time.perf_counter() - t_solve

    _p(f"  Status     : {status}  ({t_solve:.1f}s)")

    if not info["feasible"]:
        _p("  No feasible solution found — aborting.")
        log.close()
        return dict(feasible=False, status=status,
                    total_time=float("inf"),
                    wall_clock=time.perf_counter() - t_wall_start)

    _p(f"  2SP objective (E[arrival]) : {info['obj']:.3f} h")

    # ── Step 3: Extract committed plan STRUCTURE (first-stage binaries) ────────
    committed = extract_2sp_committed_schedule(model, full_data)
    n_chg = sum(1 for e in committed if e["y"])
    n_brk = sum(1 for e in committed if e["break_type"])
    n_rst = sum(1 for e in committed if e["rest_type"])
    _p(f"  Committed  : {n_chg} charge(s),  {n_brk} break(s),  {n_rst} rest(s)")
    _p(f"  (structure only — durations re-optimised online, SP1)")

    # ── Step 4: Execute plan with online duration recourse (SP1) ──────────────
    _p(f"\n  Executing 2SP structure with online duration recourse...")
    vehicle, tracker, events = run_plan_with_recourse(
        full_data      = full_data,
        plan           = committed,
        D_real         = D_real,
        E_real         = E_real,
        method_name    = "2SP",
        log_fn         = _p,
        cv             = cv,
        supervised     = supervised,
        prune_quantile = prune_quantile,
        verbose        = verbose,
    )
    scores_log = []
    _p(f"  Recourse   : {len(events['repairs'])} repair(s), "
       f"{len(events['plan_violations'])} plan violation(s), "
       f"{len(events['interventions'])} supervisor intervention(s)")

    wall_elapsed = time.perf_counter() - t_wall_start
    arr_h = vehicle.t_arr
    _p(f"\n{'='*65}")
    _p(f"  2SP SIMULATION COMPLETE")
    _p(f"  Arrival (absolute) : {arr_h:.3f} h  "
       f"({int(arr_h):02d}:{int((arr_h % 1) * 60):02d})")
    _p(f"  Travel duration    : {arr_h - T_START:.3f} h")
    _p(f"  Solve time (2SP)   : {t_solve:.1f} s")
    _p(f"  Wall-clock total   : {wall_elapsed:.1f} s")
    _p("=" * 65)

    # ── Step 5: Delegate epilogue to runner (oracle, JSON, figure) ────────────
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
        events      = events,
        method_meta = dict(
            method        = "2SP",
            n_scenarios   = len(scenarios),
            cv            = cv,
            twosp_obj     = info["obj"],
            twosp_status  = status,
            twosp_optimal = info["optimal"],
            twosp_warmstart_seed = bool(warmstart_seed),
            twosp_heuristics     = heuristics,
            gurobi_log    = paths["gurobi"],
            solve_time    = t_solve,
            supervised    = supervised,
            prune_quantile= prune_quantile,
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.instance_gen.instance_io import load_instance_json

    # Usage: python -m src.methods.twosp <json_file> [n_scenarios] [time_limit]
    json_file   = sys.argv[1] if len(sys.argv) > 1 else None
    n_scenarios = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    time_limit  = int(sys.argv[3]) if len(sys.argv) > 3 else 2 * 3600

    if json_file is None:
        print("Usage: python -m src.methods.twosp <json_file> [n_scenarios] [time_limit_s]")
        sys.exit(1)

    full_data, D_real, E_real, cv_file = load_instance_json(json_file)

    results = run_2sp(
        full_data         = full_data,
        D_real            = D_real,
        E_real            = E_real,
        n_scenarios       = n_scenarios,
        cv                = cv_file,
        time_limit        = time_limit,
        tee               = True,
        verbose           = True,
        oracle_tee        = True,
    )

    print(f"\n  2SP arrival  : {results['total_time']:.3f} h")
    print(f"  Wall clock   : {results['wall_clock']:.1f} s")
    print(f"  Figure       : {results['fig_path']}")
    print(f"  Solution     : {results['sol_path']}")
