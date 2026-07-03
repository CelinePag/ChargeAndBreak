"""
2SP.py — Two-stage stochastic programming for BET scheduling
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
    min (1/S) Σ_s ta[N, s]    (expected arrival time at destination)

Linking constraints:
    The second-stage break/rest durations are bounded below by the minimum
    required for the chosen first-stage type:
        taub[i,s] >= Tb45 · x_b45[i]   (etc.)
    HoS accumulator propagation uses scenario travel times D[i,s].
    Charging amount (tauc[i,s]) adapts per scenario via the PWL recourse.

Simulation step (open-loop)
---------------------------
After solving, a committed schedule is extracted:
  - First-stage binaries (y, break/rest types) are taken as-is.
  - Continuous durations (tauc, taub, taur) are averaged across all S scenarios.
  - sigma is determined by majority vote across scenarios.
This committed schedule is executed open-loop via BEHDV.advance() with the
actual realised travel times.  No recourse or re-optimisation occurs; the plan
does not use D_real at any point.  The only adaptation is clipping tauc to the
physically feasible charge time given the actual SOC at each CS stop (cannot
overcharge — this uses observable vehicle state, not future D_real).

Scenarios
---------
Scenarios are generated live at solve time via scenarios.generate_scenarios()
(start_stop=0, end_stop=N, delta=delta) — the same mechanism LA uses — rather
than read from a precomputed pool.  Runs are therefore not tied to a fixed
scenario sample across repeated calls; pass `scenario_seed` for reproducibility.

Integration with the framework
-------------------------------
  import importlib
  twosp = importlib.import_module("2SP")
  results = twosp.run_2sp(full_data, D_real, E_real, n_scenarios=10)

  Or via runner_dispatch.py:
    python runner_dispatch.py instances/RmediumCfew_7.json 2SP

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

from BEHDV     import BEHDV, _charging_time_needed
from MILP      import (
    _declare_common_params,
    _solve_quiet,
)
from scenarios import ScenarioTracker, generate_scenarios
from runner    import finalize_run


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — EXTENSIVE FORM MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_2sp_model(data: dict, scenarios: list[dict]) -> pyo.ConcreteModel:
    """
    Build the extensive-form 2SP Pyomo model.

    Parameters
    ----------
    data      : full route data dict from instances.make_data()
    scenarios : list of S scenario dicts, each with keys "D" and "E"
                (global leg index → float), from generate_scenarios().

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
    I_list  = list(data["I"])
    S_svc   = data["S"]          # service times at customer stops (Python dict)
    M_big   = data["M_big"]
    M_drv   = data["M_drv"]
    M_sd    = data["M_sd"]
    M_sw    = data["M_sw"]

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

    # Fix no activities at origin and destination
    for _v in [m.x_b45, m.x_b15, m.x_b30, m.rho1, m.rho2]:
        _v[0].fix(0)
        _v[N].fix(0)

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
    m.u        = pyo.Var(m.Kset, m.Scen, domain=pyo.NonNegativeReals)
    m.p        = pyo.Var(m.Kset, m.Scen, domain=pyo.NonNegativeReals)

    # ── Objective: expected arrival time ──────────────────────────────────────
    m.obj = pyo.Objective(
        expr  = sum(m.ta[N, s] for s in S_list) / n_scen,
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

    # Reduced-rest budget: at most 3 per route
    m.rst_lim = pyo.Constraint(
        expr=sum(m.rho2[i] for i in I_list) <= 3)

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

    # ── Initial conditions per scenario ───────────────────────────────────────
    m.init_ta = pyo.Constraint(m.Scen, rule=lambda m, s: m.ta[0, s]  == T_START)
    m.init_ea = pyo.Constraint(m.Scen, rule=lambda m, s: m.ea[0, s]  == E0)
    m.init_cd = pyo.Constraint(m.Scen, rule=lambda m, s: m.cd[0, s]  == 0.0)
    m.init_sd = pyo.Constraint(m.Scen, rule=lambda m, s: m.sd[0, s]  == 0.0)
    m.init_sw = pyo.Constraint(m.Scen, rule=lambda m, s: m.sw[0, s]  == 0.0)

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

    # Departure at customer stops: service + break + rest
    C_S = [(i, s) for i in C for s in S_list]
    m.td_C = pyo.Constraint(C_S, rule=lambda m, i, s:
        m.td[i, s] == m.ta[i, s] + m.S[i] + m.taub[i, s] + m.taur[i, s])

    # Departure at CS stops: overhead + queue + charging + break + rest
    m.td_K = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.td[i, s] == (m.ta[i, s]
                       + m.v[i, s] * m.Mstop[i]
                       + m.Q_nom[i] * m.y[i]
                       + m.tauc[i, s] + m.taub[i, s] + m.taur[i, s]
                       + m.sigma[i, s] * m.Mseq[i]))

    # Time windows at customer stops
    m.tw_hard = pyo.Constraint(C_S, rule=lambda m, i, s:
        pyo.inequality(m.Wha[i], m.ta[i, s], m.Whf[i]))

    # ── SOC propagation ───────────────────────────────────────────────────────
    def _soc_prop(m, i, s):
        if i >= N: return pyo.Constraint.Skip
        return m.ea[i+1, s] == m.ed[i, s] - m.E_sc[i, s]
    m.soc_prop   = pyo.Constraint(m.I, m.Scen, rule=_soc_prop)
    m.soc_nc_C   = pyo.Constraint(C_S, rule=lambda m, i, s: m.ed[i, s] == m.ea[i, s])
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
    def _xsum(m, i):
        return m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]

    m.v_lb_y  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.v[i, s] >= m.y[i])
    m.v_lb_xr = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.v[i, s] >= _xsum(m, i))
    m.v_ub    = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.v[i, s] <= m.y[i] + _xsum(m, i))
    m.sigma_ub_y  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.sigma[i, s] <= m.y[i])
    m.sigma_ub_xr = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.sigma[i, s] <= _xsum(m, i))

    # Concurrent charging: if σ=0 and y=1, tauc must cover the declared break/rest
    m.conc_b45 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.tauc[i, s] >= m.Tb45 * m.x_b45[i] - M_big * m.sigma[i, s] - M_big * (1 - m.y[i]))
    m.conc_b15 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.tauc[i, s] >= m.Tb15 * m.x_b15[i] - M_big * m.sigma[i, s] - M_big * (1 - m.y[i]))
    m.conc_b30 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.tauc[i, s] >= m.Tb30 * m.x_b30[i] - M_big * m.sigma[i, s] - M_big * (1 - m.y[i]))
    m.conc_r1  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.tauc[i, s] >= m.Tr1 * m.rho1[i] - M_big * m.sigma[i, s] - M_big * (1 - m.y[i]))
    m.conc_r2  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.tauc[i, s] >= m.Tr2 * m.rho2[i] - M_big * m.sigma[i, s] - M_big * (1 - m.y[i]))

    # ── p = tauc·(1−σ) at CS (concurrent charging credit toward break time) ───
    m.p_ub1 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.p[i, s] <= m.tauc[i, s])
    m.p_ub2 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.p[i, s] <= m.TK * (1 - m.sigma[i, s]))
    m.p_lb  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.p[i, s] >= m.tauc[i, s] - m.TK * m.sigma[i, s])

    # taub_hat = taub + p at CS stops, taub_hat = taub elsewhere
    non_K_pairs = [(i, s) for i in I_list if i not in K_set for s in S_list]
    m.qb_K    = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.taub_hat[i, s] == m.taub[i, s] + m.p[i, s])
    m.qb_nonK = pyo.Constraint(non_K_pairs, rule=lambda m, i, s:
        m.taub_hat[i, s] == m.taub[i, s])

    # ── Break / rest duration constraints (first↔second stage linking) ────────
    m.brk45  = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taub_hat[i, s] >= m.Tb45 * m.x_b45[i])
    m.brk15  = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taub_hat[i, s] >= m.Tb15 * m.x_b15[i])
    m.brk30  = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taub_hat[i, s] >= m.Tb30 * m.x_b30[i])
    m.brk_ub = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taub[i, s] <= M_big * (m.x_b45[i] + m.x_b15[i] + m.x_b30[i]))
    m.rst1   = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taur[i, s] >= m.Tr1 * m.rho1[i])
    m.rst2   = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taur[i, s] >= m.Tr2 * m.rho2[i])
    m.rst_ub = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.taur[i, s] <= M_big * (m.rho1[i] + m.rho2[i]))

    # ── u: charging work in shift-working accumulator (concurrent vs sequential)
    m.u_ub1 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.u[i, s] <= m.TK * (1 - _xsum(m, i) + m.sigma[i, s]))
    m.u_ub2 = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.u[i, s] <= m.tauc[i, s])
    m.u_lb  = pyo.Constraint(m.Kset, m.Scen, rule=lambda m, i, s:
        m.u[i, s] >= m.tauc[i, s] - m.TK * (_xsum(m, i) - m.sigma[i, s]))

    # ── HoS accumulators (cd, sd, sw) using scenario-specific travel times ────
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
    m.sd_ub   = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.sd[i, s] <= m.Tdrv_sh1)

    # sw: shift working (driving + service/charging overhead; reset by r1/r2)
    def _cs_work(m, j, s):
        return (m.v[j, s] * m.Mstop[j] + m.Q_nom[j] * m.y[j]
                + m.u[j, s] + m.sigma[j, s] * m.Mseq[j])

    def _sw(m, i, s):
        if i >= N: return pyo.Constraint.Skip
        ip1 = i + 1
        work_next = (_cs_work(m, ip1, s) if ip1 in K_set
                     else (S_svc.get(ip1, 0.0) if ip1 in C_set else 0.0))
        return m.sw[i+1, s] == m.sw[i, s] - m.l4[i, s] + m.D_sc[i, s] + work_next

    m.l4u1    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l4[i, s] <= M_sw * _rho(m, i))
    m.l4u2    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l4[i, s] <= m.sw[i, s])
    m.l4lb    = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.l4[i, s] >= m.sw[i, s] - M_sw * (1 - _rho(m, i)))
    m.sw_prop = pyo.Constraint(m.I, m.Scen, rule=_sw)
    m.sw_ub   = pyo.Constraint(m.I, m.Scen, rule=lambda m, i, s:
        m.sw[i, s] <= m.Twrk_sh)

    return m


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SOLVER WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def solve_2sp(model: pyo.ConcreteModel,
              time_limit: int = 2 * 3600,
              mip_gap: float  = 0.005,
              tee: bool       = True) -> tuple[dict, str]:
    """Solve the 2SP extensive form with Gurobi."""
    solver = pyo.SolverFactory("gurobi")
    solver.options["MIPGap"]    = mip_gap
    solver.options["TimeLimit"] = time_limit

    try:
        res    = _solve_quiet(solver, model, tee=tee)
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
# PART 3 — COMMITTED SCHEDULE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_2sp_committed_schedule(model: pyo.ConcreteModel, data: dict) -> list[dict]:
    """
    Extract a committed open-loop schedule from the solved 2SP model.

    First-stage binaries (y, break type, rest type) are taken as-is.
    Continuous durations (tauc, taub, taur) are averaged across all S scenarios
    to produce a single deterministic plan that is executed regardless of the
    actual realization.

    sigma (concurrent vs. sequential charging mode) is set to 1 only when the
    majority of scenarios chose sequential; otherwise 0 (concurrent).

    Returns
    -------
    list of dicts, one per stop 0..N, with keys:
        i, y, break_type, rest_type, tauc, taub, taur, tauq, sigma, is_C, is_K
    """
    N      = data["N"]
    K_set  = set(data["K"])
    S_list = list(model.Scen)
    n_scen = len(S_list)

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

        if i in K_set:
            tauc_avg  = sum(pyo.value(model.tauc[i, s])         for s in S_list) / n_scen
            sigma_sum = sum(round(pyo.value(model.sigma[i, s])) for s in S_list)
            sigma_val = 1 if sigma_sum >= n_scen / 2 else 0
        else:
            tauc_avg  = 0.0
            sigma_val = 0

        taub_avg = sum(pyo.value(model.taub[i, s]) for s in S_list) / n_scen
        taur_avg = sum(pyo.value(model.taur[i, s]) for s in S_list) / n_scen
        tauq     = data["Q"].get(i, 0.0) * y if i in K_set else 0.0

        schedule.append(dict(
            i          = i,
            y          = y,
            break_type = brk,
            rest_type  = rst,
            tauc       = tauc_avg,
            taub       = taub_avg,
            taur       = taur_avg,
            tauq       = tauq,
            sigma      = sigma_val,
            is_C       = i in set(data["C"]),
            is_K       = i in K_set,
        ))
    return schedule


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — OPEN-LOOP SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_2sp_schedule(full_data: dict,
                            committed_schedule: list[dict],
                            D_real: list,
                            E_real: list,
                            verbose: bool,
                            log_fn) -> tuple[BEHDV, ScenarioTracker, list]:
    """
    Execute the committed 2SP schedule on the precomputed realisation.

    The schedule (binary decisions + scenario-averaged continuous durations)
    is applied open-loop: no recourse or re-optimisation occurs and the plan
    does not depend on D_real in any way.

    The only adaptation is clipping tauc to physically feasible given the
    actual SOC at each CS stop (cannot overcharge — physics, not D_real info).
    """
    N       = full_data["N"]
    vehicle = BEHDV(full_data)
    tracker = ScenarioTracker(full_data)
    K_set   = set(full_data["K"])
    C_set   = set(full_data["C"])

    for stop in range(N):
        entry = committed_schedule[stop]
        brk   = entry["break_type"]
        rst   = entry["rest_type"]
        y     = entry["y"]

        # Clip tauc to physically feasible given actual SOC at arrival.
        # Cannot overcharge the battery — this is a physical constraint,
        # NOT information about future travel times.
        tauc_plan = entry["tauc"]
        if stop in K_set and y and tauc_plan > 0:
            tauc_exec = min(tauc_plan, _charging_time_needed(vehicle.e_arr, full_data))
        else:
            tauc_exec = 0.0

        stop_type = ("CS"   if stop in K_set else
                     "CUST" if stop in C_set else
                     "ORIG" if stop == 0 else "INT")
        log_fn(f"\n  stop {stop:>3} ({stop_type})"
               f"  t={vehicle.t_arr:.3f}h  soc={vehicle.e_arr:.0f}kWh"
               f"  cd={vehicle.cd:.2f}  sd={vehicle.sd:.2f}  sw={vehicle.sw:.2f}")
        log_fn(f"     -> y={y}  brk={brk or '---'}  rst={rst or '---'}"
               f"  tauc={tauc_exec*60:.0f}m (plan={tauc_plan*60:.0f}m)"
               f"  taub={entry['taub']*60:.0f}m"
               f"  taur={entry['taur']*60:.0f}m"
               f"  D_act={float(D_real[stop]):.3f}h  E_act={float(E_real[stop]):.1f}kWh")

        mock_sol = dict(
            feasible = True,
            sol      = [dict(
                i     = 0,
                taub  = entry["taub"],
                tauc  = tauc_exec,
                taur  = entry["taur"],
                tauq  = entry["tauq"],
                y     = y,
                b45   = int(brk == "b45"),
                b15   = int(brk == "b15"),
                b30   = int(brk == "b30"),
                rho1  = int(rst == "r1"),
                rho2  = int(rst == "r2"),
                sigma = entry.get("sigma", 0),
                is_C  = stop in C_set,
                is_K  = stop in K_set,
            )],
        )

        action = dict(y=y, break_type=brk, rest_type=rst)
        vehicle.advance(action=action, D_next=float(D_real[stop]),
                        E_next=float(E_real[stop]), milp_sol=mock_sol)
        tracker.record_realisation(stop, float(D_real[stop]),
                                   E_actual=float(E_real[stop]))

    return vehicle, tracker, []


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_2sp(full_data: dict,
            D_real: list,
            E_real: list,
            n_scenarios: int  = 10,
            delta: float      = 0.20,
            scenario_seed              = None,
            time_limit: int   = 2 * 3600,
            mip_gap: float    = 0.005,
            tee: bool         = True,
            verbose: bool     = True,
            run_id: str       = None,
            oracle_tee: bool  = True) -> dict:
    """
    Solve the two-stage stochastic program and simulate on D_real/E_real.

    Parameters
    ----------
    full_data         : instance dict (from instance_io.load_instance_json)
    D_real            : list[float] — precomputed realised travel times (h), length N
    E_real            : list[float] — precomputed realised energies (kWh), length N
    n_scenarios       : number of full-route scenarios to draw (default 10)
    delta             : travel-time uncertainty half-width used to draw scenarios
    scenario_seed     : seed for scenario generation (None = unseeded/random)
    time_limit        : solver wall-clock limit in seconds (default 2h)
    mip_gap           : MIP relative gap tolerance (default 0.5%)
    tee               : show Gurobi solver output (default True)
    verbose           : print per-stop trajectory to stdout
    run_id            : override auto-generated run identifier
    oracle_tee        : show Gurobi output in oracle solve

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

    for d in ("logs", "figures", "solutions"):
        os.makedirs(d, exist_ok=True)
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{title}_2SP_S{n_scenarios}_{ts}"
    paths = dict(
        log = os.path.join("logs",      f"{run_id}.txt"),
        fig = os.path.join("figures",   f"{run_id}.png"),
        sol = os.path.join("solutions", f"{run_id}.json"),
        scn = os.path.join("logs",      f"{run_id}_scenarios.json"),
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
    _p(f"  Settings   : S={n_scenarios}  δ={delta:.0%}  gap={mip_gap:.1%}"
       f"  time_limit={time_limit}s")
    _p("=" * 65)

    # ── Step 1: Draw full-route scenarios live ─────────────────────────────────
    scenarios = generate_scenarios(
        full_data     = full_data,
        start_stop    = 0,
        end_stop      = N,
        n_scenarios   = n_scenarios,
        delta         = delta,
        seed          = scenario_seed,
        include_best  = False,
        include_worst = False,
    )
    _p(f"\n  Drew {len(scenarios)} full-route scenarios (δ={delta:.0%})")

    # ── Step 2: Build and solve the extensive form ─────────────────────────────
    _p(f"\n  Building 2SP extensive form (S={len(scenarios)}, N={N})...")
    model = build_2sp_model(full_data, scenarios)

    _p(f"  Solving...")
    t_solve = time.perf_counter()
    info, status = solve_2sp(model, time_limit=time_limit, mip_gap=mip_gap, tee=True)
    t_solve = time.perf_counter() - t_solve

    _p(f"  Status     : {status}  ({t_solve:.1f}s)")

    if not info["feasible"]:
        _p("  No feasible solution found — aborting.")
        log.close()
        return dict(feasible=False, status=status,
                    total_time=float("inf"),
                    wall_clock=time.perf_counter() - t_wall_start)

    _p(f"  2SP objective (E[arrival]) : {info['obj']:.3f} h")

    # ── Step 3: Extract committed schedule (binary + scenario-averaged durations)
    committed = extract_2sp_committed_schedule(model, full_data)
    n_chg = sum(1 for e in committed if e["y"])
    n_brk = sum(1 for e in committed if e["break_type"])
    n_rst = sum(1 for e in committed if e["rest_type"])
    _p(f"  Committed  : {n_chg} charge(s),  {n_brk} break(s),  {n_rst} rest(s)")
    _p(f"  (durations are scenario averages over S={len(scenarios)} scenarios)")

    # ── Step 4: Execute committed plan open-loop on actual realisation ─────────
    _p(f"\n  Executing committed 2SP schedule (open-loop, no recourse)...")
    vehicle, tracker, scores_log = _simulate_2sp_schedule(
        full_data, committed, D_real, E_real,
        verbose=verbose, log_fn=_p,
    )

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
        method_meta = dict(
            method        = "2SP",
            n_scenarios   = len(scenarios),
            delta         = delta,
            twosp_obj     = info["obj"],
            twosp_status  = status,
            twosp_optimal = info["optimal"],
            solve_time    = t_solve,
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from instance_io import load_instance_json

    # Usage: python 2SP.py <json_file> [n_scenarios] [time_limit]
    json_file   = sys.argv[1] if len(sys.argv) > 1 else None
    n_scenarios = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    time_limit  = int(sys.argv[3]) if len(sys.argv) > 3 else 2 * 3600

    if json_file is None:
        print("Usage: python 2SP.py <json_file> [n_scenarios] [time_limit_s]")
        sys.exit(1)

    full_data, D_real, E_real, delta_file = load_instance_json(json_file)

    results = run_2sp(
        full_data         = full_data,
        D_real            = D_real,
        E_real            = E_real,
        n_scenarios       = n_scenarios,
        delta             = delta_file,
        time_limit        = time_limit,
        tee               = True,
        verbose           = True,
        oracle_tee        = True,
    )

    print(f"\n  2SP arrival  : {results['total_time']:.3f} h")
    print(f"  Wall clock   : {results['wall_clock']:.1f} s")
    print(f"  Figure       : {results['fig_path']}")
    print(f"  Solution     : {results['sol_path']}")
