"""
MILP.py — Electric Truck Scheduling: Full-Route & Rolling-Horizon Models
=========================================================================
Single file for ALL Pyomo modelling code.  Organised into four parts:

  PART 1 — Shared helpers
      _declare_common_vars  : declare the full set of decision variables
      _add_pwl_*            : PWL charging constraints (Montoya et al. 2017)
      _add_break_rest_*     : HoS break / rest binary constraints
      _add_hos_accumulator_*: cd / sd / sw propagation with big-M resets
      _add_manoeuver_*      : z_man activity-indicator constraints
      _model_xsum / _ri / _rho : Pyomo expression shorthands
      _solve_quiet          : solver wrapper that suppresses output when tee=False

  PART 2 — Full-route deterministic model
      build_model(data)       : build Pyomo model for complete route 0..N
      solve_model(model)      : solve with HiGHS (0.5% gap, 2h limit)
      extract_solution(model) : extract per-stop solution dicts
      run_instance(data)      : convenience wrapper: build→solve→report→save
      Entry point: python MILP.py [instance_name]

  PART 3 — Rolling-horizon sub-problem
      make_subproblem_data(...)  : slice full data dict to sub-route window
      build_horizon_model(...)   : build Pyomo model for [start_stop, end_stop]
      extract_horizon_solution() : extract per-stop solution dicts (local idx)
      solve_horizon(...)         : end-to-end: build → warm-start → solve → extract
      Called by Simulation.py (look-ahead scorer) and oracle.py (hindsight solve).

  PART 4 — IO helpers
      save_solution / load_solution / solution_path
      check_solution / print_schedule

Mathematical notation
---------------------
  Stops 0..N : origin=0, intermediate={1..N-1}, destination=N
  C ⊆ {1..N-1} : customer stops (mandatory service time S_i)
  K ⊆ {1..N-1} : charging station (CS) stops   (C ∩ K = ∅)
  All times in HOURS, energy in kWh.

Data dict
---------
  Produced by instances.make_data().  Full key listing in instances.py docstring.

Dependencies
------------
  MILP.py imports compute_time_bounds from instances.py (arrival-time bounds
  needed when building sub-problems at run-time with perturbed travel times).
  No other local imports at module level.

References
----------
  Montoya et al. (2017) "The electric vehicle routing problem with nonlinear
  charging function", Transportation Research Part B, 103, pp. 87-110.
  https://doi.org/10.1016/j.trb.2017.02.004
"""

from __future__ import annotations

import contextlib as _ctx
import io as _io
import json
import os
import random
import math as _mi

import pyomo.environ as pyo

from instances import compute_time_bounds

FIGURES_DIR        = "figures"
SOLUTIONS_DIR      = "solutions"
EPS                = 1e-5
INFEASIBLE_PENALTY = 1e9


def _ensure_dirs():
    os.makedirs(FIGURES_DIR,   exist_ok=True)
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Note: compute_time_bounds lives in instances.py (imported above).
# It is the single authoritative implementation; both build_model and
# make_subproblem_data call it via the imported name.

# ── Expression helpers ────────────────────────────────────────────────────────

def _model_xsum(m, i):
    """All binary break/rest decisions at stop i."""
    return m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]

def _model_ri(m, i):
    """Consecutive-driving reset indicator (b45, b30, or any rest)."""
    return m.x_b45[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]

def _model_rho(m, i):
    """Shift reset indicator (any rest: r1 or r2)."""
    return m.rho1[i] + m.rho2[i]


# ── Variable declaration ──────────────────────────────────────────────────────

def _declare_common_vars(m):
    """
    Declare all shared decision variables on model m.
    Precondition: m.I, m.Cset, m.Kset, m.Rset, m.RsegS must already exist.
    """
    m.ta  = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.td  = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.ea  = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.ed  = pyo.Var(m.I, domain=pyo.NonNegativeReals)

    m.y      = pyo.Var(m.Kset, domain=pyo.Binary)
    m.tauc   = pyo.Var(m.Kset, domain=pyo.NonNegativeReals)
    m.lam_a  = pyo.Var(m.Kset, m.Rset,  domain=pyo.NonNegativeReals)
    m.lam_d  = pyo.Var(m.Kset, m.Rset,  domain=pyo.NonNegativeReals)
    m.mu_a   = pyo.Var(m.Kset, m.RsegS, domain=pyo.Binary)
    m.mu_d   = pyo.Var(m.Kset, m.RsegS, domain=pyo.Binary)

    m.x_b45    = pyo.Var(m.I, domain=pyo.Binary)
    m.x_b15    = pyo.Var(m.I, domain=pyo.Binary)
    m.x_b30    = pyo.Var(m.I, domain=pyo.Binary)
    m.phi      = pyo.Var(m.I, domain=pyo.Binary)
    m.taub     = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.taub_hat = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.rho1     = pyo.Var(m.I, domain=pyo.Binary)
    m.rho2     = pyo.Var(m.I, domain=pyo.Binary)
    m.taur     = pyo.Var(m.I, domain=pyo.NonNegativeReals)

    m.cd = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.sd = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.sw = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.l1 = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.l2 = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.l4 = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.u  = pyo.Var(m.Kset, domain=pyo.NonNegativeReals)

    m.z_man = pyo.Var(m.I, domain=pyo.Binary)


# ── Parameters declaration ──────────────────────────────────────────────────────

def _declare_common_params(m, data):

    N     = data["N"]
    C     = data["C"]
    K     = data["K"]
    R     = data["R"]
    Rseg  = data["Rseg"]
    lb_t  = data["lb_t"]
    ub_t  = data["ub_t"]

    m.TK    = pyo.Param(initialize=data["Tbar"][max(R)])
    m.M_drv = pyo.Param(initialize=data["M_drv"])
    m.M_sd  = pyo.Param(initialize=data["M_sd"])
    m.M_sw  = pyo.Param(initialize=data["M_sw"])
    m.M_big = pyo.Param(initialize=data["M_big"])


    # ── Sets ──────────────────────────────────────────────────────────────────
    m.I     = pyo.Set(initialize=data["I"], ordered=True)
    m.Cset  = pyo.Set(initialize=C)
    m.Kset  = pyo.Set(initialize=K)
    m.Rset  = pyo.Set(initialize=R,    ordered=True)
    m.RsegS = pyo.Set(initialize=Rseg, ordered=True)
    m.Legs  = pyo.Set(initialize=list(range(N)), ordered=True)

    # ── Parameters ────────────────────────────────────────────────────────────
    m.D_nom = pyo.Param(m.Legs, initialize=data["D"])
    m.Q_nom = pyo.Param(m.Kset, initialize=data["Q"],  default=0)
    m.Man   = pyo.Param(m.I,    initialize={**data["M"], N: 0}, default=0)
    m.S     = pyo.Param(m.Cset, initialize=data["S"],  default=0)
    m.Eparam= pyo.Param(m.Legs, initialize=data["E"])
    m.E0    = pyo.Param(initialize=data["E0"])
    m.Ecap  = pyo.Param(initialize=data["Ecap"])
    m.Emin  = pyo.Param(initialize=data["Emin"])
    m.Ebar  = pyo.Param(m.Rset, initialize=data["Ebar"])
    m.Tbar  = pyo.Param(m.Rset, initialize=data["Tbar"])
    m.Wha   = pyo.Param(m.Cset, initialize=data["Wha"], default=0)
    m.Whf   = pyo.Param(m.Cset, initialize=data["Whf"], default=1e6)
    m.Tb45  = pyo.Param(initialize=data["Tb45"])
    m.Tb15  = pyo.Param(initialize=data["Tb15"])
    m.Tb30  = pyo.Param(initialize=data["Tb30"])
    m.Tr1   = pyo.Param(initialize=data["Tr1"])
    m.Tr2   = pyo.Param(initialize=data["Tr2"])
    m.Tdrv_cons = pyo.Param(initialize=data["Tdrv_cons"])
    m.Tdrv_sh1  = pyo.Param(initialize=data["Tdrv_sh1"])
    m.Twrk_sh   = pyo.Param(initialize=data["Twrk_sh"])

    return N, C, K, R, Rseg, lb_t, ub_t

# ── Constraint blocks ─────────────────────────────────────────────────────────

def _add_pwl_charging_constraints(m, K, R, Rseg):
    """
    Piecewise-linear charging constraints (Montoya et al. 2017)
    + SOS2 adjacency constraints that linearise them.
    """
    R_list = sorted(R)
    K_max  = max(Rseg)
    mid    = [(i, k) for i in K for k in Rseg[:-1]]

    m.pwl_ea = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.ea[i] == sum(m.lam_a[i, k] * m.Ebar[k] for k in R))
    m.pwl_ed = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.ed[i] == sum(m.lam_d[i, k] * m.Ebar[k] for k in R))
    m.pwl_tc = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.tauc[i] == (sum(m.lam_d[i, k] * m.Tbar[k] for k in R)
                    - sum(m.lam_a[i, k] * m.Tbar[k] for k in R)))
    m.pwl_ca = pyo.Constraint(m.Kset, rule=lambda m, i:
        sum(m.lam_a[i, k] for k in R) == 1)
    m.pwl_cd = pyo.Constraint(m.Kset, rule=lambda m, i:
        sum(m.lam_d[i, k] for k in R) == 1)
    m.pwl_sa = pyo.Constraint(m.Kset, rule=lambda m, i:
        sum(m.mu_a[i, k] for k in Rseg) == 1)
    m.pwl_sd = pyo.Constraint(m.Kset, rule=lambda m, i:
        sum(m.mu_d[i, k] for k in Rseg) == 1)

    m.sos2_lo_a  = pyo.Constraint(m.Kset,
        rule=lambda m, i: m.lam_a[i, R_list[0]] <= m.mu_a[i, R_list[1]])
    m.sos2_hi_a  = pyo.Constraint(m.Kset,
        rule=lambda m, i: m.lam_a[i, R_list[-1]] <= m.mu_a[i, K_max])
    m.sos2_lo_d  = pyo.Constraint(m.Kset,
        rule=lambda m, i: m.lam_d[i, R_list[0]] <= m.mu_d[i, R_list[1]])
    m.sos2_hi_d  = pyo.Constraint(m.Kset,
        rule=lambda m, i: m.lam_d[i, R_list[-1]] <= m.mu_d[i, K_max])
    m.sos2_mid_a = pyo.Constraint(mid,
        rule=lambda m, i, k: m.lam_a[i, k] <= m.mu_a[i, k] + m.mu_a[i, k + 1])
    m.sos2_mid_d = pyo.Constraint(mid,
        rule=lambda m, i, k: m.lam_d[i, k] <= m.mu_d[i, k] + m.mu_d[i, k + 1])


def _add_break_rest_constraints(m, N, I_list, K_set, M_big, rho2_limit=3):
    """
    All break/rest constraints including the split-break phi tracker.

    Parameters
    ----------
    N           : int  — local destination index (last stop, activities forbidden)
    I_list      : list — all local stop indices
    K_set       : set  — CS stop indices (local)
    M_big       : float — big-M constant
    rho2_limit  : int  — max reduced rests (r2) allowed
    """
    non_K = [i for i in I_list if i not in K_set]

    m.qb_K    = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.taub_hat[i] == m.taub[i] + m.tauc[i])
    m.qb_nonK = pyo.Constraint(non_K,  rule=lambda m, i:
        m.taub_hat[i] == m.taub[i])

    m.one_brk = pyo.Constraint(m.I, rule=lambda m, i:
        m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i] <= 1)

    m.brk45  = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb45 * m.x_b45[i])
    m.brk15  = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb15 * m.x_b15[i])
    m.brk30  = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb30 * m.x_b30[i])
    m.brk_ub = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub[i] <= M_big * (m.x_b45[i] + m.x_b15[i] + m.x_b30[i]))

    m.split_ord = pyo.Constraint(m.I, rule=lambda m, i: m.x_b30[i] <= m.phi[i])

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

    m.rst1   = pyo.Constraint(m.I, rule=lambda m, i: m.taur[i] >= m.Tr1 * m.rho1[i])
    m.rst2   = pyo.Constraint(m.I, rule=lambda m, i: m.taur[i] >= m.Tr2 * m.rho2[i])
    m.rst_ub = pyo.Constraint(m.I, rule=lambda m, i:
        m.taur[i] <= M_big * (m.rho1[i] + m.rho2[i]))
    m.rst_lim = pyo.Constraint(
        expr=sum(m.rho2[i] for i in I_list) <= rho2_limit)


def _add_hos_accumulator_constraints(m, N, I_list, C_set, K_set,
                                     S_dict, M_drv, M_sd, M_sw, TK):
    """
    Hours-of-Service accumulator propagation: cd, sd, sw.

    Big-M linearisation: l_k[i] = accumulator_k[i] if reset at stop i, else 0.
    """
    # --- Consecutive driving (reset by b45, b30, or any rest) ---
    m.l1u1 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l1[i] <= M_drv * _model_ri(m, i))
    m.l1u2 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l1[i] <= m.cd[i])
    m.l1lb  = pyo.Constraint(m.I,
        rule=lambda m, i: m.l1[i] >= m.cd[i] - M_drv * (1 - _model_ri(m, i)))

    def _cd(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.cd[i+1] == m.cd[i] + m.D_nom[i] - m.l1[i]
    m.cd_prop = pyo.Constraint(m.I, rule=_cd)
    m.cd_ub   = pyo.Constraint(m.I, rule=lambda m, i: m.cd[i] <= m.Tdrv_cons)

    # --- Shift driving (reset by any rest only) ---
    m.l2u1 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l2[i] <= M_sd * _model_rho(m, i))
    m.l2u2 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l2[i] <= m.sd[i])
    m.l2lb  = pyo.Constraint(m.I,
        rule=lambda m, i: m.l2[i] >= m.sd[i] - M_sd * (1 - _model_rho(m, i)))

    def _sd(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.sd[i+1] == m.sd[i] + m.D_nom[i] - m.l2[i]
    m.sd_prop = pyo.Constraint(m.I, rule=_sd)
    m.sd_ub   = pyo.Constraint(m.I, rule=lambda m, i: m.sd[i] <= m.Tdrv_sh1)

    # --- Shift working (reset by any rest; charge counts as work unless break) ---
    m.u_ub1 = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.u[i] <= TK * (1 - _model_xsum(m, i)))
    m.u_ub2 = pyo.Constraint(m.Kset,
        rule=lambda m, i: m.u[i] <= m.tauc[i])
    m.u_lb  = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.u[i] >= m.tauc[i] - TK * _model_xsum(m, i))

    m.l4u1 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l4[i] <= M_sw * _model_rho(m, i))
    m.l4u2 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l4[i] <= m.sw[i])
    m.l4lb  = pyo.Constraint(m.I,
        rule=lambda m, i: m.l4[i] >= m.sw[i] - M_sw * (1 - _model_rho(m, i)))

    def _sw(m, i):
        if i >= N: return pyo.Constraint.Skip
        man_i = m.Man[i] * m.z_man[i]   # manoeuver counts as working time
        ip1   = i + 1
        if ip1 in K_set:
            work_next = m.Q_nom[ip1] * m.y[ip1] + m.u[ip1]
        elif ip1 in C_set:
            work_next = S_dict.get(ip1, 0)
        else:
            work_next = 0
        return m.sw[i+1] == m.sw[i] - m.l4[i] + man_i + m.D_nom[i] + work_next
    m.sw_prop = pyo.Constraint(m.I, rule=_sw)
    m.sw_ub   = pyo.Constraint(m.I, rule=lambda m, i: m.sw[i] <= m.Twrk_sh)


def _add_manoeuver_constraints(m, I_list, K_set):
    """
    Link z_man to break/rest/charge decisions.

    z_man[i] = 1 iff any activity requiring a physical manoeuver happens:
      - All stops : break or rest  (x_b45/b15/b30 or rho1/rho2)
      - CS stops  : additionally, charging (y=1) triggers a manoeuver

    """
    # Lower bound from break/rest decisions at all stops
    m.z_man_brk = pyo.Constraint(m.I, rule=lambda m, i:
        m.z_man[i] >= _model_xsum(m, i))

    # Lower bound from charging at CS stops
    def _z_man_chg(m, i):
        if i not in K_set:
            return pyo.Constraint.Skip
        return m.z_man[i] >= m.y[i]
    m.z_man_chg = pyo.Constraint(m.I, rule=_z_man_chg)


def _add_soc_constraints(m, N, data):
    def _soc(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.ea[i + 1] == m.ed[i] - m.Eparam[i]
    m.soc_prop   = pyo.Constraint(m.I,    rule=_soc)
    m.soc_nc_C   = pyo.Constraint(m.Cset, rule=lambda m, i: m.ed[i] == m.ea[i])
    m.soc_mono_K = pyo.Constraint(m.Kset, rule=lambda m, i: m.ed[i] >= m.ea[i])
    m.soc_lb     = pyo.Constraint(m.I,    rule=lambda m, i: m.ea[i] >= m.Emin)
    m.soc_ub     = pyo.Constraint(m.I,    rule=lambda m, i: m.ed[i] <= m.Ecap)
    m.chg_act    = pyo.Constraint(m.Kset, rule=lambda m, i: m.tauc[i] <= m.TK * m.y[i])
    m.chg_act2   = pyo.Constraint(m.Kset, rule=lambda m, i: m.tauc[i] >= 0.25 * m.y[i])

    usable_cap   = data["Ecap"] - data["Emin"]
    initial_free = data["E0"]   - data["Emin"]
    total_energy = sum(data["E"].values())
    n_min = max(0, _mi.ceil((total_energy - initial_free) / usable_cap))

    #m.ineq1 = pyo.Constraint(expr=sum(m.y[i] for i in m.Kset) >= n_min)

    m.pwl_no_free_charge = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.ed[i] - m.ea[i] <= m.Ecap * m.y[i])

def _add_time_constraints(m, N):
    def _tp(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.ta[i + 1] == m.td[i] + m.D_nom[i]
    m.time_prop = pyo.Constraint(m.I, rule=_tp)

    m.td_C = pyo.Constraint(m.Cset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.S[i] + m.taub[i] + m.taur[i]
                 + m.Man[i] * m.z_man[i])

    m.td_K = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.Q_nom[i]*m.y[i] + m.tauc[i] + m.taub[i] + m.taur[i]
                 + m.Man[i] * m.z_man[i])

    m.tw_hard = pyo.Constraint(m.Cset, rule=lambda m, i:
        pyo.inequality(m.Wha[i], m.ta[i], m.Whf[i]))

# ── Quiet solver wrapper ──────────────────────────────────────────────────────

def _solve_quiet(solver, model, tee, warmstart=False):
    """
    Run solver.solve(), suppressing all output when tee=False.
    Exported for use by the horizon sub-problem solver.
    """
    if tee:
        return solver.solve(model, tee=True, warmstart=warmstart)
    _sink = _io.StringIO()
    with _ctx.redirect_stdout(_sink), _ctx.redirect_stderr(_sink):
        return solver.solve(model, tee=False, warmstart=warmstart)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — FULL-ROUTE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(data: dict) -> pyo.ConcreteModel:
    """
    Build and return the full-route Pyomo model for data['I'] = 0..N.

    The model minimises arrival time at stop N subject to:
      - Time propagation (nominal drive times D)
      - Battery SOC (PWL charging, energy per leg)
      - Hours-of-Service: consecutive driving cd, shift driving sd, shift work sw
      - Break / rest decisions (b45, b15, b30, r1, r2) with split-break logic
      - Time windows at customer stops
      - Manoeuver times

    data : dict produced by instances.make_data().
    """
    m = pyo.ConcreteModel()

    N, C, K, R, Rseg, lb_t, ub_t= _declare_common_params(m, data)

    # ── Variables ─────────────────────────────────────────────────────────────
    _declare_common_vars(m)

    # ── Objective ─────────────────────────────────────────────────────────────
    m.obj = pyo.Objective(expr=m.ta[N], sense=pyo.minimize)

    #for i in data["I"]:
     #   m.ta[i].setlb(lb_t.get(i, 0.0))
      #  m.ta[i].setub(ub_t.get(i, data["T_hor"]))

    # ── Initial conditions ────────────────────────────────────────────────────
    m.init_ta  = pyo.Constraint(expr=m.ta[0] == data.get("T_START", 0.0))
    m.init_ea  = pyo.Constraint(expr=m.ea[0] == m.E0)
    m.init_cd  = pyo.Constraint(expr=m.cd[0] == 0)
    m.init_sd  = pyo.Constraint(expr=m.sd[0] == 0)
    m.init_sw  = pyo.Constraint(expr=m.sw[0] == 0)
    m.init_phi = pyo.Constraint(expr=m.phi[0] == 0)

    for v in [m.x_b45, m.x_b15, m.x_b30, m.rho1, m.rho2]:
        v[0].fix(0); v[N].fix(0)
    m.taub[0].fix(0); m.taur[0].fix(0)
    m.taub[N].fix(0); m.taur[N].fix(0)

    m.td_orig = pyo.Constraint(expr=m.td[0] == m.ta[0])
    m.td_dest = pyo.Constraint(expr=m.td[N] == m.ta[N])

    m.soc_nc_orig= pyo.Constraint(expr=m.ed[0] == m.ea[0])
    m.soc_nc_dest= pyo.Constraint(expr=m.ed[N] == m.ea[N])


    # ── Shared constraint blocks ───────────────────────────────────────────────
    _add_soc_constraints(m, N, data)
    _add_time_constraints(m, N)
    _add_manoeuver_constraints(m, data["I"], set(K))
    _add_pwl_charging_constraints(m, K, R, Rseg)
    _add_break_rest_constraints(m, N, data["I"], set(K), m.M_big, rho2_limit=3)
    _add_hos_accumulator_constraints(m, N, data["I"], set(C), set(K),
                                     data["S"], m.M_drv, m.M_sd, m.M_sw, m.TK)
    return m


def solve_model(model, tee=True):
    """Solve the full-route MIP to near-optimality (0.5% gap, 2h limit)."""
    solver = pyo.SolverFactory("appsi_highs")
    solver.options["mip_rel_gap"] = 0.005
    solver.options["time_limit"]  = 60 * 60 * 2
    solver.options["presolve"]    = "on"
    try:
        results = _solve_quiet(solver, model, tee)
        status  = str(results.solver.termination_condition)
    except RuntimeError:
        status  = "infeasible"
        results = None
    print(f"  Solver: {status}")
    return results, status


def extract_solution(model, data: dict) -> list[dict]:
    """Extract per-stop solution dicts (same schema as extract_horizon_solution)."""
    N = data["N"]
    K = data["K"]

    sol = []
    for i in data["I"]:
        is_K  = i in K
        y_val = round(pyo.value(model.y[i])) if is_K else 0
        tauq_val = data["Q"].get(i, 0.0) * y_val if is_K else 0.0

        sol.append(dict(
            i    = i,
            ta   = pyo.value(model.ta[i]),
            td   = pyo.value(model.td[i]),
            ea   = pyo.value(model.ea[i]),
            ed   = pyo.value(model.ed[i]),
            cd   = pyo.value(model.cd[i]),
            sd   = pyo.value(model.sd[i]),
            sw   = pyo.value(model.sw[i]),
            tauc = pyo.value(model.tauc[i]) if is_K else 0.0,
            tauq = tauq_val,
            taub = pyo.value(model.taub[i]),
            taur = pyo.value(model.taur[i]),
            y    = y_val,
            b45  = round(pyo.value(model.x_b45[i])),
            b15  = round(pyo.value(model.x_b15[i])),
            b30  = round(pyo.value(model.x_b30[i])),
            rho1 = round(pyo.value(model.rho1[i])),
            rho2 = round(pyo.value(model.rho2[i])),
            is_C = i in data["C"],
            is_K = is_K,
            D_nom = data["D"].get(i, 0.0),
        ))
    return sol


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — ROLLING-HORIZON SUB-PROBLEM
# ══════════════════════════════════════════════════════════════════════════════

def make_subproblem_data(full_data: dict, start_stop: int, end_stop: int,
                         init_state: dict, D_override=None,
                         E_override=None) -> dict:
    """
    Slice the full route into a sub-problem over [start_stop, end_stop].

    Parameters
    ----------
    full_data   : dict from instances.make_data()
    start_stop  : global index of first stop  (local index 0)
    end_stop    : global index of last stop   (local index H)
    init_state  : dict with keys ta, ea, cd, sd, sw, phi  (from BEHDV.as_init_state)
    D_override  : dict {global_leg_index: hours} or None — scenario travel times
    E_override  : dict {global_leg_index: kWh}   or None — scenario energies

    Returns
    -------
    sub_data : dict — same schema as full_data but for the sub-route.
    """
    N_glob = full_data["N"]
    C_glob = set(full_data["C"])
    K_glob = set(full_data["K"])

    assert 0 <= start_stop < end_stop <= N_glob, (
        f"Invalid horizon [{start_stop}, {end_stop}] for route of length {N_glob}")

    H = end_stop - start_stop

    I_loc = list(range(H + 1))
    C_loc = [j for j in range(H + 1) if (start_stop + j) in C_glob]
    K_loc = [j for j in range(H + 1) if (start_stop + j) in K_glob]

    D_src = D_override if D_override is not None else full_data["D"]
    E_src = E_override if E_override is not None else full_data["E"]
    D_loc = {j: D_src.get(start_stop + j, 0.0) for j in range(H)}
    E_loc = {j: E_src.get(start_stop + j, 0.0) for j in range(H)}

    S_loc  = {j: full_data["S"].get(start_stop + j, 0.0) for j in C_loc}
    Q_loc  = {j: full_data["Q"].get(start_stop + j, 0.0) for j in K_loc}
    M_loc  = {}
    for j in range(H + 1):
        g = start_stop + j
        M_loc[j] = full_data["M"].get(g, 5.0 / 60) if g < N_glob else 0.0
    M_loc[H] = 0.0
    if start_stop == 0:
        M_loc[0] = 0.0

    Wha_loc = {j: full_data["Wha"].get(start_stop + j, 0.0)
               for j in C_loc if (start_stop + j) in full_data.get("Wha", {})}
    Whf_loc = {j: full_data["Whf"].get(start_stop + j, 1e6)
               for j in C_loc if (start_stop + j) in full_data.get("Whf", {})}

    t0    = init_state["ta"]
    T_hor = full_data["T_hor"]
    R     = full_data["R"]
    Rseg  = full_data["Rseg"]
    Tbar  = full_data["Tbar"]

    _man = full_data.get("M", {}).get(start_stop, 5.0 / 60)
    lb_t, ub_t = compute_time_bounds(I_loc, C_loc, K_loc, D_loc, S_loc, Q_loc,
                               Tbar, T_hor, t0=t0, Man_default=_man)

    # Minimum energy needed from each local stop to the next CS or destination.
    # Uses scenario energy within the horizon, nominal beyond.
    E_within      = E_override if E_override is not None else full_data["E"]
    E_global_all  = full_data["E"]
    K_global_set  = set(full_data["K"])
    e_to_next_cs  = {}
    for j in range(H + 1):
        g   = start_stop + j
        cum = 0.0
        k   = g
        while k < N_glob:
            E_use = E_within if k < start_stop + H else E_global_all
            cum += E_use.get(k, 0.0)
            if k + 1 in K_global_set or k + 1 == N_glob:
                break
            k += 1
        e_to_next_cs[j] = cum

    return dict(
        label=f"subproblem [{start_stop}→{end_stop}]",
        title=f"sub_{start_stop}_{end_stop}",
        N=H, I=I_loc, C=C_loc, K=K_loc, R=R, Rseg=Rseg,
        D=D_loc, E=E_loc,
        S=S_loc, Q=Q_loc, M=M_loc,
        Wha=Wha_loc, Whf=Whf_loc,
        E0=init_state["ea"],
        Ecap=full_data["Ecap"], Emin=full_data["Emin"],
        Ebar=full_data["Ebar"], Tbar=Tbar,
        T_hor=T_hor,
        lb_t=lb_t, ub_t=ub_t,
        Tb45=full_data["Tb45"], Tb15=full_data["Tb15"], Tb30=full_data["Tb30"],
        Tr1=full_data["Tr1"],   Tr2=full_data["Tr2"],
        Tdrv_cons=full_data["Tdrv_cons"],
        Tdrv_sh1=full_data["Tdrv_sh1"],
        Tdrv_sh2=full_data.get("Tdrv_sh2", 10.0),
        Twrk_sh=full_data["Twrk_sh"],
        M_drv=full_data["M_drv"], M_sd=full_data["M_sd"],
        M_sw=full_data["M_sw"],   M_big=full_data["M_big"],
        global_start=start_stop, global_end=end_stop, global_N=N_glob,
        E_global=full_data["E"],
        K_global=set(full_data["K"]),
        e_to_next_cs=e_to_next_cs,
    )


def build_horizon_model(sub_data: dict, init_state: dict,
                        fixed_action=None, rho2_remaining: int = 3,
                        tee=False) -> pyo.ConcreteModel:
    """
    Build the rolling-horizon sub-problem Pyomo model.

    Differences from build_model (full-route):
    1. Initial conditions come from init_state (not hardcoded to origin).
    2. fixed_action (if given) fixes binary decisions at local stop 0.
    3. rho2_remaining caps the reduced-rest budget in the sub-window.
    4. No activities allowed at the last horizon stop (binaries fixed 0).
    5. Extra SOC bounds tighten the LP relaxation.
    6. sw[0] override when init_state.sw is already near the limit.

    Parameters
    ----------
    sub_data       : dict from make_subproblem_data
    init_state     : dict — ta, ea, cd, sd, sw, phi
    fixed_action   : dict — y, break_type, rest_type (or None for free solve)
    rho2_remaining : int  — remaining r2 budget (3 − vehicle.rho2_used)
    """
    import logging as _lg
    _lg.getLogger("pyomo").setLevel(_lg.ERROR)

    m = pyo.ConcreteModel()

    N, C, K, R, Rseg, lb_t, ub_t = _declare_common_params(m, sub_data)


    # ── Variables ─────────────────────────────────────────────────────────────
    _declare_common_vars(m)


    m.obj = pyo.Objective(
        expr  = m.ta[N] - m.ea[N]/5000,
        sense = pyo.minimize,
    )

    #for i in sub_data["I"]:
     #   m.ta[i].setlb(lb_t.get(i, 0.0))
      #  m.ta[i].setub(ub_t.get(i, sub_data["T_hor"]))

    # ── Initial conditions from init_state ────────────────────────────────────
    m.init_ta  = pyo.Constraint(expr=m.ta[0]  == init_state["ta"])
    m.init_ea  = pyo.Constraint(expr=m.ea[0]  == init_state["ea"])
    m.init_cd  = pyo.Constraint(expr=m.cd[0]  == init_state["cd"])
    m.init_sd  = pyo.Constraint(expr=m.sd[0]  == init_state["sd"])
    m.init_phi = pyo.Constraint(expr=m.phi[0] == init_state["phi"])
    m.init_sw = pyo.Constraint(expr=m.sw[0] == init_state["sw"])

    # ── Fixed action at stop 0 ────────────────────────────────────────────────
    if fixed_action is not None:
        brk = fixed_action.get("break_type")   # None | "b45" | "b15" | "b30" | "0"
        rst = fixed_action.get("rest_type")    # None | "r1"  | "r2" | "0"

        # Fix y[0] if specified and stop 0 is a CS
        if 0 in set(K) and fixed_action.get("y") is not None:
            m.fix_y0 = pyo.Constraint(expr=m.y[0] == int(fixed_action["y"]))

        # Fix break binaries only when a specific break type is chosen
        if brk is not None:
            m.fix_b45_0 = pyo.Constraint(expr=m.x_b45[0] == (1 if brk == "b45" else 0))
            m.fix_b15_0 = pyo.Constraint(expr=m.x_b15[0] == (1 if brk == "b15" else 0))
            m.fix_b30_0 = pyo.Constraint(expr=m.x_b30[0] == (1 if brk == "b30" else 0))

        # Fix rest binaries only when a specific rest type is chosen
        if rst is not None:
            m.fix_rho1_0 = pyo.Constraint(expr=m.rho1[0] == (1 if rst == "r1" else 0))
            m.fix_rho2_0 = pyo.Constraint(expr=m.rho2[0] == (1 if rst == "r2" else 0))

        # When neither break nor rest is specified (charge_only mode): nothing


    # ── No activities at last horizon stop ─────────────────────────────────────
    # Same reasoning: use equality constraints, not .fix(), so they survive
    # the LP relaxation transformation.
    m.fix_b45_N  = pyo.Constraint(expr=m.x_b45[N] == 0)
    m.fix_b15_N  = pyo.Constraint(expr=m.x_b15[N] == 0)
    m.fix_b30_N  = pyo.Constraint(expr=m.x_b30[N] == 0)
    m.fix_rho1_N = pyo.Constraint(expr=m.rho1[N]  == 0)
    m.fix_rho2_N = pyo.Constraint(expr=m.rho2[N]  == 0)
    m.fix_taub_N = pyo.Constraint(expr=m.taub[N]  == 0)
    m.fix_taur_N = pyo.Constraint(expr=m.taur[N]  == 0)
    if N in set(K):
        m.fix_y_N = pyo.Constraint(expr=m.y[N] == 0)
    m.td_dest = pyo.Constraint(expr=m.td[N] == m.ta[N])
    m.soc_nc_dest= pyo.Constraint(expr=m.ed[N] == m.ea[N])


    # ── Shared constraint blocks ───────────────────────────────────────────────

    _add_soc_constraints(m, N, sub_data)
    _add_time_constraints(m, N)
    _add_manoeuver_constraints(m, sub_data["I"], set(K))
    _add_pwl_charging_constraints(m, K, R, Rseg)
    _add_break_rest_constraints(m, N, sub_data["I"], set(K), m.M_big,
                            rho2_limit=rho2_remaining)
    _add_hos_accumulator_constraints(m, N, sub_data["I"], set(C), set(K),
                                    sub_data["S"], m.M_drv, m.M_sd, m.M_sw, m.TK)



    return m


def _inject_warm_start(model, warm_sol, start_stop):
    """
    Inject a previous solution as warm-start hints into a horizon model.
    warm_sol : list of per-stop dicts with LOCAL indices (0 = start_stop).
    """
    import warnings as _ws
    with _ws.catch_warnings():
        _ws.simplefilter("ignore")
        for s in warm_sol:
            i = s.get("i")
            if i is None or i not in model.I:
                continue
            _sv = lambda var, key, default=0.0: (
                var[i].set_value(s.get(key, default))
                if i in var else None)
            _sv(model.ta,    "ta")
            _sv(model.td,    "td")
            _sv(model.ea,    "ea")
            _sv(model.ed,    "ed")
            _sv(model.cd,    "cd")
            _sv(model.sd,    "sd")
            _sv(model.sw,    "sw")
            _sv(model.taub,  "taub")
            _sv(model.taur,  "taur")
            _sv(model.x_b45, "b45")
            _sv(model.x_b15, "b15")
            _sv(model.x_b30, "b30")
            _sv(model.rho1,  "rho1")
            _sv(model.rho2,  "rho2")
            if i in model.Kset:
                _sv(model.y,    "y")
                _sv(model.tauc, "tauc")


def _solve_horizon_model(model, time_limit=8, tee=False, relax=True, had_warm=False):
    """
    Solve a horizon model with HiGHS.

    Returns (results, status_str, solve_info).
    solve_info keys: wall_s, obj, n_vars, n_cons, had_warm, relax, status.
    """
    import time as _tm
    n_vars_pre = sum(1 for _ in model.component_data_objects(pyo.Var, active=True))
    n_cons_pre = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))

    if relax:
        try:
            pyo.TransformationFactory("core.relax_integer_vars").apply_to(model)
        except KeyError:
            pyo.TransformationFactory("core.relax_integrality").apply_to(model)

    if False:
        for var in model.component_objects(pyo.Var, active=True):
            if var.name in ("mu_a", "mu_d"):
                continue          # keep binary — SOS2 adjacency must be integer
            for vdata in var.values():
                if vdata.domain in (pyo.Binary, pyo.Integers,
                                    pyo.NonNegativeIntegers):
                    vdata.domain = pyo.NonNegativeReals
                    if vdata.ub is None:
                        vdata.setub(1.0)

    solver = pyo.SolverFactory("appsi_highs")
    solver.options["presolve"]   = "on"
    solver.options["time_limit"] = time_limit
    if tee:
        solver.options["write_model_file"] = "debug_stop23.lp"
    if not relax:
        solver.options["mip_rel_gap"] = 0.05

    t0 = _tm.perf_counter()
    try:
        use_warmstart = had_warm and not relax
        results = _solve_quiet(solver, model, tee, warmstart=use_warmstart)
        status  = str(results.solver.termination_condition)
    except RuntimeError:
        status  = "infeasible"
        results = None
    wall_s = _tm.perf_counter() - t0

    obj_val = None
    if status in ("optimal", "feasible", "maxTimeLimit"):
        try:
            obj_val = pyo.value(model.obj)
        except Exception:
            pass

    solve_info = dict(wall_s=wall_s, obj=obj_val, n_vars=n_vars_pre,
                      n_cons=n_cons_pre, had_warm=had_warm,
                      relax=relax, status=status)
    return results, status, solve_info


def extract_horizon_solution(model, sub_data: dict) -> list[dict]:
    """Extract per-stop solution dicts from a solved horizon model (local indices)."""
    N     = sub_data["N"]
    K     = sub_data["K"]
    K_set = set(K)

    sol = []
    for i in sub_data["I"]:
        is_K  = i in K_set
        y_val = round(pyo.value(model.y[i])) if is_K else 0
        tauq_val = sub_data["Q"].get(i, 0.0) * y_val if is_K else 0.0

        sol.append(dict(
            i    = i,
            ta   = pyo.value(model.ta[i]),
            td   = pyo.value(model.td[i]),
            ea   = pyo.value(model.ea[i]),
            ed   = pyo.value(model.ed[i]),
            cd   = pyo.value(model.cd[i]),
            sd   = pyo.value(model.sd[i]),
            sw   = pyo.value(model.sw[i]),
            tauc = pyo.value(model.tauc[i]) if is_K else 0.0,
            tauq = tauq_val,
            taub = pyo.value(model.taub[i]),
            taur = pyo.value(model.taur[i]),
            y    = y_val,
            b45  = round(pyo.value(model.x_b45[i])),
            b15  = round(pyo.value(model.x_b15[i])),
            b30  = round(pyo.value(model.x_b30[i])),
            rho1 = round(pyo.value(model.rho1[i])),
            rho2 = round(pyo.value(model.rho2[i])),
            is_C = i in set(sub_data["C"]),
            is_K = is_K,
            D_nom = sub_data["D"].get(i, 0.0),
        ))
    return sol


def solve_horizon(full_data: dict, start_stop: int, end_stop: int,
                  init_state: dict, fixed_action=None,
                  D_override=None, E_override=None,
                  rho2_remaining: int = 3, tee: bool = False,
                  time_limit: int = 30, relax: bool = True,
                  warm_start=None) -> dict:
    """
    End-to-end rolling-horizon solve: build → (warm-start) → solve → extract.

    This is the primary entry point called by Simulation.py and greedy.py.

    Parameters
    ----------
    full_data       : dict from instances.make_data()
    start_stop      : global index of first stop in window
    end_stop        : global index of last stop  in window
    init_state      : dict — ta, ea, cd, sd, sw, phi  (BEHDV.as_init_state())
    fixed_action    : dict — y, break_type, rest_type — or None (free solve)
    D_override      : dict {global_leg: hours}   — scenario travel times
    E_override      : dict {global_leg: kWh}     — scenario energy consumption
    rho2_remaining  : int  — remaining r2 rest budget (3 − vehicle.rho2_used)
    tee             : bool — print solver output
    time_limit      : int  — solver wall-clock limit (seconds)
    relax           : bool — True → LP relaxation (fast); False → full MIP
    warm_start      : list of stop dicts (local indices) for warm-starting

    Returns
    -------
    dict:
        'feasible'     : bool
        'obj'          : float — ta at end_stop, or INFEASIBLE_PENALTY
        'sol'          : list of stop dicts (local indices)  or []
        'status'       : str
        'first_action' : dict summarising decisions at local stop 0
        'solve_info'   : dict (wall_s, n_vars, n_cons, …)
    """
    sub_data = make_subproblem_data(full_data, start_stop, end_stop,
                                    init_state, D_override=D_override,
                                    E_override=E_override)
    model    = build_horizon_model(sub_data, init_state,
                                   fixed_action=fixed_action,
                                   rho2_remaining=rho2_remaining,
                                   tee=tee)
    _had_warm = bool(warm_start)
    if warm_start:
        _inject_warm_start(model, warm_start, start_stop)

    _, status, solve_info = _solve_horizon_model(model, time_limit=time_limit,
                                                 tee=tee, relax=relax,
                                                 had_warm=_had_warm)

    feasible = status in ("optimal", "feasible")

    if not feasible:
        return dict(feasible=False, obj=INFEASIBLE_PENALTY,
                    sol=[], status=status, first_action=None,
                    solve_info=solve_info)

    sol = extract_horizon_solution(model, sub_data)
    s0  = sol[0]

    first_action = dict(
        y          = s0["y"],
        break_type = ("b45" if s0["b45"] else
                      "b15" if s0["b15"] else
                      "b30" if s0["b30"] else None),
        rest_type  = ("r1"  if s0["rho1"] else
                      "r2"  if s0["rho2"] else None),
        taub       = s0["taub"],
        tauc       = s0["tauc"],
        taur       = s0["taur"],
        tauq       = s0["tauq"],
    )

    return dict(
        feasible=True,
        obj=pyo.value(model.obj),
        sol=sol,
        status=status,
        first_action=first_action,
        solve_info=solve_info,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — IO HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def solution_path(name: str) -> str:
    _ensure_dirs()
    return os.path.join(SOLUTIONS_DIR, f"{name}.json")


def save_solution(sol: list, data: dict, name: str):
    _ensure_dirs()
    payload = {
        "name": name,
        "data": {
            "label": data["label"], "N": data["N"],
            "I": data["I"], "C": data["C"], "K": data["K"],
            "Emin": data["Emin"], "Ecap": data["Ecap"],
            "Tdrv_cons": data["Tdrv_cons"],
            "Tdrv_sh1":  data["Tdrv_sh1"],
            "Twrk_sh":   data["Twrk_sh"],
            "D": {str(k): v for k,v in data["D"].items()},
            "E": {str(k): v for k,v in data["E"].items()},
            "S": {str(k): v for k,v in data["S"].items()},
            "Q": {str(k): v for k,v in data["Q"].items()},
        },
        "sol": sol,
    }
    fpath = solution_path(name)
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Solution saved : {fpath}")


def load_solution(name: str) -> tuple[list, dict]:
    fpath = solution_path(name)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No saved solution at '{fpath}'.")
    with open(fpath) as f:
        payload = json.load(f)
    d = payload["data"]
    for fld in ("D", "E", "S", "Q"):
        d[fld] = {int(k): v for k, v in d[fld].items()}
    for fld in ("I", "C", "K"):
        d[fld] = [int(x) for x in d[fld]]
    sol = payload["sol"]
    for s in sol:
        s["i"] = int(s["i"])
        s.setdefault("D_nom", 0.0)
    print(f"  Solution loaded: {fpath}")
    return sol, d


def check_solution(sol: list, data: dict) -> bool:
    """Feasibility check: print any violated constraints, return True iff all OK."""
    print("\n  === Feasibility check ===")
    ok = True; N = data["N"]
    for idx, s in enumerate(sol):
        i = s["i"]
        if s["ta"] > s["td"] + EPS and i != 0:
            print(f"  WARN  ta>td stop {i}: ta={s['ta']:.3f} td={s['td']:.3f}")
        if s["cd"] > data["Tdrv_cons"] + EPS:
            print(f"  FAIL  consec_drv stop {i}: {s['cd']:.3f} > {data['Tdrv_cons']}"); ok=False
        if s["sd"] > data["Tdrv_sh1"]  + EPS:
            print(f"  FAIL  shift_drv  stop {i}: {s['sd']:.3f} > {data['Tdrv_sh1']}");  ok=False
        if s["sw"] > data["Twrk_sh"]   + EPS:
            print(f"  FAIL  shift_wk   stop {i}: {s['sw']:.3f} > {data['Twrk_sh']}");   ok=False
        if s["ea"] < data["Emin"] - EPS:
            print(f"  FAIL  ea stop {i}: {s['ea']:.2f} < {data['Emin']}");               ok=False
        if s["ed"] > data["Ecap"] + EPS:
            print(f"  FAIL  ed stop {i}: {s['ed']:.2f} > {data['Ecap']}");               ok=False
        if i < N and idx+1 < len(sol):
            D_nom = s.get("D_nom", data["D"].get(i, 0))
            exp   = s["td"] + D_nom
            act   = sol[idx+1]["ta"]
            if abs(act - exp) > 5*EPS:
                print(f"  WARN  time-prop leg {i}: td={s['td']:.3f}+D={D_nom:.3f}"
                      f" ≠ ta[{i+1}]={act:.3f} (Δ={act-exp:.4f})")
    print("  OK — all checked." if ok else "  Some constraints violated.")
    return ok


def print_schedule(sol: list, data: dict):
    """Print a human-readable stop-by-stop schedule table."""
    N = data["N"]
    hdr = (f"  {'i':>3}  {'type':>5}  {'ta':>6}  {'td':>6}  "
           f"{'ea':>6}  {'ed':>6}  {'cd':>5}  {'sd':>5}  {'sw':>5}  "
           f"{'D':>5}  activity")
    print(f"\n{hdr}\n  {'─'*90}")
    for s in sol:
        i   = s["i"]
        typ = ("ORIG" if i==0 else "DEST" if i==N else "CUST" if s["is_C"] else "CS")
        acts = []
        if s["is_K"] and s["y"]:
            acts.append(f"CHG {s['ea']:.0f}→{s['ed']:.0f}kWh ({s['tauc']:.2f}h)"
                        f" Q={s['tauq']*60:.0f}m")
        if s["b45"]:  acts.append(f"B45 {s['taub']:.2f}h")
        if s["b15"]:  acts.append(f"B15 {s['taub']:.2f}h")
        if s["b30"]:  acts.append(f"B30 {s['taub']:.2f}h")
        if s["rho1"]: acts.append(f"REST-r1 {s['taur']:.1f}h")
        if s["rho2"]: acts.append(f"REST-r2 {s['taur']:.1f}h")
        D_nom = s.get("D_nom", 0.0)
        print(f"  {i:>3}  {typ:>5}  {s['ta']:>6.2f}  {s['td']:>6.2f}  "
              f"{s['ea']:>6.1f}  {s['ed']:>6.1f}  "
              f"{s['cd']:>5.2f}  {s['sd']:>5.2f}  {s['sw']:>5.2f}  "
              f"{D_nom:>5.3f}  "
              f"{', '.join(acts) or '—'}")


# ── Re-export for plots.py callers ────────────────────────────────────────────
from plots import plot_solution   # noqa: F401


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT  (python MILP.py [instance_name])
# ══════════════════════════════════════════════════════════════════════════════

def run_instance(data: dict, tee: bool = True, run: bool = True):
    """Solve and report a single instance. data = result of make_data()."""
    print(f"\n{'='*65}")
    print(f"  {data['label']}")
    print(f"  C={data['C']}  K={data['K']}")

    if not run:
        sol, _ = load_solution(data["title"])
        print_schedule(sol, data)
        check_solution(sol, data)
        plot_solution(sol, data, title=data["title"])
        return

    model = build_model(data)
    _, status = solve_model(model, tee=tee)
    if status in ("optimal", "feasible"):
        print(f"  Arrival at dest: {pyo.value(model.ta[data['N']]):.3f} h")
        sol = extract_solution(model, data)
        print_schedule(sol, data)
        check_solution(sol, data)
        save_solution(sol, data, data["title"])
        plot_solution(sol, data, title=data["title"])
    else:
        print(f"  No feasible solution (status={status}).")


if __name__ == "__main__":
    import sys
    from instances import ALL_INSTANCES
    random.seed(5)
    name = sys.argv[1] if len(sys.argv) > 1 else "realistic"
    tee  = True

    if name == "all":
        for n, fn in ALL_INSTANCES.items():
            run_instance(fn(), tee=tee)
    elif name in ALL_INSTANCES:
        run_instance(ALL_INSTANCES[name](), tee=tee)
    else:
        print(f"Unknown instance '{name}'. Choose: {list(ALL_INSTANCES)}")