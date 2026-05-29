"""
Electric Truck Scheduling MILP
================================
Stops indexed 0..N   (origin=0, destination=N).
  C ⊆ {1..N-1} : customer stops
  K ⊆ {1..N-1} : charging station stops  (C ∩ K = ∅)

Drive times D_i and queue times Q_i are fixed (deterministic) parameters.
Uncertainty in travel time is handled externally by the simulation layer
(simulation.py), which perturbs D values when building scenario sub-problems.

Queue time Q_i (h) : fixed per CS stop i (plug-in + waiting, independent of ToD).
Drive time  D_i (h): fixed nominal value; scenarios scale it by U(1−δ, 1+δ).

All times in HOURS, energy in kWh.
"""

import pyomo.environ as pyo
import numpy as np
import random, json, os
import io as _io, contextlib as _ctx
import math as _math_top, logging as _log_top

FIGURES_DIR   = "figures"
SOLUTIONS_DIR = "solutions"

def _ensure_dirs():
    os.makedirs(FIGURES_DIR,   exist_ok=True)
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)

# ============================================================
# TIME BOUNDS  (for variable lb/ub tightening)
# ============================================================

def _time_bounds(I, C, K, D, S, Q, Tbar, T_hor, t0=0.0, rho2_remaining=3):
    """
    Simple forward-pass conservative bounds on arrival times.
    D and Q are fixed scalars (no time-of-day multipliers).
    Returns lb[i], ub[i] — absolute arrival-time windows.

    rho2_remaining : int — reduced-rest budget (0..3). When >0, the minimum
                     rest in the upper bound uses Tr2=9h (tightest feasible);
                     when 0, only Tr1=11h rests are allowed.
    """
    N   = max(I)
    TK  = Tbar[max(Tbar)]
    Tr1 = 11.0; Tr2 = 9.0; Tb = 0.75
    T_rest_min = Tr2 if rho2_remaining > 0 else Tr1   # tightest possible rest
    C_s = set(C); K_s = set(K)

    lb = {0: t0}; ub = {0: t0}
    for i in range(N):
        if i in C_s:
            dmin = S.get(i, 0.0)
            dmax = S.get(i, 0.0) + Tb + T_rest_min + 0.1
        elif i in K_s:
            dmin = 0.0
            dmax = Q.get(i, 0.0) + TK + Tb + T_rest_min + 0.1
        else:
            dmin = dmax = 0.0
        lb[i + 1] = lb[i] + dmin + D.get(i, 0.0)
        ub[i + 1] = min(T_hor, ub[i] + dmax + D.get(i, 0.0))
    return lb, ub


# ============================================================
# _make_data
# ============================================================

def _make_data(I, C, K, D, E, S, E0, Ecap, Emin,
               Ebar, Tbar, Wha, Whf, label, title, km=None, Q=None):
    """
    Assemble the data dict consumed by build_model and MILP2.

    Parameters
    ----------
    km : dict {leg_index: km} — physical leg distances.  When provided,
         scenario generation couples energy to travel speed via ECR(v).
         When None, defaults to E (1 kWh/km backward compat).
    Q  : dict {cs_stop: queue_time_h} or None.
         Fixed queue times at each CS stop.  When None, queue times are
         drawn uniformly from U[0, 30] min using the *current* global
         random state — callers should set random.seed() beforehand if
         reproducibility is required.  Pass Q explicitly for full control.
    """
    N    = max(I)
    R    = sorted(Ebar.keys())
    Rseg = R[1:]
    assert set(C) | set(K) == set(I) - {0, N}
    assert not (set(C) & set(K))

    if Q is not None:
        Q_nom = dict(Q)
    else:
        Q_nom = {i: random.randint(0, 10) / 60 for i in K}   # fixed queue (h)

    M_man = {i: 15 / 60 for i in range(N)}
    T_START = 8.0          # 08:00 departure (hours)
    T_hor   = T_START + 5 * 24   # 128h absolute planning horizon

    km_dict = km if km is not None else dict(E)

    lb_t, ub_t = _time_bounds(I, C, K, D, S, Q_nom, Tbar, T_hor, t0=T_START)

    Wha_shifted = {k: v + T_START for k, v in Wha.items()}
    Whf_shifted = {k: v + T_START for k, v in Whf.items()}

    return dict(
        label=label, title=title,
        N=N, I=I, C=C, K=K, R=R, Rseg=Rseg,
        Q=Q_nom, M=M_man,
        D=D, E=E, km=km_dict, S=S,
        E0=E0, Ecap=Ecap, Emin=Emin,
        Ebar=Ebar, Tbar=Tbar, Wha=Wha_shifted, Whf=Whf_shifted,
        T_hor=T_hor, T_START=T_START,
        lb_t=lb_t, ub_t=ub_t,
        # HoS parameters
        Tb45=0.75, Tb15=0.25, Tb30=0.50,
        Tr1=11.0, Tr2=9.0,
        Tdrv_cons=4.5, Tdrv_sh1=9.0, Tdrv_sh2=10.0,
        Twrk_cons1=6.0, Twrk_cons2=9.0, Twrk_sh=13.0,
        M_drv=4.5, M_sd=10.0, M_sw=13.0, M_big=1000.0,
    )


# ============================================================
# MODULE-LEVEL PYOMO EXPRESSION HELPERS
# (Used in both build_model and MILP2.build_horizon_model
#  via the shared constraint helpers below.)
# ============================================================

def _model_xsum(m, i):
    """All binary break/rest decisions at stop i."""
    return m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]

def _model_ri(m, i):
    """Consecutive-driving reset indicator (b45, b30, or any rest)."""
    return m.x_b45[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]

def _model_rho(m, i):
    """Shift reset indicator (any rest: r1 or r2)."""
    return m.rho1[i] + m.rho2[i]


# ============================================================
# SHARED CONSTRAINT HELPERS
# These encapsulate the ~300 lines of constraints that are
# identical between the full-route model (build_model) and
# the rolling-horizon sub-problem (MILP2.build_horizon_model).
# Both callers must have already declared the standard Sets,
# Params, and variables (via _declare_common_vars) on m.
# ============================================================

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

    # z_man[i] ∈ [0,1] — manoeuver activity indicator.
    # =1 whenever the vehicle performs any activity that requires a
    # manoeuvring move (backing into a berth, dock positioning, etc.):
    #   • At customer stops : break or rest
    #   • At CS stops       : charge OR break OR rest
    # Constraints linking z_man to decisions are added by
    # _add_manoeuver_constraints, called from each model builder.
    m.z_man = pyo.Var(m.I, bounds=(0, 1))


def _add_pwl_charging_constraints(m, K, R, Rseg):
    """
    Add piecewise-linear charging constraints (Montoya et al. 2017)
    and the SOS2 adjacency constraints that linearise them.

    Precondition: m.Kset, m.Rset, m.RsegS, m.lam_a, m.lam_d, m.mu_a, m.mu_d,
                  m.tauc, m.ea, m.ed, m.Ebar, m.Tbar must exist on m.
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
    Add all break/rest constraints including the split-break phi tracker.

    Parameters
    ----------
    N           : int  — local destination index (last stop, activities forbidden)
    I_list      : list — all local stop indices
    K_set       : set  — CS stop indices (local)
    M_big       : float — big-M constant
    rho2_limit  : int  — max reduced rests (r2) allowed; 3 globally, may be
                         lower in rolling-horizon sub-problems.

    Precondition: m.I, m.Kset, m.x_b45/b15/b30, m.phi, m.taub, m.taub_hat,
                  m.rho1, m.rho2, m.taur, m.tauc, m.Tb45/Tb15/Tb30, m.Tr1,
                  m.Tr2 must exist on m.
    """
    non_K = [i for i in I_list if i not in K_set]

    # taub_hat = break + (charge counted as break at CS stops)
    m.qb_K    = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.taub_hat[i] == m.taub[i] + m.tauc[i])
    m.qb_nonK = pyo.Constraint(non_K,  rule=lambda m, i:
        m.taub_hat[i] == m.taub[i])

    # At most one activity type per stop
    m.one_brk = pyo.Constraint(m.I, rule=lambda m, i:
        m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i] <= 1)

    # Break duration lower bounds
    m.brk45  = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb45 * m.x_b45[i])
    m.brk15  = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb15 * m.x_b15[i])
    m.brk30  = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb30 * m.x_b30[i])
    m.brk_ub = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub[i] <= M_big * (m.x_b45[i] + m.x_b15[i] + m.x_b30[i]))

    # b30 only allowed after b15 (split-break ordering)
    m.split_ord = pyo.Constraint(m.I, rule=lambda m, i: m.x_b30[i] <= m.phi[i])

    # phi tracks whether b15 has been taken since the last reset
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

    # Rest duration lower bounds
    m.rst1   = pyo.Constraint(m.I, rule=lambda m, i: m.taur[i] >= m.Tr1 * m.rho1[i])
    m.rst2   = pyo.Constraint(m.I, rule=lambda m, i: m.taur[i] >= m.Tr2 * m.rho2[i])
    m.rst_ub = pyo.Constraint(m.I, rule=lambda m, i:
        m.taur[i] <= M_big * (m.rho1[i] + m.rho2[i]))
    m.rst_lim = pyo.Constraint(
        expr=sum(m.rho2[i] for i in I_list) <= rho2_limit)


def _add_hos_accumulator_constraints(m, N, I_list, C_set, K_set,
                                     S_dict, M_drv, M_sd, M_sw, TK):
    """
    Add Hours-of-Service accumulator propagation constraints for
    consecutive driving (cd), shift driving (sd), and shift working (sw).

    The reset logic uses auxiliary variables l1, l2, l4 (big-M linearisation):
        l_k[i] = accumulator_k[i]   if reset at stop i,  else 0
    so that  accumulator[i+1] = accumulator[i] + D[i] - l_k[i].

    Parameters
    ----------
    N        : int    — local destination (loop bound)
    I_list   : list   — all local stop indices
    C_set    : set    — customer stop indices (local)
    K_set    : set    — CS stop indices (local)
    S_dict   : dict   — service times {local_stop: hours}
    M_drv    : float  — big-M for consecutive driving reset (≥ Tdrv_cons)
    M_sd     : float  — big-M for shift driving reset (≥ Tdrv_sh1)
    M_sw     : float  — big-M for shift working reset (≥ Twrk_sh)
    TK       : float  — max charging time (upper bound on tauc, used for u)

    Precondition: all model variables and Tdrv_cons, Tdrv_sh1, Twrk_sh
                  Params must already exist on m.
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
    # u[i] = tauc[i] when no break/rest at stop i, else 0
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
    Constrain z_man[i] — the manoeuver activity indicator.

    z_man[i] = 1  ⟺  any activity requiring a physical manoeuver happens:
        • All stops   : break or rest taken  (x_b45/b15/b30 or rho1/rho2)
        • CS stops only: additionally, charging (y=1) triggers a manoeuver
                         (backing into the pantograph / coupling the cable)

    z_man ∈ [0,1] continuous; since Man[i] is a cost in the objective
    direction the solver will drive z_man to 0 whenever possible, so the
    [0,1] domain is equivalent to binary in a feasible optimal solution.

    Parameters
    ----------
    I_list : list — all stop indices in this model
    K_set  : set  — CS stop indices
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


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(data):
    m = pyo.ConcreteModel()

    N     = data["N"]
    C     = data["C"]
    K     = data["K"]
    R     = data["R"]
    Rseg  = data["Rseg"]
    TK    = data["Tbar"][max(R)]
    M_drv = data["M_drv"]
    M_sd  = data["M_sd"]
    M_sw  = data["M_sw"]
    M_big = data["M_big"]
    lb_t  = data["lb_t"]
    ub_t  = data["ub_t"]

    # ── Sets ──────────────────────────────────────────────────────────────
    m.I     = pyo.Set(initialize=data["I"], ordered=True)
    m.Cset  = pyo.Set(initialize=C)
    m.Kset  = pyo.Set(initialize=K)
    m.Rset  = pyo.Set(initialize=R,    ordered=True)
    m.RsegS = pyo.Set(initialize=Rseg, ordered=True)
    m.Legs  = pyo.Set(initialize=list(range(N)), ordered=True)

    # ── Parameters ────────────────────────────────────────────────────────
    m.D_nom = pyo.Param(m.Legs, initialize=data["D"])
    m.Q_nom = pyo.Param(m.Kset, initialize=data["Q"],  default=0)
    # Man is indexed over ALL stops (I, not Legs) so the shared HoS helper
    # can access m.Man[i] for any i; stops where no manoeuver applies get 0.
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

    # ── Variables (shared declaration) ────────────────────────────────────
    _declare_common_vars(m)
    _add_manoeuver_constraints(m, data["I"], set(K))

    # ── Objective ─────────────────────────────────────────────────────────
    m.obj = pyo.Objective(expr=m.ta[N], sense=pyo.minimize)

    # Tighten variable bounds (helps LP relaxation)
    for i in data["I"]:
        m.ta[i].setlb(lb_t.get(i, 0.0))
        m.ta[i].setub(ub_t.get(i, data["T_hor"]))

    # =========================================================
    # INITIAL CONDITIONS / BOUNDARY FIXES
    # =========================================================
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

    # =========================================================
    # TIME PROPAGATION
    # =========================================================
    def _tp(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.ta[i + 1] == m.td[i] + m.D_nom[i]
    m.time_prop = pyo.Constraint(m.I, rule=_tp)

    m.td_orig = pyo.Constraint(expr=m.td[0] == m.ta[0])
    m.td_dest = pyo.Constraint(expr=m.td[N] == m.ta[N])

    m.td_C = pyo.Constraint(m.Cset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.S[i] + m.taub[i] + m.taur[i]
                 + m.Man[i] * m.z_man[i])

    m.td_K = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.Q_nom[i]*m.y[i] + m.tauc[i] + m.taub[i] + m.taur[i]
                 + m.Man[i] * m.z_man[i])

    m.tw_hard = pyo.Constraint(m.Cset, rule=lambda m, i:
        pyo.inequality(m.Wha[i], m.ta[i], m.Whf[i]))

    # =========================================================
    # BATTERY SOC
    # =========================================================
    def _soc(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.ea[i + 1] == m.ed[i] - m.Eparam[i]
    m.soc_prop   = pyo.Constraint(m.I,    rule=_soc)
    m.soc_nc_orig= pyo.Constraint(expr=m.ed[0] == m.ea[0])
    m.soc_nc_dest= pyo.Constraint(expr=m.ed[N] == m.ea[N])
    m.soc_nc_C   = pyo.Constraint(m.Cset, rule=lambda m, i: m.ed[i] == m.ea[i])
    m.soc_mono_K = pyo.Constraint(m.Kset, rule=lambda m, i: m.ed[i] >= m.ea[i])
    m.soc_lb     = pyo.Constraint(m.I,    rule=lambda m, i: m.ea[i] >= m.Emin)
    m.soc_ub     = pyo.Constraint(m.I,    rule=lambda m, i: m.ed[i] <= m.Ecap)
    m.chg_act    = pyo.Constraint(m.Kset, rule=lambda m, i: m.tauc[i] <= TK * m.y[i])
    m.chg_act2   = pyo.Constraint(m.Kset, rule=lambda m, i: m.tauc[i] >= 0.25 * m.y[i])

    # Valid inequality: minimum number of CS stops that must charge.
    #
    # Starting energy available without any charge:
    #   initial_free = E0 - Emin   (what we can spend before hitting the floor)
    # Each charge refills at most usable_cap = Ecap - Emin kWh.
    # To cover total_energy kWh we need:
    #   initial_free + n * usable_cap >= total_energy
    #   n >= ceil((total_energy - initial_free) / usable_cap)
    #
    # Using floor(total_energy / usable_cap) instead is wrong when total_energy
    # is exactly divisible by usable_cap — overcounts by 1, which can make
    # a feasible solution infeasible (especially in the LP relaxation).
    import math as _math, logging as _log_ineq
    usable_cap   = data["Ecap"] - data["Emin"]
    initial_free = data["E0"]  - data["Emin"]
    total_energy = sum(data["E"].values())
    n_min = max(0, _math.ceil((total_energy - initial_free) / usable_cap))
    _log_ineq.getLogger(__name__).debug(
        "ineq1: total=%.0f kWh  usable=%.0f kWh  E0_free=%.0f kWh  min_charges=%d",
        total_energy, usable_cap, initial_free, n_min)
    m.ineq1 = pyo.Constraint(
        expr=sum(m.y[i] for i in m.Kset) >= n_min)

    # =========================================================
    # SHARED CONSTRAINT BLOCKS
    # =========================================================
    _add_pwl_charging_constraints(m, K, R, Rseg)
    _add_break_rest_constraints(m, N, data["I"], set(K), M_big, rho2_limit=3)
    _add_hos_accumulator_constraints(m, N, data["I"], set(C), set(K),
                                     data["S"], M_drv, M_sd, M_sw, TK)

    return m


# ============================================================
# SOLVE
# ============================================================

def _solve_quiet(solver, model, tee, warmstart=False):
    """
    Run solver.solve(), suppressing all output when tee=False.
    Exported for use by MILP2 and Simulation to avoid duplication.

    warmstart : bool — pass warmstart=True to HiGHS so it uses variable
                values (set via set_value()) as a starting incumbent.
                Only meaningful for MIP solves; ignored silently for LP.
    """
    if tee:
        return solver.solve(model, tee=True, warmstart=warmstart)
    _sink = _io.StringIO()
    with _ctx.redirect_stdout(_sink), _ctx.redirect_stderr(_sink):
        return solver.solve(model, tee=False, warmstart=warmstart)


def solve_model(model, tee=True):
    solver = pyo.SolverFactory("appsi_highs")
    solver.options["mip_rel_gap"] = 0.005   # tight for full-route solve
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


# ============================================================
# EXTRACT SOLUTION
# ============================================================

def extract_solution(model, data):
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


# ============================================================
# SAVE / LOAD  (compact — time-dep data recomputed on load)
# ============================================================

def solution_path(name):
    _ensure_dirs()
    return os.path.join(SOLUTIONS_DIR, f"{name}.json")


def save_solution(sol, data, name):
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


def load_solution(name):
    fpath = solution_path(name)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No saved solution at '{fpath}'.")
    with open(fpath) as f:
        payload = json.load(f)
    d = payload["data"]
    for fld in ("D","E","S","Q"):
        d[fld] = {int(k): v for k,v in d[fld].items()}
    for fld in ("I","C","K"):
        d[fld] = [int(x) for x in d[fld]]
    sol = payload["sol"]
    for s in sol:
        s["i"] = int(s["i"])
        s.setdefault("D_nom", 0.0)
    print(f"  Solution loaded: {fpath}")
    return sol, d


# ============================================================
# VISUALISATION  →  see plots.py
# ============================================================
from plots import plot_solution          # noqa: F401 — re-exported for callers


# ============================================================
# FEASIBILITY CHECK + SCHEDULE PRINT

def check_solution(sol, data):
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


def print_schedule(sol, data):
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


# ============================================================
# MAIN
# ============================================================

def run_instance(data, tee=True, run=True):
    """Solve and report a single instance. data = result of an instance generator."""
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