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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random, json, os, time as _time_mod

FIGURES_DIR   = "figures"
SOLUTIONS_DIR = "solutions"

def _ensure_dirs():
    os.makedirs(FIGURES_DIR,   exist_ok=True)
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)

import time

# ============================================================
# TIME BOUNDS  (for variable lb/ub tightening)
# ============================================================

def _time_bounds(I, C, K, D, S, Q, Tbar, T_hor, t0=0.0):
    """
    Simple forward-pass conservative bounds on arrival times.
    D and Q are fixed scalars (no time-of-day multipliers).
    Returns lb[i], ub[i] — absolute arrival-time windows.
    """
    N   = max(I)
    TK  = Tbar[max(Tbar)]
    Tr1 = 11.0; Tb = 0.75
    C_s = set(C); K_s = set(K)

    lb = {0: t0}; ub = {0: t0}
    for i in range(N):
        if i in C_s:
            dmin = S.get(i, 0.0)
            dmax = S.get(i, 0.0) + Tb + Tr1 + 0.1
        elif i in K_s:
            dmin = 0.0
            dmax = Q.get(i, 0.0) + TK + Tb + Tr1 + 0.1
        else:
            dmin = dmax = 0.0
        lb[i + 1] = lb[i] + dmin + D.get(i, 0.0)
        ub[i + 1] = min(T_hor, ub[i] + dmax + D.get(i, 0.0))
    return lb, ub


# ============================================================
# INSTANCES
# ============================================================


# ── Backward-compatible instance wrappers ─────────────────────────────────
# Instance generators live in instances.py.  These thin wrappers let
# existing code continue to do `from MILP import instance_break_forced`.
# They use a local import inside the function body to avoid the circular
# import that would arise if we imported test_instances at module level
# (test_instances itself imports _make_data from MILP).

def instance_tiny(*a, **kw):
    from instances import instance_tiny as _f; return _f(*a, **kw)

def instance_break_forced(*a, **kw):
    from instances import instance_break_forced as _f; return _f(*a, **kw)

def instance_charging_needed(*a, **kw):
    from instances import instance_charging_needed as _f; return _f(*a, **kw)

def instance_rest_forced(*a, **kw):
    from instances import instance_rest_forced as _f; return _f(*a, **kw)

def instance_3day(*a, **kw):
    from instances import instance_3day as _f; return _f(*a, **kw)

def instance_realistic(*a, **kw):
    from instances import instance_realistic as _f; return _f(*a, **kw)

# ============================================================
# _make_data
# ============================================================

def _make_data(I, C, K, D, E, S, E0, Ecap, Emin,
               Ebar, Tbar, Wha, Whf, label, title, km=None):
    """
    km : dict {leg_index: km} — physical leg distances in kilometres.
         When provided, scenario generation can couple energy consumption
         to travel speed via E_scen[i] = km[i] * f(v_scen[i]).
         When None, defaults to E (assuming 1 kWh/km for backward compat).
    """
    N    = max(I)
    R    = sorted(Ebar.keys())
    Rseg = R[1:]
    assert set(C) | set(K) == set(I) - {0, N}
    assert not (set(C) & set(K))

    Q_nom = {i: random.randint(0, 30) / 60 for i in K}   # fixed queue (h)
    M_man = {i: 5 / 60 for i in range(N)}
    T_START = 8.0          # 08:00 departure (hours)
    T_hor   = T_START + 5 * 24   # 128h absolute planning horizon

    # km: physical distances.  Default to E values (backward compat: 1 kWh/km).
    km_dict = km if km is not None else dict(E)

    lb_t, ub_t = _time_bounds(I, C, K, D, S, Q_nom, Tbar, T_hor, t0=T_START)

    # Shift customer time windows to be relative to T_START (08:00 departure)
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
    T_hor = data["T_hor"]

    lb_t    = data["lb_t"]
    ub_t    = data["ub_t"]
    Q_nom   = data["Q"]

    # ---- sets -----------------------------------------------
    m.I     = pyo.Set(initialize=data["I"], ordered=True)
    m.Cset  = pyo.Set(initialize=C)
    m.Kset  = pyo.Set(initialize=K)
    m.Rset  = pyo.Set(initialize=R,    ordered=True)
    m.RsegS = pyo.Set(initialize=Rseg, ordered=True)
    m.Legs  = pyo.Set(initialize=list(range(N)), ordered=True)



    # ---- parameters -----------------------------------------
    m.D_nom = pyo.Param(m.Legs, initialize=data["D"])
    m.Q_nom = pyo.Param(m.Kset, initialize=data["Q"],  default=0)
    m.Man   = pyo.Param(m.Legs, initialize=data["M"],  default=0)
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

    # ---- state variables ------------------------------------
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



    # ---- objective ------------------------------------------
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

    def _xsum(m, i):
        return m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]

    m.td_C = pyo.Constraint(m.Cset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.S[i] + m.taub[i] + m.taur[i]
                 + m.Man[i] * _xsum(m, i))

    # CS: fixed queue (Q_nom*y) + charge + break + rest
    m.td_K = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.Q_nom[i]*m.y[i] + m.tauc[i] + m.taub[i] + m.taur[i]
                 + m.Man[i] * _xsum(m, i))

    # Hard time windows
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

    print(f"Energy needed: {sum(data['E'].values()):.0f} kWh  "
          f"Capacity: {data['Ecap']:.0f} kWh  "
          f"Min charges: {sum(data['E'].values())/(0.8*data['Ecap']):.1f}")

    # easy valid inequalities:
    m.ineq1 = pyo.Constraint(expr=sum(m.y[i] for i in m.Kset) >= int(sum(data["E"].values()) / (0.8 * data["Ecap"])))
    # m.ineq2 = pyo.Constraint(m.Kset, rule=lambda m: sum(m.rho1[i] + m.rho2[i] for i in m.I) >= int()

    # PWL charging (Montoya et al. 2017)
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

    R_list = sorted(R); K_max = max(Rseg)
    mid_a = [(i, k) for i in K for k in Rseg[:-1]]
    mid_d = [(i, k) for i in K for k in Rseg[:-1]]

    m.sos2_lo_a  = pyo.Constraint(m.Kset, rule=lambda m,i: m.lam_a[i,R_list[0]] <= m.mu_a[i,R_list[1]])
    m.sos2_hi_a  = pyo.Constraint(m.Kset, rule=lambda m,i: m.lam_a[i,R_list[-1]] <= m.mu_a[i,K_max])
    m.sos2_lo_d  = pyo.Constraint(m.Kset, rule=lambda m,i: m.lam_d[i,R_list[0]] <= m.mu_d[i,R_list[1]])
    m.sos2_hi_d  = pyo.Constraint(m.Kset, rule=lambda m,i: m.lam_d[i,R_list[-1]] <= m.mu_d[i,K_max])
    m.sos2_mid_a = pyo.Constraint(mid_a, rule=lambda m,i,k: m.lam_a[i,k] <= m.mu_a[i,k]+m.mu_a[i,k+1])
    m.sos2_mid_d = pyo.Constraint(mid_d, rule=lambda m,i,k: m.lam_d[i,k] <= m.mu_d[i,k]+m.mu_d[i,k+1])

    # =========================================================
    # BREAKS AND RESTS
    # =========================================================
    non_K = [i for i in data["I"] if i not in K]
    m.qb_K    = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.taub_hat[i] == m.taub[i] + m.tauc[i])
    m.qb_nonK = pyo.Constraint(non_K,  rule=lambda m, i:
        m.taub_hat[i] == m.taub[i])

    m.one_brk = pyo.Constraint(m.I, rule=lambda m, i:
        m.x_b45[i]+m.x_b15[i]+m.x_b30[i]+m.rho1[i]+m.rho2[i] <= 1)
    m.brk45   = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb45 * m.x_b45[i])
    m.brk15   = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb15 * m.x_b15[i])
    m.brk30   = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb30 * m.x_b30[i])
    m.brk_ub  = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub[i] <= M_big*(m.x_b45[i]+m.x_b15[i]+m.x_b30[i]))

    m.split_ord = pyo.Constraint(m.I, rule=lambda m, i: m.x_b30[i] <= m.phi[i])

    def _phi1(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.phi[i+1] >= m.phi[i]+m.x_b15[i]-m.x_b30[i]-m.x_b45[i]-m.rho1[i]-m.rho2[i]
    def _phi2(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.phi[i+1] <= m.phi[i]+m.x_b15[i]
    def _phi3(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.phi[i+1] <= 1-m.x_b30[i]-m.x_b45[i]-m.rho1[i]-m.rho2[i]
    m.phi1 = pyo.Constraint(m.I, rule=_phi1)
    m.phi2 = pyo.Constraint(m.I, rule=_phi2)
    m.phi3 = pyo.Constraint(m.I, rule=_phi3)

    m.rst1    = pyo.Constraint(m.I, rule=lambda m,i: m.taur[i] >= m.Tr1*m.rho1[i])
    m.rst2    = pyo.Constraint(m.I, rule=lambda m,i: m.taur[i] >= m.Tr2*m.rho2[i])
    m.rst_ub  = pyo.Constraint(m.I, rule=lambda m,i:
        m.taur[i] <= M_big*(m.rho1[i]+m.rho2[i]))
    m.rst_lim = pyo.Constraint(
        expr=sum(m.rho2[i] for i in data["I"]) <= 3)

    # =========================================================
    # HoS ACCUMULATORS
    # =========================================================
    def _ri(m, i):  return m.x_b45[i]+m.x_b30[i]+m.rho1[i]+m.rho2[i]
    def _rho(m, i): return m.rho1[i]+m.rho2[i]

    # --- consecutive driving ---
    m.l1u1 = pyo.Constraint(m.I, rule=lambda m,i: m.l1[i] <= M_drv*_ri(m,i))
    m.l1u2 = pyo.Constraint(m.I, rule=lambda m,i: m.l1[i] <= m.cd[i])
    m.l1lb  = pyo.Constraint(m.I, rule=lambda m,i:
        m.l1[i] >= m.cd[i] - M_drv*(1-_ri(m,i)))

    def _cd(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.cd[i+1] == m.cd[i] + m.D_nom[i] - m.l1[i]
    m.cd_prop = pyo.Constraint(m.I, rule=_cd)
    m.cd_ub   = pyo.Constraint(m.I, rule=lambda m,i: m.cd[i] <= m.Tdrv_cons)

    # --- shift driving ---
    m.l2u1 = pyo.Constraint(m.I, rule=lambda m,i: m.l2[i] <= M_sd*_rho(m,i))
    m.l2u2 = pyo.Constraint(m.I, rule=lambda m,i: m.l2[i] <= m.sd[i])
    m.l2lb  = pyo.Constraint(m.I, rule=lambda m,i:
        m.l2[i] >= m.sd[i] - M_sd*(1-_rho(m,i)))

    def _sd(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.sd[i+1] == m.sd[i] + m.D_nom[i] - m.l2[i]
    m.sd_prop = pyo.Constraint(m.I, rule=_sd)
    m.sd_ub   = pyo.Constraint(m.I, rule=lambda m,i: m.sd[i] <= m.Tdrv_sh1)

    # --- shift working ---
    m.u_ub1 = pyo.Constraint(m.Kset, rule=lambda m,i:
        m.u[i] <= TK*(1-m.x_b45[i]-m.x_b15[i]-m.x_b30[i]-m.rho1[i]-m.rho2[i]))
    m.u_ub2 = pyo.Constraint(m.Kset, rule=lambda m,i: m.u[i] <= m.tauc[i])
    m.u_lb  = pyo.Constraint(m.Kset, rule=lambda m,i:
        m.u[i] >= m.tauc[i] - TK*(m.x_b45[i]+m.x_b15[i]+m.x_b30[i]+m.rho1[i]+m.rho2[i]))

    m.l4u1 = pyo.Constraint(m.I, rule=lambda m,i: m.l4[i] <= M_sw*_rho(m,i))
    m.l4u2 = pyo.Constraint(m.I, rule=lambda m,i: m.l4[i] <= m.sw[i])
    m.l4lb  = pyo.Constraint(m.I, rule=lambda m,i:
        m.l4[i] >= m.sw[i] - M_sw*(1-_rho(m,i)))

    def _sw(m, i):
        if i >= N: return pyo.Constraint.Skip
        man_i = m.Man[i] * _xsum(m, i)
        ip1 = i + 1
        if ip1 in K:
            work_next = m.Q_nom[ip1] * m.y[ip1] + m.u[ip1]
        elif ip1 in C:
            work_next = data["S"].get(ip1, 0)
        else:
            work_next = 0
        return m.sw[i+1] == m.sw[i] - m.l4[i] + man_i + m.D_nom[i] + work_next
    m.sw_prop = pyo.Constraint(m.I, rule=_sw)
    m.sw_ub   = pyo.Constraint(m.I, rule=lambda m,i: m.sw[i] <= m.Twrk_sh)

    return m


# ============================================================
# SOLVE
# ============================================================

import io as _io, contextlib as _ctx


def _solve_quiet(solver, model, tee):
    if tee:
        return solver.solve(model, tee=True)
    _sink = _io.StringIO()
    with _ctx.redirect_stdout(_sink), _ctx.redirect_stderr(_sink):
        return solver.solve(model, tee=False)


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
        # back-compat defaults
        s.setdefault("D_nom", 0.0)
    print(f"  Solution loaded: {fpath}")
    return sol, d


# ============================================================
# VISUALISATION
# ============================================================

COL = dict(
    drive   = "#2C6FAC",
    service = "#27AE60",
    queue   = "#C0392B",
    charge  = "#E67E22",
    brk     = "#F1C40F",
    rest    = "#8E44AD",
)
EPS = 1e-3


def _bar(ax, start, dur, y, h, color, label=None, fontsize=7, text_color="white"):
    if dur < EPS: return
    ax.barh(y, dur, left=start, height=h, color=color,
            edgecolor="white", linewidth=0.3)
    if dur > 0.08 and label:
        ax.text(start + dur/2, y, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", clip_on=True)


def _shade_tod(ax, t_start, t_end):
    """Shade 24-h time-of-day bands across the full timeline."""
    # 3 bands per day: night 0-6, day 6-20, evening 20-24
    band_col = ["#D6EAF8", "#FEF9E7", "#E8DAEF"]   # night, day, evening
    band_hrs = [(0, 6), (6, 20), (20, 24)]
    t = 0
    while t < t_end:
        day = int(t) // 24
        for (h0, h1), col in zip(band_hrs, band_col):
            s = day*24 + h0; e = day*24 + h1
            s = max(s, t_start); e = min(e, t_end)
            if e > s:
                ax.axvspan(s, e, color=col, alpha=0.25, zorder=0, lw=0)
        t += 24


def _draw_vlines(ax, vlines):
    seen = set()
    for (t, col, lw, alpha, ls) in vlines:
        key = round(t, 4)
        if key in seen: continue
        seen.add(key)
        ax.axvline(t, color=col, lw=lw, alpha=alpha, ls=ls, zorder=1)


def plot_solution(sol, data, title="solution"):
    N    = data["N"]
    tend = sol[-1]["ta"]

    fig, axes = plt.subplots(3, 1, figsize=(17, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.suptitle(f"{title}  —  {data['label']}", fontsize=12, fontweight="bold")

    # ---- collect vlines -----
    vlines = []
    for s in sol:
        t = ta = s["ta"]
        vlines.append((ta, "gray", 0.5, 0.30, "--"))
        if s["is_C"]: t += data["S"].get(s["i"], 0)
        if s["is_K"] and s["y"]:
            if s["tauq"] > EPS: t += s["tauq"]; vlines.append((t, COL["queue"],  0.6, 0.28, ":"))
            if s["tauc"] > EPS: t += s["tauc"]; vlines.append((t, COL["charge"], 0.7, 0.33, ":"))
        if s["taub"] > EPS: vlines.append((t, COL["brk"],  0.8, 0.48, "--")); t += s["taub"]
        if s["taur"] > EPS: vlines.append((t, COL["rest"], 1.0, 0.52, "--"))

    # ============ Panel 1: Gantt ==============================
    ax = axes[0]; ax.set_title("Activity timeline", fontsize=10)
    _shade_tod(ax, 0, tend)
    Y, H = 0.5, 0.38
    for s in sol:
        i = s["i"]
        if i > 0:
            _bar(ax, sol[i-1]["td"], s["ta"]-sol[i-1]["td"], Y, H, COL["drive"],
                 label=f"drv→{i}", fontsize=6.5)
        t = s["ta"]
        if s["is_C"]:
            svc = data["S"].get(i, 0)
            _bar(ax, t, svc, Y, H, COL["service"], label=f"C{i}", fontsize=7); t += svc
        if s["is_K"] and s["y"] and s["tauq"] > EPS:
            _bar(ax, t, s["tauq"], Y, H, COL["queue"], label="Q", fontsize=7); t += s["tauq"]
        if s["is_K"] and s["y"] and s["tauc"] > EPS:
            _bar(ax, t, s["tauc"], Y, H, COL["charge"],
                 label=f"CHG\n{s['ea']:.0f}→{s['ed']:.0f}", fontsize=6.5); t += s["tauc"]
        if s["taub"] > EPS:
            lbl = "B45" if s["b45"] else ("B15" if s["b15"] else "B30")
            _bar(ax, t, s["taub"], Y, H, COL["brk"], label=lbl, fontsize=7,
                 text_color="#333"); t += s["taub"]
        if s["taur"] > EPS:
            _bar(ax, t, s["taur"], Y, H, COL["rest"],
                 label="RST-r1" if s["rho1"] else "RST-r2", fontsize=7)
        typ = "●C" if s["is_C"] else ("▲K" if s["is_K"] else ("O" if i==0 else "D"))
        ax.text(s["ta"], Y+H/2+0.06, f"{typ}{i}",
                ha="left", va="bottom", fontsize=6, color="#444", rotation=45, clip_on=True)
    _draw_vlines(ax, vlines)
    ax.set_yticks([])
    ax.set_xlim(-0.2, tend * 1.02)
    patches = [mpatches.Patch(color=v, label=k.replace("_","").title()) for k,v in COL.items()]
    patches += [mpatches.Patch(color="#D6EAF8", alpha=0.6, label="night 0-6h"),
                mpatches.Patch(color="#FEF9E7", alpha=0.6, label="day 6-20h"),
                mpatches.Patch(color="#E8DAEF", alpha=0.6, label="evening 20-24h")]
    ax.legend(handles=patches, loc="upper left", fontsize=7, ncol=5)

    # ============ Panel 2: SOC vs time ========================
    ax2 = axes[1]; ax2.set_title("Battery state of charge", fontsize=10)
    _shade_tod(ax2, 0, tend)
    tpts, spts = [], []
    for s in sol:
        ta, td, ea, ed = s["ta"], s["td"], s["ea"], s["ed"]
        tauq = s["tauq"] if s["is_K"] else 0
        tauc = s["tauc"] if s["is_K"] else 0
        tpts.append(ta); spts.append(ea)
        if td - ta > EPS:
            tcs = ta + tauq; tce = tcs + tauc
            if tauq > EPS: tpts.append(tcs); spts.append(ea)
            if tauc > EPS: tpts.append(tce); spts.append(ed)
            tpts.append(td); spts.append(ed)
    ax2.plot(tpts, spts, color=COL["drive"], lw=2, label="SOC", zorder=2)
    ax2.fill_between(tpts, spts, alpha=0.10, color=COL["drive"])
    for s in sol:
        if s["is_K"] and s["y"] and s["ed"]-s["ea"] > 0.5:
            ts = s["ta"]+s["tauq"]; te = ts+s["tauc"]
            ax2.annotate("", xy=(te, s["ed"]), xytext=(ts, s["ea"]),
                arrowprops=dict(arrowstyle="->", color=COL["charge"], lw=1.5), zorder=3)
            ax2.text((ts+te)/2, (s["ea"]+s["ed"])/2, f"+{s['ed']-s['ea']:.0f}",
                     ha="center", fontsize=7, color=COL["charge"])
    ax2.axhline(data["Emin"], color="red", ls=":", lw=1.2, label=f"E_min={data['Emin']} kWh")
    ax2.axhline(data["Ecap"], color="gray",ls=":", lw=1.2, label=f"E_cap={data['Ecap']} kWh")
    _draw_vlines(ax2, vlines)
    ax2.set_ylabel("kWh"); ax2.set_ylim(0, data["Ecap"]*1.15)
    ax2.legend(fontsize=8, ncol=3, loc="upper right")

    # ============ Panel 3: HoS counters =======================
    ax3 = axes[2]; ax3.set_title("HoS accumulators (at arrival)", fontsize=10)
    _shade_tod(ax3, 0, tend)
    cdt, cdv, sdt, sdv, swt, swv = [], [], [], [], [], []
    for s in sol:
        ta, td = s["ta"], s["td"]
        r_cd = s["b45"] or s["b30"] or s["rho1"] or s["rho2"]
        r_rho = s["rho1"] or s["rho2"]
        cdt.append(ta); cdv.append(s["cd"])
        sdt.append(ta); sdv.append(s["sd"])
        swt.append(ta); swv.append(s["sw"])
        if td-ta > EPS:
            cdt.append(td); cdv.append(0.0 if r_cd  else s["cd"])
            sdt.append(td); sdv.append(0.0 if r_rho else s["sd"])
            swt.append(td); swv.append(0.0 if r_rho else s["sw"])
    ax3.plot(cdt, cdv, "o-", color="#E74C3C", lw=1.5, ms=3, label="Consec. driving")
    ax3.plot(sdt, sdv, "s-", color="#3498DB", lw=1.5, ms=3, label="Shift driving")
    ax3.plot(swt, swv, "^-", color="#1ABC9C", lw=1.5, ms=3, label="Shift working")
    ax3.axhline(data["Tdrv_cons"], color="#E74C3C", ls=":", lw=1.2, alpha=0.7,
                label=f"max consec {data['Tdrv_cons']}h")
    ax3.axhline(data["Tdrv_sh1"],  color="#3498DB", ls=":", lw=1.2, alpha=0.7,
                label=f"max shift drv {data['Tdrv_sh1']}h")
    ax3.axhline(data["Twrk_sh"],   color="#1ABC9C", ls=":", lw=1.2, alpha=0.7,
                label=f"max shift wk {data['Twrk_sh']}h")
    _draw_vlines(ax3, vlines)
    ax3.set_ylabel("Hours"); ax3.legend(fontsize=7, ncol=3, loc="upper left")


    plt.tight_layout()
    fname = f"solution_{title}_{int(time.time())}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {fname}")
    plt.close()


# ============================================================
# FEASIBILITY CHECK + SCHEDULE PRINT
# ============================================================

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
        # time propagation: ta[i+1] == td[i] + D_nom[i]
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

INSTANCES = {
    "tiny":            instance_tiny,
    "break_forced":    instance_break_forced,
    "charging_needed": instance_charging_needed,
    "rest_forced":     instance_rest_forced,
    "3day":            instance_3day,
    "realistic":       instance_realistic,
}


def run_instance(name, tee=True, run=True):
    print(f"\n{'='*65}")
    data = INSTANCES[name]()
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
    random.seed(5)
    name = sys.argv[1] if len(sys.argv) > 1 else "realistic"
    tee  = True # "--tee" in sys.argv

    if name == "all":
        for n in INSTANCES:
            run_instance(n, tee=tee)
    elif name in INSTANCES:
        run_instance(name, tee=tee)
    else:
        print(f"Unknown instance '{name}'. Choose: {list(INSTANCES)}")