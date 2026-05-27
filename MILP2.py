"""
MILP2.py — Horizon Sub-Problem MILP
=====================================
Solves the electric-truck scheduling problem over a sub-route
[start_stop, end_stop] given an initial vehicle state and an
optional fixed action at the first stop.

Designed to be called repeatedly by the look-ahead simulator
(simulation.py) with different scenarios and candidate actions.

Public interface
----------------
    result = solve_horizon(
        full_data,          # dict from MILP._make_data
        start_stop,         # global index of first stop in window
        end_stop,           # global index of last stop  in window
        init_state,         # dict: 'ta','ea','cd','sd','sw','phi'
        fixed_action=None,  # dict: 'y','break_type','rest_type'
        D_override=None,    # dict {global_leg: duration} (scenario)
        rho2_remaining=3,   # reduced-rest budget still available
        tee=False,
        time_limit=30,
    )
    →  dict:
        'feasible' : bool
        'obj'      : float  (arrival time at end_stop, or PENALTY)
        'sol'      : list of per-stop dicts (same schema as MILP.extract_solution)
        'status'   : str

Key modelling differences from MILP.build_model
------------------------------------------------
1.  Initial conditions come from init_state (not hardcoded to 0).
2.  fixed_action (if given) fixes the binary decisions at local stop 0.
3.  rho2_remaining caps the number of reduced rests in the sub-window.
4.  No activities allowed at the last horizon stop (binaries fixed 0).
5.  Local stop 0 is an intermediate route stop → breaks/rests are
    allowed there (unless it is the global route origin).

All times are in HOURS; energy in kWh.
Indexing: local stop j = global_stop - start_stop, j in 0..H.
          local leg  j covers global leg (start_stop + j).
"""

import pyomo.environ as pyo
import logging as _logging
_logging.getLogger('pyomo').setLevel(_logging.ERROR)    # suppress W1001/W1002
import numpy as np

# ── helpers re-used from MILP.py ─────────────────────────────────────────
from MILP import _time_bounds

INFEASIBLE_PENALTY = 1e9   # returned as obj when no feasible solution found


# (time-shifted bounds delegated to MILP._time_bounds via make_subproblem_data)


# ══════════════════════════════════════════════════════════════════════════
# SUB-PROBLEM DATA ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════

def make_subproblem_data(full_data, start_stop, end_stop, init_state,
                         D_override=None, E_override=None):
    """
    Slice the full route into a sub-problem over [start_stop, end_stop].

    Parameters
    ----------
    full_data   : dict from MILP._make_data
    start_stop  : int — global index of first stop  (local index 0)
    end_stop    : int — global index of last stop   (local index H)
    init_state  : dict — vehicle state AT ARRIVAL at start_stop
    D_override  : dict {global_leg_index: hours} or None
                  Scenario-specific travel times.
    E_override  : dict {global_leg_index: kWh} or None
                  Scenario-specific energy consumptions; derived from
                  speed-consumption coupling when km data is available.

    Returns
    -------
    sub_data : dict — same schema as full_data but for the sub-route.
    """
    N_glob = full_data["N"]
    C_glob = set(full_data["C"])
    K_glob = set(full_data["K"])

    assert 0 <= start_stop < end_stop <= N_glob, (
        f"Invalid horizon [{start_stop}, {end_stop}] for route of length {N_glob}")

    H = end_stop - start_stop          # number of legs = local N

    # ── local stop sets ───────────────────────────────────────────────────
    I_loc = list(range(H + 1))
    # Every intermediate global stop is either C or K; origin/dest are neither.
    C_loc = [j for j in range(H + 1) if (start_stop + j) in C_glob]
    K_loc = [j for j in range(H + 1) if (start_stop + j) in K_glob]

    # ── per-leg parameters ────────────────────────────────────────────────
    D_src = D_override if D_override is not None else full_data["D"]
    E_src = E_override if E_override is not None else full_data["E"]
    D_loc = {j: D_src.get(start_stop + j, 0.0) for j in range(H)}
    E_loc = {j: E_src.get(start_stop + j, 0.0) for j in range(H)}

    # ── per-stop parameters ───────────────────────────────────────────────
    S_loc  = {j: full_data["S"].get(start_stop + j, 0.0) for j in C_loc}
    Q_loc  = {j: full_data["Q"].get(start_stop + j, 0.0) for j in K_loc}
    # Manoeuver time at every stop, indexed by local stop (not leg)
    M_loc  = {}
    for j in range(H + 1):
        g = start_stop + j
        M_loc[j] = full_data["M"].get(g, 5.0 / 60) if g < N_glob else 0.0
    M_loc[H] = 0.0        # no manoeuver at last horizon stop
    if start_stop == 0:
        M_loc[0] = 0.0    # no manoeuver at route origin

    Wha_loc = {j: full_data["Wha"].get(start_stop + j, 0.0)
               for j in C_loc if (start_stop + j) in full_data.get("Wha", {})}
    Whf_loc = {j: full_data["Whf"].get(start_stop + j, 1e6)
               for j in C_loc if (start_stop + j) in full_data.get("Whf", {})}

    t0    = init_state["ta"]
    T_hor = full_data["T_hor"]
    R     = full_data["R"]
    Rseg  = full_data["Rseg"]
    Tbar  = full_data["Tbar"]

    lb_t, ub_t = _time_bounds(I_loc, C_loc, K_loc, D_loc, S_loc, Q_loc,
                               Tbar, T_hor, t0=t0)

    # For each local stop j, compute the minimum energy needed to reach the
    # next charging station (or the global destination) from j.
    # ea[j] >= Emin + e_to_next_CS[j] is a necessary feasibility condition
    # that tightens the LP relaxation without cutting any integer solution.
    E_global_all = full_data["E"]
    K_global_set = set(full_data["K"])
    e_to_next_cs = {}
    for j in range(H + 1):
        g = start_stop + j          # global stop index
        cum = 0.0
        k   = g
        while k < N_glob:
            cum += E_global_all.get(k, 0.0)
            if k + 1 in K_global_set or k + 1 == N_glob:
                break
            k += 1
        e_to_next_cs[j] = cum

    return dict(
        label=f"subproblem [{start_stop}→{end_stop}]",
        title=f"sub_{start_stop}_{end_stop}",
        # structure
        N=H, I=I_loc, C=C_loc, K=K_loc, R=R, Rseg=Rseg,
        # per-leg
        D=D_loc, E=E_loc,
        # per-stop
        S=S_loc, Q=Q_loc, M=M_loc,
        Wha=Wha_loc, Whf=Whf_loc,
        # battery
        E0=init_state["ea"],
        Ecap=full_data["Ecap"], Emin=full_data["Emin"],
        Ebar=full_data["Ebar"], Tbar=Tbar,
        T_hor=T_hor,
        # pre-computed bounds
        lb_t=lb_t, ub_t=ub_t,
        # HoS
        Tb45=full_data["Tb45"], Tb15=full_data["Tb15"], Tb30=full_data["Tb30"],
        Tr1=full_data["Tr1"],   Tr2=full_data["Tr2"],
        Tdrv_cons=full_data["Tdrv_cons"],
        Tdrv_sh1=full_data["Tdrv_sh1"],
        Tdrv_sh2=full_data.get("Tdrv_sh2", 10.0),
        Twrk_sh=full_data["Twrk_sh"],
        M_drv=full_data["M_drv"], M_sd=full_data["M_sd"],
        M_sw=full_data["M_sw"],   M_big=full_data["M_big"],
        # reference
        global_start=start_stop, global_end=end_stop, global_N=N_glob,
        # global energy/CS info for horizon-end energy bound
        E_global=full_data["E"],
        K_global=set(full_data["K"]),
        e_to_next_cs=e_to_next_cs,
    )


# ══════════════════════════════════════════════════════════════════════════
# BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════


def inject_warm_start(model, warm_sol, start_stop):
    """
    Inject a previous solution into a horizon model as warm-start hints.
    warm_sol : list of per-stop dicts with local indices (0 = start_stop).
    Pyomo W1001/W1002 warnings are suppressed — HiGHS clamps values to
    feasible range so slightly out-of-bound hints are harmless.
    """
    import pyomo.environ as pyo, warnings as _ws
    with _ws.catch_warnings():
        _ws.simplefilter("ignore")
        for s in warm_sol:
            i = s.get("i")
            if i is None or i not in model.I:
                continue
            try:
                model.ta[i].set_value(max(0.0, float(s.get("ta", 0))))
                model.ea[i].set_value(max(0.0, float(s.get("ea", 0))))
                model.ed[i].set_value(max(0.0, float(s.get("ed", 0))))
                model.cd[i].set_value(max(0.0, float(s.get("cd", 0))))
                model.sd[i].set_value(max(0.0, float(s.get("sd", 0))))
                model.sw[i].set_value(max(0.0, float(s.get("sw", 0))))
                model.taub[i].set_value(max(0.0, float(s.get("taub", 0))))
                model.taur[i].set_value(max(0.0, float(s.get("taur", 0))))
                model.x_b45[i].set_value(int(s.get("b45", 0)))
                model.x_b15[i].set_value(int(s.get("b15", 0)))
                model.x_b30[i].set_value(int(s.get("b30", 0)))
                model.rho1[i].set_value(int(s.get("rho1", 0)))
                model.rho2[i].set_value(int(s.get("rho2", 0)))
            except Exception:
                pass
            if s.get("is_K"):
                try:
                    model.y[i].set_value(int(s.get("y", 0)))
                    model.tauc[i].set_value(max(0.0, float(s.get("tauc", 0))))
                except Exception:
                    pass


def build_horizon_model(sub_data, init_state, fixed_action=None,
                        rho2_remaining=3):
    """
    Build the Pyomo MILP for the sub-problem.

    Structurally identical to MILP.build_model with the differences
    listed in the module docstring.

    Parameters
    ----------
    sub_data        : dict from make_subproblem_data
    init_state      : dict — 'ta','ea','cd','sd','sw','phi'
    fixed_action    : dict or None
        Optional keys:
          'y'          : 0 or 1  (only meaningful when local stop 0 is CS)
          'break_type' : None | 'b45' | 'b15' | 'b30'
          'rest_type'  : None | 'r1'  | 'r2'
    rho2_remaining  : int 0–3 — reduced-rest budget left for this window
    """
    m = pyo.ConcreteModel()

    N      = sub_data["N"]
    C      = sub_data["C"]
    K      = sub_data["K"]
    R      = sub_data["R"]
    Rseg   = sub_data["Rseg"]
    TK     = sub_data["Tbar"][max(R)]
    M_drv  = sub_data["M_drv"]
    M_sd   = sub_data["M_sd"]
    M_sw   = sub_data["M_sw"]
    M_big  = sub_data["M_big"]
    T_hor  = sub_data["T_hor"]
    lb_t   = sub_data["lb_t"]
    ub_t   = sub_data["ub_t"]
    Q_nom  = sub_data["Q"]

    C_set  = set(C)
    K_set  = set(K)
    is_global_origin = (sub_data.get("global_start", 1) == 0)

    # ── Sets ──────────────────────────────────────────────────────────────
    m.I     = pyo.Set(initialize=sub_data["I"], ordered=True)
    m.Cset  = pyo.Set(initialize=C)
    m.Kset  = pyo.Set(initialize=K)
    m.Rset  = pyo.Set(initialize=R,    ordered=True)
    m.RsegS = pyo.Set(initialize=Rseg, ordered=True)
    m.Legs  = pyo.Set(initialize=list(range(N)), ordered=True)



    # ── Parameters ────────────────────────────────────────────────────────
    m.D_nom  = pyo.Param(m.Legs, initialize=sub_data["D"])
    m.Q_nom  = pyo.Param(m.Kset, initialize=sub_data["Q"],  default=0)
    # Manoeuver time indexed over ALL stops (needed for td_C and td_K at stop 0 or H)
    m.Man    = pyo.Param(m.I,    initialize=sub_data["M"],  default=0)
    m.S      = pyo.Param(m.Cset, initialize=sub_data["S"],  default=0)
    m.Eparam = pyo.Param(m.Legs, initialize=sub_data["E"])
    m.Ecap   = pyo.Param(initialize=sub_data["Ecap"])
    m.Emin   = pyo.Param(initialize=sub_data["Emin"])
    m.Ebar   = pyo.Param(m.Rset, initialize=sub_data["Ebar"])
    m.Tbar   = pyo.Param(m.Rset, initialize=sub_data["Tbar"])
    m.Wha    = pyo.Param(m.Cset, initialize=sub_data.get("Wha", {}), default=0)
    m.Whf    = pyo.Param(m.Cset, initialize=sub_data.get("Whf", {}), default=1e6)
    m.Tb45   = pyo.Param(initialize=sub_data["Tb45"])
    m.Tb15   = pyo.Param(initialize=sub_data["Tb15"])
    m.Tb30   = pyo.Param(initialize=sub_data["Tb30"])
    m.Tr1    = pyo.Param(initialize=sub_data["Tr1"])
    m.Tr2    = pyo.Param(initialize=sub_data["Tr2"])
    m.Tdrv_cons = pyo.Param(initialize=sub_data["Tdrv_cons"])
    m.Tdrv_sh1  = pyo.Param(initialize=sub_data["Tdrv_sh1"])
    m.Twrk_sh   = pyo.Param(initialize=sub_data["Twrk_sh"])

    # ── State variables ───────────────────────────────────────────────────
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



    # ── Objective ─────────────────────────────────────────────────────────
    m.obj = pyo.Objective(expr=m.ta[N], sense=pyo.minimize)

    # Tighten variable bounds
    for i in sub_data["I"]:
        m.ta[i].setlb(lb_t.get(i, 0.0))
        m.ta[i].setub(ub_t.get(i, T_hor))

    # ══════════════════════════════════════════════════════════════════════
    # INITIAL CONDITIONS
    # ══════════════════════════════════════════════════════════════════════
    m.init_ta  = pyo.Constraint(expr=m.ta[0]  == init_state["ta"])
    m.init_ea  = pyo.Constraint(expr=m.ea[0]  == init_state["ea"])
    m.init_cd  = pyo.Constraint(expr=m.cd[0]  == init_state["cd"])
    m.init_sd  = pyo.Constraint(expr=m.sd[0]  == init_state["sd"])
    # sw[0] = accumulated work at arrival at local stop 0, EXCLUDING work
    # done at stop 0 itself.  We add it here as a model variable so that
    # tauc[0] (unknown before solve) is accounted for exactly.
    if 0 in K_set:
        m.init_sw = pyo.Constraint(
            expr=m.sw[0] == init_state["sw"] + m.Q_nom[0]*m.y[0] + m.u[0])
    elif 0 in C_set:
        m.init_sw = pyo.Constraint(
            expr=m.sw[0] == init_state["sw"] + sub_data["S"].get(0, 0.0))
    else:
        m.init_sw = pyo.Constraint(expr=m.sw[0] == init_state["sw"])
    m.init_phi = pyo.Constraint(expr=m.phi[0] == init_state["phi"])

    # ── No activities at last horizon stop ────────────────────────────────
    # Use equality constraints instead of .fix() so they survive LP relaxation.
    # (.fix() sets variable bounds; relax_integer_vars resets those bounds.)
    m.fix_b45_N  = pyo.Constraint(expr=m.x_b45[N] == 0)
    m.fix_b15_N  = pyo.Constraint(expr=m.x_b15[N] == 0)
    m.fix_b30_N  = pyo.Constraint(expr=m.x_b30[N] == 0)
    m.fix_rho1_N = pyo.Constraint(expr=m.rho1[N]  == 0)
    m.fix_rho2_N = pyo.Constraint(expr=m.rho2[N]  == 0)
    m.fix_taub_N = pyo.Constraint(expr=m.taub[N]  == 0)
    m.fix_taur_N = pyo.Constraint(expr=m.taur[N]  == 0)

    # ── No activities at global route origin ──────────────────────────────
    if is_global_origin:
        m.fix_b45_0  = pyo.Constraint(expr=m.x_b45[0] == 0)
        m.fix_b15_0  = pyo.Constraint(expr=m.x_b15[0] == 0)
        m.fix_b30_0  = pyo.Constraint(expr=m.x_b30[0] == 0)
        m.fix_rho1_0 = pyo.Constraint(expr=m.rho1[0]  == 0)
        m.fix_rho2_0 = pyo.Constraint(expr=m.rho2[0]  == 0)
        m.fix_taub_0 = pyo.Constraint(expr=m.taub[0]  == 0)
        m.fix_taur_0 = pyo.Constraint(expr=m.taur[0]  == 0)

    # ══════════════════════════════════════════════════════════════════════
    # APPLY FIXED ACTION AT LOCAL STOP 0
    # ══════════════════════════════════════════════════════════════════════
    if fixed_action is not None and not is_global_origin:
        brk = fixed_action.get("break_type", None)
        rst = fixed_action.get("rest_type",  None)

        # Use equality constraints instead of .fix() — .fix() sets bounds,
        # which are silently reset to [0,1] by relax_integer_vars.
        # Equality constraints are linear and survive LP transformation.
        if 0 in K_set and "y" in fixed_action:
            m.fix_y0 = pyo.Constraint(expr=m.y[0] == int(fixed_action["y"]))

        m.fix_b45_act = pyo.Constraint(expr=m.x_b45[0] == (1 if brk == "b45" else 0))
        m.fix_b15_act = pyo.Constraint(expr=m.x_b15[0] == (1 if brk == "b15" else 0))
        m.fix_b30_act = pyo.Constraint(expr=m.x_b30[0] == (1 if brk == "b30" else 0))
        m.fix_r1_act  = pyo.Constraint(expr=m.rho1[0]  == (1 if rst == "r1"  else 0))
        m.fix_r2_act  = pyo.Constraint(expr=m.rho2[0]  == (1 if rst == "r2"  else 0))

    # ══════════════════════════════════════════════════════════════════════
    # TIME PROPAGATION
    # ══════════════════════════════════════════════════════════════════════
    def _tp(m, i):
        if i >= N:
            return pyo.Constraint.Skip
        return m.ta[i + 1] == m.td[i] + m.D_nom[i]
    m.time_prop = pyo.Constraint(m.I, rule=_tp)

    # Route origin: no activities → td = ta
    if is_global_origin:
        m.td_orig = pyo.Constraint(expr=m.td[0] == m.ta[0])

    # Last horizon stop: no activities → td = ta
    # (also consistent with all binaries fixed to 0 there)
    if N not in C_set and N not in K_set:
        m.td_dest = pyo.Constraint(expr=m.td[N] == m.ta[N])
    # If N is C or K, the td_C / td_K constraints handle it (with all
    # binaries = 0, they reduce to td[N] = ta[N] + S[N] or ta[N], resp.)

    def _xsum(m, i):
        return m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]

    m.td_C = pyo.Constraint(m.Cset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.S[i] + m.taub[i] + m.taur[i]
                   + m.Man[i] * _xsum(m, i))

    m.td_K = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.Q_nom[i]*m.y[i] + m.tauc[i] + m.taub[i] + m.taur[i]
                   + m.Man[i] * _xsum(m, i))

    # Hard time windows (only those within the horizon)
    if sub_data.get("Wha") or sub_data.get("Whf"):
        wha = sub_data.get("Wha", {})
        whf = sub_data.get("Whf", {})
        if wha or whf:
            m.tw_hard = pyo.Constraint(m.Cset, rule=lambda m, i:
                pyo.inequality(m.Wha[i], m.ta[i], m.Whf[i]))

    # ══════════════════════════════════════════════════════════════════════
    # BATTERY SOC
    # ══════════════════════════════════════════════════════════════════════
    def _soc(m, i):
        if i >= N:
            return pyo.Constraint.Skip
        return m.ea[i + 1] == m.ed[i] - m.Eparam[i]
    m.soc_prop = pyo.Constraint(m.I, rule=_soc)

    # No charging at global route origin
    if is_global_origin:
        m.soc_nc_orig = pyo.Constraint(expr=m.ed[0] == m.ea[0])

    # No charging at last horizon stop
    m.soc_nc_dest = pyo.Constraint(expr=m.ed[N] == m.ea[N])

    m.soc_nc_C   = pyo.Constraint(m.Cset, rule=lambda m, i: m.ed[i] == m.ea[i])
    m.soc_mono_K = pyo.Constraint(m.Kset, rule=lambda m, i: m.ed[i] >= m.ea[i])

    # Direct charge-link constraint: ed[i] - ea[i] <= (Ecap-Emin)*y[i].
    # When y[i]=0 this forces ed[i] = ea[i] without going through the PWL,
    # preventing the LP relaxation from setting ed > ea via fractional lam_d
    # weights (LP PWL relaxation is not tight: it can achieve tauc=0 while
    # ed > ea by spreading weights across breakpoints).
    _charge_headroom = sub_data["Ecap"] - sub_data["Emin"]
    m.soc_charge_link = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.ed[i] - m.ea[i] <= _charge_headroom * m.y[i])
    # Tight energy lower bounds on DEPARTURE energy ed[i]:
    #   ed[i] >= Emin + e_to_next_CS[i]
    # where e_to_next_CS[i] = energy to travel from stop i to the next CS.
    # This is a necessary feasibility condition.  Applying it to ed (not ea)
    # means CS stops can charge to meet it, while non-CS stops (ed=ea) must
    # already have sufficient energy on arrival — correctly marking states
    # infeasible when the vehicle has run too low before a CS.
    # This also tightens the LP relaxation: fractional y at future CS stops
    # must still provide enough departing energy.
    e_to_next = sub_data.get("e_to_next_cs", {})
    Emin_val   = sub_data["Emin"]
    Ecap_val   = sub_data["Ecap"]
    def _soc_lb(m, i):
        return m.ea[i] >= Emin_val   # basic arrival bound
    def _soc_dep_lb(m, i):
        lb = Emin_val + e_to_next.get(i, 0.0)
        return m.ed[i] >= min(lb, Ecap_val)
    m.soc_lb     = pyo.Constraint(m.I, rule=_soc_lb)
    m.soc_dep_lb = pyo.Constraint(m.I, rule=_soc_dep_lb)
    m.soc_ub     = pyo.Constraint(m.I, rule=lambda m, i: m.ed[i] <= m.Ecap)

    # Tighter lower bound on ea at the last horizon stop:
    # if N_h < global N, the vehicle must have enough energy to reach the
    # next charging station (or the global destination) from stop N_h.
    # Otherwise, MILP2 can set ea[N_h]=Emin and the vehicle immediately runs
    # out of battery on the very next leg after the horizon.
    global_end = sub_data.get("global_end", N)
    global_N   = sub_data.get("global_N",  N)
    if global_end < global_N:
        # Sum E[j] for legs global_end, global_end+1, ... until next CS or dest
        Eleg_glob = sub_data.get("E_global", {})
        K_glob    = sub_data.get("K_global", set())
        e_needed  = 0.0
        j = global_end
        while j < global_N:
            e_needed += Eleg_glob.get(j, 0.0)
            if j + 1 in K_glob or j + 1 == global_N:
                break
            j += 1
        if e_needed > 0:
            horizon_end_lb = sub_data["Emin"] + e_needed
            if horizon_end_lb <= sub_data["Ecap"]:
                m.soc_horizon_lb = pyo.Constraint(
                    expr=m.ea[N] >= horizon_end_lb)
    m.chg_act    = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.tauc[i] <= TK * m.y[i])
    m.chg_act2   = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.tauc[i] >= 0.25 * m.y[i])   # at least 15 min if charging

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

    R_list = sorted(R)
    K_max  = max(Rseg)
    mid    = [(i, k) for i in K for k in Rseg[:-1]]
    m.sos2_lo_a  = pyo.Constraint(m.Kset, rule=lambda m,i: m.lam_a[i,R_list[0]] <= m.mu_a[i,R_list[1]])
    m.sos2_hi_a  = pyo.Constraint(m.Kset, rule=lambda m,i: m.lam_a[i,R_list[-1]] <= m.mu_a[i,K_max])
    m.sos2_lo_d  = pyo.Constraint(m.Kset, rule=lambda m,i: m.lam_d[i,R_list[0]] <= m.mu_d[i,R_list[1]])
    m.sos2_hi_d  = pyo.Constraint(m.Kset, rule=lambda m,i: m.lam_d[i,R_list[-1]] <= m.mu_d[i,K_max])
    m.sos2_mid_a = pyo.Constraint(mid, rule=lambda m,i,k: m.lam_a[i,k] <= m.mu_a[i,k]+m.mu_a[i,k+1])
    m.sos2_mid_d = pyo.Constraint(mid, rule=lambda m,i,k: m.lam_d[i,k] <= m.mu_d[i,k]+m.mu_d[i,k+1])

    # ══════════════════════════════════════════════════════════════════════
    # BREAKS AND RESTS
    # ══════════════════════════════════════════════════════════════════════
    non_K = [i for i in sub_data["I"] if i not in K_set]
    m.qb_K    = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.taub_hat[i] == m.taub[i] + m.tauc[i])
    m.qb_nonK = pyo.Constraint(non_K, rule=lambda m, i:
        m.taub_hat[i] == m.taub[i])

    m.one_brk = pyo.Constraint(m.I, rule=lambda m, i:
        m.x_b45[i] + m.x_b15[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i] <= 1)
    m.brk45   = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb45 * m.x_b45[i])
    m.brk15   = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb15 * m.x_b15[i])
    m.brk30   = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb30 * m.x_b30[i])
    m.brk_ub  = pyo.Constraint(m.I, rule=lambda m, i:
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

    m.rst1    = pyo.Constraint(m.I, rule=lambda m, i: m.taur[i] >= m.Tr1 * m.rho1[i])
    m.rst2    = pyo.Constraint(m.I, rule=lambda m, i: m.taur[i] >= m.Tr2 * m.rho2[i])
    m.rst_ub  = pyo.Constraint(m.I, rule=lambda m, i:
        m.taur[i] <= M_big * (m.rho1[i] + m.rho2[i]))
    # KEY CHANGE: use rho2_remaining instead of the global limit of 3
    m.rst_lim = pyo.Constraint(
        expr=sum(m.rho2[i] for i in sub_data["I"]) <= rho2_remaining)

    # ══════════════════════════════════════════════════════════════════════
    # HoS ACCUMULATORS  (consecutive driving, shift driving, shift working)
    # ══════════════════════════════════════════════════════════════════════
    def _ri(m, i):  return m.x_b45[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]
    def _rho(m, i): return m.rho1[i] + m.rho2[i]

    # --- Consecutive driving ---
    m.l1u1 = pyo.Constraint(m.I, rule=lambda m, i: m.l1[i] <= M_drv * _ri(m, i))
    m.l1u2 = pyo.Constraint(m.I, rule=lambda m, i: m.l1[i] <= m.cd[i])
    m.l1lb  = pyo.Constraint(m.I, rule=lambda m, i:
        m.l1[i] >= m.cd[i] - M_drv * (1 - _ri(m, i)))
    def _cd(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.cd[i + 1] == m.cd[i] + m.D_nom[i] - m.l1[i]
    m.cd_prop = pyo.Constraint(m.I, rule=_cd)
    m.cd_ub   = pyo.Constraint(m.I, rule=lambda m, i: m.cd[i] <= m.Tdrv_cons)

    # --- Shift driving ---
    m.l2u1 = pyo.Constraint(m.I, rule=lambda m, i: m.l2[i] <= M_sd * _rho(m, i))
    m.l2u2 = pyo.Constraint(m.I, rule=lambda m, i: m.l2[i] <= m.sd[i])
    m.l2lb  = pyo.Constraint(m.I, rule=lambda m, i:
        m.l2[i] >= m.sd[i] - M_sd * (1 - _rho(m, i)))
    def _sd(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.sd[i + 1] == m.sd[i] + m.D_nom[i] - m.l2[i]
    m.sd_prop = pyo.Constraint(m.I, rule=_sd)
    m.sd_ub   = pyo.Constraint(m.I, rule=lambda m, i: m.sd[i] <= m.Tdrv_sh1)

    # --- Shift working (charging not counted as work during break/rest) ---
    m.u_ub1 = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.u[i] <= TK * (1 - m.x_b45[i] - m.x_b15[i]
                        - m.x_b30[i] - m.rho1[i] - m.rho2[i]))
    m.u_ub2 = pyo.Constraint(m.Kset, rule=lambda m, i: m.u[i] <= m.tauc[i])
    m.u_lb  = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.u[i] >= m.tauc[i] - TK * (m.x_b45[i] + m.x_b15[i]
                                     + m.x_b30[i] + m.rho1[i] + m.rho2[i]))

    m.l4u1 = pyo.Constraint(m.I, rule=lambda m, i: m.l4[i] <= M_sw * _rho(m, i))
    m.l4u2 = pyo.Constraint(m.I, rule=lambda m, i: m.l4[i] <= m.sw[i])
    m.l4lb  = pyo.Constraint(m.I, rule=lambda m, i:
        m.l4[i] >= m.sw[i] - M_sw * (1 - _rho(m, i)))

    def _sw(m, i):
        if i >= N: return pyo.Constraint.Skip
        man_i = m.Man[i] * _xsum(m, i)
        ip1   = i + 1
        if ip1 in K_set:
            work_next = m.Q_nom[ip1] * m.y[ip1] + m.u[ip1]
        elif ip1 in C_set:
            work_next = sub_data["S"].get(ip1, 0)
        else:
            work_next = 0
        return m.sw[i + 1] == m.sw[i] - m.l4[i] + man_i + m.D_nom[i] + work_next
    m.sw_prop = pyo.Constraint(m.I, rule=_sw)
    m.sw_ub   = pyo.Constraint(m.I, rule=lambda m, i: m.sw[i] <= m.Twrk_sh)

    return m


# ══════════════════════════════════════════════════════════════════════════
# SOLVE
# ══════════════════════════════════════════════════════════════════════════

import io as _io, contextlib as _ctx


def _solve_quiet(solver, model, tee):
    """
    Run solver.solve(), suppressing all stdout/stderr output when tee=False.
    This silences Pyomo's 'Loading a feasible but suboptimal solution' messages
    that appear on every time-limit hit in look-ahead calls.
    """
    if tee:
        return solver.solve(model, tee=True)
    _sink = _io.StringIO()
    with _ctx.redirect_stdout(_sink), _ctx.redirect_stderr(_sink):
        return solver.solve(model, tee=False)


def solve_horizon_model(model, time_limit=8, tee=False, relax=True,
                        had_warm=False):
    """
    Solve the horizon model with HiGHS.

    Returns (results, status_str, solve_info) where solve_info is a dict:
        wall_s   : float — solver wall-clock time (s)
        obj      : float or None — objective value if feasible
        n_vars   : int — number of variables in the model
        n_cons   : int — number of constraints
        had_warm : bool — True if warm_start was injected before this call
        relax    : bool — whether LP relaxation was used
        status   : str

    had_warm must be passed explicitly because the LP-relaxation transformation
    clears set_value() hints, making post-transform counting unreliable.
    """
    import time as _tm
    # Count model size BEFORE relaxation transform (more accurate)
    n_vars_pre = sum(1 for _ in model.component_data_objects(pyo.Var, active=True))
    n_cons_pre = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))

    if relax:
        try:
            pyo.TransformationFactory("core.relax_integer_vars").apply_to(model)
        except KeyError:
            pyo.TransformationFactory("core.relax_integrality").apply_to(model)

    solver = pyo.SolverFactory("appsi_highs")
    solver.options["presolve"]    = "on"
    solver.options["time_limit"]  = time_limit
    if not relax:
        solver.options["mip_rel_gap"] = 0.05

    t0 = _tm.perf_counter()
    try:
        results = _solve_quiet(solver, model, tee)
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

    solve_info = dict(
        wall_s   = wall_s,
        obj      = obj_val,
        n_vars   = n_vars_pre,
        n_cons   = n_cons_pre,
        had_warm = had_warm,
        relax    = relax,
        status   = status,
    )
    return results, status, solve_info


# ══════════════════════════════════════════════════════════════════════════
# EXTRACT SOLUTION
# ══════════════════════════════════════════════════════════════════════════

def extract_horizon_solution(model, sub_data):
    """
    Extract per-stop solution dicts from a solved horizon model.
    Schema matches MILP.extract_solution (local stop indices).
    """
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


# ══════════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════

def solve_horizon(full_data, start_stop, end_stop, init_state,
                  fixed_action=None, D_override=None, E_override=None,
                  rho2_remaining=3, tee=False, time_limit=30, relax=True,
                  warm_start=None):
    """
    End-to-end helper: build → solve → extract.

    Parameters
    ----------
    relax : bool (default True)
        True  → LP relaxation (fast, good enough for scenario comparison).
        False → full MIP (slower; needed when extracting activity durations
                for the nominal solution that drives advance_state).

    Returns
    -------
    dict with keys:
        'feasible'     : bool
        'obj'          : float — ta at end_stop, or INFEASIBLE_PENALTY
        'sol'          : list of stop dicts (local indices)  or []
        'status'       : str
        'first_action' : dict summarising decisions at local stop 0
    """
    sub_data = make_subproblem_data(full_data, start_stop, end_stop,
                                    init_state, D_override=D_override,
                                    E_override=E_override)
    model    = build_horizon_model(sub_data, init_state,
                                   fixed_action=fixed_action,
                                   rho2_remaining=rho2_remaining)
    _had_warm = bool(warm_start)
    if warm_start:
        inject_warm_start(model, warm_start, start_stop)
    _, status, solve_info = solve_horizon_model(model, time_limit=time_limit,
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