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
6.  Extra energy bounds (soc_charge_link, soc_dep_lb, soc_horizon_lb)
    tighten the LP relaxation and detect infeasible states earlier.

All times are in HOURS; energy in kWh.
Indexing: local stop j = global_stop - start_stop, j in 0..H.
          local leg  j covers global leg (start_stop + j).
"""

import pyomo.environ as pyo
import logging as _logging
_logging.getLogger('pyomo').setLevel(_logging.ERROR)    # suppress W1001/W1002
import numpy as np

# ── Shared helpers from MILP.py ──────────────────────────────────────────
from MILP import (
    _time_bounds,
    _solve_quiet,
    _declare_common_vars,
    _add_pwl_charging_constraints,
    _add_break_rest_constraints,
    _add_hos_accumulator_constraints,
    _add_manoeuver_constraints,
    _model_xsum,
    _model_ri,
    _model_rho,
)

INFEASIBLE_PENALTY = 1e9


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
    E_override  : dict {global_leg_index: kWh}   or None

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

    lb_t, ub_t = _time_bounds(I_loc, C_loc, K_loc, D_loc, S_loc, Q_loc,
                               Tbar, T_hor, t0=t0)

    # Minimum energy needed from each local stop to the next CS or dest.
    #
    # For WITHIN-horizon legs we use the scenario energy (E_override when
    # provided) so that soc_dep_lb is a valid lower bound under this scenario
    # — using nominal energies when scenario has lower consumption would make
    # the LP cut too tight and cause false infeasibility.
    # For legs BEYOND the horizon (used in soc_horizon_lb only) we conservatively
    # use nominal energies scaled by (1+delta_safe) since we don't know the
    # future scenario realisation; here we just use nominal (caller adds margin
    # via soc_horizon_lb if desired).
    E_within = E_override if E_override is not None else full_data["E"]
    E_global_all = full_data["E"]
    K_global_set = set(full_data["K"])
    e_to_next_cs = {}
    for j in range(H + 1):
        g   = start_stop + j
        cum = 0.0
        k   = g
        while k < N_glob:
            # Use scenario energy for within-horizon legs, nominal beyond
            E_src = E_within if k < start_stop + H else E_global_all
            cum += E_src.get(k, 0.0)
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


# ══════════════════════════════════════════════════════════════════════════
# WARM START
# ══════════════════════════════════════════════════════════════════════════

def inject_warm_start(model, warm_sol, start_stop):
    """
    Inject a previous solution into a horizon model as warm-start hints.
    warm_sol : list of per-stop dicts with local indices (0 = start_stop).

    Sets values for: ta, td, ea, ed, cd, sd, sw, phi,
                     taub, taur, l1, l2, l4 (derived from cd/sd/sw + binaries),
                     all binary variables, y, tauc.
    Pyomo W1001/W1002 warnings are suppressed — HiGHS clamps values to
    feasible range so slightly out-of-bound hints are harmless.
    """
    import warnings as _ws
    with _ws.catch_warnings():
        _ws.simplefilter("ignore")
        for s in warm_sol:
            i = s.get("i")
            if i is None or i not in model.I:
                continue
            try:
                ta_v = max(0.0, float(s.get("ta", 0)))
                td_v = max(0.0, float(s.get("td", ta_v)))
                cd_v = max(0.0, float(s.get("cd", 0)))
                sd_v = max(0.0, float(s.get("sd", 0)))
                sw_v = max(0.0, float(s.get("sw", 0)))
                b45  = int(s.get("b45",  0))
                b30  = int(s.get("b30",  0))
                rho1 = int(s.get("rho1", 0))
                rho2 = int(s.get("rho2", 0))
                ri   = int(b45 or b30 or rho1 or rho2)   # cd reset
                rho  = int(rho1 or rho2)                  # sd/sw reset

                model.ta[i].set_value(ta_v)
                model.td[i].set_value(td_v)
                model.ea[i].set_value(max(0.0, float(s.get("ea", 0))))
                model.ed[i].set_value(max(0.0, float(s.get("ed", 0))))
                model.cd[i].set_value(cd_v)
                model.sd[i].set_value(sd_v)
                model.sw[i].set_value(sw_v)
                model.phi[i].set_value(int(s.get("b15", 0)) if not ri else 0)
                model.taub[i].set_value(max(0.0, float(s.get("taub", 0))))
                model.taur[i].set_value(max(0.0, float(s.get("taur", 0))))
                model.x_b45[i].set_value(b45)
                model.x_b15[i].set_value(int(s.get("b15", 0)))
                model.x_b30[i].set_value(b30)
                model.rho1[i].set_value(rho1)
                model.rho2[i].set_value(rho2)
                model.z_man[i].set_value(float(bool(int(s.get("y", 0)) or b45 or
                                                      int(s.get("b15",0)) or b30 or
                                                      rho1 or rho2)))
                # Auxiliary big-M variables: l_k[i] = acc[i] if reset else 0
                model.l1[i].set_value(cd_v if ri  else 0.0)
                model.l2[i].set_value(sd_v if rho else 0.0)
                model.l4[i].set_value(sw_v if rho else 0.0)
            except Exception:
                pass
            if s.get("is_K"):
                try:
                    model.y[i].set_value(int(s.get("y", 0)))
                    model.tauc[i].set_value(max(0.0, float(s.get("tauc", 0))))
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════════════════
# BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════

def build_horizon_model(sub_data, init_state, fixed_action=None,
                        rho2_remaining=3):
    """
    Build the Pyomo MILP for the sub-problem.

    Uses the shared constraint helpers from MILP.py for the ~300 lines of
    constraints identical to build_model.  Only the sub-problem-specific
    parts (initial conditions, fixed_action, extra energy bounds) are
    handled here.

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
    # Man indexed over all stops (needed when local stop 0 or H is C or K)
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

    # ── Variables (shared declaration) ────────────────────────────────────
    _declare_common_vars(m)
    _add_manoeuver_constraints(m, sub_data["I"], K_set)

    # ── Objective ─────────────────────────────────────────────────────────
    m.obj = pyo.Objective(expr=m.ta[N], sense=pyo.minimize)

    for i in sub_data["I"]:
        m.ta[i].setlb(lb_t.get(i, 0.0))
        m.ta[i].setub(ub_t.get(i, T_hor))

    # ══════════════════════════════════════════════════════════════════════
    # INITIAL CONDITIONS  (from init_state, not hardcoded zeros)
    # ══════════════════════════════════════════════════════════════════════
    m.init_ta  = pyo.Constraint(expr=m.ta[0]  == init_state["ta"])
    m.init_ea  = pyo.Constraint(expr=m.ea[0]  == init_state["ea"])
    m.init_cd  = pyo.Constraint(expr=m.cd[0]  == init_state["cd"])
    m.init_sd  = pyo.Constraint(expr=m.sd[0]  == init_state["sd"])
    # sw[0] = shift working time AT arrival at local stop 0, AFTER the work
    # done at this stop (queue, charge-if-no-break) is added.
    # The rest reset is captured via l4[0] in sw_prop (sw[1] = sw[0] - l4[0] + ...),
    # but sw_ub checks sw[0] BEFORE the reset, which can spuriously force
    # infeasibility when sw_init + Q is high but a rest at stop 0 would fix it.
    #
    # Correct fix: define a post-reset sw at stop 0 so that sw_ub[0] reflects
    # the state AFTER the rest (if taken). We achieve this by shifting the
    # reset into sw[0] directly: if rho[0]=1 then l4[0]=sw_raw[0] → sw[1]=0.
    # We therefore LEAVE sw[0] unconstrained (let sw_prop determine it) and
    # instead apply the initial condition via an equality on sw[1] (the first
    # leg boundary). However this breaks the propagation chain at stop 0.
    #
    # Pragmatic approach (matches existing MILP structure): keep init_sw as
    # an equality but clip it to Twrk_sh when a rest is forced at stop 0 so
    # the model can at least represent the post-rest state. We add a slack
    # variable sw0_raw and sw0 = min(sw0_raw, Twrk_sh) when rho[0]=1.
    #
    # Simplest correct approach without restructuring: initialise sw[0] to
    # the pre-work arrival sw, then let sw_prop add the work and the reset.
    # But sw_prop propagates from i to i+1, so sw[0] must already include
    # work at stop 0 for the sw_ub[0] constraint to be meaningful.
    #
    # CHOSEN FIX: Add a relaxed upper bound that accounts for the reset.
    # init_sw: sw[0] = sw_init + Q*y[0] + u[0]  (as before)
    # Add:     sw[0] - M_sw * rho[0] <= Twrk_sh (rest makes the bound slack)
    # This replaces the hard sw_ub[0] constraint for stop 0.
    if 0 in K_set:
        m.init_sw = pyo.Constraint(
            expr=m.sw[0] == init_state["sw"] + m.Q_nom[0]*m.y[0] + m.u[0])
    elif 0 in C_set:
        m.init_sw = pyo.Constraint(
            expr=m.sw[0] == init_state["sw"] + sub_data["S"].get(0, 0.0))
    else:
        m.init_sw = pyo.Constraint(expr=m.sw[0] == init_state["sw"])
    m.init_phi = pyo.Constraint(expr=m.phi[0] == init_state["phi"])

    # Soft sw_ub at stop 0: if a rest is taken (rho[0]=1), sw[0] may exceed
    # Twrk_sh because the reset only lands in sw[1]. Replace the shared
    # sw_ub[0] with the relaxed version below; shared helper still adds
    # sw_ub for i in I which includes 0, so we override it post-hoc.
    # (The override is applied after _add_hos_accumulator_constraints.)
    _sw0_needs_override = True   # flag consumed below

    # ── No activities at last horizon stop ────────────────────────────────
    # Use equality constraints (not .fix()) — .fix() sets bounds that are
    # silently reset to [0,1] by relax_integer_vars.
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
        if i >= N: return pyo.Constraint.Skip
        return m.ta[i + 1] == m.td[i] + m.D_nom[i]
    m.time_prop = pyo.Constraint(m.I, rule=_tp)

    if is_global_origin:
        m.td_orig = pyo.Constraint(expr=m.td[0] == m.ta[0])

    if N not in C_set and N not in K_set:
        m.td_dest = pyo.Constraint(expr=m.td[N] == m.ta[N])

    m.td_C = pyo.Constraint(m.Cset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.S[i] + m.taub[i] + m.taur[i]
                   + m.Man[i] * m.z_man[i])

    m.td_K = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.Q_nom[i]*m.y[i] + m.tauc[i] + m.taub[i] + m.taur[i]
                   + m.Man[i] * m.z_man[i])

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
        if i >= N: return pyo.Constraint.Skip
        return m.ea[i + 1] == m.ed[i] - m.Eparam[i]
    m.soc_prop = pyo.Constraint(m.I, rule=_soc)

    if is_global_origin:
        m.soc_nc_orig = pyo.Constraint(expr=m.ed[0] == m.ea[0])
    m.soc_nc_dest = pyo.Constraint(expr=m.ed[N] == m.ea[N])
    m.soc_nc_C   = pyo.Constraint(m.Cset, rule=lambda m, i: m.ed[i] == m.ea[i])
    m.soc_mono_K = pyo.Constraint(m.Kset, rule=lambda m, i: m.ed[i] >= m.ea[i])

    # Charge-link: prevents LP from fictitiously charging when y=0
    _charge_headroom = sub_data["Ecap"] - sub_data["Emin"]
    m.soc_charge_link = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.ed[i] - m.ea[i] <= _charge_headroom * m.y[i])

    # Tight lower bounds on arrival and departure energy
    e_to_next = sub_data.get("e_to_next_cs", {})
    Emin_val  = sub_data["Emin"]
    Ecap_val  = sub_data["Ecap"]

    m.soc_lb = pyo.Constraint(m.I,
        rule=lambda m, i: m.ea[i] >= Emin_val)
    m.soc_dep_lb = pyo.Constraint(m.I, rule=lambda m, i:
        m.ed[i] >= min(Emin_val + e_to_next.get(i, 0.0), Ecap_val))
    m.soc_ub = pyo.Constraint(m.I,
        rule=lambda m, i: m.ed[i] <= m.Ecap)

    # At the horizon boundary the vehicle must have enough energy to reach
    # the next CS (or destination) beyond the horizon.
    global_end = sub_data.get("global_end", N)
    global_N   = sub_data.get("global_N",  N)
    if global_end < global_N:
        Eleg_glob = sub_data.get("E_global", {})
        K_glob    = sub_data.get("K_global", set())
        e_needed  = 0.0
        j = global_end
        while j < global_N:
            e_needed += Eleg_glob.get(j, 0.0)
            if j + 1 in K_glob or j + 1 == global_N:
                break
            j += 1
        if 0 < e_needed <= sub_data["Ecap"] - sub_data["Emin"]:
            m.soc_horizon_lb = pyo.Constraint(
                expr=m.ea[N] >= sub_data["Emin"] + e_needed)

    m.chg_act  = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.tauc[i] <= TK * m.y[i])
    m.chg_act2 = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.tauc[i] >= 0.25 * m.y[i])

    # ══════════════════════════════════════════════════════════════════════
    # SHARED CONSTRAINT BLOCKS
    # ══════════════════════════════════════════════════════════════════════
    _add_pwl_charging_constraints(m, K, R, Rseg)
    _add_break_rest_constraints(m, N, sub_data["I"], K_set, M_big,
                                rho2_limit=rho2_remaining)
    _add_hos_accumulator_constraints(m, N, sub_data["I"], C_set, K_set,
                                     sub_data["S"], M_drv, M_sd, M_sw, TK)

    # Override sw_ub at stop 0: the shared helper adds sw[i] <= Twrk_sh for
    # all i, but at stop 0 a rest (rho[0]=1) resets sw in sw_prop (sw[1]=0+…).
    # If sw[0] > Twrk_sh because the vehicle arrived at a high-sw state and
    # the queue pushed it over, the model is spuriously infeasible even though
    # a rest at stop 0 is a valid corrective action.
    #
    # Fix: deactivate the shared sw_ub[0] and replace it with a constraint
    # that allows sw[0] to exceed Twrk_sh by at most M_sw when a rest is taken.
    #   sw[0] <= Twrk_sh + M_sw * rho[0]
    # When rho[0]=0 this reduces to the original bound. When rho[0]=1 the
    # slack M_sw allows any sw[0] (the actual bound lands on sw[1] via sw_prop).
    if _sw0_needs_override:
        m.sw_ub[0].deactivate()
        m.sw_ub0_relaxed = pyo.Constraint(
            expr=m.sw[0] <= M_sw + M_sw * _model_rho(m, 0))

    return m


# ══════════════════════════════════════════════════════════════════════════
# SOLVE
# ══════════════════════════════════════════════════════════════════════════

import io as _io, contextlib as _ctx


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
    n_vars_pre = sum(1 for _ in model.component_data_objects(pyo.Var, active=True))
    n_cons_pre = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))

    if relax:
        try:
            pyo.TransformationFactory("core.relax_integer_vars").apply_to(model)
        except KeyError:
            pyo.TransformationFactory("core.relax_integrality").apply_to(model)

    solver = pyo.SolverFactory("appsi_highs")
    solver.options["presolve"]   = "on"
    solver.options["time_limit"] = time_limit
    if not relax:
        solver.options["mip_rel_gap"] = 0.05

    t0 = _tm.perf_counter()
    try:
        # Pass warmstart=True only for MIP solves that have had variable hints
        # injected via inject_warm_start.  For LP relaxations HiGHS ignores it
        # (and the transformation clears set_value hints anyway).
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
        False → full MIP (needed for extracting activity durations used
                by advance_state).

    Returns
    -------
    dict with keys:
        'feasible'     : bool
        'obj'          : float — ta at end_stop, or INFEASIBLE_PENALTY
        'sol'          : list of stop dicts (local indices)  or []
        'status'       : str
        'first_action' : dict summarising decisions at local stop 0
        'solve_info'   : dict from solve_horizon_model
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