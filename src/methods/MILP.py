"""
MILP.py — Electric Truck Scheduling: Full-Route & Rolling-Horizon Models
=========================================================================
Single file for ALL Pyomo modelling code.  Organised into four parts:

  PART 1 — Shared helpers

  PART 2 — Full-route deterministic model
      build_model(data)       : build Pyomo model for complete route 0..N
      solve_model(model)      : solve with HiGHS (0.5% gap, 2h limit)
      extract_solution(model) : extract per-stop solution dicts
      run_instance(data)      : convenience wrapper: build→solve→report→save
      Entry point: python -m src.methods.MILP [instance_name]

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

"""

from __future__ import annotations

import contextlib as _ctx
import io as _io
import json
import os
import random
import math as _mi

import pyomo.environ as pyo

from src.instance_gen.instances import compute_time_bounds
from src.settings import BETA_TW, TRAVEL_TIME_CV_TARGET
from src.settings import apply_solver_threads as _apply_solver_threads
from src import paths as _paths

FIGURES_DIR        = _paths.figures()
SOLUTIONS_DIR      = _paths.solutions()
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

def _model_xbrk(m, i):
    """Break-only indicator at stop i (excludes rests)."""
    return m.x_b45[i] + m.x_b15[i] + m.x_b30[i]

def _model_ri(m, i):
    """Consecutive-driving reset indicator (b45, b30, or any rest)."""
    return m.x_b45[i] + m.x_b30[i] + m.rho1[i] + m.rho2[i]

def _model_rho(m, i):
    """Shift reset indicator (any rest: r1 or r2)."""
    return m.rho1[i] + m.rho2[i]


# ── Variable declaration ──────────────────────────────────────────────────────

def _drop_split_break(m):
    """
    No-split regime (data["allow_split"] = False, 8.3 sensitivity axis).

    Art. 7 second subparagraph only PERMITS the 15'+30' split, so forbidding it
    is a legal fleet policy.  Fixing x_b15 = x_b30 = 0 removes both blocks from
    the model: phi[0] = 0 plus phi[i+1] <= phi[i] + x_b15[i] then propagates
    phi ≡ 0, so the split-break state machine collapses on its own and only the
    unsplit 45' break survives.  (phi is deliberately NOT fixed here — the
    horizon model pins phi[0] to the simulator's carried state, and letting the
    constraint chain do the work keeps the two consistent.)

    Fixing rather than skipping the declarations keeps every constraint, warm
    start, and solution extractor downstream structurally identical; presolve
    removes the columns.  Must be called AFTER _declare_common_vars.
    """
    if getattr(m, "_allow_split", True):
        return
    for i in m.I:
        m.x_b15[i].fix(0)
        m.x_b30[i].fix(0)


def _declare_common_vars(m):
    """
    Declare all shared decision variables on model m.
    Precondition: m.I, m.Cset, m.Kset, m.Lset, m.Rset, m.RsegS must already exist.

    Model revision (paper rewrite, July 2026):
      M2 — g (credited charging) replaces the orphaned p variable.
      M3 — u deleted: the work contribution of charging is the linear
           expression tauc − g.
      M5 — h (elapsed shift spread) + l5 replace the 13 h cap on sw.
      M6 — z / q_ext implement the 10 h extended-driving allowance.
      TW1/TW2 (v3) — no idle waiting; service starts at arrival.  The
           out-of-window penalty is the FIXED binary delta (early = late =
           same cost); the v2 waiting (w) and lateness (ell) variables are
           deleted.
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

    # Benchmark-parity idle wait (PGLT "APO"): free idle time appended to the
    # departure at customer/layby stops.  Fixed to 0 unless data["allow_wait"]
    # (see _add_time_constraints), preserving the v3 no-idle-waiting default.
    m.w = pyo.Var(m.I, domain=pyo.NonNegativeReals)

    m.cd = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.sd = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.sw = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.l1 = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.l2 = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.l4 = pyo.Var(m.I, domain=pyo.NonNegativeReals)

    # M5 — shift spread (elapsed on-duty time since end of last daily rest)
    m.h  = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.l5 = pyo.Var(m.I, domain=pyo.NonNegativeReals)   # rho·(h+o) linearisation

    # M6 — extended-driving allowance flags
    m.z     = pyo.Var(m.I, domain=pyo.Binary)   # 1 = current shift extended (10 h)
    m.q_ext = pyo.Var(m.I, domain=pyo.Binary)   # 1 = extension consumed at this rest

    # TW2 — fixed binary out-of-window penalty: delta_i = 1 iff service at
    # customer i starts outside its window (early OR late — one indicator).
    m.delta = pyo.Var(m.Cset, domain=pyo.Binary)

    # M2 — charging time credited toward the break requirement (parallel break)
    m.g     = pyo.Var(m.Kset, domain=pyo.NonNegativeReals)
    m.v     = pyo.Var(m.Kset, domain=pyo.Binary)            # any activity at CS stop
    m.sigma = pyo.Var(m.Kset, domain=pyo.Binary)            # sequential mode at CS stop


# ── Parameters declaration ──────────────────────────────────────────────────────

def _declare_common_params(m, data):

    N     = data["N"]
    C     = data["C"]
    K     = data["K"]
    R     = data["R"]
    Rseg  = data["Rseg"]
    lb_t  = data["lb_t"]
    ub_t  = data["ub_t"]

    # TK is the 0–100% full-charge time T_R: the natural big-M for every
    # charging-related linearisation (M9: replaces any other charging big-M).
    m.TK    = pyo.Param(initialize=data["Tbar"][max(R)])
    m.M_drv = pyo.Param(initialize=data["M_drv"])
    m.M_sd  = pyo.Param(initialize=data["M_sd"])
    m.M_sw  = pyo.Param(initialize=data["M_sw"])
    m.M_h   = pyo.Param(initialize=data.get("M_h", data.get("Tspr2", 15.0)))
    m.M_big = pyo.Param(initialize=data["M_big"])


    # ── Sets ──────────────────────────────────────────────────────────────────
    m.I     = pyo.Set(initialize=data["I"], ordered=True)
    m.Cset  = pyo.Set(initialize=C)
    m.Kset  = pyo.Set(initialize=K)
    m.Lset  = pyo.Set(initialize=data.get("L", []))   # layby stops (M8)
    m.Rset  = pyo.Set(initialize=R,    ordered=True)
    m.RsegS = pyo.Set(initialize=Rseg, ordered=True)
    m.Legs  = pyo.Set(initialize=list(range(N)), ordered=True)

    # ── Parameters ────────────────────────────────────────────────────────────
    m.D_nom = pyo.Param(m.Legs, initialize=data["D"])
    m.Q_nom = pyo.Param(m.Kset, initialize=data["Q"],      default=0)
    m.Mstop = pyo.Param(m.Kset, initialize=data["M_stop"], default=0)
    m.Mseq  = pyo.Param(m.Kset, initialize=data["M_seq"],  default=0)
    m.S     = pyo.Param(m.Cset, initialize=data["S"],      default=0)
    m.Eparam= pyo.Param(m.Legs, initialize=data["E"])
    m.E0    = pyo.Param(initialize=data["E0"])
    m.Ecap  = pyo.Param(initialize=data["Ecap"])
    m.Emin  = pyo.Param(initialize=data["Emin"])
    m.Ebar  = pyo.Param(m.Rset, initialize=data["Ebar"])
    m.Tbar  = pyo.Param(m.Rset, initialize=data["Tbar"])
    m.Wha   = pyo.Param(m.Cset, initialize=data["Wha"], default=0)
    m.Whf   = pyo.Param(m.Cset, initialize=data["Whf"], default=1e6)
    m.Mlay  = pyo.Param(m.Lset, initialize=data.get("M_lay", {}), default=0)
    m.Tb45  = pyo.Param(initialize=data["Tb45"])
    m.Tb15  = pyo.Param(initialize=data["Tb15"])
    m.Tb30  = pyo.Param(initialize=data["Tb30"])
    # 8.3 no-split axis — plain attribute (not a Param): it gates variable
    # fixing in _drop_split_break, it never enters an expression.
    m._allow_split = bool(data.get("allow_split", True))
    m.Tr1   = pyo.Param(initialize=data["Tr1"])
    m.Tr2   = pyo.Param(initialize=data["Tr2"])
    m.Tdrv_cons = pyo.Param(initialize=data["Tdrv_cons"])
    m.Tdrv_sh1  = pyo.Param(initialize=data["Tdrv_sh1"])
    m.Tdrv_sh2  = pyo.Param(initialize=data.get("Tdrv_sh2", 10.0))
    m.Twrk_sh   = pyo.Param(initialize=data["Twrk_sh"])
    # M5 — shift spread limits (13 h regular / 15 h reduced rest)
    m.Tspr1 = pyo.Param(initialize=data.get("Tspr1", 13.0))
    m.Tspr2 = pyo.Param(initialize=data.get("Tspr2", 15.0))
    # M9 — weekly working-time cap (Directive 2002/15/EC)
    m.Twk60 = pyo.Param(initialize=data.get("Twk60", 60.0))
    # TW2 — fixed out-of-window service penalty (h-equivalent per missed window)
    m.beta  = pyo.Param(initialize=data.get("beta", BETA_TW))
    # C1 — horizon big-M H: a valid upper bound on any feasible arrival span.
    # There is NO arrival deadline; H is used ONLY as the big-M in the window
    # indicators (eqs. 5–6) and the rest bound (eq. 36).  Computed per instance
    # by instances.compute_horizon_bigM; fall back to the weekly horizon span.
    _t0    = data.get("t0", data.get("T_START", lb_t.get(0, 0.0)))
    _Hbig  = data.get("H")
    if _Hbig is None:
        _Hbig = data.get("T_hor", _t0 + 7 * 24) - _t0
    m.H    = pyo.Param(initialize=max(1e-3, float(_Hbig)))

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

    # M2 (R7)–(R12): charging time is credited toward the break requirement
    # only when a break is DECLARED (x=1) and runs in parallel with charging
    # (σ=0).  g_i linearises tauc·x·(1−σ); charging with no declared break
    # earns no break credit.  Replaces the orphaned p machinery (o.23–o.27).
    m.g_ub1 = pyo.Constraint(m.Kset, rule=lambda m, i: m.g[i] <= m.tauc[i])
    m.g_ub2 = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.g[i] <= m.TK * _model_xbrk(m, i))
    m.g_ub3 = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.g[i] <= m.TK * (1 - m.sigma[i]))
    m.g_lb  = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.g[i] >= m.tauc[i] - m.TK * (1 - _model_xbrk(m, i)) - m.TK * m.sigma[i])

    # (R11)–(R12): effective break duration
    m.qb_K    = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.taub_hat[i] == m.taub[i] + m.g[i])
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
    # Named tight big-M (Appendix B): a break lies within one shift spread
    m.brk_ub = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub[i] <= m.Tspr2 * (m.x_b45[i] + m.x_b15[i] + m.x_b30[i]))

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
    # Named big-M (Appendix B, eq. 36): a rest is bounded by the horizon H
    m.rst_ub = pyo.Constraint(m.I, rule=lambda m, i:
        m.taur[i] <= m.H * (m.rho1[i] + m.rho2[i]))
    m.rst_lim = pyo.Constraint(
        expr=sum(m.rho2[i] for i in I_list) <= rho2_limit)


def _add_hos_accumulator_constraints(m, N, I_list, C_set, K_set,
                                     S_dict, M_drv, M_sd, M_sw, TK,
                                     is_subproblem: bool = False,
                                     D_wc: dict = None,
                                     ext_budget: int = 2,
                                     M_lay_dict: dict = None):
    """
    Hours-of-Service accumulator propagation: cd, sd, sw + shift spread h.

    Model revision (paper rewrite):
      M3 — charging work contribution is the linear expression tauc − g
           (no auxiliary u variable).
      M5 — the 13 h cap on sw is replaced by the elapsed shift-spread h:
           (R22) o_i := td_i − ta_i − taur_i        (pre-rest on-duty dwell)
           (R23) h_{i+1} = h_i + o_i + D_i − l5_i,  l5 = ρ·(h+o) linearised
           (R24) h_i + o_i ≤ 13 + 2·ρ2_i + 15·(1−ρ_i)   (pre-rest spread cap)
           (R25) h_i ≤ 15                               (global spread ceiling)
      M6 — 10 h extended-driving allowance:
           (R16) sd_i ≤ 9 + 1·z_i
           (R17) z_{i+1} ≥ z_i − ρ_i                (flag persists within shift)
           (R18) q_i ≥ z_i + ρ_i − 1                (budget unit consumed at rest)
           (R19) Σ q_i + z_N ≤ ext_budget           (incl. unfinished final shift)

    Big-M linearisation: l_k[i] = accumulator_k[i] if reset at stop i, else 0.

    is_subproblem
    -------------
    In the full-route model stop 0 is always the origin (no work), so sw[0]=0
    and the sw_prop formula is self-consistent: sw[i] includes work done AT i
    (injected as work_next from the previous step).

    In a subproblem, stop 0 is the current vehicle position (a CS or customer
    stop). init_state["sw"] is the arrival value BEFORE any work at stop 0,
    so the standard formula leaves work at stop 0 out of every sw constraint.
    When is_subproblem=True we inject work_here (queue + working charge, or
    service time) into the sw[1] propagation step to close this gap.

    Additionally a direct upper-bound constraint is added:
        sw[0] + work_at_stop_0 <= Twrk_sh
    This catches the case where work at stop 0 itself pushes the driver over
    the shift-working limit (not visible from sw[0] alone).

    D_wc : dict or None
    -------------------
    If provided, maps each leg index i to a worst-case travel time (h) used
    in the cd, sd, and sw propagation constraints instead of m.D_nom[i].
    Used by the Bertsimas-Sim robust counterpart (RO.py): passing
    D_wc[i] = D_nom[i]*(1+delta) guarantees HoS feasibility under any
    realised delay within the uncertainty set.  The nominal m.D_nom[i] is
    still used in the time-propagation / objective (handled by the dual
    penalty in the RO objective).
    None (default) -> nominal behaviour, uses m.D_nom[i] throughout.
    """
    # Travel time used in HoS accumulator propagation:
    # nominal m.D_nom[i] by default; worst-case float D_wc[i] for RO.
    def _d(i):
        if D_wc is not None and i in D_wc:
            return D_wc[i]
        return m.D_nom[i]
    # --- Consecutive driving (reset by b45, b30, or any rest) ---
    m.l1u1 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l1[i] <= M_drv * _model_ri(m, i))
    m.l1u2 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l1[i] <= m.cd[i])
    m.l1lb  = pyo.Constraint(m.I,
        rule=lambda m, i: m.l1[i] >= m.cd[i] - M_drv * (1 - _model_ri(m, i)))

    def _cd(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.cd[i+1] == m.cd[i] + _d(i) - m.l1[i]
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
        return m.sd[i+1] == m.sd[i] + _d(i) - m.l2[i]
    m.sd_prop = pyo.Constraint(m.I, rule=_sd)

    # ── M6: extended-driving allowance (R16)–(R19) — replaces sd ≤ 9 h ────────
    m.sd_ub = pyo.Constraint(m.I, rule=lambda m, i:
        m.sd[i] <= m.Tdrv_sh1 + (m.Tdrv_sh2 - m.Tdrv_sh1) * m.z[i])

    def _z_persist(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.z[i+1] >= m.z[i] - _model_rho(m, i)
    m.z_persist = pyo.Constraint(m.I, rule=_z_persist)
    m.q_ext_lb  = pyo.Constraint(m.I, rule=lambda m, i:
        m.q_ext[i] >= m.z[i] + _model_rho(m, i) - 1)
    m.ext_budget = pyo.Constraint(
        expr=sum(m.q_ext[i] for i in I_list) + m.z[N] <= ext_budget)

    # ── M3: shift working (reset by any rest) ─────────────────────────────────
    # The work contribution of charging is the LINEAR expression tauc − g:
    # tauc when no parallel break covers it (g=0), zero when fully credited.
    # No auxiliary u variable is needed.
    m.l4u1 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l4[i] <= M_sw * _model_rho(m, i))
    m.l4u2 = pyo.Constraint(m.I,
        rule=lambda m, i: m.l4[i] <= m.sw[i])
    m.l4lb  = pyo.Constraint(m.I,
        rule=lambda m, i: m.l4[i] >= m.sw[i] - M_sw * (1 - _model_rho(m, i)))

    _M_lay = M_lay_dict or {}

    def _cs_work(m, j):
        """Working activities at CS stop j before any break/rest (R20)."""
        return (m.v[j]*m.Mstop[j] + m.Q_nom[j]*m.y[j]
                + (m.tauc[j] - m.g[j]) + m.sigma[j]*m.Mseq[j])

    def _work_at(m, j):
        """Working activities performed at stop j (any stop type)."""
        if j in K_set:
            return _cs_work(m, j)
        if j in C_set:
            return S_dict.get(j, 0.0)
        if j in _M_lay:   # layby: parking overhead counts as other work (M8)
            return _M_lay[j] * _model_xsum(m, j)
        return 0.0

    def _sw(m, i):
        if i >= N: return pyo.Constraint.Skip
        # ── Subproblem correction ─────────────────────────────────────────────
        # init_state["sw"] is the arrival value at local stop 0 (before any
        # work there). For i=0 there is no i-1 step, so inject work at stop 0
        # explicitly into sw[1].
        work_here = _work_at(m, 0) if (i == 0 and is_subproblem) else 0.0
        return (m.sw[i+1] == m.sw[i] - m.l4[i] + work_here + _d(i)
                + _work_at(m, i + 1))
    m.sw_prop = pyo.Constraint(m.I, rule=_sw)
    # M5: the old daily cap sw ≤ 13 h (o.62) had no legal basis and is
    # REPLACED by the shift-spread constraints below.  sw is retained for the
    # weekly cap, reporting, and the ex-post Directive verification (S3).

    # ── M5/SP2: shift spread h, eqs. (h_prop)–(h_term) ────────────────────────
    # o_i = td_i − ta_i − taur_i: on-duty dwell at stop i before the rest.
    # Single rest-last convention (v3): with no idle waiting, a rest at ANY
    # stop type — customer included, since breaks/rests follow service — is
    # the last activity, so pre-rest elapsed = h + o exactly and the
    # post-reset spread starts at the outgoing leg.  Uniform for all i ∈ I.
    def _o(m, i):
        return m.td[i] - m.ta[i] - m.taur[i]

    M_h = float(pyo.value(m.M_h))

    def _h_prop(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.h[i+1] == m.h[i] + _o(m, i) + _d(i) - m.l5[i]
    m.h_prop = pyo.Constraint(m.I, rule=_h_prop)

    # l5 = rho·(h + o) linearisation (Appendix)
    m.l5u1 = pyo.Constraint(m.I, rule=lambda m, i:
        m.l5[i] <= M_h * _model_rho(m, i))
    m.l5u2 = pyo.Constraint(m.I, rule=lambda m, i:
        m.l5[i] <= m.h[i] + _o(m, i))
    m.l5lb = pyo.Constraint(m.I, rule=lambda m, i:
        m.l5[i] >= m.h[i] + _o(m, i) - M_h * (1 - _model_rho(m, i)))

    # (R24): pre-rest spread cap — 13 h before a regular rest, 15 h before a
    # reduced rest; inactive (RHS ≥ 28) when no rest is taken at the stop.
    m.spread_prerest = pyo.Constraint(m.I, rule=lambda m, i:
        m.h[i] + _o(m, i) <= m.Tspr1
                             + (m.Tspr2 - m.Tspr1) * m.rho2[i]
                             + m.Tspr2 * (1 - _model_rho(m, i)))
    # (R25): global spread ceiling
    m.spread_ub = pyo.Constraint(m.I, rule=lambda m, i: m.h[i] <= m.Tspr2)


def _add_wtd_constraints(m, N, data):
    """
    Directive 2002/15/EC working-time breaks, enforced IN-MODEL (S3 was
    ex-post only).  Gated by data["wtd_rules"]; used by the R15-PGLT
    benchmark (pglt.py) to match the reference model's constraints C11-C16
    (Peña-Arenas/Garaix main.cpp):

      C11  — no more than 6 h of working (driving + other work) without a
             break: continuous-work accumulator cw, reset by ANY declared
             break or rest (all our break types are ≥ 15 min), capped so the
             work performed at a stop BEFORE its break still fits:
                 cw_i + work_at(i) ≤ Twrk_cons1 (6 h)
      C12  — if the working time of a shift exceeds 6 h, the cumulated
             MINIMUM break durations within the shift must reach 30 min.
      C15  — beyond 9 h of shift work, 45 min.  Break minimums accumulate in
             bt (reset at rests, like sw); indicators chi30/chi45 are forced
             by sw.  Enforced at every rest stop and at the route end.

    Deterministic semantics only (uses nominal D); not wired into the RO
    worst-case propagation.
    """
    Twc1 = float(data["Twrk_cons1"])            # 6 h continuous-work cap
    Twc2 = float(data["Twrk_cons2"])            # 9 h shift-work threshold
    M_sw = float(data["M_sw"])                  # ≥ any feasible sw (15 h)
    M_bt = float(data["Tspr2"])                 # ≥ any feasible bt
    C_set = set(data["C"])
    K_set = set(data["K"])
    _M_lay = data.get("M_lay", {}) or {}

    m.cw    = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.l6    = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.bt    = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.l7    = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.chi30 = pyo.Var(m.I, domain=pyo.Binary)
    m.chi45 = pyo.Var(m.I, domain=pyo.Binary)

    def _work_at(m, j):
        """Working activities at stop j (mirrors _add_hos_accumulator_constraints)."""
        if j in K_set:
            return (m.v[j]*m.Mstop[j] + m.Q_nom[j]*m.y[j]
                    + (m.tauc[j] - m.g[j]) + m.sigma[j]*m.Mseq[j])
        if j in C_set:
            return m.S[j]
        if j in _M_lay:
            return _M_lay[j] * _model_xsum(m, j)
        return 0.0

    def _brkmin(m, i):
        """Declared break time at stop i, counted at MINIMUM durations
        (their sumB uses BD[b]); a rest stop has no break (one_brk)."""
        return (m.Tb15 * m.x_b15[i] + m.Tb30 * m.x_b30[i]
                + m.Tb45 * m.x_b45[i])

    m.init_cw = pyo.Constraint(expr=m.cw[0] == 0)
    m.init_bt = pyo.Constraint(expr=m.bt[0] == 0)

    # ── C11: continuous work ≤ 6 h; reset by any break/rest ──────────────────
    m.cw_cap = pyo.Constraint(m.I, rule=lambda m, i:
        m.cw[i] + _work_at(m, i) <= Twc1)
    # l6 = xsum·(cw + work_at) linearisation (work precedes the break)
    m.l6u1 = pyo.Constraint(m.I, rule=lambda m, i:
        m.l6[i] <= Twc1 * _model_xsum(m, i))
    m.l6u2 = pyo.Constraint(m.I, rule=lambda m, i:
        m.l6[i] <= m.cw[i] + _work_at(m, i))
    m.l6lb = pyo.Constraint(m.I, rule=lambda m, i:
        m.l6[i] >= m.cw[i] + _work_at(m, i) - Twc1 * (1 - _model_xsum(m, i)))

    def _cw(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.cw[i+1] == m.cw[i] + _work_at(m, i) + m.D_nom[i] - m.l6[i]
    m.cw_prop = pyo.Constraint(m.I, rule=_cw)

    # ── break-minimum accumulator bt (reset by any rest, like sw) ────────────
    m.l7u1 = pyo.Constraint(m.I, rule=lambda m, i:
        m.l7[i] <= M_bt * _model_rho(m, i))
    m.l7u2 = pyo.Constraint(m.I, rule=lambda m, i:
        m.l7[i] <= m.bt[i])
    m.l7lb = pyo.Constraint(m.I, rule=lambda m, i:
        m.l7[i] >= m.bt[i] - M_bt * (1 - _model_rho(m, i)))

    def _bt(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.bt[i+1] == m.bt[i] + _brkmin(m, i) - m.l7[i]
    m.bt_prop = pyo.Constraint(m.I, rule=_bt)

    # ── C13/C16: shift-work threshold indicators ─────────────────────────────
    m.chi30_def = pyo.Constraint(m.I, rule=lambda m, i:
        m.sw[i] <= Twc1 + (M_sw - Twc1) * m.chi30[i])
    m.chi45_def = pyo.Constraint(m.I, rule=lambda m, i:
        m.sw[i] <= Twc2 + (M_sw - Twc2) * m.chi45[i])

    # ── C12/C15: cumulated breaks ≥ 30'/45' by the end of the shift ─────────
    # sw[i] includes the work performed AT i (before its rest); bt[i] counts
    # the breaks strictly before i — their sumB over [u, v[.
    m.wtd30 = pyo.Constraint(m.I, rule=lambda m, i:
        m.bt[i] >= m.Tb30 * (m.chi30[i] + _model_rho(m, i) - 1))
    m.wtd45 = pyo.Constraint(m.I, rule=lambda m, i:
        m.bt[i] >= m.Tb45 * (m.chi45[i] + _model_rho(m, i) - 1))
    # unfinished final shift at the route end
    m.wtd30_term = pyo.Constraint(expr=m.bt[N] >= m.Tb30 * m.chi30[N])
    m.wtd45_term = pyo.Constraint(expr=m.bt[N] >= m.Tb45 * m.chi45[N])


def _add_v_sigma_constraints(m, M_big):
    """
    Activity / sequential-mode indicators (M4, rewrite R13–R15).

    v_i  ∈ {0,1}: = 1 if any activity occurs at CS stop i (charging, break, or rest).
    σ_i  ∈ {0,1}: = 1 if sequential mode: charging completes before the
                  break/rest begins (truck vacates the charging bay, M_seq).

    (43) σ ≤ y           — sequential mode only possible when charging occurs.
    (44) σ ≥ y + ρ − 1   — charging co-located with a REST is forced
                            sequential (parallel charge-during-rest is legally
                            contested; the sequential treatment is conservative
                            and the charging time counts as work).
    (45) σ ≥ (b_i − τ_c)/T_b45 − (1 − y)   — C2/Q1: a DECLARED break the
                            charge does not fully cover forces sequential mode.
                            b_i = 0.75·x_b45 + 0.25·x_b15 + 0.5·x_b30 is the
                            declared break minimum; if the charge τ_c is
                            shorter, the positive shortfall (b_i−τ_c)/0.75 ∈
                            (0,1] rounds σ up to 1, and (g3) then sets g = 0 so
                            the full break sits sequentially after the charge.
                            The −(1−y) term deactivates it when no charge
                            occurs.  When the charge covers the break the RHS
                            is ≤ 0 and the min-time objective keeps σ = 0
                            (parallel) — so no "force parallel" constraint is
                            needed.
    extra σ ≤ x + ρ       — no sequential flag without a break/rest to
                            sequence against (tightening; avoids spurious
                            M_seq overhead on pure-charge stops).

    The old concurrent-coverage constraints (o.42)–(o.44) are DELETED: with
    taub_hat = g + taub (M2) and the minimum-duration constraints, a parallel
    break is automatically covered by charging and/or extra break time.
    """
    # v_i activity indicator
    m.v_lb_y  = pyo.Constraint(m.Kset, rule=lambda m, i: m.v[i] >= m.y[i])
    m.v_lb_xr = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.v[i] >= _model_xsum(m, i))
    m.v_ub    = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.v[i] <= m.y[i] + _model_xsum(m, i))

    # (43)–(44) + tightening
    m.sigma_ub_y   = pyo.Constraint(m.Kset, rule=lambda m, i: m.sigma[i] <= m.y[i])
    m.sigma_lb_r   = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.sigma[i] >= m.y[i] + m.rho1[i] + m.rho2[i] - 1)
    m.sigma_ub_xr  = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.sigma[i] <= _model_xsum(m, i))

    # (45) C2/Q1 — uncovered declared break forces sequential mode
    m.sigma_lb_brk = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.sigma[i] >= (m.Tb45 * m.x_b45[i] + m.Tb15 * m.x_b15[i]
                       + m.Tb30 * m.x_b30[i] - m.tauc[i]) / m.Tb45
                      - (1 - m.y[i]))


def _add_soc_constraints(m, N, data):
    def _soc(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.ea[i + 1] == m.ed[i] - m.Eparam[i]
    m.soc_prop   = pyo.Constraint(m.I,    rule=_soc)
    m.soc_nc_C   = pyo.Constraint(m.Cset, rule=lambda m, i: m.ed[i] == m.ea[i])
    # M8 — laybys cannot charge: departure SOC equals arrival SOC (else the
    # solver gets free energy at every layby and never needs a CS).
    m.soc_nc_L   = pyo.Constraint(m.Lset, rule=lambda m, i: m.ed[i] == m.ea[i])
    m.soc_mono_K = pyo.Constraint(m.Kset, rule=lambda m, i: m.ed[i] >= m.ea[i])
    m.soc_lb     = pyo.Constraint(m.I,    rule=lambda m, i: m.ea[i] >= m.Emin)
    m.soc_ub     = pyo.Constraint(m.I,    rule=lambda m, i: m.ed[i] <= m.Ecap)
    m.chg_act    = pyo.Constraint(m.Kset, rule=lambda m, i: m.tauc[i] <= m.TK * m.y[i])
    m.chg_act2   = pyo.Constraint(m.Kset, rule=lambda m, i: m.tauc[i] >= 0.25 * m.y[i])

    m.pwl_no_free_charge = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.ed[i] - m.ea[i] <= m.Ecap * m.y[i])

def _add_time_constraints(m, N, data=None):
    """
    Time propagation, departures, and service windows (v3, eqs. (2)–(7)).

    TW1 — no idle waiting: service starts at arrival, so the customer
    departure is the additive eq. (depart_C): td = ta + S + taub + taur,
    with any break/rest taken AFTER service (rest-last at every stop type).

    TW2 — fixed binary penalty: delta_i = 1 whenever service starts outside
    the window (early OR late — a single indicator), eqs. (5)/(6):
        ta_i >= Wha_i − H·delta_i,   ta_i <= Whf_i + H·delta_i.
    data["hard_tw"]=True fixes delta = 0 (hard-window sensitivity).

    C1 — there is NO arrival deadline: the constant H in (5)/(6) is a valid
    big-M (any feasible arrival satisfies ta_N ≤ t0 + H by construction), so
    the indicators only DETECT an out-of-window arrival, never forbid a late
    one.

    TW6 (documented, not implemented): a physically-explicit variant would
    extend the service time by ΔS when out of window (S_i + ΔS·delta_i in
    td), delaying downstream arrivals; the objective-penalty form is the
    base model (paper §3.1).
    """
    data = data or {}
    hard_tw = bool(data.get("hard_tw", False))
    allow_wait = bool(data.get("allow_wait", False))

    # Idle wait w_i: only meaningful at customer/layby stops, and only in
    # benchmark-parity mode.  Everywhere else it is fixed to 0 so the default
    # model is unchanged (v3 no-idle-waiting convention).
    _waitable = (set(m.Cset) | set(m.Lset)) if allow_wait else set()
    for i in m.I:
        if i not in _waitable:
            m.w[i].fix(0.0)

    def _tp(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.ta[i + 1] == m.td[i] + m.D_nom[i]
    m.time_prop = pyo.Constraint(m.I, rule=_tp)

    # (depart_C): customer departure — service, then break/rest (TW1),
    # then optional benchmark idle wait
    m.td_C = pyo.Constraint(m.Cset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.S[i] + m.taub[i] + m.taur[i] + m.w[i])

    # (4): CS departure — stop overhead (v·Mstop) + queue + charging + break/rest
    #                    + sequential repositioning (σ·Mseq)
    m.td_K = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.v[i]*m.Mstop[i] + m.Q_nom[i]*m.y[i]
                 + m.tauc[i] + m.taub[i] + m.taur[i] + m.sigma[i]*m.Mseq[i])

    # (M8): layby departure — parking overhead + break/rest only
    if list(m.Lset):
        m.td_L = pyo.Constraint(m.Lset, rule=lambda m, i:
            m.td[i] == m.ta[i] + m.Mlay[i] * _model_xsum(m, i)
                     + m.taub[i] + m.taur[i] + m.w[i])

    # TW2 — out-of-window indicator, eqs. (5)/(6).  The single horizon big-M H
    # (C1) upper-bounds any feasible arrival span, so it relieves both the
    # early and the late side for every customer when delta = 1.
    if hard_tw:
        # Hard-window sensitivity: windows enforced, indicator fixed to 0
        m.tw_early = pyo.Constraint(m.Cset, rule=lambda m, i:
            m.ta[i] >= m.Wha[i])
        m.tw_late = pyo.Constraint(m.Cset, rule=lambda m, i:
            m.ta[i] <= m.Whf[i])
        m.delta_zero = pyo.Constraint(m.Cset, rule=lambda m, i:
            m.delta[i] == 0)
    else:
        # Early side: per-stop big-M.  With ta[i] ≥ lb_t[i] as a variable
        # bound, M_i = Wha[i] − lb_t[i] is valid (and if Wha ≤ lb_t the
        # early violation is impossible, so the row is skipped — never
        # emitted with M = 0, which would wrongly enforce the window at
        # δ = 1).  This is the side the minimizing LP exploits under the
        # global H (fractional δ ≈ (Wha−ta)/H relieves the window for free).
        _lb_t = data.get("lb_t", {}) or {}

        def _tw_early(m, i):
            M_i = pyo.value(m.Wha[i]) - float(_lb_t.get(i, 0.0))
            if M_i <= 1e-9:
                return pyo.Constraint.Skip
            return m.ta[i] >= m.Wha[i] - M_i * m.delta[i]
        m.tw_early = pyo.Constraint(m.Cset, rule=_tw_early)
        # Late side: the global H stays — there is no valid smaller M
        # without arrival upper bounds (rest durations are unbounded).
        m.tw_late = pyo.Constraint(m.Cset, rule=lambda m, i:
            m.ta[i] <= m.Whf[i] + m.H * m.delta[i])



def add_valid_inequalities(m: pyo.ConcreteModel,
                           data: dict,
                           init_state: dict | None = None,
                           rho2_limit: int | None = None) -> None:
    """
    Add valid inequalities to model *m* in-place.

    Parameters
    ----------
    m          : Pyomo ConcreteModel from build_model() or build_horizon_model().
    data       : The same data dict passed to the build function (full or sub).
    init_state : dict with keys 'sd' / 'ta' / optional 'h' (from init_state
                 passed to build_horizon_model).  Required for subproblems;
                 omit (or pass None) for the full-route model where sd=0 and
                 the route starts fresh at t0.
    rho2_limit : reduced-rest budget in effect for THIS model (the horizon
                 subproblem passes its remaining budget); None → the
                 full-route budget data["rho_bar"] (default 3).
    """
    N       = data["N"]
    I_list  = list(data["I"])       # local indices [0, 1, ..., N]
    D       = data["D"]             # {local_leg_index: hours}
    Ecap    = data["Ecap"]
    Emin    = data["Emin"]

    usable  = Ecap - Emin           # max energy gain per charging session
    D_total = sum(D.get(i, 0.0) for i in range(N))

    sd_0  = 0.0
    h_0   = 0.0
    if init_state is not None:
        sd_0  = float(init_state.get("sd", 0.0))
        h_0   = float(init_state.get("h",  0.0))
        t0    = float(init_state["ta"])
    else:
        t0    = float(data.get("t0", data.get("T_START", 8.0)))

    if rho2_limit is None:
        rho2_limit = int(data.get("rho_bar", 3))

    Tdrv_c  = float(pyo.value(m.Tdrv_cons))
    # M1: a valid inequality must hold for ALL feasible schedules, including
    # those legally using the 10 h extended-driving allowance — so the
    # shift-driving VIs use Tdrv_sh2, never the regular 9 h limit.
    Tdrv_s  = float(pyo.value(m.Tdrv_sh2))
    Tspr1   = float(pyo.value(m.Tspr1))
    Tspr2   = float(pyo.value(m.Tspr2))

    _add_vi1(m, usable)
    _add_vi3(m, I_list, D, N, Tdrv_s)
    _add_vi4(m, I_list, D, N, Tdrv_c)
    _add_vi5(m, I_list, D_total, sd_0, Tdrv_s)
    _add_vi6(m, I_list, D, N, Tdrv_s, sd_0, rho2_limit)
    _add_vi7(m, I_list, N, Tspr1, Tspr2, t0, h_0,
             is_subproblem=(init_state is not None))
    # Window energy covers: integral charge-session mass per depletion
    # window.  Each forced session drags its queue (Q·y) and stop overhead
    # (v·Mstop) into the duty, which VI-7 then converts into rest mass —
    # the LP otherwise pays these only fractionally through fractional y.
    add_window_energy_covers(m, data)


# ─────────────────────────────────────────────────────────────────────────────
# VI-1  Tightened charging energy bound
# ─────────────────────────────────────────────────────────────────────────────

def _add_vi1(m, usable):
    """
    ed[i] − ea[i]  ≤  (Ecap − Emin) · y[i]     for i ∈ K

    Tighter than the existing pwl_no_free_charge which uses Ecap instead of
    Ecap−Emin.  The existing constraint is deactivated and replaced.

    Validity: if y=0 then ed=ea; if y=1 then ed≤Ecap and ea≥Emin so
    ed−ea ≤ Ecap−Emin.  Reference: big-M tightening [1, §9.3].
    """
    if not list(m.Kset):
        return
    if hasattr(m, "pwl_no_free_charge"):
        m.pwl_no_free_charge.deactivate()

    m.vi1 = pyo.Constraint(
        m.Kset,
        rule=lambda m, i: m.ed[i] - m.ea[i] <= usable * m.y[i],
        doc="VI-1: ed-ea ≤ (Ecap-Emin)·y"
    )


# ─────────────────────────────────────────────────────────────────────────────
# VI-3  Prefix shift-rest count
# ─────────────────────────────────────────────────────────────────────────────

def _add_vi3(m, I_list, D, N, Tdrv_sh1):
    """
    For each i ∈ I:
        Σ_{l<i} (ρ1[l] + ρ2[l])  ≥  ⌈ Σ_{l<i} D_l / T^drv_shift ⌉ − 1

    Prefix shift-rest count: before reaching stop i, enough shift rests must
    have been taken to cover the accumulated driving time.
    """
    if Tdrv_sh1 <= 0:
        return

    active = {}
    cum_D = 0.0
    for i in I_list:
        if i > 0:
            cum_D += D.get(i - 1, 0.0)
        rhs = max(0, _mi.ceil(cum_D / Tdrv_sh1) - 1)
        if rhs <= 0:
            continue
        stops_before = [l for l in I_list if l < i]
        if not stops_before:
            continue
        active[i] = (stops_before, int(rhs))

    def _rule(m, i):
        if i not in active:
            return pyo.Constraint.Skip
        stops_before, rhs = active[i]
        return sum(m.rho1[l] + m.rho2[l] for l in stops_before) >= rhs

    m.vi3 = pyo.Constraint(m.I, rule=_rule,
                           doc="VI-3: prefix shift-rest count")


# ─────────────────────────────────────────────────────────────────────────────
# VI-4  Prefix consecutive-driving reset count
# ─────────────────────────────────────────────────────────────────────────────

def _add_vi4(m, I_list, D, N, Tdrv_cons):
    """
    For each i ∈ I:
        Σ_{l<i} (x_b45[l] + x_b30[l] + ρ1[l] + ρ2[l])  ≥  ⌈ Σ_{l<i} D_l / T^drv_cons ⌉ − 1

    Prefix consecutive-driving reset count: before reaching stop i, enough
    cd-resets must have been taken to cover the accumulated driving time.
    Note: x_b15 is excluded as it does not reset the cd accumulator.
    """
    if Tdrv_cons <= 0:
        return

    active = {}
    cum_D = 0.0
    for i in I_list:
        if i > 0:
            cum_D += D.get(i - 1, 0.0)
        rhs = max(0, _mi.ceil(cum_D / Tdrv_cons) - 1)
        if rhs <= 0:
            continue
        stops_before = [l for l in I_list if l < i]
        if not stops_before:
            continue
        active[i] = (stops_before, int(rhs))

    def _rule(m, i):
        if i not in active:
            return pyo.Constraint.Skip
        stops_before, rhs = active[i]
        return sum(m.x_b45[l] + m.x_b30[l] + m.rho1[l] + m.rho2[l]
                   for l in stops_before) >= rhs

    m.vi4 = pyo.Constraint(m.I, rule=_rule,
                           doc="VI-4: prefix cd-reset count")


# ─────────────────────────────────────────────────────────────────────────────
# VI-5  Minimum shift-rest count
# ─────────────────────────────────────────────────────────────────────────────

def _add_vi5(m, I_list, D_total, sd_0, Tdrv_sh1):
    """
    Σ (ρ1 + ρ2)  ≥  max(0, ⌈(D_total + sd_0) / T_drv_sh1⌉ − 1)

    Validity: same block-decomposition argument as VI-4 but applied to the
    shift-driving accumulator sd (reset ONLY by ρ1 or ρ2, not by breaks).
    sd[0]=sd_0 is the accumulated shift driving at the sub-window start.
    Reference: [5], [2].
    """
    if Tdrv_sh1 <= 0:
        return
    n_rho = max(0, _mi.ceil((D_total + sd_0) / Tdrv_sh1) - 1)
    if n_rho <= 0:
        return

    m.vi5 = pyo.Constraint(
        expr=sum(m.rho1[i] + m.rho2[i] for i in I_list) >= n_rho,
        doc=f"VI-5: shift-rests ≥ {n_rho} (sd_0={sd_0:.2f}h)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# VI-6  Rest-mix count (regular 11 h rests forced beyond the reduced budget)
# ─────────────────────────────────────────────────────────────────────────────

def _add_vi6(m, I_list, D, N, Tdrv_sh1, sd_0, rho2_limit):
    """
    Σ ρ1                ≥  n_rho − rho2_limit                       (global)
    Σ_{l<i} ρ1[l]       ≥  n_i   − rho2_limit          (prefix, for each i)

    with n_rho / n_i the VI-5 / VI-3 right-hand sides
    (⌈(sd_0 + Σ D)/T_drv_sh⌉ − 1).

    Validity: linear consequence of the model's reduced-rest budget
    Σρ2 ≤ rho2_limit (rst_lim) combined with the VI-3/VI-5 rest counts —
    every rest beyond the reduced budget must be a REGULAR (11 h) one.
    Not LP-implied: the relaxation otherwise satisfies the count with the
    cheaper 9 h reduced rests only.
    """
    if Tdrv_sh1 <= 0:
        return

    active = {}
    cum_D = sd_0
    for i in I_list:
        if i > 0:
            cum_D += D.get(i - 1, 0.0)
        rhs = max(0, _mi.ceil(cum_D / Tdrv_sh1) - 1) - int(rho2_limit)
        if rhs <= 0:
            continue
        stops_before = [l for l in I_list if l < i]
        if not stops_before:
            continue
        active[i] = (stops_before, int(rhs))

    # global version = prefix at the destination is included via i = N
    if not active:
        return

    def _rule(m, i):
        if i not in active:
            return pyo.Constraint.Skip
        stops_before, rhs = active[i]
        return sum(m.rho1[l] for l in stops_before) >= rhs

    m.vi6 = pyo.Constraint(m.I, rule=_rule,
                           doc="VI-6: prefix regular-rest count")


# ─────────────────────────────────────────────────────────────────────────────
# VI-7  Shift-spread partition (duty length forces the rest COUNT)
# ─────────────────────────────────────────────────────────────────────────────

def _add_vi7(m, I_list, N, Tspr1, Tspr2, t0, h_0, is_subproblem=False):
    """
    For each stop i ≥ 1:

        ta[i] − t0 − Σ_{l<i} taur[l]
            ≤  Tspr1·Σ_{l<i} ρ1[l] + Tspr2·Σ_{l<i} ρ2[l] + T_last − h_0

    T_last = Tspr1 at the destination in the full-route model (h_term caps the
    unfinished final shift at the regular 13 h spread), Tspr2 otherwise.

    Validity (partition argument): the wall-clock from t0 to arrival at i,
    minus rest durations, is exactly the sum of the shift spreads, and the
    model itself caps each span — R24 gives ≤ Tspr1 before a regular rest and
    ≤ Tspr2 before a reduced one; the trailing span is capped by R25
    (h ≤ Tspr2), or by h_term (≤ Tspr1) at the full-route destination.  The
    first span has h_0 of its spread already consumed at the window start.
    Feasible set unchanged — every term is implied by existing constraints.

    This is the reverse direction the LP lacks: rests adding duration is in
    the time chain, but duration forcing MORE rests is not, which is exactly
    how the relaxation certifies 4 rests on routes whose optimum needs 5.
    """
    if Tspr2 <= 0:
        return

    def _rule(m, i):
        if i <= 0:
            return pyo.Constraint.Skip
        stops_before = [l for l in I_list if l < i]
        t_last = Tspr1 if (i == N and not is_subproblem) else Tspr2
        return (m.ta[i] - t0 - sum(m.taur[l] for l in stops_before)
                <= Tspr1 * sum(m.rho1[l] for l in stops_before)
                 + Tspr2 * sum(m.rho2[l] for l in stops_before)
                 + t_last - h_0)

    m.vi7 = pyo.Constraint(m.I, rule=_rule,
                           doc="VI-7: prefix shift-spread partition")


# ─────────────────────────────────────────────────────────────────────────────
# Optional: window energy cover cuts (extension of VI-3)
# ─────────────────────────────────────────────────────────────────────────────

def add_window_energy_covers(m: pyo.ConcreteModel, data: dict,
                             max_cuts: int = 50) -> int:
    """
    For any segment [a, b] with Σ_{l=a}^{b-1} E[l] > (Ecap−Emin):
        Σ_{i∈K ∩ [a,b]} y[i]  ≥  ⌈(E_{a,b} − usable) / usable⌉

    Validity: entering [a,b] with full battery, at most Ecap−Emin kWh is
    usable without charging.  If the segment consumes more, at least one
    CS stop within [a,b] is mandatory.  Reference: [3, §4], [4, §3.2].

    Returns the number of cuts added.
    """
    N      = data["N"]
    K      = list(data["K"])
    E      = data["E"]
    usable = data["Ecap"] - data["Emin"]

    cuts = []
    for a in range(N):
        seg_E = 0.0
        for b in range(a + 1, N + 1):
            seg_E += E.get(b - 1, 0.0)
            rhs = _mi.ceil(max(0.0, (seg_E - usable) / usable))
            if rhs <= 0:
                continue
            K_win = [i for i in K if a <= i <= b]
            if len(K_win) < rhs:
                continue
            cuts.append((a, b, K_win, int(rhs)))

    cuts.sort(key=lambda c: -c[3])
    cuts = cuts[:max_cuts]

    for a, b, K_win, rhs in cuts:
        name = f"vi_win_{a}_{b}"
        setattr(m, name,
                pyo.Constraint(expr=sum(m.y[i] for i in K_win) >= rhs,
                               doc=f"window cover [{a},{b}] rhs={rhs}"))
    return len(cuts)


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
    _drop_split_break(m)

    # ── Objective (eq. 1): arrival + fixed penalty per missed window (TW2) ───
    m.obj = pyo.Objective(
        expr=m.ta[N] + m.beta * sum(m.delta[i] for i in C),
        sense=pyo.minimize)

    # Earliest-arrival lower bounds (valid: forward pass over mandatory
    # dwell — service at customers, zero elsewhere).  Upper bounds stay OFF:
    # ub_t is not valid while rest durations are unbounded above.
    for i in data["I"]:
        m.ta[i].setlb(lb_t.get(i, 0.0))

    # ── Initial conditions ────────────────────────────────────────────────────
    m.init_ta  = pyo.Constraint(expr=m.ta[0] == data.get("T_START", 0.0))
    m.init_ea  = pyo.Constraint(expr=m.ea[0] == m.E0)
    m.init_cd  = pyo.Constraint(expr=m.cd[0] == 0)
    m.init_sd  = pyo.Constraint(expr=m.sd[0] == 0)
    m.init_sw  = pyo.Constraint(expr=m.sw[0] == 0)
    m.init_phi = pyo.Constraint(expr=m.phi[0] == 0)
    m.init_h   = pyo.Constraint(expr=m.h[0]  == 0)   # M5
    m.init_z   = pyo.Constraint(expr=m.z[0]  == 0)   # M6

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
    _add_time_constraints(m, N, data)
    _add_v_sigma_constraints(m, m.M_big)
    _add_pwl_charging_constraints(m, K, R, Rseg)
    _add_break_rest_constraints(m, N, data["I"], set(K), m.M_big,
                                rho2_limit=int(data.get("rho_bar", 3)))
    _add_hos_accumulator_constraints(m, N, data["I"], set(C), set(K),
                                     data["S"], m.M_drv, m.M_sd, m.M_sw, m.TK,
                                     ext_budget=int(data.get("ext_bar", 2)),
                                     M_lay_dict=data.get("M_lay"))

    # Directive 2002/15 working-time breaks in-model (benchmark parity);
    # default False keeps the ex-post-only treatment (S3).
    if data.get("wtd_rules", False):
        _add_wtd_constraints(m, N, data)

    # (h_term / eq. 70): terminal spread — the off-model final rest after
    # arrival is assumed regular, so the unfinished final shift is bounded by
    # the 13 h regular-rest spread.  Full-route only (a horizon end may sit
    # mid-shift, where this would be wrong).
    m.spread_term = pyo.Constraint(expr=m.h[N] <= m.Tspr1)

    # ── M9 (R21): weekly working-time cap (Directive 2002/15/EC, 60 h) ────────
    # OUT OF PROBLEM SCOPE (2026-07-29).  The paper models the DAILY provisions
    # of Reg. 561/2006 / Dir. 2002/15/EC only; weekly-level accounting (weekly
    # rests, 56/90 h driving totals, the 60 h working-time week) is excluded
    # throughout — no model enforces it and the simulator does not treat a
    # breach as infeasible.  BEHDV still accumulates realized weekly working
    # time (sw_week) and records a breach as a DIAGNOSTIC note (weekly_notes /
    # metrics.weekly_cap_exceeded), reported in the paper as a compliance-
    # margin statistic on long routes.  Twk60 stays declared as a parameter
    # for that diagnostic.

    _fix_ferry_nodes(m, data)
    return m


def _fix_ferry_nodes(m, data: dict) -> None:
    """Force the mandatory break at sea-crossing (ferry) nodes.

    A ferry node is a layby at which the vehicle is aboard a vessel for a
    KNOWN duration.  It is expressed with two variable fixings rather than new
    constraints:

        x_b45[F] = 1          the break is taken, not chosen
        taub[F]  = T_cross    for exactly the crossing duration

    Everything else follows from the constraints that already exist:
      * one_brk forces x_b15 = x_b30 = rho1 = rho2 = 0, and rst_ub then forces
        taur[F] = 0 — so a daily rest can never be taken on board (Art. 9's
        rest-on-ferry option is deliberately out of scope);
      * xsum[F] = 1 makes the layby departure equation charge M_lay[F], the
        boarding/disembarking overhead, which the vehicle therefore cannot
        avoid;
      * the crossing is node dwell, so it never enters the driving
        accumulators cd / sd, and taub is not work, so it does not enter sw;
      * the consecutive-driving reset indicator already contains x_b45, so the
        4.5 h clock resets on disembarkation, as Art. 9 of Reg. (EC) 561/2006
        prescribes.

    The crossing does count toward the shift spread h (it is on-duty dwell,
    not a rest), which is the intended reading.
    """
    ferry = {int(k): float(v) for k, v in (data.get("ferry") or {}).items()}
    if not ferry:
        return
    L_set = set(data.get("L", []))
    for f, t_cross in ferry.items():
        if f not in m.I:
            continue
        if f not in L_set:
            raise ValueError(
                f"ferry node {f} must be a layby (in data['L']); got "
                f"{'customer' if f in set(data['C']) else 'CS' if f in set(data['K']) else 'unknown'}")
        m.x_b45[f].fix(1)
        m.taub[f].fix(t_cross)


def solve_model(model, tee=True, mipgap=0.005, timelimit=60 * 60 * 2):
    """Solve the full-route MIP (default: 0.5% gap, 2h limit)."""
    solver = pyo.SolverFactory("gurobi")
    _apply_solver_threads(solver)
    solver.options["MIPGap"]    = mipgap
    solver.options["TimeLimit"] = timelimit
    try:
        results = _solve_quiet(solver, model, tee)
        status  = str(results.solver.termination_condition)
    except RuntimeError:
        status  = "infeasible"
        results = None
    print(f"  Solver: {status}")
    return results, status


def _extract_stop_dict(model, i, data, C_set, K_set) -> dict:
    """Shared per-stop extraction for full-route and horizon models."""
    is_K  = i in K_set
    is_C  = i in C_set
    y_val = round(pyo.value(model.y[i])) if is_K else 0
    tauq_val = data["Q"].get(i, 0.0) * y_val if is_K else 0.0

    def _v(var, default=0.0):
        try:
            val = pyo.value(var)
            return default if val is None else val
        except Exception:
            return default

    return dict(
        i     = i,
        ta    = _v(model.ta[i]),
        td    = _v(model.td[i]),
        ea    = _v(model.ea[i]),
        ed    = _v(model.ed[i]),
        cd    = _v(model.cd[i]),
        sd    = _v(model.sd[i]),
        sw    = _v(model.sw[i]),
        h     = _v(model.h[i]),                              # M5 spread
        tauc  = _v(model.tauc[i]) if is_K else 0.0,
        tauq  = tauq_val,
        taub  = _v(model.taub[i]),
        taur  = _v(model.taur[i]),
        wait  = _v(model.w[i]),                              # benchmark idle
        g     = _v(model.g[i]) if is_K else 0.0,             # M2 break credit
        delta = round(_v(model.delta[i])) if is_C else 0,    # TW2 window miss
        y     = y_val,
        sigma = round(_v(model.sigma[i])) if is_K else 0,
        z     = round(_v(model.z[i])),                       # M6 extension flag
        b45   = round(_v(model.x_b45[i])),
        b15   = round(_v(model.x_b15[i])),
        b30   = round(_v(model.x_b30[i])),
        rho1  = round(_v(model.rho1[i])),
        rho2  = round(_v(model.rho2[i])),
        is_C  = is_C,
        is_K  = is_K,
        D_nom = data["D"].get(i, 0.0),
    )


def extract_solution(model, data: dict) -> list[dict]:
    """Extract per-stop solution dicts (same schema as extract_horizon_solution)."""
    C_set = set(data["C"])
    K_set = set(data["K"])
    return [_extract_stop_dict(model, i, data, C_set, K_set)
            for i in data["I"]]


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

    L_glob = set(full_data.get("L", []))

    I_loc = list(range(H + 1))
    C_loc = [j for j in range(H + 1) if (start_stop + j) in C_glob]
    K_loc = [j for j in range(H + 1) if (start_stop + j) in K_glob]
    L_loc = [j for j in range(H + 1) if (start_stop + j) in L_glob]

    D_src = D_override if D_override is not None else full_data["D"]
    E_src = E_override if E_override is not None else full_data["E"]
    D_loc = {j: D_src.get(start_stop + j, 0.0) for j in range(H)}
    E_loc = {j: E_src.get(start_stop + j, 0.0) for j in range(H)}

    # ── LA energy guard (settings.LA_ENERGY_QUANTILE) ────────────────────────
    # Applies ONLY to the nominal sub-problem (E_override is None) — the one
    # whose re-solve produces the COMMITTED charge quantity.  Per-scenario
    # evaluations already carry their own energy and are left untouched.
    #
    # Scope: the legs from here up to arrival at the next CS.  That is the
    # stretch over which a shortfall cannot be repaired, because there is no
    # station on it; beyond the next CS the rolling horizon re-decides anyway,
    # and only the first action is ever committed.
    _eq = full_data.get("la_energy_quantile")
    if E_override is None and _eq:
        from src.settings import energy_at_quantile as _e_at_q
        _K_glob = set(full_data["K"])
        _km_g   = full_data.get("km", {}) or {}
        _D_g    = full_data["D"]
        _cv_g   = float(full_data.get("la_energy_cv", TRAVEL_TIME_CV_TARGET))
        _limit  = full_data["N"]
        for _g in range(start_stop + 1, full_data["N"] + 1):
            if _g in _K_glob:
                _limit = _g
                break
        for j in range(H):
            _g = start_stop + j
            if _g >= _limit:
                break
            _L, _d = _km_g.get(_g), _D_g.get(_g)
            if _L and _d:
                E_loc[j] = max(E_loc[j], _e_at_q(_L, _d, _eq, _cv_g))

    S_loc      = {j: full_data["S"].get(start_stop + j, 0.0) for j in C_loc}
    Q_loc      = {j: full_data["Q"].get(start_stop + j, 0.0) for j in K_loc}
    M_stop_loc = {j: full_data["M_stop"].get(start_stop + j, 0.0) for j in K_loc}
    M_seq_loc  = {j: full_data["M_seq"].get(start_stop + j, 0.0)  for j in K_loc}
    M_lay_loc  = {j: full_data.get("M_lay", {}).get(start_stop + j, 0.0)
                  for j in L_loc}
    # ferry nodes fall inside the horizon window only sometimes; map to local
    _ferry_glob = {int(k): float(v)
                   for k, v in (full_data.get("ferry") or {}).items()}
    ferry_loc = {j: _ferry_glob[start_stop + j]
                 for j in L_loc if (start_stop + j) in _ferry_glob}
    # Keep old M dict for covering-inequality helper (uses M_dict = sub_data["M"])
    M_loc = {j: full_data["M"].get(start_stop + j, 5.0 / 60)
             for j in range(H + 1)}

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

    # C1 — the full-route horizon big-M is a valid upper bound for the
    # sub-route too (the sub-route duration is shorter), so reuse it.
    H_bigM = full_data.get("H")
    if H_bigM is None:
        H_bigM = T_hor - t0

    return dict(
        label=f"subproblem [{start_stop}→{end_stop}]",
        title=f"sub_{start_stop}_{end_stop}",
        N=H, I=I_loc, C=C_loc, K=K_loc, L=L_loc, R=R, Rseg=Rseg,
        D=D_loc, E=E_loc,
        S=S_loc, Q=Q_loc, M=M_loc, M_stop=M_stop_loc, M_seq=M_seq_loc,
        M_lay=M_lay_loc, ferry=ferry_loc,
        Wha=Wha_loc, Whf=Whf_loc,
        t0=t0,                      # sub-window start time
        H=H_bigM,                   # C1 window / rest big-M (full-route bound)
        hard_tw=full_data.get("hard_tw", False),
        beta=full_data.get("beta", BETA_TW),
        allow_split=full_data.get("allow_split", True),
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
        Tspr1=full_data.get("Tspr1", 13.0),
        Tspr2=full_data.get("Tspr2", 15.0),
        Twk60=full_data.get("Twk60", 60.0),
        rho_bar=full_data.get("rho_bar", 3),
        ext_bar=full_data.get("ext_bar", 2),
        M_drv=full_data["M_drv"], M_sd=full_data["M_sd"],
        M_sw=full_data["M_sw"],   M_big=full_data["M_big"],
        M_h=full_data.get("M_h", full_data.get("Tspr2", 15.0)),
        global_start=start_stop, global_end=end_stop, global_N=N_glob,
        E_global=full_data["E"],
        K_global=set(full_data["K"]),
        e_to_next_cs=e_to_next_cs,
    )


def build_horizon_model(sub_data: dict, init_state: dict,
                        fixed_action=None, rho2_remaining: int = 3,
                        ext_remaining: int = 2,
                        fixed_plan: dict = None,
                        plan_mode: str = "fix",
                        repair_penalty: float = 1000.0,
                        tee=False) -> pyo.ConcreteModel:
    """
    Build the rolling-horizon sub-problem Pyomo model.

    Differences from build_model (full-route):
    1. Initial conditions come from init_state (not hardcoded to origin).
    2. fixed_action (if given) fixes binary decisions at local stop 0.
    3. rho2_remaining caps the reduced-rest budget in the sub-window.
    4. ext_remaining caps the 10 h extended-driving budget (M6).
    5. No activities allowed at the last horizon stop (binaries fixed 0).
    6. fixed_plan (SP1/RO1 recourse): per-stop binary structure taken from an
       offline plan, either fixed exactly (plan_mode="fix" — duration-only
       LP recourse) or imposed as lower bounds (plan_mode="repair" — binary
       activities may be ADDED but never removed, and each addition is
       penalised by `repair_penalty` in the objective).

    Parameters
    ----------
    sub_data       : dict from make_subproblem_data
    init_state     : dict — ta, ea, cd, sd, sw, phi (+ optional h)
    fixed_action   : dict — y, break_type, rest_type (or None for free solve)
    rho2_remaining : int  — remaining r2 budget (rho_bar − vehicle.rho2_used)
    ext_remaining  : int  — remaining extension budget (ext_bar − used)
    fixed_plan     : dict {local_stop: {"y", "break_type", "rest_type"}}
    plan_mode      : "fix" | "repair"
    """
    import logging as _lg
    _lg.getLogger("pyomo").setLevel(_lg.ERROR)

    m = pyo.ConcreteModel()

    N, C, K, R, Rseg, lb_t, ub_t = _declare_common_params(m, sub_data)


    # ── Variables ─────────────────────────────────────────────────────────────
    _declare_common_vars(m)
    _drop_split_break(m)

    # ── Objective: arrival + fixed window penalty (TW2) (+ repair penalty) ────
    _obj_expr = m.ta[N] + m.beta * sum(m.delta[i] for i in C)
    if fixed_plan is not None and plan_mode == "repair":
        # SP-recourse: lexicographic-ish — heavily penalise every ADDED
        # binary activity, then ta[N] + beta·Σdelta (repair step ordering).
        _plan = fixed_plan
        _added = []
        for j in sub_data["I"]:
            p = _plan.get(j, {})
            if j in set(K) and not int(p.get("y", 0) or 0):
                _added.append(m.y[j])
            if p.get("break_type") not in ("b45",): _added.append(m.x_b45[j])
            if p.get("break_type") not in ("b15",): _added.append(m.x_b15[j])
            if p.get("break_type") not in ("b30",): _added.append(m.x_b30[j])
            if p.get("rest_type")  not in ("r1",):  _added.append(m.rho1[j])
            if p.get("rest_type")  not in ("r2",):  _added.append(m.rho2[j])
        _obj_expr = _obj_expr + repair_penalty * sum(_added)
    m.obj = pyo.Objective(expr=_obj_expr, sense=pyo.minimize)

    # Earliest-arrival lower bounds (see build_model; ub_t deliberately off).
    for i in sub_data["I"]:
        m.ta[i].setlb(lb_t.get(i, 0.0))

    # ── Initial conditions from init_state ────────────────────────────────────
    m.init_ta  = pyo.Constraint(expr=m.ta[0]  == init_state["ta"])
    m.init_ea  = pyo.Constraint(expr=m.ea[0]  == init_state["ea"])
    m.init_cd  = pyo.Constraint(expr=m.cd[0]  == init_state["cd"])
    m.init_sd  = pyo.Constraint(expr=m.sd[0]  == init_state["sd"])
    m.init_phi = pyo.Constraint(expr=m.phi[0] == init_state["phi"])
    m.init_sw = pyo.Constraint(expr=m.sw[0] == init_state["sw"])
    # M5: spread at arrival; M6: z[0] is left free — sd[0] > 9 h forces it.
    m.init_h  = pyo.Constraint(expr=m.h[0] == init_state.get("h", 0.0))

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

    # ── SP1/RO1: fixed-structure plan over the whole window ───────────────────
    if fixed_plan is not None:
        K_set_loc = set(K)

        def _plan_bin(j, key, val):
            p = fixed_plan.get(j)
            if p is None:
                return None
            if key == "y":
                return int(p.get("y", 0) or 0)
            if key == "brk":
                return int(p.get("break_type") == val)
            if key == "rst":
                return int(p.get("rest_type") == val)
            return None

        _pl = pyo.ConstraintList()
        m.plan_bins = _pl
        # stop 0 activities already happened or are being decided now; the
        # plan applies from local stop 0 (current stop) through N−1.  The
        # last horizon stop stays activity-free (fixed below).
        for j in sub_data["I"]:
            if j >= N:
                continue
            if fixed_plan.get(j) is None:
                continue
            targets = [
                (m.x_b45[j], _plan_bin(j, "brk", "b45")),
                (m.x_b15[j], _plan_bin(j, "brk", "b15")),
                (m.x_b30[j], _plan_bin(j, "brk", "b30")),
                (m.rho1[j],  _plan_bin(j, "rst", "r1")),
                (m.rho2[j],  _plan_bin(j, "rst", "r2")),
            ]
            if j in K_set_loc:
                targets.append((m.y[j], _plan_bin(j, "y", None)))
            for var, val in targets:
                if val is None:
                    continue
                if plan_mode == "fix":
                    _pl.add(var == val)
                else:   # "repair": activities may be added, never removed
                    if val == 1:
                        _pl.add(var >= 1)

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
    _add_time_constraints(m, N, sub_data)
    _add_v_sigma_constraints(m, m.M_big)
    _add_pwl_charging_constraints(m, K, R, Rseg)
    _add_break_rest_constraints(m, N, sub_data["I"], set(K), m.M_big,
                            rho2_limit=rho2_remaining)
    _add_hos_accumulator_constraints(m, N, sub_data["I"], set(C), set(K),
                                    sub_data["S"], m.M_drv, m.M_sd, m.M_sw, m.TK,
                                    is_subproblem=True,
                                    ext_budget=ext_remaining,
                                    M_lay_dict=sub_data.get("M_lay"))

    add_valid_inequalities(m, sub_data, init_state=init_state,
                           rho2_limit=rho2_remaining)


    # DEBUG
    #pyo.Constraint(m.Cset, rule=lambda m, i: m.x_b45[i] == 0)
    #pyo.Constraint(m.Cset, rule=lambda m, i: m.x_b15[i] == 0)
    #pyo.Constraint(m.Cset, rule=lambda m, i: m.x_b30[i] == 0)

    # Sea crossings inside this horizon window (local indices) — same two
    # fixings as the full-route model.  Stop 0 is the vehicle's current
    # position and its dwell is already realized, so it is never re-fixed.
    _sub_ferry = dict(sub_data.get("ferry") or {})
    _sub_ferry.pop(0, None)
    _fix_ferry_nodes(m, dict(sub_data, ferry=_sub_ferry))

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
            _sv(model.h,     "h")
            _sv(model.z,     "z")
            _sv(model.taub,  "taub")
            _sv(model.taur,  "taur")
            _sv(model.x_b45, "b45")
            _sv(model.x_b15, "b15")
            _sv(model.x_b30, "b30")
            _sv(model.rho1,  "rho1")
            _sv(model.rho2,  "rho2")
            if i in model.Cset:
                _sv(model.delta, "delta")
            if i in model.Kset:
                _sv(model.y,    "y")
                _sv(model.tauc, "tauc")
                _sv(model.g,    "g")


def _solve_horizon_model(model, time_limit=8, tee=False, relax=True, had_warm=False):
    """
    Solve a horizon model with Gurobi.

    Returns (results, status_str, solve_info).
    solve_info keys: wall_s, obj, n_vars, n_cons, had_warm, relax, status.
    """
    import time as _tm
    n_vars_pre = sum(1 for _ in model.component_data_objects(pyo.Var, active=True))
    n_cons_pre = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))

    if relax:
        # ── Partial LP relaxation ──────────────────────────────────────────────
        # keep x_b45/b15/b30/rho1/rho2 BINARY at customer stops; keep ALL
        # integer variables at local stop 1 (next stop) as integer.
        #
        # TW4: the out-of-window indicators delta stay INTEGER at every
        # customer stop.  Their big-M relaxation is terrible — in the LP,
        # delta takes value ≈ miss/H (a 2 h miss over a 100+ h horizon
        # ⇒ delta ≈ 0.02 ⇒ penalty ≈ nothing), so a fully-relaxed subproblem
        # treats windows as free and the action scoring is silently biased
        # toward window-ignoring actions.  |C ∩ horizon| binaries per
        # subproblem — negligible solve-time impact, removes the bias.

        _CUST_KEEP_BINARY = frozenset(("x_b45", "x_b15", "x_b30",
                                       "rho1", "rho2", "delta"))
        cust_set = set(model.Cset)

        for var in model.component_objects(pyo.Var, active=True):
            vname = var.local_name
            for idx, vdata in var.items():
                if vdata.domain not in (pyo.Binary, pyo.Integers,
                                        pyo.NonNegativeIntegers):
                    continue
                # Keep binary if this is a break/rest/window variable at a
                # customer stop
                if (vname in _CUST_KEEP_BINARY
                        and isinstance(idx, int) and idx in cust_set):
                    continue  # leave as Binary
                # C2: keep the sequential-mode flag sigma integer at every CS
                # stop — (45) governs feasibility/timing (M_seq overhead and
                # break-credit g), so a fractional sigma would let the LP dodge
                # the uncovered-break sequential penalty.
                if vname == "sigma":
                    continue  # leave as Binary
                # Keep all integer variables at local stop 1 (next stop) as integer
                stop_idx = idx if isinstance(idx, int) else (idx[0] if isinstance(idx, tuple) else None)
                if stop_idx == 1:
                    continue
                # Relax everything else
                vdata.domain = pyo.NonNegativeReals
                if vdata.ub is None:
                    vdata.setub(1.0)

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

    solver = pyo.SolverFactory("gurobi")
    _apply_solver_threads(solver)
    solver.options["TimeLimit"] = time_limit
    if not relax:
        solver.options["MIPGap"] = 0.005

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
    C_set = set(sub_data["C"])
    K_set = set(sub_data["K"])
    return [_extract_stop_dict(model, i, sub_data, C_set, K_set)
            for i in sub_data["I"]]


def solve_horizon(full_data: dict, start_stop: int, end_stop: int,
                  init_state: dict, fixed_action=None,
                  D_override=None, E_override=None,
                  rho2_remaining: int = 3, tee: bool = False,
                  time_limit: int = 30, relax: bool = True,
                  warm_start=None,
                  ext_remaining: int = 2,
                  fixed_plan: dict = None,
                  plan_mode: str = "fix") -> dict:
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
                                   ext_remaining=ext_remaining,
                                   fixed_plan=fixed_plan,
                                   plan_mode=plan_mode,
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
        _sd_lim = (data.get("Tdrv_sh2", 10.0) if s.get("z", 0)
                   else data["Tdrv_sh1"])
        if s["cd"] > data["Tdrv_cons"] + EPS:
            print(f"  FAIL  consec_drv stop {i}: {s['cd']:.3f} > {data['Tdrv_cons']}"); ok=False
        if s["sd"] > _sd_lim + EPS:
            print(f"  FAIL  shift_drv  stop {i}: {s['sd']:.3f} > {_sd_lim}");  ok=False
        if s.get("h", 0.0) > data.get("Tspr2", 15.0) + EPS:
            print(f"  FAIL  spread     stop {i}: {s['h']:.3f} > {data.get('Tspr2', 15.0)}"); ok=False
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
        if s["is_K"]:
            sigma_s  = int(s.get("sigma", 0))
            v_s      = bool(s["y"] or s["b45"] or s["b15"] or s["b30"] or
                            s["rho1"] or s["rho2"])
            mstop_s  = float(v_s)      * data.get("M_stop", {}).get(i, 0.0)
            mseq_s   = float(sigma_s)  * data.get("M_seq",  {}).get(i, 0.0)
            mode_s   = "SEQ" if sigma_s else "CONC"
            if v_s:
                acts.append(f"[{mode_s}] setup={mstop_s*60:.0f}m")
            if s["y"]:
                acts.append(f"CHG {s['ea']:.0f}→{s['ed']:.0f}kWh"
                            f" ({s['tauc']:.2f}h) Q={s['tauq']*60:.0f}m")
            if sigma_s and mseq_s > EPS:
                acts.append(f"repos={mseq_s*60:.0f}m")
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
from src.plot.plots import plot_solution   # noqa: F401


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT  (python -m src.methods.MILP [instance_name])
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
    from src.instance_gen.instances import ALL_INSTANCES
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