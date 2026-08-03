"""
instances.py — Route data assembly and instance generators
===========================================================
Single source of truth for all route data in the BET scheduling project.

This module owns two responsibilities:

  1. make_data(...)
       Assembles the canonical data dict consumed by every other module.
       Moving this function here (rather than in MILP.py) keeps MILP.py a
       pure modelling file with no knowledge of instance geometry.

  2. Instance generator functions (instance_tiny, instance_realistic, …)
       Each returns a data dict by calling make_data.  All instances are
       registered in ALL_INSTANCES so CLI entry points can reference them
       by name.

Import chain
------------
instances.py → (stdlib only: math, random, os, sys)

No local imports at module level.  MILP.py, Simulation.py, greedy.py etc.
all import FROM instances; instances imports from none of them.  This makes
the dependency graph a strict DAG with instances.py at the leaves.

Data dict keys (produced by make_data)
---------------------------------------
  label, title          str   — human-readable description / filename stem
  N                     int   — destination stop index  (route has stops 0..N)
  I                     list  — all stop indices [0 .. N]
  C                     list  — customer stop indices   (C ∩ K = ∅)
  K                     list  — charging station (CS) stop indices
  R, Rseg               list  — PWL charging curve segment indices
  D        {leg: h}     dict  — nominal travel time per leg (hours)
  E        {leg: kWh}   dict  — nominal energy consumption per leg (kWh)
  km       {leg: km}    dict  — physical leg distance (km); used by ECR(v)
  S        {stop: h}    dict  — service time at each customer stop (h)
  Q        {stop: h}    dict  — queue time at each CS stop (h)
  M        {stop: h}    dict  — manoeuver time per active stop (h)
  E0, Ecap, Emin        float — initial / max / min battery SOC (kWh)
  Ebar     {r: kWh}     dict  — PWL charging curve energy breakpoints
  Tbar     {r: h}       dict  — PWL charging curve time breakpoints
  Wha, Whf {stop: h}    dict  — earliest / latest arrival windows (absolute h)
  T_hor, T_START        float — absolute planning horizon / departure time (h)
  lb_t, ub_t {stop: h}  dict  — arrival-time variable bounds (for MILP tightening)
  Tb45, Tb15, Tb30      float — minimum break durations (h)
  allow_split           bool  — Art. 7 split break (15'+30') available?
  Tr1, Tr2              float — minimum rest durations: daily=11h, reduced=9h
  Tdrv_cons             float — max consecutive driving before mandatory break (4.5 h)
  Tdrv_sh1, Tdrv_sh2    float — max shift driving (9 h / 10 h split-week rule)
  Twrk_cons1/2          float — max consecutive working accumulators
  Twrk_sh               float — max shift working (13 h)
  M_drv, M_sd, M_sw     float — big-M constants for HoS linearisation
  M_big                 float — generic big-M for break/rest linking constraints

HoS regulation references
--------------------------
  EU Regulation 561/2006 (driving times and rest periods):
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32006R0561
  EU Regulation 165/2014 (tachographs):
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32014R0165

ECR model reference
-------------------
  Younes et al. (2020) "Energy consumption model for battery electric vehicles",
  Journal of Energy Storage, 32, 101758.
  https://doi.org/10.1016/j.est.2020.101758
  Formula: ECR(v) = A/v + B + C·v²  (kWh/km),  A=33.055, B=0.2256, C=7.2e−5
"""

from __future__ import annotations

import math
import os
import random
import sys
import numpy as np


from src.settings import (
    ECR_A as _ECR_A, ECR_B as _ECR_B, ECR_C as _ECR_C, ecr as _ecr,
    V_NOM, BATTERY_CAPACITY, SOC_MIN_FRAC, EBAR_FRACS, TBAR,
    Tb45, Tb15, Tb30, ALLOW_SPLIT_BREAK, Tr1, Tr2,
    Tdrv_cons, Tdrv_sh1, Tdrv_sh2, Twrk_cons1, Twrk_cons2, Twrk_sh,
    T_SPR1, T_SPR2, TWK_60, TWK_DRV, RHO_BAR, EXT_BAR, BETA_TW,
    QUEUE_WAIT_MEAN_MIN, QUEUE_WAIT_STD_MIN,
    M_STOP_H, M_SEQ_H, M_MAN_DEFAULT_H, M_LAYBY_H,
    LAYBY_SPACING_KM, LAYBY_MIN_LEG_H,
    SERVICE_TIME_H, CS_SPACING_KM, T_START as _T_START,
    CHARGER_POWER_BASE_KW, scale_tbar, XI_MAX,
)
from src import paths as _paths


# ══════════════════════════════════════════════════════════════════════════════
# TIME BOUNDS
# ══════════════════════════════════════════════════════════════════════════════

def compute_time_bounds(I, C, K, D, S, Q, Tbar, T_hor,
                        t0: float = 0.0,
                        Man_default: float = M_MAN_DEFAULT_H) -> tuple[dict, dict]:
    """
    Forward-pass conservative arrival-time bounds for variable tightening.

    Returns lb[i], ub[i] — dictionaries mapping stop index to the earliest
    and latest feasible absolute arrival time (hours).

    These bounds are used by MILP.build_model and MILP.build_horizon_model to
    set variable lower/upper bounds, which significantly tightens the LP
    relaxation and speeds up presolve.

    Parameters
    ----------
    I            : list[int] — all stop indices
    C            : list[int] — customer stop indices
    K            : list[int] — CS stop indices
    D            : dict {leg: h} — nominal travel times
    S            : dict {stop: h} — service times
    Q            : dict {stop: h} — queue times at CS stops
    Tbar         : dict {r: h} — PWL charging curve time breakpoints
    T_hor        : float — absolute planning horizon
    t0           : float — departure time (absolute hours, default 0)
    Man_default  : float — manoeuver time per active stop; must match M_man
                   in make_data to keep ub_t internally consistent
    """
    N   = max(I)
    TK  = Tbar[max(Tbar)]  # maximum possible charging duration
    C_s = set(C)
    K_s = set(K)

    lb = {0: t0}
    ub = {0: t0}
    for i in range(N):
        if i in C_s:
            dmin = S.get(i, 0.0)
            dmax = S.get(i, 0.0) + Tb45 + Tr1 + Man_default + 0.1
        elif i in K_s:
            dmin = 0.0
            dmax = Q.get(i, 0.0) + TK + Tb45 + Tr1 + Man_default + 0.1
        else:
            dmin = dmax = 0.0
        lb[i + 1] = lb[i] + dmin + D.get(i, 0.0)
        ub[i + 1] = min(T_hor, ub[i] + dmax + D.get(i, 0.0))
    return lb, ub


# ══════════════════════════════════════════════════════════════════════════════
# C1 — HORIZON BIG-M  H  (valid upper bound on any feasible arrival span)
# ══════════════════════════════════════════════════════════════════════════════

def compute_horizon_bigM(N, D, S, Q, M_stop, Tr1,
                         delta_pad: float = XI_MAX - 1.0) -> float:
    """
    C1 — a valid big-M H bounding the total route duration (arrival minus t0).

    There is NO arrival deadline in the model; H is used ONLY as the big-M in
    the window indicators (t_a ≥ W_a − H·δ, t_a ≤ W_f + H·δ) and the rest
    bound (τ_r ≤ H·ρ).  It must genuinely upper-bound any feasible arrival:

        H = Σ_i D_i·(1 + delta_pad)                (driving, worst-inflated)
            + Σ_i S_i + Σ_i Q_i + Σ_i M_stop_i     (service, queue, maneuver)
            + (N + 1)·T_rst_1                       (≤ one 11 h rest per stop)

    The rest term is deliberately loose (each stop can host at most one
    break/rest by (23), so N+1 rests of 11 h dominates every break too),
    keeping H valid for every schedule the solver can produce — optimal or
    not — which is what a big-M requires.  delta_pad defaults to XI_MAX − 1
    (= 0.6): the multiplier support is hard, so D·XI_MAX bounds every
    realised leg time.
    """
    D_total = sum(D.get(i, 0.0) for i in range(N))
    return (D_total * (1.0 + delta_pad)
            + sum(S.values()) + sum(Q.values()) + sum(M_stop.values())
            + (N + 1) * Tr1)


# ══════════════════════════════════════════════════════════════════════════════
# make_data — CANONICAL DATA DICT CONSTRUCTOR
# ══════════════════════════════════════════════════════════════════════════════

def make_data(I, C, K, D, E, Wha, Whf, label, title,
              km=None, Bcap=BATTERY_CAPACITY, Q=None, M_man_h: float = M_MAN_DEFAULT_H,
              rng: np.random.Generator | None = None,
              L=None, T_dead: float | None = None,
              charger_power_kw: float | None = None,
              hard_tw: bool = False,
              beta_tw: float = BETA_TW,
              S: dict | None = None,
              M_lay_h: float | None = None,
              allow_wait: bool = False,
              wtd_rules: bool = False,
              allow_split: bool = ALLOW_SPLIT_BREAK) -> dict:
    """
    Assemble the canonical data dict consumed by MILP.build_model,
    MILP.solve_horizon, BEHDV, oracle_solve, and all instance generators.

    Parameters
    ----------
    I     : list[int]        — all stop indices; must satisfy min(I)=0,
                               max(I)=N, and set(C)|set(K) == set(I)-{0,N}
    C     : list[int]        — customer stop indices  (C ∩ K = ∅)
    K     : list[int]        — charging station stop indices
    D     : dict {leg: h}    — nominal travel time per leg (hours)
    E     : dict {leg: kWh}  — nominal energy consumption per leg (kWh)
    S     : dict {stop: h}   — service time at each customer stop (hours)
    E0    : float            — initial battery SOC (kWh)
    Ecap  : float            — battery capacity (kWh)
    Emin  : float            — minimum allowed SOC (kWh)
    Ebar  : dict {r: kWh}    — PWL charging curve energy breakpoints
    Tbar  : dict {r: h}      — PWL charging curve cumulative-time breakpoints
    Wha   : dict {stop: h}   — earliest arrival at customer (relative to T_START)
    Whf   : dict {stop: h}   — latest   arrival at customer (relative to T_START)
    label : str              — verbose description string
    title : str              — short identifier used as JSON filename stem
    km    : dict {leg: km} or None
            Physical leg distances (km).  When provided, scenario generation
            in scenarios.py couples energy consumption to travel speed via
            ECR(v) = A/v + B + C·v².  When None, defaults to E (i.e. treats
            energy as proportional to travel time — backward-compatible).
    Q     : dict {cs_stop: h} or None
            Fixed queue times at each CS stop (plug-in + initial waiting).
            When None, drawn from a lognormal distribution using `rng`.
    rng   : np.random.Generator or None
            Generator used to draw Q when Q is None.  Pass a seeded
            np.random.default_rng(seed) for reproducible queue times; when
            None, a fresh unseeded generator is used.
    M_man_h : float
            Manoeuver time (h) applied uniformly to every stop.  A manoeuver
            is charged whenever a break or rest is taken without simultaneous
            charging (the driver must physically park/unpark the truck).
            Default 15/60 h (15 min).  Use larger values (e.g. 10.0 h) to
            model high-penalty scenarios where off-CS stops are very costly,
            forcing the optimiser to plan breaks exclusively at CS bays.

    Returns
    -------
    dict — see module docstring for full key listing.

    L     : list[int] or None
            Layby (rest-area) stop indices (M8).  Break/rest-only stops:
            no service, no charging.  Default None = no laybys.
    T_dead : float or None
            Hard deadline at the destination (absolute hours) — constraint
            (R6).  None = no deadline.
    charger_power_kw : float or None
            Average charger power (kW) for the I2 sensitivity axis.  Rescales
            the PWL time breakpoints; None keeps the 200 kW base curve.
    hard_tw : bool
            True → customer windows are hard (tight-slot sensitivity).
            False (default) → soft windows with the fixed out-of-window
            penalty delta (TW2).
    beta_tw : float
            Fixed out-of-window service penalty β (h-equivalent per missed
            window, TW2).
    S     : dict {stop: h} or None
            Per-stop service times at customer stops.  None (default) applies
            the uniform SERVICE_TIME_H to every customer.  Used by external
            benchmark instances (e.g. R15-PGLT) with heterogeneous services.
    M_lay_h : float or None
            Parking overhead (h) at layby stops.  None keeps M_LAYBY_H.
            Benchmark instances pass 0.0 (their idle stops are free).
    allow_wait : bool
            True → MILP adds a free idle-wait variable w_i at customer and
            layby stops (benchmark parity with the TDSP "APO" idle time).
            Default False preserves the v3 no-idle-waiting convention.
    wtd_rules : bool
            True → MILP enforces the Directive 2002/15/EC working-time
            breaks in-model (6 h continuous work → break; 30'/45' cumulative
            per shift).  Default False keeps them ex-post only.
    allow_split : bool
            True (default) → the Art. 7 split break (15' + 30') is available.
            False → x_b15 / x_b30 are dropped from every model and the
            heuristic rules only ever take the unsplit 45' break (8.3
            no-split sensitivity axis).

    Raises
    ------
    AssertionError if C, K, L are not disjoint or do not cover {1..N-1}.
    """
    N    = max(I)
    L    = list(L) if L is not None else []

    assert set(C) | set(K) | set(L) == set(I) - {0, N}, (
        "C ∪ K ∪ L must equal the intermediate stops {1..N-1}")
    assert not (set(C) & set(K)), "C and K must be disjoint"
    assert not ((set(C) | set(K)) & set(L)), "L must be disjoint from C and K"

    # M9: instance-generation guard — each route must fit between two weekly
    # rests, so weekly driving caps never bind (paper §3.4(a)).
    _D_total = sum(D.get(i, 0.0) for i in range(N))
    assert _D_total <= TWK_DRV + 1e-9, (
        f"total nominal driving {_D_total:.1f}h exceeds the weekly cap "
        f"{TWK_DRV}h — regenerate the instance (M9)")

    # Queue times at CS stops (lognormal distribution)
    mu    = np.log(QUEUE_WAIT_MEAN_MIN**2 / np.sqrt(QUEUE_WAIT_STD_MIN**2 + QUEUE_WAIT_MEAN_MIN**2))
    sigma = np.sqrt(np.log(1 + (QUEUE_WAIT_STD_MIN / QUEUE_WAIT_MEAN_MIN)**2))
    _rng  = rng if rng is not None else np.random.default_rng()
    Q_nom = dict(Q) if Q is not None else {
        i: _rng.lognormal(mu, sigma) / 60 for i in K
    }

    # Maneuver overhead at CS stops.
    # M_stop: incurred whenever any activity occurs at a CS (charging, break, or rest).
    # M_seq:  additional repositioning in sequential mode (truck vacates charging bay
    #         before the break begins).
    M_man  = {i: float(M_man_h) for i in range(N + 1)}   # kept for compat
    M_stop = {i: M_STOP_H for i in K}
    M_seq  = {i: M_SEQ_H  for i in K}
    _M_lay_h = M_LAYBY_H if M_lay_h is None else float(M_lay_h)
    M_lay  = {i: _M_lay_h for i in L}    # parking overhead at layby stops (M8)

    S    = dict(S) if S is not None else {c: SERVICE_TIME_H for c in C}
    E0   = Bcap
    Ecap = Bcap
    Emin = SOC_MIN_FRAC * Bcap
    Ebar = {r: EBAR_FRACS[r] * Bcap for r in EBAR_FRACS}
    # I2: rescale the PWL charging-time breakpoints to the requested power class
    Tbar = (scale_tbar(charger_power_kw) if charger_power_kw else dict(TBAR))

    R    = sorted(Ebar.keys())
    Rseg = R[1:]

    T_hor = _T_START + 7 * 24   # 7-day planning horizon

    km_dict = km if km is not None else dict(E)

    lb_t, ub_t = compute_time_bounds(
        I, C, K, D, S, Q_nom, Tbar, T_hor,
        t0=_T_START,
        Man_default=float(M_man_h),
    )

    # Time windows: convert from relative (hours-since-T_START) to absolute
    Wha_abs = {k: v + _T_START for k, v in Wha.items()}
    Whf_abs = {k: v + _T_START for k, v in Whf.items()}

    # C1 — horizon big-M (valid upper bound on the route-duration span)
    H_bigM = compute_horizon_bigM(N, D, S, Q_nom, M_stop, Tr1)

    data = dict(
        label=label, title=title,
        N=N, I=I, C=C, K=K, L=L, R=R, Rseg=Rseg,
        Q=Q_nom, M=M_man, M_stop=M_stop, M_seq=M_seq, M_lay=M_lay,
        D=D, E=E, km=km_dict, S=S,
        E0=E0, Ecap=Ecap, Emin=Emin,
        Ebar=Ebar, Tbar=Tbar,
        Wha=Wha_abs, Whf=Whf_abs,
        T_dead=T_dead,          # retained for I/O compatibility; NOT used as a
                                # constraint (C1: there is no arrival deadline)
        H=H_bigM,               # C1 window / rest big-M
        hard_tw=bool(hard_tw), beta=float(beta_tw),
        allow_wait=bool(allow_wait), wtd_rules=bool(wtd_rules),
        allow_split=bool(allow_split),
        T_hor=T_hor, T_START=_T_START,
        lb_t=lb_t, ub_t=ub_t,
        # Break / rest minimum durations (EU Regulation 561/2006)
        Tb45=Tb45, Tb15=Tb15, Tb30=Tb30,
        Tr1=Tr1,   Tr2=Tr2,
        # HoS accumulator limits
        Tdrv_cons=Tdrv_cons,   Tdrv_sh1=Tdrv_sh1,   Tdrv_sh2=Tdrv_sh2,
        Twrk_cons1=Twrk_cons1, Twrk_cons2=Twrk_cons2, Twrk_sh=Twrk_sh,
        # M5 shift spread limits / M9 weekly caps and exception budgets
        Tspr1=T_SPR1, Tspr2=T_SPR2, Twk60=TWK_60,
        rho_bar=RHO_BAR, ext_bar=EXT_BAR,
        # Big-M constants for HoS linearisation (must be ≥ respective limit).
        # M_sd covers the extended 10 h shift (M6); M_sw and M_h cover the
        # 15 h spread ceiling (M5) which bounds both sw and h.
        M_drv=Tdrv_cons, M_sd=Tdrv_sh2, M_sw=T_SPR2, M_h=T_SPR2, M_big=1000.0,
    )
    return data


# ══════════════════════════════════════════════════════════════════════════════
# LAYBY INSERTION (M8)
# ══════════════════════════════════════════════════════════════════════════════

def insert_laybys(I, C, K, D, km,
                  spacing_km: float = LAYBY_SPACING_KM,
                  min_leg_h: float = LAYBY_MIN_LEG_H):
    """
    M8 — Insert layby (rest-area) nodes along long legs.

    On every leg with nominal duration > `min_leg_h`, layby nodes are inserted
    at (roughly) every `spacing_km` km, splitting the leg into equal segments.
    Laybys allow breaks/rests but no charging and no service.

    Parameters use the ORIGINAL indexing; the function reindexes everything.

    Returns
    -------
    (I2, C2, K2, L2, D2, E2, km2) — reindexed stop sets and leg dicts.
    Leg energies E2 are recomputed from the segment length at the leg's
    implied nominal speed.
    """
    N     = max(I)
    C_set = set(C)
    K_set = set(K)

    C2, K2, L2 = [], [], []
    D2, E2, km2 = {}, {}, {}
    idx = 0
    for i in range(N):
        # classify original stop i under the new numbering
        if i in C_set:
            C2.append(idx)
        elif i in K_set:
            K2.append(idx)

        leg_km = km.get(i, D.get(i, 0.0) * V_NOM)
        leg_h  = D.get(i, 0.0)
        v_leg  = leg_km / leg_h if leg_h > 1e-9 else V_NOM

        if leg_h > min_leg_h and leg_km > spacing_km:
            n_lay  = max(1, int(math.ceil(leg_km / spacing_km)) - 1)
            seg_km = leg_km / (n_lay + 1)
            for j in range(n_lay + 1):
                D2[idx]  = seg_km / v_leg
                km2[idx] = seg_km
                E2[idx]  = seg_km * _ecr(v_leg)
                idx += 1
                if j < n_lay:
                    L2.append(idx)
        else:
            D2[idx]  = leg_h
            km2[idx] = leg_km
            E2[idx]  = leg_km * _ecr(v_leg)
            idx += 1

    I2 = list(range(idx + 1))
    return I2, C2, K2, L2, D2, E2, km2


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK INSTANCES
# ══════════════════════════════════════════════════════════════════════════════

def instance_realistic(route_class: str = "medium",
                       clusters: int = 3,
                       customers_class: str = "few",
                       rng: np.random.Generator | None = None,
                       cs_spacing_km: float | None = None,
                       charger_power_kw: float | None = None,
                       add_laybys: bool = True,
                       layby_spacing_km: float = LAYBY_SPACING_KM) -> dict:
    """
    Randomly generated long-haul route with realistic geometry.

    The route is built by placing CS stops every `CS_spacing` km along a
    corridor of length `route_distance`, then inserting customer stops drawn
    from Gaussian clusters along the route.  Leg energies are computed using
    ECR(v_nom) so that scenario generation can consistently vary speed.

    Parameters
    ----------
    route_class      : "short" (800–1200 km) | "medium" (1500–2500 km)
                       | "long" (3000–4000 km)
    clusters         : 1 | 2 | 3 — number of customer delivery clusters
    customers_class  : "few" (1–3) | "medium" (4–6) | "many" (7–15)

    Notes
    -----
    Call random.seed() before this function for a reproducible instance.
    The title encodes the parameter choices:
      realistic_{route_class}_{customers_class}_{clusters}
    """
    distances = {"short": [800, 1200], "medium": [1500, 2500], "long": [3000, 4000]}
    customers = {"few": (1, 3), "medium": (4, 6), "many": (7, 15)}
    average_speed    = V_NOM
    CS_spacing       = int(cs_spacing_km) if cs_spacing_km else CS_SPACING_KM
    Battery_capacity = BATTERY_CAPACITY

    nb_customers   = random.randint(*customers[customers_class])
    route_distance = random.randint(*distances[route_class])

    # Cluster centres spread evenly along the route
    if clusters == 1:
        cluster_centers = [random.randint(int(0.5 * route_distance),
                                          int(0.6 * route_distance))]
    elif clusters == 2:
        cluster_centers = [
            random.randint(int(0.35 * route_distance), int(0.45 * route_distance)),
            random.randint(int(0.55 * route_distance), int(0.65 * route_distance)),
        ]
    else:
        cluster_centers = [
            random.randint(int(0.25 * route_distance), int(0.30 * route_distance)),
            random.randint(int(0.50 * route_distance), int(0.55 * route_distance)),
            random.randint(int(0.70 * route_distance), int(0.75 * route_distance)),
        ]

    customer_locations = sorted(
        random.choice(cluster_centers) + random.randint(-75, 75) + 0.5
        for _ in range(nb_customers)
    )

    I = [0]; C = []; K = []
    D = {0: CS_spacing / average_speed}
    E = {0: CS_spacing * _ecr(average_speed)}
    I_nb = 1; cur_c = 0; prev_cs = 0

    for dist in range(CS_spacing, route_distance, CS_spacing):
        real = dist + random.randint(-19, 19)
        prev_stop = prev_cs
        # Insert any customer stops between the last CS and this one
        while (cur_c < len(customer_locations) and
               prev_cs < customer_locations[cur_c] < real):
            I.append(I_nb); C.append(I_nb)
            _km = customer_locations[cur_c] - prev_stop
            D[I_nb] = _km / average_speed
            E[I_nb] = _km * _ecr(average_speed)
            I_nb += 1; prev_stop = customer_locations[cur_c]; cur_c += 1
        I.append(I_nb); K.append(I_nb)
        _km = real - prev_stop
        D[I_nb] = _km / average_speed
        E[I_nb] = _km * _ecr(average_speed)
        I_nb += 1; prev_cs = real

    I.append(I_nb)

    km = {i: average_speed * D[i] for i in D}

    # M8 — optionally insert layby (rest-area) nodes along long legs
    L = []
    if add_laybys:
        I, C, K, L, D, E, km = insert_laybys(
            I, C, K, D, km, spacing_km=layby_spacing_km)

    return make_data(
        I=I, C=C, K=K, L=L, D=D, E=E, km=km, Bcap=Battery_capacity,
        Wha={c: 0        for c in C},
        Whf={c: 20000000 for c in C},
        label="realistic — randomly generated long-haul route",
        title=f"realistic_{route_class}_{customers_class}_{clusters}",
        rng=rng,
        charger_power_kw=charger_power_kw,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

ALL_INSTANCES: dict[str, callable] = {
    "realistic"         : instance_realistic,
}