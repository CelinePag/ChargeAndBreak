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
  Formula: ECR(v) = A/v + B + C·v²  (kWh/km),  A=33.055, B=−0.257, C=7.2e−5
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# ECR CURVE
# ══════════════════════════════════════════════════════════════════════════════

# Constants must match _ecr() in scenarios.py and Simulation.py exactly.
_ECR_A, _ECR_B, _ECR_C = 33.055, -0.257, 7.2e-5


def _ecr(v: float) -> float:
    """
    Energy consumption rate (kWh/km) at speed v (km/h).

    Piecewise physics model: rolling resistance (A/v), constant drag (B),
    and aerodynamic drag (C·v²).  Clipped to [5, 150] km/h for numerical
    stability at very low or very high speeds.
    """
    v = max(5.0, min(float(v), 150.0))
    return _ECR_A / v + _ECR_B + _ECR_C * v ** 2


# ══════════════════════════════════════════════════════════════════════════════
# TIME BOUNDS
# ══════════════════════════════════════════════════════════════════════════════

def compute_time_bounds(I, C, K, D, S, Q, Tbar, T_hor,
                        t0: float = 0.0,
                        Man_default: float = 5.0 / 60) -> tuple[dict, dict]:
    """
    Forward-pass conservative arrival-time bounds for variable tightening.

    Returns lb[i], ub[i] — dictionaries mapping stop index to the earliest
    and latest feasible absolute arrival time (hours).

    These bounds are used by MILP.build_model and MILP.build_horizon_model to
    set variable lower/upper bounds, which significantly tightens the LP
    relaxation and speeds up HiGHS presolve.

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

    Notes
    -----
    Upper bounds use Tr1=11h (the longest possible rest period) so that ub_t
    is always large enough to accommodate any rest type.  Using Tr2=9h instead
    was a previous bug that caused r1 actions to appear spuriously infeasible
    during presolve.
    """
    N   = max(I)
    TK  = Tbar[max(Tbar)]          # maximum possible charging duration
    Tr1 = 11.0                     # longest rest period (EU 561/2006)
    Tb  = 0.75                     # longest break (b45)
    C_s = set(C)
    K_s = set(K)

    lb = {0: t0}
    ub = {0: t0}
    for i in range(N):
        if i in C_s:
            dmin = S.get(i, 0.0)
            dmax = S.get(i, 0.0) + Tb + Tr1 + Man_default + 0.1
        elif i in K_s:
            dmin = 0.0
            dmax = Q.get(i, 0.0) + TK + Tb + Tr1 + Man_default + 0.1
        else:
            dmin = dmax = 0.0
        lb[i + 1] = lb[i] + dmin + D.get(i, 0.0)
        ub[i + 1] = min(T_hor, ub[i] + dmax + D.get(i, 0.0))
    return lb, ub


# ══════════════════════════════════════════════════════════════════════════════
# make_data — CANONICAL DATA DICT CONSTRUCTOR
# ══════════════════════════════════════════════════════════════════════════════

def make_data(I, C, K, D, E, S, E0, Ecap, Emin,
              Ebar, Tbar, Wha, Whf, label, title,
              km=None, Q=None, M_man_h: float = 15.0 / 60) -> dict:
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
            When None, drawn uniformly from U[0, 10] min using the current
            global random state; call random.seed() first for reproducibility.
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

    Raises
    ------
    AssertionError if C and K are not disjoint or do not cover {1..N-1}.
    """
    N    = max(I)
    R    = sorted(Ebar.keys())
    Rseg = R[1:]
    assert set(C) | set(K) == set(I) - {0, N}, (
        "C ∪ K must equal the intermediate stops {1..N-1}")
    assert not (set(C) & set(K)), "C and K must be disjoint"

    # Queue times at CS stops
    Q_nom = dict(Q) if Q is not None else {
        i: random.randint(0, 10) / 60 for i in K
    }

    # Manoeuver time per active stop.
    # Default 15 min; override via M_man_h to penalise off-CS breaks/rests.
    # Note: lb_t / ub_t use Man_default so that MILP variable bounds remain
    # consistent with the manoeuver time used in the model.
    M_man = {i: float(M_man_h) for i in range(N + 1)}

    T_START = 8.0                   # 08:00 departure (absolute hours)
    T_hor   = T_START + 5 * 24     # 5-day planning horizon

    km_dict = km if km is not None else dict(E)

    lb_t, ub_t = compute_time_bounds(
        I, C, K, D, S, Q_nom, Tbar, T_hor,
        t0=T_START,
        Man_default=float(M_man_h),
    )

    # Time windows: convert from relative (hours-since-T_START) to absolute
    Wha_abs = {k: v + T_START for k, v in Wha.items()}
    Whf_abs = {k: v + T_START for k, v in Whf.items()}

    return dict(
        label=label, title=title,
        N=N, I=I, C=C, K=K, R=R, Rseg=Rseg,
        Q=Q_nom, M=M_man,
        D=D, E=E, km=km_dict, S=S,
        E0=E0, Ecap=Ecap, Emin=Emin,
        Ebar=Ebar, Tbar=Tbar,
        Wha=Wha_abs, Whf=Whf_abs,
        T_hor=T_hor, T_START=T_START,
        lb_t=lb_t, ub_t=ub_t,
        # Break / rest minimum durations (EU Regulation 561/2006)
        Tb45=0.75, Tb15=0.25, Tb30=0.50,
        Tr1=11.0,  Tr2=9.0,
        # HoS accumulator limits
        Tdrv_cons=4.5,   Tdrv_sh1=9.0,   Tdrv_sh2=10.0,
        Twrk_cons1=6.0,  Twrk_cons2=9.0, Twrk_sh=13.0,
        # Big-M constants for HoS linearisation (must be ≥ respective limit)
        M_drv=4.5, M_sd=10.0, M_sw=13.0, M_big=1000.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK INSTANCES
# ══════════════════════════════════════════════════════════════════════════════

def instance_tiny() -> dict:
    """
    Minimal 5-stop route (4 legs).
    Tests basic SOC propagation and time feasibility on a small problem.
    """
    N = 4
    return make_data(
        I=list(range(N + 1)), C=[1], K=[2, 3],
        D={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
        E={0: 8.0, 1: 8.0, 2: 8.0, 3: 8.0},
        S={1: 0.5},
        E0=60, Ecap=100, Emin=10,
        Ebar={0: 0, 1: 40, 2: 80, 3: 100},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={1: 0}, Whf={1: 5},
        label="tiny — 5 stops, basic SOC + timing check",
        title="tiny",
    )


def instance_break_forced() -> dict:
    """
    10-stop route where the 4.5h consecutive driving limit is tight.
    All legs are 1h → cd reaches 4.5h exactly after 4.5 legs, so a
    b45 break must be inserted somewhere within the first 4 CS stops.
    """
    N = 10
    C, K = [2, 7], [1, 3, 4, 5, 6, 8, 9]
    return make_data(
        I=list(range(N + 1)), C=C, K=K,
        D={i: 1.0 for i in range(N)},
        E={i: 7.0 for i in range(N)},
        S={2: 0.5, 7: 0.5},
        E0=90, Ecap=100, Emin=10,
        Ebar={0: 0, 1: 40, 2: 80, 3: 100},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={2: 0, 7: 0}, Whf={2: 20, 7: 20},
        label="break_forced — 10 stops, 4.5h driving limit binds",
        title="break_forced",
    )


def instance_charging_needed() -> dict:
    """
    8-stop route with high energy consumption (22 kWh/leg).
    Battery would deplete without at least one charge stop; tests that
    the MILP selects a charging action and the simulation follows.
    """
    N = 8
    C, K = [2, 6], [1, 3, 4, 5, 7]
    return make_data(
        I=list(range(N + 1)), C=C, K=K,
        D={i: 1.0 for i in range(N)},
        E={i: 22.0 for i in range(N)},
        S={2: 0.5, 6: 0.5},
        E0=80, Ecap=100, Emin=10,
        Ebar={0: 0, 1: 40, 2: 80, 3: 100},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={2: 0, 6: 0}, Whf={2: 20, 6: 20},
        label="charging_needed — 8 stops, high consumption forces charging",
        title="charging_needed",
    )


def instance_rest_forced() -> dict:
    """
    14-stop route where the 9h shift-driving limit forces a daily rest.
    Legs of 1h each → after 9 legs without rest, sd=9h exactly.
    """
    N = 14
    C = [3, 8, 12]
    K = [1, 2, 4, 5, 6, 7, 9, 10, 11, 13]
    return make_data(
        I=list(range(N + 1)), C=C, K=K,
        D={i: 1.0 for i in range(N)},
        E={i: 7.0 for i in range(N)},
        S={3: 0.5, 8: 0.5, 12: 0.5},
        E0=90, Ecap=100, Emin=10,
        Ebar={0: 0, 1: 40, 2: 80, 3: 100},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={3: 0, 8: 0, 12: 0}, Whf={3: 30, 8: 30, 12: 30},
        label="rest_forced — 14 stops, 9h shift limit forces daily rest",
        title="rest_forced",
    )


def instance_3day() -> dict:
    """
    34-stop three-day route with 10 customers and 23 CS stops.
    Requires 3 mandatory daily rests and multiple charges.
    Representative of a realistic multi-day long-haul mission.
    """
    N = 34
    C = [3, 7, 11, 15, 19, 22, 25, 28, 30, 32]
    K = [i for i in range(1, N) if i not in C]
    return make_data(
        I=list(range(N + 1)), C=C, K=K,
        D={i: 1.0  for i in range(N)},
        E={i: 8.0  for i in range(N)},
        S={c: 0.75 for c in C},
        E0=90, Ecap=100, Emin=10,
        Ebar={0: 0, 1: 40, 2: 80, 3: 100},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={c: 0   for c in C},
        Whf={c: 200 for c in C},
        label="3-day — 34 legs, 10 customers, 23 CS, 3 mandatory rests",
        title="3day",
    )


def instance_realistic(route_class: str = "medium",
                       clusters: int = 3,
                       customers_class: str = "few") -> dict:
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
    customers_class  : "few" (1–3) | "medium" (4–5) | "many" (6–15)

    Notes
    -----
    Call random.seed() before this function for a reproducible instance.
    The title encodes the parameter choices:
      realistic_{route_class}_{customers_class}_{clusters}
    """
    distances = {"short": [800, 1200], "medium": [1500, 2500], "long": [3000, 4000]}
    customers = {"few": (1, 3), "medium": (4, 5), "many": (6, 15)}
    average_speed    = 80        # km/h nominal highway speed
    CS_spacing       = 40        # km between consecutive CS stops
    Battery_capacity = 350       # kWh

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
        random.choice(cluster_centers) + random.randint(-75, 75)
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
    print(f"Route: {route_distance} km, {len(C)} customers, {len(K)} CS")

    km = {i: average_speed * D[i] for i in D}
    Bcap = Battery_capacity
    return make_data(
        I=I, C=C, K=K, D=D, E=E, km=km,
        S={c: 0.5 for c in C},
        E0=Bcap, Ecap=Bcap, Emin=0.2 * Bcap,
        Ebar={0: 0, 1: 0.40 * Bcap, 2: 0.80 * Bcap, 3: Bcap},
        Tbar={0: 0.0, 1: 0.55, 2: 1.367, 3: 2.50},
        Wha={c: 0        for c in C},
        Whf={c: 20000000 for c in C},
        label="realistic — randomly generated long-haul route",
        title=f"realistic_{route_class}_{customers_class}_{clusters}",
    )


# ── Targeted edge-case instances ──────────────────────────────────────────────

def instance_split_break() -> dict:
    """
    Forces the b15→b30 split-break sequence.

    Leg sizes are chosen so that cd approaches 4.5h but stays under even
    with δ=20%: max drawn cd = (1.5+1.4)×1.2 = 3.48h < 4.5h.
    A b15 at stop 1 sets phi=1, enabling b30 at stop 3.
    """
    return make_data(
        I=[0, 1, 2, 3, 4], C=[2], K=[1, 3],
        D={0: 1.5, 1: 0.4, 2: 1.4, 3: 0.7},
        E={0: 45,  1: 12,  2: 42,  3: 20},
        S={2: 0.5},
        E0=200, Ecap=200, Emin=40,
        Ebar={0: 0, 1: 80, 2: 160, 3: 200},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={2: 0}, Whf={2: 1e7},
        label="split_break — forces b15+b30 sequence",
        title="split_break",
    )


def instance_phi_inherited() -> dict:
    """
    Start mid-route with phi=1 (b15 already taken in a previous window).
    The b30 option must be available at the first sub-problem stop.
    Tests that phi is correctly propagated through init_state in solve_horizon.
    """
    return make_data(
        I=[0, 1, 2, 3], C=[], K=[1, 2],
        D={0: 1.5, 1: 1.0, 2: 0.8},
        E={0: 45,  1: 30,  2: 25},
        S={},
        E0=150, Ecap=200, Emin=30,
        Ebar={0: 0, 1: 80, 2: 160, 3: 200},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={}, Whf={},
        label="phi_inherited — b30 available from start",
        title="phi_inherited",
    )


def instance_rho2_budget() -> dict:
    """
    Route requiring four daily rests.  The first three may use r2 (9h);
    the fourth must use r1 (11h) once the rho2_used budget is exhausted.
    Tests that the rho2_used counter is tracked and the budget enforced.
    """
    shift_d = 8.5 / 3
    C = [3, 6, 9]
    K = [1, 2, 4, 5, 7, 8, 10, 11]
    return make_data(
        I=list(range(13)), C=C, K=K,
        D={i: shift_d for i in range(12)},
        E={i: shift_d * 30 for i in range(12)},
        S={3: 0.5, 6: 0.5, 9: 0.5},
        E0=350, Ecap=350, Emin=50,
        Ebar={0: 0, 1: 140, 2: 280, 3: 350},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={c: 0   for c in C},
        Whf={c: 1e7 for c in C},
        label="rho2_budget — 4 rests, first 3 can be r2",
        title="rho2_budget",
    )


def instance_tight_energy_chain() -> dict:
    """
    Battery must charge at every CS stop: each leg consumes 85% of usable
    capacity (136 kWh), so skipping any CS charge causes energy infeasibility.

    NOTE: use delta=0 only (listed in DET_ONLY_INSTANCES).
    With δ>0 the drawn energy may exceed the usable capacity within a single
    leg, which no policy can recover from.
    """
    Ecap = 200; Emin = 40
    E_leg = (Ecap - Emin) * 0.85
    D_leg = round(E_leg / 100, 3)
    N_legs = 5
    return make_data(
        I=list(range(N_legs + 1)), C=[], K=list(range(1, N_legs)),
        D={i: D_leg for i in range(N_legs)},
        E={i: E_leg for i in range(N_legs)},
        S={},
        E0=Ecap, Ecap=Ecap, Emin=Emin,
        Ebar={0: 0, 1: 80, 2: 160, 3: 200},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={}, Whf={},
        label="tight_energy_chain — must charge at every CS",
        title="tight_energy",
    )


def instance_sd_boundary() -> dict:
    """
    Route where shift driving approaches 9h, forcing a preemptive rest.
    Three legs of 2.9h each push sd to 8.7h; the fourth leg would
    exceed the 9h limit, so a rest must be inserted before it.

    NOTE: use delta=0 only (listed in DET_ONLY_INSTANCES).
    With δ=20%: max sd = 2.9×1.2×3 = 10.44h which exceeds the limit
    mid-leg — something no look-ahead policy can avoid.
    """
    return make_data(
        I=[0, 1, 2, 3, 4, 5], C=[], K=[1, 2, 3, 4],
        D={0: 2.9, 1: 2.9, 2: 2.9, 3: 0.4, 4: 0.4},
        E={0: 87,  1: 87,  2: 87,  3: 12,  4: 12},
        S={},
        E0=350, Ecap=350, Emin=50,
        Ebar={0: 0, 1: 140, 2: 280, 3: 350},
        Tbar={0: 0.0, 1: 0.55, 2: 1.37, 3: 2.50},
        Wha={}, Whf={},
        label="sd_boundary — sd approaches 9h, rest required",
        title="sd_boundary",
    )


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

ALL_INSTANCES: dict[str, callable] = {
    "tiny"              : instance_tiny,
    "break_forced"      : instance_break_forced,
    "charging_needed"   : instance_charging_needed,
    "rest_forced"       : instance_rest_forced,
    "3day"              : instance_3day,
    "realistic"         : instance_realistic,
    "split_break"       : instance_split_break,
    "phi_inherited"     : instance_phi_inherited,
    "rho2_budget"       : instance_rho2_budget,
    "tight_energy_chain": instance_tight_energy_chain,
    "sd_boundary"       : instance_sd_boundary,
}

# These instances are only meaningful with delta=0 (noise would violate
# the constraints mid-leg, which no policy can prevent).
DET_ONLY_INSTANCES: set[str] = {"tight_energy_chain", "sd_boundary"}