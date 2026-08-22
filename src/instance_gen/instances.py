"""
instances.py — Route data assembly and instance generators
===========================================================


Parameters notation:
    - I : list[int] — all stop indices, 0..N
    - C : list[int] — customer stop indices
    - K : list[int] — charging station stop indices
    - L : list[int] — layby (rest-area) stop indices
    - D : dict {leg: h} — nominal travel time per leg (hours)
    - S : dict {stop: h} — service time at each customer stop (hours)
    - E : dict {leg: kWh} — nominal energy consumption per leg (kWh)
    - Wha : dict {stop: h} — earliest arrival at customer (relative to T_START)
    - Whf : dict {stop: h} — latest   arrival at customer (relative to T_START)
    - Q : dict {cs_stop: h} — queue times at each CS stop
    - M_stop : dict {stop: h} — manoeuver time at stops (charging, break, or rest)
    - M_seq : dict {cs_stop: h} — additional manoeuver time at CS stops in sequential mode
    - Tbar : dict {r: h} — PWL charging curve cumulative-time breakpoints
    - t0 : float — departure time (absolute hours)
    - E0 : float — initial battery SOC (kWh)
    - Ecap : float — battery capacity (kWh)
    - Emin : float — minimum allowed SOC (kWh)
    - Ebar : dict {r: kWh} — PWL charging curve energy breakpoints
    - T_hor : float — planning horizon (absolute hours)
"""

from __future__ import annotations

import math

import numpy as np

from src.settings import (
    ecr as _ecr,
    V_NOM, BATTERY_CAPACITY, SOC_MIN_FRAC, EBAR_FRACS,
    Tb45, Tb15, Tb30, ALLOW_SPLIT_BREAK, Tr1, Tr2,
    Tdrv_cons, Tdrv_sh1, Tdrv_sh2, Twrk_cons1, Twrk_cons2, Twrk_sh,
    T_SPR1, T_SPR2, TWK_60, TWK_DRV, RHO_BAR, EXT_BAR, BETA_TW,
    QUEUE_WAIT_MEAN_MIN, QUEUE_WAIT_STD_MIN,
    M_STOP_H, M_SEQ_H, M_MAN_DEFAULT_H, M_LAYBY_H,
    LAYBY_SPACING_KM,
    SERVICE_TIME_H, CS_SPACING_KM, T_START as _T_START,
    CHARGER_POWER_BASE_KW, charging_curve, XI_MAX,
    CUSTOMERS_PER_CLASS, DISTANCES_CLASS, CLUSTERS_CUSTOMERS,
    CUSTOMERS_SHIFT, CS_SHIFT,
)


# ══════════════════════════════════════════════════════════════════════════════
# TIME BOUNDS
# ══════════════════════════════════════════════════════════════════════════════

def compute_time_bounds(I, C, K, D, S, Q, Tbar, T_hor,
                        t0: float = 0.0,
                        Man_default: float = M_MAN_DEFAULT_H) -> tuple[dict, dict]:
    """
    Returns lb[i], ub[i] — dictionaries mapping stop index to the earliest
    and latest feasible absolute arrival time (hours).
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
    valid big-M H bounding the total route duration (arrival minus t0).
    """
    D_total = sum(D.get(i, 0.0) for i in range(N))
    return (D_total * (1.0 + delta_pad)
            + sum(S.values()) + sum(Q.values()) + sum(M_stop.values())
            + (N + 1) * Tr1)


# ══════════════════════════════════════════════════════════════════════════════
# make_data — DATA DICT CONSTRUCTOR
# ══════════════════════════════════════════════════════════════════════════════

def make_data(I, C, K, L, D, E, km, label, title,
              rng: np.random.Generator | None = None,
              Bcap: float = BATTERY_CAPACITY,
              charger_power_kw: float | None = None,
              Wha: dict | None = None,
              Whf: dict | None = None,
              S: dict | None = None,
              Q: dict | None = None,
              M_man_h: float = M_MAN_DEFAULT_H,
              M_lay_h: float | None = None,
              T_dead: float | None = None,
              hard_tw: bool = False,
              beta_tw: float = BETA_TW,
              allow_wait: bool = False,
              wtd_rules: bool = False,
              allow_split: bool = ALLOW_SPLIT_BREAK) -> dict:
    """
    Attach every non-geometric attribute to a route and return the data dict
    consumed by MILP.build_model, MILP.solve_horizon, BEHDV, oracle_solve and
    the simulation.

    The route itself (I, C, K, L, km) comes from create_geometry_instance; the
    per-leg nominal travel time D and energy E are derived from it by the
    caller, because both depend on the assumed cruising speed.

    Everything the caller does not supply is drawn or defaulted here:

    rng     : generator for the CS queue draws.  Pass a seeded one for a
              reproducible instance; None draws from a fresh generator.
    Q       : fixed queue times {cs_stop: h}.  None draws them lognormally.
    S       : per-customer service times.  None applies SERVICE_TIME_H to all.
    Wha/Whf : customer windows, RELATIVE to T_START (they are returned as
              absolute hours).  None means unconstrained, which is what the
              "none" window class and the benchmark loaders want; the windowed
              classes are written afterwards by instance_io.generate_time_windows.
    M_man_h : legacy uniform manoeuvre time, kept because MILP's covering
              inequalities still index the flat "M" dict.
    M_lay_h : layby parking overhead; None keeps M_LAYBY_H.

    Raises AssertionError when C/K/L do not partition the intermediate stops,
    or when the route cannot fit between two weekly rests (M9) — instance_io
    catches the latter and regenerates with an advanced seed.
    """
    # ── geometry sanity ──────────────────────────────────────────────────────
    N = max(I)
    L = list(L) if L is not None else []
    assert set(C) | set(K) | set(L) == set(I) - {0, N}, (
        "C ∪ K ∪ L must equal the intermediate stops {1..N-1}")
    assert not (set(C) & set(K)), "C and K must be disjoint"
    assert not ((set(C) | set(K)) & set(L)), "L must be disjoint from C and K"

    # M9 — each route must fit between two weekly rests, so the weekly driving
    # cap never binds and the model needs no week-level accumulator.
    _D_total = sum(D.get(i, 0.0) for i in range(N))
    assert _D_total <= TWK_DRV + 1e-9, (
        f"total nominal driving {_D_total:.1f}h exceeds the weekly cap "
        f"{TWK_DRV}h — regenerate the instance (M9)")

    # ── per-stop attributes ──────────────────────────────────────────────────
    # Queue times at CS stops (lognormal, mean/std in minutes -> hours)
    _rng  = rng if rng is not None else np.random.default_rng()
    mu    = np.log(QUEUE_WAIT_MEAN_MIN**2
                   / np.sqrt(QUEUE_WAIT_STD_MIN**2 + QUEUE_WAIT_MEAN_MIN**2))
    sigma = np.sqrt(np.log(1 + (QUEUE_WAIT_STD_MIN / QUEUE_WAIT_MEAN_MIN)**2))
    Q_nom = dict(Q) if Q is not None else {
        i: _rng.lognormal(mu, sigma) / 60 for i in K
    }

    # Manoeuvre overheads.  These are three DISJOINT dicts because the model
    # indexes them over disjoint sets:
    #   M_stop  over K  — any activity at a charging station (charge/break/rest)
    #   M_seq   over K  — extra repositioning when the break follows the charge
    #   M_lay   over L  — parking overhead at a layby
    # MILP builds m.Mstop on Kset and m.Mlay on Lset, and BEHDV guards each with
    # is_CS / is_lay, so putting layby stops into M_stop would be ignored by the
    # model and silently drop their overhead.  Merging the two into one
    # "overhead at any non-customer stop" is a model change (Kset -> Kset|Lset
    # in MILP, plus BEHDV) — worth doing, but not something make_data can do
    # on its own.
    M_stop = {i: M_STOP_H for i in K}
    M_seq  = {i: M_SEQ_H  for i in K}
    M_lay  = {i: (M_LAYBY_H if M_lay_h is None else float(M_lay_h)) for i in L}
    M_man  = {i: float(M_man_h) for i in range(N + 1)}   # legacy flat dict

    S_nom = dict(S) if S is not None else {c: SERVICE_TIME_H for c in C}

    # ── vehicle and charge point ─────────────────────────────────────────────
    E0   = Bcap
    Ecap = Bcap
    Emin = SOC_MIN_FRAC * Bcap
    Ebar = {r: EBAR_FRACS[r] * Bcap for r in EBAR_FRACS}
    # The curve is derived from the charge point's rated output AND this
    # instance's pack, so the base case and the power classes share one code
    # path and the taper tracks Bcap instead of assuming 500 kWh.
    Tbar = charging_curve(charger_power_kw or CHARGER_POWER_BASE_KW, Bcap)
    R    = sorted(Ebar.keys())
    Rseg = R[1:]

    # ── horizon, bounds, windows ─────────────────────────────────────────────
    T_hor = _T_START + 7 * 24            # 7-day planning horizon

    lb_t, ub_t = compute_time_bounds(
        I, C, K, D, S_nom, Q_nom, Tbar, T_hor,
        t0=_T_START, Man_default=float(M_man_h),
    )

    # Windows arrive RELATIVE to T_START and are stored ABSOLUTE.  Unconstrained
    # is the default: instance_io.generate_time_windows overwrites these in
    # place for the tight/medium/large classes.
    Wha_rel = dict(Wha) if Wha is not None else {c: 0.0 for c in C}
    Whf_rel = dict(Whf) if Whf is not None else {c: 2e7 for c in C}
    Wha_abs = {k: v + _T_START for k, v in Wha_rel.items()}
    Whf_abs = {k: v + _T_START for k, v in Whf_rel.items()}

    # C1 — horizon big-M (valid upper bound on the route-duration span)
    H_bigM = compute_horizon_bigM(N, D, S_nom, Q_nom, M_stop, Tr1)

    data = dict(
        label=label, title=title,
        N=N, I=I, C=C, K=K, L=L, R=R, Rseg=Rseg,
        Q=Q_nom, M=M_man, M_stop=M_stop, M_seq=M_seq, M_lay=M_lay,
        D=D, E=E, km=(dict(km) if km is not None else dict(E)), S=S_nom,
        E0=E0, Ecap=Ecap, Emin=Emin,
        Ebar=Ebar, Tbar=Tbar,
        Wha=Wha_abs, Whf=Whf_abs,
        T_dead=T_dead,          # retained for I/O; NOT enforced as a
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
        # Big-M constants for HoS linearisation (must be >= respective limit).
        # M_sd covers the extended 10 h shift (M6); M_sw and M_h cover the
        # 15 h spread ceiling (M5), which bounds both sw and h.
        M_drv=Tdrv_cons, M_sd=Tdrv_sh2, M_sw=T_SPR2, M_h=T_SPR2, M_big=1000.0,
    )
    return data


def create_geometry_instance(route_class: str = "medium",
                             clusters: int = 3,
                             customers_class: str = "few",
                             rng: np.random.Generator | None = None,
                             cs_spacing_km: float = CS_SPACING_KM,
                             layby: bool = True,
                             layby_spacing_km: float = LAYBY_SPACING_KM):
    """Build the route geometry of a given instance.

    Returns
    -------
    I  : list[int]          all node indices, in route order
    C  : list[int]          indices that are customers
    K  : list[int]          indices that are charging stations
    L  : list[int]          indices that are laybys
    KM : dict[int, float]   KM[i] = distance (km) from node i to node i+1
    """

    rng = rng if rng is not None else np.random.default_rng()

    nb_customers   = int(rng.integers(*CUSTOMERS_PER_CLASS[customers_class],
                                      endpoint=True))
    route_distance = int(rng.integers(*DISTANCES_CLASS[route_class],
                                      endpoint=True))

    # Cluster centres spread evenly along the route
    cluster_centers = [
        int(rng.integers(int(CLUSTERS_CUSTOMERS[clusters][c][0] * route_distance),
                         int(CLUSTERS_CUSTOMERS[clusters][c][1] * route_distance),
                         endpoint=True))
        for c in range(clusters)
    ]
    # Customers are placed around a randomly chosen cluster centre.  The +0.5
    # keeps every customer off the integer grid the charging stations sit on,
    # so the two never collide exactly.
    customer_locations = sorted(
        float(rng.choice(cluster_centers))
        + int(rng.integers(-CUSTOMERS_SHIFT, CUSTOMERS_SHIFT, endpoint=True))
        + 0.5
        for _ in range(nb_customers)
    )

    # --- 1. collect (position, type) nodes ---------------------------------
    nodes: list[tuple[float, str]] = [(0.0, "depot")]

    for x in customer_locations:
        nodes.append((float(x), "customer"))

    # charging stations: k * spacing +/- shift, for k = 1 .. floor(D/spacing)
    n_cs = int(route_distance // cs_spacing_km)
    for k in range(1, n_cs + 1):
        pos = k * cs_spacing_km + int(rng.integers(-CS_SHIFT, CS_SHIFT,
                                                   endpoint=True))
        pos = min(max(pos, 0.0), route_distance)
        if any(t == "customer" and abs(pos - p) < 1 for p, t in nodes):
            pos -= 1 # don't drop a CS on top of a fixed customer
        nodes.append((pos, "cs"))

    nodes.append((float(route_distance), "depot"))

    # --- 2. sort by position ----------------------------------------------
    nodes.sort(key=lambda t: t[0])

    # --- 3. insert laybys so every gap <= layby_spacing_km ----------------
    if layby:
        filled: list[tuple[float, str]] = [nodes[0]]
        for p1, t1 in nodes[1:]:
            p0 = filled[-1][0]
            gap = p1 - p0
            if gap > layby_spacing_km:
                n_seg = math.ceil(gap / layby_spacing_km)   # >= 2
                step = gap / n_seg
                for j in range(1, n_seg):
                    filled.append((p0 + j * step, "layby"))
            filled.append((p1, t1))
        nodes = filled

    # --- 4. index & build outputs -----------------------------------------
    I, C, K, L = [], [], [], []
    for idx, (_, typ) in enumerate(nodes):
        I.append(idx)
        if typ == "customer":
            C.append(idx)
        elif typ == "cs":
            K.append(idx)
        elif typ == "layby":
            L.append(idx)

    KM = {idx: nodes[idx + 1][0] - nodes[idx][0] for idx in range(len(nodes) - 1)}

    return I, C, K, L, KM




# ══════════════════════════════════════════════════════════════════════════════
# GENERATED INSTANCES
# ══════════════════════════════════════════════════════════════════════════════

def instance_realistic(route_class: str = "medium",
                       clusters: int = 3,
                       customers_class: str = "few",
                       rng: np.random.Generator | None = None,
                       cs_spacing_km: float | None = None,
                       charger_power_kw: float | None = None,
                       battery_kwh: float | None = None,
                       add_laybys: bool = True,
                       layby_spacing_km: float = LAYBY_SPACING_KM) -> dict:
    """Randomly generated long-haul route with realistic geometry.

    Geometry comes from create_geometry_instance; this function only adds the
    speed-dependent leg quantities and hands everything to make_data.

    Windows are left UNCONSTRAINED here — instance_io.generate_time_windows
    writes the tight/medium/large classes afterwards, because centring them
    needs a nominal MILP solve of the finished data dict.

    Parameters
    ----------
    rng              : seeded generator; drives geometry, queue draws and,
                       downstream in instance_io, the realisation and windows.
    cs_spacing_km    : charging-station spacing; None keeps CS_SPACING_KM.
    charger_power_kw : rated charge-point output; None keeps the base curve.
    battery_kwh      : pack capacity; None keeps BATTERY_CAPACITY.  make_data
                       derives Emin, Ebar and Tbar from it, so the SOC floor
                       scales WITH the pack and the tail acceptance
                       (TAIL_C_RATE*Ecap) moves too — a bigger pack therefore
                       also shifts where the charge curve tapers.
    """
    rng     = rng if rng is not None else np.random.default_rng()
    spacing = float(cs_spacing_km) if cs_spacing_km else CS_SPACING_KM
    Bcap    = float(battery_kwh) if battery_kwh else BATTERY_CAPACITY

    I, C, K, L, km = create_geometry_instance(
        route_class=route_class,
        clusters=clusters,
        customers_class=customers_class,
        rng=rng,
        cs_spacing_km=spacing,
        layby=add_laybys,
        layby_spacing_km=layby_spacing_km,
    )

    # Nominal travel time and energy per leg, both at the cruising speed.
    # ecr() is kWh per KM, so the energy of a leg is ecr(v) * km — not
    # ecr(v) * D, which would be out by a factor of v.
    D = {i: km[i] / V_NOM for i in km}
    E = {i: _ecr(V_NOM) * km[i] for i in km}

    return make_data(
        I=I, C=C, K=K, L=L, D=D, E=E, km=km,
        label="realistic — randomly generated long-haul route",
        title=f"realistic_{route_class}_{customers_class}_{clusters}",
        rng=rng,
        Bcap=Bcap,
        charger_power_kw=charger_power_kw,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
# Consumed by the __main__ blocks of MILP.py and Simulation.py, which solve a
# one-off instance by name.

ALL_INSTANCES: dict[str, callable] = {
    "realistic": instance_realistic,
}
