"""
Electric Truck Scheduling MILP
================================
Stops indexed 0..N   (origin=0, destination=N).
  C ⊆ {1..N-1} : customer stops
  K ⊆ {1..N-1} : charging station stops  (C ∩ K = ∅, C ∪ K = {1..N-1})

Every interior stop is either a customer or a CS — no transit stops.
All times in HOURS, energy in kWh.
"""

import pyomo.environ as pyo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import json
import os
import time as _time_mod

# Output folders — created on first use
FIGURES_DIR   = "figures"
SOLUTIONS_DIR = "solutions"

def _ensure_dirs():
    os.makedirs(FIGURES_DIR,   exist_ok=True)
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)
import time

# ============================================================
# INSTANCES
# ============================================================

def instance_tiny():
    """
    5 stops total (0..4). 1 customer (stop 1), 2 CS (stops 2,3).
    Short legs, low consumption — no HoS constraint should bind.
    Use to verify: SOC propagation, PWL charging, basic timing.
    Expected: truck charges lightly at one CS, arrives at dest ~2.5h.
    """
    N = 4
    I = list(range(N + 1))
    return _make_data(
        I=I, C=[1], K=[2, 3],
        D={0:0.5, 1:0.5, 2:0.5, 3:0.5},
        E={0:8.0, 1:8.0, 2:8.0, 3:8.0},
        S={1: 0.5},
        E0=60, Ecap=100, Emin=10,
        Ebar={0:0, 1:50, 2:100},
        Tbar={0:0.0, 1:0.8, 2:2.0},
        Wha={1:0}, Whf={1:5},
        label="tiny — 5 stops, basic SOC + timing check",
        title="tiny"
    )

def instance_break_forced():
    """
    11 stops (0..10). 2 customers, 7 CS available.
    Legs of 1h → cumulative driving hits 4.5h limit → break mandatory.
    Many CS available so solver chooses optimal charging + break combo.
    Expected: one 45-min break around stop 5; some charging.
    """
    N = 10
    I = list(range(N + 1))
    C = [2, 7]
    K = [1, 3, 4, 5, 6, 8, 9]   # many CS to choose from
    return _make_data(
        I=I, C=C, K=K,
        D={i: 1.0 for i in range(N)},
        E={i: 7.0 for i in range(N)},
        S={2: 0.5, 7: 0.5},
        E0=90, Ecap=100, Emin=10,
        Ebar={0:0, 1:40, 2:70, 3:100},
        Tbar={0:0.0, 1:0.55, 2:1.10, 3:2.50},
        Wha={2:0, 7:0}, Whf={2:20, 7:20},
        label="break_forced — 10 stops, 4.5h driving limit binds",
        title="break_forced"
    )

def instance_charging_needed():
    """
    9 stops (0..8). 2 customers, 5 CS.
    High consumption (22 kWh/leg) drains 100 kWh battery in ~4 legs.
    PWL constraints are active: solver charges partially at cheap stops.
    Expected: charging at ≥2 CS stops, SOC stays above 10 kWh.
    """
    N = 8
    I = list(range(N + 1))
    C = [2, 6]
    K = [1, 3, 4, 5, 7]
    return _make_data(
        I=I, C=C, K=K,
        D={i: 1.0 for i in range(N)},
        E={i: 22.0 for i in range(N)},
        S={2: 0.5, 6: 0.5},
        E0=80, Ecap=100, Emin=10,
        Ebar={0:0, 1:40, 2:70, 3:100},
        Tbar={0:0.0, 1:0.55, 2:1.10, 3:2.50},
        Wha={2:0, 6:0}, Whf={2:20, 6:20},
        label="charging_needed — 8 stops, high consumption forces charging",
        title="charging_needed"
    )

def instance_rest_forced():
    """
    15 stops (0..14). 3 customers, 10 CS.
    Legs of 1h → 14h total driving far exceeds 9h shift limit.
    A daily rest (11h) is mandatory mid-route.
    Expected: rest around stop 9-10; counters reset; objective ~25h+.
    """
    N = 14
    I = list(range(N + 1))
    C = [3, 8, 12]
    K = [1, 2, 4, 5, 6, 7, 9, 10, 11, 13]
    return _make_data(
        I=I, C=C, K=K,
        D={i: 1.0 for i in range(N)},
        E={i: 7.0 for i in range(N)},
        S={3: 0.5, 8: 0.5, 12: 0.5},
        E0=90, Ecap=100, Emin=10,
        Ebar={0:0, 1:40, 2:70, 3:100},
        Tbar={0:0.0, 1:0.55, 2:1.10, 3:2.50},
        Wha={3:0, 8:0, 12:0},
        Whf={3:30, 8:30, 12:30},
        label="rest_forced — 14 stops, 9h shift limit forces daily rest",
        title="rest_forced"
    )

def instance_3day():
    """
    3-day long-haul route: 35 stops (0..34).
    10 customers spread across the route.
    23 CS stops available — solver picks optimal charging locations.

    Structure:
      - 34 legs of 1h → 34h total driving
      - Shift driving limit (9h) forces 3 mandatory daily rests (~11h each)
      - Battery (100 kWh, 8 kWh/leg) needs ~3 charges over the trip
      - Service time 0.75h per customer (loading/unloading heavy freight)
      - Expected total duration ≈ 34h driving + 33h rest + 7.5h service
                                + breaks ≈ 75–80h (~3.2 days)

    HoS events expected by the solver:
      Day 1: drive ~9h (stops 0–9) → mandatory rest ~stop 9-10
      Day 2: drive ~9h (stops 10–19) → mandatory rest ~stop 19-20
      Day 3: drive ~9h (stops 20–29) → mandatory rest ~stop 29-30
      Day 4 (partial): drive remaining ~7h to destination
    """
    N = 34
    I = list(range(N + 1))
    C = [3, 7, 11, 15, 19, 22, 25, 28, 30, 32]
    K = [i for i in range(1, N) if i not in C]

    return _make_data(
        I=I, C=C, K=K,
        D={i: 1.0  for i in range(N)},   # 1h per leg (highway speed)
        E={i: 8.0  for i in range(N)},   # 8 kWh/leg → ~3 charges needed
        S={c: 0.75 for c in C},           # 45min service per customer
        E0=90, Ecap=100, Emin=10,
        # Realistic CC-CV curve (5 segments):
        # CC phase up to ~80% (fast), CV phase above (slowing)
        Ebar={0: 0,  1: 20, 2: 40, 3: 70, 4: 85, 5: 100},
        Tbar={0: 0.0, 1: 0.27, 2: 0.55, 3: 1.00, 4: 1.55, 5: 2.50},
        Wha={c: 0   for c in C},          # no hard lower bound
        Whf={c: 200 for c in C},          # generous hard upper bound
        label="3-day — 34 legs, 10 customers, 23 CS, 3 mandatory rests",
        title="3day"
    )

def instance_realistic(route_class="medium", clusters=3, customers_class="many", ):

    distances = {"short":[800, 1200], "medium":[1500, 2500], "long":[3000, 4000]}
    customers = {"few": (1, 3), "medium": (4, 5), "many": (6, 15)}

    average_speed = 80 # km/h for a heavy-duty electric truck on highways
    energy_consumption = 1 # kWh/km for a heavy-duty electric truck
    CS_spacing = 60 # km between CS stops

    Battery_capacity = 350 # kWh

    route_distance = random.randint(*distances[route_class])
    nb_customers = random.randint(*customers[customers_class])

    if clusters == 1:
        cluster_centers = [random.randint(int(0.5*route_distance), int(0.6*route_distance))]
    elif clusters == 2:
        cluster_centers = [random.randint(int(0.35*route_distance), int(0.45*route_distance)),
                           random.randint(int(0.55*route_distance), int(0.65*route_distance))]
    elif clusters == 3:
        cluster_centers = [random.randint(int(0.25*route_distance), int(0.30*route_distance)),
                           random.randint(int(0.50*route_distance), int(0.55*route_distance)),
                           random.randint(int(0.70*route_distance), int(0.75*route_distance))]
    else:
        raise ValueError("Invalid number of clusters. Choose 1, 2, or 3.")

    customer_locations = []
    chosen_clusters = []

    for c in range(nb_customers):
        if c < clusters:
            chosen_cluster = cluster_centers[c]
        else:
            chosen_cluster = random.choice(cluster_centers)
        chosen_clusters.append(chosen_cluster)
        customer_location = chosen_cluster + random.randint(-75, 75)
        customer_locations.append(customer_location)

    customer_locations = sorted(customer_locations)
    I = [0]
    C = []
    K = []
    D = {0: CS_spacing / average_speed}
    E = {0: CS_spacing * energy_consumption}
    I_nb = 1
    current_customerindex = 0
    previous_CS_dist = 0
    for dist in range(CS_spacing, route_distance, CS_spacing):
        real_dist = dist + random.randint(-20, 20) # add some randomness to the stop locations
        customer_between = False
        previous_stop_dist = previous_CS_dist
        while current_customerindex <= len(customer_locations) - 1 and real_dist > customer_locations[current_customerindex] > previous_CS_dist:
            customer_between = True
            I.append(I_nb)
            C.append(I_nb)
            D[I_nb] = (customer_locations[current_customerindex] - previous_stop_dist) / average_speed
            E[I_nb] = (customer_locations[current_customerindex] - previous_stop_dist) * energy_consumption
            I_nb += 1
            previous_stop_dist = customer_locations[current_customerindex]
            current_customerindex += 1


        if customer_between:
            I.append(I_nb)
            K.append(I_nb)
            D[I_nb] = (real_dist - previous_stop_dist) / average_speed
            E[I_nb] = (real_dist - previous_stop_dist) * energy_consumption
            I_nb += 1

        if not customer_between:
            I.append(I_nb)
            K.append(I_nb)
            D[I_nb] = (real_dist - previous_CS_dist) / average_speed
            E[I_nb] = (real_dist - previous_CS_dist) * energy_consumption
            I_nb += 1
        previous_CS_dist = real_dist

    I.append(I_nb)  # destination
    print(f"Generated route distance: {route_distance} km")
    print(f"Generated customer locations: {customer_locations}")

    return _make_data(
        I=I, C=C, K=K,
        D=D,   # 1h per leg (highway speed)
        E=E,   # 8 kWh/leg → ~3 charges needed
        S={c: 0.5 for c in C},           # 45min service per customer
        E0=Battery_capacity, Ecap=Battery_capacity, Emin=0.2*Battery_capacity,
        # Realistic CC-CV curve (5 segments):
        # CC phase up to ~80% (fast), CV phase above (slowing)
        Ebar={0: 0,  1: 0.2*Battery_capacity, 2: 0.4*Battery_capacity, 3: 0.7*Battery_capacity, 4: 0.85*Battery_capacity, 5: Battery_capacity},
        Tbar={0: 0.0, 1: 0.27, 2: 0.55, 3: 1.00, 4: 1.55, 5: 2.50},
        Wha={c: 0   for c in C},          # no hard lower bound
        Whf={c: 20000000 for c in C},          # generous hard upper bound
        label="realistic — randomly generated route with realistic parameters",
        title="realistic_" + route_class + "_" + customers_class + "_" + str(clusters)
    )

def _make_data(I, C, K, D, E, S, E0, Ecap, Emin,
               Ebar, Tbar, Wha, Whf, label, title):
    N    = max(I)
    R    = sorted(Ebar.keys())
    Rseg = R[1:]      # segment indices 1..K_pwl (SOS2 selectors)
    assert set(C) | set(K) == set(I) - {0, N}, \
        "Every interior stop must be C or K"
    assert not (set(C) & set(K)), "C and K must be disjoint"
    return dict(
        label=label,
        title=title,
        N=N, I=I, C=C, K=K, R=R, Rseg=Rseg,
        Q={i: random.randint(5, 20)/60 for i in K},
        M= {i: 5/60 for i in K},
        D=D, E=E, S=S,
        E0=E0, Ecap=Ecap, Emin=Emin,
        Ebar=Ebar, Tbar=Tbar,
        Wha=Wha, Whf=Whf,
        Tb45=0.75, Tb15=0.25, Tb30=0.50,
        Tr1=11.0,  Tr2=9.0,
        Tdrv_cons=4.5, Tdrv_sh1=9.0, Tdrv_sh2=10.0,
        Twrk_cons1=6.0, Twrk_cons2=9.0, Twrk_sh=13.0,
        M_drv=4.5,      # tight Big-M: consecutive driving
        M_sd =10.0,     # tight Big-M: shift driving (extended)
        M_sw =13.0,     # tight Big-M: shift working
        M_big=1000.0,
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
    TK    = data["Tbar"][max(R)]   # time to fully charge from empty
    M_drv = data["M_drv"]
    M_sd  = data["M_sd"]
    M_sw  = data["M_sw"]
    M_big = data["M_big"]

    # ---- sets -----------------------------------------------
    m.I     = pyo.Set(initialize=data["I"],    ordered=True)
    m.Cset  = pyo.Set(initialize=C)
    m.Kset  = pyo.Set(initialize=K)
    m.Rset  = pyo.Set(initialize=R,    ordered=True)
    m.RsegS = pyo.Set(initialize=Rseg, ordered=True)
    m.Legs  = pyo.Set(initialize=list(range(N)), ordered=True)  # 0..N-1

    # ---- parameters -----------------------------------------
    m.D    = pyo.Param(m.Legs,  initialize=data["D"])
    m.E    = pyo.Param(m.Legs,  initialize=data["E"])
    m.Q =    pyo.Param(m.Kset, initialize=data["Q"], default=0)
    m.M =    pyo.Param(m.Kset, initialize=data["M"], default=0)
    m.S    = pyo.Param(m.Cset, initialize=data["S"], default=0)
    m.E0   = pyo.Param(initialize=data["E0"])
    m.Ecap = pyo.Param(initialize=data["Ecap"])
    m.Emin = pyo.Param(initialize=data["Emin"])
    m.Ebar = pyo.Param(m.Rset, initialize=data["Ebar"])
    m.Tbar = pyo.Param(m.Rset, initialize=data["Tbar"])
    m.Wha  = pyo.Param(m.Cset, initialize=data["Wha"], default=0)
    m.Whf  = pyo.Param(m.Cset, initialize=data["Whf"], default=1e6)
    m.Tb45 = pyo.Param(initialize=data["Tb45"])
    m.Tb15 = pyo.Param(initialize=data["Tb15"])
    m.Tb30 = pyo.Param(initialize=data["Tb30"])
    m.Tr1  = pyo.Param(initialize=data["Tr1"])
    m.Tr2  = pyo.Param(initialize=data["Tr2"])
    m.Tdrv_cons = pyo.Param(initialize=data["Tdrv_cons"])
    m.Tdrv_sh1  = pyo.Param(initialize=data["Tdrv_sh1"])
    m.Twrk_sh   = pyo.Param(initialize=data["Twrk_sh"])

    # ---- variables ------------------------------------------
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

    m.l1 = pyo.Var(m.I, domain=pyo.NonNegativeReals)  # r_i   * cd[i]
    m.l2 = pyo.Var(m.I, domain=pyo.NonNegativeReals)  # rho_i * sd[i]
    m.l4 = pyo.Var(m.I, domain=pyo.NonNegativeReals)  # rho_i * sw[i]

    m.u = pyo.Var(m.Kset, domain=pyo.NonNegativeReals)  # charging time as work (no break/rest)
    m.w = pyo.Var(m.Kset, domain=pyo.Binary)            # y_i * rho_i: charges AND rests

    # ---- objective ------------------------------------------
    m.obj = pyo.Objective(expr=m.ta[N], sense=pyo.minimize)

    # =========================================================
    # CONSTRAINTS
    # =========================================================

    # ------ initial conditions / fix boundary binaries -------
    m.init_ta  = pyo.Constraint(expr=m.ta[0] == 0)
    m.init_ea  = pyo.Constraint(expr=m.ea[0] == m.E0)
    m.init_cd  = pyo.Constraint(expr=m.cd[0] == 0)
    m.init_sd  = pyo.Constraint(expr=m.sd[0] == 0)
    m.init_sw  = pyo.Constraint(expr=m.sw[0] == 0)
    m.init_phi = pyo.Constraint(expr=m.phi[0] == 0)

    for v in [m.x_b45, m.x_b15, m.x_b30, m.rho1, m.rho2]:
        v[0].fix(0); v[N].fix(0)
    m.taub[0].fix(0); m.taur[0].fix(0)
    m.taub[N].fix(0); m.taur[N].fix(0)

    # ------ time propagation ---------------------------------
    # Leg: ta[i+1] = td[i] + D[i]
    def _tp(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.ta[i+1] == m.td[i] + m.D[i]
    m.time_prop = pyo.Constraint(m.I, rule=_tp)

    # Origin and destination: no dwell
    m.td_orig = pyo.Constraint(expr=m.td[0] == m.ta[0])
    m.td_dest = pyo.Constraint(expr=m.td[N] == m.ta[N])

    # Customer: service + break + rest  (in that order)
    m.td_C = pyo.Constraint(m.Cset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.S[i] + m.taub[i] + m.taur[i])

    # CS: charging + break + rest + Queue
    m.td_K = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.td[i] == m.ta[i] + m.Q[i]*m.y[i] + m.tauc[i] + m.taub[i] + m.taur[i] + m.M[i]*(m.x_b45[i]+m.x_b15[i]+m.x_b30[i]+m.rho1[i]+m.rho2[i]))

    # ------ hard time windows --------------------------------
    m.tw_hard = pyo.Constraint(m.Cset, rule=lambda m, i:
        pyo.inequality(m.Wha[i], m.ta[i], m.Whf[i]))

    # ------ battery SOC --------------------------------------
    def _soc(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.ea[i+1] == m.ed[i] - m.E[i]
    m.soc_prop = pyo.Constraint(m.I, rule=_soc)

    m.soc_nc_orig = pyo.Constraint(expr=m.ed[0] == m.ea[0])
    m.soc_nc_dest = pyo.Constraint(expr=m.ed[N] == m.ea[N])
    m.soc_nc_C = pyo.Constraint(m.Cset,
        rule=lambda m, i: m.ed[i] == m.ea[i])
    m.soc_mono_K = pyo.Constraint(m.Kset,
        rule=lambda m, i: m.ed[i] >= m.ea[i])
    m.soc_lb = pyo.Constraint(m.I, rule=lambda m, i: m.ea[i] >= m.Emin)
    m.soc_ub = pyo.Constraint(m.I, rule=lambda m, i: m.ed[i] <= m.Ecap)
    m.chg_act = pyo.Constraint(m.Kset,
        rule=lambda m, i: m.tauc[i] <= TK * m.y[i])

    m.chg_act2 = pyo.Constraint(m.Kset,
        rule=lambda m, i: m.tauc[i] >= 0.25 * m.y[i]) # au minimum 15min de charge

    print(f"Total energy needed: {sum(data['E'].values())} kWh")
    print(f"Battery capacity: {data['Ecap']} kWh")
    print(f"Minimum number of charges needed (energy/capacity): {sum(data['E'].values())/(0.8*m.Ecap):.1f}")
    #m.nb_stop_charge = pyo.Constraint(expr=lambda m: sum(m.y[i] for i in m.Kset) <= 5 + sum(data["E"].values()) / (0.8*m.Ecap)) # limiter nombre totql de stop charge ?
    # ------ PWL charging (Montoya et al. 2017) ---------------
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

    # SOS2: at most two adjacent lambda weights nonzero
    R_list = sorted(R)
    K_max  = max(Rseg)
    mid_pairs_a = [(i, k) for i in K for k in Rseg[:-1]]
    mid_pairs_d = [(i, k) for i in K for k in Rseg[:-1]]

    m.sos2_lo_a = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.lam_a[i, R_list[0]] <= m.mu_a[i, R_list[1]])
    m.sos2_hi_a = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.lam_a[i, R_list[-1]] <= m.mu_a[i, K_max])
    m.sos2_lo_d = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.lam_d[i, R_list[0]] <= m.mu_d[i, R_list[1]])
    m.sos2_hi_d = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.lam_d[i, R_list[-1]] <= m.mu_d[i, K_max])

    m.sos2_mid_a = pyo.Constraint(mid_pairs_a, rule=lambda m, i, k:
        m.lam_a[i, k] <= m.mu_a[i, k] + m.mu_a[i, k+1])
    m.sos2_mid_d = pyo.Constraint(mid_pairs_d, rule=lambda m, i, k:
        m.lam_d[i, k] <= m.mu_d[i, k] + m.mu_d[i, k+1])

    # ------ qualifying break duration ------------------------
    m.qb_K = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.taub_hat[i] == m.taub[i] + m.tauc[i])
    non_K = [i for i in data["I"] if i not in K]
    m.qb_nonK = pyo.Constraint(non_K, rule=lambda m, i:
        m.taub_hat[i] == m.taub[i])

    # ------ breaks and rests ---------------------------------
    m.one_brk = pyo.Constraint(m.I, rule=lambda m, i:
        m.x_b45[i]+m.x_b15[i]+m.x_b30[i]+m.rho1[i]+m.rho2[i] <= 1)

    m.brk45 = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb45 * m.x_b45[i])
    m.brk15 = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb15 * m.x_b15[i])
    m.brk30 = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub_hat[i] >= m.Tb30 * m.x_b30[i])
    m.brk_ub = pyo.Constraint(m.I, rule=lambda m, i:
        m.taub[i] <= M_big*(m.x_b45[i]+m.x_b15[i]+m.x_b30[i]))

    m.split_ord = pyo.Constraint(m.I, rule=lambda m, i:
        m.x_b30[i] <= m.phi[i])

    def _phi1(m, i):
        if i >= N: return pyo.Constraint.Skip
        return (m.phi[i+1] >= m.phi[i]+m.x_b15[i]
                -m.x_b30[i]-m.x_b45[i]-m.rho1[i]-m.rho2[i])
    def _phi2(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.phi[i+1] <= m.phi[i]+m.x_b15[i]
    def _phi3(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.phi[i+1] <= 1-m.x_b30[i]-m.x_b45[i]-m.rho1[i]-m.rho2[i]
    m.phi1 = pyo.Constraint(m.I, rule=_phi1)
    m.phi2 = pyo.Constraint(m.I, rule=_phi2)
    m.phi3 = pyo.Constraint(m.I, rule=_phi3)

    m.rst1   = pyo.Constraint(m.I, rule=lambda m,i: m.taur[i] >= m.Tr1*m.rho1[i])
    m.rst2   = pyo.Constraint(m.I, rule=lambda m,i: m.taur[i] >= m.Tr2*m.rho2[i])
    m.rst_ub = pyo.Constraint(m.I, rule=lambda m,i:
        m.taur[i] <= M_big*(m.rho1[i]+m.rho2[i]))
    m.rst_lim = pyo.Constraint(expr=sum(m.rho2[i] for i in data["I"]) <= 3)

    # ------ consecutive driving (equality + McCormick) -------
    # r_i = x_b45 + x_b30 + rho1 + rho2   (reset indicator)
    # l1_i = r_i * cd[i]
    def _ri(m, i): return m.x_b45[i]+m.x_b30[i]+m.rho1[i]+m.rho2[i]
    def _rho(m, i): return m.rho1[i]+m.rho2[i]

    m.l1u1 = pyo.Constraint(m.I, rule=lambda m,i: m.l1[i] <= M_drv*_ri(m,i))
    m.l1u2 = pyo.Constraint(m.I, rule=lambda m,i: m.l1[i] <= m.cd[i])
    m.l1lb = pyo.Constraint(m.I, rule=lambda m,i:
        m.l1[i] >= m.cd[i] - M_drv*(1-_ri(m,i)))

    def _cd(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.cd[i+1] == m.cd[i] + m.D[i] - m.l1[i]
    m.cd_prop = pyo.Constraint(m.I, rule=_cd)
    m.cd_ub   = pyo.Constraint(m.I, rule=lambda m,i: m.cd[i] <= m.Tdrv_cons)

    # ------ shift driving (equality + McCormick) -------------
    # l2_i = rho_i * sd[i]
    m.l2u1 = pyo.Constraint(m.I, rule=lambda m,i: m.l2[i] <= M_sd*_rho(m,i))
    m.l2u2 = pyo.Constraint(m.I, rule=lambda m,i: m.l2[i] <= m.sd[i])
    m.l2lb = pyo.Constraint(m.I, rule=lambda m,i:
        m.l2[i] >= m.sd[i] - M_sd*(1-_rho(m,i)))

    def _sd(m, i):
        if i >= N: return pyo.Constraint.Skip
        return m.sd[i+1] == m.sd[i] + m.D[i] - m.l2[i]
    m.sd_prop = pyo.Constraint(m.I, rule=_sd)
    m.sd_ub   = pyo.Constraint(m.I, rule=lambda m,i: m.sd[i] <= m.Tdrv_sh1)

    # ------ shift working time (equality + McCormick) --------
    # Service counts as working; charging does NOT (EU Reg. 561/2006)

    # u_i = tauc_i * nb_i  (charging as work when no break/rest declared)
    # nb_i = 1 - x_b45 - x_b15 - x_b30 - rho1 - rho2
    m.u_ub1 = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.u[i] <= TK * (1 - m.x_b45[i] - m.x_b15[i]
                            - m.x_b30[i] - m.rho1[i] - m.rho2[i]))
    m.u_ub2 = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.u[i] <= m.tauc[i])
    m.u_lb  = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.u[i] >= m.tauc[i] - TK * (m.x_b45[i] + m.x_b15[i]
                                        + m.x_b30[i] + m.rho1[i] + m.rho2[i]))

    # w_i = y_i * rho_i  (charges AND rests at stop i)
    # Used to keep queue time Q_i*y_i in the OLD shift when a rest resets sw.
    # Product of two binaries — three-constraint linearization (exact convex hull).
    m.w_ub1 = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.w[i] <= m.rho1[i] + m.rho2[i])
    m.w_ub2 = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.w[i] <= m.y[i])
    m.w_lb  = pyo.Constraint(m.Kset, rule=lambda m, i:
        m.w[i] >= m.y[i] + m.rho1[i] + m.rho2[i] - 1)

    # l4_i = rho_i * sw[i]
    m.l4u1 = pyo.Constraint(m.I, rule=lambda m,i: m.l4[i] <= M_sw*_rho(m,i))
    m.l4u2 = pyo.Constraint(m.I, rule=lambda m,i: m.l4[i] <= m.sw[i])
    m.l4lb = pyo.Constraint(m.I, rule=lambda m,i:
        m.l4[i] >= m.sw[i] - M_sw*(1-_rho(m,i)))

    def _sw(m, i):
        if i >= N: return pyo.Constraint.Skip
        if i in K:
            # Queue always counts as work.
            # u[i] = charging time as work (0 if break/rest declared).
            # w[i] = y[i]*rho[i]: subtract queue from new shift after reset.
            return m.sw[i+1] == (m.sw[i] + m.D[i]
                                + m.Q[i]*m.y[i]
                                + m.u[i]
                                - m.l4[i]
                                - m.Q[i]*m.w[i]
                                + m.M[i]*(m.x_b45[i]+m.x_b15[i]+m.x_b30[i]+m.rho1[i]+m.rho2[i]))
        else:
            # Origin (i=0), customers, or any non-CS stop.
            # At origin: svc=0, rho=0 → sw[1] = sw[0] + D[0] = D[0]. Correct.
            svc = data["S"].get(i, 0) if i in C else 0
            return m.sw[i+1] == (m.sw[i] + m.D[i]
                                + svc*(1 - m.rho1[i] - m.rho2[i])
                                - m.l4[i])

    m.sw_prop = pyo.Constraint(m.I, rule=_sw)

    m.sw_ub   = pyo.Constraint(m.I, rule=lambda m,i: m.sw[i] <= m.Twrk_sh)

    return m


# ============================================================
# SOLVE
# ============================================================

def solve_model(model, tee=False):
    solver = pyo.SolverFactory("appsi_highs")
    solver.options["mip_rel_gap"] = 0.001
    solver.options["time_limit"]  = 60*60*12  # 12h
    results = solver.solve(model, tee=True)
    status  = str(results.solver.termination_condition)
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
        is_K = i in K
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
            tauq = data["Q"].get(i, 0) * round(pyo.value(model.y[i])) if is_K else 0.0,
            taub = pyo.value(model.taub[i]),
            taur = pyo.value(model.taur[i]),
            y    = round(pyo.value(model.y[i]))     if is_K else 0,
            b45  = round(pyo.value(model.x_b45[i])),
            b15  = round(pyo.value(model.x_b15[i])),
            b30  = round(pyo.value(model.x_b30[i])),
            rho1 = round(pyo.value(model.rho1[i])),
            rho2 = round(pyo.value(model.rho2[i])),
            is_C = i in data["C"],
            is_K = is_K,
        ))
    return sol


# ============================================================
# SAVE / LOAD SOLUTION
# ============================================================

def solution_path(name):
    """Canonical path for a named solution file."""
    _ensure_dirs()
    return os.path.join(SOLUTIONS_DIR, f"{name}.json")


def save_solution(sol, data, name):
    """
    Persist sol (list of dicts) and the instance data needed for
    plotting/checking to a JSON file in solutions/.
    """
    _ensure_dirs()
    payload = {
        "name": name,
        "data": {
            "label":      data["label"],
            "N":          data["N"],
            "I":          data["I"],
            "C":          data["C"],
            "K":          data["K"],
            "Emin":       data["Emin"],
            "Ecap":       data["Ecap"],
            "Tdrv_cons":  data["Tdrv_cons"],
            "Tdrv_sh1":   data["Tdrv_sh1"],
            "Twrk_sh":    data["Twrk_sh"],
            # leg-indexed dicts — JSON requires string keys
            "D": {str(k): v for k, v in data["D"].items()},
            "E": {str(k): v for k, v in data["E"].items()},
            "S": {str(k): v for k, v in data["S"].items()},
            "Q": {str(k): v for k, v in data["Q"].items()},
        },
        "sol": sol,
    }
    fpath = solution_path(name)
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Solution saved : {fpath}")


def load_solution(name):
    """
    Load a previously saved solution.
    Returns (sol, data) ready for plot_solution / check_solution.
    """
    fpath = solution_path(name)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"No saved solution at '{fpath}'. "
            "Solve first or check the instance name."
        )
    with open(fpath, "r") as f:
        payload = json.load(f)

    data = payload["data"]
    for field in ("D", "E", "S", "Q"):          # restore int keys
        data[field] = {int(k): v for k, v in data[field].items()}
    for field in ("I", "C", "K"):               # restore int lists
        data[field] = [int(x) for x in data[field]]

    sol = payload["sol"]
    for s in sol:                                # restore int stop index
        s["i"] = int(s["i"])

    print(f"  Solution loaded: {fpath}")
    return sol, data


# ============================================================
# VISUALISATION  (all x-axes in hours, shared across panels)
# ============================================================

COL = dict(
    drive   = "#2C6FAC",
    service = "#27AE60",
    queue   = "#C0392B",   # queue/setup time at CS (always work)
    charge  = "#E67E22",
    brk     = "#F1C40F",
    rest    = "#8E44AD",
)
EPS = 1e-3   # threshold below which we don't draw a block


def _bar(ax, start, dur, y, h, color, label=None, fontsize=7, text_color="white"):
    if dur < EPS:
        return
    ax.barh(y, dur, left=start, height=h, color=color,
            edgecolor="white", linewidth=0.3)
    if dur > 0.08 and label:
        ax.text(start + dur/2, y, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold",
                clip_on=True)


def plot_solution(sol, data, title="solution"):
    N    = data["N"]
    tend = sol[-1]["ta"]
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.suptitle(f"{title}  —  {data['label']}", fontsize=12, fontweight="bold")

    # ------------------------------------------------------------------
    # Pre-compute all activity-boundary times per stop so we can draw
    # consistent vertical lines across all three panels.
    #
    # For each stop we collect: (time, line_style, line_color, alpha)
    # line_style encodes the type of boundary:
    #   "stop"   — arrival at a new stop (gray dashed, lightest)
    #   "brk"    — break starts (yellow, medium)
    #   "rest"   — rest starts (purple, strongest)
    # ------------------------------------------------------------------
    vlines = []   # list of (t, color, lw, alpha, ls)

    for s in sol:
        ta = s["ta"]
        t  = ta

        # arrival marker (always)
        vlines.append((ta, "gray", 0.6, 0.35, "--"))

        # advance t through each activity, adding a line at each transition
        if s["is_C"]:
            t += data["S"].get(s["i"], 0)

        if s["is_K"] and s["y"]:
            if s["tauq"] > EPS:
                t += s["tauq"]
                vlines.append((t, COL["queue"],  0.6, 0.30, ":"))
            if s["tauc"] > EPS:
                t += s["tauc"]
                vlines.append((t, COL["charge"], 0.7, 0.35, ":"))

        if s["taub"] > EPS:
            vlines.append((t, COL["brk"],  0.8, 0.50, "--"))
            t += s["taub"]

        if s["taur"] > EPS:
            vlines.append((t, COL["rest"], 1.0, 0.55, "--"))

    # ------------------------------------------------------------------
    # Helper: draw all vlines on a given axis
    # ------------------------------------------------------------------
    def _draw_vlines(ax, top_panel=False):
        seen = set()
        for (t, col, lw, alpha, ls) in vlines:
            key = round(t, 4)
            if key in seen:
                continue
            seen.add(key)
            ax.axvline(t, color=col, lw=lw, alpha=alpha, ls=ls, zorder=0)

    # ============ Panel 1: Gantt =============================
    ax = axes[0]
    ax.set_title("Activity timeline", fontsize=10)
    Y, H = 0.5, 0.38

    for s in sol:
        i = s["i"]

        # driving leg arriving at stop i
        if i > 0:
            drv_start = sol[i-1]["td"]
            drv_dur   = s["ta"] - drv_start
            _bar(ax, drv_start, drv_dur, Y, H, COL["drive"],
                 label=f"drv→{i}", fontsize=7)

        t = s["ta"]

        if s["is_C"]:
            svc = data["S"].get(i, 0)
            _bar(ax, t, svc, Y, H, COL["service"],
                 label=f"C{i}", fontsize=7)
            t += svc

        if s["is_K"] and s["y"] and s["tauq"] > EPS:
            _bar(ax, t, s["tauq"], Y, H, COL["queue"],
                 label="Q", fontsize=7)
            t += s["tauq"]

        if s["is_K"] and s["y"] and s["tauc"] > EPS:
            _bar(ax, t, s["tauc"], Y, H, COL["charge"],
                 label=f"CHG\n{s['ea']:.0f}→{s['ed']:.0f}", fontsize=6.5)
            t += s["tauc"]

        if s["taub"] > EPS:
            lbl = ("B45" if s["b45"] else ("B15" if s["b15"] else "B30"))
            _bar(ax, t, s["taub"], Y, H, COL["brk"],
                 label=lbl, fontsize=7, text_color="#333")
            t += s["taub"]

        if s["taur"] > EPS:
            lbl = "RST-r1" if s["rho1"] else "RST-r2"
            _bar(ax, t, s["taur"], Y, H, COL["rest"],
                 label=lbl, fontsize=7)

        stop_type = ("●C" if s["is_C"] else
                     "▲K" if s["is_K"] else
                     ("O" if i == 0 else "D"))
        ax.text(s["ta"], Y + H/2 + 0.06, f"{stop_type}{i}",
                ha="left", va="bottom", fontsize=6.5, color="#444",
                rotation=45, clip_on=True)

    _draw_vlines(ax)
    ax.set_yticks([])
    ax.set_xlim(-0.2, tend * 1.02)
    patches = [mpatches.Patch(color=v, label=k.replace("_","").title())
               for k, v in COL.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=8, ncol=6)

    # ============ Panel 2: SOC vs time =======================
    ax2 = axes[1]
    ax2.set_title("Battery state of charge", fontsize=10)

    time_pts, soc_pts = [], []
    for s in sol:
        ta, td = s["ta"], s["td"]
        ea, ed = s["ea"], s["ed"]
        tauq   = s["tauq"] if s["is_K"] else 0.0
        tauc   = s["tauc"] if s["is_K"] else 0.0
        time_pts.append(ta)
        soc_pts.append(ea)
        if td - ta > EPS:
            t_chg_start = ta + tauq
            t_chg_end   = t_chg_start + tauc
            if tauq > EPS:
                time_pts.append(t_chg_start)
                soc_pts.append(ea)
            if tauc > EPS:
                time_pts.append(t_chg_end)
                soc_pts.append(ed)
            time_pts.append(td)
            soc_pts.append(ed)

    ax2.plot(time_pts, soc_pts, color=COL["drive"], lw=2, label="SOC", zorder=2)
    ax2.fill_between(time_pts, soc_pts, alpha=0.10, color=COL["drive"])

    # Charging arrows
    for s in sol:
        if s["is_K"] and s["y"] and s["ed"] - s["ea"] > 0.5:
            t_start = s["ta"] + s["tauq"]
            t_end   = t_start + s["tauc"]
            ax2.annotate("", xy=(t_end, s["ed"]), xytext=(t_start, s["ea"]),
                arrowprops=dict(arrowstyle="->", color=COL["charge"], lw=1.5),
                zorder=3)
            ax2.text((t_start + t_end) / 2, (s["ea"] + s["ed"]) / 2,
                     f"+{s['ed']-s['ea']:.0f}", ha="center",
                     fontsize=7, color=COL["charge"])

    # Annotate SOC panel: label why charging was triggered
    for s in sol:
        if s["is_K"] and s["y"] and s["ea"] - data["Emin"] < 0.15 * data["Ecap"]:
            ax2.text(s["ta"], s["ea"] + data["Ecap"] * 0.03,
                     "SOC\nnear\nmin", ha="center", va="bottom",
                     fontsize=6, color="red",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec="red", alpha=0.7))

    ax2.axhline(data["Emin"], color="red",  ls=":", lw=1.2,
                label=f"E_min = {data['Emin']} kWh")
    ax2.axhline(data["Ecap"], color="gray", ls=":", lw=1.2,
                label=f"E_cap = {data['Ecap']} kWh")

    _draw_vlines(ax2)
    ax2.set_ylabel("kWh")
    ax2.set_ylim(0, data["Ecap"] * 1.15)
    ax2.legend(fontsize=8, ncol=3, loc="upper right")

    # ============ Panel 3: HoS counters vs time ==============
    ax3 = axes[2]
    ax3.set_title("HoS accumulators (at arrival)", fontsize=10)

    cd_t, cd_v = [], []
    sd_t, sd_v = [], []
    sw_t, sw_v = [], []

    for s in sol:
        ta, td = s["ta"], s["td"]
        r_cd = s["b45"] or s["b30"] or s["rho1"] or s["rho2"]
        r_sd = s["rho1"] or s["rho2"]
        r_sw = s["rho1"] or s["rho2"]

        cd_t.append(ta); cd_v.append(s["cd"])
        sd_t.append(ta); sd_v.append(s["sd"])
        sw_t.append(ta); sw_v.append(s["sw"])

        if td - ta > EPS:
            cd_t.append(td); cd_v.append(0.0 if r_cd else s["cd"])
            sd_t.append(td); sd_v.append(0.0 if r_sd else s["sd"])
            sw_t.append(td); sw_v.append(0.0 if r_sw else s["sw"])

    ax3.plot(cd_t, cd_v, "o-", color="#E74C3C", lw=1.5, ms=3,
             label="Consec. driving", zorder=2)
    ax3.plot(sd_t, sd_v, "s-", color="#3498DB", lw=1.5, ms=3,
             label="Shift driving", zorder=2)
    ax3.plot(sw_t, sw_v, "^-", color="#1ABC9C", lw=1.5, ms=3,
             label="Shift working", zorder=2)

    ax3.axhline(data["Tdrv_cons"], color="#E74C3C", ls=":", lw=1.2, alpha=0.7,
                label=f"max consec. drv {data['Tdrv_cons']}h")
    ax3.axhline(data["Tdrv_sh1"],  color="#3498DB", ls=":", lw=1.2, alpha=0.7,
                label=f"max shift drv {data['Tdrv_sh1']}h")
    ax3.axhline(data["Twrk_sh"],   color="#1ABC9C", ls=":", lw=1.2, alpha=0.7,
                label=f"max shift wk {data['Twrk_sh']}h")

    # ------------------------------------------------------------------
    # Annotate what caused each break or rest.
    # Rule: at the stop where a break/rest is taken, find which
    # accumulator was closest to (or at) its limit as a fraction of the
    # limit. That is the binding constraint.  Draw a small labelled arrow
    # from the relevant counter value up toward the limit line.
    # ------------------------------------------------------------------
    _annotated = set()   # avoid duplicate labels at the same time

    for s in sol:
        has_brk  = s["b45"] or s["b30"] or s["b15"]
        has_rest = s["rho1"] or s["rho2"]
        if not (has_brk or has_rest):
            continue

        ta = s["ta"]
        if round(ta, 3) in _annotated:
            continue
        _annotated.add(round(ta, 3))

        # fractions of limit reached
        frac_cd = s["cd"] / data["Tdrv_cons"] if data["Tdrv_cons"] > 0 else 0
        frac_sd = s["sd"] / data["Tdrv_sh1"]  if data["Tdrv_sh1"]  > 0 else 0
        frac_sw = s["sw"] / data["Twrk_sh"]   if data["Twrk_sh"]   > 0 else 0

        # map to (fraction, counter_value, limit_value, label, color)
        candidates = [
            (frac_cd, s["cd"], data["Tdrv_cons"], "cd→limit", "#E74C3C"),
            (frac_sd, s["sd"], data["Tdrv_sh1"],  "sd→limit", "#3498DB"),
            (frac_sw, s["sw"], data["Twrk_sh"],   "sw→limit", "#1ABC9C"),
        ]
        # only annotate if the counter is actually close (>70% of limit)
        binding = [(f, val, lim, lbl, col)
                   for (f, val, lim, lbl, col) in candidates if f > 0.70]
        if not binding:
            continue

        # pick the most binding
        f, val, lim, lbl, col = max(binding, key=lambda x: x[0])

        # draw a small upward arrow from the counter value toward the limit
        ax3.annotate(
            lbl,
            xy=(ta, lim),           # arrowhead at the limit line
            xytext=(ta, val * 0.85 if val > 0.5 else val + 0.3),  # label below
            fontsize=6.5,
            color=col,
            ha="center",
            va="top",
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, alpha=0.85),
            zorder=4,
        )

    _draw_vlines(ax3)
    ax3.set_xlabel("Time (h)")
    ax3.set_ylabel("Hours")
    ax3.legend(fontsize=7, ncol=3, loc="upper left")
    plt.tight_layout()

    fname = f"solution_{title}_{time.time()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {fname}")
    plt.close()




# ============================================================
# FEASIBILITY CHECKER + SCHEDULE PRINT
# ============================================================

def check_solution(sol, data):
    print("\n  === Feasibility check ===")
    ok = True
    for s in sol:
        i = s["i"]
        if s["ta"] > s["td"] + EPS and i != 0:
            print(f"  WARN  ta > td at stop {i}: ta={s['ta']:.3f} td={s['td']:.3f}")
        if s["cd"] > data["Tdrv_cons"] + EPS:
            print(f"  FAIL  consec_drv stop {i}: {s['cd']:.3f} > {data['Tdrv_cons']}")
            ok = False
        if s["sd"] > data["Tdrv_sh1"] + EPS:
            print(f"  FAIL  shift_drv  stop {i}: {s['sd']:.3f} > {data['Tdrv_sh1']}")
            ok = False
        if s["sw"] > data["Twrk_sh"] + EPS:
            print(f"  FAIL  shift_wk   stop {i}: {s['sw']:.3f} > {data['Twrk_sh']}")
            ok = False
        if s["ea"] < data["Emin"] - EPS:
            print(f"  FAIL  ea stop {i}: {s['ea']:.2f} < {data['Emin']}")
            ok = False
        if s["ed"] > data["Ecap"] + EPS:
            print(f"  FAIL  ed stop {i}: {s['ed']:.2f} > {data['Ecap']}")
            ok = False
    print("  OK — all checked." if ok else "  Some constraints violated.")
    return ok


def print_schedule(sol, data):
    print(f"\n  {'i':>3}  {'type':>5}  {'ta':>6}  {'td':>6}  "
          f"{'ea':>6}  {'ed':>6}  {'cd':>5}  {'sd':>5}  {'sw':>5}  activity")
    print("  " + "─"*80)
    for s in sol:
        i   = s["i"]
        typ = ("ORIG" if i==0 else "DEST" if i==data["N"] else
               "CUST" if s["is_C"] else "CS")
        acts = []
        if s["is_K"] and s["y"]:
            acts.append(f"CHG {s['ea']:.0f}→{s['ed']:.0f}kWh ({s['tauc']:.2f}h)")
        if s["b45"]: acts.append(f"B45 {s['taub']:.2f}h")
        if s["b15"]: acts.append(f"B15 {s['taub']:.2f}h")
        if s["b30"]: acts.append(f"B30 {s['taub']:.2f}h")
        if s["rho1"]: acts.append(f"REST-r1 {s['taur']:.1f}h")
        if s["rho2"]: acts.append(f"REST-r2 {s['taur']:.1f}h")
        print(f"  {i:>3}  {typ:>5}  {s['ta']:>6.2f}  {s['td']:>6.2f}  "
              f"{s['ea']:>6.1f}  {s['ed']:>6.1f}  "
              f"{s['cd']:>5.2f}  {s['sd']:>5.2f}  {s['sw']:>5.2f}  "
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


def run_instance(name, tee=False, run=True):
    print(f"\n{'='*65}")
    data = INSTANCES[name]()
    print(f"  {data['label']}")
    print(f"  C={data['C']}   K={data['K']}")

    if not run:
        sol, _ = load_solution("realistic_medium_many_3")
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
    random.seed(10)
    name = sys.argv[1] if len(sys.argv) > 1 else "realistic"
    tee  = "--tee" in sys.argv
    run  = True # "--run" in sys.argv

    # run_instance(name, run=False)

    if name == "all":
        for n in INSTANCES:
            run_instance(n, tee=tee, run=run)
    elif name in INSTANCES:
        run_instance(name, tee=tee, run=run)
    else:
        print(f"Unknown instance '{name}'. Choose: {list(INSTANCES)}")