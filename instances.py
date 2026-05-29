"""
instances.py  —  All instance generators for the BET scheduling problem.
=========================================================================
Moved here from MILP.py so MILP.py stays a pure modelling file.

Usage
-----
    from instances import (
        instance_tiny, instance_break_forced, instance_charging_needed,
        instance_rest_forced, instance_3day, instance_realistic,
        ALL_INSTANCES, DET_ONLY_INSTANCES,
    )
"""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from MILP import _make_data


# ══════════════════════════════════════════════════════════════════════════
# ORIGINAL BENCHMARK INSTANCES
# ══════════════════════════════════════════════════════════════════════════

def instance_tiny():
    N = 4
    return _make_data(
        I=list(range(N + 1)), C=[1], K=[2, 3],
        D={0:0.5, 1:0.5, 2:0.5, 3:0.5},
        E={0:8.0, 1:8.0, 2:8.0, 3:8.0},
        S={1: 0.5},
        E0=60, Ecap=100, Emin=10,
        Ebar={0:0, 1:40, 2:80, 3:100},
        Tbar={0:0.0, 1:0.55, 2:1.37, 3:2.50},
        Wha={1:0}, Whf={1:5},
        label="tiny — 5 stops, basic SOC + timing check", title="tiny")


def instance_break_forced():
    N = 10
    C, K = [2, 7], [1, 3, 4, 5, 6, 8, 9]
    return _make_data(
        I=list(range(N + 1)), C=C, K=K,
        D={i: 1.0 for i in range(N)},
        E={i: 7.0 for i in range(N)},
        S={2: 0.5, 7: 0.5},
        E0=90, Ecap=100, Emin=10,
        Ebar={0:0, 1:40, 2:80, 3:100},
        Tbar={0:0.0, 1:0.55, 2:1.37, 3:2.50},
        Wha={2:0, 7:0}, Whf={2:20, 7:20},
        label="break_forced — 10 stops, 4.5h driving limit binds",
        title="break_forced")


def instance_charging_needed():
    N = 8
    C, K = [2, 6], [1, 3, 4, 5, 7]
    return _make_data(
        I=list(range(N + 1)), C=C, K=K,
        D={i: 1.0 for i in range(N)},
        E={i: 22.0 for i in range(N)},
        S={2: 0.5, 6: 0.5},
        E0=80, Ecap=100, Emin=10,
        Ebar={0:0, 1:40, 2:80, 3:100},
        Tbar={0:0.0, 1:0.55, 2:1.37, 3:2.50},
        Wha={2:0, 6:0}, Whf={2:20, 6:20},
        label="charging_needed — 8 stops, high consumption forces charging",
        title="charging_needed")


def instance_rest_forced():
    N = 14
    C = [3, 8, 12]
    K = [1, 2, 4, 5, 6, 7, 9, 10, 11, 13]
    return _make_data(
        I=list(range(N + 1)), C=C, K=K,
        D={i: 1.0 for i in range(N)},
        E={i: 7.0 for i in range(N)},
        S={3: 0.5, 8: 0.5, 12: 0.5},
        E0=90, Ecap=100, Emin=10,
        Ebar={0:0, 1:40, 2:80, 3:100},
        Tbar={0:0.0, 1:0.55, 2:1.37, 3:2.50},
        Wha={3:0, 8:0, 12:0}, Whf={3:30, 8:30, 12:30},
        label="rest_forced — 14 stops, 9h shift limit forces daily rest",
        title="rest_forced")


def instance_3day():
    N = 34
    C = [3, 7, 11, 15, 19, 22, 25, 28, 30, 32]
    K = [i for i in range(1, N) if i not in C]
    return _make_data(
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
        title="3day")


def instance_realistic(route_class="short", clusters=3, customers_class="medium"):
    """
    Randomly generated long-haul route.

    Parameters
    ----------
    route_class      : "short" (800–1200 km) | "medium" (1500–2500) | "long" (3000–4000)
    clusters         : 1 | 2 | 3 — number of customer delivery clusters
    customers_class  : "few" (1–3) | "medium" (4–5) | "many" (6–15)
    """
    distances = {"short": [800, 1200], "medium": [1500, 2500], "long": [3000, 4000]}
    customers = {"few": (1, 3), "medium": (4, 5), "many": (6, 15)}
    average_speed      = 80          # km/h nominal highway speed
    CS_spacing         = 40          # km between CS stops
    Battery_capacity   = 350         # kWh

    # Nominal energy is computed leg-by-leg as km * ECR(v_nom) below
    nb_customers = random.randint(*customers[customers_class])
    route_distance = random.randint(*distances[route_class])

    if clusters == 1:
        cluster_centers = [random.randint(int(0.5*route_distance),
                                          int(0.6*route_distance))]
    elif clusters == 2:
        cluster_centers = [
            random.randint(int(0.35*route_distance), int(0.45*route_distance)),
            random.randint(int(0.55*route_distance), int(0.65*route_distance))]
    else:
        cluster_centers = [
            random.randint(int(0.25*route_distance), int(0.30*route_distance)),
            random.randint(int(0.50*route_distance), int(0.55*route_distance)),
            random.randint(int(0.70*route_distance), int(0.75*route_distance))]

    customer_locations = sorted(
        random.choice(cluster_centers) + random.randint(-75, 75)
        for _ in range(nb_customers))

    # ECR at nominal speed for computing nominal energy per leg
    def _ecr_local(v):
        v = max(float(v), 5.0)
        return 33.055 / v - 0.257 + 7.2e-5 * v**2

    I = [0]; C = []; K = []
    _d0 = CS_spacing / average_speed
    D = {0: _d0}
    E = {0: CS_spacing * _ecr_local(average_speed)}
    I_nb = 1; cur_c = 0; prev_cs = 0

    for dist in range(CS_spacing, route_distance, CS_spacing):
        real = dist + random.randint(-19, 19)
        prev_stop = prev_cs
        while (cur_c < len(customer_locations) and
               prev_cs < customer_locations[cur_c] < real):
            I.append(I_nb); C.append(I_nb)
            _km = customer_locations[cur_c] - prev_stop
            D[I_nb] = _km / average_speed
            E[I_nb] = _km * _ecr_local(average_speed)
            I_nb += 1; prev_stop = customer_locations[cur_c]; cur_c += 1
        I.append(I_nb); K.append(I_nb)
        _km = real - prev_stop
        D[I_nb] = _km / average_speed
        E[I_nb] = _km * _ecr_local(average_speed)
        I_nb += 1; prev_cs = real

    I.append(I_nb)
    print(f"Route: {route_distance} km, {len(C)} customers, {len(K)} CS")

    # km: physical leg distances (km) = average_speed * D[i]
    # These are used by _ecr(v) in scenario generation.
    km = {i: average_speed * D[i] for i in D}

    Bcap = Battery_capacity
    return _make_data(
        I=I, C=C, K=K, D=D, E=E, km=km,
        S={c: 0.5 for c in C},
        E0=Bcap, Ecap=Bcap, Emin=0.2 * Bcap,
        Ebar={0: 0,
              1: 0.40 * Bcap,
              2: 0.80 * Bcap,
              3: Bcap},
        Tbar={0: 0.0, 1: 0.55, 2: 1.367, 3: 2.50},
        Wha={c: 0        for c in C},
        Whf={c: 20000000 for c in C},
        label="realistic — randomly generated long-haul route",
        title=f"realistic_{route_class}_{customers_class}_{clusters}")


# ══════════════════════════════════════════════════════════════════════════
# TARGETED EDGE-CASE INSTANCES
# ══════════════════════════════════════════════════════════════════════════

def instance_split_break():
    """
    Forces the b15→b30 split-break sequence.
    Legs sized so cd approaches 4.5h but stays under even with δ=20%:
    max drawn cd = (1.5+1.4)*1.2 = 3.48h < 4.5h.
    """
    I = [0, 1, 2, 3, 4]
    C = [2]
    K = [1, 3]
    D = {0: 1.5, 1: 0.4, 2: 1.4, 3: 0.7}
    E = {0: 45,  1: 12,  2: 42,  3: 20}
    return _make_data(
        I=I, C=C, K=K, D=D, E=E,
        S={2: 0.5},
        E0=200, Ecap=200, Emin=40,
        Ebar={0:0, 1:80, 2:160, 3:200},
        Tbar={0:0.0, 1:0.55, 2:1.37, 3:2.50},
        Wha={2:0}, Whf={2:1e7},
        label="split_break — forces b15+b30 sequence",
        title="split_break")


def instance_phi_inherited():
    """
    Start mid-route with phi=1 (b15 already taken).
    The b30 option should then be available at the first stop.
    Tests that phi is correctly inherited in init_state.
    """
    I = [0, 1, 2, 3]
    C = []
    K = [1, 2]
    D = {0: 1.5, 1: 1.0, 2: 0.8}
    E = {0: 45,  1: 30,  2: 25}
    return _make_data(
        I=I, C=C, K=K, D=D, E=E,
        S={},
        E0=150, Ecap=200, Emin=30,
        Ebar={0:0, 1:80, 2:160, 3:200},
        Tbar={0:0.0, 1:0.55, 2:1.37, 3:2.50},
        Wha={}, Whf={},
        label="phi_inherited — b30 available from start",
        title="phi_inherited")


def instance_rho2_budget():
    """
    Route requiring four rests; first three can be r2, fourth must be r1.
    Tests that the rho2_used budget is tracked and enforced correctly.
    """
    shift_d = 8.5 / 3   # three legs per shift
    I = list(range(13))
    C = [3, 6, 9]
    K = [1, 2, 4, 5, 7, 8, 10, 11]
    D = {i: shift_d for i in range(12)}
    E = {i: shift_d * 30 for i in range(12)}
    return _make_data(
        I=I, C=C, K=K, D=D, E=E,
        S={3: 0.5, 6: 0.5, 9: 0.5},
        E0=350, Ecap=350, Emin=50,
        Ebar={0:0, 1:140, 2:280, 3:350},
        Tbar={0:0.0, 1:0.55, 2:1.37, 3:2.50},
        Wha={c:0 for c in C}, Whf={c:1e7 for c in C},
        label="rho2_budget — 4 rests, 3x r2 allowed",
        title="rho2_budget")


def instance_tight_energy_chain():
    """
    Battery must charge at every CS stop: each leg consumes 85% of usable
    capacity (136 kWh) so skipping any CS charge leads to energy infeasibility.
    Legs are 1.36h so two consecutive = 2.72h < 4.5h cd limit.

    NOTE: test with delta=0 only (boundary instance — listed in DET_ONLY_INSTANCES).
    """
    Ecap = 200; Emin = 40
    E_leg = (Ecap - Emin) * 0.85    # 136 kWh > headroom per leg
    D_leg = round(E_leg / 100, 3)   # 100 kWh/h → 1.36h per leg
    N_legs = 5
    I = list(range(N_legs + 1))
    K = list(range(1, N_legs))
    C = []
    D = {i: D_leg for i in range(N_legs)}
    E = {i: E_leg for i in range(N_legs)}
    return _make_data(
        I=I, C=C, K=K, D=D, E=E,
        S={},
        E0=Ecap, Ecap=Ecap, Emin=Emin,
        Ebar={0:0, 1:80, 2:160, 3:200},
        Tbar={0:0.0, 1:0.55, 2:1.37, 3:2.50},
        Wha={}, Whf={},
        label="tight_energy_chain — must charge at every CS",
        title="tight_energy")


def instance_sd_boundary():
    """
    Route where shift driving approaches 9h, forcing a preemptive rest.
    Three legs of 2.9h each + a short final segment.

    NOTE: test with delta=0 only (boundary instance — listed in DET_ONLY_INSTANCES).
    With δ=20%: max sd = 2.9*1.2*3 = 10.44h which exceeds 9h within a leg,
    something no look-ahead policy can prevent.
    """
    I = [0, 1, 2, 3, 4, 5]
    C = []
    K = [1, 2, 3, 4]
    D = {0: 2.9, 1: 2.9, 2: 2.9, 3: 0.4, 4: 0.4}
    E = {0: 87,  1: 87,  2: 87,  3: 12,  4: 12}
    return _make_data(
        I=I, C=C, K=K, D=D, E=E,
        S={},
        E0=350, Ecap=350, Emin=50,
        Ebar={0:0, 1:140, 2:280, 3:350},
        Tbar={0:0.0, 1:0.55, 2:1.37, 3:2.50},
        Wha={}, Whf={},
        label="sd_boundary — sd approaches 9h, rest required",
        title="sd_boundary")


# ══════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════

# All instances available for CLI / test loops
ALL_INSTANCES = {
    "tiny"              : instance_tiny,
    "break_forced"      : instance_break_forced,
    "charging_needed"   : instance_charging_needed,
    "rest_forced"       : instance_rest_forced,
    "3day"              : instance_3day,
    "realistic"         : instance_realistic,
    # targeted edge-cases
    "split_break"       : instance_split_break,
    "phi_inherited"     : instance_phi_inherited,
    "rho2_budget"       : instance_rho2_budget,
    "tight_energy_chain": instance_tight_energy_chain,
    "sd_boundary"       : instance_sd_boundary,
}

# Boundary-tight instances: only meaningful with delta=0 (no noise)
DET_ONLY_INSTANCES = {"tight_energy_chain", "sd_boundary"}