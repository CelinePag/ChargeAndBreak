"""
Pyomo implementation of the Break-and-Charge MILP model
from: Brockmann & Guajardo (2025), "Break-and-charge: Leveraging EU regulations
to enhance electric truck competitiveness", Sustainability Analytics and Modeling 5,
https://doi.org/10.1016/j.samod.2025.100047

Model: Truck Driver Scheduling Problem (TDSP) for Battery Electric Trucks (BETs)
       with HOS (Hours of Service) regulation and non-linear charging.

SECTION 3 of the paper defines the MILP model. This file implements it in Pyomo.

==============================================================================
DATA STRUCTURES
==============================================================================

All model data is passed in a single Python dictionary `data`. Below is the
exact schema expected, with types, shapes, and example values.

data = {
    # -------------------------------------------------------------------
    # SETS  (all as plain Python iterables / lists)
    # -------------------------------------------------------------------
    'N':  [1, 2, 3, 4],          # Ordered list of customer location indices.
                                  # Index order matters: customer i is visited
                                  # before customer i+1 (fixed route).
                                  # The last element has no successor and is
                                  # the depot / end point.

    'F':  ['f1', 'f2', 'f3'],    # Charging station identifiers (strings or
                                  # ints, just needs to be hashable).

    'R':  [0, 1],                 # Secant-line indices for piecewise-linear
                                  # charging function. len(R) = number of
                                  # segments used to approximate the curve.
                                  # Indices must be consecutive integers
                                  # starting at 0 (needed for r+1 lookup).

    'Z':  [(1,'f1'),(1,'f2'),     # Explicit set of allowed detours.
           (2,'f1'),(3,'f2')],    # Z ⊆ N × F. A tuple (i, f) means the truck
                                  # may detour via charger f when travelling
                                  # from customer i to customer i+1.
                                  # Typically Z = {(i,f) | i ∈ N\{last}, f ∈ F}
                                  # but can be restricted to "close" chargers.

    # -------------------------------------------------------------------
    # SCALAR PARAMETERS
    # -------------------------------------------------------------------
    'h':        0.85,   # kWh/km – electricity consumption rate of the truck.
                        # Source: manufacturer data (e.g. MAN eTGX spec sheet).

    'D_safety': 50.0,   # km – safety/reserve distance the battery must always
                        # cover (SOC buffer). Paper uses D^safety = 50 km.

    'Y_max':    480.0,  # kWh – usable battery capacity (e.g. BT480 = 480 kWh,
                        # BT400 = 400 kWh, BT320 = 320 kWh in the paper).

    'B':        0.75,   # hours – minimum stop duration to count as a HOS break
                        # (45 min = 0.75 h, EU Reg. EC No 561/2006).

    'W_break':  4.5,    # hours – maximum driving time before a mandatory break
                        # (4.5 h, EU Reg. EC No 561/2006).

    'W_day':    9.0,    # hours – maximum total driving time per day
                        # (9 h standard, EU Reg. EC No 561/2006).

    'M':        1e6,    # Big-M constant. Should be large enough to be a valid
                        # upper bound but as small as possible to keep the LP
                        # relaxation tight. A value of 1e4–1e6 is typical.

    # -------------------------------------------------------------------
    # INDEXED PARAMETERS  (all as Python dicts keyed by set indices)
    # -------------------------------------------------------------------

    # Travel time in hours between any two locations in V = N ∪ F.
    # Keys: (loc_a, loc_b) where loc_a, loc_b ∈ V.
    # Must include (i, i+1), (i, f), and (f, i+1) entries for all i ∈ N, f ∈ F.
    'T_travel': {
        (1, 2): 1.2,  (2, 3): 0.9,  (3, 4): 1.5,   # direct customer arcs
        (1,'f1'): 0.6, ('f1', 2): 0.7,               # via charger f1 after cust 1
        (1,'f2'): 1.0, ('f2', 2): 0.3,
        (2,'f1'): 0.4, ('f1', 3): 0.6,
        (3,'f2'): 0.5, ('f2', 4): 1.1,
        # ... all needed pairs
    },

    # Distance in km between any two locations in V = N ∪ F.
    # Same key structure as T_travel.
    'D_dist': {
        (1, 2): 80.0, (2, 3): 60.0, (3, 4): 100.0,
        (1,'f1'): 40.0, ('f1', 2): 45.0,
        (1,'f2'): 70.0, ('f2', 2): 20.0,
        (2,'f1'): 25.0, ('f1', 3): 40.0,
        (3,'f2'): 30.0, ('f2', 4): 75.0,
    },

    # Breakpoints of the piecewise-linear charging function.
    # s[f, r] = energy level (kWh) at the start of secant segment r
    #           for charging station f.
    # Keys: (station_id, segment_index).
    # The paper uses two segments (R = {0,1}), breakpoint at 80% SOC:
    #   segment 0: constant-current phase  (SOC 0% – 80%)
    #   segment 1: constant-voltage phase  (SOC 80% – 100%)
    's': {
        ('f1', 0): 0.0,    ('f1', 1): 384.0,   # 384 = 80% of 480 kWh
        ('f2', 0): 0.0,    ('f2', 1): 384.0,
    },

    # Gradient (kW) of secant line r at station f.
    # K[f, r] is the charging power (kW) in segment r.
    # Higher in segment 0 (CC phase), lower in segment 1 (CV phase).
    'K': {
        ('f1', 0): 350.0,  ('f1', 1): 120.0,
        ('f2', 0): 300.0,  ('f2', 1): 100.0,
    },

    # Intercept of secant line r at station f (in kWh).
    # B_intercept[f, r] is derived from the piecewise approximation so that
    # energy = K[f,r] * time + B_intercept[f,r] on segment r.
    'B_intercept': {
        ('f1', 0): 0.0,    ('f1', 1): -some_value,
        ('f2', 0): 0.0,    ('f2', 1): -some_value,
    },
}

==============================================================================
NOTE ON THE SUCCESSOR MAPPING
==============================================================================
N is treated as an **ordered sequence**. The model uses i → i+1 (successor)
arcs. We build a successor dict from the sorted N list:

    N_sorted = [1, 2, 3, 4]
    successor = {1: 2,  2: 3,  3: 4}   # last element has no successor

All constraints indexed over i ∈ N that reference i+1 must skip the last
element. This is handled via `Constraint.Skip` checks.
"""

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    Binary, NonNegativeReals, minimize, value, SolverFactory
)

import brockmann_graphs as gr

# =============================================================================
# HELPER: build successor mapping
# =============================================================================

def build_successor(N_list):
    """Return {i: i+1} for all but the last element of the ordered list."""
    N_sorted = sorted(N_list)
    return {N_sorted[k]: N_sorted[k + 1] for k in range(len(N_sorted) - 1)}


# =============================================================================
# MODEL BUILDER
# =============================================================================

def build_bet_tdsp_model(data: dict) -> ConcreteModel:
    """
    Build the BET TDSP MILP (Section 3, Brockmann & Guajardo 2025).

    Parameters
    ----------
    data : dict
        See module docstring for the exact schema.

    Returns
    -------
    model : pyomo.environ.ConcreteModel
    """
    m = ConcreteModel(name="BET_TDSP")

    ORIGINAL = False # Set to True to match the original paper's constraints (that are wrong)

    # -------------------------------------------------------------------------
    # SETS
    # -------------------------------------------------------------------------
    m.N = Set(initialize=data["N"], ordered=True,
              doc="Ordered customer locations")
    m.F = Set(initialize=data["F"],
              doc="Charging stations")
    m.Z = Set(initialize=data["Z"], dimen=2,
              doc="Allowed detours (i, f): i in N\\{last}, f in F")
    m.R = Set(initialize=data["R"], ordered=True,
              doc="Secant-line segment indices (consecutive ints from 0)")

    # Successor mapping (pure Python dict, not a Pyomo object)
    successor = build_successor(list(data["N"]))
    N_inner = list(successor.keys())   # all customers with a successor

    # -------------------------------------------------------------------------
    # SCALAR PARAMETERS
    # -------------------------------------------------------------------------
    m.h = Param(initialize=data["h"],
                doc="Energy consumption rate [kWh/km]")
    m.D_safety = Param(initialize=data["D_safety"],
                       doc="Safety reserve distance [km]")
    m.Y_max = Param(initialize=data["Y_max"],
                    doc="Maximum usable battery capacity [kWh]")
    m.B_break = Param(initialize=data["B"],
                      doc="Minimum stop duration to count as HOS break [h]")
    m.W_break = Param(initialize=data["W_break"],
                      doc="Max driving time before mandatory break [h] (EU: 4.5 h)")
    m.W_day = Param(initialize=data["W_day"],
                    doc="Max total driving time per day [h] (EU: 9 h)")
    m.M = Param(initialize=data["M"],
                doc="Big-M constant")

    # -------------------------------------------------------------------------
    # INDEXED PARAMETERS
    # -------------------------------------------------------------------------
    # Travel times and distances are defined over V×V but we only populate
    # entries that are actually used (sparse dict).
    m.T_travel = Param(m.N | m.F, m.N | m.F,
                       initialize=data["T_travel"],
                       default=0.0,
                       doc="Travel time [h] between locations in V = N ∪ F")

    m.D_dist = Param(m.N | m.F, m.N | m.F,
                     initialize=data["D_dist"],
                     default=0.0,
                     doc="Distance [km] between locations in V = N ∪ F")

    # Piecewise-linear charging function parameters
    m.s_bp = Param(m.F, m.R,
                   initialize=data["s"],
                   doc="Breakpoint [kWh] at start of secant segment r for station f")

    m.K_grad = Param(m.F, m.R,
                     initialize=data["K"],
                     doc="Charging gradient [kW] of secant segment r at station f")

    m.B_int = Param(m.F, m.R,
                    initialize=data["B_intercept"],
                    doc="Intercept [kWh] of secant line r at station f")

    # -------------------------------------------------------------------------
    # BINARY DECISION VARIABLES
    # -------------------------------------------------------------------------
    m.x = Var(m.N, domain=Binary,
              doc="1 if direct arc i → i+1 is used")
    m.z = Var(m.Z, domain=Binary,
              doc="1 if detour i → f → i+1 is used")
    m.a = Var(m.Z, m.R, domain=Binary,
              doc="1 if BET arrives at charger f (on leg i→i+1) with SOC in segment r")
    m.w = Var(m.N, domain=Binary,
              doc="1 if mandatory HOS break is taken at customer location i")
    m.w_prime = Var(m.N, domain=Binary,
                    doc="1 if charging time at station f (after customer i) >= B hours")

    # -------------------------------------------------------------------------
    # CONTINUOUS DECISION VARIABLES
    # -------------------------------------------------------------------------
    m.T_drive = Var(domain=NonNegativeReals,
                    doc="Total driving duration [h]")
    m.T_stop = Var(domain=NonNegativeReals,
                   doc="Total break + charging duration [h]")
    m.T_total = Var(domain=NonNegativeReals,
                    doc="Total route duration [h] = T_drive + T_stop")

    # Energy levels
    m.y = Var(m.N, domain=NonNegativeReals,
              doc="Energy [kWh] on arrival at customer i")
    m.y_prime = Var(m.Z, domain=NonNegativeReals,
                    doc="Energy [kWh] on arrival at charger f (between i and i+1)")
    m.y_dbl_prime = Var(m.Z, domain=NonNegativeReals,
                        doc="Energy [kWh] recharged at station f on leg i→i+1")

    # Charging times (used in the piecewise-linear charging function)
    m.t_prime = Var(m.Z, domain=NonNegativeReals,
                    doc="Time [h] to charge battery from 0 to y'_{i,f} (virtual)")
    m.t_dbl_prime = Var(m.Z, domain=NonNegativeReals,
                        doc="Time [h] to charge from y'_{i,f} to y'_{i,f}+y''_{i,f}")

    # Time-before-break counters (HOS tracking)
    m.t_b = Var(m.N, domain=NonNegativeReals,
                doc="Remaining drive time before mandatory break on arrival at i")
    m.t_b_prime = Var(m.N, domain=NonNegativeReals,
                  doc="Remaining drive time before mandatory break on arrival at CS after i")

    # -------------------------------------------------------------------------
    # OBJECTIVE  (Eq. 1)
    # -------------------------------------------------------------------------
    m.obj = Objective(expr=m.T_total, sense=minimize,
                      doc="Minimize total route duration")

    # -------------------------------------------------------------------------
    # CONSTRAINT: T_total decomposition  (Eq. 1 split)
    # -------------------------------------------------------------------------
    m.c_total = Constraint(
        expr=m.T_total == m.T_drive + m.T_stop,
        doc="Eq.1: T_total = T_drive + T_stop"
    )

    # -------------------------------------------------------------------------
    # CONSTRAINT: T_drive definition  (Eq. 2)
    # -------------------------------------------------------------------------
    m.c_drive = Constraint(
        expr=m.T_drive == (
            sum(m.T_travel[i, successor[i]] * m.x[i] for i in N_inner) +
            sum((m.T_travel[i, f] + m.T_travel[f, successor[i]]) * m.z[i, f]
                for (i, f) in m.Z if i in successor)
        ),
        doc="Eq.2: driving time = direct legs + detour legs"
    )

    # -------------------------------------------------------------------------
    # CONSTRAINT: T_stop definition  (Eq. 3)
    # -------------------------------------------------------------------------
    m.c_stop = Constraint(
        expr=m.T_stop == (
            sum(m.B_break * m.w[i] for i in m.N) +
            sum(m.t_dbl_prime[i, f] for (i, f) in m.Z)
        ),
        doc="Eq.3: stop time = sum of breaks + sum of charging times"
    )

    # -------------------------------------------------------------------------
    # CONSTRAINT: Max driving per day  (Eq. 4)
    # -------------------------------------------------------------------------
    m.c_drive_day = Constraint(
        expr=m.T_drive <= m.W_day,
        doc="Eq.4: total driving <= W^day (EU: 9 h)"
    )

    # -------------------------------------------------------------------------
    # CONSTRAINT: Route coverage  (Eq. 5)
    # Every leg i→i+1 must be covered by exactly one arc (direct or one detour)
    # -------------------------------------------------------------------------
    def c_route_cover(mdl, i):
        if i not in successor:
            return Constraint.Skip
        return (mdl.x[i] +
                sum(mdl.z[i, f] for f in mdl.F if (i, f) in mdl.Z) == 1)

    m.c_route_cover = Constraint(m.N, rule=c_route_cover,
                                 doc="Eq.5: each leg covered by direct arc or one detour")

    # -------------------------------------------------------------------------
    # ENERGY CONSTRAINTS
    # -------------------------------------------------------------------------

    # Eq. 6 – energy update on direct arc i → i+1
    def c_energy_direct(mdl, i):
        if i not in successor:
            return Constraint.Skip
        ip1 = successor[i]
        return (mdl.y[ip1] <=
                mdl.y[i] - mdl.D_dist[i, ip1] * mdl.h + mdl.M * (1 - mdl.x[i]))

    m.c_energy_direct = Constraint(m.N, rule=c_energy_direct,
                                   doc="Eq.6: energy propagation on direct arc")

    if not ORIGINAL:
        def c_energy_direct_lb(mdl, i):
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.y[ip1] >=
                    mdl.y[i] - mdl.D_dist[i, ip1] * mdl.h - mdl.M * (1 - mdl.x[i]))

        m.c_energy_direct_lb = Constraint(m.N, rule=c_energy_direct_lb,
                                    doc="")

    # Eq. 7 – energy on arrival at charger f
    def c_energy_arrive_f(mdl, i, f):
        return (mdl.y_prime[i, f] <=
                mdl.y[i] - mdl.D_dist[i, f] * mdl.h + mdl.M * (1 - mdl.z[i, f]))

    m.c_energy_arrive_f = Constraint(m.Z, rule=c_energy_arrive_f,
                                     doc="Eq.7: energy on arrival at charger f")

    if not ORIGINAL:
        def c_energy_arrive_f_lb(mdl, i, f):
            return (mdl.y_prime[i, f] >=
                    mdl.y[i] - mdl.D_dist[i, f] * mdl.h - mdl.M * (1 - mdl.z[i, f]))

        m.c_energy_arrive_f_lb = Constraint(m.Z, rule=c_energy_arrive_f_lb,
                                     doc="")

    # Eq. 8 – energy update after charging at f, travelling to i+1
    def c_energy_leave_f(mdl, i, f):
        if i not in successor:
            return Constraint.Skip
        ip1 = successor[i]
        return (mdl.y[ip1] <=
                mdl.y_prime[i, f] + mdl.y_dbl_prime[i, f]
                - mdl.D_dist[f, ip1] * mdl.h
                + mdl.M * (1 - mdl.z[i, f]))

    m.c_energy_leave_f = Constraint(m.Z, rule=c_energy_leave_f,
                                    doc="Eq.8: energy propagation after charging")

    if not ORIGINAL:
        def c_energy_leave_f_lb(mdl, i, f):
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.y[ip1] >=
                    mdl.y_prime[i, f] + mdl.y_dbl_prime[i, f]
                    - mdl.D_dist[f, ip1] * mdl.h
                    - mdl.M * (1 - mdl.z[i, f]))

        m.c_energy_leave_f_lb = Constraint(m.Z, rule=c_energy_leave_f_lb,
                                    doc="")

    # Eq. 9 – safety reserve at every customer
    def c_safety_cust(mdl, i):
        return mdl.y[i] >= mdl.D_safety * mdl.h

    m.c_safety_cust = Constraint(m.N, rule=c_safety_cust,
                                 doc="Eq.9: minimum SOC at customer locations")

    if ORIGINAL:
        # Eq. 10 – safety reserve on arrival at charger
        def c_safety_charger(mdl, i, f):
            return mdl.y_prime[i, f] >= mdl.D_safety * mdl.h

        m.c_safety_charger = Constraint(m.Z, rule=c_safety_charger,
                                        doc="Eq.10: minimum SOC on arrival at charger")
    else:
        # Eq. 10 – safety reserve on arrival at charger
        def c_safety_charger(mdl, i, f):
            return mdl.y_prime[i, f] >= mdl.D_safety * mdl.h - mdl.M * (1 - mdl.z[i, f])

        m.c_safety_charger = Constraint(m.Z, rule=c_safety_charger,
                                        doc="Eq.10: minimum SOC on arrival at charger")

    # Eq. 11 – battery capacity at customer
    def c_cap_cust(mdl, i):
        return mdl.y[i] <= mdl.Y_max

    m.c_cap_cust = Constraint(m.N, rule=c_cap_cust,
                              doc="Eq.11: SOC cannot exceed battery capacity at customer")

    if ORIGINAL:
        # Eq. 12 – battery capacity after charging
        def c_cap_charger(mdl, i, f):
            return mdl.y_prime[i, f] + mdl.y_dbl_prime[i, f] <= mdl.Y_max

        m.c_cap_charger = Constraint(m.Z, rule=c_cap_charger,
                                    doc="Eq.12: SOC after charging <= Y_max")
    else:
        # Eq. 12 – battery capacity after charging
        def c_cap_charger(mdl, i, f):
            return mdl.y_prime[i, f] + mdl.y_dbl_prime[i, f] <= mdl.Y_max + mdl.M * (1 - mdl.z[i, f])

        m.c_cap_charger = Constraint(m.Z, rule=c_cap_charger,
                                    doc="Eq.12: SOC after charging <= Y_max if charger visited")

    # -------------------------------------------------------------------------
    # PIECEWISE-LINEAR CHARGING FUNCTION  (Eqs. 13–18)
    # The charging curve E(t) is approximated by |R| secant lines.
    # a[i,f,r] = 1 selects the active segment when arriving at charger f.
    # -------------------------------------------------------------------------
    R_list = sorted(data["R"])
    R_max  = max(R_list)

    # Eq. 13 – lower bound of active segment: s_{f,r} <= y'_{i,f}
    def c_seg_lower(mdl, i, f, r):
        return (mdl.s_bp[f, r] + mdl.M * (mdl.a[i, f, r] - 1)
                - (1 - mdl.z[i, f]) <= mdl.y_prime[i, f])

    m.c_seg_lower = Constraint(m.Z, m.R, rule=c_seg_lower,
                               doc="Eq.13: SOC >= s_{f,r} when segment r active")

    if ORIGINAL:
        # Eq. 14 – upper bound of active segment: y'_{i,f} <= s_{f,r+1}
        def c_seg_upper(mdl, i, f, r):
            if r == R_max:
                return Constraint.Skip   # last segment has no upper breakpoint
            return (mdl.y_prime[i, f] <=
                    mdl.s_bp[f, r + 1] + mdl.M * (mdl.a[i, f, r] - 1)
                    - (1 - mdl.z[i, f]))

        m.c_seg_upper = Constraint(m.Z, m.R, rule=c_seg_upper,
                                doc="Eq.14: SOC <= s_{f,r+1} when segment r active")
    else:
        # Eq. 14 – upper bound of active segment: y'_{i,f} <= s_{f,r+1}
        def c_seg_upper(mdl, i, f, r):
            if r == R_max:
                return Constraint.Skip   # last segment has no upper breakpoint
            return (mdl.y_prime[i, f] <=
                    mdl.s_bp[f, r + 1] + mdl.M * (1 - mdl.a[i, f, r])
                    + mdl.M *  (1 - mdl.z[i, f]))

        m.c_seg_upper = Constraint(m.Z, m.R, rule=c_seg_upper,
                                doc="Eq.14: SOC <= s_{f,r+1} when segment r active")

    # Eq. 15 – exactly one segment active per charging visit
    def c_seg_sum(mdl, i, f):
        return sum(mdl.a[i, f, r] for r in mdl.R) == mdl.z[i, f]

    m.c_seg_sum = Constraint(m.Z, rule=c_seg_sum,
                             doc="Eq.15: one segment active iff charger is visited")

    # Eqs. 16–17 – link y'_{i,f} to t'_{i,f} via the secant line
    # y'_{i,f} = K_{f,r} * t'_{i,f} + B_{f,r}  (exact when segment r active)
    def c_charge_lb(mdl, i, f, r):
        return (mdl.y_prime[i, f] >=
                mdl.K_grad[f, r] * mdl.t_prime[i, f] + mdl.B_int[f, r]
                - mdl.M * (1 - mdl.a[i, f, r]))

    m.c_charge_lb = Constraint(m.Z, m.R, rule=c_charge_lb,
                               doc="Eq.16: lower bound of charging function")

    def c_charge_ub(mdl, i, f, r):
        return (mdl.y_prime[i, f] <=
                mdl.K_grad[f, r] * mdl.t_prime[i, f] + mdl.B_int[f, r]
                + mdl.M * (1 - mdl.a[i, f, r]))

    m.c_charge_ub = Constraint(m.Z, m.R, rule=c_charge_ub,
                               doc="Eq.17: upper bound of charging function")

    # Eq. 18 – total charging time: maps (y'_{i,f} + y''_{i,f}) → t' + t''
    def c_charge_total(mdl, i, f, r):
        return ((mdl.y_prime[i, f] + mdl.y_dbl_prime[i, f]) <=
                mdl.K_grad[f, r] * (mdl.t_prime[i, f] + mdl.t_dbl_prime[i, f])
                + mdl.B_int[f, r]
                + mdl.M * (1 - mdl.a[i, f, r]))

    m.c_charge_total = Constraint(m.Z, m.R, rule=c_charge_total,
                                  doc="Eq.18: charging time for total recharged energy")

    if not ORIGINAL:
        def c_charge_total_lb(mdl, i, f, r):
            return ((mdl.y_prime[i, f] + mdl.y_dbl_prime[i, f]) >=
                    mdl.K_grad[f, r] * (mdl.t_prime[i, f] + mdl.t_dbl_prime[i, f])
                    + mdl.B_int[f, r]
                    - mdl.M * (1 - mdl.a[i, f, r]))

        m.c_charge_total_lb = Constraint(m.Z, m.R, rule=c_charge_total_lb,
                                  doc="Eq.18: charging time for total recharged energy (lower bound)")
    # -------------------------------------------------------------------------
    # HOS BREAK CONSTRAINTS  (Eqs. 19–28)
    # The truck must take a break of >= B hours every W_break driving hours.
    # w[i]=1 means the break is taken at customer i.
    # w_prime[i]=1 means the break is taken at charger after i (during charging).
    # -------------------------------------------------------------------------

    # Eqs. 19–20: charging time at f counts as a break only if >= B hours
    def c_bac_lb(mdl, i, f):
        # charging time t''_{i,f} <= B  when it does NOT count as break
        return (mdl.t_dbl_prime[i, f] - mdl.W_break * mdl.w[i]
                <= mdl.B_break + mdl.M * (1 - mdl.z[i, f]))

    m.c_bac_lb = Constraint(m.Z, rule=c_bac_lb,
                            doc="Eq.19: charging stop may serve as break")

    if ORIGINAL:
        def c_bac_ub(mdl, i, f):
            return (mdl.t_dbl_prime[i, f] - mdl.W_break * mdl.w[i]
                    + mdl.M * (1 - mdl.z[i, f]) <= mdl.B_break - mdl.W_break)

        m.c_bac_ub = Constraint(m.Z, rule=c_bac_ub,
                                doc="Eq.20: break-and-charge lower bound")
    else:
        def c_bac_ub(mdl, i, f):
            return (mdl.t_dbl_prime[i, f] - mdl.W_break * mdl.w[i]
                    + mdl.M * (1 - mdl.z[i, f]) >= mdl.B_break - mdl.W_break)

        m.c_bac_ub = Constraint(m.Z, rule=c_bac_ub,
                                doc="Eq.20: break-and-charge lower bound")

    # Eqs. 21–22: remaining drive time before required break at customer i+1
    # t^b_{i+1} tracks how close the driver is to the 4.5-h break limit.
    def c_tb_direct_ub(mdl, i):
        """Upper bound on t^b at i+1 via direct arc (resets if break taken)."""
        if i not in successor:
            return Constraint.Skip
        ip1 = successor[i]
        return (mdl.t_b[ip1] <=
                mdl.t_b[i] - mdl.T_travel[i, ip1]
                + mdl.W_break * (1 - mdl.x[i] + mdl.w[i]
                                + sum(mdl.z[i, f] for f in mdl.F
                                    if (i, f) in mdl.Z)))

    m.c_tb_direct_ub = Constraint(m.N, rule=c_tb_direct_ub,
                                doc="Eq.21: t_b update on direct arc (upper)")

    if ORIGINAL:
        def c_tb_direct_lb(mdl, i):
            """Lower bound: t^b at i+1 via direct arc."""
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] <=
                    (mdl.W_break - mdl.T_travel[i, ip1]) * (1 - mdl.x[i] + mdl.w[i])
                    + mdl.W_break * (1 - mdl.x[i] + mdl.w[i]
                                    + sum(mdl.z[i, f] for f in mdl.F
                                        if (i, f) in mdl.Z)))

        m.c_tb_direct_lb = Constraint(m.N, rule=c_tb_direct_lb,
                                    doc="Eq.22: t_b update on direct arc (lower)")
    else:
        def c_tb_direct_ub0(mdl, i):
            """Upper bound on t^b at i+1 via direct arc (resets if break taken)."""
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] <=
                    mdl.t_b[i] - mdl.T_travel[i, ip1]
                    + mdl.W_break * mdl.w[i] + mdl.M * (1 - mdl.x[i])
                                + mdl.M *
                                    sum(mdl.z[i, f] for f in mdl.F
                                        if (i, f) in mdl.Z))

        m.c_tb_direct_ub0 = Constraint(m.N, rule=c_tb_direct_ub0,
                                    doc="Eq.21: t_b update on direct arc (upper)")


        def c_tb_direct_lb(mdl, i):
            """Lower bound: t^b at i+1 via direct arc."""
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] >=
                    mdl.t_b[i] - mdl.T_travel[i, ip1]
                    + mdl.W_break * mdl.w[i] - mdl.M * (1 - mdl.x[i])
                                - mdl.M *
                                    sum(mdl.z[i, f] for f in mdl.F
                                        if (i, f) in mdl.Z))

        m.c_tb_direct_lb = Constraint(m.N, rule=c_tb_direct_lb,
                                    doc="Eq.22: t_b update on direct arc (lower)")

        def c_tb_direct_ub2(mdl, i):
            """Upper bound on t^b at i+1 via direct arc (resets if break taken)."""
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] <=
                    mdl.W_break - mdl.T_travel[i, ip1]
                    + mdl.M * (1 - mdl.w[i]) + mdl.M * (1 - mdl.x[i])
                                + mdl.M *
                                    sum(mdl.z[i, f] for f in mdl.F
                                        if (i, f) in mdl.Z))

        m.c_tb_direct_ub2 = Constraint(m.N, rule=c_tb_direct_ub2,
                                    doc="Eq.21: t_b update on direct arc (upper)")


        def c_tb_direct_lb2(mdl, i):
            """Lower bound: t^b at i+1 via direct arc."""
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] >=
                    mdl.W_break - mdl.T_travel[i, ip1]
                    - mdl.M * (1 - mdl.w[i]) - mdl.M * (1 - mdl.x[i])
                                - mdl.M *
                                    sum(mdl.z[i, f] for f in mdl.F
                                        if (i, f) in mdl.Z))

        m.c_tb_direct_lb2 = Constraint(m.N, rule=c_tb_direct_lb2,
                                    doc="Eq.22: t_b update on direct arc (lower)")



    # Eqs 23: domain of t_b
    def c_tb_domain(mdl, i):
        return mdl.t_b[i] <= mdl.W_break

    m.c_tb_domain = Constraint(m.N, rule=c_tb_domain,
                               doc="Eq.23: t_b in [0, W_break]")

    # Eqs. 24-25: t^b at charger after customer i (upper & lower)
    def c_tb_f_ub(mdl, i, f):
        return (mdl.t_b_prime[i] <=
                mdl.t_b[i] - mdl.T_travel[i, f]
                + mdl.W_break * (1 - mdl.z[i, f] + mdl.w[i]))

    m.c_tb_f_ub = Constraint(m.Z, rule=c_tb_f_ub,
                            doc="Eq.24: t_b on arrival at charger f (upper)")

    if ORIGINAL:
        def c_tb_f_lb(mdl, i, f):
            return (mdl.t_b_prime[i] <=
                    (mdl.W_break - mdl.T_travel[i, f]) * (1 - mdl.z[i, f] + mdl.w[i]) + mdl.M * (1 - mdl.z[i, f]))

        m.c_tb_f_lb = Constraint(m.Z, rule=c_tb_f_lb,
                                doc="Eq.25: t_b on arrival at charger f (lower)")
    else:
        def c_tb_f_ub0(mdl, i, f):
            return (mdl.t_b_prime[i] <=
                    mdl.t_b[i] - mdl.T_travel[i, f] + mdl.M * mdl.w[i]
                    + mdl.M * (1 - mdl.z[i, f] + mdl.w[i]))

        m.c_tb_f_ub0 = Constraint(m.Z, rule=c_tb_f_ub0,
                                doc="Eq.24: t_b on arrival at charger f (upper)")

        def c_tb_f_lb(mdl, i, f):
            return (mdl.t_b_prime[i] >=
                    mdl.t_b[i] - mdl.T_travel[i, f] - mdl.M * mdl.w[i]
                    - mdl.M * (1 - mdl.z[i, f] + mdl.w[i]))

        m.c_tb_f_lb = Constraint(m.Z, rule=c_tb_f_lb,
                                doc="Eq.25: t_b on arrival at charger f (lower)")

        def c_tb_f_ub2(mdl, i, f):
            return (mdl.t_b_prime[i] <=
                    mdl.W_break - mdl.T_travel[i, f] + mdl.M * (1 - mdl.w[i])
                    + mdl.M * (1 - mdl.z[i, f] + mdl.w[i]))

        m.c_tb_f_ub2 = Constraint(m.Z, rule=c_tb_f_ub2,
                                doc="Eq.24: t_b on arrival at charger f (upper)")

        def c_tb_f_lb2(mdl, i, f):
            return (mdl.t_b_prime[i] >=
                    mdl.W_break - mdl.T_travel[i, f] - mdl.M * (1 - mdl.w[i])
                    - mdl.M * (1 - mdl.z[i, f] + mdl.w[i]))
        m.c_tb_f_lb2 = Constraint(m.Z, rule=c_tb_f_lb2,
                                doc="Eq.25: t_b on arrival at charger f (lower)")

    # Eqs 26: domain of t_b_prime
    def c_tb_prime_domain(mdl, i):
        return mdl.t_b_prime[i] <= mdl.W_break

    m.c_tb_prime_domain = Constraint(m.N, rule=c_tb_prime_domain,
                                     doc="Eq.26: t_b' in [0, W_break]")


    # Eqs. 27-28: t^b at i+1
    def c_tb_via_f_ub(mdl, i, f):
        if i not in successor:
            return Constraint.Skip
        ip1 = successor[i]
        return (mdl.t_b[ip1] <=
                mdl.t_b[i] - (mdl.T_travel[i, f] + mdl.T_travel[f, ip1]) + mdl.W_break * (1 - mdl.z[i, f] + mdl.w_prime[i]))

    m.c_tb_via_f_ub = Constraint(m.Z, rule=c_tb_via_f_ub,
                                doc="Eq.27: t_b at i+1 via charger (upper)")
    if ORIGINAL:
        def c_tb_via_f_lb(mdl, i, f):
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] <=
                    (mdl.W_break - mdl.T_travel[f, ip1])
                    * (1 - mdl.z[i, f] + mdl.w_prime[i])
                    + mdl.M * (1 - mdl.z[i, f]))

        m.c_tb_via_f_lb = Constraint(m.Z, rule=c_tb_via_f_lb,
                                    doc="Eq.26: t_b at i+1 via charger (lower)")
    else:
        def c_tb_via_f_ub0(mdl, i, f):
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] <=
                    mdl.t_b[i] - mdl.T_travel[f, ip1] + mdl.M * mdl.w_prime[i] + mdl.M * (1 - mdl.z[i, f]))

        m.c_tb_via_f_ub0 = Constraint(m.Z, rule=c_tb_via_f_ub0,
                                    doc="Eq.27: t_b at i+1 via charger (upper)")

        def c_tb_via_f_lb(mdl, i, f):
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] >=
                    mdl.t_b[i] - mdl.T_travel[f, ip1] - mdl.M * mdl.w_prime[i] - mdl.M * (1 - mdl.z[i, f]))

        m.c_tb_via_f_lb = Constraint(m.Z, rule=c_tb_via_f_lb,
                                    doc="Eq.26: t_b at i+1 via charger (lower)")

        def c_tb_via_f_ub2(mdl, i, f):
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] <=
                    mdl.W_break - mdl.T_travel[f, ip1] + mdl.M * (1 - mdl.w_prime[i]) + mdl.M * (1 - mdl.z[i, f]))

        m.c_tb_via_f_ub2 = Constraint(m.Z, rule=c_tb_via_f_ub2,
                                    doc="Eq.27: t_b at i+1 via charger (upper)")

        def c_tb_via_f_lb2(mdl, i, f):
            if i not in successor:
                return Constraint.Skip
            ip1 = successor[i]
            return (mdl.t_b[ip1] >=
                    mdl.W_break - mdl.T_travel[f, ip1] - mdl.M * (1 - mdl.w_prime[i]) - mdl.M * (1 - mdl.z[i, f]))

        m.c_tb_via_f_lb2 = Constraint(m.Z, rule=c_tb_via_f_lb2,
                                    doc="Eq.26: t_b at i+1 via charger (lower)")

    return m, successor


# =============================================================================
# SOLVER WRAPPER
# =============================================================================

def solve_model(model, solver_name="gurobi", options=None):
    """
    Solve the Pyomo model with the specified solver.

    The paper uses Gurobi 11.0.0 (ref: Section 4.2).
    Free alternatives: 'glpk', 'cbc', 'highs' (via pip install highspy).

    Parameters
    ----------
    model  : Pyomo ConcreteModel
    solver_name : str  – 'gurobi' | 'cplex' | 'cbc' | 'glpk' | 'highs'
    options : dict     – solver-specific options, e.g. {'TimeLimit': 3600}

    Returns
    -------
    results : Pyomo SolverResults object
    """
    solver = SolverFactory(solver_name)
    if options:
        for k, v in options.items():
            solver.options[k] = v
    results = solver.solve(model, tee=True)
    return results


# =============================================================================
# RESULTS EXTRACTION
# =============================================================================

def extract_results(model, successor):
    """
    Print and return key results from a solved model instance.

    Returns
    -------
    dict with T_total, T_drive, T_stop, route (list of stops), charging_stops
    """
    print(f"\n{'='*60}")
    print(f"  Optimal Total Route Duration : {value(model.T_total):.4f} h")
    print(f"  Driving Time                 : {value(model.T_drive):.4f} h")
    print(f"  Stop Time (breaks+charging)  : {value(model.T_stop):.4f} h")
    print(f"{'='*60}")

    route = []
    for i in model.N:
        if i in successor:
            if value(model.x[i]) > 0.5:
                route.append((i, successor[i], "direct"))
            else:
                for f in model.F:
                    if (i, f) in model.Z and value(model.z[i, f]) > 0.5:
                        route.append((i, successor[i], f"via_{f}",
                                      f"charge={value(model.y_dbl_prime[i,f]):.1f}kWh",
                                      f"time={value(model.t_dbl_prime[i,f])*60:.1f}min"))

    print("\nRoute decisions:")
    for leg in route:
        print(" ", leg)

    print("\nBreaks taken at customer locations:")
    for i in model.N:
        if value(model.w[i]) > 0.5:
            print(f"  Customer {i}: break taken (45 min)")

    for (i,f) in model.Z:
        if value(model.t_dbl_prime[i, f]) > 0:
            print(f"  CS {f} from customer {i}: charge time = {value(model.t_dbl_prime[i,f])*60:.1f} min")

    print("\nEnergy levels at customer locations [kWh]:")
    print("\nEnergy levels at customer locations [kWh]:")
    for i in model.N:
        print(f"  y[{i}] = {value(model.y[i]):.1f} kWh")

    print("\nSummary of the route:")

    T_drive = 0.0 # cumulative driving time
    T_break = 0.0 # cumulative break time
    T_charge = 0.0 # cumulative charging time

    for i in model.N:

        print(f"  Customer {i}")
        print(f"    Time of arrival: {T_drive+T_break+T_charge:.2f} h")
        print(f"    y={value(model.y[i]):.1f} kWh")
        print(f"    w={value(model.w[i])}")
        print(f"    t_b={value(model.t_b[i]):.2f} h")

        T_break += value(model.w[i]) * value(model.B_break)


        for f in model.F:
            if (i, f) in model.Z and value(model.z[i, f]) > 0.5:
                T_drive += value(model.T_travel[i, f])
                print(f"    Detour: via {f}")
                print(f"      Time of arrival: {T_drive+T_break+T_charge:.2f} h")
                print(f"      y'={value(model.y_prime[i,f]):.1f} kWh on arrival at {f}")
                print(f"      y''={value(model.y_dbl_prime[i,f]):.1f} kWh recharged at {f}")
                print(f"      t''={value(model.t_dbl_prime[i,f])*60:.1f} min to charge at {f}")
                print(f"      t'_b={value(model.t_b_prime[i]):.2f} h")
                print(f"      w'={value(model.w_prime[i])}")

                T_break += value(model.w_prime[i]) * value(model.t_dbl_prime[i,f])
                T_charge += value(model.t_dbl_prime[i, f])
                T_drive += value(model.T_travel[f, successor[i]])

        T_drive += sum(model.T_travel[i, successor[i]] * value(model.x[i]) for i in model.N if i in successor)


    return {
        "T_total": value(model.T_total),
        "T_drive": value(model.T_drive),
        "T_stop":  value(model.T_stop),
        "route":   route,
    }


# =============================================================================
# TEST INSTANCES
# =============================================================================

def _charging_params(Y_max=480.0, K0=350.0, K1=120.0):
    """Return piecewise-linear charging params for a single station."""
    bp   = 0.80 * Y_max
    B1   = bp - K1 * (bp / K0)
    return bp, K0, K1, B1


def _make_data(N, F, Z, D, W_break=4.5, W_day=9.0, B=0.75,
               Y_max=480.0, h=0.85, D_safety=50.0, M=1e5, speed=80.0,
               K_vals=None):
    """
    Build a complete data dict from distances D and topology (N, F, Z).

    D      : dict of distances in km for every needed (a, b) pair
    K_vals : optional dict  {f: (K0, K1)}  per station; defaults to 350/120 kW
    """
    T = {k: v / speed for k, v in D.items()}
    R = [0, 1]

    s, K, B_int = {}, {}, {}
    for f in F:
        k0, k1 = (K_vals or {}).get(f, (350.0, 120.0))
        bp, K0, K1, Bint1 = _charging_params(Y_max, k0, k1)
        s[f, 0]     = 0.0;   s[f, 1]     = bp
        K[f, 0]     = K0;    K[f, 1]     = K1
        B_int[f, 0] = 0.0;   B_int[f, 1] = Bint1

    return dict(N=N, F=F, R=R, Z=Z, T_travel=T, D_dist=D,
                h=h, D_safety=D_safety, Y_max=Y_max,
                B=B, W_break=W_break, W_day=W_day, M=M,
                s=s, K=K, B_intercept=B_int, y0=Y_max)


# ---------------------------------------------------------------------------
# INSTANCE 1 – No charging needed, charger is out of the way
#
# Expected: x[0]=1 (direct arc), no charger visit.
# Route: 0 → 1  (300 km, 3.75 h)
# Battery: 480 - 300*0.85 = 225 kWh  → well above safety (42.5 kWh)
# HOS:     3.75 h < 4.5 h limit → no break needed
# Charger f1 adds a 60 km detour each way → z[0,f1] must not be chosen.
# ---------------------------------------------------------------------------
def make_instance_no_charging_needed():
    """
    Charger is available but out of the way and not needed energetically.
    Expected optimal: direct arc, no stop.
    """
    N = [0, 1]
    F = ["f1"]
    Z = [(0, "f1")]
    D = {
        (0, 1):      300.0,          # direct leg: 300 km, 3.75 h
        (0, "f1"):   200.0,          # detour adds 200+160=360 km vs 300
        ("f1", 1):   160.0,
    }
    return _make_data(N, F, Z, D)


# ---------------------------------------------------------------------------
# INSTANCE 2 – Charging is necessary (battery would run out otherwise)
#
# Expected: z[0,f1]=1, truck charges enough to complete the route.
# Route: 0 → f1 → 1  (total 650 km)
# Direct would require: 530 km × 0.85 = 450.5 kWh; battery is 480 kWh but
# safety reserve is 50*0.85=42.5 kWh so usable = 437.5 kWh → infeasible direct.
# Via charger: 300 km to f1 (255 kWh used, 225 kWh remaining) then charge,
# then 350 km from f1 to 1 (297.5 kWh needed).
# ---------------------------------------------------------------------------
def make_instance_charging_necessary():
    """
    Single leg that exceeds battery range without a stop.
    Expected optimal: detour via charger.
    """
    N = [0, 1]
    F = ["f1"]
    Z = [(0, "f1")]
    D = {
        (0, 1):      530.0,   # direct: 530*0.85=450.5 kWh > 437.5 usable
        (0, "f1"):   300.0,
        ("f1", 1):   350.0,
    }
    return _make_data(N, F, Z, D)


# ---------------------------------------------------------------------------
# INSTANCE 3 – HOS break required but NOT at a charger
#
# Two legs both under battery range. Total drive = 5.5 h > W_break=4.5 h
# so a break is mandatory. No charger is on a useful path.
# Expected: x[0]=1, x[1]=1, w[1]=1 (break at customer 1).
# ---------------------------------------------------------------------------
def make_instance_break_at_customer():
    """
    HOS break required; charger is so far out of the way it is never chosen.
    Expected: direct arcs, break at the intermediate customer.
    """
    N = [0, 1, 2]
    F = ["f1"]
    Z = [(0, "f1"), (1, "f1")]
    D = {
        (0, 1):      280.0,   # 3.5 h
        (1, 2):      160.0,   # 2.0 h  → total 5.5 h > 4.5 h limit
        (0, "f1"):   400.0,   # huge detour → never chosen
        ("f1", 1):   400.0,
        (1, "f1"):   400.0,
        ("f1", 2):   400.0,
    }
    return _make_data(N, F, Z, D)


# ---------------------------------------------------------------------------
# INSTANCE 4 – Break-and-charge: charger on route, long enough to be a break
#
# Leg 0→1 is 3.0 h; leg 1→2 is 3.0 h.  Total = 6 h > W_break.
# Charger f1 lies between 1 and 2, adding only 10 km detour.
# Charging needed (leg 1→2 = 350 km, available after leg 0→1 = 480-200=280 kWh,
# need 350*0.85=297.5 kWh → must charge at least 297.5-280+42.5 = 60 kWh).
# Charging 60 kWh at 350 kW takes ~10 min < 45 min → w_prime must be 0.
# Still need a break somewhere.  The model should place w[1]=1.
# ---------------------------------------------------------------------------
def make_instance_charge_and_separate_break():
    """
    Charging is needed AND a HOS break is needed, but charging time < 45 min
    so the charging stop alone cannot serve as the break.
    Expected: z[1,f1]=1 (charge), w[1]=1 (separate break at customer 1).
    """
    N = [0, 1, 2]
    F = ["f1"]
    Z = [(1, "f1")]
    D = {
        (0, 1):      240.0,   # 3.0 h, uses 204 kWh → y[1]=276 kWh
        (1, 2):      350.0,   # direct would use 297.5 kWh → y[2]=276-297.5=-21 infeasible
        (1, "f1"):    50.0,   # small detour
        ("f1", 2):   305.0,   # 305*0.85=259.25 kWh needed from charger
    }
    return _make_data(N, F, Z, D)


# ---------------------------------------------------------------------------
# INSTANCE 5 – Break-and-charge: charging time >= 45 min serves as the break
#
# Leg 0→1 is 3.0 h; leg 1→2 needs heavy charging (low SOC on arrival).
# Battery after leg 0→1: 480 - 380*0.85 = 157 kWh.
# Leg to charger: 30 km → SOC at charger = 157 - 25.5 = 131.5 kWh.
# Leg from charger to 2: 330 km needs 280.5 kWh → need at least 149 kWh charge.
# At 350 kW, 149 kWh takes ~25 min; but HOS requires 45 min break after 3.0 h
# of driving into leg 1→f1 (another 0.375 h = 3.375 h total).
# Next leg f1→2 adds 4.125 h → total from last break = 3.375+4.125 = 7.5 h >> 4.5 h
# So w_prime[1] must be 1 and t'' >= 45 min → truck charges >= 350*0.75 = 262.5 kWh.
# ---------------------------------------------------------------------------
def make_instance_break_and_charge():
    """
    Charging stop naturally lasts >= 45 min (large energy deficit),
    so it can serve as the mandatory HOS break (w_prime=1).
    Expected: z[1,f1]=1, w_prime[1]=1, no separate customer break needed.
    """
    N = [0, 1, 2]
    F = ["f1"]
    Z = [(1, "f1")]
    D = {
        (0, 1):      380.0,   # 4.75 h, y[1] = 480-323=157 kWh
        (1, 2):      450.0,   # direct: needs 382.5 kWh > 157-42.5=114.5 available
        (1, "f1"):    30.0,   # short detour to charger
        ("f1", 2):   330.0,   # 330*0.85=280.5 kWh needed
    }
    return _make_data(N, F, Z, D)


# ---------------------------------------------------------------------------
# INSTANCE 6 – Two chargers available, only the closer one should be chosen
#
# Leg 0→1: 350 km. Both f1 and f2 can service it; f1 is 20 km detour,
# f2 is 100 km detour. Both provide enough charge.
# Expected: z[0,f1]=1 (f1 chosen, shorter detour), z[0,f2]=0.
# ---------------------------------------------------------------------------
def make_instance_two_chargers_pick_closer():
    """
    Two chargers available on the same leg; model should pick the closer one.
    Expected: z[0,f1]=1, z[0,f2]=0.
    """
    N = [0, 1]
    F = ["f1", "f2"]
    Z = [(0, "f1"), (0, "f2")]
    D = {
        (0, 1):       430.0,  # direct: 430*0.85=365.5 > usable 437.5 kWh?
                               # y[1] = 480-365.5=114.5 > 42.5 → direct feasible
                               # but both chargers are faster
        (0, "f1"):    200.0,  ("f1", 1): 220.0,   # detour: +220+200-430=−10 → shorter!
        (0, "f2"):    260.0,  ("f2", 1): 270.0,   # detour: +260+270-430=+100 → longer
    }
    # Make direct infeasible so charging is forced
    D[(0, 1)] = 530.0  # 530*0.85=450.5 > 437.5 → must charge
    return _make_data(N, F, Z, D)


# ---------------------------------------------------------------------------
# INSTANCE 7 – Three legs, charger only needed on the middle leg
#
# Expected: x[0]=1, z[1,f1]=1, x[2]=1.
# The first and last legs are short; only the middle leg is long enough
# to require charging.
# ---------------------------------------------------------------------------
def make_instance_charge_middle_leg_only():
    """
    Three-leg route; charging only necessary on leg 1→2.
    Expected: direct on legs 0→1 and 2→3, charger detour on leg 1→2.
    """
    N = [0, 1, 2, 3]
    F = ["f1"]
    Z = [(0, "f1"), (1, "f1"), (2, "f1")]
    D = {
        (0, 1):      100.0,   # short: 85 kWh used
        (1, 2):      480.0,   # long: 408 kWh > 395-42.5=352.5 usable → must charge
        (2, 3):      100.0,   # short
        (0, "f1"):   200.0, ("f1", 1): 200.0,  # detour on leg 0→1: adds 400-100=300km
        (1, "f1"):   200.0, ("f1", 2): 280.0,  # detour on leg 1→2: small
        (2, "f1"):   200.0, ("f1", 3): 200.0,  # detour on leg 2→3: adds 300km
    }
    return _make_data(N, F, Z, D)


def make_example_data(instance=1):
    """
    Return a data dict for one of the test instances.

    instance : int 1–7
      1  No charging needed, charger out of the way  → expect: direct, no stop
      2  Charging necessary (range exceeded)          → expect: charger detour
      3  HOS break needed, no useful charger          → expect: break at customer
      4  Charge needed + break needed separately      → expect: z=1, w=1
      5  Break-and-Charge (charging time >= 45 min)  → expect: z=1, w'=1
      6  Two chargers: pick the closer one            → expect: z[0,f1]=1
      7  Three legs: charge only on middle leg        → expect: z[1,f1]=1 only
    """
    dispatch = {
        1: make_instance_no_charging_needed,
        2: make_instance_charging_necessary,
        3: make_instance_break_at_customer,
        4: make_instance_charge_and_separate_break,
        5: make_instance_break_and_charge,
        6: make_instance_two_chargers_pick_closer,
        7: make_instance_charge_middle_leg_only,
    }
    assert instance in dispatch, f"instance must be 1–7, got {instance}"
    data = dispatch[instance]()
    print(f"[Instance {instance}] {dispatch[instance].__doc__.strip().splitlines()[0]}")
    return data

def validate_solution(model, successor, data, tol=1e-4):
    """
    Simulate the optimal route chronologically and verify:
      1. Route coverage  – every leg has exactly one arc chosen
      2. SOC evolution   – energy consumed matches distances, stays in bounds
      3. Charging        – energy gained matches time × rate (piecewise linear)
      4. HOS regulation  – cumulative drive time never exceeds W_break without a break
      5. Break logic     – w=1 only when a real break of length B is taken;
                           w'=1 only when t'' >= B at the charging station

    Returns a dict with keys 'passed' (bool) and 'events' (list of step dicts).
    Prints a formatted report.
    """
    from pyomo.environ import value as V

    W   = data["W_break"]   # 4.5 h
    B   = data["B"]         # 0.75 h
    h   = data["h"]         # kWh/km
    Ym  = data["Y_max"]
    Ds  = data["D_safety"]
    N_s = sorted(data["N"])

    PASS = "\033[92m PASS\033[0m"
    FAIL = "\033[91m FAIL\033[0m"

    failures = []
    events   = []
    clock    = 0.0
    soc      = data["y0"]
    hos      = 0.0   # cumulative drive since last break (0 = fresh)

    def fail(msg):
        failures.append(msg)
        return f"{FAIL}  {msg}"

    def ok(msg):
        return f"{PASS}  {msg}"

    def check(cond, pass_msg, fail_msg):
        if cond:
            return ok(pass_msg)
        return fail(fail_msg)

    sep = "─" * 62
    print(f"\n{'═'*62}")
    print("  SOLUTION VALIDATOR")
    print(f"{'═'*62}")

    for idx, i in enumerate(N_s):
        print(f"\n{sep}")
        print(f"  CUSTOMER {i}   clock={clock:.4f}h  SOC={soc:.2f}kWh  HOS={hos:.4f}h")
        print(sep)

        # ── SOC at customer from model ────────────────────────────────────────
        y_model = V(model.y[i])
        ev_soc  = check(abs(y_model - soc) < tol,
                        f"SOC model={y_model:.4f} sim={soc:.4f} match",
                        f"SOC MISMATCH model={y_model:.4f} sim={soc:.4f}")
        print(f"  {ev_soc}")

        # ── SOC bounds ───────────────────────────────────────────────────────
        print(f"  {check(soc >= Ds*h - tol, f'SOC >= safety ({Ds*h:.2f} kWh)', f'SOC BELOW SAFETY: {soc:.4f} < {Ds*h:.4f}')}")
        print(f"  {check(soc <= Ym + tol,   f'SOC <= Y_max ({Ym:.0f} kWh)',    f'SOC EXCEEDS Y_max: {soc:.4f}')}")

        # ── HOS at customer from model ────────────────────────────────────────
        tb_model   = V(model.t_b[i])
        tb_sim     = W - hos
        ev_hos = check(abs(tb_model - tb_sim) < tol,
                       f"t_b model={tb_model:.4f} sim={tb_sim:.4f} match",
                       f"t_b MISMATCH model={tb_model:.4f} sim(W-hos)={tb_sim:.4f}")
        print(f"  {ev_hos}")

        # ── HOS limit ────────────────────────────────────────────────────────
        print(f"  {check(hos <= W + tol, f'Drive since break={hos:.4f}h <= {W}h limit', f'HOS VIOLATION: {hos:.4f}h > {W}h')}")

        w_i = V(model.w[i]) > 0.5

        # ── Break at this customer ────────────────────────────────────────────
        if w_i:
            print(f"  → Break taken ({B*60:.0f} min)")
            clock += B
            hos    = 0.0  # reset drive counter

        # ── No more legs after last node ──────────────────────────────────────
        if i not in successor:
            events.append(dict(node=i, clock=clock, soc=soc, hos=hos))
            break

        ip1 = successor[i]

        # ── Check route coverage ──────────────────────────────────────────────
        x_i   = V(model.x[i]) > 0.5
        z_if  = {f: V(model.z[i, f]) > 0.5
                 for f in data["F"] if (i, f) in data["Z"]}
        n_arcs = int(x_i) + sum(z_if.values())
        print(f"  {check(n_arcs == 1, f'Route coverage: {n_arcs} arc chosen', f'Route coverage ERROR: {n_arcs} arcs active')}")

        # ── Charger detour ────────────────────────────────────────────────────
        if any(z_if.values()):
            f = next(f for f, active in z_if.items() if active)
            T_if  = data["T_travel"][(i, f)]
            T_fi1 = data["T_travel"][(f, ip1)]
            D_if  = data["D_dist"][(i, f)]
            D_fi1 = data["D_dist"][(f, ip1)]

            # Drive i → f
            soc_before_f = soc - D_if * h
            hos          = hos + T_if
            clock        = clock + T_if

            print(f"\n  [Drive to charger {f}: {T_if*60:.1f} min, {D_if:.0f} km]")
            print(f"  {check(soc_before_f >= Ds*h - tol, f'SOC on arrival at {f}: {soc_before_f:.2f} >= {Ds*h:.2f}', f'SOC BELOW SAFETY at {f}: {soc_before_f:.4f}')}")
            print(f"  {check(hos <= W + tol, f'HOS at {f}: {hos:.4f}h <= {W}h', f'HOS VIOLATION at {f}: {hos:.4f}h')}")

            # Check model y_prime
            yp_model = V(model.y_prime[(i, f)])
            print(f"  {check(abs(yp_model - soc_before_f) < tol, f'y_prime model={yp_model:.4f} sim={soc_before_f:.4f}', f'y_prime MISMATCH model={yp_model:.4f} sim={soc_before_f:.4f}')}")

            # Check model t_b_prime
            tbf_model = V(model.t_b_prime[i])
            tbf_sim   = W - hos
            print(f"  {check(abs(tbf_model - tbf_sim) < tol, f't_b_prime model={tbf_model:.4f} sim={tbf_sim:.4f}', f't_b_prime MISMATCH model={tbf_model:.4f} sim={tbf_sim:.4f}')}")

            # Charging
            t2    = V(model.t_dbl_prime[(i, f)])
            y2    = V(model.y_dbl_prime[(i, f)])
            wp    = V(model.w_prime[i]) > 0.5

            # Check break-and-charge logic
            print(f"\n  [Charging at {f}: t''={t2*60:.1f} min, y''={y2:.2f} kWh, w'={int(wp)}]")
            if wp:
                print(f"  {check(t2 >= B - tol, f't'' >= B (break valid): {t2*60:.1f} >= {B*60:.0f} min', f't'' < B but w=1 (invalid break): {t2*60:.1f} min')}")
            else:
                print(f"  {check(t2 <= B + tol, f't'' <= B (no break): {t2*60:.1f} <= {B*60:.0f} min', f't'' > B but w=0 (missed break flag): {t2*60:.1f} min')}")

            # Verify energy gain matches charging rate (piecewise linear)
            soc_total_after = soc_before_f + y2
            # Find active segment
            seg_ok = False
            for r in data["R"]:
                a_r = V(model.a[(i, f, r)]) > 0.5
                if a_r:
                    K   = data["K"][(f, r)]
                    Bi  = data["B_intercept"][(f, r)]
                    t1  = V(model.t_prime[(i, f)])
                    # y' + y'' = K*(t'+t'') + B_int
                    expected = K * (t1 + t2) + Bi
                    seg_ok = abs(soc_total_after - expected) < tol
                    print(f"  {check(seg_ok, f'Charging energy correct (seg {r}): {soc_total_after:.4f} = {expected:.4f}', f'Charging energy ERROR (seg {r}): {soc_total_after:.4f} ≠ {expected:.4f}')}")
                    break
            if not seg_ok and not any(V(model.a[(i, f, r)]) > 0.5 for r in data["R"]):
                print(f"  {fail(f'No active segment for charger {f}')}")

            soc   = soc_total_after
            if wp:
                hos = 0.0  # break at charger resets HOS
            clock = clock + t2

            # Drive f → i+1
            soc   = soc - D_fi1 * h
            hos   = hos + T_fi1
            clock = clock + T_fi1
            print(f"\n  [Drive from {f} to customer {ip1}: {T_fi1*60:.1f} min, {D_fi1:.0f} km]")

        else:
            # Direct arc i → i+1
            T_dir = data["T_travel"][(i, ip1)]
            D_dir = data["D_dist"][(i, ip1)]
            soc   = soc - D_dir * h
            hos   = hos + T_dir
            clock = clock + T_dir
            print(f"\n  [Direct drive to {ip1}: {T_dir*60:.1f} min, {D_dir:.0f} km]")

        events.append(dict(node=i, clock=clock, soc=soc, hos=hos))

    print(f"\n{'═'*62}")
    if failures:
        print(f"  RESULT: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        print("  RESULT: ALL CHECKS PASSED ✓")
    print(f"{'═'*62}\n")

    return {"passed": len(failures) == 0, "events": events, "failures": failures}



if __name__ == "__main__":
    import brockmann_graphs as gr
    for ex in range(1, 8):
        data = make_example_data(ex)
        model, successor = build_bet_tdsp_model(data)

        N_sorted = sorted(data["N"])
        model.y[N_sorted[0]].fix(data["y0"])

        print("Model built successfully.")
        print(f"  Variables : {model.nvariables()}")
        print(f"  Constraints: {model.nconstraints()}")
        print(f"  Objectives : {model.nobjectives()}")

        # Uncomment to solve (requires Gurobi, CBC, or HiGHS):
        try:
            results = solve_model(model, solver_name="highs")
            extract_results(model, successor)
            validate_solution(model, successor, data)
        except Exception as e:
            print(f"Solver error: {e}")
            print("Skipping solution and validation for this instance.")
        #gr.plot_all(model, successor, data)
