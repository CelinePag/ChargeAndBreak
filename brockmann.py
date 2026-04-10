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
    m.w_prime = Var(m.Z, domain=Binary,
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
    m.t_b_f = Var(m.Z, domain=NonNegativeReals,
                  doc="Remaining drive time before mandatory break on arrival at f")

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

    # Eq. 7 – energy on arrival at charger f
    def c_energy_arrive_f(mdl, i, f):
        return (mdl.y_prime[i, f] <=
                mdl.y[i] - mdl.D_dist[i, f] * mdl.h + mdl.M * (1 - mdl.z[i, f]))

    m.c_energy_arrive_f = Constraint(m.Z, rule=c_energy_arrive_f,
                                     doc="Eq.7: energy on arrival at charger f")

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

    # Eq. 9 – safety reserve at every customer
    def c_safety_cust(mdl, i):
        return mdl.y[i] >= mdl.D_safety * mdl.h

    m.c_safety_cust = Constraint(m.N, rule=c_safety_cust,
                                 doc="Eq.9: minimum SOC at customer locations")

    # Eq. 10 – safety reserve on arrival at charger
    def c_safety_charger(mdl, i, f):
        return mdl.y_prime[i, f] >= mdl.D_safety * mdl.h

    m.c_safety_charger = Constraint(m.Z, rule=c_safety_charger,
                                    doc="Eq.10: minimum SOC on arrival at charger")

    # Eq. 11 – battery capacity at customer
    def c_cap_cust(mdl, i):
        return mdl.y[i] <= mdl.Y_max

    m.c_cap_cust = Constraint(m.N, rule=c_cap_cust,
                              doc="Eq.11: SOC cannot exceed battery capacity at customer")

    # Eq. 12 – battery capacity after charging
    def c_cap_charger(mdl, i, f):
        return mdl.y_prime[i, f] + mdl.y_dbl_prime[i, f] <= mdl.Y_max

    m.c_cap_charger = Constraint(m.Z, rule=c_cap_charger,
                                 doc="Eq.12: SOC after charging <= Y_max")

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

    # Eq. 14 – upper bound of active segment: y'_{i,f} <= s_{f,r+1}
    def c_seg_upper(mdl, i, f, r):
        if r == R_max:
            return Constraint.Skip   # last segment has no upper breakpoint
        return (mdl.y_prime[i, f] <=
                mdl.s_bp[f, r + 1] + mdl.M * (mdl.a[i, f, r] - 1)
                - (1 - mdl.z[i, f]))

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

    # -------------------------------------------------------------------------
    # HOS BREAK CONSTRAINTS  (Eqs. 19–28)
    # The truck must take a break of >= B hours every W_break driving hours.
    # w[i]=1 means the break is taken at customer i.
    # w_prime[i,f]=1 means the break is taken at charger f (during charging).
    # -------------------------------------------------------------------------

    # Eqs. 19–20: charging time at f counts as a break only if >= B hours
    def c_bac_lb(mdl, i, f):
        # charging time t''_{i,f} <= B  when it does NOT count as break
        return (mdl.t_dbl_prime[i, f] - mdl.W_break * mdl.w[i]
                <= mdl.B_break + mdl.M * (1 - mdl.z[i, f]))

    m.c_bac_lb = Constraint(m.Z, rule=c_bac_lb,
                            doc="Eq.19: charging stop may serve as break")

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

    # Eqs. 23–24: t^b at charger f after customer i (upper & lower)
    def c_tb_f_ub(mdl, i, f):
        return (mdl.t_b_f[i, f] <=
                mdl.t_b[i] - mdl.T_travel[i, f]
                + mdl.M * (1 - mdl.z[i, f]))

    m.c_tb_f_ub = Constraint(m.Z, rule=c_tb_f_ub,
                             doc="Eq.23: t_b on arrival at charger f (upper)")

    def c_tb_f_lb(mdl, i, f):
        return (mdl.t_b_f[i, f] <=
                (mdl.W_break - mdl.T_travel[i, f]) * mdl.z[i, f])

    m.c_tb_f_lb = Constraint(m.Z, rule=c_tb_f_lb,
                             doc="Eq.24: t_b on arrival at charger f (lower)")

    # Eqs. 25–26: t^b at i+1 when travelling via charger f
    def c_tb_via_f_ub(mdl, i, f):
        if i not in successor:
            return Constraint.Skip
        ip1 = successor[i]
        return (mdl.t_b[ip1] <=
                mdl.t_b_f[i, f] + mdl.W_break * mdl.w_prime[i, f]
                - mdl.T_travel[f, ip1]
                + mdl.M * (1 - mdl.z[i, f]))

    m.c_tb_via_f_ub = Constraint(m.Z, rule=c_tb_via_f_ub,
                                 doc="Eq.25: t_b at i+1 via charger (upper)")

    def c_tb_via_f_lb(mdl, i, f):
        if i not in successor:
            return Constraint.Skip
        ip1 = successor[i]
        return (mdl.t_b[ip1] <=
                (mdl.W_break - mdl.T_travel[f, ip1])
                * (mdl.z[i, f] + mdl.w_prime[i, f])
                + mdl.M * (1 - mdl.z[i, f]))

    m.c_tb_via_f_lb = Constraint(m.Z, rule=c_tb_via_f_lb,
                                 doc="Eq.26: t_b at i+1 via charger (lower)")

    # Eqs. 27–28: domain of t_b
    def c_tb_domain(mdl, i):
        return mdl.t_b[i] <= mdl.W_break

    m.c_tb_domain = Constraint(m.N, rule=c_tb_domain,
                               doc="Eq.27-28: t_b in [0, W_break]")

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

    print("\nEnergy levels at customer locations [kWh]:")
    for i in model.N:
        print(f"  y[{i}] = {value(model.y[i]):.1f} kWh")

    return {
        "T_total": value(model.T_total),
        "T_drive": value(model.T_drive),
        "T_stop":  value(model.T_stop),
        "route":   route,
    }


# =============================================================================
# MINIMAL EXAMPLE (2 customers, 1 charger, 1 secant segment)
# =============================================================================

def make_example_data():
    """
    Small toy instance to verify the model builds and solves correctly.

    Route: depot(0) → customer_1 → customer_2 → depot_end(3)
    Charger: 'f1' located between customer 1 and 2
    Battery: 480 kWh (BT480), two secant segments (CC + CV phase)
    """
    N = [0, 1, 2]       # 3 nodes: start depot, 1 customer, end depot
    F = ["f1"]
    R = [0, 1]          # two piecewise segments
    Z = [(0, "f1"), (1, "f1")]   # detour option on both legs

    # All distances in km, travel times in hours (approx 80 km/h)
    D = {
        (0, 1): 200.0, (1, 2): 200.0,
        (0, "f1"): 100.0, ("f1", 1): 105.0,
        (1, "f1"): 100.0, ("f1", 2): 105.0,
    }
    T = {k: v / 80.0 for k, v in D.items()}  # time = dist / speed

    Y_max = 480.0   # kWh
    h     = 0.85    # kWh/km (MAN eTGX typical)

    # Two-segment piecewise charging:
    #  Segment 0 (CC): 0–80% SOC at 350 kW  =>  E = 350*t + 0
    #  Segment 1 (CV): 80–100% SOC at ~120 kW
    bp = 0.80 * Y_max   # 384 kWh
    # Segment 1 intercept: at t=bp/350 hours, E=bp, so B_int_1 = bp - 350*(bp/350) = 0
    # After 384 kWh charged at 350 kW, time elapsed = 384/350 h
    # CV phase: E = 120*t + (384 - 120*(384/350))
    t_bp = bp / 350.0
    B1 = bp - 120.0 * t_bp

    s       = {("f1", 0): 0.0,   ("f1", 1): bp}
    K       = {("f1", 0): 350.0, ("f1", 1): 120.0}
    B_int   = {("f1", 0): 0.0,   ("f1", 1): B1}

    return {
        "N": N, "F": F, "R": R, "Z": Z,
        "T_travel": T,
        "D_dist":   D,
        "h":        h,
        "D_safety": 50.0,
        "Y_max":    Y_max,
        "B":        0.75,       # 45 min break
        "W_break":  4.5,        # 4.5 h max before break
        "W_day":    9.0,        # 9 h max driving per day
        "M":        1e5,
        "s":        s,
        "K":        K,
        "B_intercept": B_int,
        # Initial energy (add as a separate parameter or as y[0] fixed):
        "y0": Y_max,            # start fully charged
    }


if __name__ == "__main__":
    data = make_example_data()
    model, successor = build_bet_tdsp_model(data)

    # Fix initial energy level at the depot/start node
    N_sorted = sorted(data["N"])
    model.y[N_sorted[0]].fix(data["y0"])

    print("Model built successfully.")
    print(f"  Variables : {model.nvariables()}")
    print(f"  Constraints: {model.nconstraints()}")
    print(f"  Objectives : {model.nobjectives()}")

    # Uncomment to solve (requires Gurobi, CBC, or HiGHS):
    results = solve_model(model, solver_name="gurobi")
    extract_results(model, successor)