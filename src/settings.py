"""
settings.py — Project-wide shared constants.

Import from here rather than re-defining values in each module.
All times in hours, energies in kWh, distances in km.
"""

from __future__ import annotations
import numpy as np

# ── ECR energy model — simplified road load ────────────────────────────────────
# ECR(v) = A/v + B + C·v²  (kWh/km) is not a fitted curve: it is the road-load
# equation divided by speed, so every coefficient is DERIVED from a physical
# parameter rather than regressed.
#
#     A/v   auxiliaries            P_aux / v
#     B     rolling resistance     C_rr·m·g / η
#     C·v²  aerodynamic drag       ½·ρ·C_d·A_f·v² / η
#
# Parameters follow the 40-t tractor-semitrailer of
#   Earl, Mathieu, Cornelis, Kenny, Calvo Ambel & Nix (2018), "Analysis of
#   long haul battery electric trucks in EU", 8th Commercial Vehicle Workshop,
#   Graz — European Federation for Transport and Environment.
#   https://www.transportenvironment.org/uploads/files/20180725_T.pdf
# Running the block below with C_d = 0.60 reproduces their published 1.44 kWh/km
# at 90 km/h, and with C_d = 0.36 their 1.15 kWh/km — the model is theirs.
VEH_MASS_KG:     float = 40000.0  # EU max GVW, 5-axle artic (Directive 96/53/EC)
C_RR:            float = 0.0055   # C-rated tyres (Earl et al. 2018)
FRONTAL_AREA_M2: float = 10.0     # 2.5 m wide × 4.0 m high (Directive 96/53/EC)
RHO_AIR:         float = 1.2      # kg/m³ (Earl et al. 2018)
ETA_DRIVETRAIN:  float = 0.85     # BET total drivetrain (Earl et al. 2018, Tab. 1)
# Continuous cabin HVAC + electrical load.  Earl et al. exclude auxiliaries;
# VECTO (Reg. (EU) 2017/2400) models them explicitly (air compressor, HVAC,
# alternator, cooling fan, steering pump) and the BET powertrain literature puts
# the aggregate at ~3.2 kW.  Only material below ~40 km/h.
P_AUX_KW:        float = 3.2
# The one judgement call.  Earl et al. bracket it: 0.60 = 2018 EU fleet average,
# 0.36 = best in class but requires a shortened trailer (less load volume).  0.50
# reflects the aerodynamic redesign of current long-haul BETs (Mercedes eActros
# 600, Volvo FH Aero).  The bracket spans 1.09–1.32 kWh/km at 80 km/h and is the
# natural sensitivity axis if the base value is challenged.
C_D:             float = 0.50

_G = 9.81  # m/s²
# N·km → kWh is a factor 1/3600; v in km/h → (v/3.6)² introduces the 12.96.
ECR_A: float = P_AUX_KW
ECR_B: float = C_RR * VEH_MASS_KG * _G / 3600.0 / ETA_DRIVETRAIN
ECR_C: float = (0.5 * RHO_AIR * C_D * FRONTAL_AREA_M2
                / 12.96 / 3600.0 / ETA_DRIVETRAIN)

# Speed range for ECR evaluation — truck operating envelope
ECR_V_MIN: float = 20.0   # km/h
ECR_V_MAX: float = 100.0  # km/h

V_NOM: float = 80.0  # km/h — nominal highway cruising speed


def ecr(v_kmh: float) -> float:
    """Energy consumption rate (kWh/km) at speed v_kmh (km/h)."""
    v = max(ECR_V_MIN, min(float(v_kmh), ECR_V_MAX))
    return ECR_A / v + ECR_B + ECR_C * v ** 2


# E1: the derived ECR must land in the range measured for 40-t battery electric
# tractors — 1.1–1.4 kWh/km.  Anchors: 1.3 kWh/km on a 40-t Stockholm–Malmö run
# (600 kWh usable, cold); ~1.3 kWh/km for electric tractor units in real
# operation; Öko-Institut real-world BET data for Germany.  ECR(80) ≈ 1.23.
assert 1.1 <= ecr(V_NOM) <= 1.4, (
    f"ECR({V_NOM}) = {ecr(V_NOM):.3f} kWh/km outside the measured "
    f"[1.1, 1.4] kWh/km range for a loaded 40-t BET (E1)")


# ── Battery defaults ───────────────────────────────────────────────────────────
BATTERY_CAPACITY: float = 500.0  # kWh default battery capacity
# Emin = 20% of capacity: range-anxiety / battery-health buffer (deep discharge
# accelerates degradation and leaves no margin for detours or cold weather).
SOC_MIN_FRAC: float = 0.20       # Emin = SOC_MIN_FRAC * Ecap

# ── PWL charging curve ─────────────────────────────────────────────────────────
# Two concave segments, DERIVED from two physical inputs — the charge point's
# rated output P and the pack capacity Ecap:
#
#   0 → 80 % SOC    charger-limited, flat at the rated output P.
#   80 → 100 % SOC  battery-limited (CC→CV transition), flat at
#                   min(P, TAIL_C_RATE * Ecap).
#
# Ebar breakpoints are fractions of Ecap; Tbar is cumulative hours.
EBAR_KNEE:  float = 0.80   # SOC at which the pack, not the charger, takes over
EBAR_FRACS: dict[int, float] = {0: 0.0, 1: EBAR_KNEE, 2: 1.0}

# Tail acceptance as a fraction of Ecap per hour: 0.40/h = 200 kW on a 500 kWh
# pack.  This is a BATTERY property, so it does NOT rise with charger power once
# the charger exceeds it, and a charge point rated below it never tapers at all.
# Modelling it as charger-proportional (as the old scale_tbar did) had a 150 kW
# point tapering to 86 kW — below its own rated output, which is unphysical.
TAIL_C_RATE: float = 0.40

# Base case: 350 kW, the minimum station output that Regulation (EU) 2023/1804
# (AFIR) mandates every 60 km on the TEN-T core network by 2030
# (http://data.europa.eu/eli/reg/2023/1804/oj).  Current tractors accept this
# over CCS: Volvo FH Aero 350 kW, MAN eTGX 375 kW, Mercedes eActros 600 400 kW.
# NOTE: these are RATED outputs, not 0-100 % averages.  The base curve averages
# 304 kW over a full charge and sustains the full 350 kW across 20-80 % SOC.
CHARGER_POWER_BASE_KW: float = 350.0
# I2 sensitivity axis.  700 kW = Volvo FH Aero / MAN eTGX over MCS;
# 1000 kW = Mercedes eActros 600 over MCS; 150 kW = pre-AFIR / depot CCS.
CHARGER_POWER_CLASSES_KW: tuple = (150.0, 350.0, 700.0, 1000.0)


def charging_curve(power_kw: float, ecap: float = BATTERY_CAPACITY) -> dict:
    """PWL cumulative-time breakpoints (h) for a charge point rated `power_kw`.

    Flat at the rated output up to EBAR_KNEE, then flat at the pack's tail
    acceptance.  Keys align with EBAR_FRACS.  The base case and every power
    class go through this one function, so Tbar cannot drift out of sync with
    the power it claims to represent.
    """
    p      = float(power_kw)
    ecap   = float(ecap)
    p_tail = min(p, TAIL_C_RATE * ecap)
    t: dict[int, float] = {0: 0.0}
    for r in sorted(EBAR_FRACS)[1:]:
        de   = (EBAR_FRACS[r] - EBAR_FRACS[r - 1]) * ecap
        rate = p if EBAR_FRACS[r] <= EBAR_KNEE + 1e-12 else p_tail
        t[r] = t[r - 1] + de / rate
    return t


TBAR: dict[int, float] = charging_curve(CHARGER_POWER_BASE_KW)

# Emin = SOC_MIN_FRAC * Ecap confines the truck to the 20-80 % band, where the
# charger must be the binding constraint (mirrors the E1 ECR check above).
_e_lo   = SOC_MIN_FRAC * BATTERY_CAPACITY
_e_hi   = EBAR_KNEE * BATTERY_CAPACITY
_p_band = (_e_hi - _e_lo) / (TBAR[1] - _e_lo / CHARGER_POWER_BASE_KW)
assert abs(_p_band - CHARGER_POWER_BASE_KW) < 1e-9, (
    f"base curve sustains {_p_band:.1f} kW over the {100*SOC_MIN_FRAC:.0f}-"
    f"{100*EBAR_KNEE:.0f}% band, not the rated {CHARGER_POWER_BASE_KW:.0f} kW")


# ── HoS limits (EU Regulation 561/2006 + Directive 2002/15/EC) ─────────────────
# Reference: http://data.europa.eu/eli/reg/2006/561/oj
Tb45: float = 0.75   # 45-min break (h)
Tb15: float = 0.25   # split-break part 1 — 15 min (h)
Tb30: float = 0.50   # split-break part 2 — 30 min (h)
Tr1:  float = 11.0   # daily rest (h)
Tr2:  float = 9.0    # reduced daily rest (h)

# Art. 7 second subparagraph PERMITS the 45-min break to be split into a 15-min
# block followed by a 30-min block.  It is an option, not an obligation, so the
# no-split regime is a legal operating policy: False drops x_b15/x_b30 from the
# models and forces every break to be the unsplit 45 min (8.3 sensitivity axis).
ALLOW_SPLIT_BREAK: bool = True

Tdrv_cons:  float = 4.5   # max consecutive driving (h)
Tdrv_sh1:   float = 9.0   # max shift driving, regular (h)
Tdrv_sh2:   float = 10.0  # max shift driving, extended (h) — Art. 6(1)
Twrk_cons1: float = 6.0   # working-time break trigger, Directive 2002/15/EC (h)
Twrk_cons2: float = 9.0   # working-time 45-min break trigger (h)
Twrk_sh:    float = 13.0  # legacy shift-working cap (h) — superseded by spread (M5)

# M5 — shift spread (elapsed on-duty time since the end of the last daily rest).
# Derived from "daily rest completed within 24h": 24 − 11 = 13 h under a full
# rest, 24 − 9 = 15 h under a reduced rest.
T_SPR1: float = 13.0  # max spread before a REGULAR daily rest (h)
T_SPR2: float = 15.0  # max spread before a REDUCED daily rest (h) & global cap

# M9 — weekly caps and exception budgets (per week, i.e. per route here since
# each route is completed between two weekly rests — see paper §3.4(a)).
TWK_60:  float = 60.0  # weekly working-time cap, Directive 2002/15/EC (h)
TWK_DRV: float = 56.0  # weekly driving cap, Reg. 561/2006 Art. 6(2) (h)
RHO_BAR: int = 3       # reduced daily rests allowed between weekly rests (Art. 8(4))
EXT_BAR: int = 2       # extended (10 h) driving shifts allowed per week (Art. 6(1))

# TW2 — fixed out-of-window service penalty in the objective
#     min  ta[N] + BETA_TW * sum(delta_i),   delta_i ∈ {0,1}.
# Expressed in objective-hours per missed window (early = late = same cost:
# the disruption is being unannounced, not its magnitude — paper §3.1).
# Base case 0.5 h (30 min) — one missed window costs the same as half an hour
# of route duration.  Report the beta-sensitivity {1, 2, 5} h above it.
BETA_TW: float = 0.5
BETA_TW_CLASSES: tuple = (0.5, 1.0, 2.0, 5.0)

# ── Queue wait time at CS stops — lognormal distribution ──────────────────────
# S6: Q_i is a KNOWN parameter (expected access delay at the charger), drawn
# once per instance at generation time and visible to every method.  It is NOT
# revealed on arrival and there is no endogenous queuing model.
QUEUE_WAIT_MEAN_MIN: float = 10.0  # mean (minutes)
QUEUE_WAIT_STD_MIN:  float = 8.0   # std dev (minutes)

# ── Maneuver / overhead times at CS stops ─────────────────────────────────────
M_STOP_H: float = 10.0 / 60   # stop overhead per CS activity (h)
M_SEQ_H:  float = 5.0 / 60    # sequential-mode repositioning overhead (h)
M_MAN_DEFAULT_H: float = 10.0 / 60  # default maneuver at non-CS stop (h)

# ── Service time at customer stops ────────────────────────────────────────────
SERVICE_TIME_H: float = 0.5  # 30 min per customer delivery (h)

# ── Route / planning defaults ─────────────────────────────────────────────────
# I2: 60 km base spacing per AFIR — Regulation (EU) 2023/1804 mandates HDV
# charging pools every ~60 km on the TEN-T core network
# (http://data.europa.eu/eli/reg/2023/1804/oj).  30/90 km are sensitivities.
CS_SPACING_KM: int = 60
CS_SPACING_CLASSES_KM: tuple = (30, 60, 90)
T_START: float = 8.0       # departure time (absolute hours, 08:00)

# M8 — layby (rest-area) nodes: optional break/rest-only stops inserted along
# long legs.  Gated behind a flag in instance generation (instances.py).
LAYBY_SPACING_KM: float = 25.0     # spacing of layby nodes along a leg
LAYBY_MIN_LEG_H:  float = 0.5      # only insert laybys on legs longer than this
M_LAYBY_H:        float = M_STOP_H # same maneuver overhead as a CS stop (h)


# ──────────────────────────────────────────────────────────────────────────────
# Travel-time uncertainty (S5) — shifted lognormal, bounded by physical speeds
# ──────────────────────────────────────────────────────────────────────────────
# Per-leg multiplicative deviation on the nominal travel time:
#
#     xi = min(XI_MIN + eta, XI_MAX),   eta ~ Lognormal(mu, sigma)
#
# The support is set by PHYSICAL speed bounds, not by a statistical choice:
#   - fast side : EU HGVs carry a mandatory 90 km/h speed limiter
#                 (Directive 92/6/EEC), so xi >= V_NOM / 90.  The lognormal
#                 density of eta vanishes at 0, so the distribution approaches
#                 this floor smoothly (no probability atom on the fast side).
#   - slow side : a leg-average congestion floor of 50 km/h (e.g. half of a
#                 25 km leg jammed at 30 km/h around a city ring), so
#                 xi <= V_NOM / 50 = 1.6.  At the base CV this cap sits at
#                 ~q99 of the unclipped law and absorbs ~1% of the mass.
#
# Calibration: sigma is set from the target CV of xi via the unclipped
# relation; mu is then recalibrated numerically (closed-form clipped mean +
# bisection) so that E[xi] = 1 EXACTLY after the cap — the nominal plan stays
# an unbiased benchmark.
#
# The hard support lets the conservative box RO keep its Soyster logic with
# probability-1 feasibility: time corner at XI_MAX, energy corner at XI_MIN
# (fastest speed, highest ECR).
V_LIMITER_KMH: float = 90.0   # HGV speed limiter (Directive 92/6/EEC)
V_FLOOR_KMH:   float = 50.0   # worst leg-average speed under congestion
XI_MIN: float = V_NOM / V_LIMITER_KMH   # ≈ 0.889 — fast bound
XI_MAX: float = V_NOM / V_FLOOR_KMH     # = 1.6   — slow bound

# TARGET coefficient of variation of xi — an INPUT to the calibration, NOT the
# dispersion you get out.  sigma is solved from the UNCLIPPED relation and only
# mu is re-solved after the cap (see lognormal_params), so the mass above XI_MAX
# is folded onto XI_MAX.  A point sitting on the boundary carries far less
# variance than the tail it replaced, and the realised sd therefore always falls
# short of the target — by more as the target grows and more mass hits the cap:
#
#     target    realised sd    shortfall    legs at cap
#      0.10       0.0958         -4.2%         0.26%
#      0.15       0.1249        -16.7%         1.08%     <- base
#      0.25       0.1530        -38.8%         2.55%
#
# ALWAYS report xi_realised_sd(cv), never the target: the labels overstate the
# spread, and because the shortfall is non-linear the sensitivity axis is
# compressed at the top end (0.25 delivers only ~23% more dispersion than the
# base, not the ~67% the labels imply).
#
# The realised base value (0.125) sits inside the empirically reported range for
# road travel-time CV (~0.08-0.17, motorways at the lower end; Van Lint & Van
# Zuylen 2005, TRR 1917; Tu, Van Lint & Van Zuylen 2007, TRR 1993), so the base
# case remains defensible — it is the LABEL that was wrong, not the model.
TRAVEL_TIME_CV_TARGET: float = 0.15
TRAVEL_TIME_CV_TARGET_CLASSES: tuple = (0.10, 0.15, 0.25)

# AR(1) correlation between consecutive-leg multipliers (0 = i.i.d. base case).
# With ~25 km legs one congestion event spans several consecutive legs, so
# rho ≈ 0.4 is the standard sensitivity.
TRAVEL_TIME_AR1_RHO: float = 0.0

# One-step feasibility guard level (greedy decision rule, LA RH2 action
# pruning, and the opt-in S1 supervisor).  Probability level on xi:
#   0.95  — guard against the 95% quantile of the multiplier (xi ≈ 1.25 at
#           the base CV); residual risk alpha = 5% per leg is reported.
#   1.0   — guard the full support corners [XI_MIN, XI_MAX].
#   None  — guard DISABLED (default): greedy's rule degrades to nominal
#           checks (xi = 1 on both sides) and LA does NO flag-based pruning
#           at all — infeasible actions are exposed by the scenario scores.
GUARD_QUANTILE: float | None = None

_SQRT2 = float(np.sqrt(2.0))


def _phi(z: float) -> float:
    """Standard-normal CDF."""
    from math import erf
    return 0.5 * (1.0 + erf(z / _SQRT2))


def _probit(p: float) -> float:
    """Standard-normal quantile (Acklam's rational approximation, |eps|<1e-9)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    from math import log, sqrt
    if p < 0.02425:
        q = sqrt(-2 * log(p))
        return ((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    if p <= 1 - 0.02425:
        q = p - 0.5
        r = q * q
        return ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
                / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))
    q = sqrt(-2 * log(1 - p))
    return -((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
             / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))


def _clipped_mean(mu: float, sigma: float) -> float:
    """E[min(XI_MIN + eta, XI_MAX)] for eta ~ LN(mu, sigma), closed form."""
    from math import exp, log
    k = XI_MAX - XI_MIN
    d1 = (mu + sigma * sigma - log(k)) / sigma
    d2 = (mu - log(k)) / sigma
    mean_eta = exp(mu + 0.5 * sigma * sigma)
    excess = mean_eta * _phi(d1) - k * _phi(d2)   # E[(eta - k)^+]
    return XI_MIN + mean_eta - excess


_LN_CACHE: dict[float, tuple[float, float]] = {}


def lognormal_params(cv: float = TRAVEL_TIME_CV_TARGET) -> tuple[float, float]:
    """
    (mu, sigma) of the eta driver for a target CV of xi.

    sigma comes from the unclipped relation on eta with mean m = 1 - XI_MIN
    and sd = cv; mu is then shifted (bisection on the closed-form clipped
    mean) so that E[min(XI_MIN + eta, XI_MAX)] = 1 exactly.
    """
    cv = float(cv)
    if cv in _LN_CACHE:
        return _LN_CACHE[cv]
    from math import log, sqrt
    m = 1.0 - XI_MIN
    s2 = log(1.0 + (cv / m) ** 2)
    sigma = sqrt(s2)
    mu0 = log(m) - 0.5 * s2
    lo, hi = mu0 - 0.5, mu0 + 0.5          # clipped mean is increasing in mu
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _clipped_mean(mid, sigma) < 1.0:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    _LN_CACHE[cv] = (mu, sigma)
    return mu, sigma


def xi_realised_sd(cv: float = TRAVEL_TIME_CV_TARGET) -> float:
    """
    Standard deviation of xi that `cv` ACTUALLY produces, in closed form.

    E[xi] = 1 by calibration, so this is also the realised coefficient of
    variation.  It is strictly below `cv` because the cap at XI_MAX collapses
    the upper tail onto a point (see the note beside TRAVEL_TIME_CV_TARGET).
    Quote this in write-ups, not the target.  Matches Monte Carlo to 4 d.p.
    """
    cv = float(cv)
    if cv <= 0.0:
        return 0.0
    from math import exp, log, sqrt
    mu, sigma = lognormal_params(cv)
    k    = XI_MAX - XI_MIN
    lk   = log(k)
    p_gt = 1.0 - _phi((lk - mu) / sigma)          # P(eta > k), the atom mass
    # truncated moments of eta below k, plus the atom at k
    m1 = exp(mu + 0.5 * sigma ** 2) * _phi((lk - mu - sigma ** 2) / sigma) \
        + k * p_gt
    m2 = exp(2 * mu + 2 * sigma ** 2) * _phi((lk - mu - 2 * sigma ** 2) / sigma) \
        + k * k * p_gt
    return sqrt(max(0.0, m2 - m1 * m1))


def xi_atom_mass(cv: float = TRAVEL_TIME_CV_TARGET) -> float:
    """Fraction of legs landing exactly on the XI_MAX cap (P(xi = XI_MAX))."""
    cv = float(cv)
    if cv <= 0.0:
        return 0.0
    from math import log
    mu, sigma = lognormal_params(cv)
    return 1.0 - _phi((log(XI_MAX - XI_MIN) - mu) / sigma)


def xi_quantile(q: float, cv: float = TRAVEL_TIME_CV_TARGET) -> float:
    """
    q-quantile of the bounded multiplier xi (quantiles commute with the cap).
    cv <= 0 degenerates to the deterministic nominal case (xi = 1); q = 1
    returns XI_MAX and q = 0 returns XI_MIN — the RO box corners.
    """
    if cv <= 0.0:
        return 1.0
    if q <= 0.0:
        return XI_MIN
    if q >= 1.0:
        return XI_MAX
    from math import exp
    mu, sigma = lognormal_params(cv)
    return min(XI_MIN + exp(mu + sigma * _probit(q)), XI_MAX)


# LA energy guard (default None = off).  See energy_at_quantile below and
# MILP._build_sub_data: when the LA look-ahead commits an action it re-solves
# the NOMINAL sub-problem, which sizes the charge for nominal energy and
# therefore plans to arrive at the next CS at exactly Emin.  On corridors
# where consecutive charging stations are far apart there is no station at
# which to repair the shortfall, so the committed charge must instead cover
# the legs to the next CS at an ADVERSE energy quantile.
LA_ENERGY_QUANTILE: float | None = None


def energy_at_quantile(km_leg: float, d_nom: float, q: float,
                       cv: float = TRAVEL_TIME_CV_TARGET) -> float:
    """Leg energy (kWh) at the q-quantile of CONSUMPTION.

    Consumption rises with speed (the C·v² term), and speed rises as the
    travel-time multiplier xi FALLS, so the q-quantile of energy is driven by
    the (1−q)-quantile of xi — a "lucky" fast leg is an expensive one.

        q = 0.95  guard against the 95th percentile of consumption
        q = 1.0   worst case: xi = XI_MIN, i.e. the 90 km/h speed limiter

    Returns 0.0 for degenerate legs (zero distance or zero nominal time), so
    ferry legs and the destination sentinel are unaffected.
    """
    if km_leg <= 0.0 or d_nom <= 0.0 or not q:
        return 0.0
    xi_lo = xi_quantile(1.0 - float(q), cv)
    if xi_lo <= 0.0:
        return 0.0
    return km_leg * ecr(km_leg / (d_nom * xi_lo))


def sample_multipliers(
    n_legs: int,
    rng,
    cv: float = TRAVEL_TIME_CV_TARGET,
    ar1_rho: float = TRAVEL_TIME_AR1_RHO,
):
    """
    S5 — Draw a vector of `n_legs` travel-time multipliers xi.

    xi = min(XI_MIN + eta, XI_MAX), eta lognormal, E[xi] = 1 by calibration.

    Parameters
    ----------
    n_legs  : number of legs
    rng     : np.random.Generator
    cv      : target coefficient of variation of xi (0 → xi = 1 exactly)
    ar1_rho : AR(1) correlation between consecutive legs (0 = independent).
              Implemented on the underlying standard-normal driver, so the
              marginal distribution of each multiplier is preserved.

    Returns
    -------
    np.ndarray of length n_legs with mean-1 multipliers in [XI_MIN, XI_MAX].
    """
    if n_legs <= 0:
        return np.array([])
    if cv <= 0.0:
        return np.ones(n_legs)

    # Underlying standard-normal driver z (AR(1) if requested)
    if abs(ar1_rho) > 1e-12:
        z = np.empty(n_legs)
        z[0] = rng.standard_normal()
        innov = rng.standard_normal(n_legs)
        c = np.sqrt(1.0 - ar1_rho ** 2)
        for j in range(1, n_legs):
            z[j] = ar1_rho * z[j - 1] + c * innov[j]
    else:
        z = rng.standard_normal(n_legs)

    mu, sigma = lognormal_params(cv)
    eta = np.exp(mu + sigma * z)
    return np.minimum(XI_MIN + eta, XI_MAX)


def sample_travel_time(
    D_i,
    rng,
    cv: float = TRAVEL_TIME_CV_TARGET,
    n: int = 1,
):
    """Sample realised travel time(s) D_i * xi for one leg (i.i.d. draws)."""
    T = D_i * sample_multipliers(n, rng, cv=cv)
    return T if n > 1 else float(T[0])
