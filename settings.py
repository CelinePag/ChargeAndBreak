"""
settings.py — Project-wide shared constants.

Import from here rather than re-defining values in each module.
All times in hours, energies in kWh, distances in km.
"""

from __future__ import annotations
import numpy as np

# ── ECR energy model (Younes et al. 2020) ──────────────────────────────────────
# Formula: ECR(v) = A/v + B + C·v²  (kWh/km)
# Reference: https://doi.org/10.1016/j.est.2020.101758
# Calibration cross-check (E1): a loaded 40-t BET at highway speed should land
# in ~1.0–1.5 kWh/km.  See Nykvist & Olsson (2021, Joule,
# https://doi.org/10.1016/j.joule.2021.06.007) and NACFE Run on Less—Electric
# (https://runonless.com).  ECR(80) ≈ 1.10 kWh/km with the parameters below.
ECR_A: float = 33.055
ECR_B: float = 0.2256
ECR_C: float = 7.2e-5

# Speed range for ECR evaluation — truck operating envelope
ECR_V_MIN: float = 20.0   # km/h
ECR_V_MAX: float = 100.0  # km/h

V_NOM: float = 80.0  # km/h — nominal highway cruising speed


def ecr(v_kmh: float) -> float:
    """Energy consumption rate (kWh/km) at speed v_kmh (km/h)."""
    v = max(ECR_V_MIN, min(float(v_kmh), ECR_V_MAX))
    return ECR_A / v + ECR_B + ECR_C * v ** 2


# E1: assert the calibrated loaded-highway ECR is in the plausible BET range.
assert 1.0 <= ecr(V_NOM) <= 1.5, (
    f"ECR({V_NOM}) = {ecr(V_NOM):.3f} kWh/km outside the calibrated "
    f"[1.0, 1.5] kWh/km range for a loaded 40-t BET (E1)")


# ── Battery defaults ───────────────────────────────────────────────────────────
BATTERY_CAPACITY: float = 500.0  # kWh default battery capacity
# Emin = 20% of capacity: range-anxiety / battery-health buffer (deep discharge
# accelerates degradation and leaves no margin for detours or cold weather).
SOC_MIN_FRAC: float = 0.20       # Emin = SOC_MIN_FRAC * Ecap

# ── PWL charging curve ─────────────────────────────────────────────────────────
# Ebar breakpoints as fractions of Ecap; Tbar is in hours.
# Base case: 500 kWh in 2.5 h ≈ 200 kW average (CCS-class fast charging).
EBAR_FRACS: dict[int, float] = {0: 0.0, 1: 0.40, 2: 0.80, 3: 1.0}
TBAR: dict[int, float] = {0: 0.0, 1: 0.55, 2: 1.367, 3: 2.50}

# I2 sensitivity axis: charger power classes.  The PWL time breakpoints scale
# inversely with average power: Tbar_scaled = Tbar * (P_BASE / P_charger).
CHARGER_POWER_BASE_KW: float = 200.0   # implied by TBAR above (500 kWh / 2.5 h)
CHARGER_POWER_CLASSES_KW: tuple = (150.0, 200.0, 350.0, 1000.0)  # incl. MCS 1 MW


def scale_tbar(power_kw: float, tbar: dict | None = None) -> dict:
    """Return TBAR rescaled to a charger of average power `power_kw` (I2)."""
    base = dict(TBAR if tbar is None else tbar)
    f = CHARGER_POWER_BASE_KW / float(power_kw)
    return {r: t * f for r, t in base.items()}


# ── HoS limits (EU Regulation 561/2006 + Directive 2002/15/EC) ─────────────────
# Reference: http://data.europa.eu/eli/reg/2006/561/oj
Tb45: float = 0.75   # 45-min break (h)
Tb15: float = 0.25   # split-break part 1 — 15 min (h)
Tb30: float = 0.50   # split-break part 2 — 30 min (h)
Tr1:  float = 11.0   # daily rest (h)
Tr2:  float = 9.0    # reduced daily rest (h)
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
# Base case 2 h (Table 5); report the beta-sensitivity {1, 2, 5} h.
BETA_TW: float = 1.0
BETA_TW_CLASSES: tuple = (1.0, 2.0, 5.0)

# ── Queue wait time at CS stops — lognormal distribution ──────────────────────
# S6: Q_i is a KNOWN parameter (expected access delay at the charger), drawn
# once per instance at generation time and visible to every method.  It is NOT
# revealed on arrival and there is no endogenous queuing model.
QUEUE_WAIT_MEAN_MIN: float = 10.0  # mean (minutes)
QUEUE_WAIT_STD_MIN:  float = 8.0   # std dev (minutes)

# ── Maneuver / overhead times at CS stops ─────────────────────────────────────
M_STOP_H: float = 10.0 / 60   # stop overhead per CS activity (h)
M_SEQ_H:  float = 5.0 / 60    # sequential-mode repositioning overhead (h)
M_MAN_DEFAULT_H: float = 15.0 / 60  # default maneuver at non-CS stop (h)

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
# Travel-time uncertainty (S5)
# ──────────────────────────────────────────────────────────────────────────────
# Base case: independent multiplicative deviations xi ~ U[1−δ, 1+δ], δ = 0.15.
# (The old 0.25 base case is kept available as a sensitivity, resolving the
# δ inconsistency flagged in the paper rewrite.)
LOWER_PCT = 0.15

# Available distribution families for the realisation / scenario draws:
#   "uniform"   : xi ~ U[1−δ, 1+δ]                       (base case, bounded)
#   "lognormal" : xi lognormal matched to mean 1 and the same CV as the
#                 uniform (CV = δ/√3), clipped to [1−3δ, 1+3δ] for sanity
TRAVEL_TIME_DIST: str = "uniform"

# AR(1) correlation between consecutive-leg multipliers (0 = i.i.d. base case).
# Positive correlation models persistent congestion.
TRAVEL_TIME_AR1_RHO: float = 0.0


def sample_travel_time(
    D_i,
    rng,
    lower_pct=LOWER_PCT,
    upper_pct=LOWER_PCT,
    n=1,
):
    """Sample travel time uniformly from [D_i*(1-lower_pct), D_i*(1+upper_pct)]."""
    T_min = D_i * (1 - lower_pct)
    T_max = D_i * (1 + upper_pct)
    T = rng.uniform(T_min, T_max, size=n)
    return T if n > 1 else float(T[0])


def sample_multipliers(
    n_legs: int,
    rng,
    delta: float = LOWER_PCT,
    dist: str = TRAVEL_TIME_DIST,
    ar1_rho: float = TRAVEL_TIME_AR1_RHO,
):
    """
    S5 — Draw a vector of `n_legs` travel-time multipliers xi.

    Parameters
    ----------
    n_legs  : number of legs
    rng     : np.random.Generator
    delta   : uncertainty half-width (uniform) / CV-matching parameter (lognormal)
    dist    : "uniform" | "lognormal"
    ar1_rho : AR(1) correlation between consecutive legs (0 = independent).
              Implemented on the underlying standard-normal driver, so the
              marginal distribution of each multiplier is preserved.

    Returns
    -------
    np.ndarray of length n_legs with mean ≈ 1 multipliers.
    """
    if n_legs <= 0:
        return np.array([])

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

    # Map z → uniform quantile u in (0,1)
    from math import erf
    u = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
    u = np.clip(u, 1e-9, 1 - 1e-9)

    if dist == "uniform":
        xi = (1.0 - delta) + 2.0 * delta * u
    elif dist == "lognormal":
        # Match mean 1 and the CV of U[1−δ,1+δ]: CV = δ/√3.
        cv = delta / np.sqrt(3.0)
        s2 = np.log(1.0 + cv ** 2)
        mu = -0.5 * s2
        # Inverse-CDF via the normal driver directly (z already standard normal)
        xi = np.exp(mu + np.sqrt(s2) * z)
        xi = np.clip(xi, max(1e-3, 1.0 - 3.0 * delta), 1.0 + 3.0 * delta)
    else:
        raise ValueError(f"unknown travel-time distribution '{dist}'")
    return xi
