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


# ── Battery defaults ───────────────────────────────────────────────────────────
BATTERY_CAPACITY: float = 500.0  # kWh default battery capacity
SOC_MIN_FRAC: float = 0.20       # Emin = SOC_MIN_FRAC * Ecap

# ── PWL charging curve ─────────────────────────────────────────────────────────
# Ebar breakpoints as fractions of Ecap; Tbar is in hours.
EBAR_FRACS: dict[int, float] = {0: 0.0, 1: 0.40, 2: 0.80, 3: 1.0}
TBAR: dict[int, float] = {0: 0.0, 1: 0.55, 2: 1.367, 3: 2.50}

# ── HoS limits (EU Regulation 561/2006) ───────────────────────────────────────
# Reference: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32006R0561
Tb45: float = 0.75   # 45-min break (h)
Tb15: float = 0.25   # split-break part 1 — 15 min (h)
Tb30: float = 0.50   # split-break part 2 — 30 min (h)
Tr1:  float = 11.0   # daily rest (h)
Tr2:  float = 9.0    # reduced daily rest (h)
Tdrv_cons:  float = 4.5   # max consecutive driving (h)
Tdrv_sh1:   float = 9.0   # max shift driving, regular (h)
Tdrv_sh2:   float = 10.0  # max shift driving, split-week (h)
Twrk_cons1: float = 6.0   # max consecutive working (h)
Twrk_cons2: float = 9.0   # max extended consecutive working (h)
Twrk_sh:    float = 13.0  # max shift working (h)

# ── Queue wait time at CS stops — lognormal distribution ──────────────────────
QUEUE_WAIT_MEAN_MIN: float = 10.0  # mean (minutes)
QUEUE_WAIT_STD_MIN:  float = 8.0   # std dev (minutes)

# ── Maneuver / overhead times at CS stops ─────────────────────────────────────
M_STOP_H: float = 10.0 / 60   # stop overhead per CS activity (h)
M_SEQ_H:  float = 5.0 / 60    # sequential-mode repositioning overhead (h)
M_MAN_DEFAULT_H: float = 15.0 / 60  # default maneuver at non-CS stop (h)

# ── Service time at customer stops ────────────────────────────────────────────
SERVICE_TIME_H: float = 0.5  # 30 min per customer delivery (h)

# ── Route / planning defaults ─────────────────────────────────────────────────
CS_SPACING_KM: int = 40    # km between consecutive CS stops
T_START: float = 8.0       # departure time (absolute hours, 08:00)


# ──────────────────────────────────────────────────────────────────────────────
LOWER_PCT = 0.25


def sample_travel_time(
    D_i,
    rng,
    lower_pct = LOWER_PCT,
    upper_pct = LOWER_PCT,
    n = 1,
):
    """Sample travel time uniformly from [D_i*(1-lower_pct), D_i*(1+upper_pct)]."""
    T_min = D_i * (1 - lower_pct)
    T_max = D_i * (1 + upper_pct)
    T = rng.uniform(T_min, T_max, size=n)
    return T if n > 1 else float(T[0])