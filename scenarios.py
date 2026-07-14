"""
scenarios.py — Scenario generation, tracking, and realisation logging
======================================================================
Centralises everything related to uncertainty representation:

  generate_scenarios(full_data, start_stop, end_stop, ...)
      Draw travel-time + energy scenarios for a look-ahead window.
      Used by Simulation.py at every stop.

  ScenarioTracker
      Keeps a per-stop record of:
        • the generated scenarios (forward-looking, at decision time)
        • the realised uncertainty (after the leg is driven)
      Provides summary statistics and JSON serialisation for post-analysis.

Design
------
Scenario dict schema
    {
        "D": {leg_index: travel_time_h, ...},
        "E": {leg_index: energy_kWh, ...},
        "warm_start": <optional list — injected by Simulation.py>,
    }

Realisation dict schema (recorded by ScenarioTracker.record_realisation)
    {
        "stop"    : int,
        "leg"     : int,          # same as stop (leg from stop to stop+1)
        "D_nom"   : float,        # nominal travel time (h)
        "D_actual": float,        # actual drawn travel time (h)
        "E_nom"   : float,        # nominal energy (kWh)
        "E_actual": float,        # actual energy (kWh) — if available
        "mult"    : float,        # D_actual / D_nom
    }

References
----------
Log-normal scenario generation with spatial correlation follows the approach
described in:
  Berthold et al. (2023) "Robust scheduling for electric trucks under travel
  time uncertainty", Transportation Research Part C, 156, 104325.
  https://doi.org/10.1016/j.trc.2023.104325

The ECR(v) = A/v + B + C·v² formula is taken from:
  Younes et al. (2020) "Energy consumption model for battery electric vehicles",
  Journal of Energy Storage, 32, 101758.
  https://doi.org/10.1016/j.est.2020.101758
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np


# ── ECR curve ─────────────────────────────────────────────────────────────────
from settings import (ecr as _ecr, sample_travel_time, sample_multipliers,
                      V_NOM, TRAVEL_TIME_DIST, TRAVEL_TIME_AR1_RHO)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_scenarios(full_data: dict,
                       start_stop: int,
                       end_stop: int,
                       n_scenarios: int = 10,
                       delta: float = 0.20,
                       seed: Optional[int] = None,
                       include_best: bool = False,
                       include_worst: bool = False,
                       dist: str = None,
                       ar1_rho: float = None) -> list[dict]:
    """
    Draw `n_scenarios` travel-time and energy scenarios for legs
    [start_stop, end_stop).

    Each scenario is a dict:
        "D": {leg: travel_time_h}
        "E": {leg: energy_kWh}   (derived from D via ECR curve)

    S5 — distribution options:
      dist="uniform"   (base case): xi ~ U[1−delta, 1+delta], i.i.d.
      dist="lognormal": lognormal matched to mean 1 and the same CV
                        (robustness check; unbounded — combine with
                        prune_quantile < 1 in the supervisor).
      ar1_rho > 0     : positively correlated deviations across consecutive
                        legs (AR(1) on the normal driver) — congestion
                        persists in time; i.i.d. draws tend to understate the
                        value of adaptive policies.
    Defaults (None) fall back to settings.TRAVEL_TIME_DIST /
    settings.TRAVEL_TIME_AR1_RHO.

    Parameters
    ----------
    full_data     : dict from instances.make_data()
    start_stop    : first leg index (inclusive)
    end_stop      : last  leg index (exclusive)
    n_scenarios   : number of random scenarios to generate
    delta         : uncertainty half-width (e.g. 0.15 = ±15%)
    seed          : RNG seed for reproducibility (None = unseeded)
    include_best  : also append a deterministic best-case scenario (1-delta)
    include_worst : also append a deterministic worst-case scenario (1+delta)

    Returns
    -------
    list of scenario dicts.
    """
    rng   = np.random.default_rng(seed)
    N     = full_data["N"]
    D_nom = full_data["D"]
    L_nom = full_data.get("km", {})   # leg lengths in km (optional)

    dist    = dist    if dist    is not None else TRAVEL_TIME_DIST
    ar1_rho = ar1_rho if ar1_rho is not None else TRAVEL_TIME_AR1_RHO

    legs   = list(range(start_stop, min(end_stop, N)))
    n_legs = len(legs)

    scenarios = []

    for k in range(n_scenarios):
        mults = sample_multipliers(n_legs, rng, delta=delta,
                                   dist=dist, ar1_rho=ar1_rho)
        D_scen: dict[int, float] = {}
        E_scen: dict[int, float] = {}
        for j, leg in enumerate(legs):
            d_s = D_nom.get(leg, 0.0) * float(mults[j])
            D_scen[leg] = d_s
            # Derive energy from ECR: E = L_km * ECR(v_s), v_s = L_km / d_s
            L_km = L_nom.get(leg, D_nom.get(leg, 0.0) * V_NOM)
            v_s  = L_km / d_s if d_s > 0 else V_NOM
            E_scen[leg] = L_km * _ecr(v_s)
        scenarios.append({"D": D_scen, "E": E_scen})

    # ── Deterministic corner cases ─────────────────────────────────────────────
    def _corner_energies(mult: float) -> dict:
        """Compute energy for a deterministic speed-multiplier corner case."""
        E_corner: dict[int, float] = {}
        for leg in legs:
            d_c  = D_nom.get(leg, 0.0) * mult
            L_km = L_nom.get(leg, D_nom.get(leg, 0.0) * V_NOM)
            v_c  = L_km / d_c if d_c > 0 else V_NOM
            E_corner[leg] = L_km * _ecr(v_c)
        return E_corner

    if include_best:
        mult = (1 - delta)
        scenarios.append({
            "D": {l: D_nom.get(l, 0.0) * mult for l in legs},
            "E": _corner_energies(mult),
        })
    if include_worst:
        mult = (1 + delta)
        scenarios.append({
            "D": {l: D_nom.get(l, 0.0) * mult for l in legs},
            "E": _corner_energies(mult),
        })

    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioTracker:
    """
    Records, at every decision stop, the scenarios used for look-ahead and the
    uncertainty that was actually realised on the following leg.

    Typical usage inside run_simulation / run_greedy
    ------------------------------------------------
    tracker = ScenarioTracker(full_data)

    # At each stop, after generating scenarios:
    tracker.record_scenarios(stop, scenarios)

    # After vehicle.advance() returns D_actual:
    tracker.record_realisation(stop, D_actual, E_actual=None)

    # At the end of the run:
    summary = tracker.summary()
    tracker.save("logs/my_run_scenarios.json")

    Attributes
    ----------
    scenarios_by_stop   : dict {stop: list[scenario_dict]}
    realisations        : list[realisation_dict]
    """

    def __init__(self, full_data: dict):
        self._fd              = full_data
        self.scenarios_by_stop: dict[int, list[dict]] = {}
        self.realisations     : list[dict]             = []

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_scenarios(self, stop: int, scenarios: list[dict]):
        """
        Store the look-ahead scenarios generated at `stop`.

        Parameters
        ----------
        stop      : current decision stop
        scenarios : list of scenario dicts (from generate_scenarios)
        """
        # Store a lightweight copy (drop warm_start arrays — large & redundant)
        self.scenarios_by_stop[stop] = [
            {k: v for k, v in s.items() if k != "warm_start"}
            for s in scenarios
        ]

    def record_realisation(self, stop: int, D_actual: float,
                           E_actual: Optional[float] = None):
        """
        Record the actual travel time (and optionally energy) realised on leg
        `stop → stop+1`.

        Parameters
        ----------
        stop     : stop index at which the vehicle just made its decision
        D_actual : actual travel time on leg `stop` (h)
        E_actual : actual energy consumed on leg `stop` (kWh), or None
        """
        D_nom = self._fd["D"].get(stop, 0.0)
        E_nom = self._fd["E"].get(stop, 0.0)
        mult  = D_actual / D_nom if D_nom > 1e-9 else 1.0

        rec = dict(
            stop     = stop,
            leg      = stop,
            D_nom    = round(D_nom, 6),
            D_actual = round(D_actual, 6),
            E_nom    = round(E_nom, 4),
            E_actual = round(E_actual, 4) if E_actual is not None else None,
            mult     = round(mult, 6),
        )
        self.realisations.append(rec)

    # ── Statistics ────────────────────────────────────────────────────────────

    def scenario_stats(self, stop: int) -> Optional[dict]:
        """
        Return mean / std / min / max of D multipliers for scenarios at `stop`.
        Returns None if no scenarios were recorded for that stop.
        """
        scens = self.scenarios_by_stop.get(stop)
        if not scens:
            return None
        D_nom = self._fd["D"]
        mults = []
        for s in scens:
            for leg, d in s["D"].items():
                nom = D_nom.get(leg, 0.0)
                if nom > 1e-9:
                    mults.append(d / nom)
        if not mults:
            return None
        arr = np.array(mults)
        return dict(mean=float(arr.mean()), std=float(arr.std()),
                    min=float(arr.min()), max=float(arr.max()),
                    n_values=len(mults))

    def summary(self) -> dict:
        """
        Return a summary dict with:
          - per-stop scenario stats
          - realisation stats (mean / std of multipliers over the whole route)
          - coverage: fraction of realisations that fell within the scenario fan
        """
        per_stop_stats = {
            s: self.scenario_stats(s)
            for s in sorted(self.scenarios_by_stop)
        }

        # Realisation statistics
        mults = [r["mult"] for r in self.realisations]
        if mults:
            arr = np.array(mults)
            real_stats = dict(mean=float(arr.mean()), std=float(arr.std()),
                              min=float(arr.min()), max=float(arr.max()),
                              n_legs=len(mults))
        else:
            real_stats = None

        # Coverage: realisation within [min_scen_D, max_scen_D] for that leg
        covered = 0; checked = 0
        for r in self.realisations:
            stop  = r["stop"]
            scens = self.scenarios_by_stop.get(stop, [])
            if not scens:
                continue
            leg     = r["leg"]
            scen_ds = [s["D"].get(leg) for s in scens if leg in s["D"]]
            if not scen_ds:
                continue
            lo, hi = min(scen_ds), max(scen_ds)
            covered += int(lo - 1e-9 <= r["D_actual"] <= hi + 1e-9)
            checked += 1

        coverage = covered / checked if checked > 0 else None

        return dict(
            n_stops_with_scenarios = len(self.scenarios_by_stop),
            n_realisations         = len(self.realisations),
            scenario_stats_per_stop= per_stop_stats,
            realisation_stats      = real_stats,
            coverage_fraction      = coverage,
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of the full tracker state."""
        def _ser(o):
            if isinstance(o, (int, float, bool, str, type(None))): return o
            if isinstance(o, dict):  return {str(k): _ser(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [_ser(v) for v in o]
            return str(o)

        return dict(
            scenarios_by_stop = _ser(self.scenarios_by_stop),
            realisations      = self.realisations,
            summary           = self.summary(),
        )

    def save(self, path: str):
        """Save the tracker state to a JSON file."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  ScenarioTracker saved: {path}")

    @classmethod
    def load(cls, path: str, full_data: dict) -> "ScenarioTracker":
        """Restore a ScenarioTracker from a saved JSON file."""
        with open(path) as f:
            state = json.load(f)
        tracker = cls(full_data)
        # Restore integer keys
        tracker.scenarios_by_stop = {
            int(k): v for k, v in state["scenarios_by_stop"].items()
        }
        tracker.realisations = state["realisations"]
        return tracker

    def __repr__(self) -> str:
        n_s = len(self.scenarios_by_stop)
        n_r = len(self.realisations)
        return f"ScenarioTracker(stops_with_scenarios={n_s}, realisations={n_r})"
