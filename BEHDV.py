"""
BEHDV.py — Battery Electric Heavy Duty Vehicle
===============================================
Owns vehicle state, history, and the single state-transition method advance().

Responsibilities
----------------
  State storage
      Keeps the complete history of every quantity at every stop arrival:
      t_arr (arrival time), e_arr (battery SOC), cd/sd/sw (HoS accumulators),
      phi (split-break flag), rho2_used (reduced-rest budget consumed).

  State transition
      advance(action, drivetime, milp_sol) executes one stop decision, draws the
      actual travel time, updates all accumulators, and appends the new state.

  Compatibility interface
      as_init_state() returns the current state as a dict for solve_horizon.
      The scalar properties (stop, t_arr, e_arr, …) mirror the old
      VehicleState namedtuple API so that all call sites are unchanged.

  Energy utility
      _energy_after_charging(ea, tauc, full_data) evaluates the PWL charging
      curve analytically.  Imported by oracle.py and greedy.py.

Non-responsibilities (deliberately excluded)
--------------------------------------------
  Action enumeration lives in Simulation.py (enumerate_actions).
    Rationale: enumeration depends on simulation policy flags (charge_only)
    and pruning rules, making it decision-layer logic rather than vehicle logic.
    BEHDV.py need not import from Simulation.py at all.

  MILP solving, scenario generation, plotting, oracle — all in their own files.

Import chain
------------
  BEHDV.py → numpy, warnings, math, collections (stdlib only; no local imports)
  Simulation.py → BEHDV
  greedy.py     → BEHDV
  oracle.py     → BEHDV (_energy_after_charging)
"""

from __future__ import annotations

import math
import warnings
from collections import namedtuple
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# ENERGY UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _energy_after_charging(ea: float, tauc: float, full_data: dict) -> float:
    """
    Return departure SOC (kWh) after charging for `tauc` hours starting from
    initial SOC `ea`, using the piecewise-linear charging curve defined by
    full_data["Ebar"] (energy breakpoints) and full_data["Tbar"]
    (cumulative-charge-time breakpoints).

    The PWL curve is evaluated analytically: invert ea → t, add tauc, invert
    back t → e.  This is correct for any tauc, including fractional values
    from an LP relaxation.

    Parameters
    ----------
    ea        : float — arrival SOC (kWh), before charging
    tauc      : float — charging duration (h)
    full_data : dict  — must contain Ebar, Tbar, Ecap, Emin

    Returns
    -------
    float — departure SOC clipped to [Emin, Ecap]
    """
    Ebar = full_data["Ebar"]
    Tbar = full_data["Tbar"]
    Ecap = full_data["Ecap"]
    Emin = full_data["Emin"]

    rs = sorted(Ebar)
    Es = [Ebar[r] for r in rs]
    Ts = [Tbar[r] for r in rs]

    def _e2t(e: float) -> float:
        """Map energy → cumulative charge time on the PWL curve."""
        e = max(Es[0], min(Es[-1], e))
        for k in range(len(Es) - 1):
            if Es[k] <= e <= Es[k + 1]:
                span = Es[k + 1] - Es[k]
                return (Ts[k] + (e - Es[k]) / span * (Ts[k + 1] - Ts[k])
                        if span else Ts[k])
        return Ts[-1]

    def _t2e(t: float) -> float:
        """Map cumulative charge time → energy on the PWL curve."""
        t = max(Ts[0], min(Ts[-1], t))
        for k in range(len(Ts) - 1):
            if Ts[k] <= t <= Ts[k + 1]:
                span = Ts[k + 1] - Ts[k]
                return (Es[k] + (t - Ts[k]) / span * (Es[k + 1] - Es[k])
                        if span else Es[k])
        return Es[-1]

    ed = _t2e(_e2t(ea) + tauc)

    if ed < Emin - 1e-3:
        raise ValueError(
            f"[BEHDV._energy_after_charging] Energy violation: "
            f"ea={ea:.2f} kWh + tauc={tauc:.2f} h → ed={ed:.2f} kWh < Emin={Emin:.2f} kWh.  "
            f"Check scenario feasibility and PWL curve definition.",
            stacklevel=2,
        )

    if ed > Ecap + 1e-3:
        raise ValueError(
            f"[BEHDV._energy_after_charging] Energy violation: "
            f"ea={ea:.2f} kWh + tauc={tauc:.2f} h → ed={ed:.2f} kWh > Ecap={Ecap:.2f} kWh.  "
            f"Check scenario feasibility and PWL curve definition.",
            stacklevel=2,
        )

    return ed


def _charging_time_needed(ea: float, full_data: dict) -> float:
    """
    Charging time (h) to bring battery from ``ea`` kWh to full capacity.

    Used by BEHDV.advance as a fallback when milp_sol is unavailable or
    infeasible, ensuring the vehicle actually charges when y=1 rather than
    departing at its current SOC.  Returns 0.0 if already at full capacity.
    """
    Ebar = full_data["Ebar"]
    Tbar = full_data["Tbar"]
    Ecap = full_data["Ecap"]

    rs = sorted(Ebar)
    Es = [Ebar[r] for r in rs]
    Ts = [Tbar[r] for r in rs]

    def _e2t(e):
        e = max(Es[0], min(Es[-1], e))
        for k in range(len(Es) - 1):
            if Es[k] <= e <= Es[k + 1]:
                span = Es[k + 1] - Es[k]
                return (Ts[k] + (e - Es[k]) / span * (Ts[k + 1] - Ts[k])
                        if span else Ts[k])
        return Ts[-1]

    return max(0.0, _e2t(Ecap) - _e2t(ea))


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE SNAPSHOT NAMEDTUPLE
# ══════════════════════════════════════════════════════════════════════════════

_Snapshot = namedtuple(
    "_Snapshot",
    ["stop", "t_arr", "e_arr", "cd", "sd", "sw", "phi", "rho2_used"],
)
"""
Immutable per-stop snapshot.  Returned by BEHDV.states and consumed by
oracle.py, plots.py, and check_simulation_feasibility.  Field names match
the old VehicleState dataclass so all existing call sites are unchanged.
"""


# ══════════════════════════════════════════════════════════════════════════════
# BEHDV CLASS
# ══════════════════════════════════════════════════════════════════════════════

class BEHDV:
    """
    Battery Electric Heavy Duty Vehicle — state tracker and transition engine.

    The vehicle maintains the FULL HISTORY of its state at every stop arrival.
    Each ``*_history`` attribute is a list whose k-th element is the value at
    the k-th stop visited (index 0 = origin, before any driving).

    Scalar properties (``stop``, ``t_arr``, ``e_arr``, ``cd``, ``sd``,
    ``sw``, ``phi``, ``rho2_used``) return the CURRENT (most recent) value
    and are intentionally identical to the old VehicleState field names, so
    all existing function signatures that accept a state object work unchanged.

    Parameters
    ----------
    full_data : dict
        Route and physics data dict produced by instances.make_data().
    """

    def __init__(self, full_data: dict):
        self._fd = full_data

        T_START = full_data.get("T_START", 8.0)
        E0      = full_data["E0"]

        # ── State history at arrival (index = number of stops visited) ────────
        self.stop_history      : list[int]   = [0]
        self.t_arr_history     : list[float] = [T_START]
        self.e_arr_history     : list[float] = [E0]
        self.cd_history        : list[float] = [0.0]
        self.sd_history        : list[float] = [0.0]
        self.sw_history        : list[float] = [0.0]
        self.phi_history       : list[int]   = [0]
        self.rho2_used_history : list[int]   = [0]

        # ── Execution history (index = stop departed from) ────────────────────
        self.actions       : list[dict]  = []   # action taken at each stop
        self.durations     : list[dict]  = []   # {taub, tauc, taur, tauq}
        self.td_list       : list[float] = []   # departure times (h)
        self.D_actual_list : list[float] = []   # actual leg travel times (h)

    # ── Current-state scalar properties ──────────────────────────────────────

    @property
    def stop(self)      -> int:   return self.stop_history[-1]
    @property
    def t_arr(self)     -> float: return self.t_arr_history[-1]
    @property
    def e_arr(self)     -> float: return self.e_arr_history[-1]
    @property
    def cd(self)        -> float: return self.cd_history[-1]
    @property
    def sd(self)        -> float: return self.sd_history[-1]
    @property
    def sw(self)        -> float: return self.sw_history[-1]
    @property
    def phi(self)       -> int:   return self.phi_history[-1]
    @property
    def rho2_used(self) -> int:   return self.rho2_used_history[-1]

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def n_stops_visited(self) -> int:
        """Number of stops arrived at so far (including origin, i.e. ≥ 1)."""
        return len(self.stop_history)

    @property
    def states(self) -> list[_Snapshot]:
        """
        Backward-compatible list of _Snapshot namedtuples, one per arrival.
        Used by oracle.py, plots.py, and check_simulation_feasibility to read
        the trajectory without depending on BEHDV internals.
        """
        return [
            _Snapshot(
                stop      = self.stop_history[k],
                t_arr     = self.t_arr_history[k],
                e_arr     = self.e_arr_history[k],
                cd        = self.cd_history[k],
                sd        = self.sd_history[k],
                sw        = self.sw_history[k],
                phi       = self.phi_history[k],
                rho2_used = self.rho2_used_history[k],
            )
            for k in range(len(self.stop_history))
        ]

    # ── Interface for solve_horizon ───────────────────────────────────────────

    def as_init_state(self) -> dict:
        """
        Return the current state as a dict for MILP.solve_horizon.

        Keys: ta (arrival time), ea (battery SOC), cd, sd, sw, phi.
        This is the only coupling point between BEHDV and the MILP layer.
        """
        return dict(
            ta  = self.t_arr,
            ea  = self.e_arr,
            cd  = self.cd,
            sd  = self.sd,
            sw  = self.sw,
            phi = self.phi,
        )

    # ── State transition ──────────────────────────────────────────────────────

    def advance(
        self,
        action: dict,
        D_next: float,
        E_next: float,
        milp_sol: Optional[dict],
    ):
        """
        Execute ``action`` at the current stop, draw the actual travel time,
        update all accumulators, and append the resulting state to all histories.

        This is the ONLY mutation method of BEHDV.  It advances the vehicle
        from ``self.stop`` to ``self.stop + 1``.

        Execution durations (taub, tauc, taur, tauq) are taken from
        ``milp_sol`` when available (the nominal MIP re-solve from
        select_best_action).  When milp_sol is None (e.g. greedy uses a mock
        sol), minimum-required durations are derived from the action dict.

        The actual travel time to the next stop is drawn from ``D_next``.

        State-transition formulas mirror the MILP's cd_prop / sd_prop /
        sw_prop constraints, evaluated with the ACTUAL travel time so that
        the simulation reflects true uncertainty.

        Parameters
        ----------
        action            : dict with keys y, break_type, rest_type
        D_next            : float — use this exact travel time
        E_next            : float — use this exact energy for the next leg
        milp_sol          : dict from solve_horizon (keys: sol, feasible) or None

        Returns
        -------
        td    : float — departure time from the current stop (h)
        D_act : float — actual travel time to the next stop (h)
        """
        full_data = self._fd
        stop      = self.stop
        N         = full_data["N"]
        C_set     = set(full_data["C"])
        K_set     = set(full_data["K"])
        is_CS     = stop in K_set
        is_cust   = stop in C_set
        S_stop    = full_data["S"].get(stop, 0.0)

        # sequence of actions at a stop:
        #   - Q + C + B/R + M (if CS)
        #   - S + B/R + M (if customer)
        # taub is extra break time, b45 possible with taub<45min if taub+tauc>45min


        # ── 1. Extract durations from MILP solution ───────────────────────────
        if milp_sol is not None and milp_sol.get("sol"):
            s0        = milp_sol["sol"][0]
            taub_exec = float(s0["taub"])
            taur_exec = float(s0["taur"])
            tauc_exec = float(s0["tauc"])
            tauq_exec = float(s0["tauq"])
            brk = ("b45" if s0["b45"] else
                   "b15" if s0["b15"] else
                   "b30" if s0["b30"] else None)
            rst = ("r1" if s0["rho1"] else
                   "r2" if s0["rho2"] else None)
            y   = int(s0.get("y", action.get("y", 0)))
        else:
            # Fallback: minimum required durations from the action dict.
            # Used when milp_sol is None (greedy mock sol) or infeasible.
            # When y=1 at a CS stop we compute tauc from the PWL charging
            # curve so the vehicle actually charges rather than departing empty.
            brk       = action.get("break_type")
            rst       = action.get("rest_type")
            taur_exec = (full_data["Tr1"]  if rst == "r1"  else
                         full_data["Tr2"]  if rst == "r2"  else 0.0)
            y         = int(action.get("y", 0))
            tauq_exec = full_data["Q"].get(stop, 0.0) * y if is_CS else 0.0
            if y and is_CS:
                # Charge to full using the PWL curve (same as greedy heuristic)
                tauc_exec = _charging_time_needed(self.e_arr, full_data)
            else:
                tauc_exec = 0.0

            taub_exec = (max(0, full_data["Tb45"] - tauc_exec) if brk == "b45" else
                         max(0, full_data["Tb15"] - tauc_exec) if brk == "b15" else
                         max(0, full_data["Tb30"] - tauc_exec) if brk == "b30" else 0.0)

        # ── 2. Manoeuver time ─────────────────────────────────────────────────
        # Required when:
        #   - a rest is taken (always: proper parking needed for 9-11 h)
        #   - a break is taken WITHOUT simultaneous charging
        #     (break synchronized with charging uses no extra parking;
        #      the driver is already at the CS bay)
        # Charging alone (y=1, no break/rest) does NOT trigger a manoeuver;
        # plug-in/out is already accounted for in Q (queue/setup time).
        _brk_active = brk in ("b45", "b15", "b30")
        _rst_active  = rst in ("r1", "r2")
        _brk_unsync  = _brk_active and not (is_CS and bool(y))
        man_time = (full_data["M"].get(stop, 5.0 / 60)
                    if (_rst_active or _brk_unsync) else 0.0)

        # ── 3. Departure time ─────────────────────────────────────────────────
        if is_CS:
            td = self.t_arr + tauq_exec + tauc_exec + taub_exec + taur_exec + man_time
        elif is_cust:
            td = self.t_arr + S_stop + taub_exec + taur_exec + man_time
        else:
            td = self.t_arr

        self.actions.append(action)
        self.durations.append(
            dict(taub=taub_exec, tauc=tauc_exec, taur=taur_exec, tauq=tauq_exec)
        )
        self.td_list.append(td)

        # ── 4. Actual travel time ─────────────────────────────────────────────
        if stop >= N:
            # Already at destination — record and return without advancing
            self.D_actual_list.append(0.0)
            return td, 0.0

        D_act = float(D_next)
        self.D_actual_list.append(D_act)
        t_arr_new = td + D_act

        # ── 5. Energy update ──────────────────────────────────────────────────
        E_act = float(E_next)
        if y and is_CS and tauc_exec > 0:
            e_dep = _energy_after_charging(self.e_arr, tauc_exec, full_data)
        else:
            e_dep = self.e_arr

        e_new_raw = e_dep - E_act
        if e_new_raw < full_data["Emin"] - 1e-3:
            raise ValueError(
                f"[BEHDV.advance] Energy violation on leg {stop}→{stop+1}: "
                f"ed={e_dep:.2f} − E={E_act:.2f} = {e_new_raw:.2f} kWh "
                f"< Emin={full_data['Emin']:.2f} kWh.  "
                f"Clipping to Emin — check scenario feasibility."
            )

        # ── 6. HoS accumulator update ─────────────────────────────────────────
        # ri  = True iff this stop resets consecutive-driving (b45, b30, or rest)
        # rho = True iff this stop resets shift accumulators (any rest)
        ri  = (brk in ("b45", "b30")) or (rst in ("r1", "r2"))
        rho = rst in ("r1", "r2")

        cd_dep = 0.0 if ri  else self.cd
        sd_dep = 0.0 if rho else self.sd

        if is_CS:
            work_now = (tauq_exec + (tauc_exec if not brk else 0.0)) if not rho else 0.0
        elif is_cust:
            work_now = S_stop if not rho else 0.0
        else:
            work_now = 0.0

        sw_dep = 0.0 if rho else self.sw + work_now     # reset sw on rest

        cd_new = cd_dep + D_act
        sd_new = sd_dep + D_act
        sw_new = sw_dep + man_time + D_act   # carry manoeuver + driving to next stop

        # ── 7. phi (split-break flag) and rho2_used ───────────────────────────
        if ri or rho:
            phi_new = 0
        elif brk == "b15":
            phi_new = 1
        else:
            phi_new = self.phi

        rho2_new = self.rho2_used + (1 if rst == "r2" else 0)

        # ── 8. Append new state ───────────────────────────────────────────────
        self.stop_history.append(stop + 1)
        self.t_arr_history.append(t_arr_new)
        self.e_arr_history.append(e_new_raw)
        self.cd_history.append(cd_new)
        self.sd_history.append(sd_new)
        self.sw_history.append(sw_new)
        self.phi_history.append(phi_new)
        self.rho2_used_history.append(rho2_new)


    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of stops arrived at (including origin)."""
        return len(self.stop_history)

    def __repr__(self) -> str:
        return (
            f"BEHDV("
            f"stop={self.stop}, "
            f"t={self.t_arr:.2f}h, "
            f"soc={self.e_arr:.0f}kWh, "
            f"cd={self.cd:.2f}h, "
            f"sd={self.sd:.2f}h, "
            f"sw={self.sw:.2f}h, "
            f"phi={self.phi}, "
            f"rho2={self.rho2_used})"
        )