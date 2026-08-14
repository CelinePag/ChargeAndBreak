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
    ["stop", "t_arr", "e_arr", "cd", "sd", "sw", "phi", "rho2_used",
     "ext_shift_used", "h"],
    defaults=(0, 0.0),   # ext_shift_used / h default for backward compatibility
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
    strict_spread : bool, default False
        Record the two Art. 8 spread rules the MILP enforces but the original
        violation semantics never tested: the PRE-REST cap (a daily rest must
        BEGIN within 13 h of the shift start, 15 h if reduced — MILP R24
        spread_prerest) and the TERMINAL cap (h[N] <= 13 — MILP spread_term).

        OFF by default so the feasibility labels of every policy that predates
        this check are unchanged.  greedy.run_greedy opts in; LA and the rest
        keep their previous semantics until they are re-validated separately.
    """

    def __init__(self, full_data: dict, strict_spread: bool = False):
        self._fd = full_data
        self.strict_spread = bool(strict_spread)

        T_START = full_data.get("T_START", 8.0)
        E0      = full_data["E0"]

        # ── State history at arrival (index = number of stops visited) ────────
        self.stop_history           : list[int]   = [0]
        self.t_arr_history          : list[float] = [T_START]
        self.e_arr_history          : list[float] = [E0]
        self.cd_history             : list[float] = [0.0]
        self.sd_history             : list[float] = [0.0]
        self.sw_history             : list[float] = [0.0]
        self.h_history              : list[float] = [0.0]   # M5 shift spread
        self.phi_history            : list[int]   = [0]
        self.rho2_used_history      : list[int]   = [0]
        self.ext_shift_used_history : list[int]   = [0]
        # M9 — cumulative weekly working time.  Unlike sw, this is NOT reset by
        # a daily rest (only a weekly rest, out of model scope, would reset it).
        # The weekly cap (Twk60) is OUT OF PROBLEM SCOPE (2026-07-29): the
        # paper models the daily provisions only, so a breach is recorded as a
        # DIAGNOSTIC note (weekly_notes), never as a run-infeasible violation.
        # The counter itself is kept for the compliance-margin statistic.
        self.sw_week_history        : list[float] = [0.0]
        self._weekly_flagged        : bool        = False
        self.weekly_notes           : list[dict]  = []

        # ── Execution history (index = stop departed from) ────────────────────
        self.actions       : list[dict]  = []   # action taken at each stop
        self.durations     : list[dict]  = []   # {taub, tauc, taur, tauq}
        self.td_list       : list[float] = []   # departure times (h)
        self.D_actual_list : list[float] = []   # actual leg travel times (h)

        # ── S2: violation semantics & metrics ─────────────────────────────────
        # Because uncertainty is revealed only after departure, a leg may
        # retroactively cause a violation that no stop-level action can
        # repair.  The simulator RECORDS these (rather than raising) and the
        # run is marked infeasible for the violating method.
        self.violations    : list[dict]  = []   # {type, stop, amount, detail}
        # TW2/SIM2: out-of-window service starts.  No waiting is ever
        # inserted (SIM1): service starts at arrival; an early or late
        # arrival records delta = 1 with the miss direction and magnitude.
        self.tw_misses     : dict[int, dict] = {}  # {stop: {type, amount}}

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
    @property
    def ext_shift_used(self) -> int: return self.ext_shift_used_history[-1]
    @property
    def h(self) -> float: return self.h_history[-1]
    @property
    def sw_week(self) -> float: return self.sw_week_history[-1]

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
                stop           = self.stop_history[k],
                t_arr          = self.t_arr_history[k],
                e_arr          = self.e_arr_history[k],
                cd             = self.cd_history[k],
                sd             = self.sd_history[k],
                sw             = self.sw_history[k],
                phi            = self.phi_history[k],
                rho2_used      = self.rho2_used_history[k],
                ext_shift_used = self.ext_shift_used_history[k],
                h              = self.h_history[k],
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
            h   = self.h,
            phi = self.phi,
        )

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    _CKPT_LISTS = (
        "stop_history", "t_arr_history", "e_arr_history", "cd_history",
        "sd_history", "sw_history", "h_history", "phi_history",
        "rho2_used_history", "ext_shift_used_history", "sw_week_history",
        "actions", "durations", "td_list", "D_actual_list",
    )

    def to_checkpoint(self) -> dict:
        """JSON-serialisable snapshot of the full execution state, so a crashed
        stop-by-stop run can be resumed instead of restarted (see
        Simulation.run_simulation_precomputed)."""
        ck = {name: list(getattr(self, name)) for name in self._CKPT_LISTS}
        ck["violations"]      = list(self.violations)
        ck["tw_misses"]       = {str(k): v for k, v in self.tw_misses.items()}
        ck["_weekly_flagged"] = bool(self._weekly_flagged)
        ck["weekly_notes"]    = list(self.weekly_notes)
        # stops advanced past the origin == completed loop iterations
        ck["n_done"]          = len(self.stop_history) - 1
        return ck

    def load_checkpoint(self, ck: dict) -> None:
        """Restore the state saved by to_checkpoint()."""
        for name in self._CKPT_LISTS:
            setattr(self, name, list(ck[name]))
        self.violations      = list(ck.get("violations", []))
        self.tw_misses       = {int(k): v
                                for k, v in ck.get("tw_misses", {}).items()}
        self._weekly_flagged = bool(ck.get("_weekly_flagged", False))
        self.weekly_notes    = list(ck.get("weekly_notes", []))

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

        If ``self.stop`` is a customer, the realized arrival is checked
        against its time window [Wha, Whf]: service starts immediately at
        arrival either way (SIM1 — no waiting is ever inserted); an arrival
        outside the window records an early/late window miss (delta = 1,
        fixed penalty) in ``self.tw_misses`` (see step 1.5 below).

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
        L_set     = set(full_data.get("L", []))
        is_CS     = stop in K_set
        is_cust   = stop in C_set
        is_lay    = stop in L_set                 # M8: layby / rest-area node
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

        # ── 1.5. Time-window compliance (customer stops only) ──────────────────
        # SIM1 (v3): NO waiting logic.  Service starts immediately at
        # arrival, whether or not the window is open.  An arrival outside
        # [Wha, Whf] — early or late — sets delta = 1 (fixed penalty in the
        # objective) and is logged with its direction and magnitude (SIM2).
        # Under data["hard_tw"]=True it is additionally recorded as a
        # violation (hard-window sensitivity).
        if is_cust:
            Wha_stop = full_data.get("Wha", {}).get(stop)
            Whf_stop = full_data.get("Whf", {}).get(stop)
            miss = None
            if Whf_stop is not None and self.t_arr > Whf_stop + 1e-3:
                miss = dict(type="late", amount=self.t_arr - Whf_stop)
            elif Wha_stop is not None and self.t_arr < Wha_stop - 1e-3:
                miss = dict(type="early", amount=Wha_stop - self.t_arr)
            if miss is not None:
                self.tw_misses[stop] = miss
                if full_data.get("hard_tw", False):
                    self.violations.append(dict(
                        type="tw_miss", stop=stop, amount=miss["amount"],
                        detail=(f"{miss['type']} arrival {self.t_arr:.3f}h "
                                f"outside [{Wha_stop:.3f}, {Whf_stop:.3f}]h")))

        # ── 2. Activity indicator (v) and sequential-mode indicator (sigma) ─────
        _brk_active = brk in ("b45", "b15", "b30")
        _rst_active = rst in ("r1", "r2")

        if milp_sol is not None and milp_sol.get("sol"):
            sigma = int(s0.get("sigma", 0))
        else:
            # M4: charging co-located with a REST is forced sequential; a
            # break may run in parallel (concurrent) by default.
            sigma = 1 if (y and _rst_active) else 0

        # v=1 whenever any activity (charging, break, rest) occurs at a CS stop
        v_val = int(is_CS and (bool(y) or _brk_active or _rst_active))

        M_stop_val = full_data["M_stop"].get(stop, 0.0) if is_CS else 0.0
        M_seq_val  = full_data["M_seq"].get(stop, 0.0)  if is_CS else 0.0
        mstop_time = v_val * M_stop_val
        mseq_time  = sigma * M_seq_val

        # M8: layby parking overhead — charged once when any break/rest is
        # taken at a layby (mirrors the model term M_lay·Σx, 2SP/MILP).
        M_lay_val  = full_data.get("M_lay", {}).get(stop, 0.0) if is_lay else 0.0
        lay_active = int(is_lay and (_brk_active or _rst_active))
        mlay_time  = lay_active * M_lay_val

        # ── 3. Departure time ─────────────────────────────────────────────────
        # CS:       ta + v·Mstop + Q·y + tauc + taub + taur + sigma·Mseq
        # Customer: ta + S + taub + taur  (no maneuver overhead)
        # Layby:    ta + Mlay·(brk|rst) + taub + taur  (no service, no charging)
        if is_CS:
            td = (self.t_arr + mstop_time + tauq_exec + tauc_exec
                  + taub_exec + taur_exec + mseq_time)
        elif is_cust:
            td = self.t_arr + S_stop + taub_exec + taur_exec
        elif is_lay:
            td = self.t_arr + mlay_time + taub_exec + taur_exec
        else:
            td = self.t_arr

        self.actions.append(action)
        self.durations.append(
            dict(taub=taub_exec, tauc=tauc_exec, taur=taur_exec, tauq=tauq_exec,
                 sigma=sigma, v=v_val, mstop=mstop_time, mseq=mseq_time,
                 mlay=mlay_time)
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
            # S2: stranding event — recorded, run marked infeasible; the SOC
            # is clipped to Emin so the simulation can continue and the full
            # trajectory / metrics remain observable.
            self.violations.append(dict(
                type="stranding", stop=stop + 1,
                amount=full_data["Emin"] - e_new_raw,
                detail=(f"leg {stop}->{stop+1}: ed={e_dep:.2f} - "
                        f"E={E_act:.2f} = {e_new_raw:.2f} kWh "
                        f"< Emin={full_data['Emin']:.2f} kWh")))
            e_new_raw = full_data["Emin"]

        # ── 6. HoS accumulator update ─────────────────────────────────────────
        # ri  = True iff this stop resets consecutive-driving (b45, b30, or rest)
        # rho = True iff this stop resets shift accumulators (any rest)
        # (v3: the forced-wait qualifying-break logic is gone with SIM1 —
        # only declared activities reset the accumulators.)
        ri  = (brk in ("b45", "b30")) or (rst in ("r1", "r2"))
        rho = rst in ("r1", "r2")

        cd_dep = 0.0 if ri  else self.cd
        sd_dep = 0.0 if rho else self.sd

        # u: charging counted as working time
        #   sigma=0 (concurrent, break overlaps charge): u=0
        #   sigma=1 (sequential) or no break/rest: u=tauc
        _any_brk_rst = _brk_active or _rst_active
        u_exec = tauc_exec if (not _any_brk_rst or sigma) else 0.0

        if is_CS:
            work_now = (mstop_time + tauq_exec + u_exec + mseq_time) if not rho else 0.0
        elif is_cust:
            work_now = S_stop if not rho else 0.0
        elif is_lay:
            work_now = mlay_time if not rho else 0.0   # M8 parking overhead is work
        else:
            work_now = 0.0

        # M9 weekly working time: same work terms as work_now but WITHOUT the
        # rest-zeroing, so work performed at a rest stop still counts toward the
        # week (mirrors the offline weekly_work term, which counted work at
        # every stop regardless of any rest taken there).
        if is_CS:
            work_wk_stop = mstop_time + tauq_exec + u_exec + mseq_time
        elif is_cust:
            work_wk_stop = S_stop
        elif is_lay:
            work_wk_stop = mlay_time
        else:
            work_wk_stop = 0.0

        sw_dep = 0.0 if rho else self.sw + work_now     # reset sw on rest

        cd_new = cd_dep + D_act
        sd_new = sd_dep + D_act
        sw_new = sw_dep + D_act   # driving to next stop (work_now already in sw_dep)
        # weekly total: this stop's work + the leg driven from it, no reset
        sw_week_new = self.sw_week_history[-1] + work_wk_stop + D_act

        # ── 6.5. Shift spread h (M5/SP2): elapsed time since end of last rest ──
        # o = on-duty dwell at this stop before the rest.  Single rest-last
        # convention (v3): breaks/rests follow service at every stop type,
        # so the rest is always the last activity; a rest resets the spread,
        # which then accumulates the drive to the next stop.
        o_dwell = max(0.0, td - self.t_arr - taur_exec)
        # h_pre = the spread AT THE INSTANT the rest begins.  It is the quantity
        # the pre-rest cap (R24) applies to, and it is NOT recoverable from
        # h_new: a rest zeroes the spread, so h_new is only the outgoing leg.
        h_pre = self.h + o_dwell
        h_new = (0.0 if rho else h_pre) + D_act

        # ── 6.6. S2 violation semantics: retroactive mid-leg violations ────────
        _Tspr2   = full_data.get("Tspr2", 15.0)
        _sd_lim  = (full_data.get("Tdrv_sh2", 10.0)
                    if self.ext_shift_used < full_data.get("ext_bar", 2)
                    else full_data.get("Tdrv_sh1", 9.0))
        if cd_new > full_data["Tdrv_cons"] + 1e-3:
            self.violations.append(dict(
                type="hos_cd", stop=stop + 1, amount=cd_new - full_data["Tdrv_cons"],
                detail=f"cd={cd_new:.3f}h > {full_data['Tdrv_cons']}h after leg {stop}"))
        if sd_new > _sd_lim + 1e-3:
            self.violations.append(dict(
                type="hos_sd", stop=stop + 1, amount=sd_new - _sd_lim,
                detail=f"sd={sd_new:.3f}h > {_sd_lim}h after leg {stop}"))
        if h_new > _Tspr2 + 1e-3:
            self.violations.append(dict(
                type="hos_spread", stop=stop + 1, amount=h_new - _Tspr2,
                detail=f"spread h={h_new:.3f}h > {_Tspr2}h after leg {stop}"))
        # (R24) pre-rest spread cap — MILP.spread_prerest.  A daily rest must
        # BEGIN within 13 h of the shift start (15 h when it is a reduced rest);
        # starting it later does not make the shift legal retroactively.  The
        # h_new test above can never catch this, because at a rest stop rho=1
        # zeroes the spread and h_new is just D_act — which is why 26-54% of
        # stored runs carry an undetected overrun.  Test h_pre instead.
        _Tspr1 = full_data.get("Tspr1", 13.0)
        if self.strict_spread and rho:
            _cap = _Tspr2 if rst == "r2" else _Tspr1
            if h_pre > _cap + 1e-3:
                self.violations.append(dict(
                    type="hos_spread_prerest", stop=stop, amount=h_pre - _cap,
                    detail=(f"spread h+o={h_pre:.3f}h > {_cap}h when the {rst} "
                            f"rest began at stop {stop}")))
        # (h_term) terminal spread — MILP.spread_term.  The off-model final rest
        # after arrival is assumed regular, so the unfinished last shift is
        # bounded by the 13 h regular-rest spread.
        _N_route = full_data.get("N")
        if (self.strict_spread and _N_route is not None
                and stop + 1 >= _N_route and h_new > _Tspr1 + 1e-3):
            self.violations.append(dict(
                type="hos_spread_term", stop=stop + 1, amount=h_new - _Tspr1,
                detail=(f"terminal spread h={h_new:.3f}h > {_Tspr1}h on arrival "
                        f"(final shift cannot close with a regular rest)")))
        # M9 — weekly working-time cap: OUT OF SCOPE (daily provisions only).
        # A breach is recorded once as a diagnostic note, NOT a violation —
        # it never marks the run infeasible (see class comment).
        _Twk60 = full_data.get("Twk60", 60.0)
        if sw_week_new > _Twk60 + 1e-3 and not self._weekly_flagged:
            self._weekly_flagged = True
            self.weekly_notes.append(dict(
                type="hos_weekly", stop=stop + 1, amount=sw_week_new - _Twk60,
                detail=(f"weekly working time {sw_week_new:.3f}h > {_Twk60}h "
                        f"after leg {stop} (diagnostic only)")))

        # ── 7. phi (split-break flag) and rho2_used ───────────────────────────
        if ri or rho:
            phi_new = 0
        elif brk == "b15":
            phi_new = 1
        else:
            phi_new = self.phi

        rho2_new = self.rho2_used + (1 if rst == "r2" else 0)

        # Extended shift driving exception (EU 561/2006, Art. 6(2)):
        # count shifts where sd at rest time exceeded the normal 9h limit.
        # self.sd is the cumulative shift driving on arrival at this stop;
        # the concluded shift's total is self.sd (the leg to the next stop
        # starts a fresh shift after the reset).
        _Tdrv_sh1_nom = self._fd.get("Tdrv_sh1", 9.0)
        ext_new = self.ext_shift_used + (
            1 if (rho and self.sd > _Tdrv_sh1_nom - 1e-3) else 0
        )

        # ── 8. Append new state ───────────────────────────────────────────────
        self.stop_history.append(stop + 1)
        self.t_arr_history.append(t_arr_new)
        self.e_arr_history.append(e_new_raw)
        self.cd_history.append(cd_new)
        self.sd_history.append(sd_new)
        self.sw_history.append(sw_new)
        self.h_history.append(h_new)
        self.phi_history.append(phi_new)
        self.rho2_used_history.append(rho2_new)
        self.ext_shift_used_history.append(ext_new)
        self.sw_week_history.append(sw_week_new)


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
            f"rho2={self.rho2_used}, "
            f"ext_sh={self.ext_shift_used}/2)"
        )