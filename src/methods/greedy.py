"""
greedy.py — Greedy benchmark heuristic for BET scheduling
==========================================================
Provides a fast, deterministic benchmark representative of current driver
practice, to compare against the rolling-horizon look-ahead (LA) policy.

Decision philosophy
-------------------
The greedy driver operates without planning tools and follows three rules:

  1. STOP ONLY WHEN FORCED
     Take a break or daily rest only when HoS regulations require it
     (consecutive-driving or shift-driving limit is about to be violated).
     No proactive / opportunistic breaks.

  2. CHARGE TO FULL WHENEVER STOPPING AT A CS
     Every mandatory stop at a charging station (CS) triggers a full charge.
     Charging at a CS that is not a mandatory stop is also triggered when
     the energy to reach the next CS (worst-case) falls below Emin + buffer.

  3. FREE BREAK WITHIN CHARGE
     If the driver must stop at a CS and charging takes at least Tb45 hours,
     the mandatory break is scheduled inside the charge window at no extra
     time cost (parallel model: dwell = max(tauc, Tb45)).  Similarly for
     Tb30 / Tb15 split-break rules — which are skipped entirely when the
     instance sets allow_split=False (8.3 no-split axis).

Priority order (evaluated once per stop)
-----------------------------------------
  1. MUST-REST    : sd ≥ Tdrv_sh1 − ε  OR  sw ≥ Twrk_sh − ε
                   → r2 if budget allows, else r1
                   → charge if at CS (free; parallel with rest)
  2. MUST-BREAK   : cd ≥ Tdrv_cons − ε  (and no rest needed)
                   → b45 (or b30 if split-break in progress, phi==1)
                   → charge if at CS (free)
  3. MUST-CHARGE  : energy to next CS (worst-case) < Emin + safety_buffer
                   → charge only
                   → IF tauc ≥ break_needed, insert free break inside charge
  4. PASS         : none of the above
                   → charge to full at non-busy CS (free break if tauc large)
                   → otherwise no activity

Spread escalation (SP2b/SP2c)
-----------------------------
Art. 8 caps the SPREAD — elapsed time since the shift began — at 13 h before a
regular daily rest and 15 h before a reduced one, measured at the instant the
rest BEGINS (MILP eq. R24, spread_prerest).  The driver therefore rests at the
current stop whenever continuing would make a legal rest impossible at the next
one, counting the dwell that stop obliges first (service / manoeuvring / a
charge the energy state forces).  A discretionary charge that would push the
rest past its cap is deferred to a later stop.

Charge synchronisation rule
----------------------------
When a mandatory break or rest coincides with a CS stop, y=1 is always set
because charging during a mandatory dwell is free (parallel model).
When the stop is mandatory only for charging, a break is inserted for free
only if tauc (time to charge to full) ≥ the required break duration.

Information set
---------------
The greedy uses the same information as every other method: the vehicle
state, the travel-time distribution support (worst-case checks via
supervisor.compute_flags), and the expected charger queue delays Q_i.  Q_i
is a shared MODEL parameter (it enters the departure-time and working-time
constraints of the MILP used by all methods), so greedy's queue-avoidance
rule (C4) is not privileged knowledge — it is restored as legitimate driver
behavior.

Uncertainty
-----------
Travel times and energies are consumed from a precomputed realisation stored
in the instance JSON file (see instance_io.py).  Pass the realisation list as
`D_real` and `E_real` to run_greedy; the i-th entry is used at stop i.

Result compatibility
--------------------
run_greedy returns the same results dict schema as run_simulation so that
plot_simulation_results and check_simulation_feasibility work unchanged.
The shared epilogue (oracle, JSON, tables, feasibility check) is delegated
to runner.finalize_run.

Import chain
------------
  greedy.py → BEHDV, scenarios (ScenarioTracker), runner, plots
  No circular imports.

CLI usage
---------
  python -m src.methods.greedy <json_file> [realisation_index] [queue_threshold]
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from typing import Optional

from src.simulation.BEHDV      import BEHDV, _energy_after_charging, _charging_time_needed
from src.simulation.scenarios  import ScenarioTracker
from src.simulation.supervisor import (compute_flags, supervise_action,
                                       worst_case_energy_to_next_cs)
from src.settings   import TRAVEL_TIME_CV_TARGET, GUARD_QUANTILE
from src.simulation.runner     import finalize_run
from src.plot.plots      import plot_simulation_results   # re-exported for callers
from src import paths as _paths


# ══════════════════════════════════════════════════════════════════════════════
# ENERGY LOOK-AHEAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _energy_to_next_cs(full_data: dict, stop_global: int) -> float:
    """
    Nominal energy (kWh) required to reach the next CS stop (or destination)
    from ``stop_global``, driving non-stop.

    Uses full_data["E"] (nominal per-leg energy).
    """
    N     = full_data["N"]
    K_set = set(full_data["K"])
    E_nom = full_data["E"]

    cum = 0.0
    k   = stop_global
    while k < N:
        cum += E_nom.get(k, 0.0)
        if k + 1 in K_set or k + 1 == N:
            break
        k += 1
    return cum


# ══════════════════════════════════════════════════════════════════════════════
# GREEDY DURATION CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def _greedy_durations(full_data: dict, stop: int, action: dict,
                      state: BEHDV) -> dict:
    """
    Compute execution durations (taub, tauc, taur, tauq) for ``action``
    at ``stop``, using the same parallel-charging model as the MILP:

      - When charging and taking a break/rest simultaneously:
          dwell = max(tauc, break_min)
          taub  = max(0, break_min - tauc)   <- residual break beyond charge
      - Charging only : tauc = time to full;  taub = 0
      - Break only    : tauc = 0;              taub = break_min
      - Rest          : taur = rest_min (always separate from taub in greedy)

    The greedy always charges to full when y=1 (no partial charging).

    Returns
    -------
    dict with keys taub, tauc, taur, tauq (all in hours).
    """
    is_CS     = stop in set(full_data["K"])
    y         = action.get("y", 0)
    brk       = action.get("break_type")
    rst       = action.get("rest_type")
    break_min = {"b45": full_data["Tb45"],
                 "b15": full_data["Tb15"],
                 "b30": full_data["Tb30"]}.get(brk, 0.0)
    rest_min  = (full_data["Tr2"] if rst == "r2" else
                 full_data["Tr1"] if rst == "r1" else 0.0)
    tauq = full_data["Q"].get(stop, 0.0) * y if is_CS else 0.0

    if is_CS and y:
        tauc = _charging_time_needed(state.e_arr, full_data)
        # Break runs in parallel with charging; only residual time is extra
        taub = max(0.0, break_min - tauc)
    else:
        tauc = 0.0
        taub = break_min

    return dict(taub=taub, tauc=tauc, taur=rest_min, tauq=tauq)


# ══════════════════════════════════════════════════════════════════════════════
# GREEDY DECISION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def greedy_decision(full_data: dict, stop_global: int, state: BEHDV,
                    cv: float = TRAVEL_TIME_CV_TARGET,
                    guard_quantile: float | None = GUARD_QUANTILE,
                    safety_buffer_frac: float = 0.0,
                    queue_threshold: Optional[float] = None,
                    verbose: bool = False) -> tuple[dict, str]:
    """
    Make a greedy action decision at ``stop_global`` (paper §5.3).

    Fixed priority rule, evaluated once per stop:
      (i)   if the remaining SOC does not cover the WORST-CASE energy demand
            to the next charging station, charge to full;
      (ii)  if the consecutive-driving budget does not cover the next leg,
            take the cheapest qualifying break — in parallel with charging
            when a charge is already planned and long enough;
      (iii) if the spread or shift-driving budget does not cover the next
            leg, take a daily rest (reduced if the budget allows, regular
            otherwise);
      (iv)  otherwise, continue;
      (v)   escalate to a daily rest whenever a legal rest would no longer be
            possible at the next stop — the pre-rest spread cap is 13 h before
            a regular rest and 15 h before a reduced one, and it applies to
            h + o (spread at the instant the rest starts), not to the
            post-reset value.

    The greedy policy solves no optimization problem and uses the SAME
    information as every other method: the vehicle state, the distribution
    support (via supervisor.compute_flags), and the expected charger queue
    delays Q_i.  Q_i is a shared MODEL parameter — it already enters the
    departure-time and working-time constraints of the MILP used by every
    method — so greedy consulting it is NOT privileged knowledge (C4).

    Queue avoidance (C4): when a stop at a CS is discretionary (a break/rest
    dwell rather than an energy-forced charge), the driver skips charging at a
    station whose expected queue delay Q_i exceeds a threshold, preferring a
    lower-queue station downstream.  An energy-forced charge (must_charge) is
    never skipped.

    Parameters
    ----------
    cv : float
        CV of the travel-time multiplier — same value as the simulator draws
        from; cv = 0 forces nominal checks regardless of guard_quantile.
    guard_quantile : float or None
        Probability level of the one-step feasibility guard
        (settings.GUARD_QUANTILE default).  None = nominal checks (xi = 1,
        no uncertainty margin); 0.95 = guard against the xi 95% quantile;
        1.0 = full support corners.
    safety_buffer_frac : float [0, 1]
        OPTIONAL extra SOC buffer on top of the worst-case energy check
        (default 0.0 — the worst case is already conservative).
    queue_threshold : float (hours) or None
        CS stops with queue delay Q_i above this are not charged at unless a
        mandatory energy condition (must_charge) forces it.  None (default)
        → adaptive 80% of the instance's maximum CS queue delay.

    Returns
    -------
    action : dict  -- {y, break_type, rest_type}
    reason : str   -- human-readable explanation for logging
    """
    K_set  = set(full_data["K"])
    is_CS  = stop_global in K_set

    # Sea crossing: aboard for a known duration — no decision to make.  The
    # break is forced exactly as in the MILP (x_b45 = 1, taub = T_cross); the
    # executed duration is substituted by BEHDV.
    if stop_global in {int(k) for k in (full_data.get("ferry") or {})}:
        t_cross = float(full_data["ferry"][stop_global])
        return ({"y": 0, "break_type": "b45", "rest_type": None},
                f"FERRY: forced {t_cross:.2f} h crossing (counts as break)")

    Ecap   = full_data["Ecap"]
    Emin   = full_data["Emin"]
    usable = Ecap - Emin
    Tb45   = full_data["Tb45"]         # 0.75 h
    Tb30   = full_data["Tb30"]         # 0.50 h
    # 8.3 no-split axis: without the Art. 7 split, phi can never leave 0, so
    # every "b30 because a b15 is already banked" branch below is dead.
    allow_split = bool(full_data.get("allow_split", True))
    phi         = state.phi if allow_split else 0

    # One-step feasibility checks — single source of truth shared with the
    # rolling-horizon pruning and the S1 safety supervisor.
    flags = compute_flags(full_data, stop_global, state, cv, guard_quantile)

    must_rest   = flags["must_rest"]
    must_break  = flags["must_reset_cd"] and not must_rest
    must_b30    = (phi == 1) and must_break
    must_charge = (flags["must_charge"]
                   or (is_CS and state.e_arr - flags["e_needed"]
                       < Emin + safety_buffer_frac * usable))

    batt_full = (state.e_arr >= Ecap - 1.0)

    # ── C4: queue avoidance — skip a discretionary charge at a busy station ────
    # Q_i is a shared model parameter (no information asymmetry); an
    # energy-forced charge (must_charge) is never skipped.
    Q_all = full_data.get("Q", {})
    if queue_threshold is None:
        queue_threshold = 0.8 * max(Q_all.values()) if Q_all else float("inf")
    queue_busy  = is_CS and Q_all.get(stop_global, 0.0) > queue_threshold
    avoid_queue = queue_busy and not must_charge

    y   = 0
    brk = None
    rst = None

    # ── Priority (iii): MUST-REST (spread or shift-driving budget) ─────────────
    if must_rest:
        rst = ("r2" if state.rho2_used < int(full_data.get("rho_bar", 3))
               else "r1")
        # Charge for free during the pre-rest dwell at a CS, unless it is busy
        y      = 1 if (is_CS and not batt_full and not avoid_queue) else 0
        reason = f"MUST-REST ({rst.upper()})"
        if avoid_queue:
            reason += "  [queue busy, no charge]"

    # ── Priority (ii): MUST-BREAK ─────────────────────────────────────────────
    elif must_break:
        brk = "b30" if must_b30 else "b45"
        # Break runs in parallel with charging at a CS, unless it is busy
        y      = 1 if (is_CS and not batt_full and not avoid_queue) else 0
        reason = f"MUST-BREAK ({brk.upper()})"
        if avoid_queue:
            reason += "  [queue busy, no charge]"

    # ── Priority (i): MUST-CHARGE ─────────────────────────────────────────────
    elif must_charge and is_CS:
        y        = 1
        tauc_est = _charging_time_needed(state.e_arr, full_data)
        # Insert a break for free only if the charge is long enough to cover it
        if phi == 1 and tauc_est >= Tb30:
            brk    = "b30"
            reason = "MUST-CHARGE + free b30"
        elif tauc_est >= Tb45:
            brk    = "b45"
            reason = "MUST-CHARGE + free b45"
        else:
            reason = "MUST-CHARGE"

    # ── Priority (iv): PASS — no forced condition, do nothing ──────────────────
    else:
        reason = "PASS"

    # ── SP2: spread-aware escalation ───────────────────────────────────────────
    # compute_flags' spread check uses o(a)=0, so `must_rest` misses the on-duty
    # dwell (queue + charge + break + service + setup) that THIS action adds —
    # all of which counts toward the 15 h spread ceiling (BEHDV: h_new = h +
    # o_dwell + D_act).  A long dwell (e.g. a full charge with a parallel break)
    # can therefore breach the ceiling before the next stop without any rest
    # being triggered.  A daily rest is the only reset, so when the action we are
    # about to execute would bust the ceiling, escalate it to a rest here (any
    # planned charge still runs for free in parallel with the rest).
    # SP2b: the cap depends on WHICH rest is taken.  Art. 8 gives 13 h of spread
    # before a regular daily rest and 15 h before a reduced one (MILP eq. R24,
    # spread_prerest) — and the cap applies to h + o, the spread at the instant
    # the rest BEGINS, not to the post-reset value.  The old rule compared
    # against a hardcoded 15 h and so under-rested by up to 2 h on every route
    # whose reduced-rest budget was spent: 68% of greedy's r1 rests started
    # after the 13 h deadline, and the simulator could not see it (BEHDV zeroes
    # the spread at the rest, so its h > 15 test never fires at a rest stop).
    Tspr1_v  = full_data.get("Tspr1", 13.0)
    Tspr2_v  = flags.get("Tspr2", full_data.get("Tspr2", 15.0))
    rho_bar  = int(full_data.get("rho_bar", 3))
    next_rst = "r2" if state.rho2_used < rho_bar else "r1"
    D_wc     = flags.get("D_next_wc", 0.0)

    def _pre_rest_dwell(_y, _brk):
        """On-duty dwell o at this stop, i.e. everything preceding the rest."""
        _d = _greedy_durations(
            full_data, stop_global,
            {"y": _y, "break_type": _brk, "rest_type": None}, state)
        o = _d["tauq"] + _d["tauc"] + _d["taub"]
        if is_CS and (_y or _brk):
            o += full_data.get("M_stop", {}).get(stop_global, 0.0)
        if stop_global in set(full_data.get("C", [])):
            o += full_data["S"].get(stop_global, 0.0)
        return o

    def _forced_dwell(_stop: int) -> float:
        """Unavoidable on-duty dwell BEFORE a rest could start at ``_stop``.

        A rest is the last activity at a stop (rest-last convention), so
        whatever the stop type obliges — customer service, CS manoeuvring,
        layby parking, and a charge the energy state leaves no choice about —
        is already spent when the rest begins.  Ignoring it is what made the
        old rule rest one stop too late: it assumed a rest at the next stop
        could start the instant the truck arrived, when in practice the truck
        often arrives at a CS it MUST plug into for two hours first.
        """
        if _stop >= full_data["N"]:
            return 0.0
        if _stop in set(full_data.get("C", [])):
            return full_data["S"].get(_stop, 0.0)
        if _stop in set(full_data.get("L", [])):
            return full_data.get("M_lay", {}).get(_stop, 0.0)
        if _stop not in K_set:
            return 0.0
        o = full_data.get("M_stop", {}).get(_stop, 0.0)
        # Worst-case SOC on arrival at _stop.  flags["e_needed"] is the
        # worst-case energy from HERE to the next charging opportunity, so when
        # _stop is that opportunity it is exactly this leg's demand.
        soc_next = state.e_arr - flags["e_needed"]
        need_on  = worst_case_energy_to_next_cs(full_data, _stop, cv,
                                                guard_quantile)
        if soc_next - need_on < Emin + safety_buffer_frac * usable:
            o += (full_data.get("Q", {}).get(_stop, 0.0)
                  + _charging_time_needed(soc_next, full_data))
        return o

    if rst is None:
        cap    = Tspr2_v if next_rst == "r2" else Tspr1_v
        o_plan = _pre_rest_dwell(y, brk)
        # Continuing is only safe if a rest at the NEXT stop would still START
        # inside the cap — i.e. after this dwell, the leg, AND the dwell that
        # stop obliges before any rest can begin there.
        h_at_next_rest = (state.h + o_plan + D_wc
                          + _forced_dwell(stop_global + 1))
        if h_at_next_rest > cap + 1e-9:
            rst    = next_rst
            brk    = None    # a rest supersedes the break and resets cd + spread
            reason = (f"MUST-REST ({rst.upper()}) "
                      f"[spread {h_at_next_rest:.2f}h > {cap:.0f}h"
                      + (", +charge" if y else "") + "]")

    # SP2c: a full charge is 1.5-3.3 h of pre-rest dwell.  When that dwell is
    # what pushes the rest past its cap, a driver charges LATER rather than
    # starting the daily rest illegally — so drop a discretionary charge here.
    # An energy-forced charge (must_charge) is never dropped: stranding is worse
    # than an overrun, and the overrun is then recorded honestly.
    if rst is not None and y and not must_charge:
        cap = Tspr2_v if rst == "r2" else Tspr1_v
        if state.h + _pre_rest_dwell(y, None) > cap + 1e-9 \
                and state.h + _pre_rest_dwell(0, None) <= cap + 1e-9:
            y       = 0
            reason += "  [charge deferred: would breach pre-rest spread cap]"


    # ── Safety guards ──────────────────────────────────────────────────────────
    if not is_CS:
        y = 0
    # Respect split-break state machine
    if brk == "b30" and phi == 0:
        brk = "b45"    # b30 requires a prior b15 (phi==1)
    if brk == "b15" and phi == 1:
        brk = "b30"    # b15 when phi==1 would create phi==2; use b30 instead
    if not allow_split and brk in ("b15", "b30"):
        brk = "b45"    # no-split regime: the 45' block is the only legal break

    return {"y": y, "break_type": brk, "rest_type": rst}, reason


# ══════════════════════════════════════════════════════════════════════════════
# NOMINAL ARRIVAL-TIME PASS (used to centre time windows before D_real is drawn)
# ══════════════════════════════════════════════════════════════════════════════

def compute_nominal_arrivals(full_data: dict) -> list[float]:
    """
    Run the greedy policy with nominal travel times/energies (full_data["D"],
    full_data["E"]) and no time-window constraints, returning the absolute
    arrival time (h) at every stop 0..N.

    This is a stripped-down version of run_greedy's main loop: no file I/O,
    no oracle validation, no plotting.  It exists so that instance_io.py can
    obtain a realistic per-customer arrival estimate to centre time windows
    on, without paying for a full simulation + MILP-oracle run per instance.

    Returns
    -------
    list[float] -- vehicle.t_arr_history, length N+1 (index i = arrival at stop i)
    """
    N     = full_data["N"]
    D_nom = full_data["D"]
    E_nom = full_data["E"]

    vehicle = BEHDV(full_data)
    for stop in range(N):
        # nominal pass: no uncertainty margin (cv=0)
        action, _ = greedy_decision(full_data, stop, vehicle, cv=0.0)
        dur       = _greedy_durations(full_data, stop, action, vehicle)
        brk       = action.get("break_type")
        rst       = action.get("rest_type")
        mock_sol  = dict(
            feasible = True,
            sol = [dict(
                i    = 0,
                taub = dur["taub"], tauc = dur["tauc"],
                taur = dur["taur"], tauq = dur["tauq"],
                y    = action.get("y", 0),
                b45  = int(brk == "b45"), b15 = int(brk == "b15"),
                b30  = int(brk == "b30"),
                rho1 = int(rst == "r1"),  rho2 = int(rst == "r2"),
                is_C = (stop in set(full_data["C"])),
                is_K = (stop in set(full_data["K"])),
            )],
        )
        vehicle.advance(
            action   = action,
            D_next   = float(D_nom.get(stop, 0.0)),
            E_next   = float(E_nom.get(stop, 0.0)),
            milp_sol = mock_sol,
        )

    return list(vehicle.t_arr_history)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GREEDY SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_greedy(full_data: dict,
               D_real: list,
               E_real: list,
               cv: float                        = TRAVEL_TIME_CV_TARGET,
               guard_quantile: float | None     = GUARD_QUANTILE,
               safety_buffer: float             = 0.0,
               queue_threshold: Optional[float] = None,   # deprecated, ignored
               verbose: bool                    = True,
               run_id: str                      = None,
               oracle_tee: bool                 = True,
               supervised: bool                 = False,
               prune_quantile: float | None     = GUARD_QUANTILE,
               persist: bool                    = True) -> dict:
    """
    Run the greedy benchmark simulation from stop 0 to stop N.

    At each stop, greedy_decision evaluates the priority rules and returns
    an action immediately (no look-ahead, no scenario solving).  Duration
    parameters (tauc, taub, taur, tauq) are computed by _greedy_durations
    and passed to vehicle.advance.

    Travel times and energies are consumed from the precomputed realisation
    lists D_real and E_real (one entry per leg, leg i = stop i -> stop i+1).
    These must come from the same precomputed JSON as those used by LA and RO,
    ensuring fair comparison on identical uncertainty realisations.

    Parameters
    ----------
    full_data       : dict from instances.make_data() or loaded from JSON
    D_real          : list[float] -- realised travel times per leg (h), length N
    E_real          : list[float] -- realised energy per leg (kWh), length N
    safety_buffer   : minimum SOC buffer above Emin (fraction of usable)
    queue_threshold : CS queue time (h) above which charging is skipped
                      (unless mandatory).  None = adaptive 80% of max queue.
    verbose         : print per-stop decisions to stdout
    run_id          : base name for output files (auto-generated if None)
    oracle_tee      : pass tee=True to oracle_solve (shows HiGHS output)

    Returns
    -------
    dict -- canonical results dict from runner.finalize_run.
    """
    t_wall_start = time.perf_counter()
    N            = full_data["N"]
    T_START      = full_data.get("T_START", 8.0)
    label        = full_data.get("label", "greedy")

    assert len(D_real) == N, f"D_real length {len(D_real)} != N={N}"
    assert len(E_real) == N, f"E_real length {len(E_real)} != N={N}"

    # ── Output directories and file paths ─────────────────────────────────────
    _paths.ensure_dirs()
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{full_data.get('title', 'inst')}_GREEDY_{ts}"
    # persist=False -> an INTERNAL run (the ORACLE MIP warm start): write no
    # solution/figure/scenario artefacts, so it can never be mistaken for a
    # greedy method result by the dedup, the tables, or the figures.
    #
    # The log goes to logs/_internal/ rather than logs/.  Suppressing only the
    # SOLUTION was not enough: compile_solutions.find_failed_runs synthesises a
    # run row from any logs/*.txt that has no matching solution file, so each
    # warm start produced a phantom INCOMPLETE row for (instance, greedy) whose
    # timestamp — the oracle runs after the method batch — beat the real greedy
    # run in the latest-run dedup and evicted it from every table and figure.
    # 1163 real greedy runs were hidden that way, almost all of them on the
    # __variant and __diesel instances of sections 8.3/8.4, which are exactly
    # where ORACLE is run alongside greedy.  find_failed_runs scans only the top
    # level of logs/, so a subdirectory keeps the traceability without the
    # phantom.
    _int_dir = _paths.logs("_internal")
    if not persist:
        os.makedirs(_int_dir, exist_ok=True)
    paths = dict(
        log = (_paths.logs(f"{run_id}.txt") if persist
               else os.path.join(_int_dir, f"{run_id}.txt")),
        fig = _paths.figures(f"{run_id}.png") if persist else None,
        sol = _paths.solutions(f"{run_id}.json") if persist else None,
        scn = (_paths.logs(f"{run_id}_scenarios.json")
               if persist else None),
    )
    log = open(paths["log"], "w", encoding="utf-8")

    def _p(msg):
        # The log handle is utf-8, but stdout is whatever the console/pipe
        # gives us — cp1252 on a redirected Windows run, which cannot encode
        # the "dwell~=" glyph below and killed EVERY run in a piped batch
        # (3/3 failed with UnicodeEncodeError before this guard).  Same
        # treatment as oracle._safe_print: degrade the glyph, never the run.
        if verbose:
            try:
                print(msg)
            except UnicodeEncodeError:
                enc = getattr(sys.stdout, "encoding", None) or "ascii"
                print(msg.encode(enc, "replace").decode(enc, "replace"))
        try: print(msg, file=log)
        except Exception: pass

    _p("=" * 65)
    _p(f"  GREEDY SIMULATION START   ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    _p(f"  Instance : {label}   run_id={run_id}")
    _p(f"  Route    : {N} stops  departure={T_START:.0f}:00")
    _p(f"  Settings : cv={cv:.2f}  guard_q={guard_quantile}  "
       f"safety={safety_buffer:.0%}  supervised={supervised}")
    _p("=" * 65)

    # strict_spread: greedy is the one policy re-validated against the Art. 8
    # pre-rest / terminal spread caps, so it is the only one that records them
    # as violations.  LA and the rest keep their previous feasibility semantics.
    vehicle    = BEHDV(full_data, strict_spread=True)
    tracker    = ScenarioTracker(full_data)   # records realisations only
    scores_log = []                           # empty -- greedy has no look-ahead
    events     = dict(interventions=[], decision_times=[], cmp_log=[],
                      repairs=[], plan_violations=[])

    # ── Main loop ─────────────────────────────────────────────────────────────
    for stop in range(N):
        t_dec = time.perf_counter()
        action, reason = greedy_decision(
            full_data,
            stop,
            vehicle,
            cv                 = cv,
            guard_quantile     = guard_quantile,
            safety_buffer_frac = safety_buffer,
        )

        # S1: identical safety-supervisor layer as every other policy.
        # greedy_decision already uses the same compute_flags checks, so
        # interventions should be rare; the call keeps the guarantee uniform.
        if supervised and stop > 0:
            action, itv = supervise_action(full_data, stop, vehicle, action,
                                           cv=cv,
                                           quantile=prune_quantile)
            if itv is not None:
                events["interventions"].append(itv)
                _p(f"  [SUPERVISOR] stop {stop}: {itv['fixes']} "
                   f"({', '.join(itv['checks'])})")
        events["decision_times"].append(time.perf_counter() - t_dec)

        y   = action.get("y", 0)
        brk = action.get("break_type") or "---"
        rst = action.get("rest_type")  or "---"

        stop_type = ("CS"    if stop in set(full_data["K"]) else
                     "CUST"  if stop in set(full_data["C"]) else
                     "LAYBY" if stop in set(full_data.get("L", [])) else
                     "ORIG"  if stop == 0 else "INT")

        _p(f"\n  stop {stop:>3} ({stop_type})"
           f"  t={vehicle.t_arr:.3f}h  soc={vehicle.e_arr:.0f}kWh"
           f"  cd={vehicle.cd:.2f}  sd={vehicle.sd:.2f}  sw={vehicle.sw:.2f}"
           f"  phi={vehicle.phi}  r2={vehicle.rho2_used}")
        _p(f"     -> {reason:<40}  y={y}  brk={brk}  rst={rst}")

        dur      = _greedy_durations(full_data, stop, action, vehicle)
        mock_sol = dict(
            feasible = True,
            sol = [dict(
                i    = 0,
                taub = dur["taub"], tauc = dur["tauc"],
                taur = dur["taur"], tauq = dur["tauq"],
                y    = y,
                b45  = int(brk == "b45"), b15 = int(brk == "b15"),
                b30  = int(brk == "b30"),
                rho1 = int(rst == "r1"),  rho2 = int(rst == "r2"),
                is_C = (stop in set(full_data["C"])),
                is_K = (stop in set(full_data["K"])),
            )],
        )

        D_act = float(D_real[stop])
        E_act = float(E_real[stop])

        vehicle.advance(
            action   = action,
            D_next   = D_act,
            E_next   = E_act,
            milp_sol = mock_sol,
        )
        tracker.record_realisation(stop, D_act, E_actual=E_act)

        soc_dep = (_energy_after_charging(vehicle.e_arr_history[-2],
                                          dur["tauc"], full_data)
                   if y else vehicle.e_arr_history[-2])
        dwell_min = (vehicle.t_arr_history[-1] - vehicle.t_arr_history[-2]
                     - D_act) * 60
        _p(f"     dwell≈{dwell_min:.0f}min"
           f"  (tauc={dur['tauc']*60:.0f}m"
           f"  taub={dur['taub']*60:.0f}m"
           f"  taur={dur['taur']*60:.0f}m"
           f"  tauq={dur['tauq']*60:.0f}m)"
           f"  D_act={D_act:.3f}h  E_act={E_act:.1f}kWh"
           f"  -> soc_dep={soc_dep:.0f}kWh")

        scores_log.append([])

    # ── Summary ───────────────────────────────────────────────────────────────
    wall_elapsed = time.perf_counter() - t_wall_start
    arr_h        = vehicle.t_arr
    n_charges    = sum(1 for a in vehicle.actions if a.get("y", 0))
    n_breaks     = sum(1 for a in vehicle.actions if a.get("break_type"))
    n_rests      = sum(1 for a in vehicle.actions if a.get("rest_type"))
    n_sync       = sum(1 for a in vehicle.actions
                       if a.get("y", 0) and
                          (a.get("break_type") or a.get("rest_type")))

    _p(f"\n{'='*65}")
    _p(f"  GREEDY COMPLETE")
    _p(f"  Arrival (absolute) : {arr_h:.3f} h  ({int(arr_h):02d}:{int((arr_h%1)*60):02d})")
    _p(f"  Travel duration    : {arr_h - T_START:.3f} h")
    _p(f"  Charges  : {n_charges}  breaks: {n_breaks}  rests: {n_rests}"
       f"  sync: {n_sync}")
    _p(f"  Wall-clock         : {wall_elapsed:.1f} s")
    _p("=" * 65)

    # ── Delegate epilogue to runner ───────────────────────────────────────────
    results = finalize_run(
        vehicle     = vehicle,
        full_data   = full_data,
        tracker     = tracker,
        run_id      = run_id,
        paths       = paths,
        timing      = dict(wall_clock=wall_elapsed, T_START=T_START),
        log_fh      = log,
        verbose     = verbose,
        oracle_tee  = oracle_tee,
        scores_log  = scores_log,
        events      = events,
        method_meta = dict(
            method          = "greedy",
            cv              = cv,
            guard_quantile  = guard_quantile,
            safety_buffer   = safety_buffer,
            supervised      = supervised,
            prune_quantile  = prune_quantile,
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.instance_gen.instance_io import load_instance_json

    # Usage: python -m src.methods.greedy <json_file> [queue_threshold]
    json_file = sys.argv[1] if len(sys.argv) > 1 else None
    queue_thr = float(sys.argv[2]) if len(sys.argv) > 2 else None

    if json_file is None:
        print("Usage: python -m src.methods.greedy <json_file> [queue_threshold]")
        sys.exit(1)

    full_data, D_real, E_real, _ = load_instance_json(json_file)

    results = run_greedy(
        full_data,
        D_real          = D_real,
        E_real          = E_real,
        queue_threshold = queue_thr,
        verbose         = True,
        oracle_tee      = True,
    )

    print(f"\n  Figure   : plot later with `python -m src.plot.plots {results['run_id']}`")