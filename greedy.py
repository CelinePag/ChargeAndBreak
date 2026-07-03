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

  3. AVOID BUSY STATIONS
     If a CS stop has a queue time above `queue_threshold`, skip charging
     there unless charging is mandatory (energy constraint forces a stop).
     When `queue_threshold` is not given, it defaults to 80% of the
     instance's maximum CS queue time (0.8 * max(full_data["Q"].values())).

  4. FREE BREAK WITHIN CHARGE
     If the driver must stop at a CS and charging takes at least Tb45 hours,
     the mandatory break is scheduled inside the charge window at no extra
     time cost (parallel model: dwell = max(tauc, Tb45)).  Similarly for
     Tb30 / Tb15 split-break rules.

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

Charge synchronisation rule
----------------------------
When a mandatory break or rest coincides with a CS stop, y=1 is always set
because charging during a mandatory dwell is free (parallel model).
When the stop is mandatory only for charging, a break is inserted for free
only if tauc (time to charge to full) ≥ the required break duration.

Queue avoidance
---------------
A CS stop with queue_time > queue_threshold is skipped (y=0, no break)
UNLESS one of the following mandatory conditions holds:
  - must_rest or must_break (HoS violation imminent)
  - must_charge (energy infeasibility imminent)

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
  python greedy.py <json_file> [realisation_index] [queue_threshold]
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from typing import Optional

from BEHDV     import BEHDV, _energy_after_charging, _charging_time_needed
from scenarios import ScenarioTracker
from runner    import finalize_run
from plots     import plot_simulation_results   # re-exported for callers


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
                    safety_buffer_frac: float    = 0.10,
                    queue_threshold: Optional[float] = None,
                    verbose: bool                 = False) -> tuple[dict, str]:
    """
    Make a greedy action decision at ``stop_global``.

    The driver stops only when forced by HoS regulations or energy constraints.
    Before each leg, it checks whether driving that leg would violate cd, sd,
    or sw limits, and if so takes a break/rest at the current stop.  When
    stopping at a CS for any mandatory reason, the driver always charges to
    full (parallel model: charging is free during mandatory dwell).

    Parameters
    ----------
    safety_buffer_frac : float [0, 1]
        Extra SOC buffer above Emin when assessing mandatory charges.
        Charge if  e_arr - e_to_next  <  Emin + safety_buffer_frac * usable.
        Default 0.10 (10% of usable capacity).
    queue_threshold : float (hours) or None
        CS stops with queue_time > queue_threshold are not charged at unless
        a mandatory energy condition (must_charge) forces a stop there.
        When None (default), computed adaptively as 80% of the instance's
        maximum CS queue time: 0.8 * max(full_data["Q"].values()).

    Returns
    -------
    action : dict  -- {y, break_type, rest_type}
    reason : str   -- human-readable explanation for logging
    """
    K_set  = set(full_data["K"])
    is_CS  = stop_global in K_set

    Ecap   = full_data["Ecap"]
    Emin   = full_data["Emin"]
    usable = Ecap - Emin
    Tdrv   = full_data["Tdrv_cons"]    # 4.5 h max consecutive driving
    Tsd    = full_data["Tdrv_sh1"]     # 9.0 h max shift driving
    Tsw    = full_data["Twrk_sh"]      # 13.0 h max shift working
    Tb45   = full_data["Tb45"]         # 0.75 h
    Tb30   = full_data["Tb30"]         # 0.50 h
    # Nominal duration of the leg departing this stop (used for look-ahead).
    # The greedy checks whether driving the next leg would violate HoS limits,
    # and if so forces a break/rest NOW at this stop.  This is not planning —
    # it is simply what any driver does before starting a leg.
    D_next_nom = full_data["D"].get(stop_global, 0.0)

    # ── Mandatory condition flags (proactive: would next leg cause violation?) ──
    must_rest  = ((state.sd + D_next_nom >= Tsd) or
                  (state.sw + D_next_nom >= Tsw))
    must_break = (state.cd + D_next_nom >= Tdrv) and not must_rest
    # Split-break: phi==1 means a b15 was already taken; b30 completes the pair
    must_b30   = (state.phi == 1) and must_break

    e_to_next   = _energy_to_next_cs(full_data, stop_global)
    safety_buf  = safety_buffer_frac * usable
    must_charge = is_CS and (state.e_arr - e_to_next < Emin + safety_buf)

    batt_full = (state.e_arr >= Ecap - 1.0)

    # ── Queue avoidance: skip charging at busy CS unless energy forces it ──────
    Q_all = full_data.get("Q", {})
    if queue_threshold is None:
        queue_threshold = 0.8 * max(Q_all.values()) if Q_all else float("inf")
    queue_busy  = is_CS and Q_all.get(stop_global, 0.0) > queue_threshold
    avoid_queue = queue_busy and not must_charge

    y   = 0
    brk = None
    rst = None

    # ── Priority 1: MUST-REST ──────────────────────────────────────────────────
    if must_rest:
        rst = "r2" if state.rho2_used < 3 else "r1"
        # Charging is free during rest dwell at a CS, unless the station is busy
        y      = 1 if (is_CS and not batt_full and not avoid_queue) else 0
        reason = f"MUST-REST ({rst.upper()})"
        if avoid_queue:
            reason += "  [queue busy, no charge]"

    # ── Priority 2: MUST-BREAK ────────────────────────────────────────────────
    elif must_break:
        brk = "b30" if must_b30 else "b45"
        # Charging is free during break dwell at a CS, unless the station is busy
        y      = 1 if (is_CS and not batt_full and not avoid_queue) else 0
        reason = f"MUST-BREAK ({brk.upper()})"
        if avoid_queue:
            reason += "  [queue busy, no charge]"

    # ── Priority 3: MUST-CHARGE ───────────────────────────────────────────────
    elif must_charge:
        # must_charge implies is_CS already
        y        = 1
        tauc_est = _charging_time_needed(state.e_arr, full_data)
        # Insert a break for free only if the charge is long enough to cover it
        if state.phi == 1 and tauc_est >= Tb30:
            brk    = "b30"
            reason = "MUST-CHARGE + free b30"
        elif tauc_est >= Tb45:
            brk    = "b45"
            reason = "MUST-CHARGE + free b45"
        else:
            reason = "MUST-CHARGE"

    # ── Priority 4: PASS — no forced condition, do nothing ──────────────────────
    else:
        reason = "PASS"


    # ── Safety guards ──────────────────────────────────────────────────────────
    if not is_CS:
        y = 0
    # Respect split-break state machine
    if brk == "b30" and state.phi == 0:
        brk = "b45"    # b30 requires a prior b15 (phi==1)
    if brk == "b15" and state.phi == 1:
        brk = "b30"    # b15 when phi==1 would create phi==2; use b30 instead

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
        action, _ = greedy_decision(full_data, stop, vehicle)
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
               safety_buffer: float             = 0.10,
               queue_threshold: Optional[float] = None,
               verbose: bool                    = True,
               run_id: str                      = None,
               oracle_tee: bool                 = True) -> dict:
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

    Q_all = full_data.get("Q", {})
    if queue_threshold is None:
        queue_threshold = 0.8 * max(Q_all.values()) if Q_all else float("inf")

    # ── Output directories and file paths ─────────────────────────────────────
    for d in ("logs", "figures", "solutions"):
        os.makedirs(d, exist_ok=True)
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{full_data.get('title', 'inst')}_GREEDY_{ts}"
    paths = dict(
        log = os.path.join("logs",      f"{run_id}.txt"),
        fig = os.path.join("figures",   f"{run_id}.png"),
        sol = os.path.join("solutions", f"{run_id}.json"),
        scn = os.path.join("logs",      f"{run_id}_scenarios.json"),
    )
    log = open(paths["log"], "w", encoding="utf-8")

    def _p(msg):
        if verbose: print(msg)
        try: print(msg, file=log)
        except Exception: pass

    _p("=" * 65)
    _p(f"  GREEDY SIMULATION START   ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    _p(f"  Instance : {label}   run_id={run_id}")
    _p(f"  Route    : {N} stops  departure={T_START:.0f}:00")
    _p(f"  Settings : safety={safety_buffer:.0%}  queue_thresh={queue_threshold:.2f}h")
    _p("=" * 65)

    vehicle    = BEHDV(full_data)
    tracker    = ScenarioTracker(full_data)   # records realisations only
    scores_log = []                           # empty -- greedy has no look-ahead

    # ── Main loop ─────────────────────────────────────────────────────────────
    for stop in range(N):
        action, reason = greedy_decision(
            full_data,
            stop,
            vehicle,
            safety_buffer_frac = safety_buffer,
            queue_threshold    = queue_threshold,
        )

        y   = action.get("y", 0)
        brk = action.get("break_type") or "---"
        rst = action.get("rest_type")  or "---"

        stop_type = ("CS"   if stop in set(full_data["K"]) else
                     "CUST" if stop in set(full_data["C"]) else
                     "ORIG" if stop == 0 else "INT")

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
        method_meta = dict(
            method          = "greedy",
            safety_buffer   = safety_buffer,
            queue_threshold = queue_threshold,
        ),
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from instance_io import load_instance_json

    # Usage: python greedy.py <json_file> [queue_threshold]
    json_file = sys.argv[1] if len(sys.argv) > 1 else None
    queue_thr = float(sys.argv[2]) if len(sys.argv) > 2 else None

    if json_file is None:
        print("Usage: python greedy.py <json_file> [queue_threshold]")
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

    plot_simulation_results(
        results, full_data,
        title = f"greedy_{os.path.splitext(os.path.basename(json_file))[0]}",
        save  = True,
    )