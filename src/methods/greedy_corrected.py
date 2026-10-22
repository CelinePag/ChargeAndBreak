"""
greedy_corrected.py — the greedy heuristic executed under the MILP's rules
==========================================================================
Configuration variant "CORR" of the greedy baseline.  greedy.py is NOT
modified and remains the baseline of the submitted paper; this module is a
second, separately labelled method.

The decision philosophy is identical to greedy.py — stop only when forced,
charge to full whenever stopping at a charging station, the same one-step
feasibility guard (quantile of the travel-time multiplier), the same SOC
safety buffer and the same queue avoidance — but every stop is executed
exactly as the deterministic MILP (paper Section 4.4) executes it:

  C1  Charge + daily rest at a charging station is SEQUENTIAL (MILP eq. 14,
      sigma >= y + rho - 1): the repositioning time M_seq is paid and the
      charging time is working time.  greedy.py never passed sigma to the
      simulator, so it skipped both.
  C2  Charge + break runs in PARALLEL (the charge credited as break) only when
      the charge lasts at least the declared break minimum (eq. 15).  Otherwise
      it is sequential: charging is work, M_seq is paid and the full break
      follows the charge.  greedy.py credited the charge and let the truck idle
      at the bay for the remainder of the break.
  C3  A charge lasts at least T_min = 15 min (eq. 9; MILP constraint chg_act2).
      greedy.py allowed top-ups of a few minutes.

Two decision rules follow from C1-C3.  Both apply the principle greedy.py
already uses at a must-charge stop ("insert a break only if the charge covers
it"), now in the other direction:

  D1  At a mandatory break at a charging station the charge is added only when
      it is free — a charge to full lasts at least the break, so it runs in
      parallel — or when the energy state forces it (must_charge), in which case
      a charge shorter than the break runs sequentially.
  D2  The dwell estimates used by the spread rules (the dwell at this stop
      before a rest, and the dwell the next stop obliges) equal the dwell the
      simulator will execute: they include M_seq when a charge precedes a rest,
      and the access time at a charging station or rest area whenever an
      activity takes place there.

Two rule sets
-------------
  rules="milp"  (variant CORR)  - C1-C3 and D1-D2 above, exactly.
  rules="bay"   (variant CORR2, the default since 2026-10-21) - C1, C3 and D2,
      but the truck may STAY AT THE BAY after a charge to complete a break
      (C2 relaxed), and the break decision no longer depends on the charge time:
        B1  Charge + break always runs in parallel: the charge is credited as
            break and the stop lasts max(charge, break).  No repositioning.
        B2  Whenever the greedy stops to charge (must-charge stop) it also takes
            the break there (b45, or b30 to complete a split), staying at the
            bay if the charge ends first.  greedy.py inserted the break only if
            the charge lasted >= 45 min, so a faster charger could REMOVE the
            break from a charging stop and push it to a later standalone stop:
            the cause of its worse results at 1000 kW than at 700 kW.
        B3  At a mandatory break at a charging station the greedy charges (as
            greedy.py does); the stop lasts max(charge, break).
      With these rules no decision uses the charging time, so for the same
      sequence of activities every dwell is non-increasing in charger power:
      max(tauc, b) for a charge with a break, tauc + M_seq before a rest.  The
      only remaining channel is the spread rule, which reads the elapsed duty
      time (shorter at higher power) to decide where to rest.

Run ids carry the variant label:  <instance>_GREEDY_<VARIANT>_<ts>[_<idx>].
The reporting dedup is keyed on the variant, so these runs sit next to the
baseline greedy runs and never displace them; compile_solutions and
paper_figures exclude variants from the published tables and figures.

Usage
-----
  python -m src.simulation.runner_dispatch "instances/*.json" greedy \
      --greedy_corrected [--greedy_rules bay|milp] --prune_quantile 0.95 \
      --safety 0.1 --jobs 8
"""

from __future__ import annotations

import datetime
import os
import sys
import time

from src.simulation.BEHDV      import BEHDV, _energy_after_charging, _charging_time_needed
from src.simulation.scenarios  import ScenarioTracker
from src.simulation.supervisor import (compute_flags, supervise_action,
                                       worst_case_energy_to_next_cs)
from src.settings   import TRAVEL_TIME_CV_TARGET, GUARD_QUANTILE
from src.simulation.runner     import finalize_run
from src import paths as _paths

# Variant label stamped into the run_id (and from there into the solution JSON),
# per rule set.  VARIANT is the default rule set's label.
VARIANTS      = {"milp": "CORR", "bay": "CORR2"}
DEFAULT_RULES = "bay"
VARIANT       = VARIANTS[DEFAULT_RULES]

# Minimum charging duration when charging at all (MILP chg_act2: tauc >= 0.25 y).
TMIN_CHARGE_H = 0.25


def _break_min(full_data: dict, brk) -> float:
    return {"b45": full_data["Tb45"], "b15": full_data["Tb15"],
            "b30": full_data["Tb30"]}.get(brk, 0.0)


def _durations_corrected(full_data: dict, stop: int, action: dict,
                         state: BEHDV, rules: str = DEFAULT_RULES) -> dict:
    """Execution durations under the MILP's charge/break/rest rules.

    Returns taub, tauc, taur, tauq (h) and sigma (sequential-mode flag):

      charge + rest              -> sigma = 1 (eq. 14), no break time
      charge + break, tauc >= b  -> sigma = 0, taub = 0 (charge credited)
      charge + break, tauc <  b  -> sigma = 1 (eq. 15), taub = b (full break
                                    after the charge)
      charge only                -> sigma = 0, charging is work
      break / rest without charge-> sigma = 0
    The charge always runs to full (greedy rule).
    """
    is_CS     = stop in set(full_data["K"])
    y         = int(action.get("y", 0) or 0)
    brk       = action.get("break_type")
    rst       = action.get("rest_type")
    break_min = _break_min(full_data, brk)
    rest_min  = (full_data["Tr2"] if rst == "r2" else
                 full_data["Tr1"] if rst == "r1" else 0.0)
    tauq = full_data["Q"].get(stop, 0.0) * y if is_CS else 0.0

    sigma = 0
    if is_CS and y:
        tauc = _charging_time_needed(state.e_arr, full_data, stop=stop)
        if rst in ("r1", "r2"):
            sigma, taub = 1, 0.0
        elif brk in ("b45", "b15", "b30"):
            if tauc >= break_min - 1e-9:
                sigma, taub = 0, 0.0
            elif rules == "bay":
                # B1: stay at the bay until the break is complete
                sigma, taub = 0, break_min - tauc
            else:
                sigma, taub = 1, break_min
        else:
            taub = 0.0
    else:
        tauc = 0.0
        taub = break_min

    return dict(taub=taub, tauc=tauc, taur=rest_min, tauq=tauq, sigma=sigma)


# ══════════════════════════════════════════════════════════════════════════════
# DECISION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def greedy_corrected_decision(full_data: dict, stop_global: int, state: BEHDV,
                              cv: float = TRAVEL_TIME_CV_TARGET,
                              guard_quantile: float | None = GUARD_QUANTILE,
                              safety_buffer_frac: float = 0.0,
                              verbose: bool = False,
                              rules: str = DEFAULT_RULES) -> tuple[dict, str]:
    """Greedy action at ``stop_global`` (same priority rules as greedy.py).

    Priority: (iii) must-rest, (ii) must-break, (i) must-charge, (iv) pass —
    then the spread escalation and the charge deferral of greedy.py.  The
    differences are D1 (a charge is attached to a mandatory break only if it
    is free or energy-forced), C3 (no charge shorter than T_min) and D2 (exact
    dwell estimates in the spread rules).
    """
    K_set  = set(full_data["K"])
    C_set  = set(full_data.get("C", []))
    L_set  = set(full_data.get("L", []))
    is_CS  = stop_global in K_set
    is_lay = stop_global in L_set

    # Sea crossing: aboard for a known duration — identical to greedy.py.
    if stop_global in {int(k) for k in (full_data.get("ferry") or {})}:
        t_cross = float(full_data["ferry"][stop_global])
        return ({"y": 0, "break_type": "b45", "rest_type": None},
                f"FERRY: forced {t_cross:.2f} h crossing (counts as break)")

    Ecap   = full_data["Ecap"]
    Emin   = full_data["Emin"]
    usable = Ecap - Emin
    Tb45   = full_data["Tb45"]
    Tb30   = full_data["Tb30"]
    M_seq  = full_data.get("M_seq", {}).get(stop_global, 0.0) if is_CS else 0.0
    M_stop = full_data.get("M_stop", {}).get(stop_global, 0.0) if is_CS else 0.0
    M_lay  = full_data.get("M_lay", {}).get(stop_global, 0.0) if is_lay else 0.0

    allow_split = bool(full_data.get("allow_split", True))
    phi         = state.phi if allow_split else 0

    flags = compute_flags(full_data, stop_global, state, cv, guard_quantile)

    must_rest   = flags["must_rest"]
    must_break  = flags["must_reset_cd"] and not must_rest
    must_b30    = (phi == 1) and must_break
    must_charge = (flags["must_charge"]
                   or (is_CS and state.e_arr - flags["e_needed"]
                       < Emin + safety_buffer_frac * usable))

    # C3: a charge that would end before T_min is not possible at all.
    tauc_full  = (_charging_time_needed(state.e_arr, full_data, stop=stop_global)
                  if is_CS else 0.0)
    can_charge = is_CS and tauc_full >= TMIN_CHARGE_H - 1e-9

    Q_all = full_data.get("Q", {})
    queue_threshold = 0.8 * max(Q_all.values()) if Q_all else float("inf")
    queue_busy  = is_CS and Q_all.get(stop_global, 0.0) > queue_threshold
    avoid_queue = queue_busy and not must_charge

    y, brk, rst = 0, None, None

    # ── (iii) MUST-REST ──────────────────────────────────────────────────────
    if must_rest:
        rst = ("r2" if state.rho2_used < int(full_data.get("rho_bar", 3))
               else "r1")
        # Charge before the rest (sequential, C1) unless busy or nothing to fill.
        y      = 1 if (can_charge and not avoid_queue) else 0
        reason = f"MUST-REST ({rst.upper()})"
        if avoid_queue:
            reason += "  [queue busy, no charge]"

    # ── (ii) MUST-BREAK ──────────────────────────────────────────────────────
    elif must_break:
        brk   = "b30" if must_b30 else "b45"
        b_min = Tb30 if brk == "b30" else Tb45
        reason = f"MUST-BREAK ({brk.upper()})"
        if can_charge and not avoid_queue and rules == "bay":
            y = 1                                        # B3: max(charge, break)
        elif can_charge and not avoid_queue:
            if tauc_full >= b_min - 1e-9:
                y = 1                                    # parallel: free (D1)
            elif must_charge:
                y = 1                                    # forced: sequential
                reason += "  [forced charge shorter than break: sequential]"
            else:
                reason += "  [charge would not cover the break: skipped]"
        elif avoid_queue:
            reason += "  [queue busy, no charge]"

    # ── (i) MUST-CHARGE ──────────────────────────────────────────────────────
    elif must_charge and is_CS:
        if can_charge and rules == "bay":
            # B2: take the break at every charging stop, whatever the charge
            # time; the truck stays at the bay if the charge ends first.
            y = 1
            brk = "b30" if phi == 1 else "b45"
            reason = f"MUST-CHARGE + {brk} (stay at bay if needed)"
        elif can_charge:
            y = 1
            if phi == 1 and tauc_full >= Tb30:
                brk, reason = "b30", "MUST-CHARGE + free b30"
            elif tauc_full >= Tb45:
                brk, reason = "b45", "MUST-CHARGE + free b45"
            else:
                reason = "MUST-CHARGE"
        else:
            reason = "MUST-CHARGE impossible (charge would last < T_min)"

    # ── (iv) PASS ────────────────────────────────────────────────────────────
    else:
        reason = "PASS"

    Tspr1_v  = full_data.get("Tspr1", 13.0)
    Tspr2_v  = flags.get("Tspr2", full_data.get("Tspr2", 15.0))
    rho_bar  = int(full_data.get("rho_bar", 3))
    next_rst = "r2" if state.rho2_used < rho_bar else "r1"
    D_wc     = flags.get("D_next_wc", 0.0)

    def _dwell(_y: int, _brk, rest_here: bool) -> float:
        """On-duty dwell at THIS stop (everything before a rest, if any) —
        exactly what BEHDV will execute (D2)."""
        d = _durations_corrected(
            full_data, stop_global,
            {"y": _y, "break_type": None if rest_here else _brk,
             "rest_type": "r1" if rest_here else None}, state, rules)
        o = d["tauq"] + d["tauc"] + d["taub"] + d["sigma"] * M_seq
        active = bool(_y or _brk or rest_here)
        if is_CS and active:
            o += M_stop
        if is_lay and (_brk or rest_here):
            o += M_lay
        if stop_global in C_set:
            o += full_data["S"].get(stop_global, 0.0)
        return o

    def _forced_dwell(_stop: int) -> float:
        """Unavoidable dwell at ``_stop`` before a rest could begin there."""
        if _stop >= full_data["N"]:
            return 0.0
        if _stop in C_set:
            return full_data["S"].get(_stop, 0.0)
        if _stop in L_set:
            return full_data.get("M_lay", {}).get(_stop, 0.0)
        if _stop not in K_set:
            return 0.0
        o = full_data.get("M_stop", {}).get(_stop, 0.0)
        soc_next = state.e_arr - flags["e_needed"]
        need_on  = worst_case_energy_to_next_cs(full_data, _stop, cv,
                                                guard_quantile)
        if soc_next - need_on < Emin + safety_buffer_frac * usable:
            # forced charge, then the rest: sequential (C1) -> + M_seq
            o += (full_data.get("Q", {}).get(_stop, 0.0)
                  + _charging_time_needed(soc_next, full_data, stop=_stop)
                  + full_data.get("M_seq", {}).get(_stop, 0.0))
        return o

    if rst is None:
        cap    = Tspr2_v if next_rst == "r2" else Tspr1_v
        o_plan = _dwell(y, brk, False)
        h_at_next_rest = (state.h + o_plan + D_wc
                          + _forced_dwell(stop_global + 1))
        if h_at_next_rest > cap + 1e-9:
            rst    = next_rst
            brk    = None
            reason = (f"MUST-REST ({rst.upper()}) "
                      f"[spread {h_at_next_rest:.2f}h > {cap:.0f}h"
                      + (", +charge" if y else "") + "]")

    # Charge deferral before a rest (greedy.py SP2c), with the exact dwell.
    if rst is not None and y and not must_charge:
        cap = Tspr2_v if rst == "r2" else Tspr1_v
        if (state.h + _dwell(y, None, True) > cap + 1e-9
                and state.h + _dwell(0, None, True) <= cap + 1e-9):
            y       = 0
            reason += "  [charge deferred: would breach pre-rest spread cap]"

    # ── Guards (as greedy.py) ─────────────────────────────────────────────────
    if not is_CS:
        y = 0
    if brk == "b30" and phi == 0:
        brk = "b45"
    if brk == "b15" and phi == 1:
        brk = "b30"
    if not allow_split and brk in ("b15", "b30"):
        brk = "b45"
    # D1 must still hold after the guards changed the break type.
    if rules != "bay" and y and brk and rst is None and not must_charge \
            and tauc_full < _break_min(full_data, brk) - 1e-9:
        y = 0
        reason += "  [charge dropped: would not cover the break]"

    return {"y": y, "break_type": brk, "rest_type": rst}, reason


# ══════════════════════════════════════════════════════════════════════════════
# RUN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_greedy_corrected(full_data: dict,
                         D_real: list,
                         E_real: list,
                         cv: float                    = TRAVEL_TIME_CV_TARGET,
                         guard_quantile: float | None = GUARD_QUANTILE,
                         safety_buffer: float         = 0.0,
                         verbose: bool                = True,
                         run_id: str                  = None,
                         oracle_tee: bool             = True,
                         supervised: bool             = False,
                         prune_quantile: float | None = GUARD_QUANTILE,
                         persist: bool                = True,
                         rules: str                   = DEFAULT_RULES) -> dict:
    """Simulate the corrected greedy from stop 0 to N (same schema as greedy).

    Identical to greedy.run_greedy except for the decision function and the
    duration model, and for passing sigma to the simulator so that BEHDV pays
    M_seq and books the charge as work exactly as for the MILP-based methods.
    """
    t_wall_start = time.perf_counter()
    N            = full_data["N"]
    T_START      = full_data.get("T_START", 8.0)
    label        = full_data.get("label", "greedy")

    assert len(D_real) == N, f"D_real length {len(D_real)} != N={N}"
    assert len(E_real) == N, f"E_real length {len(E_real)} != N={N}"

    _paths.ensure_dirs()
    if run_id is None:
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = _paths.make_run_id(full_data.get("title", "inst"), "GREEDY",
                                    ts, variant=VARIANTS[rules])

    _int_dir = _paths.logs("_internal")
    if not persist:
        os.makedirs(_int_dir, exist_ok=True)
    paths = dict(
        log = (_paths.log_out(f"{run_id}.txt") if persist
               else os.path.join(_int_dir, f"{run_id}.txt")),
        fig = _paths.figure_out(f"{run_id}.png") if persist else None,
        sol = _paths.solution_out(f"{run_id}.json") if persist else None,
        scn = (_paths.log_out(f"{run_id}_scenarios.json")
               if persist else None),
    )
    log = open(paths["log"], "w", encoding="utf-8")

    def _p(msg):
        if verbose:
            try:
                print(msg)
            except UnicodeEncodeError:
                enc = getattr(sys.stdout, "encoding", None) or "ascii"
                print(msg.encode(enc, "replace").decode(enc, "replace"))
        try: print(msg, file=log)
        except Exception: pass

    _p("=" * 65)
    _p(f"  GREEDY (CORRECTED, rules={rules}) START   "
       f"({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    _p(f"  Instance : {label}   run_id={run_id}")
    _p(f"  Route    : {N} stops  departure={T_START:.0f}:00")
    _p(f"  Settings : cv={cv:.2f}  guard_q={guard_quantile}  "
       f"safety={safety_buffer:.0%}  supervised={supervised}")
    _p("=" * 65)

    vehicle    = BEHDV(full_data, strict_spread=True)
    tracker    = ScenarioTracker(full_data)
    scores_log = []
    events     = dict(interventions=[], decision_times=[], cmp_log=[],
                      repairs=[], plan_violations=[])
    K_set, C_set, L_set = (set(full_data["K"]), set(full_data["C"]),
                           set(full_data.get("L", [])))

    for stop in range(N):
        t_dec = time.perf_counter()
        action, reason = greedy_corrected_decision(
            full_data, stop, vehicle,
            cv                 = cv,
            guard_quantile     = guard_quantile,
            safety_buffer_frac = safety_buffer,
            rules              = rules,
        )
        if supervised and stop > 0:
            action, itv = supervise_action(full_data, stop, vehicle, action,
                                           cv=cv, quantile=prune_quantile)
            if itv is not None:
                events["interventions"].append(itv)
                _p(f"  [SUPERVISOR] stop {stop}: {itv['fixes']} "
                   f"({', '.join(itv['checks'])})")
        events["decision_times"].append(time.perf_counter() - t_dec)

        y   = action.get("y", 0)
        brk = action.get("break_type") or "---"
        rst = action.get("rest_type")  or "---"
        stop_type = ("CS" if stop in K_set else "CUST" if stop in C_set else
                     "LAYBY" if stop in L_set else
                     "ORIG" if stop == 0 else "INT")
        _p(f"\n  stop {stop:>3} ({stop_type})"
           f"  t={vehicle.t_arr:.3f}h  soc={vehicle.e_arr:.0f}kWh"
           f"  cd={vehicle.cd:.2f}  sd={vehicle.sd:.2f}  sw={vehicle.sw:.2f}"
           f"  phi={vehicle.phi}  r2={vehicle.rho2_used}")
        _p(f"     -> {reason:<40}  y={y}  brk={brk}  rst={rst}")

        dur      = _durations_corrected(full_data, stop, action, vehicle, rules)
        mock_sol = dict(
            feasible = True,
            sol = [dict(
                i     = 0,
                taub  = dur["taub"], tauc = dur["tauc"],
                taur  = dur["taur"], tauq = dur["tauq"],
                sigma = dur["sigma"],
                y     = y,
                b45   = int(brk == "b45"), b15 = int(brk == "b15"),
                b30   = int(brk == "b30"),
                rho1  = int(rst == "r1"),  rho2 = int(rst == "r2"),
                is_C  = (stop in C_set),
                is_K  = (stop in K_set),
            )],
        )

        D_act = float(D_real[stop])
        E_act = float(E_real[stop])
        vehicle.advance(action=action, D_next=D_act, E_next=E_act,
                        milp_sol=mock_sol)
        tracker.record_realisation(stop, D_act, E_actual=E_act)

        if vehicle.is_halted:
            _p(f"  [!] {vehicle.halt_reason.upper()} at stop "
               f"{vehicle.halted_at} — run halted, route not completed")
            break

        soc_dep = (_energy_after_charging(vehicle.e_arr_history[-2],
                                          dur["tauc"], full_data, stop=stop)
                   if y else vehicle.e_arr_history[-2])
        dwell_min = (vehicle.t_arr_history[-1] - vehicle.t_arr_history[-2]
                     - D_act) * 60
        _p(f"     dwell≈{dwell_min:.0f}min"
           f"  (tauc={dur['tauc']*60:.0f}m  taub={dur['taub']*60:.0f}m"
           f"  taur={dur['taur']*60:.0f}m  tauq={dur['tauq']*60:.0f}m"
           f"  sigma={dur['sigma']})"
           f"  D_act={D_act:.3f}h  E_act={E_act:.1f}kWh"
           f"  -> soc_dep={soc_dep:.0f}kWh")
        scores_log.append([])

    wall_elapsed = time.perf_counter() - t_wall_start
    arr_h        = vehicle.t_arr
    _p(f"\n{'='*65}")
    _p(f"  GREEDY (CORRECTED) COMPLETE")
    _p(f"  Travel duration    : {arr_h - T_START:.3f} h")
    _p(f"  Wall-clock         : {wall_elapsed:.1f} s")
    _p("=" * 65)

    return finalize_run(
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
            greedy_rules    = rules,       # "milp" (CORR) or "bay" (CORR2)
            cv              = cv,
            guard_quantile  = guard_quantile,
            safety_buffer   = safety_buffer,
            supervised      = supervised,
            prune_quantile  = prune_quantile,
        ),
    )
