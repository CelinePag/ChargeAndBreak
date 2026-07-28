"""
oracle.py — Hindsight-optimal benchmark and simulation reporting
================================================================
Provides two services:

  1. oracle_solve(full_data, D_actual_list, sim_results, ...)
       Solves the full MILP with the travel times that actually occurred
       during the simulation (perfect hindsight).  The simulation trajectory
       is injected as a warm-start incumbent so HiGHS starts from a known
       feasible solution.  Returns arrival time, gap, and per-stop schedule.

  2. Reporting helpers
       check_simulation_feasibility(results, full_data)  — constraint audit
       print_simulation_log(results, full_data)          — trajectory table
       print_oracle_log(oracle_result, full_data)        — oracle schedule table

Design
------
All imports are at module level (no lazy imports).  oracle.py depends on:
  BEHDV     — _energy_after_charging (energy utility)
  MILP      — build_model, extract_solution (Pyomo model construction)
  instances — compute_time_bounds (re-computes bounds for actual travel times)

oracle.py is NOT imported by MILP.py or instances.py; it sits above them in
the dependency graph.  runner.py calls oracle_solve as part of finalize_run.

Import chain
------------
  oracle.py → BEHDV, MILP, instances
  runner.py → oracle
  (Simulation.py and greedy.py no longer import oracle directly;
   they use runner.finalize_run which calls it internally.)
"""

from __future__ import annotations

import contextlib as _ctx
import io as _io
import json
import os
import re
import warnings

import numpy as np
import pyomo.environ as pyo

from BEHDV     import _energy_after_charging
from MILP      import build_model, extract_solution
from instances import compute_time_bounds


# ══════════════════════════════════════════════════════════════════════════════
# ORACLE CACHE  (shared by the ORACLE runner, compile_solutions, plots)
# ══════════════════════════════════════════════════════════════════════════════
# The oracle depends only on the instance geometry + the realised travel times
# (D_real), both fixed per instance, so ONE oracle per instance is shared by
# every method.  It is stored at solutions/oracle_<instance>.json and is NOT
# recomputed by method runs — methods and the oracle are solved independently;
# the gap to oracle is derived on demand (compile_solutions / plots) from this
# cache, whenever it exists.

def oracle_cache_path(instance: str, solutions_dir: str = "solutions") -> str:
    return os.path.join(solutions_dir, f"oracle_{instance}.json")


def _reint_keys(obj):
    """Recursively turn digit string keys back into ints (JSON loses int keys)."""
    if isinstance(obj, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k):
                _reint_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_reint_keys(v) for v in obj]
    return obj


def load_oracle_cache(instance: str, solutions_dir: str = "solutions"):
    """Return the cached oracle dict for an instance, or None if not solved yet."""
    path = oracle_cache_path(instance, solutions_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return _reint_keys(json.load(fh))
    except Exception:
        return None


def save_oracle_cache(instance: str, result: dict,
                      solutions_dir: str = "solutions") -> str:
    """Write an oracle result to its per-instance cache file; returns the path."""
    def _ser(o):
        if isinstance(o, (int, float, bool, str, type(None))):
            return o
        if isinstance(o, dict):
            return {str(k): _ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_ser(v) for v in o]
        return str(o)
    path = oracle_cache_path(instance, solutions_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_ser(result), fh, indent=2)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# FEASIBILITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_simulation_feasibility(results: dict, full_data: dict,
                                  tol: float = 1e-3):
    """
    Verify that the simulated trajectory satisfies all HoS and energy
    constraints.

    Parameters
    ----------
    results   : dict returned by run_simulation or run_greedy
    full_data : dict from instances.make_data()
    tol       : float — numerical tolerance for constraint checks

    Returns
    -------
    ok     : bool          — True if all constraints satisfied
    issues : list[str]     — human-readable description of each violation
    """
    states  = results["states"]
    actions = results["actions"]
    durs    = results.get("durations_list", [])
    K_set   = set(full_data["K"])
    C_set   = set(full_data["C"])
    L_set   = set(full_data.get("L", []))

    Tdrv_cons = full_data["Tdrv_cons"]
    Tdrv_sh1  = full_data["Tdrv_sh1"]
    Tdrv_sh2  = full_data.get("Tdrv_sh2", Tdrv_sh1)   # extended exception limit
    Tspr2     = full_data.get("Tspr2", 15.0)          # M5: global spread ceiling
    Emin      = full_data["Emin"]
    hard_tw   = bool(full_data.get("hard_tw", False))
    issues    = []

    for i, state in enumerate(states):
        s   = state.stop
        act = actions[i] if i < len(actions) else {}
        dur = durs[i]    if i < len(durs)    else {}
        y   = int(act.get("y", 0))
        brk = act.get("break_type")
        rst = act.get("rest_type")
        is_CS   = s in K_set
        is_cust = s in C_set

        tauq = dur.get("tauq", 0.0)
        tauc = dur.get("tauc", 0.0)
        _brk_active  = brk in ("b45", "b15", "b30")
        _rst_active  = rst in ("r1", "r2")
        _any_brk_rst = _brk_active or _rst_active
        v_i          = int(is_CS and (bool(y) or _any_brk_rst))
        # Approximate sigma=0 (concurrent) for the feasibility check
        u_i   = tauc if (is_CS and y and not _any_brk_rst) else 0.0
        M_s   = full_data.get("M_stop", {}).get(s, 0.0) if is_CS else 0.0
        if is_CS:
            work = v_i * M_s + tauq * y + u_i
        elif is_cust:
            work = full_data["S"].get(s, 0.0)
        elif s in L_set:
            # M8: layby parking overhead (counts as work when a break/rest is taken)
            work = full_data.get("M_lay", {}).get(s, 0.0) * int(_any_brk_rst)
        else:
            work = 0.0
        sw_k = state.sw + work

        # Effective shift-driving limit: 10h when extended-exception budget
        # is not yet exhausted (ext_shift_used < 2), otherwise the regular 9h.
        # ext_shift_used at stop s reflects exceptions consumed in rests taken
        # at all stops before s, so it correctly governs the current shift.
        _ext_used   = getattr(state, "ext_shift_used", 0)
        _sd_limit   = Tdrv_sh2 if _ext_used < 2 else Tdrv_sh1

        if state.cd > Tdrv_cons + tol:
            issues.append(
                f"stop {s:>2}: cd={state.cd:.3f}h > {Tdrv_cons}h (consec driving)")
        if state.sd > _sd_limit + tol:
            issues.append(
                f"stop {s:>2}: sd={state.sd:.3f}h > {_sd_limit}h (shift driving)")
        # M5: the 13 h working cap is replaced by the shift-spread ceiling
        _h_state = getattr(state, "h", 0.0)
        if _h_state > Tspr2 + tol:
            issues.append(
                f"stop {s:>2}: spread h={_h_state:.3f}h > {Tspr2}h "
                f"(daily rest not completed within 24h window)")
        if state.e_arr < Emin - tol:
            issues.append(
                f"stop {s:>2}: soc={state.e_arr:.1f}kWh < Emin={Emin}kWh")

        if is_cust:
            # TW2: with the fixed-penalty windows an out-of-window service
            # start (early OR late) is a penalty (delta = 1, reported in the
            # metrics), not a violation; under hard_tw both directions are
            # violations.  No waiting is ever inserted (SIM1).
            Wha_s = full_data.get("Wha", {}).get(s)
            Whf_s = full_data.get("Whf", {}).get(s)
            if hard_tw and Whf_s is not None and state.t_arr > Whf_s + tol:
                issues.append(
                    f"stop {s:>2}: arrival ta={state.t_arr:.3f}h > "
                    f"Whf={Whf_s:.3f}h (hard time window — too late)")
            if hard_tw and Wha_s is not None and state.t_arr < Wha_s - tol:
                issues.append(
                    f"stop {s:>2}: arrival ta={state.t_arr:.3f}h < "
                    f"Wha={Wha_s:.3f}h (hard time window — too early)")

        # Check departure energy using realized energy (derived from SOC history).
        # Using full_data["E"] (nominal) here would cause false positives when
        # realized energy differs from nominal — the BEHDV stores the realized
        # SOC in e_arr_history, so e_dep_c − states[i+1].e_arr gives the true leg energy.
        if i < len(actions) and s < full_data["N"]:
            e_dep_c = state.e_arr
            if is_CS and y:
                tauc_i = (durs[i] if i < len(durs) else {}).get("tauc", 0.0)
                if tauc_i > 0:
                    e_dep_c = _energy_after_charging(state.e_arr, tauc_i, full_data)
            if i + 1 < len(states):
                E_leg_actual = e_dep_c - states[i + 1].e_arr
            else:
                E_leg_actual = full_data["E"].get(s, 0.0)
            if e_dep_c - E_leg_actual < Emin - tol:
                issues.append(
                    f"stop {s:>2}: energy violation — "
                    f"ed={e_dep_c:.1f} − E_act[{s}]={E_leg_actual:.1f} = "
                    f"{e_dep_c-E_leg_actual:.1f} < Emin={Emin}kWh")

    return len(issues) == 0, issues


# ══════════════════════════════════════════════════════════════════════════════
# S3 — EX-POST DIRECTIVE 2002/15/EC COMPLIANCE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_directive_compliance(results: dict, full_data: dict,
                               tol: float = 1e-3) -> dict:
    """
    S3 — Ex-post verification of the working-time break provisions of
    Directive 2002/15/EC, which are NOT modelled explicitly (paper §3.3
    argues the 4.5 h driving-break rule dominates them in long-haul
    operation; this check quantifies that claim).

    Scans the realized timeline for:
      (a) more than 6 h of consecutive WORK without a break;
      (b) total break time < 30 min in shifts with 6–9 h working time, or
          < 45 min in shifts with > 9 h working time.

    Work items per stop: driving legs, customer service, CS queue and
    maneuver time, and charging time not covered by a parallel break.
    Any declared break (or qualifying forced wait) resets the consecutive
    counter; a rest ends the shift.

    Returns
    -------
    dict(compliant, n_shifts, issues, max_consec_work)
    """
    states  = results["states"]
    actions = results["actions"]
    durs    = results.get("durations_list", [])
    K_set   = set(full_data["K"])
    C_set   = set(full_data["C"])
    L_set   = set(full_data.get("L", []))
    Twk1    = full_data.get("Twrk_cons1", 6.0)   # 6 h consecutive-work trigger
    issues  = []

    consec_work    = 0.0
    max_consec     = 0.0
    shift_work     = 0.0
    shift_break    = 0.0
    n_shifts       = 0

    def _close_shift():
        nonlocal shift_work, shift_break, n_shifts
        n_shifts += 1
        if shift_work > 9.0 + tol and shift_break < 0.75 - tol:
            issues.append(
                f"shift {n_shifts}: working {shift_work:.2f}h > 9h but total "
                f"break {shift_break*60:.0f}min < 45min")
        elif shift_work > 6.0 + tol and shift_break < 0.50 - tol:
            issues.append(
                f"shift {n_shifts}: working {shift_work:.2f}h in (6,9]h but "
                f"total break {shift_break*60:.0f}min < 30min")
        shift_work  = 0.0
        shift_break = 0.0

    n_legs = len(results.get("D_actual_list", []))
    for idx, state in enumerate(states):
        s   = state.stop
        act = actions[idx] if idx < len(actions) else {}
        dur = durs[idx]    if idx < len(durs)    else {}
        brk = act.get("break_type")
        rst = act.get("rest_type")
        y   = int(act.get("y", 0))
        sigma = int(dur.get("sigma", 0))

        # work performed AT the stop (before any break/rest)
        if s in K_set:
            work_here = (dur.get("mstop", 0.0) + dur.get("tauq", 0.0)
                         + dur.get("mseq", 0.0))
            if y and (sigma or brk is None):
                work_here += dur.get("tauc", 0.0)   # charging counted as work
        elif s in C_set:
            work_here = full_data["S"].get(s, 0.0)
        elif s in L_set:
            work_here = dur.get("mlay", 0.0)         # M8 layby parking overhead
        else:
            work_here = 0.0

        consec_work += work_here
        shift_work  += work_here

        taub = dur.get("taub", 0.0)
        if brk in ("b45", "b15", "b30") or taub >= 0.25 - tol:
            eff_break = taub + (dur.get("tauc", 0.0)
                                if (y and brk and not sigma) else 0.0)
            shift_break += eff_break
            if consec_work > Twk1 + tol:
                issues.append(
                    f"stop {s}: {consec_work:.2f}h consecutive work before "
                    f"the break (> {Twk1}h)")
            max_consec  = max(max_consec, consec_work)
            consec_work = 0.0

        if rst in ("r1", "r2"):
            max_consec  = max(max_consec, consec_work)
            consec_work = 0.0
            _close_shift()

        # drive the leg to the next stop
        if idx < n_legs:
            d_leg = results["D_actual_list"][idx]
            consec_work += d_leg
            shift_work  += d_leg
            if consec_work > Twk1 + tol:
                # violation materialises mid-leg; flagged at arrival
                pass

    max_consec = max(max_consec, consec_work)
    if consec_work > Twk1 + tol:
        issues.append(
            f"route end: {consec_work:.2f}h consecutive work without a break")
    _close_shift()   # final (unfinished) shift

    return dict(
        compliant=len(issues) == 0,
        n_shifts=n_shifts,
        issues=issues,
        max_consec_work=max_consec,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ORACLE WARM-START  (private)
# ══════════════════════════════════════════════════════════════════════════════

def _warmstart_oracle(model, full_data: dict, sim_results: dict):
    """
    Inject the simulation trajectory as a complete MIP warm-start incumbent.

    HiGHS requires EVERY variable to have a consistent value before it
    accepts a user solution (Src "X" in the B&B log).  A partial assignment
    is silently discarded — HiGHS validates the solution against all
    constraints before accepting it.

    Variables initialised
    ---------------------
    Continuous : ta, td, ea, ed, tauc, taub, taur, taub_hat, u, p,
                 cd, sd, sw, l1, l2, l4, lam_a, lam_d
    Binary     : y, v, sigma, x_b45/b15/b30, rho1/rho2, mu_a/mu_d, phi
    """
    states    = sim_results["states"]
    actions   = sim_results["actions"]
    durs      = sim_results.get("durations_list", [])
    td_list   = sim_results.get("td_list", [])
    K_set     = set(full_data["K"])
    C_set     = set(full_data["C"])
    Ebar      = full_data["Ebar"]
    Tbar      = full_data["Tbar"]
    R         = full_data["R"]
    Emin      = full_data["Emin"]

    def _pwl_weights(e_kWh):
        """Convex PWL weights for energy value e_kWh on the charging curve."""
        e = max(Ebar[R[0]], min(float(e_kWh), Ebar[R[-1]]))
        lam = {r: 0.0 for r in R}
        for j in range(len(R) - 1):
            r_lo, r_hi = R[j], R[j + 1]
            e_lo, e_hi = Ebar[r_lo], Ebar[r_hi]
            if e_lo <= e <= e_hi + 1e-9:
                span      = max(e_hi - e_lo, 1e-9)
                lam[r_hi] = (e - e_lo) / span
                lam[r_lo] = 1.0 - lam[r_hi]
                return lam, r_hi
        lam[R[-1]] = 1.0
        return lam, R[-1]

    sim_by_stop = {}
    phi_track   = 0

    for idx, state in enumerate(states):
        s   = state.stop
        act = actions[idx] if idx < len(actions) else {}
        dur = durs[idx]    if idx < len(durs)    else {}
        y   = int(act.get("y", 0))
        brk = act.get("break_type")
        rst = act.get("rest_type")
        is_CS = s in K_set

        tauc = dur.get("tauc", 0.0)
        taub = dur.get("taub", 0.0)
        taur = dur.get("taur", 0.0)
        tauq = dur.get("tauq", 0.0)

        b45  = int(brk == "b45"); b15 = int(brk == "b15"); b30 = int(brk == "b30")
        rho1 = int(rst == "r1");  rho2 = int(rst == "r2")
        xsum = b45 + b15 + b30 + rho1 + rho2
        ri   = b45 + b30 + rho1 + rho2
        rho  = rho1 + rho2

        _brk_active = bool(b45 + b15 + b30)
        # sigma from the recorded execution; M4 forces charge+rest sequential
        sigma_val = int(dur.get("sigma", 1 if (y and (rho1 + rho2)) else 0))
        v_val     = int(is_CS and (bool(y) or _brk_active or bool(rho1 + rho2)))

        # M2: g = credited charging — full tauc only when a break is declared
        # and runs in parallel with the charge
        g_val = tauc if (is_CS and y and _brk_active and sigma_val == 0) else 0.0

        # taub_hat = taub + g at CS (break credit includes parallel charging)
        taub_hat_v = taub + g_val if is_CS else taub

        # TW2: out-of-window indicator from the realized arrival (no waiting
        # is ever inserted — SIM1, so executed taub needs no correction)
        delta_val = 0
        if s in C_set:
            _wha = full_data.get("Wha", {}).get(s)
            _whf = full_data.get("Whf", {}).get(s)
            if ((_wha is not None and state.t_arr < _wha - 1e-3)
                    or (_whf is not None and state.t_arr > _whf + 1e-3)):
                delta_val = 1

        # M6: extension flag — shift currently beyond the regular 9 h limit
        z_val = int(state.sd > full_data["Tdrv_sh1"] + 1e-9)

        M_stop_val = full_data["M_stop"].get(s, 0.0) if is_CS else 0.0
        M_seq_val  = full_data["M_seq"].get(s, 0.0)  if is_CS else 0.0

        phi_now = phi_track
        if ri or b45:  phi_track = 0
        elif b15:      phi_track = 1

        if is_CS and y and tauc > 0:
            ed_val = _energy_after_charging(state.e_arr, tauc, full_data)
        else:
            ed_val = state.e_arr

        if idx < len(td_list):
            td_val = float(td_list[idx])
        elif s == 0:
            td_val = state.t_arr
        elif is_CS:
            td_val = (state.t_arr + v_val * M_stop_val + tauq * y
                      + tauc + taub + taur + sigma_val * M_seq_val)
        elif s in C_set:
            td_val = state.t_arr + full_data["S"].get(s, 0.0) + taub + taur
        else:
            td_val = state.t_arr

        lam_a_vals, mu_a_seg = _pwl_weights(state.e_arr)
        lam_d_vals, mu_d_seg = _pwl_weights(ed_val)

        # spread h and its pre-rest dwell (o = td − ta − taur)
        h_val = float(getattr(state, "h", 0.0))
        o_val = max(0.0, td_val - state.t_arr - taur)

        sim_by_stop[s] = dict(
            ta=state.t_arr, td=td_val,
            ea=state.e_arr, ed=ed_val,
            cd=state.cd, sd=state.sd, sw=state.sw, phi=phi_now,
            h=h_val, z=z_val, delta=delta_val,
            y=y, b45=b45, b15=b15, b30=b30, rho1=rho1, rho2=rho2,
            tauc=tauc, taub=taub, taur=taur, taub_hat=taub_hat_v,
            v=v_val, sigma=sigma_val, g=g_val,
            l1=float(state.cd) if ri  else 0.0,
            l2=float(state.sd) if rho else 0.0,
            l4=float(state.sw) if rho else 0.0,
            l5=(h_val + o_val) if rho else 0.0,
            lam_a=lam_a_vals, mu_a_seg=mu_a_seg,
            lam_d=lam_d_vals, mu_d_seg=mu_d_seg,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress Pyomo W1001/W1002

        for i in model.I:
            sv = sim_by_stop.get(i)
            if sv is None:
                continue
            try:
                model.ta[i].set_value(sv["ta"])
                model.td[i].set_value(sv["td"])
                model.ea[i].set_value(max(sv["ea"], Emin))
                model.ed[i].set_value(max(sv["ed"], Emin))
                model.cd[i].set_value(sv["cd"])
                model.sd[i].set_value(sv["sd"])
                model.sw[i].set_value(sv["sw"])
                model.h[i].set_value(sv["h"])
                model.z[i].set_value(sv["z"])
                model.phi[i].set_value(sv["phi"])
                model.taub[i].set_value(sv["taub"])
                model.taur[i].set_value(sv["taur"])
                model.taub_hat[i].set_value(sv["taub_hat"])
                model.x_b45[i].set_value(sv["b45"])
                model.x_b15[i].set_value(sv["b15"])
                model.x_b30[i].set_value(sv["b30"])
                model.rho1[i].set_value(sv["rho1"])
                model.rho2[i].set_value(sv["rho2"])
                model.l1[i].set_value(sv["l1"])
                model.l2[i].set_value(sv["l2"])
                model.l4[i].set_value(sv["l4"])
                model.l5[i].set_value(sv["l5"])
            except Exception:
                pass

        for i in model.Cset:
            sv = sim_by_stop.get(i)
            if sv is None:
                continue
            try:
                model.delta[i].set_value(sv["delta"])
            except Exception:
                pass

        for i in model.Kset:
            sv = sim_by_stop.get(i)
            if sv is None:
                continue
            try:
                model.y[i].set_value(sv["y"])
                model.tauc[i].set_value(sv["tauc"])
                model.g[i].set_value(sv["g"])
                model.v[i].set_value(sv["v"])
                model.sigma[i].set_value(sv["sigma"])
                for r in model.Rset:
                    model.lam_a[i, r].set_value(sv["lam_a"].get(r, 0.0))
                    model.lam_d[i, r].set_value(sv["lam_d"].get(r, 0.0))
                mu_a_seg = sv["mu_a_seg"]
                mu_d_seg = sv["mu_d_seg"]
                for r in model.RsegS:
                    model.mu_a[i, r].set_value(1 if r == mu_a_seg else 0)
                    model.mu_d[i, r].set_value(1 if r == mu_d_seg else 0)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER-RESULT INTROSPECTION  (gap + why the solve stopped)
# ══════════════════════════════════════════════════════════════════════════════

def _bounds_from_results(res):
    """Best incumbent (upper_bound) and best proven bound (lower_bound) from a
    Pyomo results object, or (None, None) if unavailable."""
    if res is None:
        return None, None
    try:
        prob = res.problem[0] if hasattr(res.problem, "__getitem__") else res.problem
        def _f(x):
            try:
                x = float(x)
                return x if abs(x) < 1e18 else None
            except (TypeError, ValueError):
                return None
        return _f(getattr(prob, "lower_bound", None)), \
               _f(getattr(prob, "upper_bound", None))
    except Exception:
        return None, None


def _relative_gap(incumbent, best_bound):
    """Gurobi-style relative MIP gap |inc - bnd| / |inc|, or nan if unknown."""
    if incumbent is None or best_bound is None:
        return float("nan")
    denom = max(abs(incumbent), 1e-10)
    return abs(incumbent - best_bound) / denom


def _classify_stop_reason(status: str, gap_val: float, mip_gap: float,
                          has_solution: bool) -> str:
    """
    Map the solver termination + gap onto WHY the solve stopped:

      'optimal'       — proven optimal (gap ~ 0)
      'gap_threshold' — stopped because the MIPGap tolerance was reached
      'time_limit'    — the wall-clock TimeLimit was hit
      'aborted'       — user interrupt / other early termination
      'infeasible'    — model proven infeasible (no schedule exists)
      'no_solution'   — stopped with no incumbent (e.g. time limit, 0 feasible)
    """
    st = (status or "").lower()
    if "infeasible" in st:
        return "infeasible"
    if not has_solution:
        # stopped without ever finding a feasible schedule
        return "time_limit" if ("time" in st or "limit" in st) else "no_solution"
    if "time" in st and "limit" in st:      # maxTimeLimit
        return "time_limit"
    if st in ("optimal",):
        # Gurobi returns OPTIMAL both when proven optimal AND when it merely
        # reaches the MIPGap tolerance; separate the two on the realised gap.
        if not np.isnan(gap_val):
            if gap_val <= 1e-6:
                return "optimal"
            if gap_val <= mip_gap * (1.0 + 1e-6):
                return "gap_threshold"
        return "optimal"
    if st in ("userinterrupt", "interrupted", "maxiterations", "other",
              "unknown", "solverfailure", "internalsolvererror"):
        return "aborted"
    return st or "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# ORACLE SOLVE
# ══════════════════════════════════════════════════════════════════════════════

def oracle_solve(full_data: dict, D_actual_list: list,
                 sim_results: dict = None,
                 time_limit: int   = 12 * 3600,
                 mip_gap: float    = 0.005,
                 tee: bool         = True,
                 verbose: bool     = True,
                 log_fh            = None,
                 log_file: str | None = None) -> dict:
    """
    Solve the full deterministic MILP with the travel times that actually
    occurred during the simulation (perfect hindsight).

    The simulation trajectory is injected as a warm-start incumbent so that
    HiGHS starts with a known feasible solution and focuses on improving it.
    This is critical for large instances where proving optimality takes hours.

    Parameters
    ----------
    full_data     : dict from instances.make_data()
    D_actual_list : list[float], length N — realised leg travel times (h)
    sim_results   : dict from run_simulation / run_greedy (for warm-start),
                    or None to solve without warm-start
    time_limit    : int  — solver wall-clock limit in seconds (default 6 h)
    tee           : bool — print HiGHS solver log to stdout
    verbose       : bool — print summary lines to stdout and log_fh
    log_fh        : open file handle for log output (optional)

    Returns
    -------
    dict with keys:
        feasible    : bool
        optimal     : bool  — True only if proven optimal (gap ~ 0)
        obj         : float — best arrival time found (h), or inf if infeasible
        gap         : float — relative MIP optimality gap at termination
        best_bound  : float — best proven lower bound (None if unavailable)
        stop_reason : str   — WHY the solve stopped: 'optimal', 'gap_threshold',
                              'time_limit', 'aborted', 'infeasible', 'no_solution'
        time_limit  : int   — the wall-clock limit (s) the solve ran under
        mip_gap     : float — the MIPGap tolerance the solve ran under
        status      : str   — raw solver termination-condition string
        sol         : list  — per-stop dicts from MILP.extract_solution
        D_actual    : dict  — {leg_index: duration_h} (realised travel times)
    """
    N = full_data["N"]
    assert len(D_actual_list) == N, (
        f"D_actual_list has {len(D_actual_list)} entries but route has {N} legs")

    D_actual_dict = {i: D_actual_list[i] for i in range(N)}
    oracle_data   = dict(full_data)
    oracle_data["D"] = D_actual_dict

    # Hindsight energy MUST match the REALISED speed on each leg — the same
    # ECR coupling used to draw E_real: E = L·ECR(L / D_actual).  Overriding D
    # without E would leave the oracle on NOMINAL energy while every policy
    # executes on realised energy, making the "hindsight" benchmark unfairly
    # easy (feasible where the policies strand).
    from scenarios import _ecr
    _km  = full_data.get("km", {})
    _Enom = full_data.get("E", {})
    E_actual_dict = {}
    for i in range(N):
        d = D_actual_dict[i]
        if i in _km and d > 1e-9:
            L = _km[i]
            E_actual_dict[i] = L * _ecr(L / d)
        else:
            E_actual_dict[i] = _Enom.get(i, 0.0)   # no km → keep nominal
    oracle_data["E"] = E_actual_dict

    lb_t, ub_t = compute_time_bounds(
        oracle_data["I"], oracle_data["C"], oracle_data["K"],
        D_actual_dict, oracle_data["S"], oracle_data["Q"],
        oracle_data["Tbar"], oracle_data["T_hor"],
        t0=oracle_data.get("T_START", 8.0))
    oracle_data["lb_t"] = lb_t
    oracle_data["ub_t"] = ub_t

    def _op(msg):
        if verbose: print(msg)
        if log_fh:
            try:
                print(msg, file=log_fh)
                log_fh.flush()   # flush per line so the .txt is readable MID-run,
                                 # not only after the (possibly hour-long) solve
                                 # finishes and the handle is closed.
            except Exception: pass

    ws_str = " + sim warm-start" if sim_results is not None else ""
    h, m   = time_limit // 3600, (time_limit % 3600) // 60
    tl_str = f"{h}h{m:02d}m" if h else f"{time_limit}s"
    _op(f"\n{'='*65}")
    _op(f"  ORACLE SOLVE  (hindsight-optimal{ws_str})")
    _op(f"  time_limit={tl_str}")
    _op(f"{'='*65}")

    model = build_model(oracle_data)

    if sim_results is not None:
        _warmstart_oracle(model, oracle_data, sim_results)
        _op(f"  Warm-start: sim arrival {sim_results['total_time']:.3f}h "
            f"injected as incumbent")

    solver = pyo.SolverFactory("gurobi")
    solver.options["MIPGap"]     = mip_gap
    solver.options["TimeLimit"]  = time_limit
    solver.options["Heuristics"] = 0.2
    if log_file is not None:
        # Persist Gurobi's full branch-and-bound log (incumbent / best-bound
        # node table) to disk so the bound evolution can be traced/plotted; the
        # normal pipeline leaves this None and the solver log is not saved.
        solver.options["LogFile"] = log_file

    if tee:
        # Write solver output directly to stdout.
        # No redirect_stdout wrapper: Gurobi output is guaranteed to appear on
        # the terminal regardless of how the solver writes to stdout/stderr.
        # If log_fh is provided the solver log is not duplicated there (summary
        # lines from _op still go to log_fh as usual).
        try:
            res    = solver.solve(model, tee=True, warmstart=True,
                                  load_solution=False)
            status = str(res.solver.termination_condition)
            if status not in ("infeasible",):
                model.solutions.load_from(res)
        except Exception:
            try:
                res    = solver.solve(model, tee=True, warmstart=True)
                status = str(res.solver.termination_condition)
            except RuntimeError:
                status = "infeasible"; res = None
        if log_fh:
            print("\n[ORACLE SOLVER OUTPUT: printed to terminal (tee=True)]",
                  file=log_fh)
    else:
        _sink = _io.StringIO()
        try:
            with _ctx.redirect_stdout(_sink), _ctx.redirect_stderr(_sink):
                res = solver.solve(model, tee=False, warmstart=True)
            status = str(res.solver.termination_condition)
        except RuntimeError:
            status = "infeasible"; res = None

    _op(f"  Status : {status}")

    has_solution = False
    obj_val      = float("inf")

    if status in ("optimal", "feasible", "maxTimeLimit"):
        try:
            obj_val      = pyo.value(model.obj)
            has_solution = obj_val is not None and obj_val < 1e8
        except Exception:
            pass

    # Best proven bound + gap, taken from the solver's reported bounds where
    # available (reliable), falling back to the termination message regex.
    best_bound, _ub = _bounds_from_results(res)
    incumbent = obj_val if has_solution else _ub
    gap_val   = _relative_gap(incumbent, best_bound)
    if np.isnan(gap_val):
        try:
            gap_raw = res.solver.termination_condition_message
            m_      = re.search(r"gap[^0-9]*([0-9.e+-]+)%", str(gap_raw), re.I)
            if m_:
                gap_val = float(m_.group(1)) / 100
            elif status == "optimal":
                gap_val = 0.0
        except Exception:
            if status == "optimal":
                gap_val = 0.0

    stop_reason = _classify_stop_reason(status, gap_val, mip_gap, has_solution)

    if not has_solution:
        _op(f"  No feasible solution found (stop_reason={stop_reason}, "
            f"best_bound={best_bound}).")
        return dict(feasible=False, optimal=False, obj=float("inf"),
                    gap=float("inf"), best_bound=best_bound,
                    stop_reason=stop_reason, status=status,
                    time_limit=time_limit, mip_gap=mip_gap,
                    sol=[], D_actual=D_actual_dict)

    is_optimal = (stop_reason == "optimal")
    sol        = extract_solution(model, oracle_data)

    if verbose:
        gap_str = "" if np.isnan(gap_val) else f", gap ≈ {gap_val:.2%}"
        _op(f"  Oracle arrival : {obj_val:.3f} h  "
            f"[stop_reason={stop_reason}{gap_str}]")

    return dict(feasible=True, optimal=is_optimal, obj=obj_val,
                gap=gap_val, best_bound=best_bound, stop_reason=stop_reason,
                status=status, time_limit=time_limit, mip_gap=mip_gap,
                sol=sol, D_actual=D_actual_dict)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULE REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_simulation_log(results: dict, full_data: dict):
    """Print the simulation trajectory as a formatted table."""
    N     = full_data["N"]
    C_set = set(full_data["C"])
    K_set = set(full_data["K"])

    hdr = (f"  {'stop':>4}  {'type':>5}  {'t_arr':>7}  {'soc':>6}  "
           f"{'cd':>5}  {'sd':>5}  {'sw':>5}  "
           f"{'y':>2}  {'brk':>4}  {'rst':>4}  action")
    print(f"\n  === SIMULATION TRAJECTORY ===")
    print(f"{hdr}\n  {'─'*95}")

    for state, action in zip(results["states"], results["actions"]):
        stop = state.stop
        typ  = ("ORIG" if stop == 0 else "DEST" if stop == N else
                "CUST" if stop in C_set else "CS")
        brk  = action.get("break_type") or "—"
        rst  = action.get("rest_type")  or "—"
        y    = action.get("y", 0)
        acts = []
        if y:           acts.append("CHARGE")
        if brk != "—":  acts.append(f"BRK-{brk.upper()}")
        if rst != "—":  acts.append(f"REST-{rst.upper()}")

        print(f"  {stop:>4}  {typ:>5}  "
              f"{state.t_arr:>7.3f}  {state.e_arr:>6.1f}  "
              f"{state.cd:>5.2f}  {state.sd:>5.2f}  {state.sw:>5.2f}  "
              f"{y:>2}  {brk:>4}  {rst:>4}  "
              f"{', '.join(acts) or '—'}")


def print_oracle_log(oracle: dict, full_data: dict):
    """Print the oracle (hindsight-optimal) schedule as a formatted table."""
    if not oracle.get("feasible") or not oracle.get("sol"):
        print("  Oracle: no feasible solution available.")
        return

    N     = full_data["N"]
    C_set = set(full_data["C"])
    K_set = set(full_data["K"])
    sol   = oracle["sol"]

    gap   = oracle.get("gap", float("nan"))
    opt_s = (" (optimal)" if oracle.get("optimal") else
             f" (gap ≈ {gap:.1%})" if not np.isnan(gap) else " (feasible)")

    hdr = (f"  {'stop':>4}  {'type':>5}  {'t_arr':>7}  {'soc':>6}  "
           f"{'cd':>5}  {'sd':>5}  {'sw':>5}  "
           f"{'y':>2}  {'brk':>4}  {'rst':>4}  action")
    print(f"\n  === ORACLE SCHEDULE  (arrival {oracle['obj']:.3f}h{opt_s}) ===")
    print(f"{hdr}\n  {'─'*95}")

    for s in sol:
        stop = s["i"]
        typ  = ("ORIG" if stop == 0 else "DEST" if stop == N else
                "CUST" if stop in C_set else "CS")
        brk  = ("b45" if s["b45"] else "b15" if s["b15"] else
                "b30" if s["b30"] else "—")
        rst  = ("r1"  if s["rho1"] else "r2" if s["rho2"] else "—")
        y    = s.get("y", 0)
        acts = []
        if y:           acts.append("CHARGE")
        if brk != "—":  acts.append(f"BRK-{brk.upper()}")
        if rst != "—":  acts.append(f"REST-{rst.upper()}")

        print(f"  {stop:>4}  {typ:>5}  "
              f"{s['ta']:>7.3f}  {s['ea']:>6.1f}  "
              f"{s['cd']:>5.2f}  {s['sd']:>5.2f}  {s['sw']:>5.2f}  "
              f"{y:>2}  {brk:>4}  {rst:>4}  "
              f"{', '.join(acts) or '—'}")