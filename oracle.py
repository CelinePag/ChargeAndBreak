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
import re
import warnings

import numpy as np
import pyomo.environ as pyo

from BEHDV     import _energy_after_charging
from MILP      import build_model, extract_solution
from instances import compute_time_bounds


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

    Tdrv_cons = full_data["Tdrv_cons"]
    Tdrv_sh1  = full_data["Tdrv_sh1"]
    Twrk_sh   = full_data["Twrk_sh"]
    Emin      = full_data["Emin"]
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
        if is_CS:
            work = tauq * y + (tauc if (y and not brk and not rst) else 0.0)
        elif is_cust:
            work = full_data["S"].get(s, 0.0)
        else:
            work = 0.0
        sw_k = state.sw + work

        if state.cd > Tdrv_cons + tol:
            issues.append(
                f"stop {s:>2}: cd={state.cd:.3f}h > {Tdrv_cons}h (consec driving)")
        if state.sd > Tdrv_sh1 + tol:
            issues.append(
                f"stop {s:>2}: sd={state.sd:.3f}h > {Tdrv_sh1}h (shift driving)")
        if sw_k > Twrk_sh + tol:
            issues.append(
                f"stop {s:>2}: sw={sw_k:.3f}h > {Twrk_sh}h (shift working)")
        if state.e_arr < Emin - tol:
            issues.append(
                f"stop {s:>2}: soc={state.e_arr:.1f}kWh < Emin={Emin}kWh")

        # Check departure energy: advance clips e_arr to Emin, hiding violations
        if i < len(actions) and s < full_data["N"]:
            E_leg   = full_data["E"].get(s, 0.0)
            e_dep_c = state.e_arr
            if is_CS and y:
                tauc_i = (durs[i] if i < len(durs) else {}).get("tauc", 0.0)
                if tauc_i > 0:
                    e_dep_c = _energy_after_charging(state.e_arr, tauc_i, full_data)
            if e_dep_c - E_leg < Emin - tol:
                issues.append(
                    f"stop {s:>2}: energy violation — "
                    f"ed={e_dep_c:.1f} − E[{s}]={E_leg:.1f} = "
                    f"{e_dep_c-E_leg:.1f} < Emin={Emin}kWh (clipped in sim)")

    return len(issues) == 0, issues


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
    Continuous : ta, td, ea, ed, tauc, taub, taur, taub_hat, u, z_man,
                 cd, sd, sw, l1, l2, l4, lam_a, lam_d
    Binary     : y, x_b45/b15/b30, rho1/rho2, mu_a/mu_d, phi
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

        # Manoeuver: rest always; break only when NOT synchronized with charging
        _brk_active = bool(b45 + b15 + b30)
        _brk_unsync = _brk_active and not (is_CS and bool(y))
        z_man_val   = float(_brk_unsync or bool(rho1 + rho2))
        taub_hat_v = taub + tauc if is_CS else taub
        u_val      = tauc if (is_CS and y and not xsum) else 0.0

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
            td_val = (state.t_arr + tauq * y + tauc + taub + taur
                      + full_data["M"].get(s, 0.0) * z_man_val)
        elif s in C_set:
            td_val = (state.t_arr + full_data["S"].get(s, 0.0) + taub + taur
                      + full_data["M"].get(s, 0.0) * z_man_val)
        else:
            td_val = state.t_arr

        lam_a_vals, mu_a_seg = _pwl_weights(state.e_arr)
        lam_d_vals, mu_d_seg = _pwl_weights(ed_val)

        sim_by_stop[s] = dict(
            ta=state.t_arr, td=td_val,
            ea=state.e_arr, ed=ed_val,
            cd=state.cd, sd=state.sd, sw=state.sw, phi=phi_now,
            y=y, b45=b45, b15=b15, b30=b30, rho1=rho1, rho2=rho2,
            tauc=tauc, taub=taub, taur=taur, taub_hat=taub_hat_v,
            u=u_val, z_man=z_man_val,
            l1=float(state.cd) if ri  else 0.0,
            l2=float(state.sd) if rho else 0.0,
            l4=float(state.sw) if rho else 0.0,
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
                model.phi[i].set_value(sv["phi"])
                model.taub[i].set_value(sv["taub"])
                model.taur[i].set_value(sv["taur"])
                model.taub_hat[i].set_value(sv["taub_hat"])
                model.x_b45[i].set_value(sv["b45"])
                model.x_b15[i].set_value(sv["b15"])
                model.x_b30[i].set_value(sv["b30"])
                model.rho1[i].set_value(sv["rho1"])
                model.rho2[i].set_value(sv["rho2"])
                model.z_man[i].set_value(sv["z_man"])
                model.l1[i].set_value(sv["l1"])
                model.l2[i].set_value(sv["l2"])
                model.l4[i].set_value(sv["l4"])
            except Exception:
                pass

        for i in model.Kset:
            sv = sim_by_stop.get(i)
            if sv is None:
                continue
            try:
                model.y[i].set_value(sv["y"])
                model.tauc[i].set_value(sv["tauc"])
                model.u[i].set_value(sv["u"])
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
# ORACLE SOLVE
# ══════════════════════════════════════════════════════════════════════════════

def oracle_solve(full_data: dict, D_actual_list: list,
                 sim_results: dict = None,
                 time_limit: int   = 12 * 3600,
                 tee: bool         = True,
                 verbose: bool     = True,
                 log_fh            = None) -> dict:
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
        feasible : bool
        optimal  : bool  — True only if proven optimal within time_limit
        obj      : float — best arrival time found (h), or inf if infeasible
        gap      : float — MIP optimality gap at termination (0.0 if optimal)
        sol      : list  — per-stop dicts from MILP.extract_solution
        status   : str   — HiGHS termination condition string
        D_actual : dict  — {leg_index: duration_h} (the realised travel times)
    """
    N = full_data["N"]
    assert len(D_actual_list) == N, (
        f"D_actual_list has {len(D_actual_list)} entries but route has {N} legs")

    D_actual_dict = {i: D_actual_list[i] for i in range(N)}
    oracle_data   = dict(full_data)
    oracle_data["D"] = D_actual_dict

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
            try: print(msg, file=log_fh)
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
    solver.options["MIPGap"]     = 0.005
    solver.options["TimeLimit"]  = time_limit
    solver.options["Heuristics"] = 0.2

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

    if not has_solution:
        _op("  No feasible solution found within time limit.")
        return dict(feasible=False, optimal=False, obj=float("inf"),
                    gap=float("inf"), sol=[], status=status,
                    D_actual=D_actual_dict)

    is_optimal = (status == "optimal")
    sol        = extract_solution(model, oracle_data)

    try:
        gap_raw = res.solver.termination_condition_message
        m_      = re.search(r"gap[^0-9]*([0-9.e+-]+)%", str(gap_raw), re.I)
        gap_val = float(m_.group(1)) / 100 if m_ else (
                  0.0 if is_optimal else float("nan"))
    except Exception:
        gap_val = 0.0 if is_optimal else float("nan")

    if verbose:
        opt_str = (" (optimal)" if is_optimal else
                   f" (gap ≈ {gap_val:.1%})" if not np.isnan(gap_val) else "")
        _op(f"  Oracle arrival : {obj_val:.3f} h{opt_str}")

    return dict(feasible=True, optimal=is_optimal, obj=obj_val,
                gap=gap_val, sol=sol, status=status, D_actual=D_actual_dict)


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