"""
fixed_y_milp.py — Fixed-Charging-Location MILP for BET scheduling
===================================================================
Solves the full-route deterministic MILP with the binary charging-location
decisions y_i pre-fixed.  All other decisions (break/rest types, charging
amounts, activity durations) are optimised freely by the solver.

Theoretical motivation
----------------------
The full MILP contains two layers of binary decisions:

  (1) Charging locations : y_i ∈ {0,1}  for i ∈ K
  (2) Break/rest types   : x_b45, x_b15, x_b30, rho1, rho2  at every stop

Fixing layer (1) eliminates the bilinear coupling term
    u_i = tau_i^c · (1 − x_i − rho_i)
from depending on y_i, and forces tau_i^c = 0 at non-charging stops,
removing the PWL segment variables mu_a, mu_d from those stops entirely
(they become trivially determined by the SOC propagation).

The remaining sub-problem P(y) — optimising over break/rest types and
charging amounts given fixed y — is the "charging-location sub-problem"
analysed in the paper.  Its structure (SPPRC on a path) makes it
solvable in polynomial time via label-setting DP; for now we delegate
to the MILP solver, using this module to isolate the sub-problem.

Y-source interface
------------------
Any object that implements  get_y(data: dict) -> dict[int, int]  is a
valid y-source.  The method must return a mapping  {cs_stop: 0_or_1}
covering every stop in data["K"].

Current y-sources
-----------------
  FromSolutionFile(solution_name)
      Reads y_i from a previously saved solution JSON in solutions/.
      Expects the MILP.save_solution() schema: { "data": {...}, "sol": [...] }.

  FromOracleFile(name)
      Reads y_i from an oracle cache JSON in solutions/ written by
      runner.finalize_run().  Schema: { "feasible": ..., "sol": [...] }.
      Accepts the name with or without the "oracle_" prefix and with or
      without the ".json" extension.

  FromSolList(sol)
      Like FromSolutionFile but takes a solution list already in memory,
      avoiding a round-trip through the filesystem.

  AllChargeY()
      y_i = 1 at every CS stop.  Upper bound on charging cost; useful to
      check whether the route is energy-feasible with greedy full-charging.

  NoChargeY()
      y_i = 0 everywhere.  The sub-problem will be infeasible whenever
      charging is required to survive a long leg.

  ManualY(y_dict)
      Pass an arbitrary {cs_stop: 0_or_1} dict directly.

Adding new y-sources
--------------------
Implement a class with:
    def get_y(self, data: dict) -> dict[int, int]: ...
    @property
    def name(self) -> str: ...

Then pass an instance to run_fixed_y().  Example future sources:
  GreedyChargeY()     — extract y from run_greedy()
  RollingHorizonY()   — extract y from run_simulation()
  TheoreticalMinY()   — minimum charging stops (energy lower bound)
  BendersOracleY()    — y from Benders master-problem relaxation

Usage
-----
  # Named instance + oracle cache (your original command, now works)
  python fixed_y_milp.py instances/RmediumCmany_1.json oracle_RmediumCmany_1.json

  # Named instance + MILP solution
  python fixed_y_milp.py realistic realistic

  # Compare fixed-y vs free MILP
  python fixed_y_milp.py instances/RmediumCmany_1.json oracle_RmediumCmany_1.json --compare

  # AllChargeY — no solution file needed
  python fixed_y_milp.py instances/RmediumCmany_1.json --all-charge --tee

Import chain
------------
  fixed_y_milp.py → MILP (build_model, solve_model, extract_solution,
                            save_solution, load_solution, print_schedule,
                            check_solution, solution_path)
                  → plots (plot_solution)
                  → instances (ALL_INSTANCES)
  No new local dependencies beyond what MILP.py already imports.
"""

from __future__ import annotations

import json
import os
import sys

import pyomo.environ as pyo

from MILP import (
    build_model,
    solve_model,
    extract_solution,
    save_solution,
    load_solution,
    print_schedule,
    check_solution,
    SOLUTIONS_DIR,
)
from plots import plot_solution


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Y-SOURCE PROTOCOL AND CONCRETE IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

class FromSolutionFile:
    """
    Derive y_i from a previously saved solution JSON in solutions/.

    Parameters
    ----------
    solution_name : str
        Stem of the file (without .json).  E.g. "realistic" reads
        solutions/realistic.json.

    Compatibility
    -------------
    The solution was saved by MILP.save_solution(), which stores each stop
    as a dict with a "y" key (0 or 1) and an "is_K" key.  Stops not in
    data["K"] are silently ignored; CS stops present in data["K"] but
    absent from the solution default to y=0 with a printed warning.
    """

    def __init__(self, solution_name: str):
        self._solution_name = solution_name

    @property
    def name(self) -> str:
        return f"solution_file({self._solution_name})"

    def get_y(self, data: dict) -> dict[int, int]:
        sol, _ = load_solution(self._solution_name)
        return _y_from_sol_with_validation(sol, data, self._solution_name)


class FromSolList:
    """
    Derive y_i from a solution list already in memory.

    Useful when the unconstrained MILP has just been solved in the same
    script and you want to immediately re-solve with those y values fixed.

    Parameters
    ----------
    sol  : list of stop-dicts (output of MILP.extract_solution())
    name : optional label for reporting (default: "in_memory_solution")
    """

    def __init__(self, sol: list, name: str = "in_memory_solution"):
        self._sol  = sol
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_y(self, data: dict) -> dict[int, int]:
        return _y_from_sol_with_validation(self._sol, data, self._name)


class FromOracleFile:
    """
    Read charging decisions from an oracle cache JSON in solutions/.

    Oracle cache files are written by runner.finalize_run() at
    ``solutions/oracle_{title}.json``.  Their schema differs from the
    MILP.save_solution() format: the ``sol`` list sits at the top level
    of the JSON, with no enclosing ``"data"`` wrapper.

    Parameters
    ----------
    name : str
        Stem of the file in solutions/, with or without the ``oracle_``
        prefix and with or without the ``.json`` extension.
        All four of these refer to the same file:
          ``oracle_RmediumCmany_1``
          ``oracle_RmediumCmany_1.json``
          ``RmediumCmany_1``   (prefix added automatically)
          ``RmediumCmany_1.json``

    Notes
    -----
    The oracle cache uses ``str(k)`` for all dict keys (JSON limitation)
    and stores them with a ``_ser`` serialiser in runner.py.  This class
    applies the inverse ``_restore_int_keys`` pass on load so that stop
    indices and leg indices are integers, matching the format expected by
    _y_from_sol_with_validation.
    """

    def __init__(self, name: str):
        # Normalise: strip .json suffix, ensure oracle_ prefix
        stem = name
        if stem.endswith(".json"):
            stem = stem[:-5]
        if not stem.startswith("oracle_"):
            stem = "oracle_" + stem
        self._stem = stem

    @property
    def name(self) -> str:
        return f"oracle_file({self._stem})"

    def get_y(self, data: dict) -> dict[int, int]:
        path = os.path.join(SOLUTIONS_DIR, f"{self._stem}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Oracle cache not found at '{path}'.\n"
                f"Run the simulation or oracle solve first so that "
                f"runner.finalize_run() creates the cache file.")

        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        # Restore integer keys (runner._ser converts all keys to str)
        payload = _restore_int_keys(payload)

        sol = payload.get("sol")
        if sol is None:
            raise KeyError(
                f"Oracle cache at '{path}' has no 'sol' field. "
                f"The file may be corrupted or from an older format.")

        return _y_from_sol_with_validation(sol, data, self._stem)


class AllChargeY:
    """
    y_i = 1 at every CS stop.

    Useful as an energy-feasibility upper bound: if the route is infeasible
    even when charging at every CS, the instance has no solution.
    Also provides an upper bound on charging cost for Benders decomposition.
    """

    @property
    def name(self) -> str:
        return "all_charge"

    def get_y(self, data: dict) -> dict[int, int]:
        return {k: 1 for k in data["K"]}


class NoChargeY:
    """
    y_i = 0 at every CS stop.

    The resulting sub-problem is infeasible whenever charging is required.
    Useful only for testing the MILP feasibility check or for instances
    with no energy constraints.
    """

    @property
    def name(self) -> str:
        return "no_charge"

    def get_y(self, data: dict) -> dict[int, int]:
        return {k: 0 for k in data["K"]}


class ManualY:
    """
    Pass a {cs_stop: 0_or_1} dict directly.

    Stops in data["K"] that are absent from the dict default to y=0.

    Parameters
    ----------
    y_dict : dict[int, int]
        Charging decisions, e.g. {3: 1, 7: 0, 11: 1}.
    name   : optional label for reporting.
    """

    def __init__(self, y_dict: dict[int, int], name: str = "manual"):
        self._y_dict = dict(y_dict)
        self._name   = name

    @property
    def name(self) -> str:
        return self._name

    def get_y(self, data: dict) -> dict[int, int]:
        y_fixed = {k: 0 for k in data["K"]}
        for k, v in self._y_dict.items():
            if k in y_fixed:
                y_fixed[k] = int(v)
            else:
                print(f"  [ManualY] WARNING: stop {k} is not in data['K'] "
                      f"— ignored.")
        return y_fixed


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def extract_y_from_sol(sol: list) -> dict[int, int]:
    """
    Extract the charging-decision vector from a solution list.

    Parameters
    ----------
    sol : list of stop-dicts from MILP.extract_solution() or
          MILP.extract_horizon_solution()

    Returns
    -------
    dict[int, int] — {cs_stop: 0_or_1} for every stop where is_K is True.

    Notes
    -----
    This is a standalone convenience function; the y-source classes call
    _y_from_sol_with_validation() internally which adds consistency checks.
    """
    return {s["i"]: int(round(s.get("y", 0)))
            for s in sol if s.get("is_K", False)}


def validate_y_fixed(y_fixed: dict[int, int], data: dict) -> bool:
    """
    Validate y_fixed for structural and energy feasibility.

    Structural checks (raise ValueError):
      - All keys in y_fixed must be in data["K"].
      - All values must be 0 or 1.
      - Every stop in data["K"] must be present in y_fixed.

    Energy check (print warning, return False):
      Run a forward pass assuming "charge to full" at every y=1 stop.
      If the SOC still drops below Emin on any leg, the sub-problem will
      be infeasible regardless of HoS decisions.

    Parameters
    ----------
    y_fixed : dict[int, int]
    data    : dict from instances.make_data()

    Returns
    -------
    bool — True if the advisory energy check passes, False otherwise.
           Structural errors raise ValueError rather than returning False.
    """
    K_set = set(data["K"])

    # ── Structural checks ─────────────────────────────────────────────────────
    extra = set(y_fixed.keys()) - K_set
    if extra:
        raise ValueError(
            f"y_fixed contains stops {sorted(extra)} that are not in "
            f"data['K'] = {sorted(K_set)}.")

    missing = K_set - set(y_fixed.keys())
    if missing:
        raise ValueError(
            f"y_fixed is missing CS stops {sorted(missing)}. "
            f"All CS stops in data['K'] must be covered.")

    for k, v in y_fixed.items():
        if v not in (0, 1):
            raise ValueError(
                f"y_fixed[{k}] = {v!r} is not a valid binary value (must be 0 or 1).")

    # ── Advisory energy forward check ─────────────────────────────────────────
    # Assumes "charge to full" at each y=1 stop (optimistic upper bound on SOC).
    # Any violation here guarantees infeasibility of the sub-problem.
    N      = data["N"]
    Ecap   = data["Ecap"]
    Emin   = data["Emin"]
    E      = data["E"]
    charged = {k for k, v in y_fixed.items() if v == 1}

    e_arr = float(data["E0"])
    violations = []

    for stop in range(N + 1):
        if e_arr < Emin - 1e-6:
            violations.append((stop, round(e_arr, 3)))
        e_dep = Ecap if stop in charged else e_arr
        if stop < N:
            e_arr = e_dep - E.get(stop, 0.0)

    if violations:
        print(f"  [validate_y_fixed] WARNING: energy infeasibility on "
              f"{len(violations)} stop(s) even with full charging at y=1 stops: "
              f"{violations[:4]}{'...' if len(violations) > 4 else ''}.")
        print(f"  The sub-problem will likely be infeasible with this y_fixed.")
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — MODEL CONSTRUCTION WITH FIXED y
# ══════════════════════════════════════════════════════════════════════════════

def apply_fixed_y(model: pyo.ConcreteModel,
                  y_fixed: dict[int, int],
                  data: dict) -> None:
    """
    Fix the charging-location binary variables on an already-built Pyomo model.

    Modifies ``model`` in place.  For each CS stop i ∈ K:
      - m.y[i] is fixed to y_fixed[i].
      - If y_fixed[i] = 0: m.tauc[i] is also fixed to 0.0.
        This makes the "no charging" condition explicit to the solver
        without relying on the big-M propagation through chg_act.
        It also makes the PWL variables (lam_a, lam_d, mu_a, mu_d)
        trivially consistent at that stop (ed[i] = ea[i] is forced by
        the combination of soc_mono_K and pwl_no_free_charge).
      - If y_fixed[i] = 1: only m.y[i] is fixed; tauc[i] and all PWL
        variables remain free so the solver optimises the charging amount.

    After this call the model retains only break/rest binaries
    (x_b45, x_b15, x_b30, rho1, rho2, z_man) and, at y=1 stops, the
    PWL segment binaries (mu_a, mu_d) as integer degrees of freedom.

    Parameters
    ----------
    model   : ConcreteModel from MILP.build_model(data)
    y_fixed : dict {cs_stop: 0_or_1} — validated by validate_y_fixed()
    data    : the same data dict used to build the model
    """
    K_set = set(data["K"])

    for k in K_set:
        if k not in model.y:
            # Should not happen for a correctly built model, but guard anyway.
            continue

        y_val = int(y_fixed.get(k, 0))
        model.y[k].fix(y_val)

        if y_val == 0:
            # Fix tauc to zero explicitly so the big-M constraint tauc <= TK*y
            # does not leave a spurious degree of freedom in the LP relaxation.
            model.tauc[k].fix(0.0)
            # Note: lam_a, lam_d, mu_a, mu_d are NOT fixed here.
            # With tauc=0 fixed, the PWL equation
            #   tauc[k] = sum(lam_d*Tbar) - sum(lam_a*Tbar) = 0
            # together with ed[k]=ea[k] (forced by pwl_no_free_charge + soc_mono)
            # uniquely determines lam_d=lam_a, mu_d=mu_a at the solver level.
            # Fixing them here would require knowing ea[k] a priori.

        # y_val == 1: leave tauc free so the solver picks the optimal amount.


def build_fixed_y_model(data: dict,
                        y_fixed: dict[int, int]) -> pyo.ConcreteModel:
    """
    Build the full-route MILP with charging locations fixed to y_fixed.

    Thin wrapper: calls MILP.build_model(data), then apply_fixed_y().
    All constraints are identical to the unconstrained full-route model;
    only the y variables (and tauc at y=0 stops) are fixed.

    Parameters
    ----------
    data    : dict from instances.make_data()
    y_fixed : dict {cs_stop: 0_or_1} — from any y-source's get_y()

    Returns
    -------
    Pyomo ConcreteModel with y[i].fixed = True for every i ∈ K.
    """
    model = build_model(data)
    apply_fixed_y(model, y_fixed, data)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — SOLVE AND REPORT
# ══════════════════════════════════════════════════════════════════════════════

def run_fixed_y(data: dict,
                y_source,
                tee:   bool = False,
                save:  bool = True,
                plot:  bool = True) -> dict:
    """
    Full pipeline: obtain y → validate → build → solve → report.

    Parameters
    ----------
    data     : dict from instances.make_data()
    y_source : object with  get_y(data) -> dict[int, int]
               (any of the classes in Part 1, or a custom implementation)
    tee      : pass solver output to stdout
    save     : write solution JSON to solutions/
    plot     : render the Gantt/schedule plot

    Returns
    -------
    dict with keys:
      feasible    bool       — True if the solver found a solution
      obj         float      — arrival time at destination (h), or inf
      sol         list       — per-stop solution dicts (global indices)
      status      str        — solver termination condition string
      y_fixed     dict       — the {cs_stop: 0_or_1} that was used
      y_source    str        — human-readable description of the y-source
      sol_name    str|None   — solution name used to save (None if save=False)
    """
    src_name = getattr(y_source, "name",
                       getattr(y_source, "__name__", repr(y_source)))

    print(f"\n{'='*65}")
    print(f"  Fixed-y MILP  |  {data['label']}")
    print(f"  y-source      : {src_name}")
    print(f"  C = {data['C']}")
    print(f"  K = {data['K']}")

    # ── Step 1: obtain and validate y_fixed ───────────────────────────────────
    y_fixed = y_source.get_y(data)
    validate_y_fixed(y_fixed, data)

    charging_stops = sorted(k for k, v in y_fixed.items() if v == 1)
    print(f"  Charging at   : {charging_stops}"
          f"  ({len(charging_stops)}/{len(data['K'])} CS stops)")

    # ── Step 2: build and solve ───────────────────────────────────────────────
    model = build_fixed_y_model(data, y_fixed)
    _, status = solve_model(model, tee=tee)

    feasible = status in ("optimal", "feasible")
    if not feasible:
        print(f"  No feasible solution (status={status}).")
        return dict(feasible=False, obj=float("inf"), sol=[],
                    status=status, y_fixed=y_fixed,
                    y_source=src_name, sol_name=None)

    # ── Step 3: extract, display, and optionally save ─────────────────────────
    obj = pyo.value(model.ta[data["N"]])
    sol = extract_solution(model, data)

    print(f"  Arrival at dest : {obj:.4f} h")
    print_schedule(sol, data)
    check_solution(sol, data)

    # Build a deterministic, filesystem-safe name for the solution.
    # Characters that are problematic in filenames are replaced with '_'.
    safe_src = (src_name
                .replace("(", "_").replace(")", "")
                .replace("/", "_").replace(" ", "_"))
    sol_name = f"{data['title']}_fixY_{safe_src}"

    if save:
        save_solution(sol, data, sol_name)

    if plot:
        plot_solution(sol, data, title=sol_name)

    return dict(feasible=True, obj=obj, sol=sol,
                status=status, y_fixed=y_fixed,
                y_source=src_name,
                sol_name=sol_name if save else None)


def compare_fixed_vs_free(data: dict,
                          y_source,
                          tee:  bool = False,
                          save: bool = True,
                          plot: bool = True) -> dict:
    """
    Solve both the unconstrained MILP and the fixed-y sub-problem, then
    print a side-by-side comparison of objectives and charging decisions.

    The unconstrained solve is the standard MILP.run_instance() pipeline.
    The fixed-y solve uses the y values found by the unconstrained solve
    (via FromSolList) UNLESS a different y_source is provided.

    Calling this with y_source=FromSolutionFile("name") is useful when the
    unconstrained solution already exists on disk and you do not want to
    re-solve it.

    Parameters
    ----------
    data     : dict from instances.make_data()
    y_source : y-source for the fixed-y solve.  Pass None to use the
               y values from the free solve performed in this call.
    tee, save, plot : forwarded to both solves.

    Returns
    -------
    dict with keys:
      free   : result dict from the unconstrained MILP
      fixed  : result dict from the fixed-y MILP
      gap_h  : obj_fixed − obj_free  (non-negative if y_fixed is optimal)
    """
    print(f"\n{'='*65}")
    print(f"  compare_fixed_vs_free  |  {data['label']}")

    # ── Unconstrained solve ───────────────────────────────────────────────────
    print(f"\n  [1/2] Unconstrained full-route MILP …")
    m_free = build_model(data)
    _, status_free = solve_model(m_free, tee=tee)
    feasible_free  = status_free in ("optimal", "feasible")

    if not feasible_free:
        print(f"  Unconstrained solve failed (status={status_free}).")
        return dict(free=dict(feasible=False, obj=float("inf")),
                    fixed=dict(feasible=False, obj=float("inf")),
                    gap_h=float("nan"))

    obj_free  = pyo.value(m_free.ta[data["N"]])
    sol_free  = extract_solution(m_free, data)
    print(f"  Free obj : {obj_free:.4f} h")

    if save:
        save_solution(sol_free, data, data["title"])
    if plot:
        plot_solution(sol_free, data, title=data["title"])

    # ── Fixed-y solve ─────────────────────────────────────────────────────────
    print(f"\n  [2/2] Fixed-y sub-problem …")
    effective_source = (y_source if y_source is not None
                        else FromSolList(sol_free, name="from_free_solve"))
    result_fixed = run_fixed_y(data, effective_source,
                               tee=tee, save=save, plot=plot)

    gap = (result_fixed["obj"] - obj_free
           if result_fixed["feasible"] else float("nan"))

    print(f"\n  {'─'*45}")
    print(f"  Free MILP obj   : {obj_free:.4f} h")
    if result_fixed["feasible"]:
        print(f"  Fixed-y obj     : {result_fixed['obj']:.4f} h")
        print(f"  Gap (fixed−free): {gap:+.4f} h  "
              f"({'optimal decomposition' if abs(gap) < 1e-4 else 'suboptimal y'})")
    else:
        print(f"  Fixed-y         : INFEASIBLE  (y source: {result_fixed['y_source']})")
    print(f"  {'─'*45}")

    return dict(
        free  = dict(feasible=True, obj=obj_free,
                     sol=sol_free, status=status_free),
        fixed = result_fixed,
        gap_h = gap,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _restore_int_keys(obj):
    """
    Recursively convert string-encoded integer keys back to int.

    runner.py serialises all dict keys as strings via json.dumps.  This
    is the inverse pass, matching the _load_oracle_cache helper in runner.py.
    Only keys whose string value is a valid integer (possibly negative) are
    converted; all other string keys are left unchanged.
    """
    if isinstance(obj, dict):
        return {
            (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k):
            _restore_int_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_restore_int_keys(v) for v in obj]
    return obj


def _y_from_sol_with_validation(sol: list,
                                data: dict,
                                source_label: str) -> dict[int, int]:
    """
    Extract y values from a solution list and cross-check against data["K"].

    CS stops present in data["K"] but absent from the solution are set to
    y=0 with a printed warning.  Stops in the solution but not in data["K"]
    are silently ignored (they belong to a different route configuration).
    """
    K_set = set(data["K"])

    sol_by_stop: dict[int, dict] = {s["i"]: s for s in sol}

    y_fixed: dict[int, int] = {}
    for k in K_set:
        s = sol_by_stop.get(k)
        if s is None:
            print(f"  [{source_label}] WARNING: CS stop {k} not found in "
                  f"solution — defaulting to y=0.")
            y_fixed[k] = 0
        else:
            y_fixed[k] = int(round(s.get("y", 0)))

    return y_fixed


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   python fixed_y_milp.py <instance_name> <solution_name> [--tee] [--no-save]
#                          [--no-plot] [--compare] [--all-charge] [--no-charge]
#
# Positional arguments:
#   instance_name  : key in instances.ALL_INSTANCES, e.g. "realistic"
#   solution_name  : stem of a JSON in solutions/, e.g. "realistic"
#                    (ignored when --all-charge or --no-charge is given)
#
# Flags:
#   --tee          : print solver output to stdout
#   --no-save      : do not write a solution JSON
#   --no-plot      : do not render the plot
#   --compare      : run compare_fixed_vs_free() (also solves the free MILP)
#   --all-charge   : use AllChargeY instead of FromSolutionFile
#   --no-charge    : use NoChargeY instead of FromSolutionFile

if __name__ == "__main__":
    from instances import ALL_INSTANCES

    args = sys.argv[1:]

    # ── Parse flags ───────────────────────────────────────────────────────────
    tee       = "--tee"        in args
    no_save   = "--no-save"    in args
    no_plot   = "--no-plot"    in args
    compare   = "--compare"    in args
    all_chrg  = "--all-charge" in args
    no_chrg   = "--no-charge"  in args

    positional = [a for a in args if not a.startswith("--")]

    if len(positional) < 1:
        print(
            "Usage: python fixed_y_milp.py <instance> [solution_stem] [flags]\n"
            "\n"
            "  <instance> can be:\n"
            "    - A key in ALL_INSTANCES, e.g.  realistic\n"
            "    - A path to an instance JSON,   e.g.  instances/RmediumCmany_1.json\n"
            "\n"
            "  [solution_stem] is the y-source; can be:\n"
            "    - A stem in solutions/ (MILP.save_solution format), e.g.  realistic\n"
            "    - An oracle cache stem, e.g.  oracle_RmediumCmany_1  (or with .json)\n"
            "      The oracle_ prefix is detected automatically.\n"
            "\n"
            f"  Named instances : {list(ALL_INSTANCES)}\n"
            "  Flags           : --tee  --no-save  --no-plot  --compare\n"
            "                    --all-charge  --no-charge"
        )
        sys.exit(1)

    inst_arg = positional[0]
    sol_arg  = positional[1] if len(positional) >= 2 else None

    # ── Load instance ─────────────────────────────────────────────────────────
    # Accept either a key from ALL_INSTANCES or a direct path to a .json file
    # produced by instance_io.py (instances/R*C*.json).
    if inst_arg.endswith(".json"):
        # File-path mode: use instance_io.load_instance_json
        try:
            from instance_io import load_instance_json
        except ImportError:
            print("ERROR: instance_io.py not found. Make sure it is in the "
                  "same directory.")
            sys.exit(1)
        if not os.path.exists(inst_arg):
            print(f"ERROR: instance file not found: '{inst_arg}'")
            sys.exit(1)
        data, _, _, _ = load_instance_json(inst_arg)
        # Default solution stem: title field from the instance
        default_sol_stem = data.get("title", os.path.splitext(
                                    os.path.basename(inst_arg))[0])
    else:
        # Named-instance mode: look up in ALL_INSTANCES
        if inst_arg not in ALL_INSTANCES:
            print(f"Unknown instance '{inst_arg}'. "
                  f"Available: {list(ALL_INSTANCES)}")
            sys.exit(1)
        data = ALL_INSTANCES[inst_arg]()
        default_sol_stem = inst_arg

    # ── Normalise solution stem ───────────────────────────────────────────────
    # Strip .json extension if the user supplied it.
    raw_stem = sol_arg if sol_arg is not None else default_sol_stem
    if raw_stem.endswith(".json"):
        raw_stem = raw_stem[:-5]

    # ── Choose y-source ───────────────────────────────────────────────────────
    if all_chrg:
        y_src = AllChargeY()
    elif no_chrg:
        y_src = NoChargeY()
    elif raw_stem.startswith("oracle_"):
        # Oracle cache format: top-level "sol" field, no "data" wrapper
        y_src = FromOracleFile(raw_stem)
    else:
        # Standard MILP.save_solution format
        y_src = FromSolutionFile(raw_stem)

    # ── Run ───────────────────────────────────────────────────────────────────
    if compare:
        compare_fixed_vs_free(data, y_src,
                              tee=tee, save=not no_save, plot=not no_plot)
    else:
        run_fixed_y(data, y_src,
                    tee=tee, save=not no_save, plot=not no_plot)