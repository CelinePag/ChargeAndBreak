"""
instance_io.py — Precomputed instance JSON files for reproducible comparison
=============================================================================
Each JSON file represents ONE independent instance: a unique route geometry
(route length, CS placement, customer count and locations) drawn with a fixed
seed, together with its uncertainty realisation (actual travel times and
energies per leg).

Scenario pools are NOT precomputed or stored here.  2SP and LA (the two
algorithms that need forward-looking scenarios) each draw them live via
scenarios.generate_scenarios() at solve/decision time — see 2SP.run_2sp and
Simulation.run_simulation_precomputed / select_best_action.

File naming
-----------
  RshortCfewTnone_1.json    route_class="short", customers_class="few",
                            window_class="none" (unconstrained), seed 1
  RshortCfewTtight_1.json   ...                  window_class="tight"
  RlongCfewTmedium_3.json
  RlongCmanyTlarge_7.json
  ...

Time windows
------------
  window_class ∈ {"none", "tight", "medium", "large"} controls the width of
  each customer's arrival-time window [Wha, Whf]:

    none    : unconstrained (Wha=T_START, Whf=T_START+2e7) -- today's behaviour
    tight   : half-width ~ Uniform(0.5h, 1.0h) per customer
    medium  : half-width ~ Uniform(1.0h, 3.0h) per customer
    large   : half-width ~ Uniform(3.0h, 6.0h) per customer

  Each customer on a route draws its OWN half-width independently, so
  customers on the same route generally have DIFFERENT window widths.  The
  window is centred on the nominal service start from the deterministic MILP
  solve of the uncapacitated-window instance (greedy warm start, greedy
  fallback on timeout — see _nominal_milp_arrivals).  Concretely, with a small
  ±10% centre jitter j:

    Wha[c] = max(T_START, t_nominal[c] + j - half_width_c)
    Whf[c] = t_nominal[c] + j + half_width_c

  Each window class is an INDEPENDENT random instance: window_class is folded
  into the geometry seed (see _geometry_seed), so the four window classes of a
  given seed have DIFFERENT routes, customer placements, and D_real/E_real --
  not merely different Wha/Whf.

Generating all files
--------------------
  python -m src.instance_gen.instance_io [output_dir] [n_seeds] [cv]

  Defaults: output_dir="instances"  n_seeds=25  cv=0.15

  This produces n_seeds files per (route_class, customers_class, window_class)
  combo, i.e. 9 route/customer combos × 4 window classes × 25 seeds = 900
  files total.

Loading a file
--------------
  from src.instance_gen.instance_io import load_instance_json
  full_data, D_real, E_real, cv = load_instance_json(
      "instances/RmediumCfewTtight_7.json")

JSON schema
-----------
{
  "meta": {
    "route_class":        str,
    "customers_class":    str,
    "window_class":       str,       // "none" | "tight" | "medium" | "large"
    "seed":               int,
    "cv":                 float,     // CV of the shifted-lognormal multiplier
    "xi_min":             float,     // hard support bounds of the multiplier
    "xi_max":             float,
    "created_at":         str,
    "window_half_widths": {str: float}   // per-customer half-width (h), omitted for "none"
  },
  "instance":   { ...make_data dict, int keys stored as strings... },
  "D_real":     [float, ...],      // N floats, one per leg
  "E_real":     [float, ...],      // N floats, one per leg
}

All originally-integer dict keys are stored as strings (JSON requirement).
load_instance_json() restores them to integers on load.

Import chain
------------
  instance_io.py -> instances, scenarios
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

from src.instance_gen.instances import instance_realistic
from src.simulation.scenarios import _ecr
from src.settings  import (V_NOM, sample_multipliers,
                       XI_MIN, XI_MAX,
                       TRAVEL_TIME_CV, TRAVEL_TIME_AR1_RHO)
from src.methods.greedy    import compute_nominal_arrivals
from src import paths as _paths


# ══════════════════════════════════════════════════════════════════════════════
# COMBO REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_ROUTE_CLASSES = ["short", "medium", "long"]
_CUST_CLASSES  = ["few", "medium", "many"]
_WINDOW_CLASSES = ["none", "tight", "medium", "large"]

_COMBOS: list[tuple[str, str, str]] = [
    (rc, cc, wc)
    for rc in _ROUTE_CLASSES
    for cc in _CUST_CLASSES
    for wc in _WINDOW_CLASSES
]

_ROUTE_TAG  = {"short": "Rshort", "medium": "Rmedium", "long": "Rlong"}
_CUST_TAG   = {"few":   "Cfew",   "medium": "Cmedium", "many": "Cmany"}
_WINDOW_TAG = {"none": "Tnone", "tight": "Ttight", "medium": "Tmedium", "large": "Tlarge"}

# Per-customer half-width sampling range (hours) for each window class.
# Every customer on a route draws its OWN half-width independently, uniformly
# from the class range — so customers on the same route have different window
# widths.  "none" is unconstrained and draws no half-widths.
_WINDOW_HALF_WIDTH_RANGE: dict[str, tuple[float, float]] = {
    "tight":  (0.5, 1.0),
    "medium": (1.0, 3.0),
    "large":  (3.0, 6.0),
}

# Destination deadline (R6) slack relative to the nominal arrival
_DEADLINE_KAPPA = 0.20
_DEADLINE_DMIN  = 2.0


def instance_filename(route_class: str, customers_class: str,
                      window_class: str, seed: int) -> str:
    """Return the canonical filename for one instance file."""
    return (f"{_ROUTE_TAG[route_class]}{_CUST_TAG[customers_class]}"
            f"{_WINDOW_TAG[window_class]}_{seed}.json")


def _geometry_seed(route_class: str, customers_class: str,
                   window_class: str, seed: int, attempt: int) -> int:
    """Deterministic geometry seed for one instance, folding in ALL axes.

    Every (route_class, customers_class, window_class, seed) is an INDEPENDENT
    random instance — the window class is mixed into the seed so the four
    window classes of a given seed draw DIFFERENT routes and customer
    placements (not just different windows).  ``attempt`` advances the seed on
    M9-rejection regeneration.  SHA-256-based so the axes vary independently
    with no structured collisions; the filename still encodes the requested
    ``seed``, and the derived value is recorded in the JSON meta for
    traceability.  Returns a 32-bit int accepted by both random.seed and
    np.random.default_rng.
    """
    key = f"{route_class}|{customers_class}|{window_class}|{seed}|{attempt}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def _nominal_milp_arrivals(full_data: dict,
                           solver_time_limit: int = 120,
                           mip_gap: float = 0.05) -> tuple:
    """
    I1 — Solve the DETERMINISTIC MILP of the uncapacitated-window instance
    (nominal travel times, unconstrained windows) and return the per-stop
    nominal arrival times used to centre the customer time windows.

    Warm-started by the greedy nominal pass; on timeout / infeasibility the
    greedy schedule itself is used and flagged.

    The extracted arrival times (``model.ta``) come from the INCUMBENT, which
    on these instances stabilises within a few seconds — nearly all remaining
    solve time is spent closing the (loose) lower bound to prove optimality,
    which does not change ``ta`` at all.  Since the arrival times only set
    window CENTRES (which then get a ±10% jitter, see generate_time_windows),
    a loose gap (default 5%) plus a short time limit gives essentially the
    same centres as a 1% gap while avoiding the bound-proving cost that made
    medium/long routes take up to the full time limit per instance.

    Returns
    -------
    (t_nominal, source) — list of absolute arrival hours per stop 0..N and
    the string "milp" or "greedy" identifying which schedule produced it.
    """
    import contextlib as _ctx
    import io as _io

    import pyomo.environ as pyo
    from src.methods.MILP import build_model

    t_greedy = compute_nominal_arrivals(full_data)

    try:
        model  = build_model(full_data)
        solver = pyo.SolverFactory("gurobi")
        solver.options["MIPGap"]    = mip_gap
        solver.options["TimeLimit"] = solver_time_limit
        _sink = _io.StringIO()
        with _ctx.redirect_stdout(_sink), _ctx.redirect_stderr(_sink):
            res = solver.solve(model, tee=False)
        status = str(res.solver.termination_condition)
        if status in ("optimal", "feasible", "maxTimeLimit"):
            obj = pyo.value(model.obj)
            if obj is not None and obj < 1e8:
                t_nom = [float(pyo.value(model.ta[i]))
                         for i in full_data["I"]]
                return t_nom, "milp"
    except Exception:
        pass
    return list(t_greedy), "greedy"


def generate_time_windows(full_data: dict, window_class: str,
                          rng: np.random.Generator,
                          solver_time_limit: int = 120,
                          mip_gap: float = 0.05) -> dict:
    """
    I1 — Generate per-customer time windows and the destination deadline,
    writing them into full_data["Wha"] / full_data["Whf"] / full_data["T_dead"]
    (absolute hours, in place).

    Windows are centred on the NOMINAL SERVICE START from the deterministic
    MILP solution of the uncapacitated-window instance (greedy warm start,
    time limit; greedy fallback flagged via the returned meta).  Each customer
    draws its OWN half-width independently,

        Delta_c ~ Uniform(lo, hi),   (lo, hi) = _WINDOW_HALF_WIDTH_RANGE[class]

    so customers on the same route generally have DIFFERENT window widths.

    ``rng`` is consumed twice per customer (the half-width draw, then a small
    jitter on the window CENTRE, uniform ±10% of that half-width) so the draw
    order contract with generate_instance_file is preserved.

    Returns
    -------
    dict {customer_stop: half_width_h} plus keys "_source" ("milp"|"greedy")
    and "_deadline" — for the JSON meta.
    """
    T_START  = full_data["T_START"]
    lo, hi   = _WINDOW_HALF_WIDTH_RANGE[window_class]

    t_nominal, source = _nominal_milp_arrivals(
        full_data, solver_time_limit=solver_time_limit, mip_gap=mip_gap)

    half_widths = {}
    for c in sorted(full_data["C"]):
        t_c = t_nominal[c]
        hw  = float(rng.uniform(lo, hi))  # each customer draws its own width
        jit = float(rng.uniform(-0.1, 0.1)) * hw
        half_widths[c] = hw
        full_data["Wha"][c] = max(T_START, t_c + jit - hw)
        full_data["Whf"][c] = t_c + jit + hw

    # Destination deadline (R6)
    t_N = t_nominal[full_data["N"]]
    full_data["T_dead"] = t_N + max(_DEADLINE_DMIN,
                                    _DEADLINE_KAPPA * (t_N - T_START))

    half_widths["_source"]   = source
    half_widths["_deadline"] = full_data["T_dead"]
    return half_widths


# ══════════════════════════════════════════════════════════════════════════════
# SERIALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_json_safe(obj):
    """Recursively make obj JSON-serialisable (int keys -> str, numpy -> Python)."""
    if isinstance(obj, (bool, str, type(None))):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if hasattr(obj, "item"):      # numpy scalar
        return obj.item()
    return str(obj)


def _restore_int_keys(obj):
    """Recursively restore numeric string keys to int after JSON load."""
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            new_k = int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k
            new[new_k] = _restore_int_keys(v)
        return new
    if isinstance(obj, list):
        return [_restore_int_keys(v) for v in obj]
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE FILE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_instance_file(route_class: str,
                           customers_class: str,
                           window_class: str,
                           seed: int,
                           output_dir: str  = _paths.instances(),
                           cv: float        = TRAVEL_TIME_CV,
                           verbose: bool    = True,
                           ar1_rho: float   = TRAVEL_TIME_AR1_RHO,
                           cs_spacing_km: float | None = None,
                           charger_power_kw: float | None = None,
                           add_laybys: bool = True,
                           max_attempts: int = 5) -> str:
    """
    Generate one precomputed instance JSON file.

    The (route_class, customers_class, window_class, seed) tuple controls
    everything via a single derived geometry seed (see _geometry_seed):
      - random.seed(geo_seed)  fixes the route geometry (via instance_realistic)
      - numpy RNG seeded from geo_seed fixes CS queue-time draws (via
        instance_realistic -> make_data), the uncertainty realisation
        (D_real, E_real), and (when window_class != "none") the per-customer
        time-window half-widths.

    Because window_class is folded into geo_seed, each window class is an
    INDEPENDENT random instance: the four window classes of a given seed have
    DIFFERENT routes, customer placements, and D_real / E_real — not merely
    different Wha/Whf.

    No scenario pool is generated or stored here — 2SP and LA draw scenarios
    live at solve/decision time (see scenarios.generate_scenarios).

    Parameters
    ----------
    route_class      : "short" | "medium" | "long"
    customers_class  : "few" | "medium" | "many"
    window_class     : "none" | "tight" | "medium" | "large"
    seed             : integer seed — uniquely identifies this instance file
    output_dir       : directory where the JSON file is written
    cv               : CV of the travel-time multiplier (e.g. 0.15)
    verbose          : print progress

    Returns
    -------
    str -- absolute path to the written JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = instance_filename(route_class, customers_class, window_class, seed)
    filepath = os.path.join(output_dir, filename)

    # ── 1. Generate route geometry ─────────────────────────────────────────────
    # I1: instances whose deterministic model is infeasible under nominal data
    # (e.g. the weekly 56 h driving guard, M9) are discarded and regenerated
    # with an advanced attempt seed; the filename keeps the REQUESTED seed.
    full_data = None
    geo_seed  = None
    for attempt in range(max_attempts):
        geo_seed = _geometry_seed(route_class, customers_class,
                                  window_class, seed, attempt)
        random.seed(geo_seed)
        rng = np.random.default_rng(geo_seed)
        try:
            full_data = instance_realistic(
                route_class      = route_class,
                customers_class  = customers_class,
                clusters         = 3,
                rng              = rng,
                cs_spacing_km    = cs_spacing_km,
                charger_power_kw = charger_power_kw,
                add_laybys       = add_laybys,
            )
            break
        except AssertionError as e:
            if verbose:
                print(f"  {filename}: geometry rejected "
                      f"(attempt {attempt+1}: {e}); regenerating")
            full_data = None
    if full_data is None:
        raise RuntimeError(
            f"could not generate a feasible instance for {filename} "
            f"after {max_attempts} attempts")

    # Override title/label with the canonical seed-based name so that all
    # output files (logs, figures, solutions) are named after the instance.
    stem = instance_filename(route_class, customers_class, window_class, seed).replace(".json", "")
    full_data["title"] = stem
    full_data["label"] = (
        f"{stem} — {route_class} route, {customers_class} customers, "
        f"{window_class} windows, seed={seed}"
    )
    N     = full_data["N"]
    D_nom = full_data["D"]
    km    = full_data.get("km", {})

    if verbose:
        print(f"  {filename}  N={N}  |C|={len(full_data['C'])}"
              f"  |K|={len(full_data['K'])}", end="  ")

    # ── 2. Draw uncertainty realisation (S5: distribution + correlation) ──────
    mults  = sample_multipliers(N, rng, cv=cv, ar1_rho=ar1_rho)
    D_real = []
    E_real = []
    for leg in range(N):
        d_nom = D_nom.get(leg, 0.0)
        d_act = d_nom * float(mults[leg])
        L_km  = km.get(leg, d_nom * V_NOM)
        v_act = L_km / d_act if d_act > 0 else V_NOM
        e_act = L_km * _ecr(v_act)
        D_real.append(round(d_act, 6))
        E_real.append(round(e_act, 4))

    # ── 3. Time windows (I1: MILP nominal service starts, exposure-scaled) ────
    half_widths = {}
    if window_class != "none":
        half_widths = generate_time_windows(full_data, window_class, rng)
        if verbose:
            _hw = [v for k, v in half_widths.items() if not str(k).startswith("_")]
            print(f"  [{window_class} windows ({half_widths.get('_source')}): "
                  f"{min(_hw):.1f}-{max(_hw):.1f}h half-width]",
                  end="  ")

    # ── 4. Write JSON ──────────────────────────────────────────────────────────
    payload = dict(
        meta = dict(
            route_class        = route_class,
            customers_class    = customers_class,
            window_class       = window_class,
            seed               = seed,
            geometry_seed      = geo_seed,
            cv                 = cv,
            dist               = "shifted-lognormal",
            xi_min             = XI_MIN,
            xi_max             = XI_MAX,
            ar1_rho            = ar1_rho,
            cs_spacing_km      = cs_spacing_km,
            charger_power_kw   = charger_power_kw,
            add_laybys         = add_laybys,
            created_at         = datetime.now().isoformat(timespec="seconds"),
            window_half_widths = _to_json_safe(half_widths),
        ),
        instance  = _to_json_safe(full_data),
        D_real    = D_real,
        E_real    = E_real,
    )

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    size_mb = os.path.getsize(filepath) / 1e6
    if verbose:
        print(f"-> {size_mb:.1f} MB")

    return os.path.abspath(filepath)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_all(output_dir: str  = _paths.instances(),
                 n_seeds: int     = 25,
                 cv: float        = TRAVEL_TIME_CV,
                 first_seed: int  = 1,
                 combos           = None,
                 verbose: bool    = True) -> list[str]:
    """
    Generate all precomputed instance JSON files.

    For each (route_class, customers_class, window_class) combination and
    each seed in [first_seed, first_seed + n_seeds), one JSON file is written.
    Each (route_class, customers_class, window_class) combination is an
    INDEPENDENT random instance: the window class is folded into the geometry
    seed (see _geometry_seed), so the four window classes of a given seed have
    DIFFERENT routes, customer placements, and D_real / E_real — not just
    different Wha/Whf.

    Total files: len(combos) * n_seeds  (default: 9*4 * 25 = 900 files)

    Parameters
    ----------
    output_dir   : directory where files are written (created if absent)
    n_seeds      : how many independent instances per combo (default 25)
    cv           : CV of the travel-time multiplier (default settings.TRAVEL_TIME_CV)
    first_seed   : starting seed value (default 1); seeds are first_seed..first_seed+n_seeds-1
    combos       : list of (route_class, customers_class, window_class) to
                   generate, or None for all 36 combinations
    verbose      : print per-file progress

    Returns
    -------
    list[str] -- absolute paths of all written files
    """
    if combos is None:
        combos = _COMBOS

    paths = []
    t0    = time.perf_counter()
    total = len(combos) * n_seeds
    done  = 0

    for rc, cc, wc in combos:
        if verbose:
            print(f"\n  [{_ROUTE_TAG[rc]}{_CUST_TAG[cc]}{_WINDOW_TAG[wc]}]"
                  f"  seeds {first_seed}..{first_seed + n_seeds - 1}")
        for seed in range(first_seed, first_seed + n_seeds):
            p = generate_instance_file(
                route_class     = rc,
                customers_class = cc,
                window_class    = wc,
                seed            = seed,
                output_dir      = output_dir,
                cv              = cv,
                verbose         = verbose,
            )
            paths.append(p)
            done += 1
            if verbose:
                elapsed = time.perf_counter() - t0
                remaining = elapsed / done * (total - done)
                print(f"    [{done}/{total}]  {elapsed:.0f}s elapsed"
                      f"  ~{remaining:.0f}s remaining", end="\r")

    if verbose:
        print(f"\n\n  Done.  {len(paths)} files written to '{output_dir}/'")
        print(f"  Total time: {time.perf_counter() - t0:.1f}s")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_instance_json(filepath: str) -> tuple[dict, list, list, float]:
    """
    Load a precomputed instance JSON file.

    Parameters
    ----------
    filepath      : path to a JSON file produced by generate_instance_file

    Returns
    -------
    full_data  : dict  -- instance data dict with int keys restored
    D_real     : list[float] -- N realised travel times (h)
    E_real     : list[float] -- N realised energies (kWh)
    cv_file    : float -- CV of the travel-time multiplier used to draw D_real
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    full_data = _restore_int_keys(data["instance"])
    D_real    = data["D_real"]
    E_real    = data["E_real"]
    cv_file   = data["meta"].get("cv", TRAVEL_TIME_CV)

    # Back-fill M_stop / M_seq for JSON files generated before model_v5.
    # Use the legacy M dict as the stop-overhead value; default M_seq to 5 min.
    if "M_stop" not in full_data:
        M_legacy = full_data.get("M", {})
        K_set    = set(full_data.get("K", []))
        full_data["M_stop"] = {k: M_legacy.get(k, 5.0 / 60) for k in K_set}
    if "M_seq" not in full_data:
        K_set = set(full_data.get("K", []))
        full_data["M_seq"] = {k: 5.0 / 60 for k in K_set}

    # Back-fill keys introduced by the July-2026 model revision (M2–M9) so
    # that JSON files generated before it still load and solve.
    _defaults = dict(
        L=[], M_lay={},
        T_dead=None, hard_tw=False, beta=2.0, allow_split=True,
        Tspr1=13.0, Tspr2=15.0, Twk60=60.0,
        rho_bar=3, ext_bar=2,
        Tdrv_sh2=10.0,
    )
    for k, v in _defaults.items():
        full_data.setdefault(k, v)
    # Big-Ms must cover the extended limits (M5/M6)
    full_data["M_sd"] = max(full_data.get("M_sd", 0.0), full_data["Tdrv_sh2"])
    full_data["M_sw"] = max(full_data.get("M_sw", 0.0), full_data["Tspr2"])
    full_data.setdefault("M_h", full_data["Tspr2"])

    # C1 — backfill the horizon big-M H for JSON files generated before it.
    if "H" not in full_data:
        from src.instance_gen.instances import compute_horizon_bigM
        N = full_data["N"]
        full_data["H"] = compute_horizon_bigM(
            N, full_data["D"], full_data.get("S", {}),
            full_data.get("Q", {}), full_data.get("M_stop", {}),
            full_data.get("Tr1", 11.0))

    return full_data, D_real, E_real, cv_file


def list_available(output_dir: str = _paths.instances()) -> list[str]:
    """Return sorted list of .json files in output_dir."""
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".json")
    )


def describe_file(filepath: str) -> dict:
    """Return a summary of a precomputed instance file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = dict(data.get("meta", {}))
    inst = data.get("instance", {})
    meta["N"]    = inst.get("N", "?")
    meta["|K|"]  = len(inst.get("K", []))
    meta["|C|"]  = len(inst.get("C", []))
    meta["size_mb"] = round(os.path.getsize(filepath) / 1e6, 1)
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Usage: python -m src.instance_gen.instance_io [output_dir] [n_seeds] [cv] [first_seed]
    output_dir   = sys.argv[1] if len(sys.argv) > 1 else _paths.instances()
    n_seeds      = int(sys.argv[2])   if len(sys.argv) > 2 else 25
    cv           = float(sys.argv[3]) if len(sys.argv) > 3 else TRAVEL_TIME_CV
    first_seed   = int(sys.argv[4])   if len(sys.argv) > 4 else 1

    print("=" * 60)
    print("  instance_io.py — precomputing instance JSON files")
    print(f"  output_dir  = {output_dir}")
    print(f"  n_seeds     = {n_seeds}  (seeds {first_seed}..{first_seed+n_seeds-1})")
    print(f"  cv          = {cv:.2f}  (xi in [{XI_MIN:.3f}, {XI_MAX:.3f}])")
    print(f"  total files = {len(_COMBOS) * n_seeds}")
    print("=" * 60)

    generate_all(
        output_dir  = output_dir,
        n_seeds     = n_seeds,
        cv          = cv,
        first_seed  = first_seed,
        verbose     = True,
    )