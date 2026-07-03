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
    tight   : half-width ~ Uniform(1h, 2h)  per customer
    medium  : half-width ~ Uniform(3h, 6h)  per customer
    large   : half-width ~ Uniform(6h, 12h) per customer

  The window is centred on the arrival time the GREEDY policy reaches that
  customer at, running with nominal (undisturbed) travel times and no time
  windows (see greedy.compute_nominal_arrivals).  Concretely:

    Wha[c] = max(T_START, t_nominal[c] - half_width)
    Whf[c] = t_nominal[c] + half_width

  Route geometry and D_real/E_real are IDENTICAL across window classes for
  the same seed -- only Wha/Whf differ -- because the half-width draws
  happen last, after every other rng-consuming step.

Generating all files
--------------------
  python instance_io.py [output_dir] [n_seeds] [delta]

  Defaults: output_dir="instances"  n_seeds=50  delta=0.20

  This produces n_seeds files per (route_class, customers_class, window_class)
  combo, i.e. 9 route/customer combos × 4 window classes × 50 seeds = 1800
  files total.

Loading a file
--------------
  from instance_io import load_instance_json
  full_data, D_real, E_real, delta = load_instance_json(
      "instances/RmediumCfewTtight_7.json")

JSON schema
-----------
{
  "meta": {
    "route_class":        str,
    "customers_class":    str,
    "window_class":       str,       // "none" | "tight" | "medium" | "large"
    "seed":               int,
    "delta":              float,
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

import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

from instances import instance_realistic
from scenarios import _ecr
from settings  import V_NOM, sample_travel_time
from greedy    import compute_nominal_arrivals


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
# "none" is unconstrained and draws no half-widths.
_WINDOW_HALF_WIDTH_RANGE: dict[str, tuple[float, float]] = {
    "tight":  (0.5, 1.0),
    "medium": (1.0, 3.0),
    "large":  (3.0, 6.0),
}


def instance_filename(route_class: str, customers_class: str,
                      window_class: str, seed: int) -> str:
    """Return the canonical filename for one instance file."""
    return (f"{_ROUTE_TAG[route_class]}{_CUST_TAG[customers_class]}"
            f"{_WINDOW_TAG[window_class]}_{seed}.json")


def generate_time_windows(full_data: dict, window_class: str,
                          rng: np.random.Generator) -> dict:
    """
    Draw per-customer time windows and write them into full_data["Wha"] /
    full_data["Whf"] (absolute hours, in place).

    The window is centred on the arrival time the greedy policy reaches that
    customer at under nominal (undisturbed) travel times and no time-window
    constraints.  A half-width is drawn independently per customer from
    Uniform(*_WINDOW_HALF_WIDTH_RANGE[window_class]) using ``rng``, so this
    must be called AFTER every other rng-consuming step (D_real, scenarios)
    for a given instance file, to keep the underlying instance identical
    across window classes for the same seed.

    Parameters
    ----------
    full_data     : dict from instances.make_data() -- Wha/Whf mutated in place
    window_class  : "tight" | "medium" | "large"  ("none" should not call this)
    rng           : np.random.Generator -- shared instance rng

    Returns
    -------
    dict {customer_stop: half_width_h} -- the draws made, for the JSON meta
    """
    lo, hi  = _WINDOW_HALF_WIDTH_RANGE[window_class]
    T_START = full_data["T_START"]

    t_nominal = compute_nominal_arrivals(full_data)

    half_widths = {}
    for c in sorted(full_data["C"]):
        hw = float(rng.uniform(lo, hi))
        half_widths[c] = hw
        t_c = t_nominal[c]
        full_data["Wha"][c] = max(T_START, t_c - hw)
        full_data["Whf"][c] = t_c + hw

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
                           output_dir: str  = "instances",
                           delta: float     = 0.20,
                           verbose: bool    = True) -> str:
    """
    Generate one precomputed instance JSON file.

    The seed controls everything:
      - random.seed(seed)  fixes the route geometry (via instance_realistic)
      - numpy RNG seeded from seed fixes CS queue-time draws (via
        instance_realistic -> make_data), the uncertainty realisation
        (D_real, E_real), and (when window_class != "none") the per-customer
        time-window half-widths.

    Because the half-width draws happen LAST (after D_real), route geometry
    and D_real/E_real are bit-identical across window classes for the same
    seed -- only Wha/Whf differ.

    No scenario pool is generated or stored here — 2SP and LA draw scenarios
    live at solve/decision time (see scenarios.generate_scenarios).

    Parameters
    ----------
    route_class      : "short" | "medium" | "long"
    customers_class  : "few" | "medium" | "many"
    window_class     : "none" | "tight" | "medium" | "large"
    seed             : integer seed — uniquely identifies this instance file
    output_dir       : directory where the JSON file is written
    delta            : travel-time uncertainty half-width (e.g. 0.20)
    verbose          : print progress

    Returns
    -------
    str -- absolute path to the written JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = instance_filename(route_class, customers_class, window_class, seed)
    filepath = os.path.join(output_dir, filename)

    # ── 1. Generate route geometry ─────────────────────────────────────────────
    random.seed(seed)
    rng = np.random.default_rng(seed)
    full_data = instance_realistic(
        route_class     = route_class,
        customers_class = customers_class,
        clusters        = 3,
        rng             = rng,
    )
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

    # ── 2. Draw uncertainty realisation ───────────────────────────────────────
    D_real = []
    E_real = []
    for leg in range(N):
        d_nom = D_nom.get(leg, 0.0)
        d_act = sample_travel_time(D_nom.get(leg, 0.0), rng, lower_pct = delta, upper_pct = delta)
        L_km  = km.get(leg, d_nom * V_NOM)
        v_act = L_km / d_act if d_act > 0 else V_NOM
        e_act = L_km * _ecr(v_act)
        D_real.append(round(d_act, 6))
        E_real.append(round(e_act, 4))

    # ── 3. Draw time windows (nominal greedy arrival ± random half-width) ─────
    half_widths = {}
    if window_class != "none":
        half_widths = generate_time_windows(full_data, window_class, rng)
        if verbose:
            print(f"  [{window_class} windows: "
                  f"{min(half_widths.values()):.1f}-{max(half_widths.values()):.1f}h half-width]",
                  end="  ")

    # ── 4. Write JSON ──────────────────────────────────────────────────────────
    payload = dict(
        meta = dict(
            route_class        = route_class,
            customers_class    = customers_class,
            window_class       = window_class,
            seed               = seed,
            delta              = delta,
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

def generate_all(output_dir: str  = "instances",
                 n_seeds: int     = 50,
                 delta: float     = 0.20,
                 first_seed: int  = 1,
                 combos           = None,
                 verbose: bool    = True) -> list[str]:
    """
    Generate all precomputed instance JSON files.

    For each (route_class, customers_class, window_class) combination and
    each seed in [first_seed, first_seed + n_seeds), one JSON file is written.
    The same seed is reused across all 4 window classes of a given
    (route_class, customers_class) pair, so those 4 files share identical
    route geometry / D_real / E_real and differ only in Wha/Whf.

    Total files: len(combos) * n_seeds  (default: 9*4 * 50 = 1800 files)

    Parameters
    ----------
    output_dir   : directory where files are written (created if absent)
    n_seeds      : how many independent instances per combo (default 50)
    delta        : uncertainty half-width (default 0.20)
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
                delta           = delta,
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
    delta_file : float -- travel-time uncertainty half-width used to draw D_real
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    full_data = _restore_int_keys(data["instance"])
    D_real    = data["D_real"]
    E_real    = data["E_real"]
    delta_file = data["meta"].get("delta", 0.20)  # back-fill delta if missing from meta

    # Back-fill M_stop / M_seq for JSON files generated before model_v5.
    # Use the legacy M dict as the stop-overhead value; default M_seq to 5 min.
    if "M_stop" not in full_data:
        M_legacy = full_data.get("M", {})
        K_set    = set(full_data.get("K", []))
        full_data["M_stop"] = {k: M_legacy.get(k, 5.0 / 60) for k in K_set}
    if "M_seq" not in full_data:
        K_set = set(full_data.get("K", []))
        full_data["M_seq"] = {k: 5.0 / 60 for k in K_set}

    return full_data, D_real, E_real, delta_file


def list_available(output_dir: str = "instances") -> list[str]:
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
    # Usage: python instance_io.py [output_dir] [n_seeds] [delta] [first_seed]
    output_dir   = sys.argv[1] if len(sys.argv) > 1 else "instances"
    n_seeds      = int(sys.argv[2])   if len(sys.argv) > 2 else 25
    delta        = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    first_seed   = int(sys.argv[4])   if len(sys.argv) > 4 else 1

    print("=" * 60)
    print("  instance_io.py — precomputing instance JSON files")
    print(f"  output_dir  = {output_dir}")
    print(f"  n_seeds     = {n_seeds}  (seeds {first_seed}..{first_seed+n_seeds-1})")
    print(f"  delta       = {delta:.0%}")
    print(f"  total files = {len(_COMBOS) * n_seeds}")
    print("=" * 60)

    generate_all(
        output_dir  = output_dir,
        n_seeds     = n_seeds,
        delta       = delta,
        first_seed  = first_seed,
        verbose     = True,
    )