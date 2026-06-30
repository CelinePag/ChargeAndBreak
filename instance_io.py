"""
instance_io.py — Precomputed instance JSON files for reproducible comparison
=============================================================================
Each JSON file represents ONE independent instance: a unique route geometry
(route length, CS placement, customer count and locations) drawn with a fixed
seed, together with its uncertainty realisation (actual travel times and
energies per leg) and a pool of 500 forward-looking scenarios per stop.

File naming
-----------
  RshortCfew_1.json      route_class="short",  customers_class="few",  seed 1
  RshortCfew_2.json      ...                                            seed 2
  RshortCmedium_1.json
  RmediumCfew_1.json
  ...

Generating all files
--------------------
  python instance_io.py [output_dir] [n_seeds] [n_scenarios] [delta]

  Defaults: output_dir="instances"  n_seeds=50  n_scenarios=500  delta=0.20

  This produces n_seeds files per (route_class, customers_class) combo,
  i.e. 9 combos × 50 seeds = 450 files total.

Loading a file
--------------
  from instance_io import load_instance_json
  full_data, D_real, E_real, scenarios_by_stop = load_instance_json(
      "instances/RmediumCfew_7.json",
      max_scenarios=10)   # take first 10 of 500 for fast runs

JSON schema
-----------
{
  "meta": {
    "route_class":      str,
    "customers_class":  str,
    "seed":             int,
    "n_scenarios":      int,       // 500
    "delta":            float,
    "created_at":       str
  },
  "instance":   { ...make_data dict, int keys stored as strings... },
  "D_real":     [float, ...],      // N floats, one per leg
  "E_real":     [float, ...],      // N floats, one per leg
  "scenarios":  [                  // one list per stop 0..N-1
    [{"D": {"0": float, ...}, "E": {"0": float, ...}}, ...],
    ...
  ]
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
from typing import Optional

import numpy as np

from instances import instance_realistic
from scenarios import generate_scenarios, _ecr
from settings  import V_NOM, sample_travel_time


# ══════════════════════════════════════════════════════════════════════════════
# COMBO REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_COMBOS: list[tuple[str, str]] = [
    ("short",  "few"),
    ("short",  "medium"),
    ("short",  "many"),
    ("medium", "few"),
    ("medium", "medium"),
    ("medium", "many"),
    ("long",   "few"),
    ("long",   "medium"),
    ("long",   "many"),
]

_ROUTE_TAG = {"short": "Rshort", "medium": "Rmedium", "long": "Rlong"}
_CUST_TAG  = {"few":   "Cfew",   "medium": "Cmedium", "many": "Cmany"}


def instance_filename(route_class: str, customers_class: str, seed: int) -> str:
    """Return the canonical filename for one instance file."""
    return f"{_ROUTE_TAG[route_class]}{_CUST_TAG[customers_class]}_{seed}.json"


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
                           seed: int,
                           output_dir: str  = "instances",
                           n_scenarios: int = 500,
                           delta: float     = 0.20,
                           verbose: bool    = True) -> str:
    """
    Generate one precomputed instance JSON file.

    The seed controls everything:
      - random.seed(seed)  fixes the route geometry (via instance_realistic)
      - numpy RNG seeded from seed fixes the uncertainty realisation (D_real,
        E_real) and all 500 scenario draws per stop.

    Parameters
    ----------
    route_class      : "short" | "medium" | "long"
    customers_class  : "few" | "medium" | "many"
    seed             : integer seed — uniquely identifies this instance file
    output_dir       : directory where the JSON file is written
    n_scenarios      : number of scenarios per stop (500)
    delta            : travel-time uncertainty half-width (e.g. 0.20)
    verbose          : print progress

    Returns
    -------
    str -- absolute path to the written JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = instance_filename(route_class, customers_class, seed)
    filepath = os.path.join(output_dir, filename)

    # ── 1. Generate route geometry ─────────────────────────────────────────────
    random.seed(seed)
    full_data = instance_realistic(
        route_class     = route_class,
        customers_class = customers_class,
        clusters        = 3,
    )
    # Override title/label with the canonical seed-based name so that all
    # output files (logs, figures, solutions) are named after the instance.
    stem = instance_filename(route_class, customers_class, seed).replace(".json", "")
    full_data["title"] = stem
    full_data["label"] = (
        f"{stem} — {route_class} route, {customers_class} customers, seed={seed}"
    )
    N     = full_data["N"]
    D_nom = full_data["D"]
    km    = full_data.get("km", {})

    if verbose:
        print(f"  {filename}  N={N}  |C|={len(full_data['C'])}"
              f"  |K|={len(full_data['K'])}", end="  ")

    # ── 2. Draw uncertainty realisation ───────────────────────────────────────
    rng    = np.random.default_rng(seed)
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

    # ── 3. Draw scenario pool: 500 scenarios per stop ─────────────────────────
    scenarios_by_stop = []
    for stop in range(N):
        scen_seed = int(rng.integers(0, 2**31))
        scens = generate_scenarios(
            full_data     = full_data,
            start_stop    = stop,
            end_stop      = N,
            n_scenarios   = n_scenarios,
            delta         = delta,
            seed          = scen_seed,
            include_best  = False,
            include_worst = False,
        )
        scenarios_by_stop.append(_to_json_safe(scens))

    # ── 4. Write JSON ──────────────────────────────────────────────────────────
    payload = dict(
        meta = dict(
            route_class     = route_class,
            customers_class = customers_class,
            seed            = seed,
            n_scenarios     = n_scenarios,
            delta           = delta,
            created_at      = datetime.now().isoformat(timespec="seconds"),
        ),
        instance  = _to_json_safe(full_data),
        D_real    = D_real,
        E_real    = E_real,
        scenarios = scenarios_by_stop,
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
                 n_scenarios: int = 500,
                 delta: float     = 0.20,
                 first_seed: int  = 1,
                 combos           = None,
                 verbose: bool    = True) -> list[str]:
    """
    Generate all precomputed instance JSON files.

    For each (route_class, customers_class) combination and each seed in
    [first_seed, first_seed + n_seeds), one JSON file is written.

    Total files: len(combos) * n_seeds  (default: 9 * 50 = 450 files)

    Parameters
    ----------
    output_dir   : directory where files are written (created if absent)
    n_seeds      : how many independent instances per combo (default 50)
    n_scenarios  : scenarios per stop per file (default 500)
    delta        : uncertainty half-width (default 0.20)
    first_seed   : starting seed value (default 1); seeds are first_seed..first_seed+n_seeds-1
    combos       : list of (route_class, customers_class) to generate,
                   or None for all 9 combinations
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

    for rc, cc in combos:
        if verbose:
            print(f"\n  [{_ROUTE_TAG[rc]}{_CUST_TAG[cc]}]"
                  f"  seeds {first_seed}..{first_seed + n_seeds - 1}")
        for seed in range(first_seed, first_seed + n_seeds):
            p = generate_instance_file(
                route_class     = rc,
                customers_class = cc,
                seed            = seed,
                output_dir      = output_dir,
                n_scenarios     = n_scenarios,
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

def load_instance_json(filepath: str,
                       max_scenarios: Optional[int] = None
                       ) -> tuple[dict, list, list, list]:
    """
    Load a precomputed instance JSON file.

    Parameters
    ----------
    filepath      : path to a JSON file produced by generate_instance_file
    max_scenarios : if given, truncate each stop's scenario list to this many
                    (e.g. max_scenarios=10 for fast LA runs)

    Returns
    -------
    full_data         : dict  -- instance data dict with int keys restored
    D_real            : list[float] -- N realised travel times (h)
    E_real            : list[float] -- N realised energies (kWh)
    scenarios_by_stop : list[list[dict]] -- scenarios_by_stop[i] is the
                        scenario list at stop i (up to max_scenarios entries)
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

    scenarios_by_stop = []
    for stop_scens in data["scenarios"]:
        restored = [_restore_int_keys(s) for s in stop_scens]
        if max_scenarios is not None:
            restored = restored[:max_scenarios]
        scenarios_by_stop.append(restored)

    return full_data, D_real, E_real, scenarios_by_stop, delta_file


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
    """
    Return a summary of a precomputed instance file without loading scenarios.
    """
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
    # Usage: python instance_io.py [output_dir] [n_seeds] [n_scenarios] [delta] [first_seed]
    output_dir   = sys.argv[1] if len(sys.argv) > 1 else "instances"
    n_seeds      = int(sys.argv[2])   if len(sys.argv) > 2 else 2
    n_scenarios  = int(sys.argv[3])   if len(sys.argv) > 3 else 100
    delta        = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    first_seed   = int(sys.argv[5])   if len(sys.argv) > 5 else 1

    print("=" * 60)
    print("  instance_io.py — precomputing instance JSON files")
    print(f"  output_dir  = {output_dir}")
    print(f"  n_seeds     = {n_seeds}  (seeds {first_seed}..{first_seed+n_seeds-1})")
    print(f"  n_scenarios = {n_scenarios} per stop")
    print(f"  delta       = {delta:.0%}")
    print(f"  total files = {9 * n_seeds}")
    print("=" * 60)

    generate_all(
        output_dir  = output_dir,
        n_seeds     = n_seeds,
        n_scenarios = n_scenarios,
        delta       = delta,
        first_seed  = first_seed,
        verbose     = True,
    )