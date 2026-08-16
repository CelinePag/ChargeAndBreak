"""
usecase_instance.py — real-life "usecase" instances from the trip-report data
=============================================================================
Builds instance JSON files in exactly the schema of instance_io.py, so every
method runs on them unchanged:

    python -m src.simulation.runner_dispatch "instances_usecase/usecase_*.json" all

What is REAL here and what is SYNTHETIC
---------------------------------------
REAL (taken from the telematics export + OpenStreetMap overlay):
    * the route geometry — total length and the km-position of every stop
    * which stops are customers, and their service times (observed dwells)
    * the charging-station set K (heavy-vehicle accessible sites) and the
      per-site manoeuvre time (detour in excess of a normal stop)
    * the layby set L (rest areas, aggregated)

SYNTHETIC (drawn exactly as for the RxxxCxxxTxxx instances, NOT from the
trip data — this is the deliberate choice for this first pass):
    * nominal leg travel times      D[leg] = km[leg] / V_NOM
    * nominal leg energies          E[leg] = km[leg] * ecr(V_NOM)
    * the travel-time realisation   D_real = D * xi,  xi ~ shifted lognormal
      (settings.sample_multipliers, cv = TRAVEL_TIME_CV_TARGET), and
      E_real recomputed at the realised speed
    * queue times Q at charging stations (lognormal, as in make_data)

Geometry is IDENTICAL across all generated seeds; only the travel-time
realisation changes.  This is enforced by using two separate generators: a
FIXED geometry seed for everything structural (queue draws), and the
per-instance seed only for the xi draw.  Five seeds therefore give five
realisations of one and the same instance.

Windows: none (the delivery appointments are not in the export), so these are
"Tnone" instances — Wha = T_START, Whf = T_START + 2e7.

Charging: a single PWL curve at the BASE rated power and the BASE pack, as in
the synthetic study.  The per-site rated powers recorded in the CSV (150 to
1400 kW) are carried into the JSON meta for reference but are NOT used: the
data dict holds one global Tbar.

FERRY — NOT YET ENFORCED.  The two sea crossings appear as layby nodes at
their true km-positions (the crossing adds no distance).  Forcing a break of
the crossing duration needs the model change described in the case-study
notes (fix x_b45 = 1 and taub = T_cross); until that lands, the crossing
duration is recorded in meta["ferry_nodes"] and costs the truck nothing.

Usage
-----
    python -m src.instance_gen.usecase_instance                # seeds 1..5
    python -m src.instance_gen.usecase_instance --seeds 10     # seeds 1..10
    python -m src.instance_gen.usecase_instance --out-dir instances_usecase
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np

from src.instance_gen.instances import make_data
from src.instance_gen.instance_io import _to_json_safe
from src.settings import (V_NOM, ecr, XI_MIN, XI_MAX, BATTERY_CAPACITY,
                          CHARGER_POWER_BASE_KW, TRAVEL_TIME_CV_TARGET,
                          TRAVEL_TIME_AR1_RHO, sample_multipliers, M_LAYBY_H)
from src.simulation.scenarios import _ecr
from src import paths as _paths

# ── inputs produced by route_from_data.py / osm_overlay.py ───────────────────
STOPS_CSV  = _paths.data_output("real_route_stops.csv")
CS_CSV     = _paths.data_output("real_route_cs_hgv.csv")
RA_CSV     = _paths.data_output("real_route_restareas.csv")

OUT_DIR_DEFAULT = os.path.join(_paths.ROOT, "instances_usecase")

LAYBY_MERGE_KM = 5.0    # rest areas closer than this share one node
MIN_SEP_KM     = 0.5    # nodes closer than this are merged (no degenerate legs)
GEOMETRY_SEED  = 20250922   # fixed: geometry must not vary with the seed

# node priority when two nodes fall within MIN_SEP_KM of each other
_PRIORITY = {"customer": 3, "ferry": 2, "charger": 2, "layby": 1}


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def build_geometry(verbose: bool = True) -> dict:
    """Assemble the node list of the real route from the three CSV files.

    Returns a dict with the total length and the ordered node list, each node
    carrying its km-position, its type, and the type-specific attributes
    (service time, rated power, manoeuvre time, crossing duration).
    """
    for p in (STOPS_CSV, CS_CSV, RA_CSV):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"{p} not found — regenerate it first with\n"
                "  python -m src.instance_gen.route_from_data\n"
                "  python -m src.instance_gen.osm_overlay")

    stops    = _read(STOPS_CSV)
    total_km = max(float(r["km"]) for r in stops)

    nodes = []
    for r in stops:
        km = float(r["km"])
        if r["type"] == "customer":
            nodes.append(dict(km=km, kind="customer",
                              service_h=float(r["dwell_h"] or 0.0)))
        elif r["type"] == "ferry":
            nodes.append(dict(km=km, kind="ferry",
                              crossing_h=float(r["ferry_dwell_h"] or 0.0)))

    for r in _read(CS_CSV):
        nodes.append(dict(km=float(r["km"]), kind="charger",
                          power_kw=float(r["kw"] or CHARGER_POWER_BASE_KW),
                          m_man_h=float(r["m_man_h"]),
                          off_route_km=float(r["dist_km"])))

    # rest areas: aggregate clusters closer than LAYBY_MERGE_KM
    ra_km = sorted(float(r["km"]) for r in _read(RA_CSV))
    cluster = []
    for x in ra_km:
        if cluster and x - cluster[-1] >= LAYBY_MERGE_KM:
            nodes.append(dict(km=cluster[0], kind="layby"))
            cluster = []
        cluster.append(x)
    if cluster:
        nodes.append(dict(km=cluster[0], kind="layby"))

    # order, then drop lower-priority nodes that sit on top of another node
    nodes.sort(key=lambda n: (n["km"], -_PRIORITY[n["kind"]]))
    kept, dropped = [], 0
    for n in nodes:
        if kept and n["km"] - kept[-1]["km"] < MIN_SEP_KM:
            if _PRIORITY[n["kind"]] > _PRIORITY[kept[-1]["kind"]]:
                kept[-1] = n          # higher priority wins the position
            dropped += 1
            continue
        kept.append(n)

    # the depot occupies km 0 and km total_km; drop anything colliding with it
    kept = [n for n in kept
            if MIN_SEP_KM <= n["km"] <= total_km - MIN_SEP_KM]

    if verbose:
        counts = {k: sum(1 for n in kept if n["kind"] == k)
                  for k in ("customer", "charger", "layby", "ferry")}
        print(f"geometry: {total_km:.0f} km, {len(kept) + 2} stops "
              f"({counts['customer']} customers, {counts['charger']} CS, "
              f"{counts['layby']} laybys, {counts['ferry']} ferry); "
              f"{dropped} nodes merged at < {MIN_SEP_KM} km")

    return dict(total_km=total_km, nodes=kept)


def build_data(geom: dict, rng_geo: np.random.Generator, title: str,
               battery_kwh: float = BATTERY_CAPACITY,
               charger_power_kw: float = CHARGER_POWER_BASE_KW) -> dict:
    """Turn the node list into the canonical make_data dict."""
    nodes    = geom["nodes"]
    total_km = geom["total_km"]

    I = list(range(len(nodes) + 2))
    N = len(nodes) + 1
    C, K, L = [], [], []
    D, E, km_leg, S = {}, {}, {}, {}

    prev_km = 0.0
    for idx, n in enumerate(nodes, start=1):
        leg = idx - 1
        d_km = n["km"] - prev_km
        km_leg[leg] = d_km
        D[leg] = d_km / V_NOM
        E[leg] = d_km * ecr(V_NOM)
        prev_km = n["km"]
        if n["kind"] == "customer":
            C.append(idx)
            S[idx] = n["service_h"]
        elif n["kind"] == "charger":
            K.append(idx)
        else:                       # layby and (for now) ferry
            L.append(idx)
    # final leg into the destination depot
    km_leg[N - 1] = total_km - prev_km
    D[N - 1] = km_leg[N - 1] / V_NOM
    E[N - 1] = km_leg[N - 1] * ecr(V_NOM)

    data = make_data(
        I=I, C=C, K=K, L=L, D=D, E=E, km=km_leg, S=S,
        Wha={c: 0.0 for c in C},          # Tnone: no windows
        Whf={c: 2e7 for c in C},
        label=(f"usecase — real long-haul tour, {total_km:.0f} km, "
               f"{len(C)} customers, {len(K)} CS, {len(L)} laybys "
               f"(synthetic travel-time realisation)"),
        title=title,
        Bcap=battery_kwh,
        charger_power_kw=charger_power_kw,
        rng=rng_geo,
    )

    # per-site manoeuvre time (detour in excess of a normal stop).  M_stop is a
    # per-stop dict in the model, so the real heterogeneity survives even
    # though the charging curve does not.
    for idx, n in zip(range(1, len(nodes) + 1), nodes):
        if n["kind"] == "charger":
            data["M_stop"][idx] = n["m_man_h"]

    # Ferry nodes: {stop: crossing duration (h)}.  A layby at which a break of
    # exactly this duration is FORCED (x_b45 = 1, taub = crossing).  M_lay at
    # the node carries the boarding/disembarking overhead.
    data["ferry"] = {idx: float(n["crossing_h"])
                     for idx, n in enumerate(nodes, start=1)
                     if n["kind"] == "ferry"}
    for idx, n in enumerate(nodes, start=1):
        if n["kind"] == "ferry":
            data["M_lay"][idx] = n.get("boarding_h", M_LAYBY_H)

    return data


# ══════════════════════════════════════════════════════════════════════════════
# INSTANCE FILES
# ══════════════════════════════════════════════════════════════════════════════

def generate_usecase_file(seed: int, geom: dict, out_dir: str,
                          cv: float = TRAVEL_TIME_CV_TARGET,
                          ar1_rho: float = TRAVEL_TIME_AR1_RHO,
                          battery_kwh: float = BATTERY_CAPACITY,
                          charger_power_kw: float = CHARGER_POWER_BASE_KW,
                          verbose: bool = True) -> str:
    """Write instances_usecase/usecase_<seed>.json.

    The geometry (and the queue draws) come from GEOMETRY_SEED and are the
    same in every file; only the xi realisation depends on `seed`.
    """
    os.makedirs(out_dir, exist_ok=True)
    title    = f"usecase_{seed}"
    filepath = os.path.join(out_dir, f"{title}.json")

    data = build_data(geom, np.random.default_rng(GEOMETRY_SEED), title,
                      battery_kwh=battery_kwh,
                      charger_power_kw=charger_power_kw)

    # ── travel-time realisation: identical machinery to instance_io ──────────
    N      = data["N"]
    rng    = np.random.default_rng(seed)
    mults  = sample_multipliers(N, rng, cv=cv, ar1_rho=ar1_rho)
    D_real, E_real = [], []
    for leg in range(N):
        d_nom = data["D"].get(leg, 0.0)
        d_act = d_nom * float(mults[leg])
        L_km  = data["km"].get(leg, d_nom * V_NOM)
        v_act = L_km / d_act if d_act > 0 else V_NOM
        D_real.append(round(d_act, 6))
        E_real.append(round(L_km * _ecr(v_act), 4))

    ferry = {str(i): n.get("crossing_h")
             for i, n in enumerate(geom["nodes"], start=1)
             if n["kind"] == "ferry"}
    powers = {str(i): n.get("power_kw")
              for i, n in enumerate(geom["nodes"], start=1)
              if n["kind"] == "charger"}

    payload = dict(
        meta=dict(
            route_class="usecase", customers_class="real",
            window_class="none", seed=seed,
            geometry_seed=GEOMETRY_SEED,
            cv=cv, dist="shifted-lognormal",
            xi_min=XI_MIN, xi_max=XI_MAX, ar1_rho=ar1_rho,
            cs_spacing_km=None,
            charger_power_kw=charger_power_kw,
            battery_kwh=battery_kwh,
            add_laybys=True,
            created_at=datetime.now().isoformat(timespec="seconds"),
            window_half_widths={},
            source="telematics trip report (geometry) + OpenStreetMap (K, L)",
            total_km=geom["total_km"],
            # recorded but NOT enforced yet — see module docstring
            ferry_nodes=ferry,
            charger_power_kw_per_site=powers,
        ),
        instance=_to_json_safe(data),
        D_real=D_real,
        E_real=E_real,
    )
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if verbose:
        tot_nom = sum(data["D"].values())
        tot_act = sum(D_real)
        print(f"  {title}.json  N={N}  |C|={len(data['C'])} "
              f"|K|={len(data['K'])} |L|={len(data['L'])}  "
              f"driving {tot_nom:.1f}h nominal / {tot_act:.1f}h realised  "
              f"-> {os.path.getsize(filepath)/1e6:.1f} MB")
    return os.path.abspath(filepath)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of travel-time realisations (default 5)")
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--cv", type=float, default=TRAVEL_TIME_CV_TARGET)
    ap.add_argument("--battery-kwh", type=float, default=BATTERY_CAPACITY)
    ap.add_argument("--charger-power-kw", type=float,
                    default=CHARGER_POWER_BASE_KW)
    args = ap.parse_args()

    print("=" * 68)
    print("  usecase_instance.py — real geometry, synthetic travel times")
    print(f"  out_dir = {args.out_dir}")
    print(f"  seeds   = {args.first_seed}..{args.first_seed + args.seeds - 1}"
          f"   cv = {args.cv}")
    print(f"  battery = {args.battery_kwh:.0f} kWh   "
          f"charger = {args.charger_power_kw:.0f} kW")
    print("=" * 68)

    geom = build_geometry()
    for s in range(args.first_seed, args.first_seed + args.seeds):
        generate_usecase_file(
            s, geom, args.out_dir, cv=args.cv,
            battery_kwh=args.battery_kwh,
            charger_power_kw=args.charger_power_kw)

    print(f"\nrun them with:\n"
          f'  python -m src.simulation.runner_dispatch '
          f'"{os.path.basename(args.out_dir)}/usecase_*.json" all --jobs 4')


if __name__ == "__main__":
    main()
