"""
norway_instance.py — I4: real Norwegian long-haul corridor loader (skeleton)
=============================================================================
Builds a real E6-corridor instance (Oslo – Trondheim – Bodø/Tromsø) in the
same data-dict format as instances.make_data(), so every method runs on it
unchanged via runner_dispatch.

STATUS: loader skeleton — the code path is complete, but it needs the data
files below (none are bundled with the repo):

  data/norway/chargers.csv      HDV-capable charger sites on the corridor.
      Source: NOBIL database export (https://info.nobil.no — the Norwegian
      public charging registry; filter for HDV-capable sites), supplemented
      with announced Milence / E.ON HDV sites.
      Columns: name, km_from_oslo, power_kw
  data/norway/customers.csv     Customer terminals.
      Columns: name, km_from_oslo, service_h, window_open_h, window_close_h
      (window columns optional — leave empty to generate I1-style windows)
  data/norway/legs.csv          (optional) Per-leg distance and nominal speed
      from OpenStreetMap/OSRM; if absent, distances follow from km_from_oslo
      and speed defaults to settings.V_NOM.
  data/norway/dispersion.json   (optional) Travel-time dispersion calibrated
      from Statens vegvesen traffic data
      (https://trafikkdata.atlas.vegvesen.no): {"delta": float,
      "ar1_rho": float}.

Note: gradients on the corridor make the constant-ECR assumption
conservative — discuss in the paper's conclusion (§8).

Usage
-----
  python norway_instance.py [--out instances/norway_e6.json] [--seed 1]

The output JSON has the same schema as instance_io.generate_instance_file,
so:  python runner_dispatch.py instances/norway_e6.json all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

from instances import make_data
from instance_io import _to_json_safe, generate_time_windows
from settings import (V_NOM, ecr, LOWER_PCT, TRAVEL_TIME_DIST,
                      TRAVEL_TIME_AR1_RHO, sample_multipliers, scale_tbar)

DATA_DIR = os.path.join("data", "norway")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_norway_instance(seed: int = 1) -> tuple[dict, float, float]:
    """
    Assemble the E6 corridor instance from the CSV files in data/norway/.

    Returns (full_data, delta, ar1_rho).
    Raises FileNotFoundError with a data-sourcing hint if files are missing.
    """
    chargers_csv  = os.path.join(DATA_DIR, "chargers.csv")
    customers_csv = os.path.join(DATA_DIR, "customers.csv")
    if not (os.path.isfile(chargers_csv) and os.path.isfile(customers_csv)):
        raise FileNotFoundError(
            f"Norway corridor data not found in {DATA_DIR}/.\n"
            "Required: chargers.csv (NOBIL export, https://info.nobil.no, "
            "HDV-capable sites) and customers.csv.\n"
            "Optional: legs.csv (OSRM distances/speeds), dispersion.json "
            "(Statens vegvesen, https://trafikkdata.atlas.vegvesen.no).")

    chargers  = _read_csv(chargers_csv)
    customers = _read_csv(customers_csv)

    # merge stops by corridor position
    stops = ([dict(kind="K", km=float(r["km_from_oslo"]),
                   power=float(r.get("power_kw") or 200.0), name=r["name"])
              for r in chargers]
             + [dict(kind="C", km=float(r["km_from_oslo"]),
                     service=float(r.get("service_h") or 0.5),
                     wopen=r.get("window_open_h") or "",
                     wclose=r.get("window_close_h") or "",
                     name=r["name"])
                for r in customers])
    stops.sort(key=lambda s: s["km"])
    route_km = stops[-1]["km"] + 20.0     # 20 km past the last stop to dest

    # dispersion calibration (Statens vegvesen) or defaults
    disp_path = os.path.join(DATA_DIR, "dispersion.json")
    if os.path.isfile(disp_path):
        disp = json.load(open(disp_path, encoding="utf-8"))
        delta, ar1 = float(disp.get("delta", LOWER_PCT)), \
                     float(disp.get("ar1_rho", TRAVEL_TIME_AR1_RHO))
    else:
        delta, ar1 = LOWER_PCT, TRAVEL_TIME_AR1_RHO

    # index stops 0..N
    I = list(range(len(stops) + 2))
    C, K, D, E, km, Wha, Whf = [], [], {}, {}, {}, {}, {}
    prev_km = 0.0
    for j, s in enumerate(stops, start=1):
        leg_km = s["km"] - prev_km
        D[j-1]  = leg_km / V_NOM
        km[j-1] = leg_km
        E[j-1]  = leg_km * ecr(V_NOM)
        if s["kind"] == "C":
            C.append(j)
            if s["wopen"] != "" and s["wclose"] != "":
                Wha[j] = float(s["wopen"])
                Whf[j] = float(s["wclose"])
            else:
                Wha[j], Whf[j] = 0.0, 2e7      # I1 windows generated below
        else:
            K.append(j)
        prev_km = s["km"]
    last = len(stops)
    D[last]  = (route_km - prev_km) / V_NOM
    km[last] = route_km - prev_km
    E[last]  = km[last] * ecr(V_NOM)

    # mean charger power on the corridor sets the PWL curve
    powers = [s["power"] for s in stops if s["kind"] == "K"] or [200.0]

    full_data = make_data(
        I=I, C=C, K=K, D=D, E=E, km=km,
        Wha=Wha, Whf=Whf,
        label=f"Norway E6 corridor — {route_km:.0f} km, "
              f"{len(C)} terminals, {len(K)} HDV chargers (NOBIL)",
        title=f"norway_e6_{seed}",
        rng=np.random.default_rng(seed),
        charger_power_kw=float(np.mean(powers)),
    )

    # generate exposure-scaled windows for customers without explicit ones
    if any(Whf.get(c, 2e7) >= 1e6 for c in C):
        generate_time_windows(full_data, "medium",
                              np.random.default_rng(seed))
    return full_data, delta, ar1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",  default=os.path.join("instances", "norway_e6.json"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    full_data, delta, ar1 = build_norway_instance(seed=args.seed)
    N   = full_data["N"]
    rng = np.random.default_rng(args.seed)

    mults  = sample_multipliers(N, rng, delta=delta,
                                dist=TRAVEL_TIME_DIST, ar1_rho=ar1)
    D_real, E_real = [], []
    for leg in range(N):
        d = full_data["D"][leg] * float(mults[leg])
        L = full_data["km"][leg]
        v = L / d if d > 0 else V_NOM
        D_real.append(round(d, 6))
        E_real.append(round(L * ecr(v), 4))

    payload = dict(
        meta=dict(route_class="norway_e6", customers_class="real",
                  window_class="medium", seed=args.seed, delta=delta,
                  ar1_rho=ar1, created_at="",
                  window_half_widths={}),
        instance=_to_json_safe(full_data),
        D_real=D_real, E_real=E_real,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"written {args.out}  (N={N}, |C|={len(full_data['C'])}, "
          f"|K|={len(full_data['K'])})")


if __name__ == "__main__":
    main()
