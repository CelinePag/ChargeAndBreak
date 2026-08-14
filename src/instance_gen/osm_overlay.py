# -*- coding: utf-8 -*-
"""Overlay charging stations and rest areas from OSM (Overpass) onto the
reconstructed PD74720 route.

- Corridor = polyline of the GPS spine (downsampled to ~4 km spacing; sea
  segments collapse because their spine-km does not advance).
- Chargers:  amenity=charging_station within 3 km of the corridor.
- Rest areas: highway=rest_area / highway=services within 2.5 km.
- Each POI gets one km-position PER PASS of the route (out and return legs
  both pass through Denmark), by clustering nearby spine points on km gaps.

Responses are cached as JSON next to this script so reruns are offline.
"""
import json
import os
import re
import sys
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from route_from_data import (build_route, spine_km, make_figure, write_csv,
                             print_table)

OUT_DIR = r"c:/Users/celinep/Documents/GitHub/ChargeAndBreak/data_output"
OVERPASS = "https://overpass-api.de/api/interpreter"


MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter"]


CACHE_DIR = (r"c:/Users/celinep/Documents/GitHub/ChargeAndBreak/"
             r"data_output/osm_cache")


def fetch(cache_name, query, tries=4):
    import time
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, cache_name)
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)
    body = urllib.parse.urlencode({"data": query}).encode()
    last_err = None
    for attempt in range(tries):
        url = MIRRORS[attempt % len(MIRRORS)]
        try:
            req = urllib.request.Request(url, data=body, headers={
                "User-Agent": "ChargeAndBreak-research/1.0"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.loads(resp.read().decode())
            if data.get("remark", "").startswith("runtime error"):
                raise RuntimeError(data["remark"])
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            time.sleep(5)          # be polite between heavy queries
            return data
        except Exception as e:              # 429/504/timeout -> back off, retry
            last_err = e
            wait = 15 * (attempt + 1)
            print(f"    fetch failed ({e}); retrying in {wait}s on next mirror")
            time.sleep(wait)
    raise last_err


def parse_kw(tags):
    """Best-effort max output (kW) from OSM charging-station tags."""
    vals = []
    for k, v in tags.items():
        if k in ("charging_station:output", "maxoutput", "output") \
                or (k.startswith("socket:") and k.endswith(":output")):
            for m in re.finditer(r"([\d.]+)\s*(kW|kVA|W)?", str(v), re.I):
                try:
                    x = float(m.group(1))
                except ValueError:
                    continue
                unit = (m.group(2) or "kW").lower()
                if unit == "w" or x > 2000:
                    x /= 1000.0
                if 1 <= x <= 5000:
                    vals.append(x)
    return max(vals) if vals else None


def main():
    route = build_route()
    lat, lon, km = spine_km(route)
    total_km = route["total_km"]

    # downsample: one point per >=4 km of spine progress (sea collapses)
    keep = [0]
    for j in range(1, len(km)):
        if km[j] - km[keep[-1]] >= 4.0:
            keep.append(j)
    pl_lat, pl_lon, pl_km = lat[keep], lon[keep], km[keep]
    poly = ",".join(f"{a:.4f},{b:.4f}" for a, b in zip(pl_lat, pl_lon))
    print(f"corridor polyline: {len(keep)} points "
          f"({len(poly)/1000:.0f} kB query payload)")

    q_ra = (f'[out:json][timeout:180];'
            f'nwr["highway"~"^(rest_area|services)$"](around:2500,{poly});'
            f'out center tags;')

    # charging_station is a heavy tag class in DE/NL: one query over the whole
    # corridor times out server-side, so fetch in overlapping corridor chunks
    # and dedupe by osm id.
    CHUNK = 90
    pts = list(zip(pl_lat, pl_lon))
    cs_raw, seen = [], set()

    def add_elements(data):
        n = 0
        for el in data["elements"]:
            key = (el["type"], el["id"])
            if key not in seen:
                seen.add(key)
                cs_raw.append(el)
                n += 1
        return n

    def fetch_chargers(seg, name):
        """Fetch one corridor segment; on server timeout split it in half
        (dense regions need smaller queries)."""
        seg_poly = ",".join(f"{a:.4f},{b:.4f}" for a, b in seg)
        q = (f'[out:json][timeout:120];'
             f'nwr["amenity"="charging_station"](around:3000,{seg_poly});'
             f'out center tags;')
        try:
            data = fetch(f"cache_chargers_{name}.json", q, tries=2)
        except Exception as e:
            if len(seg) < 12:
                raise
            print(f"  chunk {name}: splitting after failure ({e})")
            mid = len(seg) // 2
            fetch_chargers(seg[:mid + 1], name + "a")
            fetch_chargers(seg[mid:], name + "b")
            return
        n = add_elements(data)
        print(f"  chunk {name}: +{n} (unique so far {len(cs_raw)})")

    n_chunks = (len(pts) - 1) // CHUNK + 1
    for ci in range(n_chunks):
        seg = pts[max(0, ci * CHUNK - 1):(ci + 1) * CHUNK]
        fetch_chargers(seg, f"{ci:02d}")
    ra_raw = fetch("cache_restareas.json", q_ra)["elements"]
    print(f"OSM: {len(cs_raw)} charging stations, {len(ra_raw)} rest areas "
          f"within corridor")

    def coords(el):
        if "lat" in el:
            return el["lat"], el["lon"]
        c = el.get("center")
        return (c["lat"], c["lon"]) if c else (None, None)

    # vectorized km assignment: cluster spine points within radius by km gap
    plat_r = np.radians(pl_lat)
    plon_r = np.radians(pl_lon)

    def km_positions(la, lo, radius_km):
        la_r, lo_r = np.radians(la), np.radians(lo)
        dlat = plat_r - la_r
        dlon = plon_r - lo_r
        h = (np.sin(dlat / 2) ** 2
             + np.cos(la_r) * np.cos(plat_r) * np.sin(dlon / 2) ** 2)
        d = 2 * 6371.0 * np.arcsin(np.sqrt(h))
        near = np.where(d <= radius_km)[0]
        if near.size == 0:
            return []
        out = []
        grp = [near[0]]
        for i in near[1:]:
            if pl_km[i] - pl_km[grp[-1]] > 100.0:   # new pass of the route
                out.append(grp)
                grp = []
            grp.append(i)
        out.append(grp)
        return [(float(pl_km[min(g, key=lambda i: d[i])]),
                 float(d[min(g, key=lambda i: d[i])])) for g in out]

    def collect(raw, radius_km, want_kw=False):
        pois = []
        for el in raw:
            la, lo = coords(el)
            if la is None:
                continue
            tags = el.get("tags", {})
            kw = parse_kw(tags) if want_kw else None
            name = tags.get("name") or tags.get("operator") or ""
            for k, dist in km_positions(la, lo, radius_km):
                pois.append(dict(km=k, lat=la, lon=lo, dist_km=dist,
                                 kw=kw, name=name,
                                 osm=f"{el['type']}/{el['id']}",
                                 hgv=tags.get("hgv", ""),
                                 truck=tags.get("truck", "")))
        pois.sort(key=lambda p: p["km"])
        return pois

    # rest areas: 500 m corridor (agreed)
    restareas = collect(ra_raw, 0.5)
    print(f"rest-area entries on km-axis (0.5 km corridor): {len(restareas)}")

    # ── K set: HGV-tagged chargers only, corridor widened until feasible ────
    # hgv/truck=yes is a rare tag, so one selective wide-band query is cheap.
    BAND = 50.0   # km — search band; the instance radius is chosen by sweep
    sparse = [0]
    for j in range(1, len(pl_km)):
        if pl_km[j] - pl_km[sparse[-1]] >= 25.0:
            sparse.append(j)
    wide_poly = ",".join(f"{pl_lat[i]:.4f},{pl_lon[i]:.4f}" for i in sparse)
    q_hgv = (f'[out:json][timeout:180];('
             f'nwr["amenity"="charging_station"]["hgv"="yes"]'
             f'(around:{BAND*1000:.0f},{wide_poly});'
             f'nwr["amenity"="charging_station"]["truck"="yes"]'
             f'(around:{BAND*1000:.0f},{wide_poly});'
             f');out center tags;')
    hgv_raw = fetch(f"cache_chargers_hgv{BAND:.0f}km.json", q_hgv)["elements"]
    print(f"HGV-tagged stations within {BAND:.0f} km band: {len(hgv_raw)}")

    hgv_all = collect(hgv_raw, BAND, want_kw=True)
    for p in hgv_all:
        p["origin"] = "osm"

    # curated additions: documented HDV sites missing hgv tags in OSM
    import csv as _csv
    curated_path = (r"c:/Users/celinep/Documents/GitHub/ChargeAndBreak/"
                    r"data/curated_cs.csv")
    if os.path.isfile(curated_path):
        with open(curated_path, encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                la, lo = float(row["lat"]), float(row["lon"])
                for k, dist in km_positions(la, lo, BAND):
                    hgv_all.append(dict(
                        km=k, lat=la, lon=lo, dist_km=dist,
                        kw=float(row["power_kw"]), name=row["name"],
                        osm=row["source"], hgv="curated", truck="",
                        origin="curated"))
                    print(f"curated: {row['name']} at km {k:.0f} "
                          f"({dist:.1f} km off-route)")

    def usable(p):
        """Junk filter: tagged sub-150 kW hardware is car/bike equipment;
        unknown-power sites are trusted only right next to the road."""
        if p["origin"] == "curated":
            return True
        if p["kw"] is not None:
            return p["kw"] >= 150.0
        return p["dist_km"] <= 5.0

    def kset_at(radius):
        """Site list at a corridor radius: usable entries within radius
        (curated always kept), merged into sites when <=1.5 km apart on the
        km-axis (same charging park)."""
        sel = [p for p in hgv_all if usable(p)
               and (p["dist_km"] <= radius or p["origin"] == "curated")]
        sites = []
        for p in sorted(sel, key=lambda p: p["km"]):
            if sites and p["km"] - sites[-1]["km"] <= 1.5:
                s = sites[-1]
                s["kw"] = max(s["kw"] or 0, p["kw"] or 0) or None
                s["dist_km"] = min(s["dist_km"], p["dist_km"])
                continue
            sites.append(dict(p))
        return sites

    def gap_profile(sites):
        """(first_gap, max_internal_gap w/ endpoints, last_gap).

        First/last legs start/end at the depot with a full battery, so they
        tolerate more than internal gaps (where arrival SOC may be at Emin
        before reaching the NEXT charger)."""
        kms = [s["km"] for s in sites]
        first, last = kms[0], total_km - kms[-1]
        if len(kms) < 2:
            return first, 0.0, 0.0, 0.0, last
        g = np.diff(kms)
        i = int(np.argmax(g))
        return first, float(g[i]), kms[i], kms[i + 1], last

    # feasibility targets at base battery (500 kWh, Emin 20% -> 400 kWh
    # usable = 325 km at ECR(80); observed leg speeds are lower, ECR ~1.05,
    # so ~350 km is the realistic bound used for the depot-anchored ends)
    print("\ncorridor-radius sweep (usable HGV sites + curated):")
    print(f"{'radius':>7} {'sites':>6} {'first':>7} {'max internal':>13} {'last':>6}")
    chosen = None
    for r in (1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50):
        sites = kset_at(r)
        if not sites:
            continue
        first, g, a, b, last = gap_profile(sites)
        print(f"{r:6.0f}km {len(sites):6d} {first:6.0f}km {g:7.0f}km "
              f"(km {a:.0f}->{b:.0f}) {last:5.0f}km")
        if chosen is None and g <= 300.0 and first <= 350.0 and last <= 350.0:
            chosen = r
    if chosen is None:
        chosen = int(BAND)
        print(f"  no radius satisfies the gap targets — using {chosen} km")
    kset = kset_at(chosen)
    first, g, a, b, last = gap_profile(kset)
    print(f"\nchosen corridor: {chosen} km -> K = {len(kset)} sites; "
          f"first {first:.0f} km, max internal {g:.0f} km "
          f"(km {a:.0f}->{b:.0f}), last {last:.0f} km")
    print(f"{'km':>6}  {'kW':>6}  {'off-route':>9}  name")
    for s in kset:
        print(f"{s['km']:6.0f}  {str(s['kw'] or '?'):>6}  "
              f"{s['dist_km']:7.1f}km  {s['name'][:44]}  [{s['osm']}]")
    hgv_tag = kset   # downstream: CSV + figure

    # ── M_maneuver per CS (agreed formula) ──────────────────────────────────
    # Only the EXCESS detour beyond the driver's typical off-road stop
    # distance counts, at 80 km/h round trip:
    #   m_man = M_STOP + 2*max(0, dist - baseline)/80,
    # baseline = mean off-route distance of the rest areas the driver
    # actually used (matched by km-position to the observed break/rest stops).
    from src.settings import M_STOP_H
    used = []
    for v in route["visits"]:
        if v["type"] in ("break", "rest"):
            cands = [p for p in restareas if abs(p["km"] - v["km"]) <= 3.0]
            if cands:
                used.append(min(cands,
                                key=lambda p: abs(p["km"] - v["km"]))["dist_km"])
    base_det = float(np.mean(used)) if used else 0.0
    print(f"\nmaneuver baseline: mean off-route distance of driver-used rest "
          f"areas = {base_det:.2f} km ({len(used)} matched stops)")

    def m_man(dist_km):
        return round(M_STOP_H + 2.0 * max(0.0, dist_km - base_det) / 80.0, 3)

    for s in kset:
        s["m_man_h"] = m_man(s["dist_km"])

    # ── alternatives layer: >=350 kW sites near the route, ANY tag status ───
    # (cross-check against the HGV-only K set; OpenChargeMap would be the
    # true second source but needs a registered API key — slot below.)
    alt = [p for p in collect(cs_raw, 3.0, want_kw=True)
           if (p["kw"] or 0) >= 350.0]
    kset_pos = [s["km"] for s in kset]
    alt = [p for p in alt
           if not any(abs(p["km"] - k) <= 1.5 for k in kset_pos)]
    alt_sites = []
    for p in sorted(alt, key=lambda p: p["km"]):
        if alt_sites and p["km"] - alt_sites[-1]["km"] <= 1.5:
            s = alt_sites[-1]
            s["kw"] = max(s["kw"], p["kw"])
            s["dist_km"] = min(s["dist_km"], p["dist_km"])
            continue
        p = dict(p, origin="osm>=350kW", m_man_h=m_man(p["dist_km"]))
        alt_sites.append(p)

    # promote name-evident truck sites into K (certain: Milence is a
    # truck-only operator; "Truck" in the site name is explicit)
    sure = [p for p in alt_sites
            if re.search(r"milence|truck", p["name"], re.I)]
    alt_sites = [p for p in alt_sites if p not in sure]
    for p in sure:
        p["origin"] = "osm-name-truck"
        print(f"promoted to K: km {p['km']:.0f}  {p['name']}  "
              f"({p['dist_km']:.1f} km off)")
    kset = sorted(kset + sure, key=lambda s: s["km"])
    first, g, a, b, last = gap_profile(kset)
    print(f"final K = {len(kset)} sites; first {first:.0f} km, "
          f"max internal {g:.0f} km (km {a:.0f}->{b:.0f}), last {last:.0f} km")

    print(f"\nalternative >=350 kW sites within 3 km (not in K): "
          f"{len(alt_sites)}")
    for s in alt_sites:
        print(f"  km {s['km']:6.0f}  {s['kw']:5.0f} kW  "
              f"{s['dist_km']:4.1f} km off  {s['name'][:44]}  [{s['osm']}]")
    union = sorted(kset + alt_sites, key=lambda s: s["km"])
    fu, gu, au, bu, lu = gap_profile(union)
    print(f"K + alternatives: {len(union)} sites; first {fu:.0f} km, "
          f"max internal {gu:.0f} km (km {au:.0f}->{bu:.0f}), last {lu:.0f} km")

    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, pois, cols in (
            ("real_route_cs_hgv.csv", kset,
             ["km", "kw", "dist_km", "m_man_h", "origin", "lat", "lon",
              "name", "osm"]),
            ("real_route_cs_alternatives.csv", alt_sites,
             ["km", "kw", "dist_km", "m_man_h", "origin", "lat", "lon",
              "name", "osm"]),
            ("real_route_restareas.csv", restareas,
             ["km", "dist_km", "lat", "lon", "name", "osm"])):
        lines = [",".join(cols)]
        for p in pois:
            row = []
            for c in cols:
                v = p.get(c)
                if isinstance(v, float):
                    row.append(f"{v:.2f}")
                else:
                    row.append('"' + str(v or "").replace('"', "'") + '"'
                               if c in ("name",) else str(v or ""))
            lines.append(",".join(row))
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print(f"wrote {os.path.join(OUT_DIR, fname)}")

    png = make_figure(route, chargers=kset, restareas=restareas,
                      alt_chargers=alt_sites)

    # ── instance figure: aggregate laybys (<5 km apart -> keep one) ─────────
    laybys = []
    cluster = []
    for p in sorted(restareas, key=lambda p: p["km"]):
        if cluster and p["km"] - cluster[-1]["km"] >= 5.0:
            laybys.append(min(cluster, key=lambda q: q["dist_km"]))
            cluster = []
        cluster.append(p)
    if cluster:
        laybys.append(min(cluster, key=lambda q: q["dist_km"]))
    print(f"laybys after <5 km aggregation: {len(laybys)} "
          f"(from {len(restareas)} rest areas)")
    from route_from_data import make_instance_figure
    print(f"instance figure -> {make_instance_figure(route, kset, laybys)}")
    print(f"figure -> {png}")


if __name__ == "__main__":
    main()
