# -*- coding: utf-8 -*-
"""
real_route_strip.py — the real-life instance as a linear route strip.

Same visual grammar as the concept route diagram
(src/plot/concept_solution_figure.py): black squares for origin/destination,
green triangles for charging stations, orange diamonds for customers, thin
grey ticks for rest areas, plus a blue ferry marker this instance needs.

Anonymised by default: customers are labelled C1..C7, no place names.

Usage
-----
    python -m src.plot.real_route_strip [--out figures/real_route_strip.png]
                                        [--named]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src import paths as _paths

# ---- palette: identical to the concept figure ------------------------------
INK, MUT = "#222222", "#666666"
C_CHARGE = "#1e7a3c"
C_SERV   = "#e8822d"
C_FERRY  = "#2b7bba"
C_LAYBY  = "#999999"

DATA = os.path.join(os.path.dirname(_paths.instances()), "data_output")
STOPS_CSV = os.path.join(DATA, "real_route_stops.csv")
CS_CSV    = os.path.join(DATA, "real_route_cs_hgv.csv")
RA_CSV    = os.path.join(DATA, "real_route_restareas.csv")

LAYBY_MERGE_KM = 5.0     # rest areas closer than this share one node


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_instance():
    """Return (total_km, customers, chargers, laybys, ferries)."""
    stops = _read(STOPS_CSV)
    total_km = max(float(r["km"]) for r in stops)

    customers, ferries = [], []
    for r in stops:
        km = float(r["km"])
        if r["type"] == "customer":
            customers.append(dict(km=km, service_h=float(r["dwell_h"] or 0),
                                  addr=r["address"]))
        elif r["type"] == "ferry":
            ferries.append(dict(km=km,
                                dwell_h=float(r["ferry_dwell_h"] or 0)))

    chargers = [dict(km=float(r["km"]), kw=float(r["kw"] or 0))
                for r in _read(CS_CSV)]

    laybys, cluster = [], []
    for p in sorted((float(r["km"]) for r in _read(RA_CSV))):
        if cluster and p - cluster[-1] >= LAYBY_MERGE_KM:
            laybys.append(cluster[0])
            cluster = []
        cluster.append(p)
    if cluster:
        laybys.append(cluster[0])

    return total_km, customers, chargers, laybys, ferries


def make_figure(out_png: str, named: bool = False) -> str:
    total_km, customers, chargers, laybys, ferries = load_instance()

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9.5,
        "axes.edgecolor": "#aaaaaa", "axes.linewidth": 0.8,
    })
    fig = plt.figure(figsize=(15.0, 3.0))
    ax = fig.add_axes([0.012, 0.08, 0.976, 0.70])
    ax.set_xlim(-total_km * 0.030, total_km * 1.030)
    ax.set_ylim(-2.9, 2.15)
    ax.axis("off")

    ax.plot([0, total_km], [0, 0], color="#444444", lw=1.4, zorder=1)
    for x in laybys:
        ax.plot([x, x], [-0.16, 0.16], color=C_LAYBY, lw=1.0, zorder=2)
    ax.scatter([p["km"] for p in chargers], [0.34] * len(chargers),
               marker="^", s=110, c=C_CHARGE, edgecolors="#0d3c1d",
               linewidths=0.7, zorder=3)
    ax.scatter([p["km"] for p in customers], [0.34] * len(customers),
               marker="D", s=95, c=C_SERV, edgecolors="#7a3d0d",
               linewidths=0.7, zorder=4)
    ax.scatter([p["km"] for p in ferries], [0.34] * len(ferries),
               marker="o", s=115, c=C_FERRY, edgecolors="#14456b",
               linewidths=0.7, zorder=4)

    for x in (0.0, total_km):
        ax.scatter([x], [0], marker="s", s=190, c="#111111", zorder=5)

    # ── labels: name above, km below, stacked into rows so that nothing
    # overlaps.  On a 3339 km axis the depot and its first customer are 8 km
    # apart, so single-row placement is not an option.
    labels = [dict(km=0.0, name="O", sub="0 km", col=INK, bold=11)]
    for n, c in enumerate(sorted(customers, key=lambda p: p["km"]), start=1):
        nm = f"C{n}"
        if named and "," in c["addr"]:
            nm = c["addr"].split(",")[-2].strip()
        labels.append(dict(km=c["km"], name=nm, sub=f"{c['km']:.0f} km",
                           col="#7a3d0d", bold=10))
    for f in ferries:
        labels.append(dict(km=f["km"], name="F", sub=f"{f['dwell_h']:.1f} h",
                           col=C_FERRY, bold=10))
    labels.append(dict(km=total_km, name="D", sub=f"{total_km:.0f} km",
                       col=INK, bold=11))
    labels.sort(key=lambda d: d["km"])

    # crude but reliable text width in data units (~0.0058 of the axis per
    # character at this figure width and font size)
    char_km = total_km * 0.0058
    rows_x: list[float] = []                    # right edge occupied per row
    for lb in labels:
        half = 0.5 * max(len(lb["name"]), len(lb["sub"])) * char_km
        r = 0
        while r < len(rows_x) and lb["km"] - half < rows_x[r]:
            r += 1
        if r == len(rows_x):
            rows_x.append(-1e9)
        rows_x[r] = lb["km"] + half + 0.004 * total_km
        y_up = 0.62 + 0.58 * r
        y_dn = -0.52 - 0.53 * r
        ax.text(lb["km"], y_up, lb["name"], ha="center", va="bottom",
                fontsize=lb["bold"], fontweight="bold", color=lb["col"])
        ax.text(lb["km"], y_dn, lb["sub"], ha="center", va="top",
                fontsize=8.5, color=lb["col"])
        if r:                                   # tie the label to its stop
            ax.plot([lb["km"], lb["km"]], [0.30, y_up - 0.04],
                    color=lb["col"], lw=0.6, alpha=0.45, zorder=2)

    leg = [Line2D([], [], color=C_LAYBY, lw=1.2, marker="|", markersize=11,
                  linestyle="none", label=f"Rest area ({len(laybys)})"),
           Line2D([], [], marker="^", color=C_CHARGE, markeredgecolor="#0d3c1d",
                  markersize=9, linestyle="none",
                  label=f"Charging station ({len(chargers)})"),
           Line2D([], [], marker="D", color=C_SERV, markeredgecolor="#7a3d0d",
                  markersize=8, linestyle="none",
                  label=f"Customer ({len(customers)})"),
           Line2D([], [], marker="o", color=C_FERRY, markeredgecolor="#14456b",
                  markersize=9, linestyle="none",
                  label="Ferry (forced break)")]
    ax.legend(handles=leg, loc="upper center", ncol=4, frameon=True,
              framealpha=1.0, edgecolor="#33415c", fontsize=8.8,
              bbox_to_anchor=(0.5, 1.15), borderpad=0.55, handletextpad=0.35)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=_paths.figure_out("real_route_strip.png"))
    ap.add_argument("--named", action="store_true",
                    help="label customers by place instead of C1..Cn "
                         "(NOT for the paper)")
    args = ap.parse_args()
    print(f"written {make_figure(args.out, named=args.named)}")


if __name__ == "__main__":
    main()
