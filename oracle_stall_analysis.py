"""
oracle_stall_analysis.py — Why do some oracle solves certify and others stall?
==============================================================================
Joins the oracle caches (solutions/oracle_<instance>.json) with the instance
JSONs and compares SOLVED (certified: final gap is a number, i.e. optimal or
gap-threshold) against STALLED (gap = nan at the time limit) instances, per
route class:

  * instance features   — N, total realised driving time, |C|, TW class
  * optimal-schedule composition — number of rests (rho1 11 h / rho2 9 h),
    breaks, charges, total rest and charge time

Headline finding (2026-07, long routes): solved optima use exactly 4 rests
(3x9 h reduced — the full rho_bar budget — + 1x11 h); stalled optima need a
5th rest (2x11 h + 3x9 h).  The driving-arithmetic count bound
ceil(D/Tdrv_sh2)-1 equals 4 for BOTH groups, so certifying the 5th rest is
what stalls: it is forced by shift-SPREAD interactions (charging + breaks +
queues consume the 15 h spread windows), the mechanism VI-7 encodes.

Usage
-----
  python oracle_stall_analysis.py                 # all classes
  python oracle_stall_analysis.py --class long    # one route class
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import Counter, defaultdict

_CLASS_ORDER = ["short", "medium", "long"]


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float)
                                                and math.isnan(x))


def load_rows(solutions_dir: str = "solutions",
              instances_dir: str = "instances") -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(solutions_dir, "oracle_*.json")):
        inst = os.path.basename(f)[len("oracle_"):-len(".json")]
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mcls = re.search(r"R(short|medium|long)", inst)
        mtw = re.search(r"T(tight|medium|large|none)", inst)
        row = dict(
            instance=inst,
            route_class=mcls.group(1) if mcls else "?",
            tw_class=mtw.group(1) if mtw else "?",
            solved=_is_num(d.get("gap")),
            gap=d.get("gap"),
            obj=d.get("obj"),
            status=d.get("status"),
        )
        # optimal / best-found schedule composition
        sol = d.get("sol") or []
        if sol:
            row.update(
                n_rho1=sum(int(s.get("rho1", 0)) for s in sol),
                n_rho2=sum(int(s.get("rho2", 0)) for s in sol),
                n_breaks=sum(int(s.get("b45", 0)) + int(s.get("b15", 0))
                             + int(s.get("b30", 0)) for s in sol),
                n_charges=sum(int(s.get("y", 0)) for s in sol),
                taur_tot=sum(float(s.get("taur", 0) or 0) for s in sol),
                tauc_tot=sum(float(s.get("tauc", 0) or 0) for s in sol),
            )
            row["n_rests"] = row["n_rho1"] + row["n_rho2"]
        # instance features
        ip = os.path.join(instances_dir, inst + ".json")
        if os.path.exists(ip):
            try:
                idata = json.load(open(ip, encoding="utf-8"))
                D_real = idata.get("D_real", [])
                dd = idata.get("instance", {})
                if not (isinstance(dd, dict) and "N" in dd):
                    dd = idata.get("data", {})
                row.update(
                    N=dd.get("N", len(D_real)),
                    D_total=sum(D_real),
                    n_cust=len(dd.get("C", []) or []),
                    n_rho_bound=max(0, math.ceil(sum(D_real) / 10.0) - 1),
                )
            except (OSError, json.JSONDecodeError):
                pass
        rows.append(row)
    return rows


def _summ(rows: list[dict], key: str) -> str:
    v = [r[key] for r in rows if _is_num(r.get(key))]
    if not v:
        return f"  {key:12s} (no data)"
    return (f"  {key:12s} mean={st.mean(v):8.2f}  med={st.median(v):8.2f}  "
            f"min={min(v):7.2f}  max={max(v):7.2f}")


def report(rows: list[dict], route_class: str):
    rs = [r for r in rows if r["route_class"] == route_class]
    if not rs:
        return
    S = [r for r in rs if r["solved"]]
    U = [r for r in rs if not r["solved"]]
    print(f"\n{'='*72}\n  {route_class.upper()} routes:  n={len(rs)}  "
          f"solved={len(S)}  stalled={len(U)}\n{'='*72}")
    for name, grp in (("SOLVED", S), ("STALLED", U)):
        if not grp:
            continue
        print(f"\n  -- {name} (n={len(grp)}) --")
        for key in ("N", "D_total", "n_cust", "obj",
                    "n_rho1", "n_rho2", "n_rests", "n_breaks", "n_charges",
                    "taur_tot", "tauc_tot"):
            print(_summ(grp, key))
        print(f"  {'tw_class':12s}",
              dict(Counter(r['tw_class'] for r in grp)))
        have = [r for r in grp if "n_rests" in r]
        if have:
            print(f"  {'rests dist':12s}",
                  dict(sorted(Counter(r['n_rests'] for r in have).items())),
                  "  (rho1 dist:",
                  dict(sorted(Counter(r['n_rho1'] for r in have).items())),
                  ")")
        bd = [r for r in grp if _is_num(r.get("n_rho_bound"))]
        if bd:
            print(f"  {'VI-5 bound':12s}",
                  dict(sorted(Counter(r['n_rho_bound'] for r in bd).items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compare solved vs stalled oracle instances.")
    ap.add_argument("--class", dest="route_class", default=None,
                    choices=_CLASS_ORDER,
                    help="restrict to one route class (default: all)")
    ap.add_argument("--solutions-dir", default="solutions")
    ap.add_argument("--instances-dir", default="instances")
    args = ap.parse_args()

    rows = load_rows(args.solutions_dir, args.instances_dir)
    if not rows:
        raise SystemExit("no oracle caches found in "
                         f"{args.solutions_dir}/oracle_*.json")
    classes = [args.route_class] if args.route_class else _CLASS_ORDER
    for rc in classes:
        report(rows, rc)
