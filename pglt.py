"""
pglt.py — R15-PGLT benchmark (Peña-Arenas / Garaix) for the deterministic MILP
===============================================================================
Validates MILP.build_model against the published optima of the 40 R15-PGLT
truck-driver-scheduling instances (No-Night configuration).

Source data:  data/R15-PGLT/  (see the README there for provenance & format).
Reference:    data/R15-PGLT/Agg-NO-Night.txt — optimal completion (minutes).

Mapping (their TDSP -> our BET model with the battery switched off)
-------------------------------------------------------------------
  * minutes -> hours (/60); objective reported as (ta[N] - T_START)*60
  * their node 0 (depot loading, counts as work) -> customer stop with
    per-stop service S (make_data S= parameter), zero-length leg from origin
  * their nodes 1..n (clients)  -> customer stops, hard TWs on arrival
  * their node n+1 (end depot)  -> dummy end customer (S = s[n+1], usually 0)
    carrying the MANDATORY terminal daily rest (their constraint C0.1);
    enforced post-build:  rho1 + rho2 == 1  at that stop
  * preemptive driving breaks   -> legs split into GRID-minute layby segments
    (their solutions place all events on the 15' grid; --scan-reference checks)
  * pure idle time ("APO")      -> allow_wait=True (w_i at customer/laybys)
  * delayed schedule start (e.g. Test_40, ready[0]=480) -> init_ta relaxed to
    ta[0] >= T_START post-build (their shift clocks start at first activity,
    which our accumulators reproduce since h[0]=cd[0]=sw[0]=0 and o_0=0)
  * battery disabled: K=[], E=0 on every leg (E is never a divisor in MILP.py)

Usage
-----
  python pglt.py all                 solve all 40, compare, write CSV
  python pglt.py 1,5,13              solve a subset (instance numbers)
  python pglt.py Test_7 --tee        single instance with solver log
  python pglt.py --scan-reference    audit their solution files (grid/APO/splits)

Options: --grid 15   layby grid in minutes (default 15)
         --mipgap 0  MIP gap (default 0 — exact comparison)
         --timelimit 600
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import pyomo.environ as pyo

import MILP
from instances import make_data

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "R15-PGLT")
INSTANCE_DIR = os.path.join(DATA_DIR, "instances")
REFERENCE_TXT = os.path.join(DATA_DIR, "Agg-NO-Night.txt")
NO_NIGHT_DIR = os.path.join(DATA_DIR, "no_night")


# ══════════════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_pglt(path: str) -> dict:
    """
    Parse one Test_k.txt file.  Returns a dict with all values in MINUTES:
      name    str
      n       int              — number of clients (nodes are 0..n+1)
      c       list[n+2]        — travel time of leg i -> i+1 (last entry 0)
      s       list[n+2]        — service time per node (s[0] = depot loading)
      e, l    list[n+2]        — ready time / due date (window on START)
    """
    lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
    name = lines[0].strip()
    rows = {}
    for ln in lines[1:]:
        parts = ln.split("\t")
        rows[parts[0].strip()] = [float(x) for x in parts[1:] if x.strip() != ""]
    n = int(rows["CLIENTS"][0])
    out = dict(name=name, n=n,
               c=rows["TRAVEL TIME"], s=rows["SERVICE TIME"],
               e=rows["READY TIME"], l=rows["DUE DATE"])
    for key in ("c", "s", "e", "l"):
        assert len(out[key]) == n + 2, f"{path}: row {key} has {len(out[key])} values, expected {n+2}"
    assert out["e"][0] == 0 or True  # delayed start handled via ta[0] >= T_START
    return out


def load_reference(path: str = REFERENCE_TXT) -> dict[int, tuple[float, float]]:
    """Agg-NO-Night.txt -> {instance_number: (objective_min, solve_time_s)}."""
    ref = {}
    for ln in open(path).read().splitlines()[1:]:
        mres = re.match(r"Instance\s+(\d+)\s+([\d.]+)\s+([\d.]+)", ln)
        if mres:
            ref[int(mres.group(1))] = (float(mres.group(2)), float(mres.group(3)))
    return ref


# ══════════════════════════════════════════════════════════════════════════════
# INSTANCE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_pglt_data(parsed: dict, grid_min: int = 15,
                    allow_wait: bool = True) -> tuple[dict, int]:
    """
    Build the canonical data dict for one parsed PGLT instance.

    Stop layout (our indexing):
      0                      origin (departure, no activity)
      …                      their node j becomes:
                               [zero-length layby]  — hosts a rest/break
                                 BEFORE service (their arrive->rest->serve)
                               [service chunks]     — customer stops of
                                 grid_min each (service preemptable on the
                                 grid, as in their model; window on the
                                 FIRST chunk = window on service start)
      j_end                  last chunk of their end depot node n+1
      N = j_end + 1          destination, reached by a zero-length leg

    Every driving leg is split into `grid_min`-minute segments separated by
    layby stops (break/rest only, zero overhead) so that breaks can preempt
    driving as in the reference model.

    Returns (data, j_end) where j_end is the stop index that must carry the
    mandatory terminal daily rest.
    """
    n = parsed["n"]
    C, L = [], []
    D, E, S_h, Wha, Whf = {}, {}, {}, {}, {}
    idx = 0            # index of the stop we are currently AT (starts at origin)

    def _customer(their_node: int):
        """Zero-layby (pre-service rest slot) + service chunks of grid_min."""
        nonlocal idx
        # zero-length layby immediately before the customer: a rest taken
        # here is a rest AT the customer location before service starts
        L.append(idx)
        D[idx] = 0.0
        E[idx] = 0.0
        idx += 1
        s_min = parsed["s"][their_node]
        e_min = parsed["e"][their_node]
        l_min = parsed["l"][their_node]
        k = max(1, math.ceil(s_min / grid_min))
        first = idx
        for j in range(k):
            C.append(idx)
            chunk = min(grid_min, s_min - j * grid_min) if s_min > 0 else 0.0
            chunk = max(chunk, 0.0)
            S_h[idx] = chunk / 60.0
            if j == 0:
                # window on the activity START (their [ew, lw])
                Wha[idx] = e_min / 60.0
                Whf[idx] = l_min / 60.0
            else:
                # later pieces: their per-node bound X[u] + P[u] <= lw + s —
                # the whole (possibly preempted) service must COMPLETE by
                # lw + s, so piece j may start no later than lw + s - chunk
                Whf[idx] = (l_min + s_min - chunk) / 60.0
            if j < k - 1:
                D[idx] = 0.0          # zero-length leg between service chunks
                E[idx] = 0.0
                idx += 1

    def _leg(minutes: float):
        """Advance from stop idx to idx+segments, inserting laybys between."""
        nonlocal idx
        if minutes <= 0:
            D[idx] = 0.0
            E[idx] = 0.0
            idx += 1
            return
        k = max(1, math.ceil(minutes / grid_min))
        seg = minutes / k / 60.0          # equal segments (exact on 15' data)
        for j in range(k):
            D[idx] = seg
            E[idx] = 0.0
            idx += 1
            if j < k - 1:
                L.append(idx)

    # origin 0 -> loading customer (their node 0) via a zero-length leg
    _leg(0.0)
    _customer(0)

    # their nodes 1..n+1: leg c[j-1] then the node itself
    for j in range(1, n + 2):
        _leg(parsed["c"][j - 1])
        _customer(j)

    j_end = idx            # dummy end customer (their node n+1) — terminal rest
    _leg(0.0)              # zero-length leg to the destination stop N
    I = list(range(idx + 1))

    data = make_data(
        I=I, C=C, K=[], L=L, D=D, E=E,
        Wha=Wha, Whf=Whf,
        S=S_h, Q={}, M_man_h=0.0, M_lay_h=0.0,
        hard_tw=True, allow_wait=allow_wait, wtd_rules=True,
        label=f"R15-PGLT {parsed['name']} (no-night benchmark)",
        title=f"pglt_{parsed['name']}",
    )
    return data, j_end


# ══════════════════════════════════════════════════════════════════════════════
# SOLVE
# ══════════════════════════════════════════════════════════════════════════════

def solve_pglt(data: dict, j_end: int,
               mipgap: float = 0.0, timelimit: int = 600,
               tee: bool = False) -> dict:
    """
    Build, adapt (terminal rest + relaxed start), and solve one instance.

    Returns dict(obj_min, wall_s, status, model, sol).
    obj_min is the completion time in minutes measured from T_START —
    directly comparable to the Agg-NO-Night reference values.
    """
    model = MILP.build_model(data)
    N = data["N"]
    t0 = data["T_START"]

    # Their C0.1: the schedule must END with a daily rest (11 h or reduced 9 h,
    # the reduced one counting against the weekly budget rho_bar).
    model.pglt_terminal_rest = pyo.Constraint(
        expr=model.rho1[j_end] + model.rho2[j_end] == 1)

    # Delayed-start semantics (their X[0] is a variable in [e_0, ...]): the
    # schedule may start after T_START; shift clocks stay at 0 until it does.
    model.init_ta.deactivate()
    model.pglt_start = pyo.Constraint(expr=model.ta[0] >= t0)

    t_wall = time.perf_counter()
    _, status = MILP.solve_model(model, tee=tee,
                                 mipgap=mipgap, timelimit=timelimit)
    wall = time.perf_counter() - t_wall

    obj_min = None
    sol = None
    if status not in ("infeasible",):
        try:
            obj_min = round((pyo.value(model.ta[N]) - t0) * 60.0, 4)
            sol = MILP.extract_solution(model, data)
        except Exception:
            status = f"{status} (no incumbent)"
    return dict(obj_min=obj_min, wall_s=wall, status=status,
                model=model, sol=sol)


def run_one(k: int, grid_min: int = 15, mipgap: float = 0.0,
            timelimit: int = 600, tee: bool = False,
            allow_wait: bool = True) -> dict:
    parsed = parse_pglt(os.path.join(INSTANCE_DIR, f"Test_{k}.txt"))
    data, j_end = build_pglt_data(parsed, grid_min=grid_min,
                                  allow_wait=allow_wait)
    res = solve_pglt(data, j_end, mipgap=mipgap, timelimit=timelimit, tee=tee)
    res["instance"] = k
    res["n_stops"] = data["N"] + 1
    return res


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE-SOLUTION AUDIT (--scan-reference)
# ══════════════════════════════════════════════════════════════════════════════

def scan_reference() -> None:
    """
    Audit their No-Night solution files for the assumptions our mapping makes:
      * all event times / durations on the 15-minute grid,
      * no pure idle time (APO column),
      * no preempted (split) SERVICE activities.
    """
    off_grid, apo_used, svc_split = [], [], []
    for path in sorted(glob.glob(os.path.join(NO_NIGHT_DIR, "TEST_*.txt")),
                       key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
        name = os.path.basename(path).replace(".txt", "")
        svc_rows = {}
        for ln in open(path).read().splitlines():
            parts = ln.split("\t")
            if len(parts) < 7 or not parts[0].strip().isdigit():
                continue
            vals = [p.strip() for p in parts if p.strip() != ""]
            client, act, start, service, btype, full, apo = vals[:7]
            for v in (start, service, btype):
                if float(v) % 15 != 0:
                    off_grid.append((name, ln.strip()))
            if float(apo) > 0:
                apo_used.append((name, ln.strip()))
            a = int(act)
            if a % 2 == 0 and float(service) > 0:      # even act = service node
                svc_rows.setdefault(a, 0)
                svc_rows[a] += 1
        for a, cnt in svc_rows.items():
            if cnt > 1:
                svc_split.append((name, a, cnt))

    print(f"off-grid events   : {len(off_grid)}")
    for x in off_grid[:10]:
        print("   ", x)
    print(f"rows with APO > 0 : {len(apo_used)}")
    for x in apo_used[:10]:
        print("   ", x)
    print(f"split service acts: {len(svc_split)}")
    for x in svc_split[:10]:
        print("   ", x)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_selection(arg: str) -> list[int]:
    if arg.lower() == "all":
        return list(range(1, 41))
    ids = []
    for tok in arg.split(","):
        tok = tok.strip()
        m = re.match(r"(?:Test_)?(\d+)$", tok, re.IGNORECASE)
        if not m:
            raise SystemExit(f"cannot parse instance selector: {tok!r}")
        ids.append(int(m.group(1)))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("instances", nargs="?", default="all",
                    help="'all', comma list of numbers, or Test_k")
    ap.add_argument("--grid", type=int, default=15,
                    help="layby grid in minutes (default 15)")
    ap.add_argument("--mipgap", type=float, default=0.0)
    ap.add_argument("--timelimit", type=int, default=600)
    ap.add_argument("--tee", action="store_true", help="show solver log")
    ap.add_argument("--no-wait", action="store_true",
                    help="disable the idle-wait variable (gaps must be "
                         "bridged by declared breaks/rests, as their "
                         "published solutions do)")
    ap.add_argument("--scan-reference", action="store_true",
                    help="audit their solution files instead of solving")
    ap.add_argument("--out", default=os.path.join("solutions",
                                                  "pglt_comparison.csv"))
    args = ap.parse_args()

    if args.scan_reference:
        scan_reference()
        return

    ref = load_reference()
    ids = _parse_selection(args.instances)

    rows = []
    hdr = (f"{'inst':>4} {'stops':>5} {'ours(min)':>10} {'theirs':>8} "
           f"{'diff':>8} {'ours(s)':>8} {'theirs(s)':>9}  status")
    print(hdr)
    print("-" * len(hdr))
    for k in ids:
        res = run_one(k, grid_min=args.grid, mipgap=args.mipgap,
                      timelimit=args.timelimit, tee=args.tee,
                      allow_wait=not args.no_wait)
        ref_obj, ref_t = ref.get(k, (float("nan"), float("nan")))
        ours = res["obj_min"]
        diff = (ours - ref_obj) if ours is not None else float("nan")
        rows.append(dict(instance=k, n_stops=res["n_stops"],
                         ours_min=ours, theirs_min=ref_obj, diff_min=diff,
                         ours_s=round(res["wall_s"], 2), theirs_s=ref_t,
                         status=res["status"]))
        ours_s = f"{ours:10.1f}" if ours is not None else f"{'—':>10}"
        print(f"{k:>4} {res['n_stops']:>5} {ours_s} {ref_obj:>8.1f} "
              f"{diff:>8.1f} {res['wall_s']:>8.2f} {ref_t:>9.2f}  "
              f"{res['status']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    n_match = sum(1 for r in rows
                  if r["ours_min"] is not None
                  and abs(r["diff_min"]) < 1e-3)
    print(f"\n{n_match}/{len(rows)} exact matches — table written to {args.out}")


if __name__ == "__main__":
    main()
