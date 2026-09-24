"""
evaluate.py — closed-loop comparison: student vs LA / GREEDY / ORACLE
=====================================================================
Offline regret is a diagnostic.  The number that decides whether this works is
what the policy does when it drives the whole route and lives with its own
mistakes: ~88 sequential decisions, each one moving the state distribution
away from the teacher's.

Everything here is read-only with respect to the main tree.  Student runs are
written under ML/results, never into solutions/<bucket>/, because the
reporting pipeline discovers runs by globbing those buckets by method name.

Baselines
---------
  LA      stored duration_h from the same LA_MIPTAIL run the student learned
          from -- the teacher, on the same realisation
  GREEDY  stored duration_h
  ORACLE  stored `obj` from the hindsight solve, minus T_START.  NOTE: on
          instances with time windows `obj` carries the beta*miss penalty, so
          the oracle column is a duration only on Tnone routes; it is reported
          as the objective everywhere and flagged.

A violation ends the run at that stop, so an INFEASIBLE run has no duration
(the repo's own term: `metrics.run_infeasible`, `n_infeasible_excluded`).
Infeasible runs are counted, never averaged, and percentage comparisons use
only routes where BOTH the student and the baseline completed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.instance_gen.instance_io import load_instance_json      # noqa: E402

from policy_core import load_policy, run_student                  # noqa: E402

SOL = os.path.join(_ROOT, "solutions", "basecase")
INST = os.path.join(_ROOT, "instances")
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

SPLITS = dict(fit=range(1, 20), stop=range(20, 22), test=range(22, 26),
              # legacy names, kept so older commands still resolve
              train=range(1, 20), val=range(20, 22))

POL = None   # set by main(), read by report() for the policy counters


def _latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def baselines(inst):
    out = {}
    f = _latest(os.path.join(SOL, f"{inst}_LA_MIPTAIL_*.json"))
    if f:
        with open(f) as fh:
            s = json.load(fh)
        m = s.get("metrics", {})
        out["LA"] = s.get("duration_h")
        out["LA_tw"] = m.get("tw_n_misses", 0)
        out["LA_infeasible"] = bool(m.get("run_infeasible"))
        out["LA_completed"] = s.get("duration_h") is not None
        out["LA_wall_s"] = s.get("wall_clock_s")
        # A rest is 9-11 h, so resting once less is worth ~10% of a 100 h
        # route.  Counting them is the check on any "student beats teacher".
        # NOTE the stored form: "no rest" appears as None, 0, "0" or "none",
        # and the string "0" is TRUTHY -- normalise before counting.
        def _norm(v):
            return None if str(v).lower() in ("0", "none", "-", "") else str(v)
        out["LA_rests"] = sum(1 for a in (s.get("actions") or [])
                              if _norm(a.get("rest_type")) is not None)
        out["LA_charges"] = sum(1 for a in (s.get("actions") or [])
                                if int(a.get("y", 0)) == 1)
    f = _latest(os.path.join(SOL, f"{inst}_GREEDY_*.json"))
    if f:
        with open(f) as fh:
            s = json.load(fh)
        m = s.get("metrics", {})
        out["GREEDY"] = s.get("duration_h")
        out["GREEDY_tw"] = m.get("tw_n_misses", 0)
        out["GREEDY_infeasible"] = bool(m.get("run_infeasible"))
    f = os.path.join(SOL, f"oracle_{inst}.json")
    if os.path.exists(f):
        with open(f) as fh:
            s = json.load(fh)
        if s.get("feasible"):
            out["ORACLE_obj"] = s.get("obj")
            out["ORACLE_gap"] = s.get("gap")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gbt", choices=["gbt", "nn", "clf"],
                    help="which arm: gbt (trees), nn (MLP cost regression), "
                         "clf (MLP classifier — the deleted project's framing)")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--split", default="val", choices=list(SPLITS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--feas-thr", type=float, default=0.5)
    ap.add_argument("--guard-q", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    insts = [str(x) for x in d["instances"]]
    seeds = np.array([int(x.rsplit("_", 1)[1]) for x in insts])
    keep = [i for i, s in zip(insts, seeds) if s in SPLITS[args.split]]
    if args.limit:
        keep = keep[: args.limit]
    print(f"[eval] kind={args.kind} tag={args.tag} on {args.split}: "
          f"{len(keep)} instances")

    pol = load_policy(args.kind, args.tag, feas_thr=args.feas_thr,
                      guard_q=args.guard_q)
    rows = []
    t0 = time.time()
    for i, inst in enumerate(keep):
        fd, D_real, E_real, cv = load_instance_json(
            os.path.join(INST, inst + ".json"))
        fd["_horizon_h"] = 24.0
        r = run_student(fd, D_real, E_real, pol, cv=cv)
        r["instance"] = inst
        r["family"] = inst.rsplit("_", 1)[0]
        r.update(baselines(inst))
        # keep the action MIX, not the sequence: enough to see whether the
        # student buys its time by resting less than the teacher, which is the
        # first thing to suspect whenever a student "beats" its teacher.
        acts = r.pop("actions", [])
        r["action_mix"] = {k: acts.count(k) for k in sorted(set(acts))}
        r["n_rests"] = sum(v for k, v in r["action_mix"].items() if "_r" in k)
        r["n_charges"] = sum(v for k, v in r["action_mix"].items()
                             if k.startswith("y1"))
        rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(keep)}  {time.time()-t0:.0f}s", flush=True)
    wall = time.time() - t0
    print(f"[eval] {len(rows)} routes in {wall:.1f}s\n")

    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or f"eval_{args.tag}_{args.split}.json"
    with open(os.path.join(RESULTS, out), "w") as fh:
        json.dump(rows, fh, indent=1)

    global POL
    POL = pol
    report(rows, args, wall)
    print(f"\n[saved] {os.path.join(RESULTS, out)}")


def report(rows, args, wall):
    n = len(rows)
    comp = [r for r in rows if r["route_completed"]]
    halted = [r for r in rows if not r["route_completed"]]
    print("=" * 74)
    print(f"CLOSED-LOOP RESULT  —  {args.kind}:{args.tag} on {args.split}")
    print("=" * 74)
    print(f"routes                {n}")
    print(f"completed             {len(comp)}")
    print(f"infeasible            {len(halted)}", end="")
    if halted:
        why = {}
        for r in halted:
            why[r["halt_reason"]] = why.get(r["halt_reason"], 0) + 1
        print("   " + ", ".join(f"{k}={v}" for k, v in sorted(why.items())))
    else:
        print()
    ms = np.array([r["ms_per_decision"] for r in rows])
    print(f"latency               {ms.mean():.3f} ms/decision "
          f"(median {np.median(ms):.3f})")
    la_wall = np.array([r["LA_wall_s"] for r in rows
                        if r.get("LA_wall_s")], dtype=float)
    nd = np.array([r["decisions"] for r in rows], dtype=float)
    if len(la_wall):
        la_ms = 1000.0 * la_wall.sum() / nd.sum()
        print(f"teacher latency       {la_ms:.0f} ms/decision "
              f"({la_ms/max(ms.mean(),1e-9):,.0f}x slower)")

    for base in ("LA", "GREEDY"):
        pair = [(r["duration_h"], r[base]) for r in comp
                if r.get(base) is not None and not r.get(f"{base}_infeasible")]
        if not pair:
            continue
        a = np.array([p[0] for p in pair])
        b = np.array([p[1] for p in pair])
        pct = 100.0 * (a - b) / b
        print(f"\nvs {base}  (paired on {len(pair)} routes both completed)")
        print(f"   median  {np.median(pct):+7.2f} %")
        print(f"   mean    {pct.mean():+7.2f} %")
        print(f"   IQR     [{np.quantile(pct,.25):+.2f}, {np.quantile(pct,.75):+.2f}]")
        print(f"   best/worst {pct.min():+.2f} % / {pct.max():+.2f} %")
        print(f"   student faster on {int((pct < 0).sum())}/{len(pct)} routes")

    orc = [(r["duration_h"], r["ORACLE_obj"]) for r in comp
           if r.get("ORACLE_obj") is not None]
    if orc:
        a = np.array([p[0] for p in orc])
        o = np.array([p[1] - 8.0 for p in orc])
        pct = 100.0 * (a - o) / o
        print(f"\nvs ORACLE (hindsight objective - T_START, {len(orc)} routes)")
        print(f"   median  {np.median(pct):+7.2f} %     "
              f"NOTE: obj includes the TW penalty on windowed routes")

    # -- time windows, compared like for like --------------------------------
    BETA = 0.5
    tw_s = sum(r["tw_misses"] for r in rows)
    tw_la = sum(r.get("LA_tw", 0) for r in rows)
    tw_g = sum(r.get("GREEDY_tw", 0) for r in rows)
    ncu = sum(r["n_customers"] for r in rows)
    print(f"\ntime-window misses over {ncu} customer visits")
    print(f"   student {tw_s}   LA {tw_la}   GREEDY {tw_g}")

    # -- penalised objective (duration + beta*misses), the paper's metric -----
    pair = [(r["duration_h"] + BETA * r["tw_misses"],
             r["LA"] + BETA * r.get("LA_tw", 0))
            for r in comp
            if r.get("LA") is not None and not r.get("LA_infeasible")]
    if pair:
        a = np.array([p[0] for p in pair]); b = np.array([p[1] for p in pair])
        pct = 100.0 * (a - b) / b
        print(f"\npenalised objective vs LA (duration + {BETA}*misses, "
              f"{len(pair)} routes)")
        print(f"   median {np.median(pct):+7.2f} %   mean {pct.mean():+7.2f} %")

    # -- is the student buying its time by resting less? ---------------------
    pr = [(r["n_rests"], r.get("LA_rests"), r["n_charges"], r.get("LA_charges"))
          for r in comp if r.get("LA_rests") is not None]
    if pr:
        a = np.array([p[0] for p in pr]); b = np.array([p[1] for p in pr])
        c = np.array([p[2] for p in pr]); e = np.array([p[3] for p in pr])
        print(f"\nrests   student {a.sum():5d}  LA {b.sum():5d}   "
              f"mean {np.mean(a - b):+.3f}/route   "
              f"fewer on {int((a < b).sum())}/{len(pr)} routes")
        print(f"charges student {c.sum():5d}  LA {e.sum():5d}   "
              f"mean {np.mean(c - e):+.3f}/route")

    # -- where the tail lives ------------------------------------------------
    per = {}
    for r in comp:
        if r.get("LA") and not r.get("LA_infeasible"):
            per.setdefault(r["family"], []).append(
                100.0 * (r["duration_h"] - r["LA"]) / r["LA"])
    if per:
        worst = sorted(per.items(), key=lambda kv: -np.median(kv[1]))[:5]
        print("\nworst 5 families (median % vs LA)")
        for fam, v in worst:
            print(f"   {fam:26s} n={len(v):3d}  median {np.median(v):+6.2f} %"
                  f"  worst {max(v):+6.2f} %")

    # -- the teacher's own failures on the same instances --------------------
    la_inf = sum(1 for r in rows if r.get("LA_infeasible"))
    la_none = sum(1 for r in rows if r.get("LA") is None)
    print(f"\nteacher on these routes: infeasible {la_inf}, no duration {la_none}")
    if POL is not None:
        print(f"policy: forcing left 1 action {POL.n_forced}x, left none "
              f"{POL.n_empty}x, charge clamp moved {POL.n_clamped}x")


if __name__ == "__main__":
    main()
