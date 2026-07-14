"""
experiments/rh_sweep.py — RH5: rolling-horizon hyperparameter sweep
====================================================================
Sweeps the look-ahead policy over T_hor x S on one instance class and
reports, per configuration:

  - mean gap-to-oracle over the swept instances
  - per-stop online decision time (mean / max)  <- the real-time argument:
    at 40–60 km charger spacing the driver needs a decision within minutes,
    so a ~60 s per-stop time is comfortably real-time (paper §7)
  - supervisor interventions and violations

Also supports the RH3 aggregation ablation (mean vs cvar_0.8 vs worst) and
the RH4 LP-vs-MILP agreement table (--solve_mode both).

Usage
-----
  python experiments/rh_sweep.py "instances/RmediumCfewTmedium_*.json" \
      --horizons 12 24 36 --scenarios 5 10 20 --max_instances 3

Results are appended as JSON lines to experiments/rh_sweep_results.jsonl.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob of instance JSON files")
    ap.add_argument("--horizons",  type=float, nargs="+", default=[12, 24, 36])
    ap.add_argument("--scenarios", type=int,   nargs="+", default=[5, 10, 20])
    ap.add_argument("--criteria",  type=str,   nargs="+", default=["mean"],
                    help="RH3 ablation: e.g. mean cvar_0.8 worst")
    ap.add_argument("--solve_mode", type=str, default="lp",
                    choices=["lp", "mip", "both"],
                    help="'both' also produces the RH4 agreement table")
    ap.add_argument("--max_instances", type=int, default=3)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--n_workers",  type=int, default=4)
    ap.add_argument("--out", type=str,
                    default=os.path.join("experiments", "rh_sweep_results.jsonl"))
    args = ap.parse_args()

    from runner_dispatch import run_algorithm

    files = sorted(glob.glob(args.pattern))[: args.max_instances]
    if not files:
        raise SystemExit(f"no instances match '{args.pattern}'")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for H in args.horizons:
        for S in args.scenarios:
            for crit in args.criteria:
                for jf in files:
                    stem = os.path.splitext(os.path.basename(jf))[0]
                    print(f"\n=== H={H}h S={S} crit={crit} {stem} ===")
                    try:
                        res = run_algorithm(
                            json_file=jf, algorithm="LA",
                            n_scenarios=S, horizon_hours=H,
                            time_limit=args.time_limit,
                            n_workers=args.n_workers,
                            solve_mode=args.solve_mode,
                            criterion=crit,
                            verbose=False, oracle_tee=False,
                            run_id=f"sweep_{stem}_H{H:g}_S{S}_{crit}",
                        )
                        met = res.get("metrics", {})
                        ora = res.get("oracle", {})
                        gap = (res["total_time"] - ora.get("obj", float("nan"))
                               if ora.get("feasible") else float("nan"))
                        row = dict(
                            instance=stem, horizon=H, n_scenarios=S,
                            criterion=crit, solve_mode=args.solve_mode,
                            arrival_h=res["total_time"],
                            oracle_h=ora.get("obj"),
                            gap_h=gap,
                            wall_s=res["wall_clock"],
                            dec_time_mean_s=met.get("decision_time_mean_s"),
                            dec_time_max_s=met.get("decision_time_max_s"),
                            n_violations=met.get("n_violations"),
                            n_interventions=met.get("n_interventions"),
                            lp_vs_mip=met.get("lp_vs_mip"),
                        )
                    except Exception as e:
                        row = dict(instance=stem, horizon=H, n_scenarios=S,
                                   criterion=crit, error=f"{type(e).__name__}: {e}")
                    rows.append(row)
                    with open(args.out, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")

    # summary table
    print(f"\n{'H':>5} {'S':>4} {'crit':>9} {'gap(h)':>8} {'dec(s)':>7} "
          f"{'max(s)':>7} {'viol':>5}")
    for r in rows:
        if "error" in r:
            print(f"{r['horizon']:>5} {r['n_scenarios']:>4} "
                  f"{r['criterion']:>9}  ERROR: {r['error']}")
            continue
        print(f"{r['horizon']:>5} {r['n_scenarios']:>4} {r['criterion']:>9} "
              f"{r['gap_h']:>8.3f} {r['dec_time_mean_s']:>7.1f} "
              f"{r['dec_time_max_s']:>7.1f} {r['n_violations']:>5}")
    print(f"\nresults appended to {args.out}")


if __name__ == "__main__":
    main()
