"""
run_la_mixed.py — the look-ahead teacher (LA) on routes whose chargers differ
=============================================================================
Runs src's own LA, unchanged, on the mixed routes of mixed_instances.py.  That
is possible since 2026-10-01, when src learned per-charger charging curves (the
optional instance field "TbarK"; see the "Per-charger charging curves" section
of src/methods/MILP.py): the LA's scenario sub-MILPs, its nominal re-solve and
its vehicle model all read each station's own curve, in the parallel workers
too, because the support is in src itself rather than patched in at run time.

Configuration: exactly that of every stored base-case LA run (the teacher the
students learned from), read off their solution JSONs —

    solve_mode mip   n_scenarios 25   horizon 24 h   criterion mean
    tiebreak 5 min   la_energy_quantile 0.5   prune_quantile None
    unsupervised     8 workers        per-scenario time limit 300 s (default)

Outputs: paths.redirect_outputs() sends every output tree (solutions, logs,
figures) under ML/la_mixed/ before the first write, so no run reaches the main
solutions/ or logs/ — the manuscript's reporting discovers runs by globbing
those by method name.  A run is skipped when its solution is already there,
so a rerun resumes at the next route.

The default route set is the reduced one agreed for a first look (2026-10-01):
8 pmix routes, every short and medium family once, seeds 22-25.  The LA took
11.3 h on their uniform originals.  mixed_eval.py `la` reads the results.

--split train labels TRAINING routes instead (mixed_instances.py --split
train): the 2026-10-01 pilot is 48 short pmix routes, seeds 1-12, run seed by
seed across the four families so that a partial run is still balanced.  Their
logs are what extract.py --physics pmix turns into rows.

    python ML/code/run_la_mixed.py [--variant pmix] [--routes a,b,...]
    python ML/code/run_la_mixed.py --split train      # every built training route
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

ML = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LA_ROOT = os.path.join(ML, "la_mixed")          # redirected output trees
INST = os.path.join(ML, "instances_mixed")

SUBSET = ["RshortCfewTnone_22", "RshortCfewTtight_23", "RshortCmanyTnone_24",
          "RshortCmanyTtight_25", "RmediumCfewTnone_22", "RmediumCfewTtight_23",
          "RmediumCmanyTnone_24", "RmediumCmanyTtight_25"]

# the stored teacher's configuration (see the module docstring)
LA_CONFIG = dict(n_scenarios=25, horizon_hours=24.0, time_limit=300,
                 n_workers=8, solve_mode="mip", criterion="mean",
                 tiebreak_min=5.0, supervised=False, prune_quantile=None,
                 la_energy_quantile=0.5)


def la_solutions(name):
    """Stored LA solutions of one mixed route, oldest first."""
    return sorted(glob.glob(os.path.join(LA_ROOT, "solutions", "**",
                                         f"{name}_LA_*.json"), recursive=True))


def train_routes(variant):
    """Every built training route, seed by seed across families."""
    names = [os.path.splitext(os.path.basename(p))[0].split("__")[0]
             for p in glob.glob(os.path.join(INST, "train", variant, "*.json"))]
    return sorted(names, key=lambda n: (int(n.rsplit("_", 1)[1]), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pmix")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--routes", default=None,
                    help="base route names; the variant suffix is added "
                         "(default: SUBSET for test, every built route for train; "
                         "'all' = every built route of the split)")
    ap.add_argument("--slice", default="0/1",
                    help="i/n: this process's share of the routes, so n processes "
                         "can label one batch in parallel (each skips solved routes)")
    args = ap.parse_args()

    from src import paths
    paths.redirect_outputs(LA_ROOT)          # before anything can write
    from src.simulation.runner_dispatch import run_algorithm

    inst_dir = (os.path.join(INST, "train", args.variant) if args.split == "train"
                else os.path.join(INST, args.variant))
    if args.routes == "all":                 # every built route of this split
        routes = sorted((os.path.splitext(os.path.basename(p))[0].split("__")[0]
                         for p in glob.glob(os.path.join(inst_dir, "*.json"))),
                        key=lambda n: (int(n.rsplit("_", 1)[1]), n))
    else:
        routes = (args.routes.split(",") if args.routes
                  else train_routes(args.variant) if args.split == "train" else SUBSET)
    i, n = (int(x) for x in args.slice.split("/"))
    names = [f"{r}__{args.variant}" for r in routes if r][i::n]
    print(f"[la_mixed] {args.split}: {len(names)} routes (slice {args.slice}), "
          f"outputs under {LA_ROOT}", flush=True)
    for name in names:
        if la_solutions(name):
            print(f"  [skip] {name}", flush=True)
            continue
        path = os.path.join(inst_dir, name + ".json")
        t0 = time.time()
        res = run_algorithm(json_file=path, algorithm="LA", verbose=False,
                            **LA_CONFIG)
        m = (res or {}).get("metrics", {})
        print(f"  {name}: duration {res.get('duration_h')} h  "
              f"infeasible={m.get('run_infeasible')}  tw={m.get('tw_n_misses')}  "
              f"{(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
