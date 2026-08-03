"""
resync_oracle.py — Refresh the oracle snapshot embedded in method solution files
================================================================================
The gap-to-oracle reported by compile_solutions.py / paper_figures.py is NOT a
stored number: it is recomputed on the fly from an ``oracle`` block that
finalize_run copies into every method's solution JSON at run time.  So after you
RE-SOLVE an oracle (updating solutions/oracle_<instance>.json), the already-run
method files still carry the OLD oracle snapshot and their gaps do not change.

This script copies each fresh oracle cache back into the ``oracle`` block of the
matching method solution files, so the gap is recomputed against the new oracle
WITHOUT re-running the method's simulation.  The oracle is deterministic on the
instance's fixed realised travel times and identical across methods, so this
substitution is exact.

A method file is only rewritten when its embedded oracle.obj / optimal / status
actually differs from the cache, so re-runs are idempotent and touch nothing
that is already in sync.

Usage
-----
  python -m src.misc.resync_oracle                 # dry run: report what WOULD change
  python -m src.misc.resync_oracle --apply         # actually rewrite the method files
  python -m src.misc.resync_oracle --apply --instance RlongCfewTlarge_12
  python -m src.misc.resync_oracle --apply --route long      # only long-route instances
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from src import paths as _paths


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _oracle_cache_path(sol_dir, instance):
    return os.path.join(sol_dir, f"oracle_{instance}.json")


def _differs(embedded: dict, cache: dict) -> bool:
    """True when the embedded oracle snapshot is out of date vs the cache."""
    if not isinstance(embedded, dict):
        return True
    for k in ("obj", "optimal", "status"):
        if embedded.get(k) != cache.get(k):
            return True
    # sol drives the penalised-gap / TW-miss terms; compare length + last arrival
    e_sol, c_sol = embedded.get("sol") or [], cache.get("sol") or []
    if len(e_sol) != len(c_sol):
        return True
    if e_sol and c_sol and e_sol[-1].get("ta") != c_sol[-1].get("ta"):
        return True
    return False


def resync(sol_dir: str, apply: bool,
           instance_filter: str | None, route_filter: str | None):
    caches = {}   # instance -> cache dict
    for f in os.listdir(sol_dir):
        if f.startswith("oracle_") and f.endswith(".json"):
            inst = f[len("oracle_"):-len(".json")]
            try:
                caches[inst] = _load(os.path.join(sol_dir, f))
            except Exception as e:
                print(f"  SKIP cache {f}: {e}", file=sys.stderr)

    n_scanned = n_update = n_nocache = n_insync = 0
    per_instance = {}

    for f in sorted(os.listdir(sol_dir)):
        if not f.endswith(".json") or f.startswith("oracle_"):
            continue
        path = os.path.join(sol_dir, f)
        try:
            d = _load(path)
        except Exception as e:
            print(f"  SKIP {f}: {e}", file=sys.stderr)
            continue
        inst = d.get("instance")
        if not inst:
            continue
        if instance_filter and inst != instance_filter:
            continue
        if route_filter and not inst.startswith("R" + route_filter):
            continue

        n_scanned += 1
        cache = caches.get(inst)
        if cache is None:
            n_nocache += 1
            continue
        if not _differs(d.get("oracle"), cache):
            n_insync += 1
            continue

        n_update += 1
        per_instance.setdefault(inst, 0)
        per_instance[inst] += 1
        if apply:
            d["oracle"] = cache
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=2)
            os.replace(tmp, path)

    verb = "Rewrote" if apply else "Would rewrite"
    print(f"  Scanned {n_scanned} method file(s); {n_insync} already in sync, "
          f"{n_nocache} with no oracle cache.")
    print(f"  {verb} {n_update} method file(s) across "
          f"{len(per_instance)} instance(s).")
    if per_instance:
        for inst in sorted(per_instance):
            print(f"    {inst:30} {per_instance[inst]} file(s)")
    if not apply and n_update:
        print("\n  (dry run — re-run with --apply to write, then re-run "
              "compile_solutions.py / paper_figures.py to refresh the gaps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Refresh the oracle block embedded in method solution "
                    "files from the re-solved oracle caches, so gap-to-oracle "
                    "updates without re-running the methods.")
    p.add_argument("--dir", default=_paths.solutions(),
                   help="solutions directory (default: solutions)")
    p.add_argument("--apply", action="store_true", default=False,
                   help="actually rewrite files (default: dry run)")
    p.add_argument("--instance", default=None,
                   help="only this instance id (e.g. RlongCfewTlarge_12)")
    p.add_argument("--route", default=None, choices=["short", "medium", "long"],
                   help="only instances of this route class")
    args = p.parse_args()
    resync(args.dir, args.apply, args.instance, args.route)
