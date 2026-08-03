"""
coverage_report.py — what has been run, what is missing, what is duplicated.

Scans solutions/ and prints (and optionally writes) a per-(instance class,
method) inventory:

  runs      distinct instances with at least one finished run  (of --seeds)
  dup       superseded extra files for those instances (compile keeps the
            latest per (instance, method, supervised) — the rest are dead
            weight, NOT an error)
  miss      seeds with no run at all
  bad       solution files that fail to parse (must be deleted / re-run)
  mixed     runs of the same method that disagree on a key parameter
            (e.g. greedy guard 0.95 vs unguarded) — these WILL bias the
            tables because the dedup key ignores solver parameters

Base coverage deliberately ignores variant runs (tagged '__cs30', '_diesel',
...) — pass --variants to inventory those instead.

Usage
  python -m src.output_analysis.coverage_report                      # base case, seeds 1-50
  python -m src.output_analysis.coverage_report --seeds 10           # only the first 10 seeds
  python -m src.output_analysis.coverage_report --csv coverage.csv   # also write a CSV
  python -m src.output_analysis.coverage_report --variants           # variant/tagged runs
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from src import paths as _paths

METHODS = ["GREEDY", "RO", "ROBU", "LA", "2SP"]
_MLBL   = {"GREEDY": "greedy", "RO": "RO", "ROBU": "ROBU", "LA": "LA",
           "2SP": "2SP"}
ROUTES  = ["Rshort", "Rmedium", "Rlong"]
CUSTS   = ["Cfew", "Cmedium", "Cmany"]
TWS     = ["tight", "medium", "large", "none"]

# parameter whose disagreement across runs of one method silently biases the
# tables (the dedup key does not include it)
# ROBU's gamma is eps-derived and legitimately varies with N, so it is NOT
# checked.  Only parameters that are supposed to be constant across a class.
_KEY_PARAM = {"GREEDY": "prune_quantile", "LA": "n_scenarios",
              "2SP": "n_scenarios", "ROBU": None, "RO": None}

_RUN_RE = re.compile(
    r"^(?P<inst>R[a-z]+C[a-z]+T[a-z]+_\d+)_(?P<alg>GREEDY|ROBU|RO|LA|2SP)_"
    r"(?P<ts>\d{8}_\d{6})(?:_(?P<idx>\d+))?$")


def _classes():
    return [f"{r}{c}T{t}" for r in ROUTES for c in CUSTS for t in TWS]


def scan(sol_dir: str, variants: bool):
    """-> runs[(cls, alg)] = {seed: n_files}, params[(cls,alg)] = set, bad[..]"""
    runs    = defaultdict(lambda: defaultdict(int))
    params  = defaultdict(set)
    bad     = defaultdict(int)
    latest  = {}          # (cls,alg,seed) -> (ts, param value)
    for path in glob.glob(os.path.join(sol_dir, "*.json")):
        base = os.path.basename(path)
        if base.startswith("oracle_"):
            continue
        stem = base[:-5]
        tagged = ("__" in stem) or ("_diesel_" in base)
        if tagged != variants:
            continue
        m = _RUN_RE.match(stem.replace("__", "@@").split("@@")[0]
                          if variants else stem)
        if not m:
            continue
        cls  = re.match(r"(R[a-z]+C[a-z]+T[a-z]+)_", m.group("inst")).group(1)
        alg  = m.group("alg")
        seed = int(m.group("inst").rsplit("_", 1)[1])
        key  = (cls, alg)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            bad[key] += 1
            continue
        if d.get("sim_arrival_h") is None:      # started but never finished
            continue
        runs[key][seed] += 1
        p = _KEY_PARAM.get(alg)
        if p:
            lk = (cls, alg, seed)
            ts = m.group("ts")
            if lk not in latest or ts > latest[lk][0]:
                latest[lk] = (ts, str(d.get(p)))
    for (cls, alg, _seed), (_ts, val) in latest.items():
        params[(cls, alg)].add(val)
    return runs, params, bad


def oracle_coverage(sol_dir: str):
    have = defaultdict(set)
    for path in glob.glob(os.path.join(sol_dir, "oracle_R*.json")):
        b = os.path.basename(path)[7:-5]
        if "__" in b or b.endswith("_diesel"):
            continue
        m = re.match(r"(R[a-z]+C[a-z]+T[a-z]+)_(\d+)$", b)
        if not m:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if d.get("feasible") and d.get("sol"):
            have[m.group(1)].add(int(m.group(2)))
    return have


def main() -> None:
    ap = argparse.ArgumentParser(description="Run coverage / duplicate report")
    ap.add_argument("--dir", default=_paths.solutions())
    ap.add_argument("--seeds", type=int, default=50,
                    help="expected seeds per class (default 50)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--variants", action="store_true",
                    help="inventory tagged variant runs instead of base runs")
    args = ap.parse_args()

    runs, params, bad = scan(args.dir, args.variants)
    ora  = oracle_coverage(args.dir)
    want = set(range(1, args.seeds + 1))

    hdr = (f"{'class':<22}" + "".join(f"{_MLBL[m]:>16}" for m in METHODS)
           + f"{'oracle':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    tot = defaultdict(lambda: [0, 0, 0, 0])
    for cls in _classes():
        line = f"{cls:<22}"
        for alg in METHODS:
            seen = runs.get((cls, alg), {})
            n    = len(seen)
            dup  = sum(v - 1 for v in seen.values())
            miss = len(want - set(seen))
            mx   = len(params.get((cls, alg), set())) > 1
            bd   = bad.get((cls, alg), 0)
            flag = ("!" if mx else "") + ("x" if bd else "")
            line += f"{n:>4}{'/'+str(args.seeds):<4}{'+'+str(dup) if dup else '  ':>4}{flag:<4}"
            t = tot[alg]
            t[0] += n; t[1] += dup; t[2] += miss; t[3] += bd
            rows.append([cls, _MLBL[alg], n, dup, miss, bd,
                         ";".join(sorted(params.get((cls, alg), set())))])
        no = len(ora.get(cls, set()))
        line += f"{no:>5}/{args.seeds:<4}"
        print(line)
    print("-" * len(hdr))
    print(f"{'TOTAL runs':<22}" + "".join(
        f"{tot[m][0]:>4}{'/'+str(args.seeds*36):<12}" for m in METHODS)
        + f"{sum(len(v) for v in ora.values()):>5}/{args.seeds*36}")
    print(f"{'TOTAL dup files':<22}" + "".join(
        f"{tot[m][1]:>4}{'':<12}" for m in METHODS))
    print(f"{'TOTAL corrupt':<22}" + "".join(
        f"{tot[m][3]:>4}{'':<12}" for m in METHODS))
    print("\nlegend: n/50 distinct instances run, +k superseded duplicate "
          "files,\n        '!' = runs of this method disagree on "
          f"{ {k: v for k, v in _KEY_PARAM.items() if v} }, 'x' = corrupt file")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["class", "method", "n_instances", "n_duplicate_files",
                        "n_missing_seeds", "n_corrupt_files",
                        "distinct_param_values"])
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
