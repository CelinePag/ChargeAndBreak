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
import re
from collections import defaultdict
from src import paths as _paths
from src.output_analysis import run_cache

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


def _variant_of(rec: dict, stem: str) -> str | None:
    """The run's method-configuration label, or None.

    The label counts only when the run actually recorded it — a handful of
    legacy manual run_ids look labelled but predate the flag entirely — so the
    name is screened first and the stored field decides.
    """
    if not (_paths.parse_run_id(stem) or {}).get("variant"):
        return None
    return rec.get("variant") or None


def scan(sol_dir: str, variants: bool):
    """-> runs[(cls, alg)] = {seed: n_files}, params[(cls,alg)] = set, bad[..]"""
    runs    = defaultdict(lambda: defaultdict(int))
    params  = defaultdict(set)
    bad     = defaultdict(int)
    latest  = {}          # (cls,alg,seed) -> (ts, param value)
    # Parsed through run_cache (see src/output_analysis/run_cache.py).
    for base, d in run_cache.load_runs(sol_dir):
        stem = base[:-5]
        # A method-configuration sweep (--variant) runs on the BASE instances,
        # so its stem carries no "__tag": without this leg it would be counted
        # as base coverage and its differing n_scenarios would show up as a
        # parameter inconsistency of the base case.  The label in the file name
        # is only a candidate — the STORED `variant` field decides, so the few
        # legacy run_ids that merely look labelled (…_RO_box_…) keep counting as
        # base coverage exactly as before.
        labelled = _variant_of(d, stem)
        tagged = ("__" in stem) or ("_diesel_" in base) or bool(labelled)
        if tagged != variants:
            continue
        m = _RUN_RE.match(stem.replace("__", "@@").split("@@")[0]
                          if variants else stem)
        if not m:
            # a labelled run on a base instance: strip the label so the
            # class/seed still parse out of the remaining id
            if labelled:
                m = _RUN_RE.match(stem.replace(f"_{labelled}_", "_", 1))
            if not m:
                continue
        cls  = re.match(r"(R[a-z]+C[a-z]+T[a-z]+)_", m.group("inst")).group(1)
        alg  = m.group("alg")
        seed = int(m.group("inst").rsplit("_", 1)[1])
        key  = (cls, alg)
        if "_error" in d:                       # unparseable file
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
    for b, d in run_cache.load_oracles(sol_dir).items():
        if "__" in b or b.endswith("_diesel"):
            continue
        m = re.match(r"(R[a-z]+C[a-z]+T[a-z]+)_(\d+)$", b)
        if not m:
            continue
        # _n_sol is the cached len(sol): "has a schedule" is all this asked.
        if d.get("feasible") and d.get("_n_sol"):
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

    # One cell = one column, separated by '|'.  The dup/flag suffixes used to
    # be appended as space-separated tokens ("50/50  +50"), which read as an
    # extra column and made every method after greedy — which carries a "+k"
    # on every row — appear shifted one place left.  Suffixes are now bracketed
    # inside their own cell and the separators make the boundaries explicit.
    CW = 15                                   # per-method cell width
    hdr = (f"{'class':<22}" + "|".join(f"{_MLBL[m]:>{CW}}" for m in METHODS)
           + "|" + f"{'oracle':>{CW}}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    tot = defaultdict(lambda: [0, 0, 0, 0])
    for cls in _classes():
        cells = []
        for alg in METHODS:
            seen = runs.get((cls, alg), {})
            n    = len(seen)
            dup  = sum(v - 1 for v in seen.values())
            miss = len(want - set(seen))
            mx   = len(params.get((cls, alg), set())) > 1
            bd   = bad.get((cls, alg), 0)
            flag = ("!" if mx else "") + ("x" if bd else "")
            sfx  = (f"+{dup}" if dup else "") + flag
            cells.append(f"{n}/{args.seeds}" + (f"({sfx})" if sfx else ""))
            t = tot[alg]
            t[0] += n; t[1] += dup; t[2] += miss; t[3] += bd
            rows.append([cls, _MLBL[alg], n, dup, miss, bd,
                         ";".join(sorted(params.get((cls, alg), set())))])
        cells.append(f"{len(ora.get(cls, set()))}/{args.seeds}")
        print(f"{cls:<22}" + "|".join(f"{c:>{CW}}" for c in cells))
    print("-" * len(hdr))
    tot_n = args.seeds * 36
    print(f"{'TOTAL runs':<22}"
          + "|".join(f"{str(tot[m][0]) + '/' + str(tot_n):>{CW}}"
                     for m in METHODS)
          + "|" + f"{str(sum(len(v) for v in ora.values())) + '/' + str(tot_n):>{CW}}")
    print(f"{'TOTAL dup files':<22}"
          + "|".join(f"{tot[m][1]:>{CW}}" for m in METHODS))
    print(f"{'TOTAL corrupt':<22}"
          + "|".join(f"{tot[m][3]:>{CW}}" for m in METHODS))
    print(f"\nlegend: n/{args.seeds} distinct instances run; suffixes in "
          "brackets belong to THAT cell:\n"
          "        (+k) k superseded duplicate files, (!) runs of this method "
          "disagree on\n"
          f"        { {k: v for k, v in _KEY_PARAM.items() if v} }, "
          "(x) corrupt file")

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
