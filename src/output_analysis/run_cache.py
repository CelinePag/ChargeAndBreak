"""
================================================================================
 run_cache.py — parsed-once index of the solutions/ corpus
================================================================================

WHY THIS EXISTS
---------------
The reporting pipeline walks solutions/ end to end six times in a row:

    audit_runs -> coverage_report -> compile_solutions -> paper_figures
    -> additional_analysis la-report -> additional_figures

Each of those is a separate process, so each one re-read and re-parsed the whole
corpus: ~11 400 run files (~680 MB) plus ~3 350 oracle caches (~230 MB).  A full
parse costs ~1 minute, and the audit paid it twice over (it re-opened every
surviving run for the parameter check and re-opened an instance file per oracle).

Two facts make almost all of that work avoidable:

  1. A solution file is WRITE-ONCE.  The runner never edits a run it has already
     written, so a file's content is pinned by (mtime, size): if those agree with
     what we saw last time, the parse result is still valid.

  2. Reporting reads only the scalar header and the `metrics` block.  The
     trajectory arrays -- sim_trajectory, durations_list, actions, td_list,
     D_actual_list -- and the scenario_summary are 97 % of the bytes and are
     touched by NO reporting module (they are consumed by src/plot/plots.py,
     which loads its handful of files directly and does not come through here).

So each file is parsed once, stripped to the keys reporting actually reads, and
kept in a pickle keyed by (name, mtime_ns, size).  A repeat pass costs one
directory scan plus one pickle load instead of ~900 MB of JSON.

WHAT THE CALLER GETS
--------------------
`load_runs()` returns the same dicts `json.load` produced, MINUS the dropped
keys listed in ``DROPPED_RUN_KEYS`` -- so anything that needs a trajectory must
open the file itself (nothing in reporting does).

`load_oracles()` returns one record per instance with the oracle's scalars plus
the three schedule facts reporting derives from `sol`:

    _n_sol   len(sol)                    -- audit's "cache matches instance" check
    _ta_N    sol[-1]["ta"]               -- oracle arrival, for the gap
    _misses  sum of the delta indicators -- oracle time-window misses

`sol` itself is dropped: it is ~70 KB per oracle and those three numbers are all
the gap computation ever used.

CACHE INVALIDATION
------------------
The cache is a pure derivative of solutions/ and is safe to delete at any time;
the next run rebuilds it.  Entries whose (mtime, size) no longer match are
re-parsed and entries whose file is gone are dropped, so an overwritten or
deleted run is picked up automatically.  Bumping CACHE_VERSION (do this whenever
DROPPED_RUN_KEYS or the oracle record shape changes) invalidates it wholesale.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

from src import paths as _paths

# Bump when the stored record shape changes (see module docstring).
CACHE_VERSION = 1

CACHE_PATH = _paths.data_output(".run_cache.pkl")

# Payload keys reporting never reads.  Dropping them shrinks the cache from
# ~900 MB of JSON to ~20 MB of pickle and keeps the whole corpus comfortably in
# memory.  Verified against every consumer of load_solutions(): none of
# audit_runs, coverage_report, compile_solutions, paper_figures or
# additional_analysis references any of these names.
DROPPED_RUN_KEYS = frozenset({
    "sim_trajectory",     # per-event state trace   (~35 % of bytes)
    "durations_list",     # per-leg travel times    (~30 %)
    "scenario_summary",   # per-scenario draw stats (~16 %)
    "actions",            # per-stop decisions      (~12 %)
    "td_list",            # departure times         (~3 %)
    "D_actual_list",      # realised leg distances  (~2 %)
})

# Parse in worker processes only when there is enough to amortise spawning them
# (Windows spawns a fresh interpreter per worker, ~0.4 s each).
_PARALLEL_MIN_FILES = 400
_MAX_WORKERS = min(8, os.cpu_count() or 1)


# ══════════════════════════════════════════════════════════════════════════════
# PARSERS  (module level: ProcessPoolExecutor must be able to pickle them)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_run(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    return {k: v for k, v in data.items() if k not in DROPPED_RUN_KEYS}


def _parse_oracle(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    sol = data.get("sol") or []
    rec = {k: v for k, v in data.items() if k != "sol"}
    rec["_n_sol"]  = len(sol)
    rec["_ta_N"]   = sol[-1].get("ta") if sol else None
    rec["_misses"] = sum(int(s.get("delta") or 0) for s in sol) if sol else None
    return rec


_PARSERS = {"runs": _parse_run, "oracles": _parse_oracle}


def _parse_one(job: tuple[str, str, str]) -> tuple[str, str, dict]:
    """(kind, name, path) -> (kind, name, record).  The worker entry point."""
    kind, name, path = job
    return kind, name, _PARSERS[kind](path)


# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _scan(directory: str) -> tuple[dict, dict, dict]:
    """-> (runs, oracles, where), the first two name -> (mtime_ns, size).

    One scandir pass per directory: on Windows the size/mtime come back with
    the directory entry, so this is a cheap sweep rather than 15 000 stat
    calls.  solutions/ is split into experiment buckets, so the sweep covers
    the tree root and each bucket; `where` carries name -> full path so the
    caller can open a file it only knows the basename of.

    Names stay bucket-free on purpose.  A run_id is unique across the whole
    corpus, so keying the cache on the basename means moving a file between
    buckets does not invalidate its parsed record — only editing it does.
    """
    runs: dict[str, tuple[int, int]] = {}
    oracles: dict[str, tuple[int, int]] = {}
    where: dict[str, str] = {}
    for d in [directory] + [os.path.join(directory, b) for b in _paths.BUCKETS]:
        try:
            with os.scandir(d) as it:
                for e in it:
                    if not e.name.endswith(".json"):
                        continue
                    try:
                        st = e.stat()
                    except OSError:
                        continue
                    key = (st.st_mtime_ns, st.st_size)
                    where[e.name] = e.path
                    if e.name.startswith("oracle_"):
                        oracles[e.name] = key
                    else:
                        runs[e.name] = key
        except OSError:
            continue
    return runs, oracles, where


def _read_cache() -> dict:
    try:
        with open(CACHE_PATH, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("version") == CACHE_VERSION and blob.get("dir"):
            return blob
    except Exception:
        pass
    return {"version": CACHE_VERSION, "dir": None, "runs": {}, "oracles": {}}


def _write_cache(blob: dict) -> None:
    """Atomic replace, so an interrupted write cannot leave a torn cache."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CACHE_PATH) or ".",
                                   prefix=".run_cache.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, CACHE_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        # A cache we cannot persist is a performance problem, never a
        # correctness one: the data returned this run is already complete.
        print(f"  NOTE: could not write {CACHE_PATH}: {e}", file=sys.stderr)


_MEM: dict = {}          # solutions_dir -> {"runs": {...}, "oracles": {...}}

# Re-keyed views of a cache blob, built on first use.  Kept OUT of the blob so
# that what gets pickled stays exactly the on-disk record set.  Keyed by the
# blob's identity, which _MEM pins for the life of the process.
_VIEWS: dict = {}


def refresh(solutions_dir: str | None = None, *, quiet: bool = False) -> dict:
    """Bring the cache in line with solutions_dir and return it.

    Memoised per process, so the several call sites inside one script (the
    audit reads runs and oracles; compile reads runs then annotates from
    oracles) share a single scan.
    """
    solutions_dir = os.path.abspath(solutions_dir or _paths.solutions())
    hit = _MEM.get(solutions_dir)
    if hit is not None:
        return hit

    disk_runs, disk_oracles, disk_where = _scan(solutions_dir)
    blob = _read_cache()
    if blob.get("dir") != solutions_dir:      # cache belongs to another tree
        blob = {"version": CACHE_VERSION, "dir": solutions_dir,
                "runs": {}, "oracles": {}}

    jobs: list[tuple[str, str, str]] = []
    n_gone = 0
    for kind, disk in (("runs", disk_runs), ("oracles", disk_oracles)):
        have = blob[kind]
        for name, key in disk.items():
            entry = have.get(name)
            if entry is None or entry[0] != key:
                jobs.append((kind, name, disk_where[name]))
        # forget files that were deleted since the last pass
        for gone in [n for n in have if n not in disk]:
            del have[gone]
            n_gone += 1

    if jobs:
        if not quiet:
            print(f"  run_cache: parsing {len(jobs)} new/changed file(s) "
                  f"({len(disk_runs)} runs + {len(disk_oracles)} oracles on disk)")
        keys = {"runs": disk_runs, "oracles": disk_oracles}
        for kind, name, rec in _run_jobs(jobs):
            blob[kind][name] = (keys[kind][name], rec)

    if jobs or n_gone:
        # deletions alone still change the cache, so persist those too rather
        # than carrying dead entries until the next file happens to appear
        _write_cache(blob)
    elif not quiet:
        print(f"  run_cache: {len(disk_runs)} runs + {len(disk_oracles)} "
              f"oracles up to date")

    _MEM[solutions_dir] = blob
    return blob


def _run_jobs(jobs: list[tuple[str, str, str]]):
    """Parse `jobs`, spreading them over processes when it is worth it."""
    if len(jobs) < _PARALLEL_MIN_FILES or _MAX_WORKERS < 2:
        for job in jobs:
            yield _parse_one(job)
        return
    try:
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            # A chunk per worker-batch: the per-item IPC round trip dominates
            # otherwise, since a stripped record is only ~2 KB.
            chunk = max(16, len(jobs) // (_MAX_WORKERS * 8))
            yield from pool.map(_parse_one, jobs, chunksize=chunk)
    except Exception as e:      # no fork/spawn available, sandbox limits, ...
        print(f"  NOTE: parallel parse unavailable ({e}); falling back",
              file=sys.stderr)
        for job in jobs:
            yield _parse_one(job)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def load_runs(solutions_dir: str | None = None, *,
              quiet: bool = False) -> list[tuple[str, dict]]:
    """-> [(file_name, record)] for every non-oracle solutions/*.json.

    Records that failed to parse carry an "_error" key and are returned too, so
    callers keep reporting corrupt files instead of silently losing them.
    Sorted by name, matching the previous ``sorted(os.listdir(...))`` order.
    """
    blob = refresh(solutions_dir, quiet=quiet)
    return [(name, rec) for name, (_key, rec) in sorted(blob["runs"].items())]


def runs_by_name(solutions_dir: str | None = None, *,
                 quiet: bool = False) -> dict[str, dict]:
    """-> file_name -> record, for callers that resolve a file name first."""
    blob = refresh(solutions_dir, quiet=quiet)
    view = _VIEWS.get(("by_name", id(blob)))
    if view is None:
        view = _VIEWS[("by_name", id(blob))] = {
            name: rec for name, (_key, rec) in blob["runs"].items()}
    return view


def load_oracles(solutions_dir: str | None = None, *,
                 quiet: bool = False) -> dict[str, dict]:
    """-> instance title -> oracle record (see module docstring for the shape).

    Keyed by the instance the cache belongs to, i.e. oracle_<instance>.json
    without the prefix and suffix.  The view is built once and handed back on
    every call: the gap annotation asks for it per RUN, so rebuilding a
    3 000-entry dict each time cost more than the parsing this class replaced.
    """
    blob = refresh(solutions_dir, quiet=quiet)
    view = _VIEWS.get(("by_instance", id(blob)))
    if view is None:
        view = _VIEWS[("by_instance", id(blob))] = {
            name[len("oracle_"):-len(".json")]: rec
            for name, (_key, rec) in blob["oracles"].items()}
    return view


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Build or clear the solutions/ parse cache")
    ap.add_argument("--dir", default=_paths.solutions())
    ap.add_argument("--clear", action="store_true",
                    help="delete the cache file and exit")
    args = ap.parse_args()

    if args.clear:
        try:
            os.unlink(CACHE_PATH)
            print(f"  removed {CACHE_PATH}")
        except FileNotFoundError:
            print(f"  no cache at {CACHE_PATH}")
        return

    blob = refresh(args.dir)
    print(f"  {len(blob['runs'])} runs, {len(blob['oracles'])} oracles cached "
          f"in {CACHE_PATH}")


if __name__ == "__main__":
    main()
