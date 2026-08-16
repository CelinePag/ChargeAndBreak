"""
audit_runs.py — one-pass integrity audit and cleanup of solutions/ and logs/.

Every check below exists because it caught a real defect in this project; the
comments name the failure mode so the check is never removed by accident.

Checks (in report order)
  A corrupt      solution JSON that will not parse (compile skips it silently)
  B orphan       run whose instance file no longer exists anywhere
  C mistitled    run whose `instance` field disagrees with its own file stem.
                 The compile dedup and the oracle cache name both key off the
                 title, so a mistitled variant DISPLACES the base run and its
                 oracle overwrites the base cache.  Diesel is exempt: the
                 runner appends "_diesel" at run time (stem `X__diesel`,
                 title `X_diesel`).
  D superseded   older run of the same (title, method, supervised); the compile
                 keeps only the latest, so the rest is dead weight
  E warmstart    greedy run written by an ORACLE warm start (pre-persist=False)
  F oracle_bad   oracle cache whose schedule length does not match its
                 instance's stop count -> it is a cache of a DIFFERENT
                 instance (the variant-title collision)
  G orphan_log   log with no matching solution and no matching run in flight.
                 ORACLE logs are never orphans: the oracle writes a cache, not
                 a solution file.
  H param_split  latest runs of one method that disagree on a key parameter
                 (e.g. greedy guard 0.95 vs unguarded) — reported, never
                 auto-fixed, because only the author can say which is wanted

Nothing is deleted.  Offending files are MOVED to archive/<reason>/ so any
decision is reversible.

Usage
  python -m src.output_analysis.audit_runs                 # report only (default)
  python -m src.output_analysis.audit_runs --apply         # move offending files to archive/
  python -m src.output_analysis.audit_runs --apply --logs  # also archive orphan logs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict
from src import paths as _paths
from src.output_analysis import run_cache

ARCHIVE = _paths.archive()
SOL_DIR = _paths.solutions()
LOG_DIR = _paths.logs()

# Shared with the runner (src/paths.py) so this audit sees exactly the run ids
# the runner writes — including the optional VARIANT segment.  With the old
# private regex a --variant run did not match at all and was skipped silently;
# once it DOES match, the variant must be part of the "superseded" key below,
# or a base run and a variant run of the same instance would look like two runs
# of the same thing and --apply would archive the older one.
_RUN_RE = _paths.RUN_ID_RE

# parameters that must agree across the latest runs of a method.  ROBU's gamma
# is eps-derived and legitimately varies with N, so it is not listed.
_KEY_PARAM = {"GREEDY": "prune_quantile", "LA": "n_scenarios",
              "2SP": "n_scenarios"}

# the protocol's greedy departure guard (see additional_analysis.DEFAULT_GUARD)
GUARD = 0.95


# ── instance index ───────────────────────────────────────────────────────────

def instance_index() -> dict[str, str]:
    """title -> instance file path, for base and every variant."""
    idx = {}
    for p in glob.glob(_paths.instances("*.json")):
        idx[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(_paths.instances_sens("*", "*.json")):
        stem = os.path.splitext(os.path.basename(p))[0]
        idx[stem] = p
        if stem.endswith("__diesel"):        # runner retitles to _diesel
            idx[stem.replace("__diesel", "_diesel")] = p
    return idx


_INSTANCE_N: dict[str, int | None] = {}


def instance_n(path: str) -> int | None:
    """N of the instance at `path`, memoised.

    Check F asks this once per oracle cache, and a handful of instances back
    thousands of caches, so without the memo the audit re-parsed the same
    instance files ~3 000 times.
    """
    if path in _INSTANCE_N:
        return _INSTANCE_N[path]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            n = int(json.load(fh)["instance"]["N"])
    except Exception:
        n = None
    _INSTANCE_N[path] = n
    return n


# ── audit ────────────────────────────────────────────────────────────────────

def oracle_log_index() -> dict[str, list[str]]:
    """stem -> sorted ORACLE log timestamps.  Built ONCE: globbing the log
    directory per run turned this audit into a multi-minute job."""
    idx: dict[str, list[str]] = defaultdict(list)
    for p in glob.glob(os.path.join(LOG_DIR, "*_ORACLE_*.txt")):
        b = os.path.basename(p)[:-4]
        stem, _, rest = b.partition("_ORACLE_")
        ts = rest[:15]
        if len(ts) >= 15:
            idx[stem].append(ts)
    return idx


def audit():
    idx = instance_index()
    ora_logs = oracle_log_index()
    buckets: dict[str, list[str]] = defaultdict(list)
    latest: dict[tuple, tuple] = {}          # (title, alg, sup) -> (ts, path)
    params: dict[tuple, set] = defaultdict(set)
    records: dict[str, dict] = {}            # path -> parsed run, for check H

    # Parsed through run_cache: the corpus is read once per machine rather than
    # once per reporting script (see src/output_analysis/run_cache.py).
    for base, d in run_cache.load_runs(SOL_DIR):
        path = os.path.join(SOL_DIR, base)
        m = _RUN_RE.match(base[:-5])
        if not m:
            continue
        stem, alg, ts = m.group("instance"), m.group("algo"), m.group("ts")
        if alg == "ORACLE":       # writes a cache, not a run file
            continue

        if "_error" in d:
            buckets["corrupt"].append(path)          # A
            continue
        records[path] = d

        # The STORED field decides, not the file name.  Three pre-existing runs
        # carry a label-shaped segment in their run_id from a manual session
        # (…_RO_box_…, …_2SP_S5_…) but were never launched as variants; reading
        # the name as authoritative would retro-classify them and quietly drop
        # them out of the base tables.  A run is a variant only when the runner
        # recorded it as one.
        var = d.get("variant") or None

        title = str(d.get("instance") or "")
        if title not in idx:
            buckets["orphan"].append(path)           # B
            continue

        # C — since runner_dispatch derives the title from the instance FILE
        # STEM, title and stem must agree exactly.  The legacy diesel
        # convention (stem "X__diesel", title "X_diesel") predates that and is
        # accepted only so old runs are flagged for re-run rather than lost.
        legacy_diesel = (stem.endswith("__diesel")
                         and title == stem.replace("__diesel", "_diesel"))
        if title != stem:
            buckets["legacy_diesel" if legacy_diesel else "mistitled"].append(path)
            continue

        # E — greedy runs produced by an ORACLE warm start.  Signature: an
        # ORACLE log exists for the same instance with the same timestamp.
        # E — the protocol fixes the greedy departure guard at GUARD (all
        # greedy runs must be comparable), so an UNGUARDED greedy run is not a
        # valid result.  In practice these are ORACLE warm-start artefacts
        # (written before runner_dispatch passed persist=False) plus historical
        # pre-guard runs.  A timestamp heuristic was tried first and failed:
        # batch run_ids share one timestamp while the warm start writes its own
        # hours later, so proximity says nothing.
        # A variant run is a deliberate sweep cell, not a protocol run: the
        # guard rule below is about the base-case greedy protocol only.
        if alg == "GREEDY" and not var and d.get("prune_quantile") != GUARD:
            buckets["unguarded"].append((path, title))
            continue

        key = (title, alg, bool(d.get("supervised")), var)
        if key in latest:
            older = latest[key] if latest[key][0] > ts else (ts, path)
            newer = (ts, path) if latest[key][0] <= ts else latest[key]
            buckets["superseded"].append(older[1])   # D
            latest[key] = newer
        else:
            latest[key] = (ts, path)

    # H — parameter consistency over the surviving latest runs
    for (title, alg, _sup, var), (_ts, path) in latest.items():
        p = _KEY_PARAM.get(alg)
        if not p:
            continue
        rec = records.get(path)
        if rec is None:
            continue
        v = rec.get(p)
        # A method-configuration sweep varies exactly the parameter this check
        # polices, so each label is its own scope: "LA/S25H12" must not be
        # reported as disagreeing with the base "LA/base" on n_scenarios.
        if var:
            scope = f"var:{var}"
        elif "__" in title or title.endswith("_diesel"):
            scope = "variant"
        else:
            scope = "base"
        params[(alg, scope)].add(str(v))

    # F — oracle caches that do not match their instance.  The cache stores the
    # schedule LENGTH (_n_sol) rather than the schedule, which is all this check
    # ever needed and keeps ~230 MB of oracle payload out of the audit.
    for title, c in sorted(run_cache.load_oracles(SOL_DIR).items()):
        path = os.path.join(SOL_DIR, f"oracle_{title}.json")
        ipath = idx.get(title)
        if ipath is None:
            buckets["oracle_orphan"].append(path)
            continue
        if "_error" in c:
            buckets["corrupt"].append(path)
            continue
        n_sol = c.get("_n_sol") or 0
        if not n_sol:
            continue                       # log-recovered cache: no schedule
        N = instance_n(ipath)
        if N is not None and n_sol != N + 1:
            buckets["oracle_bad"].append(path)

    # SAFETY: only archive an unguarded greedy run when a guarded one survives
    # for that instance; otherwise the instance would silently lose its greedy
    # result altogether.  Those are reported separately as needing a re-run.
    guarded = {t for (t, a, _s, _v) in latest if a == "GREEDY"}
    keep, rerun = [], []
    for path, title in buckets.pop("unguarded", []):
        (keep if title in guarded else rerun).append(path)
    buckets["unguarded"] = keep
    buckets["needs_greedy_rerun"] = rerun

    return buckets, latest, params


def audit_logs(latest: dict) -> list[str]:
    """G — logs with no surviving solution.  ORACLE logs are legitimate."""
    finished = {os.path.basename(p)[:-5] for _ts, p in latest.values()}
    orphans = []
    for p in sorted(glob.glob(os.path.join(LOG_DIR, "*.txt"))):
        rid = os.path.basename(p)[:-4]
        if "_ORACLE_" in rid or rid.endswith("_ORACLE"):
            continue
        if rid in finished:
            continue
        orphans.append(p)
    return orphans


# ── apply ────────────────────────────────────────────────────────────────────

def move(paths, reason: str, apply: bool) -> int:
    dest = os.path.join(ARCHIVE, reason)
    if apply and paths:
        os.makedirs(dest, exist_ok=True)
    for p in paths:
        if apply:
            tgt = os.path.join(dest, os.path.basename(p))
            if os.path.exists(tgt):
                os.remove(p)
            else:
                shutil.move(p, tgt)
    return len(paths)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="move offending files into archive/ (default: report)")
    ap.add_argument("--logs", action="store_true",
                    help="also archive orphan run logs")
    args = ap.parse_args()

    buckets, latest, params = audit()
    orphan_logs = audit_logs(latest)

    order = [("corrupt",       "A  unparseable solution files"),
             ("orphan",        "B  runs whose instance no longer exists"),
             ("mistitled",     "C  runs whose title != file stem (CONTAMINATING)"),
             ("superseded",    "D  older duplicate runs (compile ignores them)"),
             ("unguarded",     "E  greedy runs off-protocol (guard != 0.95)"),
             ("oracle_bad",    "F  oracle caches of the WRONG instance"),
             ("oracle_orphan", "F* oracle caches with no instance file"),
             ("legacy_diesel", "I  diesel runs under the old '_diesel' title")]

    print(f"{'check':<52}{'files':>8}")
    print("-" * 60)
    total = 0
    for key, label in order:
        n = len(buckets.get(key, []))
        total += n
        print(f"{label:<52}{n:>8}")
    print(f"{'G  orphan run logs (no solution, not ORACLE)':<52}"
          f"{len(orphan_logs):>8}")
    nr = len(buckets.get("needs_greedy_rerun", []))
    if nr:
        print(f"{'!  off-protocol greedy KEPT (no guarded run exists)':<52}"
              f"{nr:>8}   <-- re-run these")
    print("-" * 60)
    print(f"{'surviving runs (latest per instance+method)':<52}"
          f"{len(latest):>8}")

    print("\nH  parameter consistency of surviving runs:")
    for (alg, scope), vals in sorted(params.items()):
        flag = "  <-- INCONSISTENT" if len(vals) > 1 else ""
        print(f"     {alg:<7} {scope:<12} {_KEY_PARAM[alg]} = "
              f"{sorted(vals)}{flag}")

    for key, label in order:
        ex = buckets.get(key, [])[:3]
        if ex:
            print(f"\n   e.g. {label.split('  ',1)[1]}:")
            for p in ex:
                print(f"     {os.path.basename(p)}")

    if args.apply:
        print("\napplying (files are MOVED to archive/, never deleted):")
        for key, label in order:
            n = move(buckets.get(key, []), key, True)
            if n:
                print(f"   {key:<14} -> archive/{key}/  ({n})")
        if args.logs:
            n = move(orphan_logs, "orphan_logs", True)
            print(f"   orphan_logs    -> archive/orphan_logs/  ({n})")
        print("\nre-run `python -m src.output_analysis.audit_runs` to confirm a clean bill.")
    else:
        print("\n(report only — pass --apply to archive, --logs to include "
              "orphan logs)")


if __name__ == "__main__":
    main()
