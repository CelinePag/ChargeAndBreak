"""
fix_variant_titles.py — one-off repair for variant instances/runs whose
``title`` still carries the BASE instance name.

Why this matters: every run records ``instance = full_data["title"]``, and both
the compile dedup (instance + method) and the oracle cache file name
(oracle_<title>.json) key off it.  Variant instances materialised before
2026-07-30 kept the base title, so their runs
  * displaced the BASE run of the same method in every table and figure, and
  * would have overwritten the BASE oracle cache once an oracle was run.

Diesel copies are exempt (runner_dispatch._apply_diesel_mode appends its own
"_diesel" suffix at run time, which already separates them).

What it does
  1. retitles every instances_sens/**/<stem>__<tag>.json to its tagged stem
  2. quarantines solution files whose `instance` field disagrees with their
     own file stem (they must be re-run; variant runs are greedy = instant)
  3. reports any oracle cache that a variant run may have overwritten

Usage
  python -m src.misc.fix_variant_titles --dry-run     # report only
  python -m src.misc.fix_variant_titles               # apply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from src import paths as _paths

QUARANTINE = "solutions_quarantine"
_RUN_RE = re.compile(r"^(?P<stem>.+?)_(?P<alg>GREEDY|ROBU|RO|LA|2SP)_"
                     r"\d{8}_\d{6}(?:_\d+)?$")


def retitle_instances(dry: bool) -> int:
    n = 0
    for path in glob.glob(_paths.instances_sens("*", "*.json")):
        stem = os.path.splitext(os.path.basename(path))[0]
        if "__" not in stem or stem.endswith("__diesel"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue
        inst = payload.get("instance", {})
        if inst.get("title") == stem:
            continue
        old = inst.get("title", "")
        if dry:
            print(f"  would retitle {os.path.basename(path)}: "
                  f"'{old}' -> '{stem}'")
        else:
            inst["title"] = stem
            if isinstance(inst.get("label"), str) and old:
                inst["label"] = inst["label"].replace(old, stem, 1)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        n += 1
    return n


def quarantine_runs(dry: bool) -> int:
    if not dry:
        os.makedirs(QUARANTINE, exist_ok=True)
    n = 0
    for path in _paths.glob_solutions("*.json"):
        base = os.path.basename(path)
        if base.startswith("oracle_"):
            continue
        m = _RUN_RE.match(base[:-5])
        if not m:
            continue
        stem = m.group("stem")
        if "__" not in stem or stem.endswith("__diesel"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                inst = str(json.load(fh).get("instance") or "")
        except Exception:
            continue
        if inst == stem:
            continue                       # already correct
        if dry:
            print(f"  would quarantine {base} (instance='{inst}')")
        else:
            shutil.move(path, os.path.join(QUARANTINE, base))
        n += 1
    return n


def suspect_oracles() -> list[str]:
    """Base oracle caches that a variant run could have overwritten: a cache
    whose schedule no longer matches the base instance is impossible to detect
    cheaply, so we just flag caches newer than the first variant run."""
    var = [p for p in _paths.glob_solutions("*.json")
           if "__" in os.path.basename(p)
           and not os.path.basename(p).startswith("oracle_")]
    if not var:
        return []
    first = min(os.path.getmtime(p) for p in var)
    return [os.path.basename(p)
            for p in _paths.glob_solutions("oracle_*.json")
            if "__" not in os.path.basename(p)
            and not os.path.basename(p).endswith("_diesel.json")
            and os.path.getmtime(p) >= first]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("1) variant instance files")
    n_i = retitle_instances(args.dry_run)
    print(f"   {'would retitle' if args.dry_run else 'retitled'}: {n_i}")

    print("2) variant solution files with a base `instance` field")
    n_s = quarantine_runs(args.dry_run)
    print(f"   {'would quarantine' if args.dry_run else 'quarantined'}: {n_s}"
          + ("" if args.dry_run else f"  -> {QUARANTINE}/"))

    sus = suspect_oracles()
    print(f"3) base oracle caches written after the first variant run: "
          f"{len(sus)}")
    for s in sus[:10]:
        print(f"     {s}")
    if len(sus) > 10:
        print(f"     ... and {len(sus) - 10} more")
    print("   (only a concern if a variant ORACLE was ever run; variant "
          "greedy runs never touch these caches)")


if __name__ == "__main__":
    main()
