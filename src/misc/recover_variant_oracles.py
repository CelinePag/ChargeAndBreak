"""
recover_variant_oracles.py — repair the fallout of the variant-title bug.

Variant instances materialised before 2026-07-30 kept the BASE instance title,
so every variant ORACLE run wrote its result to solutions/oracle_<BASE>.json:

  * the variant's own cache was never created, and
  * the BASE oracle cache was overwritten with a variant result
    (cs-spacing variants change the geometry; charger-power variants keep it,
    so the damage is invisible to any stop-count check).

The per-run .txt logs survived intact, and they carry the objective, bound and
gap — so ~1150 solves do NOT need to be repeated for the duration-based
sensitivity deltas.  Only the schedule (`sol`) is unrecoverable, so the
recovered caches are marked ``sol_unavailable`` and can be used for objectives
but not for per-stop quantities (e.g. the coupling fraction).

Actions
  1. quarantine every BASE oracle cache that a variant run may have written
     (they must be re-solved: `runner_dispatch.py <instances> ORACLE`)
  2. rebuild solutions/oracle_<stem>__<tag>.json from the variant logs

Usage
  python -m src.misc.recover_variant_oracles --dry-run
  python -m src.misc.recover_variant_oracles
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from src import paths as _paths

QUARANTINE = "oracle_quarantine"
COMBOS = ["RshortCfew", "RshortCmany", "RmediumCfew", "RmediumCmany"]
TWS    = ["tight", "medium", "large", "none"]
SEEDS  = range(1, 11)

_OBJ  = re.compile(r"Oracle arrival\s*:\s*([0-9.]+)\s*h")
_GAP  = re.compile(r"gap=([0-9.eE+-]+)")
_BND  = re.compile(r"best_bound=([0-9.eE+-]+)")
_STOP = re.compile(r"stop_reason=([a-z_]+)")
_LOG  = re.compile(r"^(?P<stem>.+?__(?P<tag>[a-z0-9.]+))_ORACLE_\d{8}_\d{6}"
                   r"(?:_\d+)?\.txt$")


def quarantine_base(dry: bool) -> int:
    if not dry:
        os.makedirs(QUARANTINE, exist_ok=True)
    n = 0
    for cb in COMBOS:
        for tw in TWS:
            for s in SEEDS:
                p = _paths.solutions(f"oracle_{cb}T{tw}_{s}.json")
                if not os.path.exists(p):
                    continue
                if dry:
                    print(f"  would quarantine {os.path.basename(p)}")
                else:
                    shutil.move(p, os.path.join(QUARANTINE,
                                                os.path.basename(p)))
                n += 1
    return n


def rebuild_variants(dry: bool) -> tuple[int, int]:
    best: dict[str, tuple[str, dict]] = {}
    for p in sorted(glob.glob(_paths.logs("*__*_ORACLE_*.txt"))):
        m = _LOG.match(os.path.basename(p))
        if not m or m.group("tag") == "diesel":
            continue                      # diesel caches are already correct
        stem = m.group("stem")
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        mo = _OBJ.search(txt)
        if not mo:
            continue
        rec = dict(
            feasible=True, optimal=False, obj=float(mo.group(1)),
            gap=float(_GAP.search(txt).group(1)) if _GAP.search(txt) else None,
            best_bound=(float(_BND.search(txt).group(1))
                        if _BND.search(txt) else None),
            stop_reason=(_STOP.search(txt).group(1)
                         if _STOP.search(txt) else "unknown"),
            sol=[],                      # NOT recoverable from the log
            sol_unavailable=True,
            recovered_from=os.path.basename(p),
        )
        ts = os.path.basename(p)
        if stem not in best or ts > best[stem][0]:
            best[stem] = (ts, rec)

    written = 0
    for stem, (_ts, rec) in best.items():
        out = _paths.solutions(f"oracle_{stem}.json")
        if dry:
            print(f"  would write {os.path.basename(out)}  obj={rec['obj']:.3f}")
        else:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2)
        written += 1
    return written, len(best)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("1) quarantine contaminated BASE oracle caches")
    n = quarantine_base(args.dry_run)
    print(f"   {'would move' if args.dry_run else 'moved'}: {n} "
          f"-> {QUARANTINE}/   (these instances must be re-solved)")

    print("2) rebuild variant oracle caches from logs")
    w, tot = rebuild_variants(args.dry_run)
    print(f"   {'would write' if args.dry_run else 'wrote'}: {w} variant "
          f"cache(s) from {tot} instance log(s)")
    print("   NOTE: recovered caches carry the objective/bound only "
          "(sol_unavailable=True)")


if __name__ == "__main__":
    main()
