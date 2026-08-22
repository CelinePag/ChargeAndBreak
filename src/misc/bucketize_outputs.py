"""
bucketize_outputs.py — file solutions/, logs/ and figures/ into their
experiment buckets.

    python -m src.misc.bucketize_outputs            # report only
    python -m src.misc.bucketize_outputs --apply    # move the files

The buckets (basecase / LAconfig / usecase / sensitivity) and the rules that
map a file name onto one of them live in src/paths.py; this script only walks
the three trees and applies those rules, so it is a pure re-run of the routing
every writer now performs at write time.  That makes it idempotent and safe to
repeat: a file already in its bucket is skipped, and a file whose bucket cannot
be derived is LEFT WHERE IT IS rather than filed under a guess.

It is also the tool to reach for after a name-classification rule changes: the
second run re-files whatever the new rule now recognises.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter

from src import paths as _paths

# (tree, "name -> bucket" function).  figures/ has its own classifier because a
# figure is named after the section it serves, not after a run.
_TREES = (
    ("solutions", _paths.SOLUTIONS, _paths.bucket_of_artefact),
    ("logs",      _paths.LOGS,      _paths.bucket_of_artefact),
    ("figures",   _paths.FIGURES,   _paths.bucket_of_figure),
)


def plan(base, classify) -> tuple[list[tuple[str, str, str]], Counter]:
    """-> ([(src, dst, bucket)], counts).

    Only the tree ROOT is walked: files already inside a bucket are in a bucket,
    and re-deriving them would just move them onto themselves.  Subdirectories
    that are not buckets (logs/_internal, solutions/.checkpoints) are working
    directories of a single tool and are left intact.
    """
    moves: list[tuple[str, str, str]] = []
    counts: Counter = Counter()
    try:
        entries = sorted(os.scandir(base), key=lambda e: e.name)
    except OSError:
        return moves, counts
    for e in entries:
        if not e.is_file():
            continue
        bucket = classify(e.name)
        if bucket not in _paths.BUCKETS:
            counts["unclassified"] += 1
            continue
        moves.append((e.path, str(base / bucket / e.name), bucket))
        counts[bucket] += 1
    return moves, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="move the files (default: report what would move)")
    ap.add_argument("--tree", choices=[n for n, _b, _c in _TREES], default=None,
                    help="only this tree (default: all three)")
    args = ap.parse_args()

    if args.apply:
        _paths.ensure_dirs()

    total = 0
    for name, base, classify in _TREES:
        if args.tree and name != args.tree:
            continue
        moves, counts = plan(base, classify)
        spread = "  ".join(f"{b}={counts[b]}" for b in _paths.BUCKETS
                           if counts[b])
        left = counts["unclassified"]
        print(f"{name}/: {len(moves)} file(s) to file"
              + (f"   [{spread}]" if spread else "")
              + (f"   ({left} left at the root: no bucket derivable)"
                 if left else ""))
        for src, dst, _b in moves:
            if not args.apply:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                # Same name in the root and in the bucket.  The bucketed copy is
                # the one every reader prefers (see paths._scan_tree), so the
                # root duplicate is what has to go — but never silently: an
                # identical size means it is the same artefact twice, anything
                # else is a real conflict the user has to look at.
                if os.path.getsize(src) == os.path.getsize(dst):
                    os.remove(src)
                else:
                    print(f"  CONFLICT (sizes differ, left in place): {src}")
                continue
            shutil.move(src, dst)
        total += len(moves)

    if args.apply:
        print(f"Filed {total} file(s).")
    else:
        print(f"{total} file(s) would move.  Re-run with --apply.")


if __name__ == "__main__":
    main()
