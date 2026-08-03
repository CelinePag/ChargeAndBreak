"""
test_no_base_contamination.py — regression test: running an additional
analysis must never touch a base-case artefact.

Three separate defects in this project all had the same symptom (base-case
numbers silently changing when a sensitivity sweep ran):

  1. variant instances kept the BASE title, so variant runs displaced base
     runs in the compile dedup and variant oracles overwrote base caches
  2. the ORACLE MIP warm start persisted a greedy solution, which then won
     the dedup against the real greedy run
  3. --skip-existing never skipped an oracle, so caches were rewritten

This test takes a fingerprint of every base-case file, runs a small but
representative additional-analysis batch (variant greedy + variant oracle +
diesel oracle, i.e. every write path), and asserts the fingerprint is
unchanged.  It is the check that would have caught all three.

Usage
  python test_no_base_contamination.py            # ~2-5 min
  python test_no_base_contamination.py --quick    # greedy paths only (~20 s)
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

SOL = "solutions"


def is_base(path: str) -> bool:
    """A base-case artefact: no '__<tag>' anywhere in the file name."""
    return "__" not in os.path.basename(path)


def fingerprint() -> dict[str, tuple]:
    """name -> (size, mtime_ns) for every base-case solution and oracle cache."""
    fp = {}
    for p in glob.glob(os.path.join(SOL, "*.json")):
        if not is_base(p):
            continue
        st = os.stat(p)
        fp[os.path.basename(p)] = (st.st_size, st.st_mtime_ns)
    return fp


def diff(before: dict, after: dict) -> tuple[list, list, list]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after)
                     if before[k] != after[k])
    return added, removed, changed


def run(cmd: list[str]) -> int:
    print("    $ " + " ".join(cmd[1:]))
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="skip the oracle legs (much faster, weaker)")
    args = ap.parse_args()

    py = sys.executable
    base_before = fingerprint()
    print(f"base-case artefacts fingerprinted: {len(base_before)}")

    # a batch that exercises EVERY write path an additional analysis can take
    algs = "greedy" if args.quick else "greedy,ORACLE"
    batches = [
        # variant instance: greedy (+ oracle, which also runs a warm-start
        # greedy internally — the path that used to persist a base-titled run)
        [py, "additional_analysis.py", "sensitivity", "--axis", "cs_spacing",
         "--values", "30", "--algorithms", algs,
         "--combos", "RshortCfew", "--tw", "none", "--seeds", "1-2"],
        # diesel copy: same, plus the _apply_diesel_mode title path
        [py, "additional_analysis.py", "diesel", "--algorithms", algs,
         "--combos", "RshortCfew", "--tw", "none", "--seeds", "1-2"],
        # patched instance (no_split): the tag carries no value suffix, so it
        # is the axis most likely to collide with a base stem
        [py, "additional_analysis.py", "sensitivity", "--axis", "no_split",
         "--algorithms", algs,
         "--combos", "RshortCfew", "--tw", "none", "--seeds", "1-2"],
    ]
    print("\nrunning additional-analysis batches:")
    for b in batches:
        rc = run(b)
        if rc != 0:
            print(f"    (exit {rc})")

    base_after = fingerprint()
    added, removed, changed = diff(base_before, base_after)

    print("\nbase-case fingerprint diff")
    print(f"  added   : {len(added)}")
    print(f"  removed : {len(removed)}")
    print(f"  changed : {len(changed)}")
    for label, lst in (("added", added), ("removed", removed),
                       ("changed", changed)):
        for name in lst[:8]:
            print(f"    {label:<8} {name}")

    ok = not (added or removed or changed)
    print("\nRESULT:", "PASS — the base case was untouched" if ok else
          "FAIL — an additional analysis wrote into the base case")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
