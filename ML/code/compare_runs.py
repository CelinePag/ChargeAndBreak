"""
compare_runs.py — closed-loop comparison table across student variants
======================================================================
Reads every student run in ML/solutions/, groups by method label
(STUDENTK3, STUDENTK10, ...), joins each instance with its baselines from
the manuscript's solutions/ tree (teacher = LA_MIPTAIL, LP tail = plain LA),
and prints the numbers that decide things: PAIRED duration deltas, failure
counts, latency.

Why paired: instances differ enormously in length, so an unpaired mean over
instances is dominated by which instances happen to be in the set.  Comparing
each run against the SAME instance's teacher run removes that variance —
this is the same convention the journal sweep uses (`delta_vs_base_pct`).

Why medians AND tails: a policy can have a good median and still be unusable
because it occasionally loses 10 h.  We print median, IQR and max.

Usage:  python ML/code/compare_runs.py [--split val]
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ML", "code"))
from extract_dataset import split_of

sys.path.insert(0, ROOT)
from src import paths as _paths                                  # noqa: E402

SOLS, REF = os.path.join(ROOT, "ML", "solutions"), os.path.join(ROOT, "solutions")


def _latest(directory, pattern):
    """Newest match for `pattern` in a bucketed solutions tree.

    Both trees are split into experiment buckets, so the search goes through
    paths.in_tree and the ranking is on the BASENAME — a run_id ends in its
    timestamp, so that is chronological order wherever the file sits.
    """
    fs = [f for f in _paths.in_tree(directory, pattern)
          if "nosplit" not in f and "LOCAL" not in f]
    return sorted(fs, key=os.path.basename)[-1] if fs else None


def baseline(inst, tag):
    f = _latest(REF, f"{inst}_{tag}*.json")
    return json.load(open(f)) if f else None


def main(a):
    # group student runs: method -> instance -> latest run
    groups: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for f in sorted(_paths.in_tree(SOLS, "*.json"), key=os.path.basename):
        d = json.load(open(f))
        m = re.match(r".*_(STUDENT[A-Za-z0-9]*)_S\d", os.path.basename(f))
        if not m:
            continue
        inst = d["instance"]
        fam, seed = inst.rsplit("_", 1)
        # NOTE: the split label depends on the split MODE the model was
        # trained under — filtering with the wrong mode silently scores only
        # a subset of the rollout (which is exactly what it did once).
        if a.split and split_of(fam, int(seed), a.mode) != a.split:
            continue
        groups[m.group(1)][inst] = d          # sorted glob => latest wins

    # baselines, cached per instance
    ref_cache: dict[str, dict] = {}
    def refs(inst):
        if inst not in ref_cache:
            # LA_MIPTAIL is now the STANDARD look-ahead (the LP tail was
            # demoted to an LPTAIL variant and no longer exists for the base
            # grid), so the teacher is the only solver baseline available.
            ref_cache[inst] = dict(teacher=baseline(inst, "LA_MIPTAIL"))
        return ref_cache[inst]

    def completed(run):
        """Route finished?  Since the halt-on-infeasible change a run that
        breaches ends AT that stop and carries duration_h = None, so a
        duration only exists for a completed route.  Halted runs must never
        enter a duration median — they are counted, not averaged."""
        return bool(run) and run.get("duration_h") is not None

    hdr = (f"{'method':16s} {'n':>3s} {'vs teacher % (completed only)':>30s} "
           f"{'halt':>5s} {'infeas':>7s} {'HoS':>4s} {'strand':>6s} {'TW':>4s} "
           f"{'ms/dec':>7s}")
    print(hdr); print("-" * len(hdr))
    for meth in sorted(groups, key=lambda s: int(re.sub(r"\D", "", s) or 0)):
        runs = groups[meth]
        dt, halt, infeas, hos, strand, tw, dec = [], 0, 0, 0, 0, 0, []
        for inst, d in runs.items():
            r, m = refs(inst), d.get("metrics", {})
            if not completed(d):
                halt += 1                      # no duration exists: count it
            elif completed(r["teacher"]):
                dt.append(100 * (d["duration_h"] - r["teacher"]["duration_h"])
                          / r["teacher"]["duration_h"])
            infeas += bool(m.get("run_infeasible"))
            hos    += int(m.get("n_hos_violations") or 0)
            strand += int(m.get("n_stranding") or 0)
            tw     += int(m.get("tw_n_misses") or 0)
            dec.append(1000 * (m.get("decision_time_mean_s") or 0))
        q = np.percentile(dt, [25, 50, 75]) if dt else [np.nan] * 3
        mx = max(dt) if dt else float("nan")
        print(f"{meth:16s} {len(runs):3d} "
              f"{q[1]:+7.2f} [{q[0]:+6.2f},{q[2]:+6.2f}] max{mx:+6.1f} n={len(dt):<3d} "
              f"{halt:5d} {infeas:7d} {hos:4d} {strand:6d} {tw:4d} "
              f"{np.median(dec):7.2f}")

    # teacher/LP-tail reference rows on the same instances
    insts = sorted({i for g in groups.values() for i in g})
    for tag, key in (("LA-MIP (teacher)", "teacher"),):
        vals = [refs(i)[key] for i in insts if refs(i)[key]]
        if not vals:
            continue
        inf = sum(bool(v["metrics"].get("run_infeasible")) for v in vals)
        h = sum(int(v["metrics"].get("n_hos_violations") or 0) for v in vals)
        s = sum(int(v["metrics"].get("n_stranding") or 0) for v in vals)
        t = sum(int(v["metrics"].get("tw_n_misses") or 0) for v in vals)
        dsec = np.median([v["metrics"].get("decision_time_mean_s") or 0
                          for v in vals])
        print(f"{tag:12s} {len(vals):3d} {'—':>22s} {'—':>14s} "
              f"{inf:7d} {h:4d} {s:6d} {t:4d} {1000*dsec:7.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--split-mode", dest="mode", default="family",
                   choices=["family", "seed"],
                   help="must match the mode the evaluated model was trained under")
    p.add_argument("--by-route", action="store_true",
                   help="break the table down by route class")
    main(p.parse_args())
