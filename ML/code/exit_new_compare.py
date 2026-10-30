"""
exit_new_compare.py — expert iteration on brand-new routes vs on the training routes
====================================================================================
Two networks after two rounds of expert iteration from the same original
network (tmlp_F95_split_list_s0):
  old  tmlp_F95_split_list_s0_exit2     labels on the 639 training routes x 2 draws
  new  tmlp_F95_split_list_s0_exitnew2  labels on brand-new routes x 1 draw
                                        (new_routes.py, hpc stage rl-exit-new)
each alone and with the decisive-state rollout, on the stored realizations of
the 66 validation and the 125 test routes, as in Table 1 of the paper:

  old: results/rl_search_exit_<split>_r2.jsonl     (student, dsr32+race8)
  new: results/rl_search_exitnew_<split>_r2.jsonl  (student, dsr32+race8, and
       the old network alone again -- it must match the old file, which checks
       that the two runs are comparable)

Gap to the oracle with the window penalty (rl_test_report.references, i.e.
paper_link's definition), the LA's gap, paired differences new - old and
vs. the LA in pp +- one standard error over routes, routes with more daily
rests than the LA, and missed windows.  Routes where any run broke a rule or
without an oracle are counted and left out.  The decision should be read on
validation; the test rows are what the paper reports for the network kept.

Writes results/exit_new_compare.json and results/exit_new_compare.txt.

    python ML/code/exit_new_compare.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_test_report import references                              # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OLD_NET = "net:torch:tmlp_F95_split_list_s0_exit2"
SPLITS = (("validation", "val", 66), ("test", "test", 125))


def _load(name):
    p = os.path.join(RESULTS, f"rl_search_{name}.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["instance"]] = {v: res[0] for v, res in zip(r["variants"], r["res"])}
    return out


def _ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def compare(split, old_name, new_name):
    old, new = _load(old_name), _load(new_name)
    if old is None or new is None:
        return None, [f"{split}: missing rl_search_{old_name if old is None else new_name}.jsonl"]
    runs = dict(old_alone=("old", "student"), old_rollout=("old", "dsr32+race8"),
                new_alone=("new", "student"), new_rollout=("new", "dsr32+race8"))
    rows, broken, no_orc, repro = [], 0, 0, []
    for inst in sorted(set(old) & set(new)):
        res = {k: (old if f == "old" else new)[inst][v] for k, (f, v) in runs.items()}
        if OLD_NET in new[inst]:
            repro.append(abs(new[inst][OLD_NET]["duration"] - old[inst]["student"]["duration"]))
        if any(r["failed"] for r in res.values()):
            broken += 1
            continue
        ref = references("base", "base", inst)
        if ref is None or ref["la"][1] is None:
            no_orc += 1
            continue
        row = dict(inst=inst, la=ref["la"][1], la_rests=ref["la_rests"])
        for k, r in res.items():
            row[k] = ref["gaps"](r["duration"], r["misses"])[1]
            row[k + "_rests"] = r["rests"]
            row[k + "_misses"] = r["misses"]
        rows.append(row)

    out = dict(routes_run=len(set(old) & set(new)), compared=len(rows), broken=broken,
               no_oracle=no_orc, policies={}, new_minus_old={})
    lines = [f"{split}: {out['routes_run']} routes run, {len(rows)} compared "
             f"({broken} with a broken run, {no_orc} without an oracle or LA run left out)"]
    if repro:
        same = sum(d < 1e-6 for d in repro)
        out["old_network_reproduced"] = [same, len(repro), float(max(repro))]
        lines.append(f"  old network alone re-run in the new job: identical on {same}/{len(repro)} "
                     f"routes (largest difference {max(repro):.3f} h)")
    if not rows:
        return out, lines
    lines.append(f"  {'policy':34s} {'gap %':>6s} {'vs LA (pp)':>14s} {'rests > LA':>10s} {'missed':>7s}")
    lines.append(f"  {'LA':34s} {np.mean([r['la'] for r in rows]):6.2f}")
    labels = dict(old_alone="network, expert iteration (old)",
                  old_rollout="  + rollout (old)",
                  new_alone="network, expert iteration (new)",
                  new_rollout="  + rollout (new)")
    for k, lab in labels.items():
        g = float(np.mean([r[k] for r in rows]))
        dl = _ms([r[k] - r["la"] for r in rows])
        more = sum(1 for r in rows if r["la_rests"] is not None and r[k + "_rests"] > r["la_rests"])
        miss = int(sum(r[k + "_misses"] for r in rows))
        out["policies"][k] = dict(gap_pen=g, vs_la_pen=dl, rests_more_than_la=more, missed=miss)
        lines.append(f"  {lab:34s} {g:6.2f} {dl[0]:+7.2f} ± {dl[1]:4.2f} {more:10d} {miss:7d}")
    for pol in ("alone", "rollout"):
        d = [r["new_" + pol] - r["old_" + pol] for r in rows]
        m = _ms(d)
        better, worse = sum(x < -1e-9 for x in d), sum(x > 1e-9 for x in d)
        out["new_minus_old"][pol] = dict(mean=m[0], se=m[1], better=better, worse=worse)
        word = ("better" if m[0] < -2 * m[1] else "worse" if m[0] > 2 * m[1] else "no clear difference")
        lines.append(f"  new - old, {'network alone' if pol == 'alone' else 'with the rollout':17s}"
                     f" {m[0]:+.2f} ± {m[1]:.2f} pp  (new better on {better}, worse on {worse} routes:"
                     f" {word})")
    return out, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="exit_{s}_r2", help="old run name, {s} = val / test")
    ap.add_argument("--new", default="exitnew_{s}_r2", help="new run name, {s} = val / test")
    ap.add_argument("--out", default="exit_new_compare")
    a = ap.parse_args()
    report, text = {}, ["Expert iteration: brand-new routes (new) vs the training routes (old), "
                        "gap to the oracle with the window penalty, stored realizations", ""]
    for split, s, _n in SPLITS:
        res, lines = compare(split, a.old.format(s=s), a.new.format(s=s))
        report[split] = res
        text += lines + [""]
    print("\n".join(text))
    with open(os.path.join(RESULTS, f"{a.out}.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(RESULTS, f"{a.out}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
