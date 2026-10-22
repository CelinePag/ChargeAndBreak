"""
gate_report.py — closed-loop results of the solver-call gate (step 3)
====================================================================
Reads the hybrid runs on base-case routes (run_hybrid.py --routes base,
results/hybrid_base[_val]_<cfg>.jsonl), the student driving alone with the same
settings (evaluate.py --mseq --nominal-features -> eval_<tag>_g99srNm_<split>.json)
and the stored LA runs, and reports, per student and configuration:

  calls per route, gap to the oracle and the difference with the LA (points,
  paired by route), without and with the 0.5 h window penalty, extra daily
  rests against the LA, broken runs; and each trigger paired against the
  student alone.  Gap code: paper_link.gaps (the manuscript's definition).

    python ML/code/gate_report.py --set val
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_link as pl                                           # noqa: E402

RES = pl.RESULTS
STUDENTS = {"gbt_F95_base_s0": "Trees", "tmlp_F95_split_list_s0": "PyTorch"}


def la_run(inst):
    f = pl._latest(os.path.join(pl.SOL, f"{inst}_LA_MIPTAIL_*.json"))
    if not f:
        return None
    s = json.load(open(f))
    m = s.get("metrics", {})
    if m.get("run_infeasible") or s.get("duration_h") is None:
        return None
    rests = sum(1 for a in (s.get("actions") or [])
                if str(a.get("rest_type")).lower() not in ("0", "none", "-", ""))
    return dict(dur=s["duration_h"], tw=int(m.get("tw_n_misses", 0)), rests=rests,
                t0=s["sim_arrival_h"] - s["duration_h"])


def gaps(inst, dur, tw, la):
    orc = pl.oracle_facts(inst)
    if orc is None or dur is None:
        return None
    return pl.gaps(dur, tw, dur + la["t0"], orc)


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="val", choices=["val", "test"])
    args = ap.parse_args()
    split = "stop" if args.set == "val" else "test"
    tail = "_val" if args.set == "val" else ""
    for tag, arm in STUDENTS.items():
        alone_f = os.path.join(RES, f"eval_{tag}_g99srNm_{split}.json")
        if not os.path.exists(alone_f):
            continue
        alone = {r["instance"]: r for r in json.load(open(alone_f))}
        configs = {"student alone": {i: dict(completed=r["route_completed"], duration_h=r["duration_h"],
                                             tw=r["tw_misses"], rests=r["n_rests"], n_calls=0)
                                     for i, r in alone.items()}}
        for f in sorted(glob.glob(os.path.join(RES, f"hybrid_base{tail}_*_ms_N.jsonl"))):
            cfg = os.path.basename(f)[len(f"hybrid_base{tail}_"):-len(".jsonl")]
            if not tail and cfg.startswith("val_"):         # test: skip the validation files
                continue
            rows = [json.loads(x) for x in open(f) if x.strip()]
            rows = {r["instance"]: r for r in rows if r["method"] == tag}
            if rows:
                configs[cfg] = rows
        LA = {i: la_run(i) for i in alone}
        print(f"\n### {arm} ({tag}), {args.set} routes: vs LA in points (paired), "
              f"and vs the student alone")
        print("| Trigger | Routes | LA calls per route | vs LA, no penalty | vs LA, with penalty "
              "| vs student alone, with penalty | Extra rests vs LA (runs) | Broken |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        base = configs["student alone"]
        for cfg, rows in configs.items():
            d, dp, ds, calls, extra, broken = [], [], [], [], 0, 0
            for i, r in rows.items():
                la = LA.get(i)
                if la is None:
                    continue
                calls.append(r.get("n_calls", 0))
                if not r["completed"]:
                    broken += 1
                    continue
                g = gaps(i, r["duration_h"], r["tw"], la)
                gl = gaps(i, la["dur"], la["tw"], la)
                if g is None or gl is None:
                    continue
                d.append(g[0] - gl[0])
                dp.append(g[1] - gl[1])
                extra += int(r["rests"] > la["rests"])
                b = base.get(i)
                if b and b["completed"]:
                    gb = gaps(i, b["duration_h"], b["tw"], la)
                    ds.append(g[1] - gb[1])
            if not d:
                continue
            vs_alone = "–" if cfg == "student alone" else f"{np.mean(ds):+.2f} ± {se(ds):.2f}"
            print(f"| {cfg} | {len(d)} | {np.mean(calls):.1f} | {np.mean(d):+.2f} ± {se(d):.2f} | "
                  f"{np.mean(dp):+.2f} ± {se(dp):.2f} | {vs_alone} | {extra} | {broken} |")


if __name__ == "__main__":
    main()
