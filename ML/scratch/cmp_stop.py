"""Closed-loop comparison on one split + guard setting:  python cmp_stop.py stop g95"""
import json
import os
import sys

import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "stop"
GUARD = sys.argv[2] if len(sys.argv) > 2 else "g95"
TAGS = ["gbt_F95_base_s0", "tmlp_F95_base_s0", "tmlp_F95_sel_s0", "tmlp_F95_list_s0",
        "tmlp_F95_list01_s0", "tmlp_F95_list3_s0", "tmlp_F95_split_s0", "tmlp_F95_split_list_s0"]


def load(tag):
    p = os.path.join(R, f"eval_{tag}_{GUARD}_{SPLIT}.json")
    return {r["instance"]: r for r in json.load(open(p))} if os.path.exists(p) else None


ref = load("gbt_F95_base_s0")
print(f"{SPLIT} / {GUARD}")
print(f"{'model':24s} {'n':>3s} {'done':>4s} {'viol':>4s} {'mean%LA':>8s} {'se':>5s} {'med%LA':>7s} "
      f"{'>5%':>4s} {'rest+':>6s} {'TW':>4s}   {'vs trees (paired)':>18s}  halts")
for t in TAGS:
    d = load(t)
    if d is None:
        continue
    done = [r for r in d.values() if r["route_completed"]]
    both = [r for r in done if r["LA_completed"] and r["LA"]]
    pct = np.array([100 * (r["duration_h"] / r["LA"] - 1) for r in both])
    rests = np.mean([r["n_rests"] - r["LA_rests"] for r in both])
    common = [k for k in d if d[k]["route_completed"] and ref[k]["route_completed"]]
    dv = np.array([100 * (d[k]["duration_h"] / ref[k]["duration_h"] - 1) for k in common])
    halts = sorted({r["halt_reason"] for r in d.values() if not r["route_completed"]})
    print(f"{t:24s} {len(d):3d} {len(done):4d} {sum(r['n_violations'] for r in d.values()):4d} "
          f"{pct.mean():+8.2f} {pct.std(ddof=1) / np.sqrt(len(pct)):5.2f} {np.median(pct):+7.2f} "
          f"{(pct > 5).sum():4d} {rests:+6.3f} {sum(r['tw_misses'] for r in d.values()):4d}   "
          f"{dv.mean():+7.2f} +- {dv.std(ddof=1) / np.sqrt(len(dv)):.2f}  {','.join(halts)}")
