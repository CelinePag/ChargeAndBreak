"""
label_replication.py — are the rl_api labels signal or noise?
=============================================================
Two label runs on the SAME states (same roll-in seed) with INDEPENDENT futures
(rl_api.py --futures-seed): run A and run B.  For every challenger:
  - corr(d_A, d_B) over challengers (all, near the pick, rest / non-rest);
  - selection test: the challengers A says beat the pick (d_A + z se_A < 0,
    or simply A's best d < 0) -- what is their advantage MEASURED IN B?
    That is the honest value of acting on a label; with pure noise it is 0
    or positive, and the in-A gain is the winner's curse.
Prints only.

    python ML/scratch/label_replication.py r1_gbt r1_gbt_rep
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def recs(name):
    out = {}
    for f in glob.glob(os.path.join(DATA, f"rl_api_{name}", "*.npz")):
        z = np.load(f, allow_pickle=True)
        for r in z["recs"]:
            out[(str(z["instance"]), int(r["stop"]))] = r
    return out


def main(a_name, b_name):
    A, B = recs(a_name), recs(b_name)
    common = sorted(set(A) & set(B))
    rows = []
    for k in common:
        ra, rb = A[k], B[k]
        if list(ra["keys"]) != list(rb["keys"]):
            print("key mismatch at", k)
            continue
        for j in range(1, len(ra["keys"])):
            rest = ra["keys"][j].endswith(("_r1", "_r2")) or ra["keys"][0].endswith(("_r1", "_r2"))
            rows.append((k, j, rest, float(ra["score"][j] - ra["score"][0]),
                         float(ra["d_mean"][j]), float(ra["d_se"][j]),
                         float(rb["d_mean"][j]), float(rb["d_se"][j])))
    print(f"{len(common)} common states, {len(rows)} challengers")
    s = np.array([r[3] for r in rows]); dA = np.array([r[4] for r in rows])
    seA = np.array([r[5] for r in rows]); dB = np.array([r[6] for r in rows])
    rest = np.array([r[2] for r in rows])
    for nm, m in (("all", np.ones(len(rows), bool)), ("near (pred <= 1 h)", s <= 1.0),
                  ("non-rest pairs", ~rest), ("rest pairs", rest)):
        if m.sum() > 2:
            print(f"  corr(d_A, d_B) {nm:20s}: {np.corrcoef(dA[m], dB[m])[0, 1]:+.3f}  (n={m.sum()})")
    print("selection: challengers chosen on run A, measured on run B (h)")
    for z in (0.0, 1.0, 2.0):
        m = dA + z * seA < 0
        if m.any():
            print(f"  d_A + {z:.0f} se < 0: n={m.sum():4d}  mean d_A {dA[m].mean():+.3f}  "
                  f"mean d_B {dB[m].mean():+.3f} ± {dB[m].std(ddof=1) / np.sqrt(m.sum()):.3f}  "
                  f"(B also < 0: {100 * (dB[m] < 0).mean():.0f} %)")
    # per state: switch to A's best challenger if d_A < 0; value measured in B
    gain_A, gain_B = [], []
    for k in common:
        idx = [i for i, r in enumerate(rows) if r[0] == k]
        if not idx:
            continue
        i = min(idx, key=lambda i: dA[i])
        gain_A.append(min(0.0, dA[i]))
        gain_B.append(dB[i] if dA[i] < 0 else 0.0)
    gA, gB = np.array(gain_A), np.array(gain_B)
    print(f"per state, act on A's best challenger: gain in A {gA.mean():+.3f}, "
          f"in B {gB.mean():+.3f} ± {gB.std(ddof=1) / np.sqrt(len(gB)):.3f} h per state")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) > 2 else ("r1_gbt", "r1_gbt_rep")))
