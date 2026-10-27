"""
rest_mispricing.py — how wrong is the student about "rest now", and when?
=========================================================================
From rl_api labels (rl_api.py): every REST candidate at a labelled state,
its measured advantage over the student's pick (rollouts, the student driving
on, shared futures) against the student's predicted advantage, split by
  - count-neutral: the nominal rest count to the destination (rl_env.
    rests_needed, with the 10 h days) is the same as after the pick;
  - lost: the driving allowance this shift gives up by resting now
    (10 h while an extended day is left, else 9 h, minus driving so far).
Prints only.

    python ML/scratch/rest_mispricing.py r1_gbt
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
DATA = os.path.join(HERE, "..", "data")
MODELS = os.path.join(HERE, "..", "models")


def main(name):
    meta = json.load(open(os.path.join(DATA, f"rl_api_{name}", "meta.json")))
    feats = json.load(open(os.path.join(MODELS, f"{meta['tag']}_meta.json")))["features"]
    i_sd, i_ext = feats.index("sd"), feats.index("ext_left")
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, f"rl_api_{name}", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        for r in z["recs"]:
            for j in range(1, len(r["keys"])):
                if not r["keys"][j].endswith(("_r1", "_r2")) or r["keys"][0].endswith(("_r1", "_r2")):
                    continue
                sd, ext = float(r["rows"][j, i_sd]), float(r["rows"][j, i_ext])
                lost = max(0.0, (10.0 if ext > 0 else 9.0) - sd)
                neutral = r["walk"][j, 0] <= r["walk"][0, 0]
                out.append((neutral, lost, float(r["score"][j] - r["score"][0]),
                            float(r["d_mean"][j]), float(r["d_se"][j]),
                            float(r["rests_mean"][j] - r["rests_mean"][0])))
    a = np.array(out, dtype=float)
    print(f"{name}: {len(a)} rest candidates vs a non-rest pick")
    print(f"{'nominal count':14s} {'lost h':>8s} {'n':>5s} {'predicted h':>12s} "
          f"{'measured h':>11s} {'meas. better':>13s} {'d rests':>8s}")
    for neu in (1.0, 0.0):
        for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 11)):
            m = (a[:, 0] == neu) & (a[:, 1] >= lo) & (a[:, 1] < hi)
            if m.sum() == 0:
                continue
            b = a[m]
            print(f"{'same' if neu else '+1 rest':14s} {f'{lo}-{hi}':>8s} {int(m.sum()):5d} "
                  f"{b[:, 2].mean():+12.2f} {b[:, 3].mean():+7.2f}±{b[:, 3].std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else 0:.2f} "
                  f"{100 * (b[:, 3] + 2 * b[:, 4] < 0).mean():12.0f}% {b[:, 5].mean():+8.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "r1_gbt")
