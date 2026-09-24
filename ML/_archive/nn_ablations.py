"""
nn_ablations.py — the NEURAL arm's grid, mirroring the tree arm's
=================================================================
`ablations.py` searched five configurations of the tree arm.  The neural arm
got one.  Comparing them and concluding "trees win" therefore compared a
SEARCHED arm against an UNSEARCHED one, which is not a result about function
approximators.  This file runs the matching grid so the comparison means
something.

Axis by axis, against `ablations.py`:

  target        `rawcost` there, `nn_rawcost` here — regress the teacher's raw
                horizon objective instead of the per-decision regret.  The
                level has std 29.6 h against a decision margin of ~40 min, so
                this is the design choice with the largest effect on the trees
                (+3.41% and 14 infeasible).  Does it hurt the network as much?

  tail          NOT on the tree grid, because LightGBM uses HUBER and is
                already protected.  sklearn's MLPRegressor is hard-wired to
                SQUARED ERROR, and the regret target has median 0.655 h
                against a p90 of 11 h — so a single tail row counts for ~280
                median rows and the network optimises almost entirely for
                "never wrongly predict a daily rest".  `nn_raw` (no transform)
                is the handicapped original; `nn_log1p` is the fair one.  The
                transform is MONOTONE, so the argmin the policy takes is
                unchanged.

  capacity      `shallow` there, `nn_small` / `nn_big` here.

  guard         identical: the one-step feasibility guard at DEPLOYMENT only,
                no retraining.

  weighting     NOT RUN.  sklearn's MLP has no `sample_weight`, so the tree
                arm's `noweight` ablation has no counterpart.  The tree grid
                found it inert anyway.

Results land in ML/results/ablations_nn_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
PY = sys.executable

# (tag, train flags, eval flags, description).  train flags None = reuse a
# model already trained under `model` (deployment-only change).
GRID = [
    # the original `nn` checkpoint predates --target-transform, so it IS the
    # untransformed model; reuse it rather than retraining an identical one
    ("nn_raw", None, [], "nn",
     "squared error on raw regret (the handicapped original)"),
    ("nn_log1p", ["--target-transform", "log1p"], [], None,
     "log1p regret — tail-robust, monotone (the fair baseline)"),
    ("nn_rawcost", ["--target", "cost", "--target-transform", "log1p"], [], None,
     "raw horizon cost instead of centred regret"),
    ("nn_small", ["--hidden", "32"], [], None, "one hidden layer of 32"),
    ("nn_big", ["--hidden", "256,128"], [], None, "hidden 256,128"),
    ("nn_log1p_g95", None, ["--guard-q", "0.95"], "nn_log1p",
     "the fair baseline + 0.95 guard at deployment"),
]

_NUM = r"([-+]?\d+\.?\d*)"
PATTERNS = dict(
    completed=r"^completed\s+(\d+)",
    infeasible=r"^infeasible\s+(\d+)",
    ms=r"^latency\s+" + _NUM,
    med_la=r"vs LA.*?\n\s+median\s+" + _NUM,
    med_greedy=r"vs GREEDY.*?\n\s+median\s+" + _NUM,
    med_oracle=r"vs ORACLE.*?\n\s+median\s+" + _NUM,
    med_pen=r"penalised objective vs LA.*?\n\s+median\s+" + _NUM,
    tw=r"^   student (\d+)   LA (\d+)",
)


def parse_report(txt):
    out = {}
    for k, pat in PATTERNS.items():
        m = re.search(pat, txt, re.M | re.S)
        if m:
            out[k] = float(m.group(1)) if k != "tw" else (
                int(m.group(1)), int(m.group(2)))
    return out


def run(cmd):
    print("   $ " + " ".join(os.path.basename(c) for c in cmd[1:2])
          + " " + " ".join(cmd[2:]), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-1500:])
        print(p.stderr[-1500:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    grid = GRID
    if args.only:
        want = set(args.only.split(","))
        grid = [g for g in grid if g[0] in want]

    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for tag, tflags, eflags, reuse, desc in grid:
        print(f"\n=== {tag}: {desc} ===", flush=True)
        t0 = time.time()
        model_tag = reuse or tag
        if tflags is not None and not args.skip_train:
            # nn_log1p may already exist from the fairness check
            ck = os.path.join(HERE, "..", "models", f"{model_tag}_nn.joblib")
            if not os.path.exists(ck):
                run([PY, os.path.join(HERE, "nn_train.py"), "--tag", model_tag]
                    + tflags)
            else:
                print(f"   (reusing existing {model_tag})", flush=True)
        txt = run([PY, os.path.join(HERE, "evaluate.py"), "--kind", "nn",
                   "--tag", model_tag, "--split", args.split,
                   "--out", f"eval_{tag}_{args.split}.json"] + eflags)
        r = parse_report(txt)
        r.update(tag=tag, desc=desc, seconds=round(time.time() - t0, 1))
        rows.append(r)
        print(f"   -> vs LA {r.get('med_la')}%  infeasible {r.get('infeasible')}"
              f"  ({r['seconds']}s)", flush=True)

    with open(os.path.join(RESULTS, f"ablations_nn_{args.split}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    print("\n" + "=" * 92)
    print(f"NEURAL ABLATIONS ({args.split})")
    print("=" * 92)
    print(f"{'variant':14s} {'vs LA':>8s} {'vs GRDY':>8s} {'vs ORCL':>8s} "
          f"{'penalis':>8s} {'infs':>5s} {'TW':>5s} {'ms/dec':>7s}  description")
    print("-" * 92)
    for r in rows:
        tw = r.get("tw", (0, 0))[0]
        print(f"{r['tag']:14s} {r.get('med_la', float('nan')):+8.2f} "
              f"{r.get('med_greedy', float('nan')):+8.2f} "
              f"{r.get('med_oracle', float('nan')):+8.2f} "
              f"{r.get('med_pen', float('nan')):+8.2f} "
              f"{int(r.get('infeasible', -1)):5d} {tw:5d} "
              f"{r.get('ms', float('nan')):7.2f}  {r['desc']}")
    print(f"\nsaved: {RESULTS}/ablations_nn_{args.split}.json")
    print("Tree arm for comparison: ML/results/ablations_val.json")


if __name__ == "__main__":
    main()
