"""
ablations.py — the pre-registered experiment programme
======================================================
Each row is one training run plus one closed-loop pass over the validation
routes.  They were chosen BEFORE any result was seen, and they answer the
design questions the project was built on rather than sweeping hyperparameters:

  target      regret (centred per decision) vs raw cost.  The cost level has
              std 29.6 h against a regret std of ~4 h, so a raw-cost regressor
              spends its capacity on "how much route is left" and none on
              ranking.  This measures how much that matters.
  weighting   margin/(margin+2*SEM) vs unweighted.  13.6% of decisions are
              inside the teacher's own sampling noise; does declining to fit
              them help?
  capacity    shallow (depth 3 / 15 leaves) vs the default (depth 6 / 63).
              With ~570 independent routes, the effective sample size is far
              below the 284k row count -- this locates where variance starts
              to win.
  guard       the one-step feasibility guard quantile at DEPLOYMENT only (no
              retraining): nominal, as the teacher ran, vs 0.95.  This is the
              lever for the halts, and it costs time.

Results land in ML/results/ablations.json.
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

# (tag, train flags, eval flags, one-line description)
GRID = [
    ("base", [], [], "regret target, margin weights, depth 6"),
    ("rawcost", ["--target", "cost"], [], "raw cost target (no centring)"),
    ("noweight", ["--weight-power", "0"], [], "no margin weighting"),
    ("shallow", ["--leaves", "15", "--depth", "3"], [], "depth 3 / 15 leaves"),
    ("base_g95", [], ["--guard-q", "0.95"], "base model, 0.95 guard at deploy"),
]

_NUM = r"([-+]?\d+\.?\d*)"
PATTERNS = dict(
    completed=r"^completed\s+(\d+)",
    halted=r"^halted\s+(\d+)",
    ms=r"^latency\s+" + _NUM,
    med_la=r"vs LA.*?\n\s+median\s+" + _NUM,
    med_greedy=r"vs GREEDY.*?\n\s+median\s+" + _NUM,
    med_oracle=r"vs ORACLE.*?\n\s+median\s+" + _NUM,
    med_pen=r"penalised objective vs LA.*?\n\s+median\s+" + _NUM,
    tw=r"^   student (\d+)   LA (\d+)",
)


def parse_report(txt: str) -> dict:
    out = {}
    for k, pat in PATTERNS.items():
        m = re.search(pat, txt, re.M | re.S)
        if m:
            out[k] = float(m.group(1)) if k != "tw" else (
                int(m.group(1)), int(m.group(2)))
    return out


def run(cmd):
    print("   $ " + " ".join(cmd[1:]), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--only", default=None, help="comma-separated tags")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    grid = GRID
    if args.only:
        want = set(args.only.split(","))
        grid = [g for g in grid if g[0] in want]

    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for tag, tflags, eflags, desc in grid:
        print(f"\n=== {tag}: {desc} ===", flush=True)
        t0 = time.time()
        # base_g95 reuses the base model: deployment-only change
        model_tag = "base" if tag == "base_g95" else tag
        if not args.skip_train and tag != "base_g95":
            run([PY, os.path.join(HERE, "gbt_train.py"), "--tag", model_tag]
                + tflags)
        txt = run([PY, os.path.join(HERE, "evaluate.py"), "--tag", model_tag,
                   "--split", args.split, "--out", f"eval_{tag}_{args.split}.json"]
                  + eflags)
        r = parse_report(txt)
        r.update(tag=tag, desc=desc, seconds=round(time.time() - t0, 1))
        rows.append(r)
        print(f"   -> median vs LA {r.get('med_la')}%  halted {r.get('halted')}"
              f"  ({r['seconds']}s)", flush=True)

    with open(os.path.join(RESULTS, f"ablations_{args.split}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    print("\n" + "=" * 86)
    print(f"ABLATIONS ({args.split})")
    print("=" * 86)
    hdr = (f"{'variant':10s} {'vs LA':>8s} {'vs GRDY':>8s} {'vs ORCL':>8s} "
           f"{'penalis':>8s} {'halt':>5s} {'TW':>5s} {'ms/dec':>8s}  description")
    print(hdr)
    print("-" * 86)
    for r in rows:
        tw = r.get("tw", (0, 0))[0]
        print(f"{r['tag']:10s} {r.get('med_la', float('nan')):+8.2f} "
              f"{r.get('med_greedy', float('nan')):+8.2f} "
              f"{r.get('med_oracle', float('nan')):+8.2f} "
              f"{r.get('med_pen', float('nan')):+8.2f} "
              f"{int(r.get('halted', -1)):5d} {tw:5d} "
              f"{r.get('ms', float('nan')):8.2f}  {r['desc']}")
    if rows and "tw" in rows[0]:
        print(f"\n(teacher TW misses on the same routes: {rows[0]['tw'][1]})")
    print(f"\nsaved: {RESULTS}/ablations_{args.split}.json")


if __name__ == "__main__":
    main()
