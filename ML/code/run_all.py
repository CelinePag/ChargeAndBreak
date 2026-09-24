"""
run_all.py — train every configuration once, report them all on test
====================================================================
One command, one table.  Reads `configs.py` (the registry), trains each
distinct model once, evaluates every configuration on the WHOLE test batch
(seeds 22-25, 125 routes), and writes `ML/results/all_test.json`.

Each row is an independent model measured on the same held-out routes, so the
rows are comparable to each other and to the baselines already stored in
solutions/basecase.  Nothing is selected on the test set -- every row is
reported.

    python ML/code/run_all.py                 # everything
    python ML/code/run_all.py --only gbt,mlp  # by arm
    python ML/code/run_all.py --skip-train    # re-evaluate existing models
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import CONFIGS, LEGACY                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))
PY = sys.executable

TRAINER = {"gbt": "gbt_train.py", "mlp": "nn_train.py", "clf": "clf_train.py"}
KIND = {"gbt": "gbt", "mlp": "nn", "clf": "clf"}
CHECKPOINT = {"gbt": "{tag}_cost.txt", "mlp": "{tag}_nn.joblib",
              "clf": "{tag}_clf.joblib"}

_NUM = r"([-+]?\d+\.?\d*)"
PATTERNS = dict(
    routes=r"^routes\s+(\d+)",
    completed=r"^completed\s+(\d+)",
    infeasible=r"^infeasible\s+(\d+)",
    ms=r"^latency\s+" + _NUM,
    med_la=r"vs LA.*?\n\s+median\s+" + _NUM,
    med_greedy=r"vs GREEDY.*?\n\s+median\s+" + _NUM,
    med_pen=r"penalised objective vs LA.*?\n\s+median\s+" + _NUM,
    tw=r"^   student (\d+)   LA (\d+)",
)


def parse(txt):
    out = {}
    for k, pat in PATTERNS.items():
        m = re.search(pat, txt, re.M | re.S)
        if m:
            out[k] = float(m.group(1)) if k != "tw" else (
                int(m.group(1)), int(m.group(2)))
    return out


def run(cmd, label):
    print(f"   $ {label}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-1500:])
        print(p.stderr[-1500:])
        raise SystemExit(f"FAILED: {label}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated arms or tags")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cfgs = CONFIGS
    if args.only:
        want = set(args.only.split(","))
        cfgs = [c for c in cfgs if c.arm in want or c.tag in want
                or f"{c.tag}_{c.guard}" in want]

    os.makedirs(RESULTS, exist_ok=True)
    trained = set()
    rows = []

    for c in cfgs:
        print(f"\n=== {c.tag}  (guard "
              f"{'nominal' if c.guard is None else c.guard}) : {c.desc} ===",
              flush=True)
        t0 = time.time()
        ck = os.path.join(MODELS, CHECKPOINT[c.arm].format(tag=c.tag))
        if not args.skip_train and c.tag not in trained:
            if os.path.exists(ck):
                print(f"   (checkpoint exists, reusing {c.tag})", flush=True)
            else:
                run([PY, os.path.join(HERE, TRAINER[c.arm]), "--tag", c.tag]
                    + list(c.train), f"train {c.tag}")
            trained.add(c.tag)

        cmd = [PY, os.path.join(HERE, "evaluate.py"), "--kind", KIND[c.arm],
               "--tag", c.tag, "--split", "test", "--out", c.eval_name]
        if c.guard is not None:
            cmd += ["--guard-q", str(c.guard)]
        txt = run(cmd, f"evaluate {c.tag} -> {c.eval_name}")

        r = parse(txt)
        r.update(arm=c.arm, name=c.name, tag=c.tag, guard=c.guard,
                 desc=c.desc, display=c.display, eval_file=c.eval_name,
                 seconds=round(time.time() - t0, 1))
        rows.append(r)
        print(f"   -> vs LA {r.get('med_la')}%  infeasible "
              f"{r.get('infeasible')}  ({r['seconds']}s)", flush=True)

    # the restored 2026-08 model, if it has been run by its own scripts
    leg = os.path.join(RESULTS, LEGACY.eval_name)
    if os.path.exists(leg):
        print(f"\n(joining {LEGACY.tag} from {LEGACY.eval_name})")

    with open(os.path.join(RESULTS, "all_test.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    print("\n" + "=" * 100)
    print("ALL CONFIGURATIONS — held-out test batch (seeds 22-25, 125 routes)")
    print("=" * 100)
    print(f"{'tag':18s} {'guard':>7s} {'vs LA':>8s} {'vs GRDY':>8s} "
          f"{'penalis':>8s} {'infs':>5s} {'TW':>5s} {'ms':>6s}  description")
    print("-" * 100)
    for r in rows:
        tw = r.get("tw", (0, 0))[0]
        g = "nominal" if r["guard"] is None else f"{r['guard']}"
        print(f"{r['tag']:18s} {g:>7s} {r.get('med_la', float('nan')):+8.2f} "
              f"{r.get('med_greedy', float('nan')):+8.2f} "
              f"{r.get('med_pen', float('nan')):+8.2f} "
              f"{int(r.get('infeasible', -1)):5d} {tw:5d} "
              f"{r.get('ms', float('nan')):6.1f}  {r['desc']}")
    print(f"\nsaved: {RESULTS}/all_test.json")


if __name__ == "__main__":
    main()
