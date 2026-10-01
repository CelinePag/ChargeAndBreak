"""
run_phys.py — trees trained across physics, leave-one-value-out
===============================================================
The base-case students were trained on ONE physics (500 kWh, 350 kW,
chargers ~60 km apart) and lose 1-5 pp to the teacher when that changes
(ood_eval.py).  Nothing in their training data varied it, so they could not
have learned how a decision should depend on it.  This trains the same trees
(F95, same hyperparameters, same seed) on the teacher's runs across physics:

    phys             every value: base + kwh300/700/900, kw150/700/1000,
                     cs30/cs100
    phys_LO<v>       every value except <v>, which stays out of fit AND
                     early stopping -- the model has never seen it

    held out    what it tests
    kwh700      interpolation   (300, 500, 900 kWh seen)
    kw700       interpolation   (150, 350, 1000 kW seen)
    kwh900      extrapolation up      kwh300  extrapolation down
    kw1000      extrapolation up      kw150   extrapolation down
    cs100       extrapolation up      (30 and 60 km seen)

cs30 is only ever trained on: the teacher finished 36 of its routes and 6
test routes, too few to test on.

One model per subprocess, so each releases its memory before the next
(~1.3 M rows; the laptop has little headroom).  Skips any model already on
disk, so a rerun resumes.

    python ML/code/run_phys.py [--only phys,phys_LOkwh700] [--seed 1]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))
LOGS = os.path.abspath(os.path.join(HERE, "..", "logs"))

ALL = ["base", "kwh300", "kwh700", "kwh900", "kw150", "kw700", "kw1000",
       "cs30", "cs100"]
HOLD_OUT = ["kwh700", "kw700", "kw150", "kw1000", "kwh900", "kwh300", "cs100"]

# config name -> held-out physics ('' = none); training order = priority
CONFIGS = {"phys": ""}
CONFIGS.update({f"phys_LO{v}": v for v in HOLD_OUT})


def tag_of(config, fset="F95", seed=1):
    return f"gbt_{fset}_{config}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of configs")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    todo = [c for c in CONFIGS if not args.only or c in args.only.split(",")]
    os.makedirs(LOGS, exist_ok=True)
    for cfg in todo:
        tag = tag_of(cfg, seed=args.seed)
        if os.path.exists(os.path.join(MODELS, f"{tag}_meta.json")):
            print(f"[skip] {tag}")
            continue
        cmd = [sys.executable, os.path.join(HERE, "gbt_train.py"),
               "--fset", "F", "--seed", str(args.seed), "--tag", tag,
               "--physics", ",".join(ALL)]
        if CONFIGS[cfg]:
            cmd += ["--hold-out", CONFIGS[cfg]]
        log = os.path.join(LOGS, f"train_{tag}.log")
        t0 = time.time()
        print(f"[train] {tag}  ->  {log}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        print(f"        rc={rc}  {time.time()-t0:.0f}s", flush=True)
        if rc != 0:
            raise SystemExit(f"{tag} failed, see {log}")


if __name__ == "__main__":
    main()
