"""
learning_curve.py — how many teacher routes does the student need?
==================================================================
Asked on 2026-10-01 before spending LA time on new training routes: the LA
costs ~0.56 h per short route and ~1.9 h per medium one on this laptop, so the
number of routes it must label is the budget question.

The same trees (F95, the base model's hyperparameters, training seed 1) are
fitted on the FIRST k fitting seeds of every route family of the base case,
k in 2, 4, 7, 12, 19 (= ~70 ... 639 routes), with early stopping and the test
batch unchanged.  Each model drives the base-case test routes (seeds 22-25)
through phys_eval's evaluator: gap to the oracle, paired with the LA, extra
daily rests.  k = 19 is the full base model again, a check that this path
reproduces gbt_F95_base_s1.

    python ML/code/learning_curve.py [--ks 2,4,7,12,19] [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ML = os.path.abspath(os.path.join(HERE, ".."))
MODELS = os.path.join(ML, "models")
STORE = os.path.join(ML, "results", "lc_g99sr.jsonl")


def tag_of(k, seed=1):
    return f"gbt_F95_lc{k}_s{seed}"


def train(k, seed):
    tag = tag_of(k, seed)
    if os.path.exists(os.path.join(MODELS, f"{tag}_meta.json")):
        return tag
    log = os.path.join(ML, "logs", f"train_{tag}.log")
    cmd = [sys.executable, os.path.join(HERE, "gbt_train.py"), "--fset", "F",
           "--seed", str(seed), "--tag", tag, "--fit-seeds", f"1-{k}"]
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if rc:
        raise SystemExit(f"{tag} failed, see {log}")
    print(f"[train] {tag}  {time.time() - t0:.0f}s", flush=True)
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,4,7,12,19")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    from phys_eval import _drive, instances

    ks = [int(k) for k in args.ks.split(",")]
    tags = {k: train(k, args.seed) for k in ks}

    done = set()
    if os.path.exists(STORE):
        with open(STORE) as fh:
            done = {(r["tag"], r["instance"]) for r in map(json.loads, fh) if r}
    jobs = [(tags[k], "base", n, p, 0.99, True) for k in ks
            for n, p in instances("base") if (tags[k], n) not in done]
    print(f"[drive] {len(jobs)} routes", flush=True)
    with open(STORE, "a") as fh:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for row in pool.imap_unordered(_drive, jobs, 2):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
    report(ks, tags)


def _fit_routes(tag):
    """Routes the model learned from, as gbt_train's split report printed it."""
    try:
        with open(os.path.join(ML, "logs", f"train_{tag}.log")) as fh:
            for line in fh:
                if line.strip().startswith("fit") and "instances" in line:
                    return int(line.split("instances")[1].split()[0])
    except OSError:
        pass
    return -1


def report(ks, tags):
    from phys_eval import summarise
    with open(STORE) as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    res = summarise(rows)["base"]
    print("\nLEARNING CURVE — base case, test seeds 22-25 (g99sr); "
          "paired vs LA = mean +/- se of (model - LA) in pp")
    print(f"{'k seeds':>8s} {'routes':>7s} {'median':>8s} {'mean':>8s} "
          f"{'vs LA (pp)':>16s} {'win':>5s} {'+rest':>7s} {'infeas':>7s}")
    for k in ks:
        r = res.get(tags[k])
        if not r:
            continue
        meta = json.load(open(os.path.join(MODELS, f"{tags[k]}_meta.json")))
        n_routes = _fit_routes(tags[k])
        p = r["vs_LA"]
        print(f"{k:8d} {n_routes:7d} {r['median']:+8.2f} {r['mean']:+8.2f} "
              f"{p['mean']:+8.2f} +/- {p['se']:.2f} {100 * p['win']:4.0f}% "
              f"{r['extra_rests']:3d}/{r['rest_n']:<3d} {r['infeasible']:7d}"
              f"   (cost head {meta['trees']['cost']} trees)")
    la = res.get("LA")
    if la:
        print(f"{'LA':>8s} {'':7s} {la['median']:+8.2f} {la['mean']:+8.2f} "
              f"{'':16s} {'':5s} {la['extra_rests']:3d}/{la['rest_n']:<3d}")


if __name__ == "__main__":
    main()
