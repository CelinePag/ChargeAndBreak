"""
run_ladder.py — every arm, every feature set, several training seeds
=====================================================================
The feature-set ladder (see `fsets.py`):

    C compact  <  D dedup  <  F full  <  L full + raw lookahead

run for each arm, so the data -- not an assumption about what each model
family "prefers" -- says how many inputs each one should get.  Each cell is
repeated over several TRAINING seeds (the model's initialisation; unrelated to
the ROUTE seeds that index instances), so a difference between two cells can
be read against its own spread.

Protocol, identical for every cell: fitted on route seeds 1-19, early-stopped
on 20-21, reported on the whole test batch 22-25 (125 routes).

Every model is named  <arm>_<SET><n>_base_s<seed>,  where n is the number of
inputs it actually consumes -- so `gbt_L215_base_s1` and `clf_L197_base_s1`
are the same feature set seen by two different model families.

Resumable: a model whose checkpoint exists is not retrained, and an evaluation
whose JSON exists is not re-run.  Results are summarised from the evaluation
files themselves, not parsed from printed output.

    python ML/code/run_ladder.py --arms gbt,clf,mlp --fsets C,D,F,L --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsets import label                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
STORE = os.path.join(RESULTS, "ladder_test.json")
PY = sys.executable
GUARD = "0.95"
BETA = 0.5

ARM = {
    "gbt": dict(trainer="gbt_train.py", kind="gbt", ck="{t}_cost.txt",
                flags=(), label="Trees"),
    "mlp": dict(trainer="nn_train.py", kind="nn", ck="{t}_nn.joblib",
                flags=("--target-transform", "log1p"), label="MLP"),
    "clf": dict(trainer="clf_train.py", kind="clf", ck="{t}_clf.joblib",
                flags=("--class-weight", "none"), label="Classifier"),
}


def summarise(eval_file):
    """The headline numbers, computed from an evaluation JSON."""
    with open(os.path.join(RESULTS, eval_file)) as fh:
        rows = json.load(fh)
    comp = [r for r in rows if r.get("route_completed")]

    def med(key, pen=False):
        v = []
        for r in comp:
            b = r.get(key)
            if b is None or r.get(f"{key}_infeasible"):
                continue
            if pen:
                a = r["duration_h"] + BETA * r.get("tw_misses", 0)
                b = b + BETA * r.get(f"{key}_tw", 0)
            else:
                a = r["duration_h"]
            v.append(100.0 * (a - b) / b)
        return float(np.median(v)) if v else float("nan")

    return dict(med_la=med("LA"), med_greedy=med("GREEDY"),
                med_pen=med("LA", pen=True),
                infeasible=len(rows) - len(comp),
                tw=sum(r.get("tw_misses", 0) for r in rows),
                tw_la=sum(r.get("LA_tw", 0) for r in rows),
                routes=len(rows))


def run(cmd, what):
    print(f"   $ {what}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-1200:])
        print(p.stderr[-1200:])
        raise SystemExit(f"FAILED: {what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gbt,clf,mlp")
    ap.add_argument("--fsets", default="C,D,F,L")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    all_names = [str(x) for x in d["feature_names"]]
    all_ns = int(d["n_state"])

    rows = []
    if os.path.exists(STORE):
        with open(STORE) as fh:
            rows = json.load(fh)
    done = {r["tag"] for r in rows}

    for arm in args.arms.split(","):
        a = ARM[arm]
        for fs in args.fsets.split(","):
            lab = label(fs, arm, all_names, all_ns)
            for sd in range(args.seeds):
                tag = f"{arm}_{lab}_base_s{sd}"
                if tag in done:
                    print(f"=== {tag}: already in the ladder, skipped", flush=True)
                    continue
                print(f"\n=== {tag} ({a['label']}, set {fs} = {lab}, "
                      f"training seed {sd}) ===", flush=True)
                t0 = time.time()
                if not os.path.exists(os.path.join(MODELS, a["ck"].format(t=tag))):
                    run([PY, os.path.join(HERE, a["trainer"]), "--tag", tag,
                         "--fset", fs, "--seed", str(sd)] + list(a["flags"]),
                        f"train {tag}")
                ev = f"eval_{tag}_g95_test.json"
                if not os.path.exists(os.path.join(RESULTS, ev)):
                    run([PY, os.path.join(HERE, "evaluate.py"), "--kind",
                         a["kind"], "--tag", tag, "--split", "test",
                         "--guard-q", GUARD, "--out", ev], f"evaluate {tag}")
                r = summarise(ev)
                r.update(arm=arm, arm_label=a["label"], fset=fs, fset_label=lab,
                         n_features=int(lab[1:]), seed=sd, tag=tag,
                         eval_file=ev, seconds=round(time.time() - t0, 1))
                rows.append(r)
                done.add(tag)
                with open(STORE, "w") as fh:
                    json.dump(rows, fh, indent=1)
                print(f"   -> vs LA {r['med_la']:+.2f}%  infeasible "
                      f"{r['infeasible']}  TW {r['tw']}  ({r['seconds']}s)",
                      flush=True)

    table(rows)


def table(rows):
    print("\n" + "=" * 88)
    print("FEATURE-SET LADDER — mean ± sd over training seeds, test batch (125 routes)")
    print("=" * 88)
    print(f"{'arm':11s} {'set':6s} {'n':>4s} {'seeds':>5s} {'vs LA':>15s} "
          f"{'vs Greedy':>15s} {'infeasible':>11s} {'TW':>9s}")
    print("-" * 88)
    order = {"C": 0, "D": 1, "F": 2, "L": 3, "F91": 4}
    for arm in ("gbt", "clf", "mlp"):
        cells = sorted({(r["fset"], r["fset_label"]) for r in rows
                        if r["arm"] == arm}, key=lambda c: order.get(c[0], 9))
        for fs, lab in cells:
            sub = [r for r in rows if r["arm"] == arm and r["fset"] == fs]
            la = np.array([r["med_la"] for r in sub])
            gr = np.array([r["med_greedy"] for r in sub])
            inf = np.array([r["infeasible"] for r in sub], float)
            tw = np.array([r["tw"] for r in sub], float)
            print(f"{sub[0]['arm_label']:11s} {lab:6s} {sub[0]['n_features']:4d} "
                  f"{len(sub):5d} {la.mean():+7.2f} ± {la.std():4.2f} "
                  f"{gr.mean():+7.2f} ± {gr.std():4.2f} "
                  f"{inf.mean():6.1f} ± {inf.std():3.1f} {tw.mean():5.0f} ± {tw.std():2.0f}")
        print()
    if rows:
        print(f"(teacher window misses on the same routes: {rows[0]['tw_la']})")
    print(f"saved: {STORE}")


if __name__ == "__main__":
    main()
