"""
run_seeds.py — the same configuration, several times, so a ranking means something
=================================================================================
`run_all.py` trains one model per configuration.  With the three headline arms
separated by 0.2-0.4 pp and the simulator's own practical floor at 0.35%, a
single model per arm cannot support any ordering: the differences are the size
of training noise, and nothing measured so far distinguishes the two.

This trains N seeds of each headline configuration and reports mean +/- spread,
so "the trees are ahead of the MLP" either survives the seed spread or is
withdrawn.

Every model is fitted on seeds 1-19, early-stopped on 20-21, and reported on
the whole test batch (seeds 22-25, 125 routes) -- the same protocol as
`run_all.py`.  NOTE the word "seed" is doing two jobs here and they are
unrelated: the ROUTE seed indexes an instance within its family, while the
TRAINING seed initialises the model.  Only the latter varies below.

    python ML/code/run_seeds.py --seeds 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))
PY = sys.executable

# the three headline configurations, one per arm
HEADS = [
    ("gbt", "base", (), "Trees"),
    ("mlp", "base", ("--target-transform", "log1p"), "MLP"),
    ("clf", "base", ("--class-weight", "none"), "Classifier"),
]
TRAINER = {"gbt": "gbt_train.py", "mlp": "nn_train.py", "clf": "clf_train.py"}
KIND = {"gbt": "gbt", "mlp": "nn", "clf": "clf"}
CHECKPOINT = {"gbt": "{t}_cost.txt", "mlp": "{t}_nn.joblib", "clf": "{t}_clf.joblib"}
GUARD = "0.95"

_NUM = r"([-+]?\d+\.?\d*)"
PATTERNS = dict(
    infeasible=r"^infeasible\s+(\d+)",
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
        print(p.stdout[-1200:])
        print(p.stderr[-1200:])
        raise SystemExit(f"FAILED: {label}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma-separated arms")
    args = ap.parse_args()

    heads = HEADS
    if args.only:
        want = set(args.only.split(","))
        heads = [h for h in heads if h[0] in want]

    # resume: keep rows already produced for arms this invocation is not
    # re-running, so cutting the seed count on one arm never discards another
    store = os.path.join(RESULTS, "seeds_test.json")
    rows = []
    if os.path.exists(store):
        with open(store) as fh:
            prev = json.load(fh)
        keep_arms = {h[0] for h in heads}
        rows = [r for r in prev if r["arm"] not in keep_arms]
        if rows:
            print(f"[resume] keeping {len(rows)} rows from arms not re-run: "
                  f"{sorted({r['arm'] for r in rows})}", flush=True)

    for arm, name, flags, label in heads:
        for sd in range(args.seeds):
            tag = f"{arm}_{name}_s{sd}"
            print(f"\n=== {tag} ({label}, training seed {sd}) ===", flush=True)
            t0 = time.time()
            ck = os.path.join(MODELS, CHECKPOINT[arm].format(t=tag))
            if not os.path.exists(ck):
                run([PY, os.path.join(HERE, TRAINER[arm]), "--tag", tag,
                     "--seed", str(sd)] + list(flags), f"train {tag}")
            out = f"eval_{tag}_g95_test.json"
            txt = run([PY, os.path.join(HERE, "evaluate.py"), "--kind",
                       KIND[arm], "--tag", tag, "--split", "test",
                       "--guard-q", GUARD, "--out", out], f"evaluate {tag}")
            r = parse(txt)
            r.update(arm=arm, label=label, seed=sd, tag=tag, eval_file=out,
                     seconds=round(time.time() - t0, 1))
            rows.append(r)
            print(f"   -> vs LA {r.get('med_la')}%  infeasible "
                  f"{r.get('infeasible')}  ({r['seconds']}s)", flush=True)
            with open(os.path.join(RESULTS, "seeds_test.json"), "w") as fh:
                json.dump(rows, fh, indent=1)

    print("\n" + "=" * 86)
    print(f"SEED SPREAD — {args.seeds} training seeds per arm, held-out test batch")
    print("=" * 86)
    print(f"{'arm':12s} {'vs LA':>18s} {'vs GREEDY':>18s} {'infeasible':>14s} "
          f"{'TW misses':>12s}")
    print("-" * 86)
    for arm in ("gbt", "mlp", "clf"):
        sub = [r for r in rows if r["arm"] == arm]
        label = sub[0]["label"] if sub else arm
        if not sub:
            continue
        la = np.array([r.get("med_la", np.nan) for r in sub], float)
        gr = np.array([r.get("med_greedy", np.nan) for r in sub], float)
        inf = np.array([r.get("infeasible", np.nan) for r in sub], float)
        tw = np.array([(r.get("tw") or [np.nan, np.nan])[0] for r in sub], float)
        print(f"{label:12s} {np.nanmean(la):+8.2f} ± {np.nanstd(la):4.2f} "
              f"{np.nanmean(gr):+11.2f} ± {np.nanstd(gr):4.2f} "
              f"{np.nanmean(inf):9.1f} ± {np.nanstd(inf):3.1f} "
              f"{np.nanmean(tw):8.0f} ± {np.nanstd(tw):3.0f}")
    tw_la = (rows[0].get("tw") or [0, 0])[1] if rows else 0
    print(f"\n(teacher time-window misses on the same routes: {tw_la})")
    print(f"saved: {RESULTS}/seeds_test.json")


if __name__ == "__main__":
    main()
