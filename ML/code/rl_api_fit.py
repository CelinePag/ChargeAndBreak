"""
rl_api_fit.py — fit the policy-iteration correction G on rl_api.py labels
=========================================================================
For a labelled state with candidates j (the policy's own pick first), the
rollouts measured d_j = cost(j) - cost(pick), mean over shared futures, with
standard error se_j; the policy's score predicted s_j - s_pick.  G is fitted
so that

    s_j + G(x_j)  -  ( s_pick + G(x_pick) )   ~   d_j

by regressing, row by row, the residual  r_j = d_j - (s_j - s_pick)  (0 for
the pick) on x_j = [the candidate's feature row, the route-end features after
j (rl_api.walk_feats), their differences to the pick's, s_j - s_pick], with
weights 1 / (se_j^2 + 0.1^2).  The improved policy takes, at a decisive state,
argmin_j s_j + G(x_j) among the candidates (rl_policy.ApiChooser).

Honest offline check: states from route seeds `--hold` (default 17-19) are
held out; on them the one-step gain of the corrected pick, measured by the
labels themselves, is reported with its standard error (FPI round 1:
-0.0002 ± 0.0053 h).

    python ML/code/rl_api_fit.py --data r1_gbt --tag api1_gbt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:                    # `src` on machines without the editable install
    sys.path.insert(0, _ROOT)

import lightgbm as lgb                                               # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
WALK = ["rests_after", "slack_after", "rests_after_w", "slack_after_w"]


def g_inputs(rows, score, walk):
    """G's design matrix for one state's challengers j = 1.. (row 0 is the
    pick, whose corrected advantage is 0 by definition, so G never sees it):
    the challenger's row, its route-end features and their difference to the
    pick's, and the predicted advantage s_j - s_pick.  (Fitting G on the
    pick's row too, with target 0, and subtracting G(pick) at decision time
    let every error at the pick move all challengers at once: the first fit
    changed 49 % of states and lost +0.15 h per state in sample.)"""
    d_walk = walk[1:] - walk[0]
    d_score = (score[1:] - score[0])[:, None]
    return np.hstack([rows[1:], walk[1:], d_walk, d_score]).astype(np.float32)


REST_H = 9.5      # price of one more daily rest in the decomposition (r2 9 h, r1 11 h)


def load(name):
    """Challenger rows of every labelled state.  Returns a dict of arrays:
    X (g_inputs), s (predicted advantage), d (measured), drest (measured
    change in the number of daily rests, mean over futures), se, grp (state
    id), seed (route seed), kind."""
    cols = dict(X=[], s=[], d=[], drest=[], se=[], grp=[], seed=[], kind=[])
    files = sorted(glob.glob(os.path.join(DATA, f"rl_api_{name}", "*.npz")))
    g = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        sd = int(str(z["instance"]).rsplit("_", 1)[1])
        for r in z["recs"]:
            n = len(r["keys"]) - 1
            if n < 1:
                continue
            cols["X"].append(g_inputs(r["rows"], r["score"].astype(float),
                                      r["walk"].astype(float)))
            cols["s"].append((r["score"] - r["score"][0])[1:].astype(float))
            cols["d"].append(r["d_mean"][1:].astype(float))
            cols["drest"].append((r["rests_mean"] - r["rests_mean"][0])[1:].astype(float))
            cols["se"].append(r["d_se"][1:].astype(float))
            cols["grp"] += [g] * n
            cols["seed"] += [sd] * n
            cols["kind"] += [r["kind"] or "random"] * n
            g += 1
    out = {k: (np.vstack(v) if k == "X" else np.concatenate(v) if k in ("s", "d", "drest", "se")
               else np.array(v)) for k, v in cols.items()}
    out["files"] = files
    return out


def one_step_gain(adj, d, grp):
    """Mean measured gain (h) per state of moving to the challenger with the
    lowest corrected advantage `adj`, when that is below 0 (else stay: 0)."""
    gains = []
    for g in np.unique(grp):
        m = np.where(grp == g)[0]
        j = int(np.argmin(adj[m]))
        gains.append(d[m][j] if adj[m][j] < -1e-9 else 0.0)
    gains = np.array(gains)
    return gains.mean(), gains.std(ddof=1) / np.sqrt(len(gains)), (gains != 0).mean(), len(gains)


def summary(name):
    """Headroom straight from the labels, no model: per kind of state, how
    often some candidate beats the policy's pick by more than 2 standard
    errors, and by how much; and the policy's predicted advantage vs the
    measured one, for rest and non-rest candidates (calibration)."""
    files = sorted(glob.glob(os.path.join(DATA, f"rl_api_{name}", "*.npz")))
    by = {}
    cal = {"rest": [], "other": []}
    for f in files:
        z = np.load(f, allow_pickle=True)
        ln = "long" if "Rlong" in str(z["instance"]) else "medium" if "Rmedium" in str(z["instance"]) else "short"
        for r in z["recs"]:
            k = (r["kind"] or "random", ln)
            b = by.setdefault(k, dict(n=0, sig=0, gain=[], best=[]))
            b["n"] += 1
            d, se = r["d_mean"], r["d_se"]
            sig = [j for j in range(1, len(d)) if d[j] + 2 * se[j] < 0]
            if sig:
                b["sig"] += 1
                b["gain"].append(min(d[j] for j in sig))
            b["best"].append(min(0.0, float(d[1:].min())) if len(d) > 1 else 0.0)
            for j in range(1, len(d)):
                rest = r["keys"][j].endswith(("_r1", "_r2"))
                cal["rest" if rest else "other"].append(
                    (float(r["score"][j] - r["score"][0]), float(d[j])))
    print(f"{name}: {len(files)} routes")
    print(f"{'state kind':16s} {'states':>7s} {'pick beaten (2 se)':>19s} "
          f"{'mean gain there h':>18s} {'mean best d h':>14s}")
    for k in sorted(by):
        b = by[k]
        print(f"{k[0] + '/' + k[1]:16s} {b['n']:7d} {100 * b['sig'] / b['n']:18.1f}% "
              f"{(np.mean(b['gain']) if b['gain'] else 0):+18.3f} {np.mean(b['best']):+14.3f}")
    for k, v in cal.items():
        if v:
            v = np.array(v)
            print(f"calibration {k:5s}: predicted adv {v[:, 0].mean():+.3f} h, measured "
                  f"{v[:, 1].mean():+.3f} h, mean(measured - predicted) "
                  f"{(v[:, 1] - v[:, 0]).mean():+.3f} h  (n={len(v)})")


def main():
    if "--summary" in sys.argv:
        summary(sys.argv[sys.argv.index("--summary") + 1])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hold", default="17,18,19")
    ap.add_argument("--leaves", type=int, default=8)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--smax", type=float, default=3.0,
                    help="train on challengers predicted within this many hours of "
                         "the pick only: far-off rests (+7-10 h) need corrections "
                         "that leaked into near-contested rests (in-sample "
                         "over-correction -0.86 vs -0.33 h, 2026-10-26)")
    a = ap.parse_args()
    D = load(a.data)
    keep = D["s"] <= a.smax
    for k in ("X", "s", "d", "drest", "se", "grp", "seed", "kind"):
        D[k] = D[k][keep]
    X, s, d, drest, se, grp = D["X"], D["s"], D["d"], D["drest"], D["se"], D["grp"]
    hold = np.isin(D["seed"], [int(x) for x in a.hold.split(",")])
    meta = json.load(open(os.path.join(DATA, f"rl_api_{a.data}", "meta.json")))
    print(f"{len(D['files'])} routes, {len(np.unique(grp))} states, {len(X)} challenger rows; "
          f"held out {len(np.unique(grp[hold]))} states (seeds {a.hold})")
    # decomposition: d = smooth + REST_H * drest.  G_s corrects the student's
    # prediction of the smooth part (low noise), G_r predicts drest (the cliff)
    y_s = (d - REST_H * drest) - s
    w = 1.0 / (se ** 2 + 0.1 ** 2)
    base = dict(learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                verbose=-1, num_threads=4, seed=0)
    models = {}
    for nm, y, obj in (("s", y_s, dict(objective="huber", alpha=0.5)),
                       ("r", drest, dict(objective="l2"))):
        tr = lgb.Dataset(X[~hold], y[~hold], weight=w[~hold])
        va = lgb.Dataset(X[hold], y[hold], weight=w[hold])
        models[nm] = lgb.train(dict(base, **obj), tr, num_boost_round=a.rounds,
                               valid_sets=[va],
                               callbacks=[lgb.early_stopping(50, verbose=False)])
        p = models[nm].predict(X[hold])
        print(f"  G_{nm}: best iteration {models[nm].best_iteration}, held-out "
              f"corr(pred, target) {np.corrcoef(p, y[hold])[0, 1]:+.3f}")
    for nm, m in (("in-sample", ~hold), ("held out", hold)):
        gs, gr = models["s"].predict(X[m]), models["r"].predict(X[m])
        rows = [("student + G_s + G_r", s[m] + gs + REST_H * gr),
                ("student + G_s only", s[m] + gs),
                ("label ceiling (hindsight)", d[m])]
        for lab, adj in rows:
            for delta in ((0.0,) if lab.startswith("label") else (0.0, 0.25, 0.5, 1.0)):
                mu, sem, frac, n = one_step_gain(adj + delta, d[m], grp[m])
                print(f"one-step gain {nm:9s} {lab:26s} margin {delta:4.2f}: {mu:+.4f} ± "
                      f"{sem:.4f} h per state (changes {100 * frac:.1f} % of {n})")
    os.makedirs(MODELS, exist_ok=True)
    models["s"].save_model(os.path.join(MODELS, f"{a.tag}_Gs.txt"))
    models["r"].save_model(os.path.join(MODELS, f"{a.tag}_Gr.txt"))
    with open(os.path.join(MODELS, f"{a.tag}_meta.json"), "w") as fh:
        json.dump(dict(data=a.data, base_kind=meta["kind"], base_tag=meta["tag"],
                       contest=meta["contest"], margin=meta["margin"], topk=meta["topk"],
                       rest_h=REST_H, hold=a.hold,
                       best_iteration={k: m.best_iteration for k, m in models.items()}),
                  fh, indent=1)
    print(f"saved {a.tag}_Gs.txt / _Gr.txt")


if __name__ == "__main__":
    main()
