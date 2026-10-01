"""
fpi_train.py — fit the correction G on the branch labels of fpi_collect.py
==========================================================================
    python ML/code/fpi_train.py --data r1 --tag fpi1_gbt_F95_base_s1_g99sr

At every labelled decision the student's k best actions were each driven to
the destination under one shared draw of the remaining travel times.  The
MEASURED advantage of a branch is its route cost minus the cost of the
student's own branch, clipped to +-24 h (a branch that breaks a rule costs
FAIL_H).  The student's cost head PREDICTS an advantage for the same action:
its regret minus the regret of its own choice.  G is fitted to the difference
(hours), with squared loss: the argmin needs the MEAN, and the rare whole-rest
flips (+-10 h) that a robust loss would discount are exactly the signal.

The split is dataset.py's: seeds 1-19 fit, 20-21 stop the boosting early.
Nothing is selected on the test routes.

Offline check, on the stopping routes: at each decision, take the improved
choice (argmin of cost head + G among the candidates) and the measured
advantage of that choice in that decision's draw.  Its mean over decisions is
an unbiased estimate of what one step of improvement gains per decision --
before a single route is driven with the improved policy.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb                                              # noqa: E402

from dataset import FIT_SEEDS, STOP_SEEDS                           # noqa: E402
from fpi_policy import CLIP_H, EXTRA                                # noqa: E402
from policy_core import MODELS                                      # noqa: E402
from run_rollout import DATA                                        # noqa: E402


def load_labels(name):
    """Every branch row of a collection, with decision ids made global."""
    files = sorted(glob.glob(os.path.join(DATA, f"fpi_{name}", "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if not files:
        raise SystemExit(f"no label files in {os.path.join(DATA, 'fpi_' + name)}")
    parts, names = [], None
    for k, f in enumerate(files):
        z = np.load(f)
        if names is None:
            names = [str(x) for x in z["names"]]
        n = len(z["C"])
        if n == 0:
            continue
        parts.append(dict(
            X=z["X"], old=z["old"].astype(np.float64), pi=z["pi"], key=z["key"],
            grp=z["grp"].astype(np.int64) + k * 1_000_000, C=z["C"],
            rests=z["rests"], seed=np.full(n, int(z["seed"])),
            cls=np.full(n, str(z["cls"])), inst=np.full(n, k)))
    d = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    d["names"] = names
    d["n_files"] = len(files)
    return d


def labels(d):
    """Measured advantage A, the head's predicted advantage, and G's target."""
    order = np.argsort(d["grp"], kind="stable")
    grp = d["grp"][order]
    start = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    ref_of = np.empty(len(grp), dtype=np.int64)
    for a, b in zip(start, np.r_[start[1:], len(grp)]):
        members = order[a:b]
        ref = members[d["pi"][members]]
        ref_of[members] = ref[0] if len(ref) else -1
    if (ref_of < 0).any():
        raise ValueError("a decision without the student's own branch")
    meas = np.clip(d["C"] - d["C"][ref_of], -CLIP_H, CLIP_H)
    pred = d["old"] - d["old"][ref_of]
    d_rest = d["rests"] - d["rests"][ref_of]
    return meas, pred, meas - pred, ref_of, d_rest


def describe(d, meas, pred, d_rest, alt):
    """Where the student's cost head misprices the alternatives to its own
    choice: measured vs predicted advantage by action and route length, over
    all labelled decisions (hours; SE of the mean difference)."""
    print(f"\n  {'action':7s} {'class':6s} {'n':>6s} {'measured':>9s} {'predicted':>9s} "
          f"{'meas-pred':>16s} {'better':>7s} {'rest -1':>7s} {'rest +1':>7s}")
    keys = sorted({str(k) for k in d["key"][alt]})
    for k in keys:
        for cls in ("long", "medium", "short"):
            m = alt & (d["key"] == k) & (d["cls"] == cls)
            n = int(m.sum())
            if n < 200:
                continue
            r = meas[m] - pred[m]
            print(f"  {k:7s} {cls:6s} {n:6d} {meas[m].mean():+9.3f} {pred[m].mean():+9.3f} "
                  f"{r.mean():+8.3f} +- {r.std(ddof=1) / np.sqrt(n):.3f} "
                  f"{np.mean(meas[m] < -1e-6) * 100:6.1f}% "
                  f"{np.sum(d_rest[m] < 0):7d} {np.sum(d_rest[m] > 0):7d}")
    print("  (actions with fewer than 200 labelled alternatives per class omitted)\n")


def one_step(d, mask, meas, adj, min_gain=0.0):
    """Per decision in `mask`: the corrected argmin among the candidates and
    the measured advantage of that choice.  Returns (mean gain h per decision,
    its SE, share of decisions changed, mean measured advantage of the changed
    ones, decisions)."""
    idx = np.flatnonzero(mask)
    grp = d["grp"][idx]
    order = np.argsort(grp, kind="stable")
    idx, grp = idx[order], grp[order]
    start = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    got, changed = [], []
    for a, b in zip(start, np.r_[start[1:], len(idx)]):
        members = idx[a:b]
        ref = members[d["pi"][members]][0]
        best = members[int(np.argmin(adj[members]))]
        pick = best if adj[best] < adj[ref] - min_gain else ref
        got.append(meas[pick])
        changed.append(pick != ref)
    got, changed = np.array(got), np.array(changed)
    ch = got[changed]
    return (float(got.mean()), float(got.std(ddof=1) / np.sqrt(len(got))),
            float(changed.mean()), float(ch.mean()) if len(ch) else 0.0, len(got))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="collection name (fpi_<name>)")
    ap.add_argument("--tag", required=True, help="name of the saved correction")
    ap.add_argument("--no-extra", action="store_true",
                    help="ablation: G without the two route-end features")
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-child", type=int, default=400)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--early", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    d = load_labels(args.data)
    with open(os.path.join(DATA, f"fpi_{args.data}", "_args.json")) as fh:
        coll = json.load(fh)
    meas, pred, y, ref_of, d_rest = labels(d)
    names = d["names"]
    cols = [i for i, n in enumerate(names) if not (args.no_extra and n in EXTRA)]
    X = d["X"][:, cols]
    feat = [names[i] for i in cols]
    fit = np.isin(d["seed"], list(FIT_SEEDS))
    stop = np.isin(d["seed"], list(STOP_SEEDS))
    alt = ~d["pi"]
    print(f"[fpi-train] {d['n_files']} routes, {len(y)} rows, "
          f"{len(np.unique(d['grp']))} decisions  (fit {fit.sum()} rows, "
          f"stop {stop.sum()} rows)  {len(feat)} features")
    print(f"  measured - predicted advantage, alternatives only: mean "
          f"{y[alt].mean():+.3f} h, sd {y[alt].std():.2f} h; branches that "
          f"change the rest count: {np.mean(d_rest[alt] != 0) * 100:.1f}%")
    describe(d, meas, pred, d_rest, alt)

    params = dict(objective="regression", metric="l2", learning_rate=args.lr,
                  num_leaves=args.leaves, max_depth=args.depth,
                  min_child_samples=args.min_child, lambda_l2=args.l2,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                  verbose=-1, seed=args.seed, num_threads=0)
    ds_fit = lgb.Dataset(X[fit], label=y[fit], feature_name=feat, free_raw_data=False)
    ds_stop = lgb.Dataset(X[stop], label=y[stop], feature_name=feat,
                          reference=ds_fit, free_raw_data=False)
    G = lgb.train(params, ds_fit, num_boost_round=args.rounds,
                  valid_sets=[ds_stop], valid_names=["stop"],
                  callbacks=[lgb.early_stopping(args.early, verbose=False),
                             lgb.log_evaluation(0)])
    g = G.predict(X, num_iteration=G.best_iteration)
    l2_0 = float(np.mean(y[stop] ** 2))
    l2_g = float(np.mean((y[stop] - g[stop]) ** 2))
    print(f"[G] trees={G.best_iteration}  stop-set L2 {l2_g:.4f} vs {l2_0:.4f} "
          f"for G = 0  (R2 {1 - l2_g / l2_0:+.4f})  {time.time() - t0:.0f}s")

    # -- offline: one step of improvement, measured on held-out draws --------
    head_only = d["old"]                  # the student's own ranking
    corrected = d["old"] + g
    print(f"\n{'split':5s} {'class':7s} {'decisions':>9s} {'changed':>8s} "
          f"{'gain h/decision':>18s} {'adv. of changed':>16s}  {'hindsight best':>14s}")
    offline = {}
    for split, m in (("fit", fit), ("stop", stop)):
        for cls in ("all", "long", "medium", "short"):
            mc = m & ((d["cls"] == cls) if cls != "all" else True)
            if not mc.any():
                continue
            gain, se, share, ch, n = one_step(d, mc, meas, corrected)
            base_gain = one_step(d, mc, meas, head_only)[0]
            best = one_step(d, mc, meas, meas)[0]
            offline[f"{split}_{cls}"] = dict(gain=gain, se=se, changed=share,
                                             adv_changed=ch, n=n)
            print(f"{split:5s} {cls:7s} {n:9d} {share * 100:7.1f}% "
                  f"{gain:+8.4f} +- {se:.4f} {ch:+15.3f}  {best:+14.3f}"
                  + (f"   (head alone {base_gain:+.4f})" if abs(base_gain) > 1e-12 else ""))
    print("  gain < 0 is better: measured hours per decision vs the student's own "
          "choice, in draws G was not fitted on (stop) or was (fit).")

    os.makedirs(MODELS, exist_ok=True)
    G.save_model(os.path.join(MODELS, f"{args.tag}_delta.txt"),
                 num_iteration=G.best_iteration)
    meta = dict(tag=args.tag, base=coll["tag"], guard_q=coll["guard_q"],
                spread_room=coll["spread_room"], top_k=coll["top_k"],
                data=f"fpi_{args.data}", features=feat, extra=not args.no_extra,
                rows=int(len(y)), decisions=int(len(np.unique(d["grp"]))),
                params=params, trees=int(G.best_iteration),
                stop_l2=l2_g, stop_l2_zero=l2_0, offline=offline)
    with open(os.path.join(MODELS, f"{args.tag}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\n[saved] {MODELS}/{args.tag}_delta.txt  ({time.time() - t0:.0f}s)")
    imp = sorted(zip(feat, G.feature_importance("gain")), key=lambda x: -x[1])
    print("\ntop 12 features by gain:")
    for n, v in imp[:12]:
        print(f"   {n:26s} {v:12.0f}")


if __name__ == "__main__":
    main()
