"""
clf_train.py — the CLASSIFIER arm: the deleted project's framing, re-run fairly
==============================================================================
The 2026-08 ML tree (deleted, recoverable from git) trained a neural network to
predict WHICH of the 12 actions the teacher chose, and reported +0.435% vs the
teacher from cloning alone and +0.186% with DAgger -- far better than this
project's cost-regression MLP manages.  That invites the obvious question: was
classification simply the better framing?

Restoring that code would not answer it.  It used different features (23 + 6K
against this project's 91), a different instance set (772 runs against 830),
different splits (129 validation routes against 136), and its own decision,
forcing and clamp code.  Its number and ours differ in five ways at once, so
placing them in one table would look like evidence and be noise.

So the FRAMING is reimplemented here instead, inside this pipeline:

                        cost-scoring arms          this file
  features              the same 91                identical
  instances / splits    830 / seed-within-family   identical
  legality & forcing    the simulator's own        identical
  decision rule         policy_core argmin         identical
  ------------------------------------------------------------------
  what is learned       cost of EVERY action       WHICH action was chosen
  rows                  411,118 (state, action)    72,595 decisions
  target                regret in hours            a 12-way class label
  model                 MLPRegressor x3            MLPClassifier + tauc head

Only the framing varies, so the difference is attributable.

Why this framing might win
--------------------------
Cross-entropy never meets the regression arm's problem: the regret target has
median 0.655 h against a p90 of 11 h, and sklearn's MLPRegressor is hard-wired
to squared error, so one tail row counts for ~280 median rows.  A classifier
sidesteps that entirely.

Why it might lose
-----------------
It inherits the imbalance instead: 88.8% of decisions are "just drive on", and
the cost of confusing classes is wildly unequal (a spurious daily rest is
9-11 h, a b15/b45 mix-up is minutes) while cross-entropy treats them alike.
The deleted project's single largest debugging result was exactly this: its
class weighting put "drive on" at 0.035 and a two-example class at 5.3, a 150x
ratio that made predicting an 11-hour rest nearly free.  It ended up best with
NO weighting, which `--class-weight none` (the default here) reproduces.

At deployment the classifier scores the STATE once and reads off a probability
per action; `policy_core` then takes its usual argmin over `-log p`, masked to
the legally enumerated actions.  The decision rule is untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib                                                      # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor     # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402

from dataset import MODELS, decision_id, load, split_masks         # noqa: E402


def decision_table(d):
    """One row per DECISION: state features only, label = the chosen action.

    The cost-scoring arms use one row per (state, action); a classifier needs
    one row per decision, with the action as the target rather than an input.
    """
    chosen = d["chosen"].astype(bool)
    n_state = int(d["n_state"])
    X = d["X"][chosen][:, :n_state].astype(np.float64)
    y = d["action_ix"][chosen].astype(int)
    seed = d["seed"][chosen]
    inst = d["instance_ix"][chosen]
    did = decision_id(d)[chosen]
    tauc = d["tauc"][chosen].astype(np.float64)
    # a decision can appear once only; the teacher chose one action
    _, first = np.unique(did, return_index=True)
    first.sort()
    return X[first], y[first], seed[first], inst[first], tauc[first]


def fit_grouped(net, Xtr, ytr, Xva, yva, epochs, patience, label,
                classes=None, score="mse"):
    """Train epoch by epoch, early-stopping on a GROUPED validation set.

    sklearn's own `early_stopping=True` holds out a RANDOM fraction of rows;
    with ~88 correlated decisions per route that would leak badly.
    """
    best, best_ep, best_score, hist = None, -1, np.inf, []
    t0 = time.time()
    for ep in range(epochs):
        if classes is not None and ep == 0:
            net.partial_fit(Xtr, ytr, classes=classes)
        else:
            net.partial_fit(Xtr, ytr)
        if score == "mse":
            s = float(np.mean((net.predict(Xva) - yva) ** 2))
        else:
            p = np.clip(net.predict_proba(Xva), 1e-9, 1.0)
            s = float(-np.mean(np.log(p[np.arange(len(yva)), yva])))
        hist.append(s)
        if s < best_score - 1e-9:
            best_score, best_ep, best = s, ep, copy.deepcopy(net)
        elif ep - best_ep >= patience:
            break
    print(f"[{label}] epochs={len(hist)} best={best_ep + 1} "
          f"val_{score}={best_score:.6f}  {time.time()-t0:.0f}s")
    return best, best_ep + 1, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--tag", default="clf")
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-weight", choices=["none", "sqrt", "balanced"],
                    default="none",
                    help="the deleted project's biggest finding was that "
                         "weighting HURT: (N/n_c)**0.5 put 'drive on' at 0.035 "
                         "against 5.3 for a two-example class, so a spurious "
                         "11-hour rest became nearly free. 'none' reproduces "
                         "its best setting; the others reproduce the failure.")
    args = ap.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(","))
    os.makedirs(MODELS, exist_ok=True)
    d = load(args.data)
    names = [str(x) for x in d["feature_names"]]
    n_state = int(d["n_state"])
    vocab = [str(x) for x in d["action_vocab"]]

    X, y, seed, inst, tauc = decision_table(d)
    tr = np.isin(seed, list(range(1, 18)))
    va = np.isin(seed, list(range(18, 22)))
    te = np.isin(seed, list(range(22, 26)))
    print(f"[data] {len(X)} decisions, {n_state} state features")
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        print(f"  {nm:5s} {m.sum():6d} decisions  "
              f"{len(np.unique(inst[m])):4d} routes")

    mix = np.bincount(y, minlength=len(vocab))
    print("\nclass mix (train):")
    for i in np.argsort(-mix):
        if mix[i]:
            print(f"   {vocab[i]:9s} {mix[i]:6d}  {100*mix[i]/mix.sum():5.2f}%")

    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)

    # sklearn's MLPClassifier has no class_weight, so weighting is applied by
    # RESAMPLING the training rows -- the same effect, and it keeps the
    # ablation honest against the deleted project's `--cw-power`.
    idx = np.flatnonzero(tr)
    if args.class_weight != "none":
        cnt = np.bincount(y[tr], minlength=len(vocab)).astype(float)
        cnt[cnt == 0] = np.inf
        p = (cnt.sum() / cnt) ** (0.5 if args.class_weight == "sqrt" else 1.0)
        w = p[y[tr]]
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(idx, size=len(idx), replace=True, p=w / w.sum())
        print(f"\n[weight] resampled with {args.class_weight} weights")

    t0 = time.time()
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                        solver="adam", alpha=args.alpha,
                        learning_rate_init=args.lr, batch_size=args.batch,
                        random_state=args.seed, early_stopping=False,
                        max_iter=1, warm_start=True)
    clf, ep_c, s_c = fit_grouped(clf, Xs[idx], y[idx], Xs[va], y[va],
                                 args.epochs, args.patience, "class",
                                 classes=np.arange(len(vocab)),
                                 score="logloss")

    # charge duration: state-only head, trained where the teacher charged
    chg = np.array([vocab[i].startswith("y1") for i in y])
    reg = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                       solver="adam", alpha=args.alpha,
                       learning_rate_init=args.lr, batch_size=args.batch,
                       random_state=args.seed, early_stopping=False,
                       max_iter=1, warm_start=True)
    reg, ep_t, s_t = fit_grouped(reg, Xs[tr & chg], tauc[tr & chg],
                                 Xs[va & chg], tauc[va & chg],
                                 args.epochs, args.patience, "tauc")
    print(f"[tauc] val_MAE="
          f"{np.abs(reg.predict(Xs[va & chg]) - tauc[va & chg]).mean()*60:.2f} min")

    # offline: top-1 against the teacher's chosen action, and balanced accuracy
    for nm, m in (("train", tr), ("val", va), ("test", te)):
        pred = clf.predict(Xs[m])
        acc = float((pred == y[m]).mean())
        bal = float(np.mean([
            (pred[y[m] == c] == c).mean()
            for c in np.unique(y[m]) if (y[m] == c).sum()]))
        print(f"[offline] {nm:5s} top-1 {100*acc:5.2f}%   balanced {100*bal:5.2f}%")

    joblib.dump(dict(scaler=scaler, clf=clf, tauc=reg, vocab=vocab),
                os.path.join(MODELS, f"{args.tag}_clf.joblib"))
    with open(os.path.join(MODELS, f"{args.tag}_meta.json"), "w") as fh:
        json.dump(dict(tag=args.tag, kind="clf", args=vars(args),
                       features=names, n_state=n_state, vocab=vocab,
                       epochs=dict(clf=ep_c, tauc=ep_t)), fh, indent=1)
    print(f"\n[saved] {MODELS}/{args.tag}_clf.joblib  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
