"""
nn_train.py — the NEURAL arm: the simplest network that could work
==================================================================
Counterpart to gbt_train.py.  Deliberately the SAME problem, so that the
comparison measures the function approximator and nothing else:

                       gbt_train.py                 nn_train.py
  formulation          cost-scoring + argmin        identical
  rows                 the same 411k                identical
  targets              regret / feasible / tauc     identical
  splits               seed within family           identical
  legality & forcing   the simulator's own          identical
  offline metrics      dataset.print_offline        identical
  ---------------------------------------------------------------
  model                LightGBM, 3 boosters         sklearn MLP, 3 nets
  feature scaling      none needed (trees are       REQUIRED: a fitted
                       invariant to monotone         StandardScaler ships
                       transforms)                   with the weights
  sample weighting     margin/(margin+2*SEM)        NOT SUPPORTED by
                                                     sklearn's MLP
  early stopping       LightGBM, on grouped val     hand-rolled, on grouped
                                                     val (see below)

"Simplest that could work" means exactly that: three plain multi-layer
perceptrons, one per head, no shared trunk, no listwise loss, no custom
training loop beyond what grouped early stopping requires.  torch is
available in this environment and a shared-trunk network with a listwise loss
over the action set is the obvious next step -- that is the version with a
real structural advantage over trees -- but it is not the simplest thing that
could work, so it is not this file.

Two things this file must get right, or the comparison is worthless
-------------------------------------------------------------------
1. SCALING.  Features span kWh, hours, km, counts, signed slacks and bounded
   fractions.  A tree does not care; a network does.  StandardScaler is fitted
   on the TRAINING ROWS ONLY and saved in the checkpoint, so serving applies
   exactly the transform training saw.

2. EARLY STOPPING MUST BE GROUPED.  sklearn's `early_stopping=True` holds out
   a RANDOM fraction of ROWS.  Rows within a route are a Markov chain, so a
   random row split puts ~88 near-duplicates of every validation row into the
   training set: the internal validation score would be wildly optimistic and
   training would stop far too late.  Instead `early_stopping=False` and the
   loop below calls partial_fit epoch by epoch, scoring after each on the real
   seed-held-out validation routes, and keeps the best weights.

Sample weighting is dropped rather than approximated.  sklearn's MLP has no
`sample_weight`, and the GBT ablation found the margin weighting inert on its
own (`noweight` -0.16% vs `base` -0.20%), so nothing known to matter is lost.
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

from dataset import (MODELS, load, print_offline, split_masks,     # noqa: E402
                     split_report)


def fit_grouped(net, Xtr, ytr, Xva, yva, epochs, patience, label,
                classes=None, score="mse"):
    """Train epoch by epoch, early-stopping on a GROUPED validation set.

    Returns (best_net, best_epoch, best_score, history).
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
        else:                                   # log loss for the classifier
            p = np.clip(net.predict_proba(Xva)[:, 1], 1e-7, 1 - 1e-7)
            s = float(-np.mean(yva * np.log(p) + (1 - yva) * np.log(1 - p)))
        hist.append(s)
        if s < best_score - 1e-9:
            best_score, best_ep, best = s, ep, copy.deepcopy(net)
        elif ep - best_ep >= patience:
            break
    print(f"[{label}] epochs={len(hist)} best={best_ep + 1} "
          f"val_{score}={best_score:.6f}  {time.time()-t0:.0f}s")
    return best, best_ep + 1, best_score, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--hidden", default="64,64",
                    help="comma-separated layer widths")
    ap.add_argument("--alpha", type=float, default=1e-4, help="L2 penalty")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=["regret", "cost"], default="regret",
                    help="what the cost head regresses.  'cost' reproduces "
                         "gbt_train.py's `rawcost` ablation on this arm: the "
                         "teacher's raw horizon objective, whose LEVEL has "
                         "std 29.6 h against a decision MARGIN of ~40 min.")
    ap.add_argument("--target-transform", choices=["none", "log1p"],
                    default="log1p",
                    help="log1p compresses the regret tail (p50 0.66 h vs p90 "
                         "11 h).  sklearn's MLPRegressor is hard-wired to "
                         "SQUARED ERROR, while the GBT arm uses Huber, so "
                         "without this the network is the only arm left "
                         "unprotected against the tail -- an artefact of the "
                         "library, not a property of neural networks.  The "
                         "transform is MONOTONE, so the argmin the policy "
                         "takes is unchanged; it is inverted at serving.")
    args = ap.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(","))
    os.makedirs(MODELS, exist_ok=True)
    d = load(args.data)
    X = d["X"].astype(np.float64)
    names = [str(x) for x in d["feature_names"]]
    tr, va, te = split_report(d)
    clean = d["clean"].astype(bool)

    # -- scaling: fitted on TRAINING ROWS ONLY, shipped with the weights -----
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)
    print(f"[scale] StandardScaler fitted on {tr.sum()} training rows "
          f"({hidden} hidden units, alpha={args.alpha})")

    def net(kind):
        cls = MLPClassifier if kind == "clf" else MLPRegressor
        return cls(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                   alpha=args.alpha, learning_rate_init=args.lr,
                   batch_size=args.batch, random_state=args.seed,
                   early_stopping=False,      # see the module docstring
                   max_iter=1, warm_start=True)

    t0 = time.time()
    out = {}

    # -- HEAD 1: cost head, clean rows only ----------------------------------
    y = d["regret"] if args.target == "regret" else d["cost"]
    if args.target_transform == "log1p":
        y = np.log1p(np.maximum(y, 0.0))
    m_tr, m_va = tr & clean & np.isfinite(y), va & clean & np.isfinite(y)
    cost_m, cost_ep, cost_s, cost_hist = fit_grouped(
        net("reg"), Xs[m_tr], y[m_tr], Xs[m_va], y[m_va],
        args.epochs, args.patience, "cost")

    # -- HEAD 2: feasibility, all rows ---------------------------------------
    yf = clean.astype(int)
    feas_m, feas_ep, feas_s, _ = fit_grouped(
        net("clf"), Xs[tr], yf[tr], Xs[va], yf[va],
        args.epochs, args.patience, "feas",
        classes=np.array([0, 1]), score="logloss")

    # -- HEAD 3: charge duration, y=1 rows only ------------------------------
    is_y1 = X[:, names.index("a_y")] > 0.5
    c_tr, c_va = tr & is_y1 & clean, va & is_y1 & clean
    tauc_m, tauc_ep, tauc_s, _ = fit_grouped(
        net("reg"), Xs[c_tr], d["tauc"][c_tr], Xs[c_va], d["tauc"][c_va],
        args.epochs, args.patience, "tauc")
    pv = tauc_m.predict(Xs[c_va])
    print(f"[tauc] val_MAE={np.abs(pv - d['tauc'][c_va]).mean()*60:.2f} min "
          f"(on {c_va.sum()} charge rows)")

    # -- offline metrics, identical code to the GBT arm ----------------------
    pred = cost_m.predict(Xs)
    if args.target_transform == "log1p":
        pred = np.expm1(pred)          # back to hours for the offline metrics
    feas = feas_m.predict_proba(Xs)[:, 1]
    print_offline(d, pred, feas, (tr, va, te))

    tag = args.tag
    joblib.dump(dict(scaler=scaler, cost=cost_m, feas=feas_m, tauc=tauc_m),
                os.path.join(MODELS, f"{tag}_nn.joblib"))
    with open(os.path.join(MODELS, f"{tag}_meta.json"), "w") as fh:
        json.dump(dict(tag=tag, kind="nn", args=vars(args), features=names,
                       n_state=int(d["n_state"]),
                       target=args.target,
                       target_transform=args.target_transform,
                       epochs=dict(cost=cost_ep, feas=feas_ep, tauc=tauc_ep),
                       val=dict(cost_mse=cost_s, feas_logloss=feas_s,
                                tauc_mse=tauc_s)), fh, indent=1)
    n_par = sum(w.size for w in cost_m.coefs_) + \
        sum(b.size for b in cost_m.intercepts_)
    print(f"\n[saved] {MODELS}/{tag}_nn.joblib  ({time.time()-t0:.0f}s)")
    print(f"[size]  cost head: {n_par:,} parameters "
          f"({len(names)} inputs -> {hidden} -> 1)")


if __name__ == "__main__":
    main()
