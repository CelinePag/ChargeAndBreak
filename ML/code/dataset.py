"""
dataset.py — everything the GBT and neural TRAINERS share
=========================================================
Loading, splitting, weighting and the offline metrics, in one place so both
arms are scored by identical code:

  ML/code/dataset.py     <- this file: data handling + offline metrics
  ML/code/gbt_train.py   <- LightGBM heads
  ML/code/nn_train.py    <- sklearn MLP heads

Splits are by SEED WITHIN FAMILY, never by row: the ~88 decisions of one route
are a Markov chain, so a row-level split leaks almost perfectly.

PROTOCOL (revised): every configuration is its own model, and every
configuration is reported on the WHOLE test batch.

    FIT   seeds 1-19   gradient actually flows here
    STOP  seeds 20-21  early stopping ONLY -- never used to pick between
                       configurations, so it costs nothing in bias
    TEST  seeds 22-25  every configuration is reported here

The earlier design held out seeds 18-21 to CHOOSE a winner and reported only
that winner on test.  That answers "how good is the model we picked"; it does
not answer "how do these designs compare", which is what an ablation table is
for.  Under this protocol each row of the table is an independent model
measured on the same 125 held-out routes, so the rows are directly comparable
and nothing is selected on the test set -- the table reports all of them.

Fitting now uses 19 seeds instead of 17 (~630 routes instead of 569); the
2-seed stopping slice is the smallest that still gives early stopping a
usable signal.
"""
from __future__ import annotations

import os

import numpy as np

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

FIT_SEEDS = set(range(1, 20))        # 1-19  — gradient flows here
STOP_SEEDS = set(range(20, 22))      # 20-21 — early stopping only
TEST_SEEDS = set(range(22, 26))      # 22-25 — every configuration reported here

# kept so older result files and scripts keep resolving
TRAIN_SEEDS, VAL_SEEDS = FIT_SEEDS, STOP_SEEDS


def load(path=None):
    d = np.load(path or os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def decision_id(d):
    """A stable integer id per (instance, stop)."""
    return d["instance_ix"].astype(np.int64) * 100_000 + d["stop"].astype(np.int64)


def split_masks(d):
    """(fit, stop, test) row masks."""
    s = d["seed"]
    return (np.isin(s, list(FIT_SEEDS)),
            np.isin(s, list(STOP_SEEDS)),
            np.isin(s, list(TEST_SEEDS)))


def decision_margins(d):
    """Per-ROW: the margin of its decision (2nd-best minus best clean cost).

    Decisions with fewer than two clean actions get margin = inf: there is
    nothing to confuse, so they are never down-weighted.
    """
    did = decision_id(d)
    cost = d["cost"].copy()
    clean = d["clean"].astype(bool)
    cost[~clean] = np.inf
    order = np.lexsort((cost, did))
    did_s, cost_s = did[order], cost[order]
    margin_s = np.full(len(did_s), np.inf)
    start = np.r_[True, did_s[1:] != did_s[:-1]]
    idx = np.flatnonzero(start)
    for a, b in zip(idx, np.r_[idx[1:], len(did_s)]):
        if b - a >= 2 and np.isfinite(cost_s[a + 1]):
            margin_s[a:b] = cost_s[a + 1] - cost_s[a]
    out = np.empty_like(margin_s)
    out[order] = margin_s
    return out


def sample_weights(d, margin, power=1.0):
    """margin / (margin + 2*SEM): ~0 on coin flips, ~1 where the teacher was sure.

    13.6% of decisions have a margin below twice the teacher's own
    scenario-sampling SEM, so its preference there is not distinguishable from
    which 25 travel-time draws it happened to get.
    """
    sem = d["std"] / np.sqrt(np.maximum(d["n_scen"], 1))
    with np.errstate(invalid="ignore"):
        w = margin / (margin + 2.0 * sem + 1e-9)
    return np.clip(np.nan_to_num(w, nan=1.0, posinf=1.0), 0.01, 1.0) ** power


def argmin_policy_regret(d, mask, pred_cost, feas=None, feas_thr=0.5):
    """Mean realised regret (hours) of choosing argmin(pred) at each decision.

    This is the quantity the cost-sensitive reduction bounds, and the only
    supervised metric expressed in the units the paper reports.  Restricted to
    CLEAN actions, since a dirty action has no true regret.
    """
    did = decision_id(d)[mask]
    clean = d["clean"].astype(bool)[mask]
    true_r = d["regret"][mask]
    pred = pred_cost[mask].astype(np.float64).copy()
    if feas is not None:
        pred = pred + 1e6 * (feas[mask] < feas_thr)
    pred[~clean] = np.inf
    order = np.lexsort((pred, did))
    did_s, r_s = did[order], true_r[order]
    start = np.flatnonzero(np.r_[True, did_s[1:] != did_s[:-1]])
    chosen = r_s[start]
    ok = np.isfinite(chosen)
    return float(np.nanmean(chosen[ok])), int(ok.sum())


def teacher_top1(d, mask, pred_cost):
    """Fraction of decisions where argmin(pred) == the teacher's argmin."""
    did = decision_id(d)[mask]
    clean = d["clean"].astype(bool)[mask]
    true_r = d["regret"][mask]
    pred = pred_cost[mask].astype(np.float64).copy()
    pred[~clean] = np.inf
    order = np.lexsort((pred, did))
    did_s = did[order]
    start = np.flatnonzero(np.r_[True, did_s[1:] != did_s[:-1]])
    return float(np.nanmean(true_r[order][start] < 1e-9))


def print_offline(d, pred, feas, masks, names=("fit", "stop", "test")):
    """The comparison table both arms print, so they are directly comparable."""
    print(f"\n{'='*72}\nOFFLINE POLICY METRICS (argmin over scored actions)\n{'='*72}")
    print(f"{'split':6s} {'mean regret':>12s} {'top-1':>8s} {'decisions':>10s}   "
          f"{'always-go':>11s}")
    go_ix = list(d["action_vocab"]).index("y0_go")
    gopred = np.where(d["action_ix"] == go_ix, 0.0, 1.0)
    for nm, m in zip(names, masks):
        r, n = argmin_policy_regret(d, m, pred, feas)
        t1 = teacher_top1(d, m, pred)
        rg, _ = argmin_policy_regret(d, m, gopred)
        print(f"{nm:6s} {r*60:9.2f} min {t1*100:7.1f}% {n:10d}   {rg*60:8.2f} min")


def split_report(d):
    masks = split_masks(d)
    print(f"[data] rows {len(d['X'])}  features {d['X'].shape[1]}")
    for nm, m in zip(("fit", "stop", "test"), masks):
        print(f"  {nm:5s} rows {m.sum():7d}  instances "
              f"{len(np.unique(d['instance_ix'][m])):4d}  "
              f"decisions {len(np.unique(decision_id(d)[m])):6d}")
    return masks
