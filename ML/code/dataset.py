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


def physics_file(tag):
    """Where extract.py writes one physics value's rows."""
    return os.path.join(DATA, "dataset.npz" if tag == "base"
                        else f"dataset_phys_{tag}.npz")


_ROWWISE = ("action_ix", "regret", "cost", "std", "ok", "n_scen", "tauc",
            "taub", "clean", "stop", "chosen", "tiebreak", "best_cost",
            "n_actions", "n_clean", "seed")


def load_multi(tags, columns=None):
    """Join several physics values' files into one dataset dict.

    `columns`: feature names to keep (default all); only these are ever held
    for the whole join, which is what keeps ~1.3 M rows inside a laptop's
    memory -- X is preallocated and filled one file at a time.  Adds
    `physics_ix` per row and `physics_names`; instance and family indices are
    re-based onto the joined lists.  A file written before the physics tag
    existed (the original dataset.npz) is read as 'base'.
    """
    files = [physics_file(t) for t in tags]
    heads = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            heads.append(dict(n=len(z["stop"]),
                              names=[str(x) for x in z["feature_names"]],
                              n_state=int(z["n_state"]),
                              vocab=[str(x) for x in z["action_vocab"]]))
    all_names = heads[0]["names"]
    for h, f in zip(heads, files):
        if h["names"] != all_names or h["vocab"] != heads[0]["vocab"]:
            raise ValueError(f"{f}: feature names or action vocabulary differ")
    keep = all_names if columns is None else [n for n in all_names if n in set(columns)]
    col_ix = [all_names.index(n) for n in keep]
    n_state = sum(1 for n in keep if all_names.index(n) < heads[0]["n_state"])

    N = sum(h["n"] for h in heads)
    X = np.empty((N, len(keep)), dtype=np.float32)
    out = {k: [] for k in _ROWWISE}
    inst_ix, fam_rows, phys_ix, instances = [], [], [], []
    at = 0
    for p, (tag, f, h) in enumerate(zip(tags, files, heads)):
        with np.load(f, allow_pickle=True) as z:
            X[at:at + h["n"]] = z["X"][:, col_ix]
            for k in _ROWWISE:
                out[k].append(z[k])
            inst_ix.append(z["instance_ix"].astype(np.int32) + len(instances))
            fams = [str(x) for x in z["families"]]
            fam_rows.append(np.array(fams, dtype=object)[z["family_ix"]])
            instances += [str(x) for x in z["instances"]]
            file_tag = str(z["physics"]) if "physics" in z.files else "base"
            if file_tag != tag:
                raise ValueError(f"{f} holds physics {file_tag!r}, not {tag!r}")
        phys_ix.append(np.full(h["n"], p, dtype=np.int8))
        at += h["n"]

    fam_all = np.concatenate(fam_rows)
    families = sorted(set(fam_all))
    fam_map = {n: i for i, n in enumerate(families)}
    d = {k: np.concatenate(v) for k, v in out.items()}
    d.update(
        X=X, feature_names=np.array(keep), n_state=n_state,
        instance_ix=np.concatenate(inst_ix),
        family_ix=np.array([fam_map[n] for n in fam_all], dtype=np.int16),
        instances=np.array(instances), families=np.array(families),
        action_vocab=np.array(heads[0]["vocab"]),
        physics_ix=np.concatenate(phys_ix), physics_names=np.array(tags),
    )
    return d


def row_physics(d):
    """Per-ROW physics tag ('base' for a single-file dataset)."""
    if "physics_ix" not in d:
        return np.full(len(d["stop"]), "base", dtype=object)
    return np.asarray(d["physics_names"]).astype(object)[d["physics_ix"]]


def decision_id(d):
    """A stable integer id per (instance, stop)."""
    return d["instance_ix"].astype(np.int64) * 100_000 + d["stop"].astype(np.int64)


def route_class(d):
    """Per-ROW route length class: 'short' | 'medium' | 'long'."""
    fams = [str(x) for x in d["families"]]
    cls = np.array([f[1:].split("C")[0] for f in fams])
    return cls[d["family_ix"]]


# Training SCOPE: which route lengths the model may learn from.
#   all  every length (the default)
#   SM   short + medium only -- long routes are never seen, so any long route
#        is a genuine extrapolation test (see run_length.py).  Models trained
#        this way carry `_SM` before the seed in their name.
SCOPES = {"all": ("short", "medium", "long"), "SM": ("short", "medium")}


def split_masks(d, scope="all", hold_out=(), fit_seeds=None):
    """(fit, stop, test) row masks, restricted to the training scope.

    `hold_out`: physics tags the model may not learn from (leave-one-value-
    out); their rows leave fit AND stop, so early stopping cannot peek.
    `fit_seeds`: a subset of FIT_SEEDS to learn from (the learning curve);
    stop and test are unchanged, so every point is measured the same way.

    The TEST mask is never restricted: which routes a model is evaluated on
    is the evaluator's choice, not the trainer's.
    """
    s = d["seed"]
    inscope = np.isin(route_class(d), SCOPES[scope])
    if hold_out:
        inscope &= ~np.isin(row_physics(d), list(hold_out))
    fit = FIT_SEEDS if fit_seeds is None else set(fit_seeds) & FIT_SEEDS
    return (np.isin(s, list(fit)) & inscope,
            np.isin(s, list(STOP_SEEDS)) & inscope,
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


def split_report(d, scope="all", hold_out=(), fit_seeds=None):
    masks = split_masks(d, scope, hold_out, fit_seeds)
    print(f"[scope] {scope}: training on {', '.join(SCOPES[scope])} routes")
    if "physics_names" in d:
        phys = [str(t) for t in d["physics_names"]]
        print(f"[physics] learning from {', '.join(t for t in phys if t not in hold_out)}"
              + (f"; held out: {', '.join(hold_out)}" if hold_out else ""))
    print(f"[data] rows {len(d['X'])}  features {d['X'].shape[1]}")
    for nm, m in zip(("fit", "stop", "test"), masks):
        print(f"  {nm:5s} rows {m.sum():7d}  instances "
              f"{len(np.unique(d['instance_ix'][m])):4d}  "
              f"decisions {len(np.unique(decision_id(d)[m])):6d}")
    return masks
