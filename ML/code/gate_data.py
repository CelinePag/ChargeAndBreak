"""
gate_data.py — one row per LA-visited decision: is calling the LA worth it here?
==============================================================================
Step 1 of the solver-call gate (2026-10-20).  For a student (trees or PyTorch,
LA labels) and every decision of the LA dataset with at least two clean
actions:

  label   vo_call = the LA regret (h) of the action the student would pick,
          i.e. how much the LA's own horizon cost says calling the LA saves at
          this stop.  A pick the LA scored infeasible in some scenario (dirty)
          gets DIRTY_H.  One-step, on the LA's states: it does not see effects
          that build up over several stops (the closed loop, step 3, does).
  signals what a deployed gate can compute from the students alone
          - margin       2nd-best minus best predicted cost
          - rest_margin  |best rest - best non-rest| predicted cost
          - entropy      of softmax(-cost / 0.25 h) over the candidates
          - pick_feas, min_feas_top2, n_cand, n_legal, pick_rest/break/charge
          - other_arm    the other arm (trees <-> PyTorch) picks differently
          - other_gap    the student's own cost gap to the other arm's pick
          - oracle_dis   the oracle-trained trees pick differently
          - oracle_gap   the student's cost gap to that pick
          + the state features of the decision (same for all its rows)

Candidates are the rows the student's feasibility head accepts (>= 0.5), as
policy_core.decide does; the shield's own filter is not re-run offline.

Honest picks on the training routes: fit seeds 1-19 are split into 5 folds
(queue_gate_folds.py); a route's pick comes from the fold model that never saw
it.  On seeds 20-21 (stop) and 22-25 (test) the picks come from the deployed
models (seed 0), which never saw them either.

    python ML/code/gate_data.py            # -> ML/data/gate_<arm>.npz, both arms
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DATA, MODELS, decision_id, load                  # noqa: E402

FOLDS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 19)]
DIRTY_H = 1.0
TAU = 0.25
ARMS = {"gbt": ("gbt_F95_base_f{k}_s0", "gbt_F95_base_s0"),
        "torch": ("tmlp_F95_split_list_f{k}_s0", "tmlp_F95_split_list_s0")}
ORACLE = ("gbt_F95_oracle_f{k}_s0", "gbt_F95_oracle_s0")
# --setting mixed (2026-10-23): the mixed-route student and its trees twin,
# on the LA's mixed training routes; no oracle there, so no oracle signal
MIXED_ARMS = {"torch": ("tmlp_T144_physmix_split_list_f{k}_s0", "tmlp_T144_physmix_split_list_s0"),
              "gbt": ("gbt_P102_physmix_f{k}_s0", "gbt_P102_physmix_s0")}


def fold_of(seed):
    for k, (a, b) in enumerate(FOLDS):
        if a <= seed <= b:
            return k
    return None                       # stop / test seeds: the deployed model


def _cols(d, tag):
    with open(os.path.join(MODELS, f"{tag}_meta.json")) as fh:
        feats = json.load(fh)["features"]
    names = [str(x) for x in d["feature_names"]]
    return [names.index(f) for f in feats]


def predict(d, tag, rows):
    """(cost, feas) of model `tag` on dataset rows `rows`."""
    X = d["X"][rows][:, _cols(d, tag)]
    if tag.startswith("gbt_"):
        import lightgbm as lgb
        c = lgb.Booster(model_file=os.path.join(MODELS, f"{tag}_cost.txt")).predict(X)
        f = lgb.Booster(model_file=os.path.join(MODELS, f"{tag}_feas.txt")).predict(X)
        return c, f
    import torch
    from torch_policy import TorchPolicy
    pol = TorchPolicy(tag)
    torch.set_num_threads(4)
    out_c, out_f = [], []
    for i in range(0, len(X), 65536):
        c, f = pol._predict(X[i:i + 65536])
        out_c.append(c)
        out_f.append(f)
    return np.concatenate(out_c), np.concatenate(out_f)


def predictions(d, pair):
    """Per-row (cost, feas): fold model on fit seeds, deployed model elsewhere."""
    fold_fmt, full = pair
    seed = d["seed"]
    folds = np.array([fold_of(int(s)) for s in seed], dtype=object)
    cost = np.full(len(seed), np.nan)
    feas = np.full(len(seed), np.nan)
    for k in range(len(FOLDS)):
        rows = np.flatnonzero(folds == k)
        cost[rows], feas[rows] = predict(d, fold_fmt.format(k=k), rows)
    rows = np.flatnonzero(np.array([f is None for f in folds]))
    cost[rows], feas[rows] = predict(d, full, rows)
    return cost, feas


def pick(cost, feas, ix):
    """policy_core.decide's argmin over a decision's rows `ix`."""
    s = cost[ix] + 1e6 * (feas[ix] < 0.5)
    return ix[int(np.argmin(s))], s


SIGNALS = ["margin", "rest_margin", "entropy", "pick_feas", "min_feas_top2", "n_cand",
           "n_legal", "pick_rest", "pick_break", "pick_charge", "other_arm", "other_gap",
           "oracle_dis", "oracle_gap"]


def build(d, arm, P):
    """Gate table for student `arm`; P[name] = (cost, feas) per row."""
    other = "torch" if arm == "gbt" else "gbt"
    cost, feas = P[arm]
    vocab = [str(x) for x in d["action_vocab"]]
    a_rest = np.array([v.endswith(("_r1", "_r2")) for v in vocab])[d["action_ix"]]
    a_brk = np.array(["_b" in v for v in vocab])[d["action_ix"]]
    a_chg = np.array([v.startswith("y1") for v in vocab])[d["action_ix"]]
    clean = d["clean"].astype(bool) & np.isfinite(d["regret"])
    did = decision_id(d)
    order = np.argsort(did, kind="stable")
    bounds = np.flatnonzero(np.r_[True, did[order][1:] != did[order][:-1], True])
    ns = int(d["n_state"])
    lab, sig, st, meta = [], [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ix = order[a:b]
        if clean[ix].sum() < 2:
            continue
        p, s = pick(cost, feas, ix)
        lab.append(float(d["regret"][p]) if clean[p] else DIRTY_H)
        cand = ix[feas[ix] >= 0.5]
        if len(cand) == 0:
            cand = ix
        c = np.sort(cost[cand])
        margin = c[1] - c[0] if len(c) > 1 else 9.0
        r, n = cost[cand][a_rest[cand]], cost[cand][~a_rest[cand]]
        rm = abs(r.min() - n.min()) if len(r) and len(n) else 9.0
        z = -(cost[cand] - c[0]) / TAU
        pz = np.exp(z) / np.exp(z).sum()
        ent = float(-(pz * np.log(pz + 1e-12)).sum())
        top2 = cand[np.argsort(cost[cand])[:2]]
        po, _ = pick(*P[other], ix)
        pr = pick(*P["oracle"], ix)[0] if "oracle" in P else p
        row = [margin, rm, ent, feas[p], feas[top2].min(), len(cand), len(ix),
               a_rest[p], a_brk[p], a_chg[p], float(po != p), cost[po] - cost[p]]
        if "oracle" in P:
            row += [float(pr != p), cost[pr] - cost[p]]
        sig.append(row)
        st.append(d["X"][p, :ns])
        meta.append((int(d["instance_ix"][p]), int(d["stop"][p]), int(d["seed"][p])))
    meta = np.array(meta)
    names = SIGNALS if "oracle" in P else SIGNALS[:-2]
    return dict(label=np.array(lab), signals=np.array(sig, dtype=np.float32),
                signal_names=np.array(names), state=np.array(st, dtype=np.float32),
                state_names=np.array([str(x) for x in d["feature_names"][:ns]]),
                instance_ix=meta[:, 0], stop=meta[:, 1], seed=meta[:, 2],
                instances=d["instances"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="base", choices=["base", "mixed"])
    args = ap.parse_args()
    if args.setting == "mixed":
        from dataset import load_multi
        d = load_multi(["pmix", "mix"])
        P = {arm: predictions(d, pair) for arm, pair in MIXED_ARMS.items()}
        builds, prefix = ["torch"], "gate_mixed_"
    else:
        d = load()
        P = {arm: predictions(d, pair) for arm, pair in ARMS.items()}
        P["oracle"] = predictions(d, ORACLE)
        builds, prefix = list(ARMS), "gate_"
    for arm in builds:
        g = build(d, arm, P)
        out = os.path.join(DATA, f"{prefix}{arm}.npz")
        np.savez_compressed(out, **g)
        sp = np.where(g["seed"] <= 19, "fit", np.where(g["seed"] <= 21, "stop", "test"))
        print(f"[gate] {arm}: {len(g['label'])} decisions -> {out}")
        for s in ("fit", "stop", "test"):
            m = sp == s
            print(f"   {s:5s} {m.sum():6d} decisions, mean value of a call "
                  f"{g['label'][m].mean()*60:.2f} min, nonzero {np.mean(g['label'][m] > 1e-9)*100:.1f}%")


if __name__ == "__main__":
    main()
