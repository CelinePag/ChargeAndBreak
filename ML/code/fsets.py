"""
fsets.py — the feature-set registry: which inputs each model gets
=================================================================
Different model families have different needs, so they are not forced onto
one feature set.  The dataset holds a SUPERSET of columns; a feature set is a
named selection from it, applied identically at training and at serving
(policies look features up by NAME, so a model only ever sees its own set).

    C  compact      the top features by importance, chosen from a model fitted
                    on the FITTING routes only (seeds 1-19) -- never the
                    stopping or test routes.  The list is frozen to disk with
                    its provenance, so every C-model uses the same columns.
    D  dedup        the full set minus the columns that carry no information
                    of their own in the base case: 1 constant (`is_ferry`)
                    and 17 exact linear duplicates.  Label-free.
    F  full         every engineered feature (summarised lookahead, slacks,
                    horizon totals, action and window interactions).
    L  lookahead    F plus the next 20 stops listed raw, 6 numbers each.
    R  route-local  F minus the seven features that measure position along
                    the WHOLE route (time elapsed, stops/drive/energy/chargers/
                    customers left, fraction done).  On long routes ~23% of
                    decisions put those features beyond anything seen on short
                    and medium routes, where a tree cannot extrapolate; R keeps
                    only what is local to the decision, so it should transfer
                    across route lengths if those features are the problem.

A model's NAME carries its set and the number of inputs it consumes:

    <arm>_<SET><n>_<config>[_s<seed>]      e.g.  gbt_L215_base_s2

`n` is the count the model actually reads.  The classifier reads only STATE
features (it predicts the action rather than scoring it), so it gets its own
number: the full set is `gbt_F95` but `clf_F77`.  Two models whose names share
a set letter but differ in `n` genuinely saw different inputs.

History worth knowing: the models produced by `run_all.py` were trained before
the four window-interaction features existed, on 91 columns.  They are named
`F91` so they can never again be mistaken for the 95-column `F95` models --
which is exactly the confusion this naming exists to prevent.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
C_FILE = os.path.join(DATA, "fset_C40.json")

RAW_PREFIX = "nx"                       # the raw lookahead block

# the four features added for the time-window experiment
WINDOW = ("a_cu1_slack_after", "a_cu1_late", "a_cu1_early",
          "a_dwell_vs_cu1_drive")

# D: no information of their own in the base case, measured on the fitting
# routes.  1 constant + 17 columns that are exact linear transforms of an
# earlier column (|corr| > 0.9999) -- e.g. `cd_slack = 4.5 - cd`, and energy
# proportional to drive time because speed is constant in the base case.
D_DROP = (
    "is_ferry",
    "soc_frac", "usable_kwh", "cd_slack", "sw_slack", "spread_slack1",
    "spread_slack2", "rho2_left", "ext_left", "E_next", "D_next_wc",
    "drive_to_next_cs", "energy_to_next_cs", "soc_at_next_cs_frac",
    "e_nom_margin_next_cs", "service_here", "energy_left",
    "a_cu1_slack_after",
)

# P: charger power, per charger (features.py section I).  Constant along every
# training route (one charger type per route), so they are kept OUT of every
# older set -- F, D, R, L, C keep their meaning and their counts -- and read
# only by P = F + POWER, the set for routes whose chargers differ.
POWER = ("kw_here", "cs1_kw", "cs2_kw", "cs3_kw", "kw_next_ratio",
         "best_kw_reach", "drive_to_best_kw")

# R: features that grow with route length rather than describe the decision
ROUTE_POS = ("t_elapsed", "stops_left", "drive_left", "energy_left",
             "cs_left", "cust_left", "route_frac_done")

# the single source of truth for valid set ids: the trainers' --fset choices
# are read from here, so adding a set here is enough to make it trainable
FSET_IDS = ("C", "D", "F", "L", "R", "F91", "P")

SET_DOC = {"R": "route-local (no whole-route position)",
           "C": "compact (top by importance)", "D": "deduplicated",
           "F": "full engineered", "L": "full + raw lookahead",
           "F91": "full, before the window features",
           "P": "full + per-charger power"}


def _is_raw(n):
    return n.startswith(RAW_PREFIX) and n[len(RAW_PREFIX):len(RAW_PREFIX)+1].isdigit()


def _compact_list():
    if not os.path.exists(C_FILE):
        raise SystemExit(f"{C_FILE} missing — run `python ML/code/fsets.py "
                         f"--freeze-compact` first")
    with open(C_FILE) as fh:
        return json.load(fh)["features"]


def resolve(set_id, all_names, all_n_state, arm):
    """-> (selected names in dataset order, number of them that are STATE).

    Order is preserved -- state block first, then action block -- because the
    policies split a row into state and action parts at `n_state`.
    """
    state = list(all_names[:all_n_state])
    action = list(all_names[all_n_state:])

    if set_id != "P":
        # every older set is defined without the power block, so adding it
        # to the dataset leaves each of them exactly as it was
        all_names = [n for n in all_names if n not in POWER]

    if set_id == "P":
        keep = {n for n in all_names if not _is_raw(n)}
    elif set_id == "L":
        keep = set(all_names)
    elif set_id == "F":
        keep = {n for n in all_names if not _is_raw(n)}
    elif set_id == "F91":
        keep = {n for n in all_names if not _is_raw(n) and n not in WINDOW}
    elif set_id == "D":
        keep = {n for n in all_names if not _is_raw(n) and n not in D_DROP}
    elif set_id == "C":
        keep = set(_compact_list())
    elif set_id == "R":
        keep = {n for n in all_names if not _is_raw(n) and n not in ROUTE_POS}
    else:
        raise ValueError(f"unknown feature set {set_id!r}")

    s = [n for n in state if n in keep]
    a = [n for n in action if n in keep]
    if arm == "clf":                     # reads the state only, by design
        a = []
    return s + a, len(s)


def label(set_id, arm, all_names, all_n_state):
    """The name fragment, e.g. 'F95', 'D77', 'clf -> F77'."""
    names, _ = resolve(set_id, all_names, all_n_state, arm)
    return f"{set_id if set_id != 'F91' else 'F'}{len(names)}"


def freeze_compact(n=40, ref_model="gbt_F95_base_s0"):
    """Pick the top-n features by gain from a model fitted on seeds 1-19."""
    import lightgbm as lgb
    models = os.path.abspath(os.path.join(HERE, "..", "models"))
    m = lgb.Booster(model_file=os.path.join(models, f"{ref_model}_cost.txt"))
    names = m.feature_name()
    gain = m.feature_importance("gain")
    top = [names[i] for i in sorted(range(len(names)), key=lambda i: -gain[i])[:n]]
    with open(C_FILE, "w") as fh:
        json.dump(dict(features=top, n=n, source_model=ref_model,
                       criterion="LightGBM split gain, cost head",
                       fitted_on="route seeds 1-19 only (never stop or test)"),
                  fh, indent=1)
    print(f"froze {n} compact features from {ref_model} -> {C_FILE}")
    return top


if __name__ == "__main__":
    import argparse
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-compact", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    if args.freeze_compact:
        freeze_compact(args.n)
    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    ns = int(d["n_state"])
    print(f"\nsuperset: {len(names)} columns ({ns} state)\n")
    print(f"{'set':4s} {'meaning':34s} {'gbt/mlp':>8s} {'clf':>6s}")
    for sid in ("C", "D", "F", "L", "R", "F91"):
        try:
            g = label(sid, "gbt", names, ns)
            c = label(sid, "clf", names, ns)
            print(f"{sid:4s} {SET_DOC[sid]:34s} {g:>8s} {c:>6s}")
        except SystemExit as e:
            print(f"{sid:4s} {SET_DOC[sid]:34s}  ({e})")
