"""
gbt_policy.py — the GRADIENT-BOOSTED-TREE arm of the student policy
===================================================================
Supplies predictions to the shared decision rule in policy_core.py from three
LightGBM boosters trained by gbt_train.py:

    <tag>_cost.txt   regret in hours for a (state, action) pair
    <tag>_feas.txt   P(action feasible in all 25 scenarios)
    <tag>_tauc.txt   charge duration in hours

Nothing about the decision rule, the legality filter, the forcing rules, the
charge clamp or the simulator loop lives here -- all of that is in
policy_core.py and is shared verbatim with the neural arm (nn_policy.py), so
the two can be compared without wondering whether some other difference crept
in.

No feature scaling: trees are invariant to any monotone transform of a
feature, which is one of the reasons this arm was built first.  The neural arm
has to carry a fitted scaler alongside its weights.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lightgbm as lgb                                            # noqa: E402

from policy_core import MODELS, StudentPolicy                     # noqa: E402


class GBTPolicy(StudentPolicy):

    def __init__(self, tag="base", feas_thr=0.5, guard_q=None,
                 models_dir=MODELS):
        super().__init__(feas_thr=feas_thr, guard_q=guard_q)
        with open(os.path.join(models_dir, f"{tag}_meta.json")) as fh:
            meta = json.load(fh)
        names = meta["features"]
        self.state_names = names[: meta["n_state"]]
        self.action_names = names[meta["n_state"]:]
        self.cost = lgb.Booster(model_file=os.path.join(models_dir,
                                                        f"{tag}_cost.txt"))
        self.feas = lgb.Booster(model_file=os.path.join(models_dir,
                                                        f"{tag}_feas.txt"))
        self.tauc = lgb.Booster(model_file=os.path.join(models_dir,
                                                        f"{tag}_tauc.txt"))
        self.tag = tag
        self.kind = "gbt"

    def _predict(self, rows):
        return self.cost.predict(rows), self.feas.predict(rows)

    def _predict_tauc(self, row):
        return self.tauc.predict(row)[0]
