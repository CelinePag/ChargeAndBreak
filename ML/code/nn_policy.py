"""
nn_policy.py — the NEURAL arm of the student policy
===================================================
Counterpart to gbt_policy.py.  Supplies predictions to the shared decision
rule in policy_core.py from the three sklearn MLPs trained by nn_train.py,
stored together in one checkpoint:

    <tag>_nn.joblib   {scaler, cost, feas, tauc}

The decision rule, the legality filter, the forcing rules, the charge clamp
and the simulator loop are NOT here -- they live in policy_core.py and are
shared verbatim with the tree arm, so a GBT-vs-NN comparison measures the
regressor and nothing else.

The one thing this arm does that the tree arm does not
-----------------------------------------------------
It applies the StandardScaler that was fitted on the training rows, on every
feature vector, at every decision.  A tree is invariant to monotone feature
transforms; a network is not, and serving unscaled rows to a network trained
on scaled ones is the classic silent failure.  The scaler therefore travels
inside the same checkpoint as the weights, and is never re-fitted here.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib                                                     # noqa: E402
import numpy as np                                                # noqa: E402

from policy_core import MODELS, StudentPolicy                     # noqa: E402


class NNPolicy(StudentPolicy):

    def __init__(self, tag="nn", feas_thr=0.5, guard_q=None,
                 models_dir=MODELS):
        super().__init__(feas_thr=feas_thr, guard_q=guard_q)
        with open(os.path.join(models_dir, f"{tag}_meta.json")) as fh:
            meta = json.load(fh)
        names = meta["features"]
        self.state_names = names[: meta["n_state"]]
        self.action_names = names[meta["n_state"]:]
        ck = joblib.load(os.path.join(models_dir, f"{tag}_nn.joblib"))
        self.scaler = ck["scaler"]          # fitted on TRAIN rows, never re-fit
        self.cost = ck["cost"]
        self.feas = ck["feas"]
        self.tauc = ck["tauc"]
        # the cost head may be trained on log1p(regret); invert at serving so
        # the score stays in hours and the feasibility veto keeps its scale
        self.inv = meta.get("args", {}).get("target_transform", "none")
        self.tag = tag
        self.kind = "nn"

    def _predict(self, rows):
        z = self.scaler.transform(rows)
        c = self.cost.predict(z)
        if self.inv == "log1p":
            c = np.expm1(c)
        return c, self.feas.predict_proba(z)[:, 1]

    def _predict_tauc(self, row):
        return self.tauc.predict(self.scaler.transform(row))[0]
