"""
clf_policy.py — the CLASSIFIER arm of the student policy
========================================================
Third arm alongside `gbt_policy.py` (boosted trees) and `nn_policy.py` (MLP
regression).  It supplies predictions to the same decision rule in
`policy_core.py`, so legality, forcing, the argmin and the charge clamp are
byte-identical across all three and the comparison measures the framing.

Where the scoring arms predict a cost per (state, action) row, this arm scores
the STATE once and reads off a probability per action.  To keep
`policy_core.decide` unchanged it returns

    cost[a] = -log P(a | state)

so the argmin picks the most probable LEGAL action -- masking by legality is
already done by the caller, which is exactly the "masked argmax" the deleted
2026-08 project used, expressed in the shared rule's own terms.

Feasibility is returned as 1.0 for every action: this arm has no feasibility
head, because a classifier trained on the teacher's CHOICES never sees the
actions the teacher rejected as infeasible.  That is a real limitation of the
framing, not an oversight -- the scoring arms get that head for free from the
per-action `ok=k/25` the logs record.  The supervisor's forcing rules still
apply, as for every arm.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib                                                     # noqa: E402
import numpy as np                                                # noqa: E402

from policy_core import MODELS, StudentPolicy                     # noqa: E402


class ClfPolicy(StudentPolicy):

    def __init__(self, tag="clf", feas_thr=0.5, guard_q=None,
                 models_dir=MODELS):
        super().__init__(feas_thr=feas_thr, guard_q=guard_q)
        with open(os.path.join(models_dir, f"{tag}_meta.json")) as fh:
            meta = json.load(fh)
        names = meta["features"]
        self.n_state = int(meta["n_state"])
        self.state_names = names[: self.n_state]
        self.action_names = names[self.n_state:]
        ck = joblib.load(os.path.join(models_dir, f"{tag}_clf.joblib"))
        self.scaler = ck["scaler"]
        self.clf = ck["clf"]
        self.tauc = ck["tauc"]
        self.vocab = list(ck["vocab"])
        self.ix = {k: i for i, k in enumerate(self.vocab)}
        # MLPClassifier only knows the classes it saw in training
        self.seen = {int(c): j for j, c in enumerate(self.clf.classes_)}
        self.tag = tag
        self.kind = "clf"

    def _state_row(self, rows):
        """Every row of a decision shares the same state block."""
        return self.scaler.transform(rows[:1, : self.n_state])

    def _predict(self, rows):
        p = self.clf.predict_proba(self._state_row(rows))[0]
        out = np.empty(len(rows), dtype=float)
        for i, key in enumerate(self._legal_keys):
            c = self.ix.get(key, -1)
            j = self.seen.get(c, None)
            # an action the classifier never saw in training gets probability
            # ~0 rather than being silently ranked first
            out[i] = -np.log(max(p[j], 1e-12)) if j is not None else 27.6
        return out, np.ones(len(rows))

    def _predict_tauc(self, row):
        return self.tauc.predict(self.scaler.transform(row[:, : self.n_state]))[0]
