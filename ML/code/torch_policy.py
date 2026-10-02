"""
torch_policy.py — the PyTorch arm of the student policy
=======================================================
Counterpart to gbt_policy.py and nn_policy.py: supplies predictions to the
shared decision rule in policy_core.py from a checkpoint written by
torch_train.py,

    <tag>_torch.pt   {config, state_dict, mean, std}

and nothing else -- legality, forcing, the charge clamp, the spread-room check
and the simulator loop are policy_core's, shared verbatim with every arm.

Like the scikit-learn arm it standardises every row with the TRAINING mean and
std that travel inside the checkpoint.  Inference runs on one thread: the
evaluators drive routes in parallel processes, and one decision is a handful
of rows.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

from policy_core import MODELS, StudentPolicy                     # noqa: E402
from torch_models import build                                    # noqa: E402


class TorchPolicy(StudentPolicy):

    def __init__(self, tag, feas_thr=0.5, guard_q=None, models_dir=MODELS):
        super().__init__(feas_thr=feas_thr, guard_q=guard_q)
        with open(os.path.join(models_dir, f"{tag}_meta.json")) as fh:
            meta = json.load(fh)
        names = meta["features"]
        self.state_names = names[: meta["n_state"]]
        self.action_names = names[meta["n_state"]:]
        ck = torch.load(os.path.join(models_dir, f"{tag}_torch.pt"),
                        map_location="cpu", weights_only=True)
        self.net = build(ck["config"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.mean = ck["mean"].numpy()
        self.std = ck["std"].numpy()
        torch.set_num_threads(1)
        self.tag = tag
        self.kind = "torch"

    def _forward(self, rows):
        x = torch.from_numpy(((rows - self.mean) / self.std).astype(np.float32))
        with torch.no_grad():
            return self.net(x)

    def _predict(self, rows):
        c, f, _t = self._forward(rows)
        return c.numpy().astype(np.float64), torch.sigmoid(f).numpy()

    def _predict_tauc(self, row):
        return float(self._forward(row)[2][0])
