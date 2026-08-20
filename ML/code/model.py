"""
model.py — the student policy network (PyTorch)
===============================================
EASY MODE
---------
The network is a function with adjustable knobs (its "weights").  You put in
the 81 numbers that describe the current situation; it puts out
  (a) a score for each of the 12 possible discrete decisions, and
  (b) one number: "if we charge, charge for this many hours".
Training (see train.py) turns the knobs until, on the recorded teacher
decisions, the highest score lands on what the teacher chose and the charge
duration lands near what the teacher used.  At deployment the whole thing is
a few matrix multiplications: ~10^5 arithmetic ops, i.e. well under a
millisecond on any CPU — that is the entire speed claim of the paper.

WHY THIS SHAPE
--------------
* Two hidden layers of 128 units is deliberately SMALL.  Our input is 81
  engineered features and ~15k training rows — a big network would have
  enough capacity to memorise the training routes instead of learning the
  rule (overfitting).  Model size is an ablation, not an act of faith.
* One shared "trunk" feeds two "heads".  The discrete decision and the
  charge duration depend on the same situational understanding (battery,
  clocks, deadlines), so they share the trunk; only the last layer differs.
  This is standard multi-task design and also regularises both tasks.
* Feasibility MASKING: some decisions are structurally impossible in a given
  state (you cannot charge where there is no charger).  We set those scores
  to -inf BEFORE the softmax, so the network never wastes capacity learning
  "don't charge in the middle of nowhere" and can never pick it at
  deployment.  The mask encodes only hard structure; everything soft is
  learned.
"""
from __future__ import annotations
import torch
import torch.nn as nn

# Feature indices used by the mask — must match FEATURE_NAMES in
# extract_dataset.py (asserted in train.py at load time).
F_PHI, F_RHO2 = 5, 6
F_AT_CS, F_ALLOW_SPLIT = 8, 19
RHO2_BUDGET = 3.0        # reduced (9 h) daily rests allowed before a full one


def build_mask(X: torch.Tensor, classes: list) -> torch.Tensor:
    """(rows, n_classes) boolean: True = decision is structurally allowed.

    Rules (hard structure only — anything debatable stays learnable):
      * y=1 (charge)      requires standing at a charging station.
      * b15 (first half of a split break) requires no split in progress
        (phi == 0) and the instance allowing split breaks.
      * b30: NO phi requirement.  We first required phi==1 ("second half of
        a split") but 199 teacher decisions falsified that: a standalone
        30-min break is legal in the model (it resets the continuous-WORK
        clock, C11, regardless of the split state).  Lesson kept in code:
        masks encode the model's rules, not our reading of the regulation.
      * r2 (reduced 9 h rest) requires reduced-rest budget remaining.
    train.py verifies that every teacher action satisfies its own mask and
    warns if a rule ever contradicts the data (then the rule must go — the
    data is the authority, not our reading of the regulation).
    """
    n, C = X.shape[0], len(classes)
    ok = torch.ones(n, C, dtype=torch.bool)
    at_cs  = X[:, F_AT_CS] > 0.5
    phi    = X[:, F_PHI] > 0.5
    split  = X[:, F_ALLOW_SPLIT] > 0.5
    rho2ok = X[:, F_RHO2] < RHO2_BUDGET - 0.5
    for j, (y, brk, rst) in enumerate(classes):
        if int(y) == 1:      ok[:, j] &= at_cs
        if brk == "b15":     ok[:, j] &= (~phi) & split
        if rst == "r2":      ok[:, j] &= rho2ok
    return ok


class StudentPolicy(nn.Module):
    """Trunk MLP + (decision head, charge-duration head)."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 128):
        super().__init__()
        # nn.Sequential chains layers: input -> Linear -> ReLU -> ...
        # ReLU(x)=max(0,x) is the nonlinearity that lets stacked linear maps
        # express non-linear decision rules (without it, depth adds nothing).
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_cls  = nn.Linear(hidden, n_classes)  # raw scores ("logits")
        # Softplus keeps the predicted charge duration positive by
        # construction (smooth version of max(0, x)).
        self.head_tauc = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.trunk(x)
        logits = self.head_cls(h)
        if mask is not None:
            # -inf score => probability exactly 0 after softmax, gradient 0:
            # the invalid class is simply absent from the model's world.
            logits = logits.masked_fill(~mask, float("-inf"))
        tauc = self.head_tauc(h).squeeze(-1)          # hours, >= 0
        return logits, tauc

    @torch.no_grad()
    def act(self, x: torch.Tensor, mask: torch.Tensor):
        """Deployment: the single best valid decision (greedy argmax).
        No sampling — the policy is deterministic, like the teacher."""
        logits, tauc = self.forward(x, mask)
        return logits.argmax(dim=-1), tauc
