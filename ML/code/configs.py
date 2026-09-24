"""
configs.py — THE registry: every model, named the same way, in one place
=======================================================================
Naming got out of hand (`base`, `gbt_v1`, `noweight`, `nn`, `nn_log1p`, ...)
and none of it said which ARM a tag belonged to.  Every model is now named

    <arm>_<config>

with the arm always first, so a tag is self-describing wherever it appears --
in `ML/models/`, in `ML/results/`, in a figure legend or in a table row.

    gbt_*      boosted trees, cost-scoring       (LightGBM)
    mlp_*      neural net, cost-scoring          (sklearn MLPRegressor)
    clf_*      neural net, action classification (sklearn MLPClassifier)
    legacy_*   the restored 2026-08 model, run exactly as it was built

PROTOCOL.  Every configuration is an INDEPENDENT model: fitted on seeds 1-19,
early-stopped on seeds 20-21, and reported on the whole test batch (seeds
22-25, 125 routes).  Nothing is selected on the test set -- the table reports
every row -- so the rows are directly comparable to each other and to the
baselines already stored in solutions/basecase.

DAgger is not part of any configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    arm: str                     # gbt | mlp | clf | legacy
    name: str                    # short config name, unique within the arm
    desc: str                    # one line, appears in tables and figures
    train: tuple = ()            # flags passed to <arm>_train.py
    guard: float | None = 0.95   # deployment guard quantile; None = nominal
    label: str = ""              # display label; defaults to a readable form

    @property
    def tag(self) -> str:
        return f"{self.arm}_{self.name}"

    @property
    def eval_name(self) -> str:
        g = "nom" if self.guard is None else f"g{int(self.guard*100)}"
        return f"eval_{self.tag}_{g}_test.json"

    @property
    def display(self) -> str:
        return self.label or f"{ARM_LABEL[self.arm]} — {self.desc}"


ARM_LABEL = {"gbt": "Trees", "mlp": "MLP (regression)",
             "clf": "MLP (classifier)", "legacy": "MLP (2026-08, as built)"}

# ── the registry ────────────────────────────────────────────────────────────
# Each row is one model.  `guard` is a DEPLOYMENT setting, so two rows may
# share a trained model and differ only there; the runner trains once.
CONFIGS = [
    # -- boosted trees, cost-scoring -----------------------------------------
    Config("gbt", "base", "regret target, margin weights, depth 6",
           train=(), guard=0.95, label="Trees"),
    Config("gbt", "base", "same model, nominal guard",
           train=(), guard=None, label="Trees (nominal guard)"),
    Config("gbt", "rawcost", "raw horizon cost instead of centred regret",
           train=("--target", "cost"), guard=0.95,
           label="Trees — raw cost target"),
    Config("gbt", "noweight", "no margin weighting",
           train=("--weight-power", "0"), guard=0.95,
           label="Trees — no weighting"),
    Config("gbt", "shallow", "depth 3 / 15 leaves",
           train=("--leaves", "15", "--depth", "3"), guard=0.95,
           label="Trees — shallow"),

    # -- neural net, cost-scoring --------------------------------------------
    Config("mlp", "base", "log1p regret — tail-robust, monotone",
           train=("--target-transform", "log1p"), guard=0.95, label="MLP"),
    Config("mlp", "base", "same model, nominal guard",
           train=("--target-transform", "log1p"), guard=None,
           label="MLP (nominal guard)"),
    Config("mlp", "sqerr", "squared error on raw regret (unprotected tail)",
           train=("--target-transform", "none"), guard=0.95,
           label="MLP — squared error"),
    Config("mlp", "rawcost", "raw horizon cost instead of centred regret",
           train=("--target", "cost", "--target-transform", "log1p"),
           guard=0.95, label="MLP — raw cost target"),

    # -- neural net, classification (the 2026-08 framing, controlled) --------
    Config("clf", "base", "12-way action classifier, unweighted",
           train=("--class-weight", "none"), guard=0.95, label="Classifier"),
    Config("clf", "sqrtw", "classifier with sqrt class weights",
           train=("--class-weight", "sqrt"), guard=0.95,
           label="Classifier — sqrt weights"),
]

# The restored 2026-08 model is run by its own scripts, not by <arm>_train.py,
# so it is listed separately and joined into the table from its own results.
LEGACY = Config("legacy", "mlp",
                "restored verbatim: 141 features, 772 runs, its own rollout",
                guard=0.95, label="MLP (2026-08, as built)")


def by_tag():
    """tag -> the training flags for it (guard-only duplicates collapse)."""
    out = {}
    for c in CONFIGS:
        out.setdefault(c.tag, c.train)
    return out


def arms():
    return ["gbt", "mlp", "clf", "legacy"]
