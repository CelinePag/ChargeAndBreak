"""
ml_style.py — the manuscript's figure style, extended with the ML methods
=========================================================================
`src/plot/paper_style.py` is the single source of truth for every figure in
the paper: colour follows the ENTITY, so Greedy is the same blue in the
base-case box plot, the sensitivity bars and the diesel comparison. The ML
figures must obey the same rule or a reader who has learned that legend will
misread them.

`src/` is read-only for this project, so the two learned policies are added
here rather than there, following the module's own precedents:

* **Okabe-Ito only.** The one free slot left by `METHOD_COLOR` is the sky blue
  `#56B4E9` (the yellow `#F0E442` is documented there as too low-contrast for
  a thin box outline). The learned students take it.
* **Same approach, different approximator -> same hue, told apart by hatch.**
  This is exactly what `paper_style` does for `DET` / `DETg`, which are one
  plan under two execution rules. The GBT and MLP students are one
  formulation under two function approximators, so they share the hue and the
  MLP is hatched.

`#56B4E9` is also DET/DETg's hue. They never co-occur: DET is a VSS
diagnostic held out of the paper figures (`METHOD_ORDER_EXTRA`), and these
figures are the ML arm. If the two are ever plotted together, give the
students their own slot here rather than reusing one.

Feasibility keeps the manuscript's own encoding: the green -> yellow ->
vermillion ramp `additional_figures._INFEAS_CMAP` uses to shade a cell by its
infeasibility rate. Green = every route completed; vermillion = many halted.
"""
from __future__ import annotations

import os
import sys

from matplotlib.colors import LinearSegmentedColormap

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.plot.paper_style import (BASELINE, GRID, INK_MUTED,      # noqa: E402
                                  INK_PRIMARY, METHOD_COLOR, METHOD_HATCH,
                                  METHOD_LBL, apply_rc, shade, tint)

# ── the learned policies ────────────────────────────────────────────────────
# All four are the SAME family of thing -- a policy learned from the
# look-ahead -- so they share the one free Okabe-Ito slot and are told apart
# by hatch, exactly as paper_style does for DET / DETg.  Colour therefore
# reads as "learned" at a glance, and identity is on the y axis as always.
ML_GBT, ML_MLP, ML_CLF, ML_LEGACY = "gbt", "mlp", "clf", "legacy"
_SKY = "#56B4E9"

COLOR = dict(METHOD_COLOR)
HATCH = dict(METHOD_HATCH)
LBL = dict(METHOD_LBL)
for _k, _h, _l in ((ML_GBT, "", "Trees"),
                   (ML_MLP, "///", "MLP (regression)"),
                   (ML_CLF, "...", "MLP (classifier)"),
                   (ML_LEGACY, "xxx", "MLP (2026-08, as built)")):
    COLOR[_k] = _SKY
    if _h:
        HATCH[_k] = _h
    LBL[_k] = _l

# Maps the run-file method names used in solutions/ onto style keys.
FROM_RUN = {"LA_MIPTAIL": "LA", "GREEDY": "greedy", "2SP": "2SP",
            "RO": "RO", "ROBU": "ROBU", "oracle": "oracle"}

# ── feasibility: the manuscript's own ramp ──────────────────────────────────
# additional_figures._INFEAS_CMAP, reproduced rather than imported because
# importing that module pulls in the whole reporting stack.
INFEAS_CMAP = LinearSegmentedColormap.from_list(
    "infeas", ["#009E73", "#F0E442", "#D55E00"])


def infeas_color(rate: float, fmax: float = 0.20):
    """Colour for an infeasibility RATE in [0,1]; fmax anchors the ramp's top."""
    return INFEAS_CMAP(min(1.0, max(0.0, rate) / max(fmax, 1e-9)))


def style_axes(ax, xgrid=True):
    """The paper's chrome: light grid, muted spines, no top/right box."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
        ax.spines[s].set_linewidth(0.6)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, lw=0.6)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


__all__ = ["COLOR", "HATCH", "LBL", "FROM_RUN", "ML_GBT", "ML_MLP",
           "INFEAS_CMAP", "infeas_color", "style_axes", "apply_rc",
           "tint", "shade", "INK_PRIMARY", "INK_MUTED", "GRID", "BASELINE"]
