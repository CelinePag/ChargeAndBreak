"""
paper_style.py — the single source of truth for every figure in the paper.

Colour must follow the ENTITY, never the number of series in a given chart:
Greedy is the same blue in the base-case box plot, the sensitivity bars and
the diesel comparison, so a reader who learns the legend once can read every
figure.  Before this module the constants were duplicated in paper_figures.py
and additional_figures.py and had already started to drift.

Two encodings, deliberately kept apart:

  METHOD_COLOR  categorical — Okabe-Ito, colourblind-safe, fixed assignment.
                ORACLE is neutral dark grey on purpose: it is a benchmark
                bound, not a competing policy, so it must not read as a
                sixth method.
  ROUTE_COLOR   ordered — route length is ordinal (short < medium < long),
                so it takes steps of ONE hue rather than categorical hues.
                Never use these for methods, or the two encodings collide.

Facet grammar used throughout: route class facets (panels/rows), customer
count facets (columns), time-window class is the within-group shade, method
is the colour.
"""

from __future__ import annotations

# ── categorical: method identity (Okabe-Ito) ─────────────────────────────────
METHOD_ORDER = ["greedy", "RO", "ROBU", "LA", "2SP", "oracle"]
METHOD_LBL = {
    "greedy": "Greedy", "RO": "RO", "ROBU": "ROBU",
    "LA": "LA", "2SP": "2SP", "oracle": "Oracle",
}
METHOD_COLOR = {
    "greedy": "#0072B2",   # blue
    "RO":     "#D55E00",   # vermillion
    "ROBU":   "#E69F00",   # orange   (budgeted robust, Bertsimas-Sim)
    "LA":     "#009E73",   # bluish green
    "2SP":    "#CC79A7",   # reddish purple
    "oracle": "#3A3A3A",   # neutral dark grey — a bound, not a policy
}

# ── ordered: route class (one hue, light -> dark with length) ────────────────
ROUTE_ORDER = ["short", "medium", "long"]
ROUTE_LBL = {"short": "Short route", "medium": "Medium route",
             "long": "Long route"}
ROUTE_COLOR = {"short": "#9ecae1", "medium": "#4292c6", "long": "#08519c"}

# ── remaining canonical axis orders ─────────────────────────────────────────
CUST_ORDER = ["few", "medium", "many"]
CUST_LBL = {"few": "Few customers", "medium": "Medium customers",
            "many": "Many customers"}
TW_ORDER = ["tight", "medium", "large", "none"]
TW_LBL = {"none": "None", "tight": "Tight", "medium": "Medium",
          "large": "Large"}

# ── chrome: neutral journal-figure greys ────────────────────────────────────
INK_PRIMARY = "#000000"
INK_MUTED   = "#555555"
GRID        = "#e0e0e0"
BASELINE    = "#333333"

RC = {
    "font.size": 8,
    "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.6,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
}


def apply_rc() -> None:
    """Install the shared rcParams (call once per figure script)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)


def tint(color: str, frac: float = 0.80) -> tuple:
    """Blend toward white (frac = white share) — box fills, secondary bars."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def shade(color: str, frac: float = 0.35) -> tuple:
    """Blend toward black (frac = black share) — median lines, outlines."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r * (1 - frac), g * (1 - frac), b * (1 - frac))
