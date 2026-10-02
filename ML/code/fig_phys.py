"""
fig_phys.py — trees trained across physics: how far behind the teacher, per value
================================================================================
One group of rows per physics value (route seeds 22-25 in every group, so the
physics is the only change).  Each row is one policy's gap to the oracle MINUS
the LA's, paired route by route on the routes both completed: the marker is
the mean, the bar its 95 % interval (1.96 standard errors).  Zero is the
teacher; left of it is better than the teacher.

    Trees, base case only        fitted on 500 kWh / 350 kW / 60 km routes
    Trees, all physics           fitted on every physics value (run_phys.py)
    Trees, all except this one   that value held out of fitting AND early
                                 stopping (leave-one-value-out)
    Greedy                       the stored runs

The group header says where the held-out value sits relative to the values
the model was fitted on.  cs30 is left out: the teacher finished only 6 of
its test routes.

    python ML/code/fig_phys.py            -> ML/figures/fig_phys.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_style import (COLOR, INK_MUTED, INK_PRIMARY, apply_rc,   # noqa: E402
                      shade, style_axes)
from phys_eval import BASE_TAG, read_store, store_path, summarise  # noqa: E402
from run_phys import tag_of                                       # noqa: E402

FIGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))

GROUPS = [("base", "Base case: 500 kWh, 350 kW, a charger every ~60 km", None),
          ("kwh300", "Battery 300 kWh", "below the fitted range (500-900)"),
          ("kwh700", "Battery 700 kWh", "between fitted values (500, 900)"),
          ("kwh900", "Battery 900 kWh", "above the fitted range (300-700)"),
          ("kw150", "Charger 150 kW", "below the fitted range (350-1000)"),
          ("kw700", "Charger 700 kW", "between fitted values (350, 1000)"),
          ("kw1000", "Charger 1000 kW", "above the fitted range (150-700)"),
          ("cs100", "A charger every ~100 km", "above the fitted range (30-60 km)")]

# (key in the summary, row label, marker, filled?, style colour key)
ROWS = [(BASE_TAG, "Trees, base case only", "o", False, "gbt"),
        (tag_of("phys"), "Trees, all physics", "o", True, "gbt"),
        ("LO", "Trees, all except this value", "s", True, "gbt"),
        ("Greedy", "Greedy", "v", True, "greedy")]
XMIN, XMAX = -3.0, 12.0


def main():
    res = summarise(read_store(store_path(0.99, True)))
    rows, headers, bands = [], [], []
    y = 0.0
    for gi, (axis, label, regime) in enumerate(GROUPS):
        if axis not in res:
            continue
        headers.append((y, label + (f"  —  held-out value {regime}" if regime else "")))
        y += 0.95
        y0 = y
        for key, lab, mk, filled, ck in ROWS:
            k = tag_of(f"phys_LO{axis}") if key == "LO" else key
            p = res[axis].get(k, {}).get("vs_LA")
            if p is None:
                continue
            rows.append((y, lab, mk, filled, ck, p))
            y += 1.0
        bands.append((y0 - 0.5, y - 0.5, gi))
        y += 0.45

    apply_rc()
    fig, ax = plt.subplots(figsize=(7.2, 0.21 * len(rows) + 0.25 * len(headers) + 1.3),
                           dpi=200)
    style_axes(ax)
    for ylo, yhi, gi in bands:
        if gi % 2 == 0:
            ax.axhspan(ylo, yhi, color="#f4f4f4", zorder=0, lw=0)
    for yh, text in headers:
        ax.text(0.0, yh, text, transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=7.2, color=INK_PRIMARY, fontweight="bold",
                zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

    for yy, lab, mk, filled, ck, p in rows:
        col = COLOR[ck]
        lo, hi = p["mean"] - 1.96 * p["se"], p["mean"] + 1.96 * p["se"]
        ax.plot([max(lo, XMIN), min(hi, XMAX)], [yy, yy], color=shade(col, 0.1),
                lw=2.0, solid_capstyle="round", zorder=2)
        m = min(max(p["mean"], XMIN), XMAX)
        ax.plot([m], [yy], marker=mk, ms=5.0, color=shade(col, 0.35),
                mfc=col if filled else "white", mew=1.2, zorder=3, clip_on=False)
        txt = f"{p['mean']:+.1f}"
        if hi > XMAX:            # the bar runs off the axis: label inside, at the edge
            ax.text(XMAX - 0.12, yy, txt + " →", ha="right", va="center",
                    fontsize=5.8, color=INK_PRIMARY, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
        else:
            ax.text(hi + 0.15, yy, txt, va="center", fontsize=5.8, color=INK_MUTED)
        ax.text(1.005, yy, f"n={p['n']}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=5.4, color=INK_MUTED)

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=6.2, color=INK_MUTED)
    ax.tick_params(axis="y", pad=2, length=0)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_PRIMARY, lw=0.9, zorder=1)
    ax.text(0.08, -0.35, "the teacher (LA)", fontsize=6.0, color=INK_PRIMARY,
            va="bottom", transform=ax.transData)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("gap to the hindsight oracle minus the LA's, same routes "
                  "(percentage points; lower is better)\n"
                  "marker = mean, bar = 95 % interval")
    ax.set_title("Trees fitted on one physics, on all, or on all but the "
                 "tested value", fontsize=9.0, color=INK_PRIMARY, loc="left", pad=6)
    ax.annotate("Route seeds 22-25 in every group; paired with the LA on the "
                "routes both completed. Every tree model completed every route "
                "(0 rule violations). Spread-room check, 0.99 drive guard.",
                xy=(0, 0), xycoords="axes fraction", xytext=(-150, -40),
                textcoords="offset points", va="top", fontsize=6.0,
                color=INK_MUTED, annotation_clip=False)
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "fig_phys.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
