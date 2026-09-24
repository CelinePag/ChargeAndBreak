"""
figures.py — result figures for ONE student (shared by both arms)
=================================================================
Three figures, each answering one question, for whichever arm produced the
evaluation JSON being read:

  fig1_distributions  How does the student compare to LA / GREEDY / ORACLE,
                      distribution and tail -- not just a median.
  fig2_ablations      Which design choices actually mattered (tree arm only:
                      the ablation grid is gbt-specific).
  fig3_families       Is the failure concentrated in a few regimes or diffuse?

  fig4_arms           NOT here -- the trees-vs-network head-to-head lives in
                      compare_arms.py, which imports the palette and styling
                      from this module so both look identical.

Colour: the Okabe-Ito qualitative palette, assigned in fixed order and never
cycled.  The skill's node validator is not available here, so the two checks
that matter are computed below in `validate_palette()` and printed before
anything is drawn: a deuteranope/protanope simulation followed by pairwise
OKLab dE, plus the normal-vision floor.  Eyeballing a palette is not a check.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))

# Okabe & Ito (2008), fixed order.  Never cycled: a 9th series would fold into
# "other" rather than get a generated hue.
#
# The three-series order is blue / vermillion / sky, chosen by SEARCHING the
# palette with validate_palette() rather than by eye.  The obvious
# blue/vermillion/green triple FAILS tritanopia (dE 5.4 against a floor of 8);
# this one passes every vision type (min dE 17.9, normal 20.4).
OKABE_ITO = ["#0072B2", "#D55E00", "#56B4E9", "#CC79A7",
             "#E69F00", "#009E73", "#F0E442", "#000000"]
# Diverging pair for signed quantities: cool = faster, warm = slower.
DIV_GOOD, DIV_BAD = "#0072B2", "#D55E00"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
SURFACE = "#ffffff"


# ── palette validation (computed, not asserted) ──────────────────────────────

def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _to_oklab(rgb_lin):
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = M1 @ rgb_lin
    lms = np.cbrt(np.clip(lms, 0, None))
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return M2 @ lms


def _cvd(rgb_lin, kind):
    """Brettel/Vienot-style dichromat simulation in linear RGB."""
    if kind == "deuteranopia":
        M = np.array([[0.625, 0.375, 0.0],
                      [0.700, 0.300, 0.0],
                      [0.0, 0.300, 0.700]])
    elif kind == "protanopia":
        M = np.array([[0.567, 0.433, 0.0],
                      [0.558, 0.442, 0.0],
                      [0.0, 0.242, 0.758]])
    else:                                   # tritanopia
        M = np.array([[0.950, 0.050, 0.0],
                      [0.0, 0.433, 0.567],
                      [0.0, 0.475, 0.525]])
    return M @ rgb_lin


def validate_palette(hexes, labels=None):
    """Pairwise OKLab dE x100, for normal vision and each dichromacy."""
    labels = labels or [f"c{i}" for i in range(len(hexes))]
    lin = [_srgb_to_linear(_hex_to_rgb(h)) for h in hexes]
    report, worst = [], {}
    for kind in ("normal", "deuteranopia", "protanopia", "tritanopia"):
        lab = [_to_oklab(v if kind == "normal" else _cvd(v, kind)) for v in lin]
        de = []
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                d = 100.0 * float(np.linalg.norm(lab[i] - lab[j]))
                de.append((d, labels[i], labels[j]))
        de.sort()
        worst[kind] = de[0]
        report.append((kind, de[0]))
    print("palette check (OKLab dE x100, closest pair per vision type)")
    for kind, (d, a, b) in report:
        floor = 15.0 if kind == "normal" else 8.0
        print(f"   {'PASS' if d >= floor else 'FAIL'}  {kind:13s} "
              f"min dE {d:5.1f}  (floor {floor:.0f})  closest: {a} vs {b}")
    return worst


# ── shared styling ───────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


# ── fig 1 ────────────────────────────────────────────────────────────────────

def fig1(rows, split, out):
    comp = [r for r in rows if r["route_completed"]]
    series = []
    pairs = [("LA (teacher)", "LA", 0), ("GREEDY", "GREEDY", 1)]
    for label, key, ci in pairs:
        v = [100 * (r["duration_h"] - r[key]) / r[key] for r in comp
             if r.get(key) and not r.get(f"{key}_infeasible")]
        series.append((label, np.array(v), OKABE_ITO[ci]))
    v = [100 * (r["duration_h"] - (r["ORACLE_obj"] - 8.0)) / (r["ORACLE_obj"] - 8.0)
         for r in comp if r.get("ORACLE_obj")]
    series.append(("ORACLE (hindsight)", np.array(v), OKABE_ITO[2]))
    series = [s_ for s_ in series if len(s_[1])]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    _style(ax)
    rng = np.random.default_rng(0)
    for k, (label, v, col) in enumerate(series):
        y = len(series) - 1 - k
        ax.scatter(v, y + rng.uniform(-0.16, 0.16, len(v)), s=9,
                   color=col, alpha=0.35, linewidths=0, zorder=2)
        bp = ax.boxplot([v], positions=[y], vert=False, widths=0.46,
                        showfliers=False, patch_artist=True, zorder=3)
        bp["boxes"][0].set(facecolor="none", edgecolor=col, linewidth=2)
        for part in ("whiskers", "caps"):
            for a in bp[part]:
                a.set(color=col, linewidth=1.4)
        bp["medians"][0].set(color=col, linewidth=2.6)
        ax.text(np.median(v), y + 0.34, f"median {np.median(v):+.2f}%",
                ha="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, lw=1.4, zorder=1)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([s[0] for s in reversed(series)], fontsize=9.5, color=INK)
    ax.set_xlabel("student route duration vs baseline  (%)   "
                  "<- student faster    student slower ->",
                  fontsize=9, color=MUTED)
    ax.set_title(f"Closed-loop paired comparison, {split} split "
                 f"({len(comp)} completed routes)",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.set_xlim(np.percentile(np.concatenate([s[1] for s in series]), 0.5) - 2,
                np.percentile(np.concatenate([s[1] for s in series]), 99.5) + 2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"   wrote {out}")


# ── fig 2 ────────────────────────────────────────────────────────────────────

def fig2(abl, split, out):
    abl = [a for a in abl if "med_la" in a]
    tags = [a["tag"] for a in abl]
    med = np.array([a["med_la"] for a in abl])
    halt = np.array([a.get("halted", np.nan) for a in abl])
    y = np.arange(len(tags))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 0.55 * len(tags) + 2.1),
                             dpi=200, gridspec_kw=dict(wspace=0.45))
    for ax, vals, lab, col, fmt in (
            (axes[0], med, "median duration vs LA  (%)", OKABE_ITO[0], "{:+.2f}"),
            (axes[1], halt, "halted routes", OKABE_ITO[1], "{:.0f}")):
        _style(ax)
        ax.barh(y, vals, height=0.6, color=col, alpha=0.9,
                edgecolor=SURFACE, linewidth=2)
        for yy, v in zip(y, vals):
            ax.text(v + (0.02 * np.nanmax(np.abs(vals)) if v >= 0
                         else -0.02 * np.nanmax(np.abs(vals))),
                    yy, fmt.format(v), va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=8.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(tags, fontsize=9, color=INK)
        ax.set_xlabel(lab, fontsize=9, color=MUTED)
        ax.axvline(0, color=INK, lw=1.2)
        pad = 0.18 * max(np.nanmax(np.abs(vals)), 1e-9)
        ax.set_xlim(min(0, np.nanmin(vals)) - pad, max(0, np.nanmax(vals)) + pad)
    axes[0].set_title(f"Ablations ({split} split)", fontsize=11, color=INK,
                      loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"   wrote {out}")


# ── fig 3 ────────────────────────────────────────────────────────────────────

def fig3(rows, split, out):
    per = {}
    for r in rows:
        if r["route_completed"] and r.get("LA") and not r.get("LA_infeasible"):
            per.setdefault(r["family"], []).append(
                100 * (r["duration_h"] - r["LA"]) / r["LA"])
    if not per:
        return
    items = sorted(per.items(), key=lambda kv: np.median(kv[1]))
    labels = [k for k, _ in items]
    meds = np.array([np.median(v) for _, v in items])
    y = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(6.6, 0.22 * len(items) + 1.8), dpi=200)
    _style(ax)
    cols = [DIV_GOOD if m < 0 else DIV_BAD for m in meds]
    ax.barh(y, meds, height=0.68, color=cols, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.2, color=INK)
    ax.set_xlabel("median duration vs LA  (%)", fontsize=9, color=MUTED)
    ax.set_title(f"Per-family medians, {split} split", fontsize=11,
                 color=INK, loc="left", pad=10)
    ax.set_ylim(-0.8, len(items) - 0.2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"   wrote {out}")


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    os.makedirs(FIGS, exist_ok=True)
    validate_palette(OKABE_ITO[:3], ["LA", "GREEDY", "ORACLE"])
    print()
    rows = _load(f"eval_base_{split}.json") or _load(f"eval_gbt_v1_{split}.json")
    if rows:
        fig1(rows, split, os.path.join(FIGS, f"fig1_distributions_{split}.png"))
        fig3(rows, split, os.path.join(FIGS, f"fig3_families_{split}.png"))
    else:
        print("   (no eval rows found)")
    abl = _load(f"ablations_{split}.json")
    if abl:
        fig2(abl, split, os.path.join(FIGS, f"fig2_ablations_{split}.png"))
    else:
        print("   (no ablation results found)")


if __name__ == "__main__":
    main()
