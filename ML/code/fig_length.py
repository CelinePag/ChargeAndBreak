"""
fig_length.py — trained on short and medium routes, tested on long ones
======================================================================
Three groups of rows, one per set of test routes:

    short + medium (in distribution)   did training only on these cost anything?
    long, 30 paired                    same routes for every model, including
                                       those trained on ALL lengths
    long, all 239                      every long route; none seen in training
                                       by the short+medium models (the models
                                       trained on all lengths are absent here:
                                       they trained on 190 of these routes)

Within a group, one row per model: each arm trained on all lengths (the
reference), then trained on short + medium with the full set F and with the
route-local set R.  Marker = mean over training seeds of the median gap to the
oracle; bar = the spread (sd) over those seeds.  The strip on the right is the
infeasibility rate on the manuscript's ramp.  LA and Greedy sit in every group
as fixed references on the same routes.

    python ML/code/fig_length.py                    -> fig_length.png
    python ML/code/fig_length.py --variant g99sr    -> fig_length_g99sr.png,
        the same models re-scored by `run_length.py --guard 0.99 --spread-room`
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_style import (COLOR, HATCH, INK_MUTED, INK_PRIMARY,      # noqa: E402
                      apply_rc, infeas_color, shade, style_axes)
from paper_link import baseline_gaps, gaps, oracle_facts         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
ARM_LABEL = {"gbt": "Trees", "clf": "Classifier", "mlp": "MLP"}
MARK = {"gbt": "o", "clf": "s", "mlp": "^"}


def length_of(inst):
    return inst[1:].split("C")[0]


def per_instance(eval_file):
    """instance -> (student gap or None, LA gap, Greedy gap)."""
    with open(os.path.join(RESULTS, eval_file)) as fh:
        rows = json.load(fh)
    out = {}
    for r in rows:
        orc = oracle_facts(r["instance"])
        if orc is None:
            continue
        t0 = 8.0
        g = None
        if r.get("route_completed") and r.get("duration_h") is not None:
            g, _ = gaps(r["duration_h"], r.get("tw_misses", 0),
                        r["duration_h"] + t0, orc)
        b = baseline_gaps(r["instance"], orc)
        out[r["instance"]] = (g, (b.get("LA_MIPTAIL") or (None,))[0],
                              (b.get("GREEDY") or (None,))[0])
    return out


def stats_over_seeds(per_seed):
    """per_seed: list of {instance: gap-or-None}.  -> (mean med, sd, inf rate)."""
    meds, rates = [], []
    for m in per_seed:
        v = [g for g in m.values() if g is not None]
        if v:
            meds.append(np.median(v))
        rates.append(sum(1 for g in m.values() if g is None) / max(len(m), 1))
    if not meds:
        return None
    return float(np.mean(meds)), float(np.std(meds)), float(np.mean(rates))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="g95",
                    help="g95 (as trained), g95sr, g99sr -- see run_length.py")
    args = ap.parse_args()
    variant = args.variant
    tail = "" if variant == "g95" else f"_{variant}"
    with open(os.path.join(RESULTS, f"length_test{tail}.json")) as fh:
        cells = json.load(fh)

    groups = {"in": ("Short + medium routes (in distribution)", lambda i: length_of(i) != "long"),
              "lp": ("Long routes — 30, paired with all-length models", lambda i: length_of(i) == "long"),
              "la": ("Long routes — all 239, none seen in training", None)}

    # model rows in a fixed order: per arm, all-lengths then SM-F then SM-R
    models = []
    for arm in ("gbt", "clf", "mlp"):
        for scope in ("all", "SM"):
            for fl in sorted({c["fset_label"] for c in cells
                              if c["arm"] == arm and c["scope"] == scope},
                             key=lambda s: (s[0] != "F", s)):
                sub = [c for c in cells if c["arm"] == arm and c["scope"] == scope
                       and c["fset_label"] == fl]
                if sub:
                    who = "all lengths" if scope == "all" else "short+medium"
                    models.append((arm, scope, fl, f"{ARM_LABEL[arm]} {fl}, trained {who}", sub))

    plot_rows, headers, y = [], [], 0.0
    for gk, (glab, keep) in groups.items():
        headers.append((y, glab)); y += 0.95
        ref_la, ref_gr = None, None
        for arm, scope, fl, lab, sub in models:
            if gk == "la" and scope == "all":
                continue                         # in-sample for these models
            per_seed = []
            for c in sub:
                f = (f"eval_{c['tag']}_{variant}_longall.json" if gk == "la"
                     else f"eval_{c['tag']}_{variant}_test.json")
                if not os.path.exists(os.path.join(RESULTS, f)):
                    continue
                pi = per_instance(f)
                if keep is not None:
                    pi = {k: v for k, v in pi.items() if keep(k)}
                per_seed.append({k: v[0] for k, v in pi.items()})
                if ref_la is None:
                    ref_la = [v[1] for v in pi.values() if v[1] is not None]
                    ref_gr = [v[2] for v in pi.values() if v[2] is not None]
            st = stats_over_seeds(per_seed)
            if st:
                plot_rows.append((y, arm, scope, lab, st)); y += 1.0
        for nm, v, key in (("LA", ref_la, "LA"), ("Greedy", ref_gr, "greedy")):
            if v:
                plot_rows.append((y, key, "ref", nm, (float(np.median(v)), 0.0, 0.0)))
                y += 1.0
        y += 0.5

    apply_rc()
    fig = plt.figure(figsize=(7.6, 0.25 * len(plot_rows) + 1.9), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[11, 1], wspace=0.035)
    ax = fig.add_subplot(gs[0, 0])
    axf = fig.add_subplot(gs[0, 1], sharey=ax)
    style_axes(ax)
    style_axes(axf, xgrid=False)

    for yy, key, scope, lab, (m, sd, rate) in plot_rows:
        col = COLOR.get(key, COLOR["gbt"])
        hollow = key in ("gbt", "clf", "mlp")
        if sd > 0:
            ax.plot([m - sd, m + sd], [yy, yy], color=shade(col, 0.1), lw=2.2,
                    solid_capstyle="round", zorder=2)
        mk = MARK.get(key, "D" if key == "LA" else "v")
        ax.plot([m], [yy], marker=mk, ms=5.4,
                color=shade(col, 0.25 if scope != "all" else 0.45),
                mfc="white" if (hollow and scope == "SM") else col,
                mew=1.3, zorder=3)
        ax.text(m + max(sd, 0) + 0.05, yy, f" {m:+.2f}", va="center",
                fontsize=5.9, color=INK_MUTED)
        axf.barh([yy], [1.0], height=0.78,
                 color=infeas_color(rate) if scope != "ref" else "#ffffff",
                 edgecolor="white", linewidth=0.4)
        if scope != "ref":
            axf.text(0.5, yy, f"{100*rate:.0f}", ha="center", va="center",
                     fontsize=5.4, color=INK_PRIMARY if rate < 0.12 else "white")
    for yh, glab in headers:
        ax.text(0.0, yh, glab, transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=7.6, color=INK_PRIMARY, fontweight="bold")

    ax.set_yticks([r[0] for r in plot_rows])
    ax.set_yticklabels([r[3] for r in plot_rows], fontsize=6.3, color=INK_MUTED)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_PRIMARY, lw=0.9)
    hi = max(r[4][0] + r[4][1] for r in plot_rows)
    ax.set_xlim(-0.3, hi * 1.18)
    ax.set_xlabel("gap to the hindsight oracle (%, lower is better)   "
                  "marker = mean over training seeds, bar = ± sd")
    extra = {"g95sr": "  (+ spread-room check)",
             "g99sr": "  (+ spread-room check, 0.99 drive guard)"}.get(variant, "")
    ax.set_title("Trained on short and medium routes — tested on long ones" + extra,
                 fontsize=9.5, color=INK_PRIMARY, loc="left", pad=6)
    axf.set_xticks([]); axf.set_xlim(0, 1)
    axf.set_xlabel("infeas.\n%", fontsize=6.5, color=INK_MUTED)
    axf.spines["left"].set_visible(False)
    axf.spines["bottom"].set_visible(False)
    axf.tick_params(left=False, labelleft=False)
    fig.text(0.01, 0.005, "Filled markers: trained on all lengths.  Hollow: "
             "trained on short + medium only.  F = full engineered features; "
             "R = route-local (no whole-route position features).",
             fontsize=6.3, color=INK_MUTED)
    out = os.path.join(FIGS, f"fig_length{tail}.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
