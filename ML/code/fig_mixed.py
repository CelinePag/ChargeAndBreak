"""
fig_mixed.py — chargers that change along the route
===================================================
Two figures from mixed_eval.py's results (routes from mixed_instances.py,
each paired with its uniform original, short + medium routes, seeds 22-25).

fig_mixed_change.png
    One group per kind of mixing.  Each row is a policy's gap to the oracle
    on the MIXED route minus its gap on the SAME route with uniform chargers:
    what the mixing alone costs it.  Marker = mean, bar = 95 % interval;
    zero = no loss.  Oracles not certified to 1 % are dropped (2 `mix`
    routes).  The LA row covers only the routes run_la_mixed.py has labelled
    so far; its n says how many.

fig_energy_share.png
    Where each policy takes its energy on the mixed-POWER routes: the share
    of all kWh charged at 150 / 350 / 700 / 1000 kW chargers.  (a) all 32
    routes; (b) the routes the LA has run, with the LA.  Greedy and the
    students are re-driven (deterministic); the result is cached in
    ML/results/mixed_energy_*.json (--refresh recomputes).

    python ML/code/fig_mixed.py [--refresh]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.patches import Patch            # noqa: E402
import numpy as np                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_style import (COLOR, INK_MUTED, INK_PRIMARY, apply_rc,   # noqa: E402
                      shade, style_axes)
import mixed_eval as me                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

GROUPS = [("dmix", "Charger SPACING changes along the route (~30 / 60 / 120 km thirds)"),
          ("pmix", "Charger POWER changes along the route (150 / 350 / 700 / 1000 kW)"),
          ("mix", "Both")]
# (method label in mixed_eval rows, row label, marker, filled, style key)
ROWS = [("LA", "LA (the teacher)", "D", True, "LA"),
        ("Greedy", "Greedy", "v", True, "greedy"),
        ("trees, base only", "Trees, base case only", "o", False, "gbt"),
        ("trees, all physics", "Trees, all physics", "o", True, "gbt"),
        ("trees, all physics + power", "Trees, all physics + power features", "s", True, "gbt")]
POWER = [150, 350, 700, 1000]
POWER_GREY = {150: "#D0D0D0", 350: "#9A9A9A", 700: "#666666", 1000: "#333333"}
XMIN, XMAX = -3.0, 11.0


def changes():
    """{variant: {method: [paired change, pp]}} on certified oracles."""
    rows = me.read_rows(me.store_path(0.99, True))
    out = {}
    for v, _ in GROUPS:
        names = [n for vv, n, _p in me.instances([v], uniform=False)]
        d = {}
        for name in names:
            base = name.split("__")[0]
            o_m, o_u = me.read_oracle(v, name), me.read_oracle("uniform", base)
            if not (o_m and o_u) or (o_m["gap"] or 0) > 0.01 or (o_u["gap"] or 0) > 0.01:
                continue
            got = {}
            for vv, nm, orc in ((v, name, o_m), ("uniform", base, o_u)):
                cand = [r for r in rows if r["variant"] == vv and r["instance"] == nm]
                la = me.la_row(vv, nm) if vv == "uniform" and me.la_solution(v, name) else \
                    (me.la_row(vv, nm) if vv != "uniform" else None)
                if la:
                    cand.append(la)
                for r in cand:
                    g = me._gap(r, orc) if r["completed"] else None
                    if g is not None:
                        got[(r["method"], vv)] = g
            for m in {k[0] for k in got}:
                if (m, v) in got and (m, "uniform") in got:
                    d.setdefault(m, []).append(got[(m, v)] - got[(m, "uniform")])
        out[v] = d
    return out


def fig_change(ch):
    rows, headers, bands = [], [], []
    y = 0.0
    for gi, (v, label) in enumerate(GROUPS):
        headers.append((y, label))
        y += 0.95
        y0 = y
        for key, lab, mk, filled, ck in ROWS:
            dd = ch.get(v, {}).get(key)
            if not dd or len(dd) < 2:
                continue
            a = np.array(dd)
            se = a.std(ddof=1) / np.sqrt(len(a))
            rows.append((y, lab, mk, filled, ck, a.mean(), se, len(a)))
            y += 1.0
        bands.append((y0 - 0.5, y - 0.5, gi))
        y += 0.45

    apply_rc()
    fig, ax = plt.subplots(figsize=(7.2, 0.23 * len(rows) + 0.27 * len(headers) + 1.2),
                           dpi=200)
    style_axes(ax)
    for ylo, yhi, gi in bands:
        if gi % 2 == 0:
            ax.axhspan(ylo, yhi, color="#f4f4f4", zorder=0, lw=0)
    for yh, text in headers:
        ax.text(0.0, yh, text, transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=7.2, color=INK_PRIMARY, fontweight="bold",
                zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    for yy, lab, mk, filled, ck, mean, se, n in rows:
        col = COLOR[ck]
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        ax.plot([max(lo, XMIN), min(hi, XMAX)], [yy, yy], color=shade(col, 0.1),
                lw=2.0, solid_capstyle="round", zorder=2)
        ax.plot([min(max(mean, XMIN), XMAX)], [yy], marker=mk, ms=5.0,
                color=shade(col, 0.35), mfc=col if filled else "white", mew=1.2,
                zorder=3, clip_on=False)
        ax.text(min(hi, XMAX) + 0.12, yy, f"{mean:+.1f}", va="center",
                fontsize=5.8, color=INK_MUTED)
        ax.text(1.005, yy, f"n={n}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=5.4, color=INK_MUTED)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=6.2, color=INK_MUTED)
    ax.tick_params(axis="y", pad=2, length=0)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_PRIMARY, lw=0.9, zorder=1)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel("gap to the oracle on the mixed route minus on the same route "
                  "with uniform chargers\n(percentage points; 0 = mixing costs "
                  "nothing)   marker = mean, bar = 95 % interval")
    ax.set_title("What mixing chargers along a route costs each policy",
                 fontsize=9.0, color=INK_PRIMARY, loc="left", pad=6)
    ax.annotate("Short and medium routes, seeds 22-25, paired by route; oracles "
                "with one charging curve per charger, certified to 1 %. The LA has "
                "run on a subset so far (its n). No tree model broke a rule on "
                "any route.", xy=(0, 0), xycoords="axes fraction",
                xytext=(-150, -38), textcoords="offset points", va="top",
                fontsize=6.0, color=INK_MUTED, annotation_clip=False)
    out = os.path.join(FIGS, "fig_mixed_change.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def energy(la_only, refresh):
    path = os.path.join(RESULTS, f"mixed_energy_pmix{'_la' if la_only else ''}.json")
    if os.path.exists(path) and not refresh:
        with open(path) as fh:
            c = json.load(fh)
        if not la_only or c["n_routes"] == len(me.la_routes("pmix")):
            return c
    E, N, n = me.energy_by_power("pmix", la_only=la_only)
    c = dict(E={m: {str(k): v for k, v in d.items()} for m, d in E.items()},
             N={m: {str(k): v for k, v in d.items()} for m, d in N.items()},
             n_routes=n)
    with open(path, "w") as fh:
        json.dump(c, fh, indent=1)
    return c


def fig_energy(c_all, c_la):
    labels = {"oracle": ("Oracle (hindsight)", "oracle"), "LA": ("LA (the teacher)", "LA"),
              "Greedy": ("Greedy", "greedy"),
              "trees, base only": ("Trees, base case only", "gbt"),
              "trees, all physics": ("Trees, all physics", "gbt"),
              "trees, all physics + power": ("Trees, all physics + power features", "gbt")}
    order = list(labels)
    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), dpi=200, sharex=True)
    for ax, c, title in ((axes[0], c_all, f"(a) all {c_all['n_routes']} mixed-power routes"),
                         (axes[1], c_la, f"(b) the {c_la['n_routes']} routes the LA has run")):
        style_axes(ax, xgrid=False)
        ms = [m for m in order if m in c["E"]]
        for i, m in enumerate(ms):
            e = {int(k): v for k, v in c["E"][m].items()}
            tot = sum(e.values()) or 1.0
            left = 0.0
            for kw in POWER:
                w = 100 * e.get(kw, 0.0) / tot
                ax.barh(i, w, left=left, height=0.66, color=POWER_GREY[kw],
                        edgecolor="white", linewidth=1.2, zorder=2)
                if w >= 7:
                    ax.text(left + w / 2, i, f"{w:.0f}", ha="center", va="center",
                            fontsize=6.0, zorder=3,
                            color="white" if kw >= 700 else INK_PRIMARY)
                left += w
            # identity mark beside the label: colour follows the entity
            ax.plot([-3.5], [i], marker="s", ms=4.5, color=COLOR[labels[m][1]],
                    mec=shade(COLOR[labels[m][1]], 0.35), clip_on=False)
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([labels[m][0] for m in ms], fontsize=6.4, color=INK_MUTED)
        ax.tick_params(axis="y", pad=12, length=0)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=7.6, color=INK_PRIMARY, loc="left", pad=4)
        ax.set_xlabel("share of the energy charged (%)")
        ax.spines["left"].set_visible(False)
    fig.legend(handles=[Patch(fc=POWER_GREY[k], ec="white", label=f"{k} kW") for k in POWER],
               title="charged at a charger of", loc="upper center", ncol=4,
               frameon=False, fontsize=6.6, title_fontsize=6.6,
               bbox_to_anchor=(0.55, 1.07))
    fig.suptitle("Where each policy charges when charger power varies along the route",
                 fontsize=9.0, color=INK_PRIMARY, x=0.01, y=1.16, ha="left")
    fig.tight_layout(w_pad=2.5)
    out = os.path.join(FIGS, "fig_energy_share.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    fig_change(changes())
    fig_energy(energy(False, args.refresh), energy(True, args.refresh))


if __name__ == "__main__":
    main()
