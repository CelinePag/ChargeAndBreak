# -*- coding: utf-8 -*-
"""
check_diesel_timeline.py — audit figure for the §8.4 decomposition.

Takes ONE instance and shows, for the electric and the diesel run of the same
policy, the executed schedule as a timeline of activities, coloured by the
category the §8.4 stack puts each one in.  Underneath, the same activities as
cumulative totals per vehicle, with the EV-minus-diesel difference annotated —
which is exactly what additional_figures.section_diesel averages over the whole
class.  It exists so the categorisation can be eyeballed against a real pair of
schedules rather than trusted.

Driving and customer service are drawn in the timeline (they are most of the
elapsed time and the schedule is unreadable without them) but NOT totalled: both
vehicles cover the same legs and serve the same customers, so they cancel in the
difference and carry no information about electrification.

Everything comes from the runs' own per-stop records — durations_list for the
activity lengths, td_list and sim_trajectory for where they sit on the clock —
so nothing here re-derives a schedule that the simulator already wrote down.

Usage
-----
  python -m src.plot.check_diesel_timeline                       # a long route
  python -m src.plot.check_diesel_timeline --instance RlongCfewTnone_10
  python -m src.plot.check_diesel_timeline --method LA
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src import paths as _paths
from src.plot import additional_figures as af
from src.plot import paper_style as ps

# Category -> (label, colour).  The five dwell categories take the colours of
# the §8.4 stack so a segment here and a segment there are the same thing; the
# two "identical by construction" activities are greys, and the diesel's post-hoc
# fuel stop is hatched because it is credited rather than simulated.
_CATS = [
    ("charge_open", "Charging outside breaks", "#A6461D", None),
    ("queue",       "Charger queueing",        "#DE9350", None),
    ("reposition",  "Bay repositioning",       "#F7DCC0", None),
    ("break",       "Break (incl. charge credited to it)", "#1F4E6B", None),
    ("manoeuvre",   "Stop manoeuvring",        "#3E7FA3", None),
    ("rest",        "Rest",                    "#86B6D0", None),
    ("refuel",      "Refuelling (post hoc)",   "#BFBFBF", "///"),
]
_CTX = [                       # drawn on the timeline, never totalled
    ("drive",   "Driving",          "#F0F0F0"),
    ("service", "Customer service", "#D4D4D4"),
]
_TOTALLED = [c for c in _CATS]


def _blocks(fname: str, fuel_h: float) -> tuple[list, dict, float]:
    """-> (timeline blocks, totals per category, route duration).

    A stop's dwell is laid out in the order the simulator charges it: pull off,
    queue, charge, reposition off the bay, then the break or rest, then service.
    The order within a stop is cosmetic — the totals do not depend on it — but it
    keeps the bands legible and matches the concept figure's convention.
    """
    d = json.load(open(_paths.solution_path(fname), encoding="utf-8"))
    traj, dur, td = d["sim_trajectory"], d["durations_list"], d["td_list"]
    inst = af._instance(str(d.get("instance") or "").split("__")[0]) or {}
    C = {int(i) for i in (inst.get("C") or [])}
    S = {int(k): float(v) for k, v in (inst.get("S") or {}).items()}

    blocks, tot = [], {k: 0.0 for k, _l, _c, _h in _CATS}
    t0 = traj[0]["t_arr"]
    for i in range(len(traj) - 1):
        e = dur[i]
        f = lambda k: float(e.get(k) or 0.0)
        act = (d.get("actions") or [{}] * len(traj))[i]
        brk = str(act.get("break_type") or "0")
        # The charge credited to a break: the MILP's own rule, g_i <= tauc_i and
        # g_i <= the break block.  What is left is charging with nothing to hide
        # behind.  This is the one place the timeline splits a single physical
        # activity, because the split is the whole point of the decomposition.
        masked = min(f("tauc"), af._BREAK_BLOCK_H.get(brk, 0.0))
        seq = [("manoeuvre", f("mstop")),
               ("queue",     f("tauq")),
               ("charge_open", f("tauc") - masked),
               ("break",     masked),
               ("reposition", f("mseq")),
               ("break",     f("taub")),
               ("rest",      f("taur")),
               ("manoeuvre", f("mlay")),
               ("service",   S.get(i, 0.0) if i in C else 0.0)]
        t = traj[i]["t_arr"] - t0
        for kind, length in seq:
            if length > 1e-9:
                blocks.append((kind, t, length))
                if kind in tot:
                    tot[kind] += length
                t += length
        drive = traj[i + 1]["t_arr"] - td[i]
        if drive > 1e-9:
            blocks.append(("drive", td[i] - t0, drive))
    duration = traj[-1]["t_arr"] - t0
    if fuel_h > 0:                       # credited after the fact, so appended
        blocks.append(("refuel", duration, fuel_h))
        tot["refuel"] = fuel_h
        duration += fuel_h
    return blocks, tot, duration


def build(instance: str, method: str, out: str | None = None) -> str:
    alg = {"greedy": "GREEDY", "LA": "LA"}.get(method, method.upper())
    ev, di = af._policy_pair(instance, alg, "diesel")
    if not (ev and di):
        raise SystemExit(f"no paired {alg} runs for {instance} "
                         f"(EV {'yes' if ev else 'no'}, "
                         f"diesel {'yes' if di else 'no'})")
    route = af._route_of(instance) if hasattr(af, "_route_of") else \
        ("long" if "Rlong" in instance else
         "medium" if "Rmedium" in instance else "short")
    fuel = af._refuel_h(route)

    ev_b, ev_t, ev_d = _blocks(ev["file"], 0.0)
    di_b, di_t, di_d = _blocks(di["file"], fuel)

    fig = plt.figure(figsize=(9.6, 5.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 0.5, 1.5], hspace=0.55)
    axe, axd, axb = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    colour = {k: c for k, _l, c, _h in _CATS} | {k: c for k, _l, c in _CTX}
    hatch = {k: h for k, _l, _c, h in _CATS}
    span = max(ev_d, di_d)
    for ax, blocks, dur_h, name in ((axe, ev_b, ev_d, "Electric"),
                                    (axd, di_b, di_d, "Diesel")):
        for kind, t, length in blocks:
            ax.barh(0, length, left=t, height=0.62, color=colour[kind],
                    edgecolor="none", hatch=hatch.get(kind), zorder=3)
        ax.set_xlim(-0.01 * span, span * 1.01)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(ps.BASELINE)
        ax.tick_params(axis="x", labelsize=7, colors=ps.INK_MUTED)
        ax.set_title(f"{name} — {dur_h:.1f} h", loc="left", fontsize=8.5,
                     color=ps.INK_PRIMARY, pad=3)
    axd.set_xlabel("Elapsed time since departure (h)", fontsize=8)

    # ── cumulative totals, EV vs diesel, with the difference annotated ───────
    keys = [k for k, _l, _c, _h in _CATS]
    labels = [l for _k, l, _c, _h in _CATS]
    y = np.arange(len(keys), dtype=float)
    h = 0.36
    # Fill = activity, edge = vehicle.  The pair shares a colour on purpose:
    # the eye should compare the two lengths of one activity, not hunt for the
    # activity across two palettes.
    _EDGE = {"Electric": "#111111", "Diesel": "#9A9A9A"}
    for off, tot, lbl in ((+h / 2, ev_t, "Electric"), (-h / 2, di_t, "Diesel")):
        axb.barh(y + off, [tot[k] for k in keys], h,
                 color=[colour[k] for k in keys],
                 hatch=[hatch.get(k) for k in keys],
                 edgecolor=_EDGE[lbl], linewidth=1.0, zorder=3)
        for yi, k in zip(y, keys):
            if tot[k] > 1e-9:
                axb.annotate(f"{tot[k]:.2f}", (tot[k], yi + off), xytext=(3, 0),
                             textcoords="offset points", va="center",
                             fontsize=6.5, color=ps.INK_MUTED)
    axb.set_yticks(y, labels, fontsize=7.5)
    axb.invert_yaxis()
    axb.set_xlabel("Cumulative time (h)", fontsize=8)
    axb.xaxis.grid(True, color=ps.GRID, lw=0.6)
    axb.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        axb.spines[s_].set_visible(False)
    axb.tick_params(axis="both", length=0, labelsize=7.5)
    # The differences get a column of their own on the right, clear of both the
    # category names and the bars: they are the quantity 8.4 actually reports.
    _xmax = max(max(ev_t.values()), max(di_t.values()))
    axb.set_xlim(0, _xmax * 1.32)
    for yi, k in zip(y, keys):
        delta = ev_t[k] - di_t[k]
        axb.annotate(f"{delta:+.2f} h", (0.995, yi),
                     xycoords=("axes fraction", "data"), ha="right",
                     va="center", fontsize=7.5,
                     color="#B03A2E" if delta > 1e-9 else
                           ("#1E6B52" if delta < -1e-9 else ps.INK_MUTED))
    axb.annotate("EV - diesel", (0.995, -0.8),
                 xycoords=("axes fraction", "data"), ha="right", va="center",
                 fontsize=7.5, color=ps.INK_PRIMARY)
    axb.legend(handles=[Patch(facecolor="white", edgecolor=e, lw=1.0, label=n)
                        for n, e in _EDGE.items()],
               frameon=False, fontsize=7.5, loc="lower right", ncol=2,
               bbox_to_anchor=(0.78, -0.02))

    # Driving and service are on the timelines but out of the totals; say so
    # here rather than leaving the reader to notice the bars do not add up.
    drive_ev = sum(l for k, _t, l in ev_b if k == "drive")
    serv_ev = sum(l for k, _t, l in ev_b if k == "service")
    axb.annotate(
        f"Not totalled (identical by construction): driving {drive_ev:.1f} h, "
        f"customer service {serv_ev:.1f} h.   "
        f"Sum of $\\Delta$ = {sum(ev_t[k] - di_t[k] for k in keys):+.2f} h "
        f"= route duration difference.",
        xy=(0, -0.30), xycoords="axes fraction", fontsize=7,
        color=ps.INK_MUTED)

    handles = [Patch(facecolor=c, hatch=hh, edgecolor="#555555", lw=0.5,
                     label=l) for _k, l, c, hh in _CATS]
    handles += [Patch(facecolor=c, edgecolor="#555555", lw=0.5, label=l)
                for _k, l, c in _CTX]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 0.962), columnspacing=1.1,
               handlelength=1.1, handletextpad=0.4)
    fig.suptitle(f"{instance} — {alg}: activity timeline and totals",
                 x=0.012, ha="left", fontsize=9.5, y=0.998)
    fig.tight_layout(rect=(0.10, 0.05, 1, 0.875))

    out = out or _paths.figure_out(
        f"check_diesel_timeline_{instance}_{alg}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Figure    : {out}")
    print(f"  {'category':<38}{'EV':>9}{'diesel':>9}{'delta':>9}")
    for k, l, _c, _h in _CATS:
        print(f"  {l:<38}{ev_t[k]:9.3f}{di_t[k]:9.3f}"
              f"{ev_t[k] - di_t[k]:+9.3f}")
    print(f"  {'SUM of deltas':<38}{'':>9}{'':>9}"
          f"{sum(ev_t[k] - di_t[k] for k in keys):+9.3f}")
    print(f"  {'route duration difference':<38}{ev_d:9.3f}{di_d:9.3f}"
          f"{ev_d - di_d:+9.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--instance", default="RlongCfewTnone_10")
    ap.add_argument("--method", default="greedy", choices=["greedy", "LA"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build(a.instance, a.method, a.out)
