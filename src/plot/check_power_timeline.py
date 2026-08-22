# -*- coding: utf-8 -*-
"""
check_power_timeline.py — audit figure for the charger-power non-monotonicity.

Raising the charge point from 700 kW to 1000 kW makes the greedy WORSE on 236
of the 251 feasible instance pairs.  That is not noise, so this figure takes ONE
pair and draws what actually happens along the route.

THE MECHANISM
-------------
greedy.py's MUST-CHARGE branch credits a mandatory 45' break to a charging stop
only when the charge is long enough to cover it (``tauc_est >= Tb45``, 0.75 h --
see greedy.py:289).  A must-charge stop refills roughly the same energy either
way, so the power sets the dwell:

    700 kW  ->  tauc ~ 0.77-0.84 h  >= 0.75  ->  b45 credited, the break is FREE
    1000 kW ->  tauc ~ 0.69-0.72 h  <  0.75  ->  no break, the charge is just a charge

The faster charger saves ~5 minutes per stop and forfeits a 45-minute break in
exchange.  The break still has to happen -- it reappears later as a standalone
stop -- and because every downstream event now sits later on the duty clock, the
daily rest eventually lands on the far side of a HOS boundary and the schedule
takes an extra 9 h rest.

The figure says this in four panels:
  (1,2) the two executed schedules on a shared clock, coloured by activity, with
        each must-charge stop marked by whether the break was credited there;
  (3)   per charging stop, the charge duration against the 0.75 h threshold --
        the crossing that causes everything above it;
  (4)   cumulative totals per activity with the difference, which is where the
        makespan gap actually lands (rest, not charging).

Driving and customer service are drawn but NOT totalled: both runs cover the
same legs with the same realised travel times and serve the same customers, so
they cancel exactly (checked: 0.000 h residual) and carry no information about
charger power.

WHY LA DOES NOT DO THIS
-----------------------
Run with ``--method LA`` and the third panel inverts: LA's charges sit mostly
BELOW 0.75 h at both powers and are credited anyway (93 % of charging stops at
700 kW, 94 % at 1000 kW, against the greedy's 82 % -> 19 %).  Both methods share
the same crediting arithmetic -- MILP.py:338, ``taub_hat = taub + g`` -- so a
0.70 h charge always earns 0.70 h of credit and only the 0.05 h shortfall is
paid.  The difference is who decides where the break goes.  The greedy decides
myopically at the current stop with an all-or-nothing test on the charge it
needs THERE, so falling below the block forfeits the whole 0.75 h; the MILP
picks break placement and charge lengths jointly over the lookahead horizon, so
it can lengthen a charge to exactly fill the block (LA parks 23-30 charges at
precisely 0.750 h) or keep the break there and pay only the difference.

LA is not immune to the second-order effect: RlongCfewTnone_22 regresses by
+9.8 h with every charging category IMPROVING, purely because the faster
schedule pushes a duty boundary and buys an extra daily rest.  That is the
HOS-alignment mechanism without the crediting collapse, and it is the reason
this figure separates the two.

Usage
-----
  python -m src.plot.check_power_timeline                      # default pair
  python -m src.plot.check_power_timeline --instance RlongCfewTnone_5
  python -m src.plot.check_power_timeline --low kw350 --high kw1000
  python -m src.plot.check_power_timeline --list          # worst pairs on disk
"""

from __future__ import annotations

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src import paths as _paths
from src.plot import additional_figures as af
from src.plot import paper_style as ps

# Category -> (label, colour, hatch).  Warm = time the charger costs, cool =
# time the hours-of-service rules cost, exactly as in the section 8.4 stack, so
# a segment here means the same thing as a segment there.  The categories
# partition the dwell -- the hatch on the first row marks charge time that also
# discharges a break obligation, it is not a second row of clock time -- so the
# deltas sum to the makespan difference with nothing to net out.
_CATS = [
    ("charge_masked", "Charging, break credited to it",   "#A6461D", "///"),
    ("charge_open",   "Charging, no break credited",      "#A6461D", None),
    ("queue",         "Charger queueing",                 "#DE9350", None),
    ("reposition",    "Bay repositioning",                "#F7DCC0", None),
    ("break",         "Break time not covered by a charge", "#1F4E6B", None),
    ("manoeuvre",     "Stop manoeuvring",                 "#3E7FA3", None),
    ("rest",          "Rest",                             "#86B6D0", None),
]
_CTX = [                       # drawn on the timeline, never totalled
    ("drive",   "Driving",          "#F0F0F0"),
    ("service", "Customer service", "#D4D4D4"),
]

_TB45 = 0.75                   # h -- the break block the crediting rule tests


def _read(fname: str) -> tuple[list, dict, list, float, float, float]:
    """-> (blocks, totals, charge stops, makespan, drive total, service total).

    A stop's dwell is laid out in the order the simulator charges it: pull off,
    queue, charge, reposition off the bay, then the break or rest, then service.
    The order within a stop is cosmetic; the totals do not depend on it.
    """
    d = json.load(open(_paths.solution_path(fname), encoding="utf-8"))
    traj, dur, td = d["sim_trajectory"], d["durations_list"], d["td_list"]
    inst = af._instance(str(d.get("instance") or "").split("__")[0]) or {}
    C = {int(i) for i in (inst.get("C") or [])}
    S = {int(k): float(v) for k, v in (inst.get("S") or {}).items()}

    blocks, tot = [], {k: 0.0 for k, _l, _c, _h in _CATS}
    charge_stops, drive_h, serv_h = [], 0.0, 0.0
    t0 = traj[0]["t_arr"]
    for i in range(len(traj) - 1):
        e = dur[i]
        f = lambda k: float(e.get(k) or 0.0)
        act = (d.get("actions") or [{}] * len(traj))[i]
        brk = str(act.get("break_type") or "0")
        block = af._BREAK_BLOCK_H.get(brk, 0.0)
        tauc = f("tauc")
        # The charge credited to a break: the MILP's own rule, g_i <= tauc_i and
        # g_i <= the break block.  Splitting the physical activity here is the
        # whole point -- free charging and paid charging look identical on the
        # clock, and which one a stop gets is the entire difference between the
        # two runs.
        masked = min(tauc, block)
        if tauc > 1e-9:
            charge_stops.append((traj[i]["t_arr"] - t0, tauc, masked > 1e-9))
        svc = S.get(i, 0.0) if i in C else 0.0
        serv_h += svc
        seq = [("manoeuvre",     f("mstop")),
               ("queue",         f("tauq")),
               ("charge_masked", masked),
               ("charge_open",   tauc - masked),
               ("reposition",    f("mseq")),
               ("break",         f("taub")),
               ("rest",          f("taur")),
               ("manoeuvre",     f("mlay")),
               ("service",       svc)]
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
            drive_h += drive
    return (blocks, tot, charge_stops, traj[-1]["t_arr"] - t0, drive_h, serv_h)


def _pairs(low: str, high: str, alg: str = "GREEDY") -> list[tuple]:
    """Every feasible (low, high) variant pair on disk, worst regression first.

    Goes through run_cache rather than af._policy: one pass over the parsed
    corpus instead of four globs per instance over a 16 000-file directory.
    """
    from src.output_analysis import run_cache
    pat = re.compile(rf"^(?P<stem>.+?)__(?P<tag>{low}|{high})_{alg}_"
                     r"(?P<ts>\d{8}_\d{6})_\d+\.json$")
    best: dict = {}
    for name, d in run_cache.runs_by_name(quiet=True).items():
        m = pat.match(name)
        if not m or d.get("duration_h") is None:
            continue
        if (d.get("metrics") or {}).get("run_infeasible"):
            continue
        k = (m["stem"], m["tag"])
        if k not in best or m["ts"] > best[k][0]:     # newest run wins
            best[k] = (m["ts"], name, float(d["duration_h"]))
    out = []
    for stem, tag in list(best):
        if tag != low:
            continue
        a, b = best[(stem, low)], best.get((stem, high))
        if b:
            out.append((b[2] - a[2], stem, a[2], b[2], a[1], b[1]))
    return sorted(out, reverse=True)


def build(instance: str, low: str, high: str, alg: str = "GREEDY",
          out: str | None = None) -> str:
    rows = {r[1]: r for r in _pairs(low, high, alg)}
    if instance not in rows:
        raise SystemExit(f"no feasible {alg} pair for {instance} "
                         f"({low} vs {high}); try --list")
    delta, _s, dur_lo, dur_hi, f_lo, f_hi = rows[instance]
    kw = lambda t: t.replace("kw", "") + " kW"

    lo = _read(f_lo)
    hi = _read(f_hi)
    lo_b, lo_t, lo_cs, lo_d, lo_drv, lo_srv = lo
    hi_b, hi_t, hi_cs, hi_d, hi_drv, hi_srv = hi

    ps.apply_rc()
    fig = plt.figure(figsize=(10.0, 7.6))
    # Explicit margins rather than tight_layout: the panels carry annotations in
    # mixed ("axes fraction", "data") coordinates, which tight_layout cannot
    # measure and warns about.
    gs = fig.add_gridspec(4, 1, height_ratios=[0.42, 0.42, 0.85, 1.50],
                          hspace=0.72, left=0.20, right=0.985,
                          top=0.845, bottom=0.105)
    axl, axh, axc, axb = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                          fig.add_subplot(gs[2]), fig.add_subplot(gs[3]))

    colour = {k: c for k, _l, c, _h in _CATS} | {k: c for k, _l, c in _CTX}
    hatch = {k: h for k, _l, _c, h in _CATS}
    span = max(lo_d, hi_d)

    # -- the two executed schedules, on one clock ----------------------------
    for ax, pack, name in ((axl, lo, kw(low)), (axh, hi, kw(high))):
        blocks, _t, cs, dur_h = pack[0], pack[1], pack[2], pack[3]
        for kind, t, length in blocks:
            ax.barh(0, length, left=t, height=0.60, color=colour[kind],
                    edgecolor="none", hatch=hatch.get(kind), zorder=3)
        # Mark every must-charge stop by what the crediting rule decided there:
        # a filled tick means the 45' break came free, a hollow one means it did
        # not and will have to be taken separately later.
        for t, _tauc, credited in cs:
            ax.plot([t], [0.46], marker="v", ms=4.2, zorder=5,
                    color="#1F4E6B" if credited else "white",
                    mec="#1F4E6B", mew=0.9)
        ax.set_xlim(-0.01 * span, span * 1.01)
        ax.set_ylim(-0.42, 0.62)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(ps.BASELINE)
        ax.tick_params(axis="x", labelsize=7, colors=ps.INK_MUTED)
        ax.set_title(f"{name} charge point — {dur_h:.1f} h", loc="left",
                     fontsize=8.5, color=ps.INK_PRIMARY, pad=3)

    # -- the threshold that causes it ---------------------------------------
    # Zoomed hard on the crossing: the whole effect lives in the ~6 minutes
    # between 0.69 h and 0.81 h, and a 0-anchored axis hides it.  Stems to the
    # threshold rather than a series line, because consecutive charging stops
    # are not a trend and joining them invents one.
    axc.axhline(_TB45, color="#B03A2E", lw=1.0, ls="--", zorder=2)
    _all_v = [c[1] for c in lo_cs + hi_cs] + [_TB45]
    # A dead band under the data so the legend has somewhere to sit: LA charges
    # in many short top-ups and fills the lower half of this panel, where the
    # greedy leaves it empty.
    _pad = max(0.06, 0.16 * (max(_all_v) - min(_all_v)))
    _y0, _y1 = min(_all_v) - 2.4 * _pad, max(_all_v) + 1.4 * _pad
    for cs, mark in ((lo_cs, "o"), (hi_cs, "s")):
        for t, v, credited in cs:
            axc.plot([t, t], [_TB45, v], lw=0.7, zorder=3,
                     color="#1E6B52" if v >= _TB45 else "#B03A2E")
            axc.plot([t], [v], marker=mark, ms=5.0, zorder=4, mew=1.0,
                     mec="#1F4E6B", mfc="#1F4E6B" if credited else "white",
                     ls="none")
    axc.set_ylim(_y0, _y1)
    axc.set_xlim(-0.01 * span, span * 1.01)
    axc.set_ylabel("Charge at\nstop (h)", fontsize=7.5)
    axc.yaxis.grid(True, color=ps.GRID, lw=0.6)
    axc.set_axisbelow(True)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    axc.tick_params(axis="both", labelsize=7, length=0)
    axc.set_xlabel("Elapsed time since departure (h)", fontsize=8)
    # Above the data cloud rather than on the line: at this zoom the line runs
    # straight through the markers and any label riding it collides with them.
    axc.annotate(f"$T^{{b45}}$ = {_TB45:.2f} h — a charge shorter than this no "
                 "longer carries the break, so the break must be taken later",
                 (0.995, 0.97), xycoords="axes fraction", ha="right",
                 va="top", fontsize=7, color="#B03A2E")
    # Shape = charger power, fill = whether the break was credited (same
    # convention as the triangles on the timelines), so the legend markers are
    # both drawn hollow and the fill is explained in words.
    axc.legend(handles=[Line2D([], [], ls="none", marker=m, ms=5.0, mew=1.0,
                               mfc="white", mec="#1F4E6B", label=n)
                        for m, n in (("o", kw(low)), ("s", kw(high)))]
                       + [Line2D([], [], ls="none", marker="o", ms=5.0,
                                 mew=1.0, mfc="#1F4E6B", mec="#1F4E6B",
                                 label="filled = break credited")],
               frameon=False, fontsize=7, loc="lower left", ncol=3,
               handletextpad=0.3, columnspacing=1.2)

    # -- cumulative totals, with the difference ------------------------------
    keys = [k for k, _l, _c, _h in _CATS]
    labels = [l for _k, l, _c, _h in _CATS]
    y = np.arange(len(keys), dtype=float)
    h = 0.36
    # Fill = activity, edge = charger power.  The pair shares a colour on
    # purpose: the eye should compare the two lengths of one activity, not hunt
    # for the activity across two palettes.
    _EDGE = {kw(low): "#111111", kw(high): "#9A9A9A"}
    for off, tot, lbl in ((+h / 2, lo_t, kw(low)), (-h / 2, hi_t, kw(high))):
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
    # category names and the bars: they are the quantity the section reports.
    _xmax = max(max(lo_t.values()), max(hi_t.values()))
    axb.set_xlim(0, _xmax * 1.34)
    for yi, k in zip(y, keys):
        dk = hi_t[k] - lo_t[k]
        axb.annotate(f"{dk:+.2f} h", (0.995, yi),
                     xycoords=("axes fraction", "data"), ha="right",
                     va="center", fontsize=7.5,
                     color="#B03A2E" if dk > 1e-9 else
                           ("#1E6B52" if dk < -1e-9 else ps.INK_MUTED))
    axb.annotate(f"{kw(high)} − {kw(low)}", (0.995, -0.85),
                 xycoords=("axes fraction", "data"), ha="right", va="center",
                 fontsize=7.5, color=ps.INK_PRIMARY)
    axb.legend(handles=[Patch(facecolor="white", edgecolor=e, lw=1.0, label=n)
                        for n, e in _EDGE.items()],
               frameon=False, fontsize=7.5, loc="center left", ncol=2,
               bbox_to_anchor=(0.30, 0.45))

    # Driving and service are on the timelines but out of the totals; say so
    # rather than leaving the reader to notice the bars do not add up.
    net = sum(hi_t[k] - lo_t[k] for k in keys)
    axb.annotate(
        f"Not totalled (identical by construction): driving {lo_drv:.1f} h, "
        f"customer service {lo_srv:.1f} h — same legs, same realised travel "
        f"times, same customers.   $\\Sigma\\Delta$ = {net:+.2f} h "
        f"= makespan difference, so the dwell accounts for all of it.",
        xy=(0, -0.245), xycoords="axes fraction", fontsize=7,
        color=ps.INK_MUTED)

    handles = [Patch(facecolor=c, hatch=hh, edgecolor="#555555", lw=0.5,
                     label=l) for _k, l, c, hh in _CATS]
    handles += [Patch(facecolor=c, edgecolor="#555555", lw=0.5, label=l)
                for _k, l, c in _CTX]
    handles += [Line2D([], [], ls="none", marker="v", ms=4.5, color="#1F4E6B",
                       mec="#1F4E6B", label="Must-charge stop, break credited"),
                Line2D([], [], ls="none", marker="v", ms=4.5, color="white",
                       mec="#1F4E6B", mew=0.9,
                       label="Must-charge stop, break NOT credited")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 0.972), columnspacing=1.1,
               handlelength=1.1, handletextpad=0.4)
    # The pair is not always a regression -- LA usually improves with power --
    # so the verdict is read off the sign rather than assumed.
    verdict = "WORSE" if delta > 1e-6 else "BETTER" if delta < -1e-6 else "EQUAL"
    fig.suptitle(f"{instance} — {alg}: the {kw(high)} charge point is "
                 f"{abs(delta):.1f} h {verdict} than {kw(low)}",
                 x=0.012, ha="left", fontsize=9.5, y=0.985)

    out = out or _paths.figure_out(
        f"check_power_timeline_{instance}_{alg}_{low}_{high}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)

    n_lo = sum(1 for c in lo_cs if c[2])
    n_hi = sum(1 for c in hi_cs if c[2])
    print(f"  Figure    : {out}")
    print(f"  makespan  : {kw(low)} {dur_lo:.3f} h   {kw(high)} {dur_hi:.3f} h"
          f"   ({delta:+.3f} h)")
    print(f"  breaks credited to a charge: {n_lo}/{len(lo_cs)} stops at "
          f"{kw(low)}, {n_hi}/{len(hi_cs)} at {kw(high)}")
    print(f"  {'category':<34}{kw(low):>10}{kw(high):>10}{'delta':>10}")
    for k, l, _c, _h in _CATS:
        print(f"  {l:<34}{lo_t[k]:10.3f}{hi_t[k]:10.3f}"
              f"{hi_t[k] - lo_t[k]:+10.3f}")
    print(f"  {'SUM of deltas':<34}{'':>10}{'':>10}"
          f"{net:+10.3f}")
    print(f"  {'makespan difference':<34}{lo_d:10.3f}{hi_d:10.3f}"
          f"{hi_d - lo_d:+10.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--instance", default="RmediumCmanyTtight_13")
    ap.add_argument("--low", default="kw700")
    ap.add_argument("--high", default="kw1000")
    ap.add_argument("--method", default="greedy")
    ap.add_argument("--list", action="store_true",
                    help="print the worst pairs on disk and exit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    alg = {"greedy": "GREEDY", "LA": "LA"}.get(a.method, a.method.upper())
    if a.list:
        rows = _pairs(a.low, a.high, alg)
        worse = sum(1 for r in rows if r[0] > 1e-6)
        print(f"  {len(rows)} feasible pairs; {worse} where {a.high} is worse")
        for r in rows[:25]:
            print(f"  {r[0]:+8.3f}  {r[1]:<24}{r[2]:9.3f}{r[3]:9.3f}")
    else:
        build(a.instance, a.low, a.high, alg, a.out)
