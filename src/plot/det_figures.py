"""
det_figures.py — what a DETERMINISTIC plan costs (the VSS experiment)
=====================================================================
Reads every DET run in solutions/VSS/ (deterministic MILP built at nominal
travel times — see methods/DET.py) and reports what happened when the plan met
a realised draw, under both execution rules:

    raw       the plan is driven as is, nothing intervenes
    guarded   the S1 supervisor may force a charge / break / rest before
              departing a stop whose next leg would breach at the 0.95
              quantile (runner_dispatch --supervised)

    python -m src.plot.det_figures            # figure + tables
    python -m src.plot.det_figures --table    # tables only

Outputs
-------
    figures/VSS/vss_det_outcome.{pdf,png}     four-panel figure
    data_output/vss_det_summary.csv           per-cell table

The figure pools CUSTOMER classes: across the 24 cells the customer axis moves
the failure rate by less than two percentage points while route length halves
the distance covered, so it is route, not customer count, that the outcome
depends on.  Time windows are pooled for the same reason.

Panels (b)-(d) all answer "by how much did it miss", on the three scales a run
can fail on: distance covered, energy, and hours of driving time.  The last two
are ECDFs because the question is "what fraction of failures came within X of
surviving" — the margin a modest buffer would have covered — and that reads
straight off a cumulative curve.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import statistics as st

from src import paths as _paths
from src.plot import paper_style as ps


_INST_RE = re.compile(
    r"^R(?P<route>short|medium|long)C(?P<cust>few|medium|many)"
    r"T(?P<tw>tight|medium|large|none)_(?P<seed>\d+)$")

# The two arms share the DET hue and separate by lightness — the same pairing
# the paper gap figure uses (paper_style.METHOD_COLOR DET / DETg), so a reader
# who has seen one figure reads the other without relearning the key.
C_RAW = ps.METHOD_COLOR["DET"]
C_GRD = ps.shade(ps.METHOD_COLOR["DET"], 0.45)

_HOS_TYPES = ("hos_cd", "hos_sd", "hos_spread")


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_det_rows() -> list[dict]:
    """One row per (INSTANCE, guard mode), keeping the latest DET run of each.

    Run ids end in a timestamp, so the lexicographic max is the newest — the
    same dedup rule compile_solutions uses.  The key includes `supervised`
    because a guarded run and an unguarded run of the same instance are two
    different EXPERIMENTS, not two attempts at one: keying on the instance
    alone would let whichever batch ran last silently replace the other in
    every rate below.
    """
    by_key: dict[tuple, str] = {}
    for p in sorted(_paths.glob_solutions("*_DET_*.json")):
        base = os.path.basename(p)
        if base.startswith("det_plan_"):        # the plan cache, not a run
            continue
        inst = base.split("_DET_")[0]
        with open(p, "r", encoding="utf-8") as fh:
            sup = bool(json.load(fh).get("supervised", False))
        by_key[(inst, sup)] = p

    rows = []
    for (inst, sup), path in by_key.items():
        m = _INST_RE.match(inst)
        if not m:                       # ad-hoc or variant stem — not the grid
            continue
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        ipath = _paths.instances(f"{inst}.json")
        if not os.path.isfile(ipath):
            continue
        with open(ipath, "r", encoding="utf-8") as fh:
            N = len(json.load(fh)["D_real"])
        met = d.get("metrics", {})
        viol = (met.get("violations") or [{}])[0]
        halt = d.get("halted_at_stop")
        done = bool(d.get("route_completed"))
        rows.append(dict(
            instance=inst, supervised=sup,
            route=m["route"], cust=m["cust"], tw=m["tw"],
            seed=int(m["seed"]), N=N,
            completed=done,
            halt=halt,
            frac=(1.0 if done or halt is None else halt / N),
            reason=d.get("halt_reason"),
            # `amount` is kWh short for a stranding, hours over for a HoS
            # breach — two different units on one field, so they are split
            # here rather than at the point of use
            shortfall=(viol.get("amount") if viol.get("type") == "stranding"
                       else None),
            hos_over=(viol.get("amount") if viol.get("type") in _HOS_TYPES
                      else None),
            planned=d.get("det_obj"),
            arrival=d.get("sim_arrival_h"),
            slip=((d.get("sim_arrival_h") - d.get("det_obj"))
                  if done and d.get("sim_arrival_h") and d.get("det_obj")
                  else None),
            interventions=met.get("n_interventions"),
            tw_misses=met.get("tw_n_misses"),
            # filled in below
            gap_pen=None,
            _rec=dict(status="OK", instance=inst, metrics=met,
                      duration_h=d.get("duration_h"),
                      sim_arrival_h=d.get("sim_arrival_h")),
        ))
    _attach_gaps(rows)
    return rows


def _attach_gaps(rows: list[dict]) -> None:
    """Gap to oracle, computed by compile_solutions' OWN annotator.

    Calling the shared function rather than re-deriving the ratio here is what
    keeps this figure and the paper gap figure reporting the same number: the
    definition converts both sides to route DURATIONS and adds the window
    penalty to each, which is not something to reimplement twice.  Only a run
    that finished has a duration, so a halted run keeps gap_pen = None.
    """
    from src.output_analysis import compile_solutions as cs
    recs = [r["_rec"] for r in rows]
    cs._annotate_gap_to_oracle(recs, _paths.solutions())
    for r, rec in zip(rows, recs):
        g = rec.get("gap_pen")
        r["gap_pen"] = (100.0 * g) if g is not None else None
        del r["_rec"]


def paired(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The instances solved BOTH ways, as (raw, guarded) aligned lists.

    The comparison is paired on purpose: where the guarded batch covers fewer
    cells than the raw one, comparing the two arms whole would confound the
    guard with the cell mix.
    """
    by = {(r["instance"], r["supervised"]): r for r in rows}
    both = sorted({i for (i, s) in by if s} & {i for (i, s) in by if not s})
    return ([by[(i, False)] for i in both], [by[(i, True)] for i in both])


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════

def cell_table(rows: list[dict]) -> list[dict]:
    """Per (route, customers, TW, guard mode) cell: failure rate and cause."""
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["route"], r["cust"], r["tw"], r["supervised"])].append(r)

    out = []
    for route in ps.ROUTE_ORDER:
        for cust in ps.CUST_ORDER:
            for tw in ps.TW_ORDER:
                for sup in (False, True):
                    v = cells.get((route, cust, tw, sup))
                    if not v:
                        continue
                    broke = [x for x in v if not x["completed"]]
                    fr = [x["frac"] for x in broke]
                    out.append(dict(
                        route=route, customers=cust, tw=tw,
                        guard=("guarded" if sup else "raw"),
                        n=len(v), broke=len(broke),
                        fail_rate=round(len(broke) / len(v), 4),
                        median_frac_done=(round(st.median(fr), 4) if fr
                                          else None),
                        stranding=sum(1 for x in broke
                                      if x["reason"] == "stranding"),
                        hos=sum(1 for x in broke
                                if x["reason"] in _HOS_TYPES),
                    ))
    return out


def _arm_summary(label: str, rs: list[dict]) -> None:
    n = len(rs)
    if not n:
        return
    broke = [r for r in rs if not r["completed"]]
    reasons = collections.Counter(r["reason"] for r in broke)
    print(f"\n  {label}   ({n} runs)")
    print(f"    broke             : {len(broke)}/{n}  ({len(broke)/n:.1%})")
    for k, v in reasons.most_common():
        print(f"        {k:<12} {v:>4}")
    fr = [r["frac"] for r in broke]
    if fr:
        print(f"    route done at halt: median {st.median(fr):.0%}")
    sf = [r["shortfall"] for r in broke if r["shortfall"] is not None]
    if sf:
        thin = sum(1 for a in sf if a < 5.0)
        print(f"    stranding short by: median {st.median(sf):.1f} kWh   "
              f"({thin}/{len(sf)} under 5 kWh)")
    ho = [r["hos_over"] for r in broke if r["hos_over"] is not None]
    if ho:
        print(f"    HoS over by       : median {st.median(ho)*60:.1f} min   "
              f"(max {max(ho)*60:.1f} min)")
    sl = [r["slip"] for r in rs if r["completed"] and r["slip"] is not None]
    if sl:
        print(f"    slip vs own plan  : mean {st.mean(sl):+.2f} h  "
              f"(max {max(sl):+.2f})")
    ni = [r["interventions"] for r in rs if r["interventions"]]
    if ni:
        print(f"    interventions     : median {st.median(ni):.0f}  "
              f"(max {max(ni)}) on {len(ni)} run(s)")
    gp = [r["gap_pen"] for r in rs if r["gap_pen"] is not None]
    if gp:
        print(f"    gap to oracle     : median {st.median(gp):.2f}%  "
              f"(n={len(gp)} of {n} — only runs that FINISHED have one)")


def print_tables(rows: list[dict], un: list[dict], gu: list[dict]) -> None:
    print("\nDET — deterministic plan under two execution rules")
    print("=" * 72)
    _arm_summary("RAW      (plan as is)",
                 [r for r in rows if not r["supervised"]])
    _arm_summary("GUARDED  (0.95 departure guard)",
                 [r for r in rows if r["supervised"]])
    print("=" * 72)
    if un:
        n = len(un)
        fu = sum(1 for r in un if not r["completed"]) / n
        fg = sum(1 for r in gu if not r["completed"]) / n
        print(f"\n  PAIRED on the {n} instance(s) solved both ways: "
              f"failure {fu:.0%} -> {fg:.0%}")


def outcome_table(rows: list[dict], un: list[dict],
                  gu: list[dict]) -> list[dict]:
    """One row per route class (plus a pooled row), both arms side by side.

    This is what panels (a)-(c) of the figure used to carry.  As numbers they
    can afford the route split the figure had to pool away, so the table is
    strictly more informative than the panels it replaces — and a failure rate
    or a median gap is a quantity a reader looks UP, not one they estimate off
    a bar.
    """
    def _stats(rs):
        if not rs:
            return dict(n=0, broke=0, fail_rate=None, frac_done=None,
                        gap_median=None, gap_n=0,
                        stranding=0, hos=0, soc_rate=None, hos_rate=None,
                        forced_mean=None, forced_max=0, forced_share=None)
        broke = [r for r in rs if not r["completed"]]
        fr = [r["frac"] for r in broke]
        gp = [r["gap_pen"] for r in rs
              if r["completed"] and r["gap_pen"] is not None]
        # Forced stops the supervisor inserted.  Averaged over EVERY run of the
        # arm, zeros included — the question is how many extra stops the guard
        # costs a route, and a route it left alone cost none.  `forced_share`
        # keeps the other reading (how often it fires at all) available.
        iv = [int(r["interventions"] or 0) for r in rs]
        return dict(
            n=len(rs), broke=len(broke),
            fail_rate=round(len(broke) / len(rs), 4),
            frac_done=(round(st.median(fr), 4) if fr else None),
            gap_median=(round(st.median(gp), 3) if gp else None),
            gap_n=len(gp),
            stranding=sum(1 for r in broke if r["reason"] == "stranding"),
            hos=sum(1 for r in broke if r["reason"] in _HOS_TYPES),
            # As a share of ALL runs, not of the failures, so the two add up
            # to fail_rate and can sit beside it without a second denominator.
            soc_rate=round(sum(1 for r in broke
                               if r["reason"] == "stranding") / len(rs), 4),
            hos_rate=round(sum(1 for r in broke
                               if r["reason"] in _HOS_TYPES) / len(rs), 4),
            forced_mean=round(st.mean(iv), 2),
            forced_max=max(iv),
            forced_share=round(sum(1 for v in iv if v) / len(iv), 4),
        )

    # Paired where both arms exist, so the two columns describe the same
    # instances; falls back to the raw arm alone when no guarded run exists.
    raw_src = un if gu else [r for r in rows if not r["supervised"]]
    out = []
    for route in list(ps.ROUTE_ORDER) + ["all"]:
        sel = (lambda rs: rs if route == "all"
               else [r for r in rs if r["route"] == route])
        r_raw, r_gu = sel(raw_src), sel(gu)
        if not r_raw and not r_gu:
            continue
        row = dict(route=route)
        for tag, rs in (("raw", r_raw), ("guarded", r_gu)):
            for k, v in _stats(rs).items():
                row[f"{tag}_{k}"] = v
        out.append(row)
    return out


def print_outcome_table(tbl: list[dict]) -> None:
    print(f"\n  {'':8}{'EV':>30}   {'EV SUPERVISED':>38}")
    print(f"  {'route':<8}{'infeas':>7}{'SOC':>6}{'HoS':>6}{'compl':>7}"
          f"{'gap':>8}"
          f"   {'infeas':>7}{'SOC':>6}{'HoS':>6}{'compl':>7}{'gap':>8}"
          f"{'forced':>8}")
    print("  " + "-" * 82)
    for r in tbl:
        cells = []
        for tag in ("raw", "guarded"):
            if not r[f"{tag}_n"]:
                cells.append(f"{'-':>7}{'-':>6}{'-':>6}{'-':>7}{'-':>8}")
                continue
            fr = f"{r[f'{tag}_fail_rate']:.0%}"
            sc = f"{r[f'{tag}_soc_rate']:.0%}"
            hs = f"{r[f'{tag}_hos_rate']:.0%}"
            dn = (f"{r[f'{tag}_frac_done']:.0%}"
                  if r[f"{tag}_frac_done"] is not None else "-")
            gp = (f"{r[f'{tag}_gap_median']:.2f}%"
                  if r[f"{tag}_gap_median"] is not None else "-")
            cells.append(f"{fr:>7}{sc:>6}{hs:>6}{dn:>7}{gp:>8}")
        # Forced stops only exist under the guard — the raw arm has no
        # supervisor to fire, so the column hangs off the guarded block only
        # rather than carrying a column of structural zeros.
        fm = r.get("guarded_forced_mean")
        fc = f"{fm:.2f}" if fm is not None else "-"
        print(f"  {r['route']:<8}{cells[0]}   {cells[1]}{fc:>8}")
    print("  " + "-" * 82)
    print("  infeas = runs halted by a violation; SOC + HoS split it by cause "
          "(shares of ALL runs);")
    print("  compl  = median % of the route reached at that halt, over the "
          "runs that were HALTED;")
    print("  gap    = median gap to oracle over the runs that FINISHED "
          "(see gap_n in the CSV);")
    print("  forced = mean supervisor-forced stops per run, zeros included "
          "(share/max in the CSV)")


def write_outcome_csv(tbl: list[dict]) -> str:
    path = _paths.data_output("vss_det_outcome_table.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tbl[0].keys()))
        w.writeheader()
        w.writerows(tbl)
    return path


def write_outcome_tex(tbl: list[dict]) -> str:
    """LaTeX twin of the console table.

    Written into tex/tables/, which holds GENERATED tables only, never next to
    the manuscript prose.
    """
    lbl = {"short": "Short", "medium": "Medium", "long": "Long",
           "all": "All routes"}
    eol = r" \\"
    # The units live in the column heads, so the cells carry bare numbers: a
    # per-cent sign repeated 18 times is noise in a table this dense.
    # Per-route n, not the pooled one: the caption states the size of a route
    # CLASS, and the "all" row would overstate it by the number of classes.
    n_runs = max((r["raw_n"] for r in tbl if r["route"] != "all"), default=0)

    lines = [
        r"% GENERATED by src/plot/det_figures.py -- do not edit by hand",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Execution of the EV plan, with and without the safeguard. "
        r"\emph{SOC} and \emph{HoS} split the infeasibility rate by the "
        r"constraint that was violated. \emph{Route completed} shows how much "
        r"of the route was done before an infeasibility occurred, over the "
        r"runs that were halted; \emph{Gap} is the gap to the hindsight "
        r"oracle, over the runs that finished. "
        + f"{n_runs if n_runs else 0}"
        + r" instances per route class.}",
        r"\label{tab:vss_outcome}",
    ]

    # House style, as used by feasibility.tex and additional_sensitivity.tex:
    # bold column heads carrying their own unit, c-aligned numeric columns,
    # every label cell filled in, "--" for a missing value, booktabs rules.
    # No group-heading rows and no \multirow -- the other generated tables use
    # neither, and repeating the route label costs one word per row while
    # keeping the left edge straight and the rows independently readable.
    metrics = [
        (r"Infeasible (\%)",      "fail_rate",  0),
        (r"SOC (\%)",             "soc_rate",   0),
        (r"HoS (\%)",             "hos_rate",   0),
        (r"Route completed (\%)", "frac_done",  0),
        (r"Gap (\%)",             "gap_median", 1),
    ]
    head = [r"\textbf{Route}", r"\textbf{Method}"]
    head += [r"\textbf{" + n + "}" for n, _k, _d in metrics]
    lines += [
        r"\begin{tabular}{ll" + "c" * len(metrics) + "}",
        r"\toprule",
        " & ".join(head) + eol,
        r"\midrule",
    ]
    for i, r in enumerate(tbl):
        # the pooled block is a summary of the rows above it, so it is set off
        # by a rule the way feasibility.tex sets off its own totals
        if r["route"] == "all" and i:
            lines.append(r"\midrule")
        label = lbl.get(r["route"], r["route"])
        for tag in ("raw", "guarded"):
            cells = []
            for _name, key, dec in metrics:
                v = r[f"{tag}_{key}"] if r[f"{tag}_n"] else None
                if v is None:
                    cells.append("--")
                elif dec:
                    cells.append(f"{v:.{dec}f}")
                else:
                    cells.append(f"{v * 100:.0f}")
            mode = "EV" if tag == "raw" else "EV sup."
            lines.append(f"{label} & {mode} & " + " & ".join(cells) + eol)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = _paths.tex_tables("vss_det_outcome.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path



def write_csv(cells: list[dict]) -> str:
    path = _paths.data_output("vss_det_summary.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _ecdf(ax, vals, color, label, scale=1.0) -> bool:
    """Step ECDF; returns True if anything was drawn."""
    v = sorted(x * scale for x in vals if x is not None)
    if not v:
        return False
    y = [(i + 1) / len(v) for i in range(len(v))]
    ax.step(v, y, where="post", color=color, linewidth=1.6, label=label)
    return True


def _style(ax, pct_y: bool = True) -> None:
    ax.grid(color=ps.GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if pct_y:
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, .25, .5, .75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100"])


def make_figure(rows: list[dict], un: list[dict], gu: list[dict]) -> list[str]:
    """The two magnitude panels: by how much a broken plan missed.

    How OFTEN it broke, how FAR it got and the gap to oracle are single
    numbers per cell, so they live in the table (outcome_table) where they can
    keep the per-route split.  What is left here is the part a table cannot
    carry: two full distributions, on the two scales a run fails on.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps.apply_rc()
    raw_all = [r for r in rows if not r["supervised"]]
    grd_all = [r for r in rows if r["supervised"]]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 2.5))

    # Panel titles are omitted: each x label already names its own panel, and
    # the sample sizes sit in the corner annotation instead, so nothing is
    # lost by dropping a title row that only repeated the axis.
    def _n_note(ax, counts):
        ax.text(0.97, 0.06, f"n = {counts[0]} / {counts[1]}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color=ps.INK_MUTED)

    # ── left: stranding, energy missing ──────────────────────────────────────
    n_l = [sum(1 for r in g if r["shortfall"] is not None)
           for g in (raw_all, grd_all)]
    drew = _ecdf(axL, [r["shortfall"] for r in raw_all], C_RAW, "EV")
    drew = _ecdf(axL, [r["shortfall"] for r in grd_all], C_GRD,
                 "EV supervised") or drew
    if drew:
        axL.axvline(5.0, color=ps.INK_MUTED, linewidth=0.7,
                    linestyle=(0, (3, 2)))
        axL.text(5.6, 0.04, "5 kWh", fontsize=6.5, color=ps.INK_MUTED)
    _style(axL)
    axL.set_xlabel("Energy missing on the stranding leg (kWh)", fontsize=7.5)
    axL.set_ylabel("Strandings at or below (%)", fontsize=7.5)
    _n_note(axL, n_l)

    # ── right: HoS, minutes over the legal limit ─────────────────────────────
    n_r = [sum(1 for r in g if r["hos_over"] is not None)
           for g in (raw_all, grd_all)]
    drew = _ecdf(axR, [r["hos_over"] for r in raw_all], C_RAW, "EV",
                 scale=60.0)
    drew = _ecdf(axR, [r["hos_over"] for r in grd_all], C_GRD,
                 "EV supervised", scale=60.0) or drew
    _style(axR)
    if not drew:
        axR.text(0.5, 0.5, "no HoS breaches", transform=axR.transAxes,
                 ha="center", va="center", fontsize=7.5, style="italic",
                 color=ps.INK_MUTED)
    axR.set_xlabel("Driving time over the legal limit (min)", fontsize=7.5)
    axR.set_ylabel("HoS breaches at or below (%)", fontsize=7.5)
    _n_note(axR, n_r)

    for ax in (axL, axR):
        ax.tick_params(labelsize=7)

    handles, labs = axL.get_legend_handles_labels()
    if handles:
        fig.legend(handles=handles, labels=labs, loc="upper center",
                   ncol=len(handles), frameon=False, fontsize=7,
                   handlelength=1.4, bbox_to_anchor=(0.5, 1.06))

    fig.tight_layout()
    out = []
    for ext in ("pdf", "png"):
        p = _paths.figure_out(f"vss_det_outcome.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", action="store_true",
                    help="print the tables only, skip the figure")
    args = ap.parse_args()

    rows = load_det_rows()
    if not rows:
        raise SystemExit("no DET runs found in solutions/VSS/")
    un, gu = paired(rows)
    print_tables(rows, un, gu)
    tbl = outcome_table(rows, un, gu)
    print_outcome_table(tbl)
    print(f"\n  wrote {write_outcome_csv(tbl)}")
    print(f"  wrote {write_outcome_tex(tbl)}")
    print(f"  wrote {write_csv(cell_table(rows))}")
    if not args.table:
        for p in make_figure(rows, un, gu):
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
