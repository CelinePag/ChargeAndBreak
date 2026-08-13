"""
additional_figures.py — REAL tables & figures for the additional analyses
(paper Sections 8.3-8.5), built from whatever runs currently exist.

Unlike experiments/mock_section_figures.py (random placeholder data), this
script reads solutions/ (+ oracle caches, results_vss/) and renders the
paper artefacts with the data available NOW; cells or panels whose runs have
not finished yet are shown explicitly as "pending", mirroring the
paper_figures.py convention of drawing the full grid with empty slots.

Outputs (figures -> figures/, tables -> tex/tables/, csv -> data_output/)
  figures/additional_diesel_gap.png|pdf      §8.4 figure
  data_output/additional_diesel_stats.csv    §8.4 per-class detail
  tex/tables/additional_diesel.tex           §8.4 table
  figures/additional_sens_effects.png|pdf    §8.3 one-at-a-time
  data_output/additional_sens_stats.csv
  tex/tables/additional_sensitivity.tex
  figures/additional_la_config.png|pdf       §8.3 look-ahead configuration
  figures/additional_la_frontier.png|pdf     §8.3 cost/quality frontier
  tex/tables/additional_la.tex               (both read
  additional_analysis.py's data_output/additional_la_stats.csv)
  tex/tables/additional_vss.tex              §8.5 VSS/EVPI (skeleton until
  data_output/additional_vss_stats.csv             results_vss/ fills up)

§8.3 reports three methods per axis: greedy and LA (online policies) and the
oracle (hindsight optimum).  Each is PAIRED per instance — base and variant
must both exist and both be feasible — so a method whose variant runs have not
landed yet shows "--" in the table and no bar in the figure, rather than a
misleading zero.

Usage
  python -m src.plot.additional_figures                 # all sections
  python -m src.plot.additional_figures --section diesel|sensitivity|vss
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.settings import T_START, BETA_TW

# Shared palette + chrome (see paper_style.py): colour follows the entity, so
# Greedy is the same blue here as in the base-case figures.
from src.plot import paper_style as ps
from src import paths as _paths

INK, MUT, GRID = ps.INK_PRIMARY, ps.INK_MUTED, ps.GRID
BLUE  = ps.METHOD_COLOR["greedy"]
VERM  = ps.METHOD_COLOR["RO"]
ORAN  = ps.METHOD_COLOR["ROBU"]
GREEN = ps.METHOD_COLOR["LA"]
PURP  = ps.METHOD_COLOR["2SP"]
ps.apply_rc()

COMBOS = [("short", "few"), ("short", "many"), ("medium", "few"),
          ("medium", "many")]
# §8.4 only.  Long routes are excluded from the sweeps (intractable under the
# full protocol) but the diesel comparison needs just Greedy and the oracle,
# and the diesel oracle is cheap there because it carries no energy dimension
# — the hardest long instance closes to 0.4% in under a minute.
DIESEL_COMBOS = COMBOS + [("long", "few"), ("long", "many")]
DIESEL_ROUTES = ("short", "medium", "long")
TWS    = ["tight", "medium", "large", "none"]
SEEDS  = range(1, 11)

_RTAG = {"short": "Rshort", "medium": "Rmedium", "long": "Rlong"}
_CTAG = {"few": "Cfew", "medium": "Cmedium", "many": "Cmany"}


def _stem(route, cust, tw, seed) -> str:
    return f"{_RTAG[route]}{_CTAG[cust]}T{tw}_{seed}"


def _latest(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def _load(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _policy(stem: str, alg: str, tag: str | None = None) -> dict | None:
    """Latest simulated-policy solution for a (possibly tagged) instance.

    ``alg`` is the run-id algorithm token (GREEDY, LA, 2SP, RO, ROBU).  Accepts
    both the '__tag' stem (orchestrator batch) and the runner-normalised
    '_tag'.
    """
    pats = ([_paths.solutions(f"{stem}_{alg}_*.json")] if tag is None else
            [_paths.solutions(f"{stem}__{tag}_{alg}_*.json"),
             _paths.solutions(f"{stem}_{tag}_{alg}_*.json")])
    for p in pats:
        f = _latest(p)
        if f:
            d = _load(f)
            if d and d.get("duration_h") is not None:
                infeas = bool((d.get("metrics") or {}).get("run_infeasible"))
                return dict(duration=float(d["duration_h"]), infeasible=infeas)
    return None


def _greedy(stem: str, tag: str | None = None) -> dict | None:
    return _policy(stem, "GREEDY", tag)


def _la(stem: str, tag: str | None = None) -> dict | None:
    return _policy(stem, "LA", tag)


def _oracle(stem: str, tag: str | None = None) -> dict | None:
    """Oracle cache -> duration (h), total/coupled charging time (h)."""
    names = ([_paths.solutions(f"oracle_{stem}.json")] if tag is None else
             [_paths.solutions(f"oracle_{stem}__{tag}.json"),
              _paths.solutions(f"oracle_{stem}_{tag}.json")])
    for n in names:
        d = _load(n)
        if not (d and d.get("feasible")):
            continue
        sol = d.get("sol") or []
        if sol:
            ta_N = float(sol[-1]["ta"])
            tauc = sum(float(s.get("tauc") or 0.0) for s in sol)
            g    = sum(float(s.get("g")    or 0.0) for s in sol)
            # delta = out-of-window service starts (early OR late, §3.1).  The
            # objective is ta[N] + BETA_TW*sum(delta), so a schedule can buy
            # makespan by missing a window; reporting duration alone would hide
            # that trade, hence both are carried.
            delta = sum(int(round(float(s.get("delta") or 0.0))) for s in sol)
            return dict(duration=ta_N - T_START, tauc=tauc, g=g, delta=delta,
                        gap=float(d.get("gap") or 0.0))
        # cache recovered from a run log (see recover_variant_oracles.py):
        # the objective survives, the schedule does not — usable for duration
        # deltas, not for per-stop quantities like the coupling fraction.
        if d.get("obj") is not None:
            return dict(duration=float(d["obj"]) - T_START,
                        tauc=None, g=None, delta=None,
                        gap=float(d.get("gap") or 0.0))
    return None


_INST_CACHE: dict[str, dict] = {}


def _instance(stem: str) -> dict:
    """Base instance data (geometry + overhead parameters), memoised.

    Diesel variants are verbatim copies, so the base file is the right source
    for both worlds; the diesel-side transform is applied by the caller.
    """
    if stem not in _INST_CACHE:
        _INST_CACHE[stem] = (_load(_paths.instances(f"{stem}.json")) or {}).get(
            "instance") or {}
    return _INST_CACHE[stem]


def _int_keyed(d) -> dict[int, float]:
    """JSON stringifies int keys on the way out; restore them."""
    return {int(k): float(v) for k, v in (d or {}).items()}


# Dwell components, in the order they are reported.  These are the terms of
# the model's own departure equations (MILP td_K / td_C / td_L), so they sum
# with driving to exactly the makespan — verified per instance below.
_DWELL_ROWS = ("charging", "queue", "manoeuvre", "reposition",
               "break", "rest", "service", "wait")


def _oracle_dwell(stem: str, tag: str | None = None) -> dict | None:
    """Per-component dwell totals (h) of the hindsight-optimal schedule.

    Because driving is identical across the EV/diesel pair by construction
    (verbatim copy, same D_real), differencing these components accounts for
    the whole EV-vs-diesel makespan gap with no residual.
    """
    names = ([_paths.solutions(f"oracle_{stem}.json")] if tag is None else
             [_paths.solutions(f"oracle_{stem}__{tag}.json"),
              _paths.solutions(f"oracle_{stem}_{tag}.json")])
    sol = None
    for n in names:
        d = _load(n)
        if d and d.get("feasible") and d.get("sol"):
            sol = d["sol"]
            break
    inst = _instance(stem)
    if sol is None or not inst:
        return None

    diesel = (tag == "diesel")
    M_stop = _int_keyed(inst.get("M_stop"))
    # _apply_diesel_mode keeps M_stop (the access manoeuvre is owed by any
    # vehicle that pulls off to break) but drops the charger queue and the
    # repositioning move off the charging bay, which have no diesel analogue.
    M_seq  = {} if diesel else _int_keyed(inst.get("M_seq"))
    M_lay  = _int_keyed(inst.get("M_lay"))
    S      = _int_keyed(inst.get("S"))
    L_set  = {int(i) for i in (inst.get("L") or [])}

    out = {k: 0.0 for k in _DWELL_ROWS}
    for s in sol:
        i   = int(s["i"])
        brk = any(float(s.get(k) or 0.0) > 0.5 for k in ("b15", "b30", "b45"))
        rst = any(float(s.get(k) or 0.0) > 0.5 for k in ("rho1", "rho2"))
        out["break"] += float(s.get("taub") or 0.0)
        out["rest"]  += float(s.get("taur") or 0.0)
        if s.get("is_K"):
            y = float(s.get("y") or 0.0)
            out["charging"]   += float(s.get("tauc") or 0.0)
            out["queue"]      += float(s.get("tauq") or 0.0)   # already Q*y
            out["reposition"] += float(s.get("sigma") or 0.0) * M_seq.get(i, 0.0)
            if y > 0.5 or brk or rst:                          # MILP v[i]
                out["manoeuvre"] += M_stop.get(i, 0.0)
        else:
            out["wait"] += float(s.get("wait") or 0.0)
            if s.get("is_C"):
                out["service"] += S.get(i, 0.0)
            elif i in L_set and (brk or rst):
                out["manoeuvre"] += M_lay.get(i, 0.0)

    out["_drive"]    = sum(float(s.get("D_nom") or 0.0) for s in sol)
    out["_duration"] = float(sol[-1]["ta"]) - T_START
    return out


def _fmt(x, spec=".1f", dash="--"):
    return format(x, spec) if x is not None and np.isfinite(x) else dash


def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(_paths.figures(f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Figure    : figures/{name}.png|pdf")


def _write_csv(name, header, rows):
    path = _paths.data_output(name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Stats CSV : {path}")


def _write_tex(name, text):
    _paths.ensure_dirs()
    path = _paths.tex_tables(name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  LaTeX     : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# §8.4 — DIESEL
# ══════════════════════════════════════════════════════════════════════════════

def section_diesel():
    print("== Sec 8.4 diesel ==")
    # Scope from the diesel runs on disk, not from DIESEL_COMBOS/TWS/SEEDS:
    # the batch is routinely launched wider than the planning constants, and
    # every consumer below (including the coverage denominator) has to agree
    # with what was actually paired or "have/want" becomes fiction.
    scope    = _discover_scope(["diesel"])
    combos_d = scope["combos"] or DIESEL_COMBOS
    tws_d    = scope["tws"]    or TWS
    seeds_d  = scope["seeds"]  or list(SEEDS)
    print(f"  scope     : combos {','.join(f'R{r}C{c}' for r, c in combos_d)}")
    print(f"              tw {','.join(tws_d)}  "
          f"seeds {min(seeds_d)}-{max(seeds_d)} (n={len(seeds_d)})")

    per_class: dict[str, dict[str, list]] = {}
    detail = []
    for route, cust in combos_d:
        for tw in tws_d:
            for seed in seeds_d:
                st       = _stem(route, cust, tw, seed)
                ev_o     = _oracle(st)
                di_o     = _oracle(st, "diesel")
                ev_g     = _greedy(st)
                di_g     = _greedy(st, "diesel")

                fuel  = _refuel_h(route)         # post-hoc diesel fuel stop(s)
                dur_d = (di_o["duration"] + fuel) if di_o else None

                pen_o = pen_g = coup = None
                # Absolute counterpart of the percentage: the same difference
                # in hours, so the figure states the penalty in the unit the
                # operator actually plans in.
                dt_o = dt_g = None
                if ev_o and dur_d and dur_d > 0:
                    pen_o = 100 * (ev_o["duration"] / dur_d - 1)
                    dt_o  = ev_o["duration"] - dur_d
                    # tauc/g are None for a log-recovered cache (no schedule)
                    if ev_o["tauc"] is not None and ev_o["tauc"] > 1e-6:
                        coup = 100 * ev_o["g"] / ev_o["tauc"]
                if (ev_g and di_g and not ev_g["infeasible"]
                        and not di_g["infeasible"] and di_g["duration"] > 0):
                    pen_g = 100 * (ev_g["duration"] / (di_g["duration"] + fuel) - 1)
                    dt_g  = ev_g["duration"] - (di_g["duration"] + fuel)

                # The EV oracle is only an incumbent where the solve hit its
                # wall budget (long routes stall on the DUAL bound), so the
                # penalty there is an upper bound on the true optimal one.
                ev_gap = 100 * ev_o["gap"] if ev_o else None

                detail.append([route, cust, tw, seed,
                               _fmt(dur_d, ".3f", ""),
                               _fmt(ev_o and ev_o["duration"], ".3f", ""),
                               _fmt(pen_o, ".2f", ""), _fmt(pen_g, ".2f", ""),
                               _fmt(coup, ".1f", ""),
                               _fmt(fuel, ".3f", ""), _fmt(ev_gap, ".2f", "")])
                d = per_class.setdefault(route, dict(pen_o=[], pen_g=[],
                                                     dt_o=[], dt_g=[],
                                                     coup=[],
                                                     dur_d=[], dur_e=[],
                                                     ev_gap=[]))
                d["pen_o"].append(pen_o); d["pen_g"].append(pen_g)
                d["dt_o"].append(dt_o);   d["dt_g"].append(dt_g)
                d["coup"].append(coup)
                d["dur_d"].append(dur_d)
                d["dur_e"].append(ev_o and ev_o["duration"])
                d["ev_gap"].append(ev_gap)

    _write_csv("additional_diesel_stats.csv",
               ["route", "cust", "tw", "seed", "diesel_oracle_h",
                "ev_oracle_h", "pen_oracle_%", "pen_greedy_%",
                "coupling_%", "refuel_h",
                "ev_oracle_gap_%"], detail)

    # ── Oracle coverage ──────────────────────────────────────────────────────
    # A class is reported from whatever oracle solves exist, complete or not.
    # This is a partial average where coverage is incomplete, and on long
    # routes the instances that finish are the easy tail (both oracles stall on
    # the dual bound there, on the rest-packing structure rather than the
    # energy dimension), so an incomplete class reads slightly optimistic.
    # Rather than withhold it, every consumer below states the coverage: the
    # figure annotates "n = have/want" and the table caption names the
    # incomplete classes.  Per-instance values remain in the CSV above.
    coverage = {}
    for r, d in per_class.items():
        have = sum(1 for v in d["pen_o"] if v is not None)
        want = len(tws_d) * len(seeds_d) * sum(1 for rr, _ in combos_d
                                               if rr == r)
        coverage[r] = (have, want)
        if have < want:
            print(f"  Oracle coverage {r}: {have}/{want} — partial average, "
                  f"reported with its n")
    oracle_ok = [r for r, (have, _w) in coverage.items() if have]

    # ── figure: greedy vs oracle penalty, per route class ────────────────────
    # Two units on one mark: the percentage sits above the bar (it is what the
    # axis measures) and its absolute-hours counterpart sits inside the bar in
    # reversed-out type, so the two never read as a single stacked number.
    routes = [r for r in DIESEL_ROUTES if r in per_class]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    w, x = 0.30, np.arange(len(routes), dtype=float)
    # Colour follows the entity (paper_style): the same blue as Greedy and the
    # same neutral grey as the oracle everywhere else in the paper.
    series = [("Greedy policy", "pen_g", "dt_g", BLUE),
              ("Hindsight optimum", "pen_o", "dt_o", ps.METHOD_COLOR["oracle"])]
    for k, (lbl, key, dkey, col) in enumerate(series):
        vals = [_mean(per_class[r][key]) for r in routes]
        hrs  = [_mean(per_class[r][dkey]) for r in routes]
        pos  = x + (k - 0.5) * (w + 0.035)      # hairline gap between the pair
        # nan, not 0: a suppressed class must leave a gap, not draw a bar at
        # zero that reads as "no penalty".
        ax.bar(pos, [np.nan if v is None else v for v in vals], w, color=col,
               edgecolor="white", linewidth=0.5, zorder=3, label=lbl)
        for p, v, h in zip(pos, vals, hrs):
            if v is None:
                continue
            ax.annotate(f"{v:.1f}%", (p, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom",
                        fontsize=7.5, color=INK)
            if h is not None:
                ax.annotate(f"{h:+.1f} h", (p, v), textcoords="offset points",
                            xytext=(0, -4), ha="center", va="top",
                            fontsize=6.5, color="white")
    for xi, r in enumerate(routes):
        c = _mean(per_class[r]["coup"])
        n = sum(1 for v in per_class[r]["pen_o"] if v is not None)
        want = coverage[r][1]
        if n:
            # "n = 75/80" wherever the class is a partial average, so the bar
            # is never read as resting on the full sample.
            shown = f"{n}" if n >= want else f"{n}/{want}"
            note = f"{_fmt(c, '.0f')}% coupled  ·  n = {shown}"
        else:   # no oracle at all for this class — label the Greedy sample
            n = sum(1 for v in per_class[r]["pen_g"] if v is not None)
            note = f"greedy only  ·  n = {n}"
        ax.text(xi, -0.115, note, ha="center", va="top", fontsize=6.5,
                color=MUT, transform=ax.get_xaxis_transform())
    ax.set_xticks(x, [ps.ROUTE_LBL[r] for r in routes])
    ax.set_xlim(-0.6, len(routes) - 0.4)
    ax.set_ylabel("Route duration vs. diesel (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ps.BASELINE)
        ax.spines[side].set_linewidth(0.7)
    # Headroom so the legend row clears the tallest bar's value label.
    _top = max((v for s in series for v in
                (_mean(per_class[r][s[1]]) for r in routes)
                if v is not None), default=1.0)
    ax.set_ylim(0, _top * 1.38)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left",
              handlelength=1.0, handleheight=1.0, handletextpad=0.5,
              columnspacing=1.4, borderpad=0.0, borderaxespad=0.2)
    ax.set_title("Electrification penalty: myopic vs optimized schedules",
                 loc="left", color=INK, pad=6)
    _save(fig, "additional_diesel_gap")

    # ── LaTeX table ──────────────────────────────────────────────────────────
    # Non-finite entries are dropped rather than propagated: some long-route
    # oracle caches record no bound at all, and a bare max() over them returned
    # NaN, which _fmt rendered as "--" — reading as "no certification data"
    # when in fact 69 of 80 instances had one.  The dropped count is reported
    # in the caption instead, since a max over the certified subset says
    # nothing about the instances whose bound is unknown.
    def _maxgap(r):
        v = [x for x in per_class[r]["ev_gap"]
             if x is not None and np.isfinite(x)]
        return max(v) if v else None

    def _nogap(r):
        return sum(1 for x in per_class[r]["ev_gap"]
                   if x is not None and not np.isfinite(x))

    # Name any class whose oracle columns rest on a partial sample, in the
    # caption rather than in a footnote, so the qualification travels with the
    # table wherever it is reproduced.
    partial = [f"{r} ({h}/{w})" for r in routes
               for h, w in [coverage[r]] if 0 < h < w]
    cov_note = (r"  Oracle columns for " + ", ".join(partial) +
                r" average the solved subset rather than the full sample; on "
                r"long routes the instances that solve are the easier ones, "
                r"so those figures read slightly optimistic."
                ) if partial else ""
    nb = [f"{r} ({_nogap(r)})" for r in routes if _nogap(r)]
    if nb:
        cov_note += (r"  The certification shown is the worst gap among the "
                     r"instances that recorded one; no bound was recorded "
                     r"for " + ", ".join(nb) + r".")

    lines = [
        r"\begin{table}[ht]\centering",
        r"\caption{EV vs diesel route duration (\%) on the same instances and "
        r"realizations.  Coupling = share of charging time credited inside a "
        r"mandatory break ($\Sigma g_i/\Sigma\tau^c_i$, hindsight optimum). "
        r"``EV cert.'' is the worst remaining MIP gap on the EV oracle: "
        r"where it exceeds the solver tolerance the EV schedule is an "
        r"incumbent, not a proven optimum, so the penalty is an upper bound "
        r"on the true optimal one." + cov_note + r"}",
        r"\label{tab:diesel}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Route & Diesel (h) & EV (h) & Greedy (\%) & "
        r"Oracle (\%) & Coupling (\%) & EV cert. (\%) & $n$ \\",
        r"\hline",
    ]
    for r in routes:
        d = per_class[r]
        have, want = coverage[r]
        lines.append(
            f"{r.capitalize()} & {_fmt(_mean(d['dur_d']))} & "
            f"{_fmt(_mean(d['dur_e']))} & "
            f"{_fmt(_mean(d['pen_g']))} & {_fmt(_mean(d['pen_o']))} & "
            f"{_fmt(_mean(d['coup']), '.0f')} & "
            f"{_fmt(_maxgap(r), '.1f')} & "
            f"{have}/{want} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_diesel.tex", "\n".join(lines))

    # Both are oracle-derived, so they follow the coverage guard above.
    ok = [r for r in routes if r in oracle_ok]
    _diesel_decomposition(ok, scope)
    _refuel_sensitivity(ok, scope)
    _diesel_by_tw(routes, ok, scope)


# Window classes ordered tight -> none, i.e. loosening the constraint, so the
# trend reads left to right (paper_figures.py uses the same order).
_TW_ORDER = ["tight", "medium", "large", "none"]


def _diesel_by_tw(routes, oracle_ok, scope) -> None:
    """Split the penalty by time-window class.

    Answers whether the EV/diesel gap is a real makespan difference or an
    artefact of the window penalty: the objective is
    ta[N] + BETA_TW*sum(delta), so a schedule can trade makespan against a
    missed window.  Both the duration-based and the objective-based penalty
    are reported, along with the delta counts that separate them.
    """
    per: dict[tuple, dict] = {}
    # Canonical tight -> none order, restricted to the classes actually run, so
    # an unswept window class does not draw an empty column.
    tw_order = [t for t in _TW_ORDER if t in set(scope["tws"] or TWS)]
    for route, cust in (scope["combos"] or DIESEL_COMBOS):
        for tw in tw_order:
            for seed in (scope["seeds"] or SEEDS):
                st  = _stem(route, cust, tw, seed)
                key = (route, tw)
                d   = per.setdefault(key, dict(pen_o=[], pen_g=[], obj_o=[],
                                               d_ev=[], d_di=[], brk_di=[]))
                fuel = _refuel_h(route)
                ev_g, di_g = _greedy(st), _greedy(st, "diesel")
                if (ev_g and di_g and not ev_g["infeasible"]
                        and not di_g["infeasible"] and di_g["duration"] > 0):
                    d["pen_g"].append(
                        100 * (ev_g["duration"] / (di_g["duration"] + fuel) - 1))
                if route not in oracle_ok:
                    continue
                ev, di = _oracle(st), _oracle(st, "diesel")
                dw_ev, dw_di = _oracle_dwell(st), _oracle_dwell(st, "diesel")
                if not (ev and di and di["duration"] > 0):
                    continue
                base = di["duration"] + fuel
                d["pen_o"].append(100 * (ev["duration"] / base - 1))
                if ev.get("delta") is not None and di.get("delta") is not None:
                    # Objective-based: charge each side for the windows it
                    # misses, at the model's own BETA_TW.
                    d["obj_o"].append(
                        100 * ((ev["duration"] + BETA_TW * ev["delta"])
                               / (base + BETA_TW * di["delta"]) - 1))
                    d["d_ev"].append(ev["delta"]); d["d_di"].append(di["delta"])
                if dw_di:
                    d["brk_di"].append(dw_di["break"])

    routes = [r for r in routes if any((r, tw) in per for tw in tw_order)]
    if not routes:
        return

    # ── small multiples: one panel per route class ───────────────────────────
    fig, axes = plt.subplots(1, len(routes), figsize=(2.3 * len(routes) + 0.9,
                                                      2.9), sharey=True)
    axes = np.atleast_1d(axes)
    x, w = np.arange(len(tw_order)), 0.34
    series = [("Greedy", "pen_g", BLUE), ("Oracle", "pen_o", INK)]
    top = 0.0
    for ax, route in zip(axes, routes):
        for k, (lbl, key, col) in enumerate(series):
            vals = [_mean(per[(route, tw)][key]) if (route, tw) in per else None
                    for tw in tw_order]
            top  = max([top] + [v for v in vals if v is not None])
            ax.bar(x + (k - 0.5) * w,
                   [np.nan if v is None else v for v in vals], w,
                   color=col, edgecolor="white", linewidth=0.8,
                   label=lbl if ax is axes[0] else None)
        ax.set_xticks(x, [t.capitalize() for t in tw_order], fontsize=7)
        ax.set_title(f"{route.capitalize()} route", loc="left", fontsize=8)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Route duration vs diesel (%)")
    axes[0].set_ylim(0, top * 1.28)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Electrification penalty by time-window class",
                 x=0.02, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "additional_diesel_tw")

    # ── table: duration vs objective, and the delta counts behind it ─────────
    rows = []
    for route in routes:
        for tw in tw_order:
            d = per.get((route, tw))
            if not d:
                continue
            rows.append([route, tw, len(d["pen_o"]), len(d["pen_g"]),
                         _fmt(_mean(d["pen_g"]), ".2f", ""),
                         _fmt(_mean(d["pen_o"]), ".2f", ""),
                         _fmt(_mean(d["obj_o"]), ".2f", ""),
                         _fmt(_mean(d["d_ev"]), ".2f", ""),
                         _fmt(_mean(d["d_di"]), ".2f", ""),
                         _fmt(_mean(d["brk_di"]), ".2f", "")])
    _write_csv("additional_diesel_tw.csv",
               ["route", "tw", "n_oracle", "n_greedy", "pen_greedy_%",
                "pen_oracle_%", "pen_oracle_objective_%", "delta_ev",
                "delta_diesel", "diesel_break_h"], rows)

    tex = [
        r"\begin{table}[ht]\centering",
        r"\caption{Electrification penalty by time-window class.  "
        r"``Duration'' compares makespans; ``objective'' additionally charges "
        r"each vehicle $\beta_{TW}$ per out-of-window service start, so the "
        r"pair separates a real makespan difference from one bought by "
        r"missing a window.  $\delta$ columns give the mean number of missed "
        r"windows.  The last column is the diesel's standalone break time, "
        r"which is what drives the trend: the electric truck takes none in "
        r"any cell, charging through every break it needs.}",
        r"\label{tab:diesel-tw}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\hline",
        r"Route & Window & $n_G$ & Greedy (\%) & $n_O$ & Duration (\%) & "
        r"Objective (\%) & $\delta_{EV}$ & $\delta_{diesel}$ \\",
        r"\hline",
    ]
    for r in rows:
        # n differs by column: Greedy drops instances it cannot schedule
        # feasibly, the oracle drops classes with incomplete coverage.
        tex.append(f"{r[0].capitalize()} & {r[1].capitalize()} & "
                   f"{r[3]} & {r[4] or '--'} & "
                   f"{r[2] or '--'} & {r[5] or '--'} & {r[6] or '--'} & "
                   f"{r[7] or '--'} & {r[8] or '--'} \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_diesel_tw.tex", "\n".join(tex))


# ── Refuelling: reported post hoc, not modelled ──────────────────────────────
# The diesel runs carry no fuel constraint.  Unlike charging, refuelling has no
# binding location choice (any truck stop serves) and happens at most once on
# these corridors, so it is a constant rather than a decision — see
# runner_dispatch._apply_diesel_mode.
#
# Tank 500-1000 L on long-haul specs; consumption 32.6 L/100 km for a typical
# EU 40 t tractor-semitrailer over the regulatory long-haul cycle (ICCT),
# ~25 L/100 km for a modern tractor on real customer routes.  90% of the tank
# is treated as usable (carriers do not run to empty).
_TANK_SPECS = [(900, 25.0), (900, 33.0), (600, 25.0), (600, 33.0)]
_TANK_USABLE = 0.90
# One fuel stop = access manoeuvre (M_STOP_H) + short queue + pumping.  A
# 500 L fill at the 180 L/min truck-lane standard is ~3 min of pumping; 7 min
# covers nozzle handling and payment.
_REFUEL_EVENT_H = (10.0 + 2.0 + 7.0) / 60

# Fuel stops credited to the diesel in the headline tables.  Stated as an
# assumption rather than derived, because the derived count depends on a tank
# specification that the literature does not pin down (see _TANK_SPECS).
#
# The counts below are internally consistent under ONE specification, the
# 600 L tank at 33 L/100 km: it is the only one that puts the medium class at
# a full stop (0.95), and it puts the long class at 1.77, i.e. two.  A larger
# tank would move the pair down together (medium 0, long 1) rather than
# changing only one of them, so mixing medium=1 with long=1 would not
# correspond to any single truck.  Crediting a stop LOWERS the reported
# electrification penalty, so this is the less conservative reading;
# _refuel_sensitivity() reports the zero-stop bound and every spec alongside.
_REFUEL_STOPS = {"short": 0, "medium": 1, "long": 2}


def _refuel_h(route: str) -> float:
    """Post-hoc diesel refuelling time (h) credited to a route class."""
    return _REFUEL_STOPS.get(route, 0) * _REFUEL_EVENT_H


def _refuel_sensitivity(routes, scope) -> None:
    """Check the assumed fuel-stop count against what tank specs imply.

    The headline tables credit _REFUEL_STOPS per route class.  This reports
    what each tank specification would derive instead, and what the penalty
    would be under it, so the assumption is auditable rather than asserted.
    """
    rows, note = [], {}
    for route in routes:
        km, pen = [], []
        for r, cust in (scope["combos"] or DIESEL_COMBOS):
            if r != route:
                continue
            for tw in (scope["tws"] or TWS):
                for seed in (scope["seeds"] or SEEDS):
                    st = _stem(route, cust, tw, seed)
                    inst = _instance(st)
                    ev, di = _oracle(st), _oracle(st, "diesel")
                    if not (inst.get("km") and ev and di and di["duration"] > 0):
                        continue
                    km.append(sum(_int_keyed(inst["km"]).values()))
                    pen.append((ev["duration"], di["duration"]))
        if not km:
            continue
        note[route] = (len(km), min(km), max(km))

        def _pen(stops) -> float | None:
            return _mean([100 * (e / (d + s * _REFUEL_EVENT_H) - 1)
                          for (e, d), s in zip(pen, stops)])

        assumed = _REFUEL_STOPS.get(route, 0)
        base    = _pen([0] * len(km))
        rows.append([route, "assumed (headline)", "", "",
                     f"{assumed:.2f}", "", _fmt(_pen([assumed] * len(km)), ".2f")])
        rows.append([route, "no fuel stop", "", "", "0.00", "",
                     _fmt(base, ".2f")])
        for tank, cons in _TANK_SPECS:
            rng   = _TANK_USABLE * tank / cons * 100
            stops = [max(0, int(np.ceil(t / rng)) - 1) for t in km]
            rows.append([route, f"{tank:g} L @ {cons:g} L/100km", tank, cons,
                         f"{_mean(stops):.2f}",
                         f"{100 * np.mean([s > 0 for s in stops]):.0f}",
                         _fmt(_pen(stops), ".2f")])

    if not rows:
        print("  Refuel note: pending (no EV/diesel oracle pairs yet)")
        return
    _write_csv("additional_diesel_refuel.csv",
               ["route", "specification", "tank_L", "cons_L_per_100km",
                "mean_fuel_stops", "pct_needing_1plus", "penalty_%"], rows)

    assumed = ", ".join(f"{r} {_REFUEL_STOPS.get(r, 0)}" for r in note)
    tex = [
        r"\begin{table}[ht]\centering",
        r"\caption{Refuelling is credited post hoc rather than modelled: it has "
        r"no binding location choice, any truck stop serving, and occurs at "
        r"most once on these corridors.  Route lengths are "
        + "; ".join(rf"{r} {lo:.0f}--{hi:.0f}\,km ($n={n}$)"
                    for r, (n, lo, hi) in note.items())
        + rf".  A fuel stop costs {_REFUEL_EVENT_H * 60:.0f}\,min (access "
        r"manoeuvre, queue, pumping).  The headline tables assume "
        + assumed
        + r" stop(s), shown against what each tank specification would derive "
        r"instead.  Crediting a stop lengthens the diesel makespan and so "
        r"shrinks the reported penalty; the zero-stop row is the conservative "
        r"bound.}",
        r"\label{tab:diesel-refuel}",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Route & Specification & Stops & Needing $\geq 1$ (\%) & "
        r"Penalty (\%) \\",
        r"\hline",
    ]
    for r in rows:
        tex.append(f"{r[0].capitalize()} & {r[1]} & {r[4]} & "
                   f"{r[5] or '--'} & ${float(r[6]):+.2f}$ \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_diesel_refuel.tex", "\n".join(tex))


# Labels for the makespan decomposition, in reporting order.  "manoeuvre" is
# now an INCREMENTAL row: both vehicles pay M_stop to pull off for a break, so
# only the EV's extra charge-only stops survive the difference.
_DECOMP_LABEL = [
    ("charging",   "Total charging time"),
    ("queue",      "Charging-station queueing"),
    ("manoeuvre",  "Stop manoeuvring (net of diesel)"),
    ("reposition", "Repositioning off the charging bay"),
    ("break",      "Break time no longer taken separately"),
    ("rest",       "Rest time"),
    ("wait",       "Idle waiting"),
    ("service",    "Customer service"),
]


def _diesel_decomposition(routes, scope) -> None:
    """Where the EV-vs-diesel makespan gap actually goes, per route class.

    Both worlds are decomposed into the terms of the model's own departure
    equations and differenced; the rows sum to the observed gap by
    construction, and the residual is asserted per instance so a silent
    accounting drift cannot pass as a result.
    """
    per: dict[str, dict[str, list]] = {}
    resid_max = 0.0
    for route, cust in (scope["combos"] or DIESEL_COMBOS):
        for tw in (scope["tws"] or TWS):
            for seed in (scope["seeds"] or SEEDS):
                st = _stem(route, cust, tw, seed)
                ev, di = _oracle_dwell(st), _oracle_dwell(st, "diesel")
                if not (ev and di):
                    continue
                d = per.setdefault(route, {k: [] for k in _DWELL_ROWS})
                for k in _DWELL_ROWS:
                    d[k].append(ev[k] - di[k])
                d.setdefault("_gap", []).append(ev["_duration"] - di["_duration"])
                resid = ((ev["_duration"] - di["_duration"])
                         - sum(ev[k] - di[k] for k in _DWELL_ROWS))
                resid_max = max(resid_max, abs(resid))

    if not per:
        print("  Decomposition: pending (no EV/diesel oracle pairs yet)")
        return
    # Driving is identical in the pair, so the components must close the gap.
    # The bound is 3.6 s, far above the float accumulation over ~85 stops
    # (~1e-6 h observed) and far below anything a real accounting slip costs.
    assert resid_max < 1e-3, f"decomposition residual {resid_max:.2e} h"
    print(f"  Decomposition: closes to {resid_max * 3600:.3f} s worst case")

    routes = [r for r in routes if r in per]
    # The modelled rows close the gap exactly (asserted above); the post-hoc
    # fuel stop is then subtracted separately, so the reported net matches the
    # penalty in Table tab:diesel rather than the raw model output.
    net = {r: _mean(per[r]["_gap"]) - _refuel_h(r) for r in routes}
    _write_csv("additional_diesel_decomp.csv",
               ["component"] + [f"{r}_h" for r in routes],
               [[lbl] + [_fmt(_mean(per[r][k]), ".3f", "") for r in routes]
                for k, lbl in _DECOMP_LABEL]
               + [["Modelled subtotal"]
                  + [_fmt(_mean(per[r]["_gap"]), ".3f", "") for r in routes],
                  ["Diesel refuelling (post hoc)"]
                  + [_fmt(-_refuel_h(r), ".3f", "") for r in routes],
                  ["Net penalty vs diesel"]
                  + [_fmt(net[r], ".3f", "") for r in routes]])

    n = min(len(per[r]["_gap"]) for r in routes)
    tex = [
        r"\begin{table}[ht]\centering",
        r"\caption{Where the electrification penalty goes: mean per-instance "
        r"difference (h) between the hindsight-optimal electric and diesel "
        r"schedules, decomposed into the terms of the departure equations. "
        r"Driving is identical within each pair, so the modelled rows sum to "
        r"the subtotal exactly. Both vehicles pay the same access manoeuvre to "
        r"pull off for a mandatory break, so only the EV's charge-only stops "
        r"survive that row. The diesel's fuel stop is credited afterwards "
        r"(Table~\ref{tab:diesel-refuel}). $n \geq " + str(n) + r"$ per class.}",
        r"\label{tab:diesel-decomp}",
        r"\begin{tabular}{l" + "r" * len(routes) + r"}",
        r"\hline",
        r"Component & " + " & ".join(r.capitalize() for r in routes) + r" \\",
        r"\hline",
    ]
    for k, lbl in _DECOMP_LABEL:
        tex.append(f"{lbl} & "
                   + " & ".join(f"${_mean(per[r][k]):+.2f}$" for r in routes)
                   + r" \\")
    tex += [
        r"\hline",
        r"Modelled subtotal & "
        + " & ".join(f"${_mean(per[r]['_gap']):+.2f}$" for r in routes) + r" \\",
        r"Diesel refuelling (post hoc) & "
        + " & ".join(f"${-_refuel_h(r):+.2f}$" for r in routes) + r" \\",
        r"\hline",
        r"\textbf{Net penalty vs.\ diesel} & "
        + " & ".join(rf"$\mathbf{{{net[r]:+.2f}}}$" for r in routes) + r" \\",
        r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_diesel_decomp.tex", "\n".join(tex))


# ══════════════════════════════════════════════════════════════════════════════
# §8.3 — SENSITIVITY (one-at-a-time)
# ══════════════════════════════════════════════════════════════════════════════

# (axis label, variant tag, planned?) — extend as new sweeps land
#
# "No split break" ("nosplit") is deliberately NOT listed.  The Art. 7 split can
# save at most Tb45 - Tb30 = 0.25 h per break entitlement, and only when the stop
# that COMPLETES the break has a charging session shorter than Tb45 (the b15 is
# banked, it does not reset cd, so both regimes complete the break at the same
# stop).  Over 1-2 entitlements on a 31-37 h route that caps the axis near 0.5 %,
# well inside the oracle's own MIPGap of 0.005 — at that tolerance more than a
# third of the pairs come out NEGATIVE, which is impossible for a restriction.
# Re-solved at MIPGap=0 the effect is real but tiny (+0.00 / +0.02 / +0.21 /
# +0.23 % on RshortCfewTlarge_17/18/11 and RshortCmanyTtight_1).  Reinstating the
# row therefore requires the whole variant set re-solved to proven optimality,
# not just a rerun of this script.
#
# The BASE CASE is the zero line, not a row: settings.CS_SPACING_KM = 60 km,
# settings.CHARGER_POWER_BASE_KW = 350 kW and settings.BATTERY_CAPACITY = 500
# kWh.  Every tag below must correspond to a directory actually produced by
# additional_analysis.py sensitivity --values, because a tag with no runs
# renders as a blank "pending" row rather than an error — and a value that was
# RUN but is not listed here is dropped silently.  Keep this list in step with
# the --values you sweep.
#
# The battery rows are NOT a pure range axis, and the table/figure must not be
# read as one.  Emin = SOC_MIN_FRAC.Ecap, so a bigger pack raises the floor too
# (+100 kWh of pack is +80 kWh of usable energy); and the tail acceptance is
# TAIL_C_RATE.Ecap, so the pack also moves where the charge curve tapers — at
# the base 350 kW the taper only binds below 875 kWh.  A capacity row therefore
# mixes range with taper avoidance, and the response is not monotone: past the
# point where charge stops fall below the HoS break count, the driver starts
# paying standalone break time that charging used to mask.  Use the `grid`
# subcommand (battery x charger_power) to separate the two effects.
_SENS_ROWS = [
    ("CS spacing 30 km",        "cs30",   True),
    ("CS spacing 100 km",       "cs100",  True),
    ("Charger power 150 kW",    "kw150",  True),
    ("Charger power 700 kW",    "kw700",  True),
    ("Charger power 1000 kW",   "kw1000", True),
    ("Battery 300 kWh",         "kwh300", True),
    ("Battery 700 kWh",         "kwh700", True),
    ("Battery 900 kWh",         "kwh900", True),
]


# Tag prefix -> experiment name, longest prefix FIRST.  The order is load-
# bearing: "kwh300" also starts with "kw", so capacity must be tested before
# power or every battery level is filed under "Charger power" and the group
# label silently spans both experiments.  Add a pair here when you add a row.
_SENS_GROUPS = [("cs",  "CS spacing"),
                ("kwh", "Battery capacity"),
                ("kw",  "Charger power")]


def _sens_group(tag: str) -> str:
    for prefix, name in sorted(_SENS_GROUPS, key=lambda p: -len(p[0])):
        if tag.startswith(prefix):
            return name
    return "Other"


_ROUTE_SPLIT = ["short", "medium"]     # fallback only; see _discover_sens_scope
# Route class is ORDERED, so it is drawn as steps of ONE hue (per method colour)
# rather than categorical hues; method colours stay reserved for the base-case
# figures.  The step is a tint of the method colour: the longer the route, the
# more saturated.  Keyed by class rather than by "is it short?" so a third class
# does not silently collapse into the darkest step.
_ROUTE_TINT = {"short": 0.55, "medium": 0.28, "long": 0.0}


def _route_face(col: str, route: str):
    """Method colour stepped by route class (lighter = shorter)."""
    frac = _ROUTE_TINT.get(route, 0.0)
    return ps.tint(col, frac) if frac else col


def _discover_scope(tags) -> dict:
    """Footprint of a tagged batch as ACTUALLY RUN, read off solutions/.

    The module constants (COMBOS / TWS / SEEDS) describe the experiments as
    first planned — short+medium only, seeds 1-10 — and a batch launched over
    anything wider was silently cropped to them.  Two sections were losing runs
    that way: §8.3 kept 80 of 300 greedy runs per tag and dropped every LA pair
    (rendering as "LA pending" when the LA runs existed all along), and §8.4
    used 120 of 300 diesel pairs.  Deriving the scope from the runs on disk
    means a wider batch reports wider instead of being quietly truncated, and
    the base leg is paired against exactly the discovered set.

    Returns combos as (route, cust) pairs plus the TW classes, seeds, and route
    classes in canonical short -> medium -> long order.
    """
    route_of = {v: k for k, v in _RTAG.items()}      # "Rshort" -> "short"
    cust_of  = {v: k for k, v in _CTAG.items()}      # "Cfew"   -> "few"
    combos, tws, seeds = set(), set(), set()
    for tag in tags:
        for path in glob.glob(_paths.solutions(f"*__{tag}_*.json")):
            m = re.match(r"^(R[a-z]+)(C[a-z]+)T([a-z]+)_(\d+)__",
                         os.path.basename(path))
            if not m:
                continue
            route, cust = route_of.get(m.group(1)), cust_of.get(m.group(2))
            if route is None or cust is None:
                continue
            combos.add((route, cust))
            tws.add(m.group(3))
            seeds.add(int(m.group(4)))
    # Route classes in the canonical short -> medium -> long order, not set order
    routes = [r for r in ps.ROUTE_ORDER if any(c[0] == r for c in combos)]
    return dict(combos=sorted(combos), tws=sorted(tws), seeds=sorted(seeds),
                routes=routes)


def section_sensitivity():
    print("== Sec 8.3 sensitivity ==")
    found = _discover_scope([t for _l, t, _p in _SENS_ROWS])
    combos_s = found["combos"] or COMBOS
    tws_s    = found["tws"]    or TWS
    seeds_s  = found["seeds"]  or list(SEEDS)
    route_split = found["routes"] or _ROUTE_SPLIT
    print(f"  scope     : combos {','.join(f'R{r}C{c}' for r, c in combos_s)}")
    print(f"              tw {','.join(tws_s)}  "
          f"seeds {min(seeds_s)}-{max(seeds_s)} (n={len(seeds_s)})  "
          f"routes {','.join(route_split)}")

    rows_out, fig_rows = [], []
    for label, tag, planned in _SENS_ROWS:
        # per-route-class deltas (the figure) and pooled (the table).  One dict
        # per method; the LA leg stays empty until the LA variant runs land, and
        # every consumer below treats an empty list as "pending" rather than 0.
        dg = {r: [] for r in route_split}
        do = {r: [] for r in route_split}
        dl = {r: [] for r in route_split}
        for route, cust in combos_s:
            if route not in dg:
                continue
            for tw in tws_s:
                for seed in seeds_s:
                    st = _stem(route, cust, tw, seed)
                    bg, vg = _greedy(st), _greedy(st, tag)
                    if (bg and vg and not bg["infeasible"]
                            and not vg["infeasible"] and bg["duration"] > 0):
                        dg[route].append(
                            100 * (vg["duration"] / bg["duration"] - 1))
                    # LA is paired exactly like greedy: both legs must exist and
                    # both must be feasible, so the delta is never contaminated
                    # by a run that stranded on one side only.
                    bl, vl = _la(st), _la(st, tag)
                    if (bl and vl and not bl["infeasible"]
                            and not vl["infeasible"] and bl["duration"] > 0):
                        dl[route].append(
                            100 * (vl["duration"] / bl["duration"] - 1))
                    bo, vo = _oracle(st), _oracle(st, tag)
                    if bo and vo and bo["duration"] > 0:
                        do[route].append(
                            100 * (vo["duration"] / bo["duration"] - 1))

        all_g = [v for r in route_split for v in dg[r]]
        all_o = [v for r in route_split for v in do[r]]
        all_l = [v for r in route_split for v in dl[r]]
        n_g, n_o, n_l = len(all_g), len(all_o), len(all_l)
        status = ("pending (needs code)" if not planned and n_g == 0 else
                  "pending" if n_g == 0 else
                  f"greedy n={n_g}"
                  + (f", LA n={n_l}" if n_l else ", LA pending")
                  + (f", oracle n={n_o}" if n_o else ", oracle pending"))
        # Per-route columns follow the DISCOVERED split, so a sweep that covers
        # long routes reports them instead of dropping them off the right edge
        # of a fixed short/medium header.
        per_route = []
        for r in route_split:
            per_route += [_fmt(_mean(do[r]), ".2f", ""), len(do[r])]
        rows_out.append([label, tag,
                         _fmt(_mean(all_g), ".2f", ""), n_g,
                         _fmt(_mean(all_l), ".2f", ""), n_l,
                         _fmt(_mean(all_o), ".2f", ""), n_o,
                         *per_route, status])
        fig_rows.append((label, planned,
                         {r: {"oracle": (_mean(do[r]), len(do[r])),
                              "LA":     (_mean(dl[r]), len(dl[r])),
                              "greedy": (_mean(dg[r]), len(dg[r]))}
                          for r in route_split}))

    _write_csv("additional_sens_stats.csv",
               ["axis", "tag", "greedy_delta_%", "n_greedy",
                "la_delta_%", "n_la",
                "oracle_delta_%", "n_oracle",
                *[c for r in route_split
                  for c in (f"oracle_delta_{r}_%", f"n_{r}")], "status"],
               rows_out)

    # ── figure ───────────────────────────────────────────────────────────────
    # Facet by route class, colour by METHOD — the same grammar as the
    # base-case figures, so Greedy is the same blue everywhere and the oracle
    # keeps its neutral-grey "benchmark" identity.
    # One panel, four bars per axis: method = HUE (oracle grey / greedy blue,
    # as everywhere else in the paper), route class = SHADE within that hue
    # (light = short, full = medium).  Shading an ordinal dimension inside a
    # categorical hue is the same device the base-case figure uses for the
    # time-window class, so the two figures read the same way.
    # Landscape orientation: the swept axes run along x and the effect along y,
    # so the response variable gets the tall, gridded axis it needs to be read
    # quantitatively (the horizontal version had to be read against ticks that
    # were 5 rows apart).
    # The sweep is two distinct experiments (charger spacing vs charger power)
    # and the levels are only comparable within one.  That grouping is carried
    # by SPACING alone — levels of the same experiment sit close together, the
    # two experiments are pushed apart — so no separator rule is needed.
    _grp = [_sens_group(t) for _l, t, _p in _SENS_ROWS]
    bounds = [i for i in range(1, len(_grp)) if _grp[i] != _grp[i - 1]]
    _BLOCK_GAP = 0.85
    x = np.array([i + _BLOCK_GAP * sum(1 for b in bounds if b <= i)
                  for i in range(len(fig_rows))], dtype=float)
    # Method order = benchmark first, then the policies in increasing
    # sophistication (oracle -> LA -> greedy), matching the base-case figures.
    _SENS_METHODS = ["oracle", "LA", "greedy"]
    series = [(m, r) for m in _SENS_METHODS for r in route_split]
    # Bar width is derived from the series count so adding a method re-packs
    # the group instead of overflowing into the neighbouring column.  Near-unit
    # width leaves only a hairline between levels of the same experiment.
    _GROUP = 0.94
    w = _GROUP / len(series)
    vals = [st[m][0] for _l, _p, per in fig_rows for st in per.values()
            for m in _SENS_METHODS if st[m][0] is not None]

    fig, ax = plt.subplots(figsize=(1.3 * (x[-1] + 1) + 1.2, 3.9))
    drawn_m, drawn_r = set(), set()

    for xi, (label, planned, per) in zip(x, fig_rows):
        any_here = False
        for k, (meth, route) in enumerate(series):
            mean_v = per[route][meth][0]
            if mean_v is None:
                continue
            any_here = True
            drawn_m.add(meth)
            drawn_r.add(route)
            col = ps.METHOD_COLOR[meth]
            face = _route_face(col, route)
            off = (k - (len(series) - 1) / 2) * w
            ax.bar(xi + off, mean_v, width=w, color=face,
                   edgecolor=col, linewidth=0.5)
            ax.text(xi + off, mean_v + np.sign(mean_v) * 0.15, f"{mean_v:+.1f}",
                    ha="center", va="bottom" if mean_v >= 0 else "top",
                    rotation=90, fontsize=5.5, color=INK)
        if not any_here:
            note = "pending" if planned else "pending (needs a model flag)"
            ax.text(xi, 0.2, note, ha="center", va="bottom", rotation=90,
                    fontsize=6.5, color=MUT, style="italic")

    ax.axhline(0, color=INK, lw=0.9)
    lo = min(-2.0, (min(vals) if vals else 0) - 2.0)
    hi = max(3.0, (max(vals) if vals else 0) + 2.0)
    # Extra headroom above the tallest bar so the in-axes legend sits over empty
    # space instead of over data.
    ax.set_ylim(lo, hi + 0.24 * (hi - lo))
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    # Tick labels keep only the level ("30 km", "150 kW"); the swept quantity is
    # carried once per block by the group label under the axis.
    ax.set_xticks(x, [" ".join(r[0].split()[-2:]) for r in fig_rows])
    ax.set_ylabel("Change in route duration vs base case (%)")

    # Major + minor grid on the response axis only: the reader compares bar
    # heights across blocks, and the minor lines make ~1% differences legible.
    ax.yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.yaxis.grid(True, which="major", color=GRID, lw=0.6)
    ax.yaxis.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.6)
    ax.tick_params(axis="y", which="minor", length=2)
    ax.set_axisbelow(True)

    # One label per block, centred under its levels — the gap above does the
    # separating, so nothing is drawn inside the axes.
    span = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for s, e in zip([0] + bounds, bounds + [len(_grp)]):
        # Fixed point offset (not an axes fraction) so the block label clears
        # the tick labels by the same margin whatever the figure height is.
        ax.annotate(_grp[s], xy=(float(x[s:e].mean()), 0), xycoords=span,
                    xytext=(0, -24), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5, color=MUT)

    # ONE legend naming each drawn bar exactly (method x route), so no reader
    # has to infer that the lighter shade means the shorter route
    handles, labels = [], []
    for meth, route in series:
        if meth not in drawn_m or route not in drawn_r:
            continue
        col = ps.METHOD_COLOR[meth]
        handles.append(plt.Rectangle(
            (0, 0), 1, 1, facecolor=_route_face(col, route),
            edgecolor=col, linewidth=0.5))
        labels.append(f"{ps.METHOD_LBL[meth]} · {route}")
    # No title of any kind — the LaTeX \caption carries it (see
    # results_section).  The legend lives INSIDE the axes, in the headroom
    # reserved above the tallest bar by the set_ylim above.
    if handles:
        ax.legend(handles, labels, frameon=True, framealpha=0.9,
                  edgecolor="none", facecolor="white", fontsize=7,
                  loc="upper center", ncol=min(3, len(handles)),
                  handlelength=1.1, handletextpad=0.4, columnspacing=1.4)
    # Bottom reserve: the block labels hang below the axes and tight_layout
    # does not measure annotations drawn outside them.
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, "additional_sens_effects")

    lines = [
        r"\begin{table}[ht]\centering",
        r"\caption{One-at-a-time sensitivity: mean change in route duration "
        r"vs the base case (\%), paired per instance.  Greedy and LA are the "
        r"online policies; Oracle is the hindsight optimum.  The trailing "
        r"columns split the Oracle column by route class.  Cells shown as "
        r"``--'' have no paired runs yet.}",
        r"\label{tab:sensitivity}",
        # Column count follows the discovered route split, so a sweep that
        # covers long routes gets a column instead of overflowing the header.
        r"\begin{tabular}{lrrr" + "r" * len(route_split) + "}", r"\hline",
        r"Axis & Greedy $\Delta$ (\%) & LA $\Delta$ (\%) & Oracle $\Delta$ (\%) & "
        + " & ".join(r.capitalize() for r in route_split) + r" \\",
        r"\hline",
    ]
    for row in rows_out:
        label, g, l, o = row[0], row[2], row[4], row[6]
        per_route = row[8:8 + 2 * len(route_split):2]      # skip the n columns
        lines.append(f"{label} & {g or '--'} & {l or '--'} & {o or '--'} & "
                     + " & ".join(str(v or '--') for v in per_route) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_sensitivity.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §8.3b — LOOK-AHEAD CONFIGURATION (horizon x scenario count)
# ══════════════════════════════════════════════════════════════════════════════

# Regulatory breakpoints the horizon axis is read against (settings.T_SPR1 and
# Tr1).  A horizon shorter than the spread cannot see the end of the current
# duty; one equal to spread + daily rest sees a whole duty cycle.  They are no
# longer drawn as reference lines — the text makes the point — but the ladder
# below is chosen to straddle them, so they stay documented here.
_LA_SPREAD_H = 13.0
_LA_CYCLE_H  = 24.0

_LA_BASE = (25, 24.0)
# Ladders drawn on the two panels.  The base cell sits on BOTH, which is what
# makes the one-at-a-time design readable as two crossing lines.
_LA_HORIZONS  = [12.0, 24.0, 48.0]
_LA_SCENARIOS = [10, 25, 50]
# Time-window class -> line style.  The house grammar puts the window class in
# the SHADE, but shade is already carrying the ordered route hue here, so on a
# line chart the window class moves to the dash pattern instead.
_LA_TW_STYLE = {"none": "-", "tight": "--"}

# Traffic-light ramp for every infeasibility heat strip in this module, built
# from the project's Okabe-Ito palette so it matches the base-case box figure:
# bluish green (all feasible) -> yellow -> vermillion (all infeasible).
from matplotlib.colors import LinearSegmentedColormap as _LSC  # noqa: E402
_INFEAS_CMAP = _LSC.from_list("infeas", ["#009E73", "#F0E442", "#D55E00"])

# The LA sweep has TWO axes and they are not comparable.  S<n>H<h> tags are the
# CONFIGURATION ladder (how much compute); anything else is a POLICY variant
# (what the policy does with it) and gets its own figure, because a policy tag
# has no position on a scenario/horizon ladder.  Listing the expected ones here
# means a variant that has not been run yet still draws a labelled empty slot,
# the same convention the rest of this module uses.
_LA_POLICY_ORDER = ["TB0", "MIPTAIL"]
_LA_POLICY_LBL = {
    "TB0":     "TB0 (no 5-min tie-break)",
    "MIPTAIL": "MIPTAIL (MIP look-ahead)",
}


def _log_ticks_plain(axis) -> None:
    """Label a log axis in plain seconds at 1-2-5 steps.

    The decision-time ranges here span well under a decade, so the default
    decade locator leaves the axis unlabelled, and the default formatter would
    write '2.2 x 10^1' if it did label it.
    """
    from matplotlib import ticker
    axis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5),
                                             numticks=12))
    axis.set_major_formatter(ticker.ScalarFormatter())
    axis.set_minor_locator(ticker.LogLocator(base=10, subs=tuple(
        x / 10 for x in range(10, 100, 5)), numticks=40))
    axis.set_minor_formatter(ticker.NullFormatter())


def _la_stats() -> dict:
    """data_output/additional_la_stats.csv -> {(cfg, route, tw): row}.

    Written by `additional_analysis.py la-report`.  Absent or partial file is
    normal while the sweep is running: missing cells render as "pending".
    """
    out: dict = {}
    path = _paths.data_output("additional_la_stats.csv")
    try:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("config"), row.get("route_class"),
                       row.get("window_class"))
                out[key] = row
    except OSError:
        pass
    return out


def _la_num(row, col):
    if not row:
        return None
    v = (row.get(col) or "").strip()
    if not v:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if np.isfinite(f) else None


# Every series in this figure is the LA policy, so it takes LA's Okabe-Ito hue
# and route class becomes the ordinal SHADE within it — the same grammar as the
# sensitivity figure (method = hue, route = shade), instead of the standalone
# blue ramp this figure used to own.  The three steps are spread wide in
# lightness (pale -> full -> near-black) because three neighbouring tints of one
# hue are not separable on a thin line; the marker SHAPE repeats the same
# ordering so the series stay distinguishable in greyscale and for readers who
# cannot rank the shades.
_LA_ROUTE_SHADE = {
    "short":  ps.tint(GREEN, 0.50),
    "medium": GREEN,
    "long":   ps.shade(GREEN, 0.62),
}
_LA_ROUTE_MARK = {"short": "o", "medium": "s", "long": "^"}


def _la_cfg(n_scen: int, horizon: float) -> str:
    return f"S{n_scen}H{horizon:g}"


def _la_cell(stats, n_scen, horizon, route, tw, col):
    """One (config, route, window) value, with the base cell aliased.

    The base cell is stored under the literal tag 'base' because those runs
    predate the sweep and carry no --variant; addressing it by its (S, H)
    coordinates keeps the ladders uniform.
    """
    cfg = "base" if (n_scen, horizon) == _LA_BASE else _la_cfg(n_scen, horizon)
    return _la_num(stats.get((cfg, route, tw)), col)


def section_la():
    print("== Sec 8.3 look-ahead configuration ==")
    stats = _la_stats()
    routes = ps.ROUTE_ORDER
    tws    = ["none", "tight"]

    # ── figure 1: quality and cost on both axes ──────────────────────────────
    # Columns are the two axes of the one-at-a-time design, rows are the two
    # things a configuration trades off.  Sharing the y-axis WITHIN a row is
    # the whole point: the reader compares the shape of the horizon response
    # against the shape of the scenario response, and any difference in slope
    # is then real rather than an artefact of two independent scales.
    # A third, short row carries the infeasibility heat strip — the same device
    # the base-case box figure puts under each panel.  It is not optional
    # decoration: a configuration that strands the truck drops those runs from
    # the gap median above, so a cell can look BETTER precisely because it
    # failed more often.  The strip is where that shows up.
    fig = plt.figure(figsize=(6.6, 5.1))
    gs  = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.34],
                           hspace=0.16, wspace=0.12)
    ax_hq = fig.add_subplot(gs[0, 0])
    ax_sq = fig.add_subplot(gs[0, 1], sharey=ax_hq)
    ax_hc = fig.add_subplot(gs[1, 0], sharex=ax_hq)
    ax_sc = fig.add_subplot(gs[1, 1], sharex=ax_sq, sharey=ax_hc)
    st_h  = fig.add_subplot(gs[2, 0], sharex=ax_hq)
    st_s  = fig.add_subplot(gs[2, 1], sharex=ax_sq)
    axes  = np.array([[ax_hq, ax_sq], [ax_hc, ax_sc]])
    for ax in (ax_hq, ax_sq, ax_hc, ax_sc):
        ax.tick_params(labelbottom=False)   # rung labels live under the strip
    for ax in (ax_sq, ax_sc):
        ax.tick_params(labelleft=False)

    for ax_q, ax_c, ladder, is_h in ((ax_hq, ax_hc, _LA_HORIZONS, True),
                                     (ax_sq, ax_sc, _LA_SCENARIOS, False)):
        drew = False
        for route in routes:
            for tw in tws:
                xs, gq, gc = [], [], []
                for v in ladder:
                    ns, hh = (_LA_BASE[0], float(v)) if is_h else (int(v),
                                                                   _LA_BASE[1])
                    q = _la_cell(stats, ns, hh, route, tw, "gap_pen_median_pct")
                    c = _la_cell(stats, ns, hh, route, tw,
                                 "decision_mean_s_median")
                    if q is None and c is None:
                        continue
                    xs.append(v); gq.append(q); gc.append(c)
                if not xs:
                    continue
                drew = True
                col = _LA_ROUTE_SHADE[route]
                sty = _LA_TW_STYLE[tw]
                mk  = _LA_ROUTE_MARK[route]
                for ax, ys in ((ax_q, gq), (ax_c, gc)):
                    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
                    if not pts:
                        continue
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            sty, color=col, marker=mk, ms=3.8, lw=1.3,
                            mfc=col, mec=col)
                    # The base cell is the reference the sweep is quoted
                    # against, so it is marked rather than left as one dot
                    # among three.
                    bx = _LA_BASE[1] if is_h else _LA_BASE[0]
                    for x, y in pts:
                        if x == bx:
                            ax.plot(x, y, mk, ms=7.5, mfc="none", mec=col,
                                    lw=0.9)
        if not drew:
            for ax in (ax_q, ax_c):
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                        fontsize=7.5, color=MUT, style="italic",
                        transform=ax.transAxes)

    # Both ladders are RATIO ladders (12/24/48, 10/25/50), so a log x-axis puts
    # the rungs at equal distances and the tick set is exactly the rungs — no
    # auto-ticks between them, no minor ticks to imply a continuum that was
    # never sampled.  (The HOS reference lines at 13 h / 24 h were dropped with
    # their labels.)
    for ax_st, ladder in ((st_h, _LA_HORIZONS), (st_s, _LA_SCENARIOS)):
        rungs = [float(v) for v in ladder]
        ax_st.set_xscale("log")     # sharex carries this up the whole column
        ax_st.set_xticks(rungs, [f"{v:g}" for v in rungs])
        ax_st.xaxis.set_minor_locator(mticker.NullLocator())
        ax_st.set_xlim(min(rungs) / 1.22, max(rungs) * 1.22)

    st_h.set_xlabel(r"Look-ahead horizon $L$ (h)   [$|\Xi| = 25$]")
    st_s.set_xlabel(r"Scenarios $|\Xi|$   [$L = 24$ h]")
    ax_hq.set_ylabel("Gap to hindsight\noptimum (%)")
    ax_hc.set_ylabel("Decision time\nper stop (s)")

    # Both response axes are LINEAR and evenly ticked.  The cost axis used to be
    # log so that equal ratios read as equal distances, but a scale whose grid
    # spacing changes at 10 s invites the reader to measure it as if it were
    # linear and get the wrong slope; a uniform grid costs some resolution at the
    # short-route end and lies about nothing.
    # Grid: vertical rules at the ladder rungs (so a point can be traced to its
    # configuration) plus major+minor horizontal rules on the response axes.
    for ax in axes.ravel():
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.grid(True, which="minor", axis="y", color=GRID, lw=0.35, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", which="minor", length=2)
        ax.tick_params(axis="x", which="minor", length=0)

    # ── infeasibility heat strip ────────────────────────────────────────────
    # Same traffic-light ramp as the base-case figure (Okabe-Ito bluish green ->
    # yellow -> vermillion) and, as there, scaled to the WORST rate actually
    # observed so the red end marks a real cell rather than a hypothetical 100%.
    # Layout: one ROW per route class (top to bottom = short, medium, long, the
    # legend's order) and, at each rung, one cell per window class — solid-line
    # class ("none") left, dashed ("tight") right, keyed by the header letters.
    from matplotlib.patches import Rectangle as _Rect
    _reds = _INFEAS_CMAP

    def _infeas(ns, hh, route, tw):
        n = _la_cell(stats, ns, hh, route, tw, "n_runs")
        i = _la_cell(stats, ns, hh, route, tw, "n_infeasible")
        return (i / n) if (n and i is not None) else None

    def _coords(is_h, v):
        return (_LA_BASE[0], float(v)) if is_h else (int(v), _LA_BASE[1])

    fracs = [f for ax_st, ladder, is_h in ((st_h, _LA_HORIZONS, True),
                                           (st_s, _LA_SCENARIOS, False))
             for v in ladder for route in routes for tw in tws
             for f in [_infeas(*_coords(is_h, v), route, tw)] if f]
    fmax = max(fracs) if fracs else 1.0

    # Cells are sized in DEX because the x-axis is log: a fixed multiplicative
    # half-width keeps every group the same visual width at every rung.
    _HALF_DEX = 0.085
    for ax_st, ladder, is_h in ((st_h, _LA_HORIZONS, True),
                                (st_s, _LA_SCENARIOS, False)):
        ax_st.set_ylim(0, len(routes))
        ax_st.set_yticks([])
        ax_st.tick_params(axis="x", length=0)
        for s in ax_st.spines.values():
            s.set_visible(False)
        for v in ladder:
            ns, hh = _coords(is_h, v)
            for ri, route in enumerate(routes):
                y0 = len(routes) - 1 - ri + 0.10      # short on top
                for ti, tw in enumerate(tws):
                    f = _infeas(ns, hh, route, tw)
                    if f is None:
                        continue                       # not run -> left blank
                    l0 = np.log10(float(v)) - _HALF_DEX + ti * _HALF_DEX
                    x0, x1 = 10 ** l0, 10 ** (l0 + _HALF_DEX)
                    ax_st.add_patch(_Rect(
                        (x0, y0), x1 - x0, 0.80,
                        facecolor=_reds(min(1.0, f / fmax)),
                        edgecolor="#8a8a8a", lw=0.35, zorder=3))

    # Row key on the left panel only, in the route shades used by the lines.
    # Initials, not words: they are keyed by the SHADE, which is the same one
    # the lines above use, so "S/M/L" decodes off the legend at a glance and the
    # strip keeps its narrow left margin.
    for ri, route in enumerate(routes):
        st_h.annotate(ps.ROUTE_LBL[route][0], xy=(0, len(routes) - 0.5 - ri),
                      xycoords=("axes fraction", "data"),
                      xytext=(-3, 0), textcoords="offset points",
                      ha="right", va="center", fontsize=5.6,
                      color=_LA_ROUTE_SHADE[route])
    # Which half of a pair is which window class, stated once.
    for ti, tw in enumerate(tws):
        l0 = np.log10(float(_LA_HORIZONS[0])) - _HALF_DEX + (ti + 0.5) * _HALF_DEX
        st_h.annotate(ps.TW_LBL[tw][0], xy=(10 ** l0, len(routes)),
                      xytext=(0, 1.5), textcoords="offset points",
                      ha="center", va="bottom", fontsize=5.0, color=MUT)
    # Strip name goes in the y-label slot, outside the row keys, so it cannot
    # collide with the N/T header letters.
    st_h.set_ylabel("Infeas.\nrate", fontsize=5.8, color=MUT, labelpad=13,
                    linespacing=0.95)

    handles = [plt.Line2D([], [], color=_LA_ROUTE_SHADE[r], lw=1.4,
                          marker=_LA_ROUTE_MARK[r], ms=3.8) for r in routes]
    labels  = [ps.ROUTE_LBL[r] for r in routes]
    handles += [plt.Line2D([], [], color=MUT, lw=1.2, ls=_LA_TW_STYLE[t])
                for t in tws]
    labels  += [f"{ps.TW_LBL[t]} windows" for t in tws]
    # Explicit margins rather than tight_layout: the heat-strip axes carry only
    # patches and annotations, which tight_layout cannot measure (it warns and
    # guesses).  Fixed margins also make the colourbar placement below exact.
    fig.subplots_adjust(left=0.135, right=0.925, top=0.895, bottom=0.105,
                        hspace=0.16, wspace=0.12)
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="upper center",
               ncol=5, bbox_to_anchor=(0.5, 0.995), handlelength=1.6,
               handletextpad=0.4, columnspacing=1.4)

    # Compact colour key for the strip, tucked to its right (as in the base-case
    # figure).  Placed after tight_layout so the axes positions are final.
    import matplotlib.cm as _cm
    from matplotlib.colors import Normalize as _Norm
    _sm = _cm.ScalarMappable(norm=_Norm(0, 100.0 * fmax), cmap=_reds)
    _sm.set_array([])
    _p  = st_s.get_position()
    _cax = fig.add_axes([_p.x1 + 0.010, _p.y0, 0.010, _p.height])
    _cb  = fig.colorbar(_sm, cax=_cax, orientation="vertical",
                        ticks=[0, 100.0 * fmax])
    _cb.ax.set_yticklabels(["0", f"{100.0 * fmax:.0f}"])
    _cb.outline.set_linewidth(0.3)
    _cb.set_label("Infeas. %", fontsize=5, labelpad=1)
    _cb.ax.tick_params(labelsize=4.4, length=1.5, width=0.3, pad=1)
    _save(fig, "additional_la_config")

    # ── figure 2: what a per-decision compute budget buys ────────────────────
    # Same five cells, re-plotted as cost against quality.  The two ladders
    # become two paths from the same base point, and the question "should the
    # next second of compute go into horizon or into scenarios?" is read off
    # directly as which path is steeper.
    fig2, axs = plt.subplots(1, len(routes), figsize=(6.6, 2.5),
                             sharey=True)
    for ax, route in zip(np.atleast_1d(axs), routes):
        drew = False
        for tw in tws:
            for ladder, is_h, col, lbl in (
                    (_LA_HORIZONS, True, ps.METHOD_COLOR["LA"], "horizon"),
                    (_LA_SCENARIOS, False, ps.METHOD_COLOR["greedy"],
                     "scenarios")):
                pts = []
                for v in ladder:
                    ns, hh = (_LA_BASE[0], float(v)) if is_h else (int(v),
                                                                   _LA_BASE[1])
                    c = _la_cell(stats, ns, hh, route, tw,
                                 "decision_mean_s_median")
                    q = _la_cell(stats, ns, hh, route, tw,
                                 "gap_pen_median_pct")
                    if c is None or q is None:
                        continue
                    pts.append((c, q, v))
                if len(pts) < 2:
                    continue
                drew = True
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        _LA_TW_STYLE[tw], color=col, marker="o", ms=3.4,
                        lw=1.2)
                for c, q, v in pts:
                    ax.annotate(f"{lbl[0].upper()}{v:g}", (c, q),
                                xytext=(3, 3), textcoords="offset points",
                                fontsize=5.5, color=MUT)
        ax.set_title(ps.ROUTE_LBL[route], loc="left")
        ax.set_xscale("log")
        _log_ticks_plain(ax.xaxis)
        ax.set_xlabel("Decision time per stop (s)")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if not drew:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    fontsize=7.5, color=MUT, style="italic",
                    transform=ax.transAxes)
    np.atleast_1d(axs)[0].set_ylabel("Gap to hindsight optimum (%)")
    h2 = [plt.Line2D([], [], color=ps.METHOD_COLOR["LA"], lw=1.4, marker="o",
                     ms=3.4),
          plt.Line2D([], [], color=ps.METHOD_COLOR["greedy"], lw=1.4,
                     marker="o", ms=3.4)]
    l2 = [r"horizon ladder ($|\Xi| = 25$)", r"scenario ladder ($L = 24$ h)"]
    fig2.tight_layout(rect=(0, 0, 1, 0.87))
    fig2.legend(h2, l2, frameon=False, fontsize=7, loc="upper center", ncol=2,
                bbox_to_anchor=(0.5, 0.995), handlelength=1.6,
                handletextpad=0.4, columnspacing=1.4)
    _save(fig2, "additional_la_frontier")

    # ── table ────────────────────────────────────────────────────────────────
    # One row per cell, route classes across the columns.  Both effect measures
    # appear: the gap is the level, the paired delta is the effect, and on long
    # routes only the second is trustworthy (the oracle's own residual MIP gap
    # is structural there and enters every level equally).
    body = []
    order = ([("base", _LA_BASE[0], _LA_BASE[1])] +
             [(_la_cfg(_LA_BASE[0], h), _LA_BASE[0], h)
              for h in _LA_HORIZONS if h != _LA_BASE[1]] +
             [(_la_cfg(s, _LA_BASE[1]), s, _LA_BASE[1])
              for s in _LA_SCENARIOS if s != _LA_BASE[0]])
    for cfg, ns, hh in order:
        for tw in tws:
            cells = []
            for route in routes:
                row = stats.get((cfg, route, tw))
                gap = _la_num(row, "gap_pen_median_pct")
                dlt = _la_num(row, "delta_vs_base_pct")
                dec = _la_num(row, "decision_mean_s_median")
                cells += [_fmt(gap, ".1f"),
                          "--" if cfg == "base" else _fmt(dlt, "+.1f"),
                          _fmt(dec, ".0f")]
            lbl = "base" if cfg == "base" else ""
            body.append((f"{ns} & {hh:g} & {ps.TW_LBL[tw]}"
                         + (f"\\rlap{{\\,\\tiny {lbl}}}" if lbl else ""),
                         cells))

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Look-ahead configuration sensitivity.  Gap is the median "
        r"gap to the hindsight optimum; $\Delta$ is the median paired change "
        r"in route duration with respect to the base cell "
        r"($|\Xi| = 25$, $L = 24$\,h), positive meaning slower; $t_{\text{dec}}$ "
        r"is the median per-stop decision time.  Cells shown as ``--'' have "
        r"no runs yet.}",
        r"\label{tab:la-config}",
        r"\begin{tabular}{rrl" + "rrr" * len(routes) + "}",
        r"\toprule",
        (r"\multicolumn{3}{c}{\textbf{Configuration}}"
         + "".join(r" & \multicolumn{3}{c}{\textbf{" + ps.ROUTE_LBL[r] + r"}}"
                   for r in routes) + r" \\"),
        r"\cmidrule(lr){1-3}"
        + "".join(r"\cmidrule(lr){%d-%d}" % (4 + 3 * i, 6 + 3 * i)
                  for i in range(len(routes))),
        (r"$|\Xi|$ & $L$ (h) & Windows"
         + r" & Gap (\%) & $\Delta$ (\%) & $t_{\text{dec}}$ (s)" * len(routes)
         + r" \\"),
        r"\midrule",
    ]
    for head, cells in body:
        lines.append(head + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_la.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §8.5 — VSS / EVPI
# ══════════════════════════════════════════════════════════════════════════════

def section_vss():
    print("== Sec 8.5 vss/evpi ==")
    agg: dict[tuple, dict[str, list]] = {}
    for f in glob.glob(_paths.results_vss("*_vss.json")):
        d = _load(f)
        if not d:
            continue
        s = d.get("summary", {})
        inst = s.get("instance", "")
        cls  = inst.split("T")[0] if "T" in inst else inst
        a = agg.setdefault(cls, dict(ws=[], rp=[], eev=[], vss=[], evpi=[]))
        for k in a:
            a[k].append(s.get(f"{k}_mean") if k in ("ws", "rp", "eev")
                        else s.get(k))

    rows, lines = [], [
        r"\begin{table}[ht]\centering",
        r"\caption{Value of the stochastic solution (VSS) and expected value "
        r"of perfect information (EVPI), common-random scenarios.}",
        r"\label{tab:vss}",
        r"\begin{tabular}{lrrrrrr}", r"\hline",
        r"Class & WS (h) & RP (h) & EEV (h) & VSS (h) & EVPI (h) & n \\",
        r"\hline",
    ]
    # Report every class the harness actually produced, in canonical order.
    # Iterating COMBOS instead would aggregate a long-route result above and
    # then never print it, the same silent drop §8.3/§8.4 used to have.
    ordered = [f"{_RTAG[r]}{_CTAG[c]}"
               for r in ps.ROUTE_ORDER for c in ("few", "medium", "many")]
    classes = ([c for c in ordered if c in agg]
               + [c for c in sorted(agg) if c not in ordered])
    for cls in classes:
        a = agg.get(cls)
        n = len(a["ws"]) if a else 0
        vals = [(_mean(a[k]) if a else None)
                for k in ("ws", "rp", "eev", "vss", "evpi")]
        rows.append([cls] + [_fmt(v, ".2f", "") for v in vals] + [n])
        cells = " & ".join(_fmt(v, ".2f") for v in vals)
        lines.append(f"{cls} & {cells} & {n or '--'} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_vss.tex", "\n".join(lines))
    _write_csv("additional_vss_stats.csv",
               ["class", "ws_h", "rp_h", "eev_h", "vss_h", "evpi_h", "n"],
               rows)
    if not agg:
        print("  (results_vss/ is empty — table is a skeleton; run "
              "'python -m src.output_analysis.additional_analysis vss')")


# ══════════════════════════════════════════════════════════════════════════════

# Effect measures the policy figure can be drawn against.  "pen" is the model's
# actual objective (arrival + beta * window misses, expressed as a duration);
# "dur" is route duration alone.  They differ whenever a configuration trades
# lateness for a shorter route — the duration view scores that as a win.
_LA_EFFECT = {
    "dur": dict(col="delta_vs_base_pct", n="n_paired", sfx="",
                ylabel="Paired change in route\nduration vs base (%)",
                tex="route duration"),
    "pen": dict(col="delta_pen_vs_base_pct", n="n_paired_pen", sfx="_pen",
                ylabel="Paired change in penalised\nobjective vs base (%)",
                tex=r"penalised objective (arrival $+\ \beta\times$ misses)"),
}


def section_la_policy(effect: str = "dur"):
    """§8.3 — LA POLICY variants (TB0, MIPTAIL), one figure + one table.

    Separate from section_la on purpose.  The S/H sweep asks "how much compute
    should LA get?" and its cells sit on two ladders; these variants ask "what
    should LA do with it?" and have no position on a ladder, so plotting them
    on the same axes would invent an ordering that does not exist.

    The effect measure is the PAIRED delta against the base cell, not the gap
    level: the oracle cancels out of a difference of two policy runs on the same
    instance, so the delta is unaffected by the oracle's own residual MIP gap
    (structural, ~9% on long routes).  The gap level is still printed in the
    table as the calibration.
    """
    eff = _LA_EFFECT[effect]
    print(f"== Sec 8.3 look-ahead policy variants [{effect}] ==")
    stats = _la_stats()
    routes = ps.ROUTE_ORDER
    tws    = ["none", "tight"]

    have = {c for (c, _r, _t) in stats}
    extra = sorted(c for c in have
                   if c != "base" and c not in _LA_POLICY_ORDER
                   and not re.fullmatch(r"S\d+H\d+(\.\d+)?", c))
    cfgs = _LA_POLICY_ORDER + extra
    if not cfgs:
        print("  no policy variants in additional_la_stats.csv — nothing drawn")
        return

    def cell(cfg, route, tw, col):
        return _la_num(stats.get((cfg, route, tw)), col)

    # ── x layout: 6 cells, grouped into route blocks by SPACING ──────────────
    _BLOCK_GAP = 0.7
    xs, keys = [], []
    for ri, route in enumerate(routes):
        for ti, tw in enumerate(tws):
            xs.append(ri * (len(tws) + _BLOCK_GAP) + ti)
            keys.append((route, tw))
    xs = np.asarray(xs, dtype=float)

    # Every series here is the LA policy, so they take LA's hue and separate by
    # lightness — the same grammar as the configuration figure.
    face = {"base": ps.tint(GREEN, 0.62)}
    for i, c in enumerate(cfgs):
        face[c] = (GREEN if i == 0 else
                   ps.shade(GREEN, min(0.62, 0.30 + 0.18 * i)))

    fig = plt.figure(figsize=(7.0, 5.0))
    gs  = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.72, 0.30], hspace=0.18)
    ax_d = fig.add_subplot(gs[0])                      # paired effect
    ax_c = fig.add_subplot(gs[1], sharex=ax_d)         # cost
    ax_s = fig.add_subplot(gs[2], sharex=ax_d)         # infeasibility strip
    ax_d.tick_params(labelbottom=False)
    ax_c.tick_params(labelbottom=False)

    # ── row 0: paired delta vs base (base is the zero line by construction) ──
    w = 0.78 / max(1, len(cfgs))
    drawn = []
    for k, cfg in enumerate(cfgs):
        vals = [cell(cfg, r, t, eff["col"]) for r, t in keys]
        if all(v is None for v in vals):
            continue
        drawn.append(cfg)
        off = (k - (len(cfgs) - 1) / 2) * w
        for x, v in zip(xs, vals):
            if v is None:
                continue
            ax_d.bar(x + off, v, width=w, color=face[cfg],
                     edgecolor=ps.shade(GREEN, 0.5), linewidth=0.5)
            # Point offset, not a data offset: the two effect measures live on
            # different scales and a fixed data nudge would collide on one of
            # them.
            ax_d.annotate(f"{v:+.2f}", xy=(x + off, v),
                          xytext=(0, 2 if v >= 0 else -2),
                          textcoords="offset points", ha="center",
                          va="bottom" if v >= 0 else "top",
                          rotation=90, fontsize=5.2, color=INK)
    pending = [c for c in cfgs if c not in drawn]
    if pending:
        # Parked in the corner, not on the zero line: the zero line is the base
        # reference and a note sitting on it reads as a data label.
        ax_d.text(0.012, 0.03, "pending: " + ", ".join(pending),
                  transform=ax_d.transAxes, ha="left", va="bottom",
                  fontsize=6.5, color=MUT, style="italic")
    ax_d.axhline(0, color=INK, lw=0.9)
    ax_d.set_ylabel(eff["ylabel"])

    # ── row 1: what it costs (base drawn too — the cost may move on its own) ─
    w2 = 0.78 / (len(cfgs) + 1)
    for k, cfg in enumerate(["base"] + cfgs):
        vals = [cell(cfg, r, t, "decision_mean_s_median") for r, t in keys]
        if all(v is None for v in vals):
            continue
        off = (k - len(cfgs) / 2) * w2
        for x, v in zip(xs, vals):
            if v is None:
                continue
            ax_c.bar(x + off, v, width=w2, color=face[cfg],
                     edgecolor=ps.shade(GREEN, 0.5), linewidth=0.5)
    ax_c.set_ylabel("Decision time\nper stop (s)")

    for ax in (ax_d, ax_c):
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(True, which="major", axis="y", color=GRID, lw=0.6)
        ax.grid(True, which="minor", axis="y", color=GRID, lw=0.35, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", which="minor", length=2)

    # ── row 2: infeasibility strip, one row per config ───────────────────────
    # Same device and ramp as the base-case box figure: a variant that strands
    # more drops those runs from the delta above, so it can look better for the
    # wrong reason.  This is where that shows.
    from matplotlib.patches import Rectangle as _Rect
    strip_cfgs = ["base"] + drawn

    def infeas(cfg, route, tw):
        n = cell(cfg, route, tw, "n_runs")
        i = cell(cfg, route, tw, "n_infeasible")
        return (i / n) if (n and i is not None) else None

    fr = [f for cfg in strip_cfgs for r, t in keys
          for f in [infeas(cfg, r, t)] if f]
    fmax = max(fr) if fr else 1.0
    ax_s.set_ylim(0, len(strip_cfgs))
    ax_s.set_yticks([])
    for s in ax_s.spines.values():
        s.set_visible(False)
    for ci, cfg in enumerate(strip_cfgs):
        y0 = len(strip_cfgs) - 1 - ci + 0.12
        for x, (r, t) in zip(xs, keys):
            f = infeas(cfg, r, t)
            if f is None:
                continue
            ax_s.add_patch(_Rect((x - 0.42, y0), 0.84, 0.76,
                                 facecolor=_INFEAS_CMAP(min(1.0, f / fmax)),
                                 edgecolor="#8a8a8a", lw=0.35, zorder=3))
        ax_s.annotate(cfg, xy=(0, len(strip_cfgs) - 0.5 - ci),
                      xycoords=("axes fraction", "data"),
                      xytext=(-3, 0), textcoords="offset points",
                      ha="right", va="center", fontsize=5.6,
                      color=face[cfg] if cfg != "base" else MUT)
    ax_s.set_ylabel("Infeas.\nrate", fontsize=5.8, color=MUT, labelpad=22,
                    linespacing=0.95)

    # window class on the ticks, route class as the block label underneath
    ax_s.set_xticks(xs, [ps.TW_LBL[t] for _r, t in keys])
    ax_s.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax_s.tick_params(axis="x", length=0)
    span = ax_s.get_xaxis_transform()
    for ri, route in enumerate(routes):
        c = xs[ri * len(tws):(ri + 1) * len(tws)].mean()
        ax_s.annotate(ps.ROUTE_LBL[route], xy=(c, 0), xycoords=span,
                      xytext=(0, -22), textcoords="offset points",
                      ha="center", va="top", fontsize=7.5, color=MUT)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=face[c],
                             edgecolor=ps.shade(GREEN, 0.5), linewidth=0.5)
               for c in ["base"] + drawn]
    labels = ["base ($|\\Xi|=25$, $L=24$ h)"] + [
        _LA_POLICY_LBL.get(c, c) for c in drawn]
    ax_d.legend(handles, labels, frameon=True, framealpha=0.9,
                edgecolor="none", facecolor="white", fontsize=7,
                loc="upper center", ncol=min(3, len(handles)),
                handlelength=1.1, handletextpad=0.4, columnspacing=1.4)
    # Headroom above for the legend, room below for the rotated value labels
    # under the negative bars (they are drawn outside the data range).
    lo, hi = ax_d.get_ylim()
    ax_d.set_ylim(lo - 0.22 * (hi - lo), hi + 0.42 * (hi - lo))
    fig.subplots_adjust(left=0.145, right=0.915, top=0.975, bottom=0.115)

    # Colour key for the strip — without it "red" has no magnitude, and the
    # ramp is scaled to the worst cell in THIS figure, not to 100%.
    import matplotlib.cm as _cm
    from matplotlib.colors import Normalize as _Norm
    _sm = _cm.ScalarMappable(norm=_Norm(0, 100.0 * fmax), cmap=_INFEAS_CMAP)
    _sm.set_array([])
    _p = ax_s.get_position()
    _cax = fig.add_axes([_p.x1 + 0.012, _p.y0, 0.010, _p.height])
    _cb = fig.colorbar(_sm, cax=_cax, orientation="vertical",
                       ticks=[0, 100.0 * fmax])
    _cb.ax.set_yticklabels(["0", f"{100.0 * fmax:.0f}"])
    _cb.outline.set_linewidth(0.3)
    _cb.set_label("Infeas. %", fontsize=5, labelpad=1)
    _cb.ax.tick_params(labelsize=4.4, length=1.5, width=0.3, pad=1)
    _save(fig, f"additional_la_policy{eff['sfx']}")

    # ── table ────────────────────────────────────────────────────────────────
    body = []
    for cfg in ["base"] + drawn:
        for tw in tws:
            cells = []
            for route in routes:
                cells += [_fmt(cell(cfg, route, tw, "gap_pen_median_pct"), ".1f"),
                          "--" if cfg == "base"
                          else _fmt(cell(cfg, route, tw, eff["col"]), "+.2f"),
                          _fmt(cell(cfg, route, tw, "decision_mean_s_median"),
                               ".0f")]
            body.append((f"{cfg} & {ps.TW_LBL[tw]}", cells))

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Look-ahead POLICY variants, against the base cell "
        r"($|\Xi| = 25$, $L = 24$\,h).  TB0 removes the 5-minute tie-break that "
        r"buys opportunistic charging; MIPTAIL solves the look-ahead tail as a "
        r"MIP instead of an LP relaxation.  Gap is the median gap to the "
        r"hindsight optimum, $\Delta$ the median paired change in "
        + eff["tex"] +
        r" (positive = worse), $t_{\text{dec}}$ the median decision "
        r"time per stop.}",
        r"\label{tab:la_policy" + eff["sfx"] + r"}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "rrr" * len(routes) + r"}", r"\toprule",
        " & ".join([r"Config & Windows"]
                   + [r"\multicolumn{3}{c}{" + ps.ROUTE_LBL[r] + "}"
                      for r in routes]) + r" \\",
        r"\midrule",
        " & ".join([r" & "] + [r"Gap (\%) & $\Delta$ (\%) & $t_{\text{dec}}$ (s)"
                               for _ in routes]) + r" \\",
        r"\midrule",
    ]
    for head, cells in body:
        lines.append(f"{head} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    _write_tex(f"additional_la_policy{eff['sfx']}.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = dict(diesel=section_diesel, sensitivity=section_sensitivity,
                 la=section_la,
                 # both effect measures: route duration and the penalised
                 # objective the model actually minimises
                 la_policy=lambda: (section_la_policy("dur"),
                                    section_la_policy("pen")),
                 vss=section_vss)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real tables/figures for the "
                                             "additional analyses (8.3-8.5)")
    ap.add_argument("--section", default="all",
                    choices=["all", *_SECTIONS])
    args = ap.parse_args()
    _paths.ensure_dirs()
    for name, fn in _SECTIONS.items():
        if args.section in ("all", name):
            fn()
