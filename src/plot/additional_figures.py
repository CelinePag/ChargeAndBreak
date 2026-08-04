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
  figures/additional_gamma_frontier.png|pdf  §8.5 frontier (endpoints only yet)
  tex/tables/additional_vss.tex              §8.5 VSS/EVPI (skeleton until
  data_output/additional_vss_stats.csv             results_vss/ fills up)

§8.3 reports three methods per axis: greedy and LA (online policies) and the
oracle (hindsight optimum).  Each is PAIRED per instance — base and variant
must both exist and both be feasible — so a method whose variant runs have not
landed yet shows "--" in the table and no bar in the figure, rather than a
misleading zero.

Usage
  python -m src.plot.additional_figures                 # all sections
  python -m src.plot.additional_figures --section diesel|sensitivity|gamma|vss
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    per_class: dict[str, dict[str, list]] = {}
    detail = []
    for route, cust in DIESEL_COMBOS:
        for tw in TWS:
            for seed in SEEDS:
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

    # ── Oracle coverage guard ────────────────────────────────────────────────
    # On long routes BOTH oracles stall on the dual bound (the rest-packing
    # structure, not the energy dimension), so the class is reported from
    # Greedy alone.  The instances that DO finish there are the easy tail, so
    # averaging whatever happens to be cached would silently report a biased
    # subset as if it were the class.  Oracle-derived quantities are therefore
    # suppressed for any class whose oracle coverage is incomplete; the
    # per-instance values stay in the CSV above with their own n.
    oracle_ok = []
    for r, d in per_class.items():
        have = sum(1 for v in d["pen_o"] if v is not None)
        want = len(TWS) * len(SEEDS) * sum(1 for rr, _ in DIESEL_COMBOS
                                           if rr == r)
        if have < want:
            print(f"  Oracle coverage {r}: {have}/{want} — reporting Greedy "
                  f"only for this class")
            for key in ("pen_o", "dt_o", "coup", "ev_gap", "dur_d", "dur_e"):
                d[key] = [None] * len(d[key])
        else:
            oracle_ok.append(r)

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
        if n:
            note = f"{_fmt(c, '.0f')}% coupled  ·  n = {n}"
        else:   # oracle suppressed for this class — label the Greedy sample
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
    def _maxgap(r):
        v = [x for x in per_class[r]["ev_gap"] if x is not None]
        return max(v) if v else None

    lines = [
        r"\begin{table}[ht]\centering",
        r"\caption{EV vs diesel route duration (\%) on the same instances and "
        r"realizations.  Coupling = share of charging time credited inside a "
        r"mandatory break ($\Sigma g_i/\Sigma\tau^c_i$, hindsight optimum). "
        r"``EV cert.'' is the worst remaining MIP gap on the EV oracle: "
        r"where it exceeds the solver tolerance the EV schedule is an "
        r"incumbent, not a proven optimum, so the penalty is an upper bound "
        r"on the true optimal one.}",
        r"\label{tab:diesel}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Route & Diesel (h) & EV (h) & Greedy (\%) & "
        r"Oracle (\%) & Coupling (\%) & EV cert. (\%) \\",
        r"\hline",
    ]
    for r in routes:
        d = per_class[r]
        lines.append(
            f"{r.capitalize()} & {_fmt(_mean(d['dur_d']))} & "
            f"{_fmt(_mean(d['dur_e']))} & "
            f"{_fmt(_mean(d['pen_g']))} & {_fmt(_mean(d['pen_o']))} & "
            f"{_fmt(_mean(d['coup']), '.0f')} & "
            f"{_fmt(_maxgap(r), '.1f')} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_diesel.tex", "\n".join(lines))

    # Both are oracle-derived, so they follow the coverage guard above.
    ok = [r for r in routes if r in oracle_ok]
    _diesel_decomposition(ok)
    _refuel_sensitivity(ok)
    _diesel_by_tw(routes, ok)


# Window classes ordered tight -> none, i.e. loosening the constraint, so the
# trend reads left to right (paper_figures.py uses the same order).
_TW_ORDER = ["tight", "medium", "large", "none"]


def _diesel_by_tw(routes, oracle_ok) -> None:
    """Split the penalty by time-window class.

    Answers whether the EV/diesel gap is a real makespan difference or an
    artefact of the window penalty: the objective is
    ta[N] + BETA_TW*sum(delta), so a schedule can trade makespan against a
    missed window.  Both the duration-based and the objective-based penalty
    are reported, along with the delta counts that separate them.
    """
    per: dict[tuple, dict] = {}
    for route, cust in DIESEL_COMBOS:
        for tw in _TW_ORDER:
            for seed in SEEDS:
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

    routes = [r for r in routes if any((r, tw) in per for tw in _TW_ORDER)]
    if not routes:
        return

    # ── small multiples: one panel per route class ───────────────────────────
    fig, axes = plt.subplots(1, len(routes), figsize=(2.3 * len(routes) + 0.9,
                                                      2.9), sharey=True)
    axes = np.atleast_1d(axes)
    x, w = np.arange(len(_TW_ORDER)), 0.34
    series = [("Greedy", "pen_g", BLUE), ("Oracle", "pen_o", INK)]
    top = 0.0
    for ax, route in zip(axes, routes):
        for k, (lbl, key, col) in enumerate(series):
            vals = [_mean(per[(route, tw)][key]) if (route, tw) in per else None
                    for tw in _TW_ORDER]
            top  = max([top] + [v for v in vals if v is not None])
            ax.bar(x + (k - 0.5) * w,
                   [np.nan if v is None else v for v in vals], w,
                   color=col, edgecolor="white", linewidth=0.8,
                   label=lbl if ax is axes[0] else None)
        ax.set_xticks(x, [t.capitalize() for t in _TW_ORDER], fontsize=7)
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
        for tw in _TW_ORDER:
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


def _refuel_sensitivity(routes) -> None:
    """Check the assumed fuel-stop count against what tank specs imply.

    The headline tables credit _REFUEL_STOPS per route class.  This reports
    what each tank specification would derive instead, and what the penalty
    would be under it, so the assumption is auditable rather than asserted.
    """
    rows, note = [], {}
    for route in routes:
        km, pen = [], []
        for r, cust in DIESEL_COMBOS:
            if r != route:
                continue
            for tw in TWS:
                for seed in SEEDS:
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


def _diesel_decomposition(routes) -> None:
    """Where the EV-vs-diesel makespan gap actually goes, per route class.

    Both worlds are decomposed into the terms of the model's own departure
    equations and differenced; the rows sum to the observed gap by
    construction, and the residual is asserted per instance so a silent
    accounting drift cannot pass as a result.
    """
    per: dict[str, dict[str, list]] = {}
    resid_max = 0.0
    for route, cust in DIESEL_COMBOS:
        for tw in TWS:
            for seed in SEEDS:
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
_SENS_ROWS = [
    ("CS spacing 30 km",        "cs30",   True),
    ("CS spacing 90 km",        "cs90",   True),
    ("Charger power 150 kW",    "kw150",  True),
    ("Charger power 350 kW",    "kw350",  True),
    ("Charger power 1000 kW",   "kw1000", True),
]


_ROUTE_SPLIT = ["short", "medium"]
# route class is ORDERED (short -> medium), so it gets two steps of one hue
# rather than two categorical hues; method colours stay reserved for the
# base-case figures.
_ROUTE_COLOR = {"short": "#9ecae1", "medium": "#2171b5"}


def section_sensitivity():
    print("== Sec 8.3 sensitivity ==")
    rows_out, fig_rows = [], []
    for label, tag, planned in _SENS_ROWS:
        # per-route-class deltas (the figure) and pooled (the table).  One dict
        # per method; the LA leg stays empty until the LA variant runs land, and
        # every consumer below treats an empty list as "pending" rather than 0.
        dg = {r: [] for r in _ROUTE_SPLIT}
        do = {r: [] for r in _ROUTE_SPLIT}
        dl = {r: [] for r in _ROUTE_SPLIT}
        for route, cust in COMBOS:
            if route not in dg:
                continue
            for tw in TWS:
                for seed in SEEDS:
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

        all_g = [v for r in _ROUTE_SPLIT for v in dg[r]]
        all_o = [v for r in _ROUTE_SPLIT for v in do[r]]
        all_l = [v for r in _ROUTE_SPLIT for v in dl[r]]
        n_g, n_o, n_l = len(all_g), len(all_o), len(all_l)
        status = ("pending (needs code)" if not planned and n_g == 0 else
                  "pending" if n_g == 0 else
                  f"greedy n={n_g}"
                  + (f", LA n={n_l}" if n_l else ", LA pending")
                  + (f", oracle n={n_o}" if n_o else ", oracle pending"))
        rows_out.append([label, tag,
                         _fmt(_mean(all_g), ".2f", ""), n_g,
                         _fmt(_mean(all_l), ".2f", ""), n_l,
                         _fmt(_mean(all_o), ".2f", ""), n_o,
                         _fmt(_mean(do["short"]), ".2f", ""), len(do["short"]),
                         _fmt(_mean(do["medium"]), ".2f", ""),
                         len(do["medium"]), status])
        fig_rows.append((label, planned,
                         {r: {"oracle": (_mean(do[r]), len(do[r])),
                              "LA":     (_mean(dl[r]), len(dl[r])),
                              "greedy": (_mean(dg[r]), len(dg[r]))}
                          for r in _ROUTE_SPLIT}))

    _write_csv("additional_sens_stats.csv",
               ["axis", "tag", "greedy_delta_%", "n_greedy",
                "la_delta_%", "n_la",
                "oracle_delta_%", "n_oracle",
                "oracle_delta_short_%", "n_short",
                "oracle_delta_medium_%", "n_medium", "status"], rows_out)

    # ── figure ───────────────────────────────────────────────────────────────
    # Facet by route class, colour by METHOD — the same grammar as the
    # base-case figures, so Greedy is the same blue everywhere and the oracle
    # keeps its neutral-grey "benchmark" identity.
    # One panel, four bars per axis: method = HUE (oracle grey / greedy blue,
    # as everywhere else in the paper), route class = SHADE within that hue
    # (light = short, full = medium).  Shading an ordinal dimension inside a
    # categorical hue is the same device the base-case figure uses for the
    # time-window class, so the two figures read the same way.
    y = np.arange(len(fig_rows))[::-1]
    # Method order = benchmark first, then the policies in increasing
    # sophistication (oracle -> LA -> greedy), matching the base-case figures.
    _SENS_METHODS = ["oracle", "LA", "greedy"]
    series = [(m, r) for m in _SENS_METHODS for r in _ROUTE_SPLIT]
    # Bar height is derived from the series count so adding a method re-packs
    # the group instead of overflowing into the neighbouring row (6 bars at the
    # old hard-coded 0.19 would have spanned 1.14 of a 1.0 row).
    _GROUP = 0.78
    h = _GROUP / len(series)
    vals = [st[m][0] for _l, _p, per in fig_rows for st in per.values()
            for m in _SENS_METHODS if st[m][0] is not None]

    # +1.6 of non-bar height: no in-figure title, but the legend band above the
    # axes needs room for its caption line plus the swatch row.
    # Taller rows than the two-method version: six bars per axis need the room,
    # and the legend now wraps onto two lines.
    fig, ax = plt.subplots(figsize=(6.6, 0.95 * len(fig_rows) + 1.9))
    drawn_m, drawn_r = set(), set()

    for yi, (label, planned, per) in zip(y, fig_rows):
        any_here = False
        for k, (meth, route) in enumerate(series):
            mean_v = per[route][meth][0]
            if mean_v is None:
                continue
            any_here = True
            drawn_m.add(meth)
            drawn_r.add(route)
            col = ps.METHOD_COLOR[meth]
            face = ps.tint(col, 0.45) if route == "short" else col
            off = ((len(series) - 1) / 2 - k) * h
            ax.barh(yi + off, mean_v, height=h, color=face,
                    edgecolor=col, linewidth=0.5)
            ax.text(mean_v + np.sign(mean_v) * 0.18, yi + off, f"{mean_v:+.1f}",
                    ha="left" if mean_v >= 0 else "right", va="center",
                    fontsize=5.5, color=INK)
        if not any_here:
            note = "pending" if planned else "pending (needs a model flag)"
            ax.text(0.2, yi, note, ha="left", va="center",
                    fontsize=6.5, color=MUT, style="italic")

    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlim(min(-2.0, (min(vals) if vals else 0) - 3.0),
                max(3.0, (max(vals) if vals else 0) + 3.0))
    ax.set_ylim(-0.6, len(fig_rows) - 0.4)
    ax.set_yticks(y, [r[0] for r in fig_rows])
    ax.set_xlabel("Change in route duration vs base case (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # ONE legend naming each drawn bar exactly (method x route), so no reader
    # has to infer that the lighter shade means the shorter route
    handles, labels = [], []
    for meth, route in series:
        if meth not in drawn_m or route not in drawn_r:
            continue
        col = ps.METHOD_COLOR[meth]
        handles.append(plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=ps.tint(col, 0.45) if route == "short" else col,
            edgecolor=col, linewidth=0.5))
        labels.append(f"{ps.METHOD_LBL[meth]} · {route}")
    # No in-figure title — the LaTeX \caption carries it (see results_section).
    # The legend is a figure-level row above the axes rather than inside them:
    # every in-axes corner is claimed by a bar or its value label at some point
    # in the sweep, and the bottom-left corner it used to occupy became the
    # largest negative bar the moment an axis was dropped from _SENS_ROWS.
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    if handles:
        # Wrap at 3 columns: with three methods x two route classes the single
        # row the two-method version used would run past the figure width.
        fig.legend(handles, labels, frameon=False, fontsize=7,
                   loc="upper center", ncol=min(3, len(handles)),
                   bbox_to_anchor=(0.5, 0.995),
                   title="hindsight optimum vs online policies",
                   title_fontsize=7.5,
                   handlelength=1.1, handletextpad=0.4, columnspacing=1.4)
    _save(fig, "additional_sens_effects")

    lines = [
        r"\begin{table}[ht]\centering",
        r"\caption{One-at-a-time sensitivity: mean change in route duration "
        r"vs the base case (\%), paired per instance.  Greedy and LA are the "
        r"online policies; Oracle is the hindsight optimum.  The last two "
        r"columns split the Oracle column by route class.  Cells shown as "
        r"``--'' have no paired runs yet.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{lrrrrr}", r"\hline",
        r"Axis & Greedy $\Delta$ (\%) & LA $\Delta$ (\%) "
        r"& Oracle $\Delta$ (\%) & Short & Medium \\",
        r"\hline",
    ]
    for (label, _tag, g, _ng, l, _nl, o, _no, o_s, _ns, o_m, _nm,
         _status) in rows_out:
        lines.append(f"{label} & {g or '--'} & {l or '--'} & {o or '--'} & "
                     f"{o_s or '--'} & {o_m or '--'} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_sensitivity.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §8.5 — GAMMA FRONTIER (endpoints from base case until the sweep runs)
# ══════════════════════════════════════════════════════════════════════════════

def section_gamma():
    print("== Sec 8.5 gamma frontier ==")
    # sweep points (from tagged runs, none yet) --------------------------------
    sweep = {}
    for gam in (0, 1, 2, 4, 8):
        gaps, infeas, tot = [], 0, 0
        for route, cust in COMBOS:
            if route != "short":
                continue
            for tw in TWS:
                for seed in SEEDS:
                    st = _stem(route, cust, tw, seed)
                    f = (_latest(_paths.solutions(f"{st}__g{gam}_ROBU_*.json"))
                         or _latest(_paths.solutions(f"{st}_g{gam}_ROBU_*.json")))
                    if not f:
                        continue
                    d = _load(f) or {}
                    m = d.get("metrics") or {}
                    tot += 1
                    if m.get("run_infeasible"):
                        infeas += 1
                    o = _oracle(st)
                    if o and d.get("duration_h") and not m.get(
                            "run_infeasible"):
                        gaps.append(100 * (d["duration_h"] / o["duration"]
                                           - 1))
        if tot:
            sweep[gam] = (_mean(gaps), 100 * infeas / tot, tot)

    # base-case endpoints from paper_gap_stats.csv -----------------------------
    end = {}
    try:
        with open(_paths.data_output("paper_gap_stats.csv"), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["route_class"] != "short":
                    continue
                m = row["method"]
                if m not in ("RO", "ROBU"):
                    continue
                nf = int(row["n_feasible"]); ni = int(
                    row["n_infeasible_excluded"])
                e = end.setdefault(m, dict(g=[], w=[], inf=0, tot=0))
                if row["gap_mean_%"]:
                    e["g"].append(float(row["gap_mean_%"])); e["w"].append(nf)
                e["inf"] += ni; e["tot"] += nf + ni
    except OSError:
        pass

    cats = ["0", "1", "2", "4", "8", r"$\sqrt{N}$" + "\n(base ROBU)",
            r"$N$" + "\n(box = RO)"]
    xs   = np.arange(len(cats))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 3.6), sharex=True,
                                   height_ratios=[3, 2])
    for i, gam in enumerate((0, 1, 2, 4, 8)):
        if gam in sweep:
            ax1.plot(xs[i], sweep[gam][0], "o", color=ORAN, ms=5)
            ax2.plot(xs[i], sweep[gam][1], "o", color=ORAN, ms=5)
        else:
            for ax in (ax1, ax2):
                ax.text(xs[i], 0.5, "pending", rotation=90, ha="center",
                        va="center", fontsize=6.5, color=MUT, style="italic",
                        transform=ax.get_xaxis_transform())
    for m, xi, col in (("ROBU", xs[5], ORAN), ("RO", xs[6], VERM)):
        e = end.get(m)
        if e and e["g"]:
            gap = float(np.average(e["g"], weights=e["w"]))
            inf = 100 * e["inf"] / e["tot"] if e["tot"] else None
            ax1.plot(xi, gap, "s", color=col, ms=6)
            ax1.annotate(f"{gap:.1f}", (xi, gap), xytext=(-4, -10),
                         textcoords="offset points", ha="right", fontsize=7)
            if inf is not None:
                ax2.plot(xi, inf, "s", color=col, ms=6)
                ax2.annotate(f"{inf:.1f}", (xi, inf), xytext=(-4, 4),
                             textcoords="offset points", ha="right",
                             fontsize=7)
    ax1.margins(y=0.2)
    ax2.margins(y=0.2)
    ax1.set_ylabel("Realized gap to oracle (%)")
    ax2.set_ylabel("Infeasible runs (%)")
    ax2.set_xticks(xs, cats)
    ax2.set_xlabel(r"Uncertainty budget $\Gamma$")
    for ax in (ax1, ax2):
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax1.set_title("Price of robustness (short routes) — sweep pending, "
                  "endpoints from base case", loc="left")
    _save(fig, "additional_gamma_frontier")


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
    for route, cust in COMBOS:
        cls = f"{_RTAG[route]}{_CTAG[cust]}"
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

_SECTIONS = dict(diesel=section_diesel, sensitivity=section_sensitivity,
                 gamma=section_gamma, vss=section_vss)

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
