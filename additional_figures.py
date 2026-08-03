"""
additional_figures.py — REAL tables & figures for the additional analyses
(paper Sections 8.3-8.5), built from whatever runs currently exist.

Unlike experiments/mock_section_figures.py (random placeholder data), this
script reads solutions/ (+ oracle caches, results_vss/) and renders the
paper artefacts with the data available NOW; cells or panels whose runs have
not finished yet are shown explicitly as "pending", mirroring the
paper_figures.py convention of drawing the full grid with empty slots.

Outputs
  figures/additional_diesel_gap.png|pdf     §8.4 figure
  figures/additional_diesel_stats.csv       §8.4 per-class detail
  tables/additional_diesel.tex              §8.4 table
  figures/additional_sens_effects.png|pdf   §8.3 one-at-a-time (preliminary)
  figures/additional_sens_stats.csv
  tables/additional_sensitivity.tex
  figures/additional_gamma_frontier.png|pdf §8.5 frontier (endpoints only yet)
  tables/additional_vss.tex                 §8.5 VSS/EVPI (skeleton until
  figures/additional_vss_stats.csv                results_vss/ fills up)

Usage
  python additional_figures.py                 # all sections
  python additional_figures.py --section diesel|sensitivity|gamma|vss
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

from settings import T_START

# Shared palette + chrome (see paper_style.py): colour follows the entity, so
# Greedy is the same blue here as in the base-case figures.
import paper_style as ps

INK, MUT, GRID = ps.INK_PRIMARY, ps.INK_MUTED, ps.GRID
BLUE  = ps.METHOD_COLOR["greedy"]
VERM  = ps.METHOD_COLOR["RO"]
ORAN  = ps.METHOD_COLOR["ROBU"]
GREEN = ps.METHOD_COLOR["LA"]
PURP  = ps.METHOD_COLOR["2SP"]
ps.apply_rc()

COMBOS = [("short", "few"), ("short", "many"), ("medium", "few"),
          ("medium", "many")]
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


def _greedy(stem: str, tag: str | None = None) -> dict | None:
    """Latest greedy solution for a (possibly tagged) instance.  Accepts both
    the '__tag' stem (orchestrator batch) and the runner-normalised '_tag'."""
    pats = ([f"solutions/{stem}_GREEDY_*.json"] if tag is None else
            [f"solutions/{stem}__{tag}_GREEDY_*.json",
             f"solutions/{stem}_{tag}_GREEDY_*.json"])
    for p in pats:
        f = _latest(p)
        if f:
            d = _load(f)
            if d and d.get("duration_h") is not None:
                infeas = bool((d.get("metrics") or {}).get("run_infeasible"))
                return dict(duration=float(d["duration_h"]), infeasible=infeas)
    return None


def _oracle(stem: str, tag: str | None = None) -> dict | None:
    """Oracle cache -> duration (h), total/coupled charging time (h)."""
    names = ([f"solutions/oracle_{stem}.json"] if tag is None else
             [f"solutions/oracle_{stem}__{tag}.json",
              f"solutions/oracle_{stem}_{tag}.json"])
    for n in names:
        d = _load(n)
        if not (d and d.get("feasible")):
            continue
        sol = d.get("sol") or []
        if sol:
            ta_N = float(sol[-1]["ta"])
            tauc = sum(float(s.get("tauc") or 0.0) for s in sol)
            g    = sum(float(s.get("g")    or 0.0) for s in sol)
            return dict(duration=ta_N - T_START, tauc=tauc, g=g,
                        gap=float(d.get("gap") or 0.0))
        # cache recovered from a run log (see recover_variant_oracles.py):
        # the objective survives, the schedule does not — usable for duration
        # deltas, not for per-stop quantities like the coupling fraction.
        if d.get("obj") is not None:
            return dict(duration=float(d["obj"]) - T_START,
                        tauc=None, g=None,
                        gap=float(d.get("gap") or 0.0))
    return None


_INST_CACHE: dict[str, dict] = {}


def _instance(stem: str) -> dict:
    """Base instance data (geometry + overhead parameters), memoised.

    Diesel variants are verbatim copies, so the base file is the right source
    for both worlds; the diesel-side transform is applied by the caller.
    """
    if stem not in _INST_CACHE:
        _INST_CACHE[stem] = (_load(f"instances/{stem}.json") or {}).get(
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
    names = ([f"solutions/oracle_{stem}.json"] if tag is None else
             [f"solutions/oracle_{stem}__{tag}.json",
              f"solutions/oracle_{stem}_{tag}.json"])
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
        fig.savefig(os.path.join("figures", f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Figure    : figures/{name}.png|pdf")


def _write_csv(name, header, rows):
    path = os.path.join("figures", name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Stats CSV : {path}")


def _write_tex(name, text):
    os.makedirs("tables", exist_ok=True)
    path = os.path.join("tables", name)
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
    for route, cust in COMBOS:
        for tw in TWS:
            for seed in SEEDS:
                st       = _stem(route, cust, tw, seed)
                ev_o     = _oracle(st)
                di_o     = _oracle(st, "diesel")
                ev_g     = _greedy(st)
                di_g     = _greedy(st, "diesel")

                pen_o = pen_g = naive = coup = None
                if ev_o and di_o and di_o["duration"] > 0:
                    pen_o = 100 * (ev_o["duration"] / di_o["duration"] - 1)
                    # tauc/g are None for a log-recovered cache (no schedule)
                    if ev_o["tauc"] is not None:
                        naive = 100 * (ev_o["tauc"] / di_o["duration"])
                        coup  = (100 * ev_o["g"] / ev_o["tauc"]
                                 if ev_o["tauc"] > 1e-6 else None)
                if (ev_g and di_g and not ev_g["infeasible"]
                        and not di_g["infeasible"] and di_g["duration"] > 0):
                    pen_g = 100 * (ev_g["duration"] / di_g["duration"] - 1)

                detail.append([route, cust, tw, seed,
                               _fmt(di_o and di_o["duration"], ".3f", ""),
                               _fmt(ev_o and ev_o["duration"], ".3f", ""),
                               _fmt(pen_o, ".2f", ""), _fmt(pen_g, ".2f", ""),
                               _fmt(naive, ".2f", ""), _fmt(coup, ".1f", "")])
                d = per_class.setdefault(route, dict(pen_o=[], pen_g=[],
                                                     naive=[], coup=[],
                                                     dur_d=[], dur_e=[]))
                d["pen_o"].append(pen_o); d["pen_g"].append(pen_g)
                d["naive"].append(naive); d["coup"].append(coup)
                d["dur_d"].append(di_o and di_o["duration"])
                d["dur_e"].append(ev_o and ev_o["duration"])

    _write_csv("additional_diesel_stats.csv",
               ["route", "cust", "tw", "seed", "diesel_oracle_h",
                "ev_oracle_h", "pen_oracle_%", "pen_greedy_%",
                "naive_pen_%", "coupling_%"], detail)

    # ── figure: naive vs greedy vs oracle penalty, per route class ───────────
    routes = [r for r in ("short", "medium") if r in per_class]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    w, x = 0.26, np.arange(len(routes))
    series = [("naive (+ total charging time)", "naive", "#b0b0b0"),
              ("greedy EV vs greedy diesel",    "pen_g", BLUE),
              ("oracle EV vs oracle diesel",    "pen_o", INK)]
    for k, (lbl, key, col) in enumerate(series):
        vals = [_mean(per_class[r][key]) for r in routes]
        pos  = x + (k - 1) * w
        ax.bar(pos, [v or 0 for v in vals], w, color=col,
               edgecolor="white", label=lbl)
        for p, v in zip(pos, vals):
            if v is not None:
                ax.text(p, v + 0.25, f"{v:.1f}", ha="center", fontsize=7,
                        color=INK)
    for xi, r in enumerate(routes):
        c = _mean(per_class[r]["coup"])
        n = sum(1 for v in per_class[r]["pen_o"] if v is not None)
        ax.text(xi, -0.14, f"{_fmt(c, '.0f')}% coupled  (n={n})",
                ha="center", va="top", fontsize=6.5, color=MUT,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(x, [f"{r.capitalize()} route" for r in routes])
    ax.set_ylabel("Route duration vs diesel (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Electrification penalty: naive estimate vs optimized "
                 "schedules", loc="left")
    _save(fig, "additional_diesel_gap")

    # ── LaTeX table ──────────────────────────────────────────────────────────
    lines = [
        r"\begin{table}[ht]\centering",
        r"\caption{EV vs diesel route duration (\%), same instances and "
        r"realizations; naive = diesel + total EV charging time. "
        r"Coupling = share of charging time credited inside a mandatory "
        r"break ($\Sigma g_i/\Sigma\tau^c_i$, hindsight optimum).}",
        r"\label{tab:diesel}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Route & Diesel (h) & EV (h) & Naive (\%) & Greedy (\%) & "
        r"Oracle (\%) & Coupling (\%) \\",
        r"\hline",
    ]
    for r in routes:
        d = per_class[r]
        lines.append(
            f"{r.capitalize()} & {_fmt(_mean(d['dur_d']))} & "
            f"{_fmt(_mean(d['dur_e']))} & {_fmt(_mean(d['naive']))} & "
            f"{_fmt(_mean(d['pen_g']))} & {_fmt(_mean(d['pen_o']))} & "
            f"{_fmt(_mean(d['coup']), '.0f')} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_diesel.tex", "\n".join(lines))

    _diesel_decomposition(routes)
    _refuel_sensitivity(routes)


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


def _refuel_sensitivity(routes) -> None:
    """Bound the effect of the fuel stops the diesel model does not charge for.

    Reports, per route class, how many stops each tank spec implies and what
    crediting them would do to the reported penalty.  The headline table
    assumes zero, which is the conservative direction: a longer diesel
    makespan can only shrink the electrification penalty.
    """
    rows, note = [], {}
    for route in routes:
        km, pen = [], []
        for r, cust in COMBOS:
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
        base = _mean([100 * (e / d - 1) for e, d in pen])
        note[route] = (len(km), min(km), max(km))
        for tank, cons in _TANK_SPECS:
            rng   = _TANK_USABLE * tank / cons * 100
            stops = [max(0, int(np.ceil(t / rng)) - 1) for t in km]
            adj   = _mean([100 * (e / (d + s * _REFUEL_EVENT_H) - 1)
                           for (e, d), s in zip(pen, stops)])
            rows.append([route, tank, cons, f"{rng:.0f}",
                         f"{_mean(stops):.2f}",
                         f"{100 * np.mean([s > 0 for s in stops]):.0f}",
                         _fmt(base, ".2f"), _fmt(adj, ".2f"),
                         _fmt(adj - base if adj and base else None, ".2f")])

    if not rows:
        print("  Refuel note: pending (no EV/diesel oracle pairs yet)")
        return
    _write_csv("additional_diesel_refuel.csv",
               ["route", "tank_L", "cons_L_per_100km", "range_km",
                "mean_fuel_stops", "pct_needing_1plus", "penalty_%",
                "penalty_with_refuel_%", "delta_pp"], rows)

    worst = max(float(r[8]) for r in rows), min(float(r[8]) for r in rows)
    span  = " to ".join(f"{v:+.2f}" for v in sorted(worst))
    _write_tex("additional_diesel_refuel.tex", "\n".join([
        r"% one-sentence footnote for the diesel section",
        r"The diesel schedules charge no refuelling time.  Route lengths are "
        + "; ".join(rf"{r} {lo:.0f}--{hi:.0f}\,km ($n={n}$)"
                    for r, (n, lo, hi) in note.items())
        + r", against a tank range of "
        + rf"{_TANK_USABLE * _TANK_SPECS[-1][0] / _TANK_SPECS[-1][1] * 100:.0f}"
        + rf"--{_TANK_USABLE * _TANK_SPECS[0][0] / _TANK_SPECS[0][1] * 100:.0f}"
        + r"\,km, so no fuel stop is required on the short routes under any "
        r"specification and at most one on the medium routes.  Crediting the "
        rf"diesel with that stop ({_REFUEL_EVENT_H * 60:.0f}\,min: access "
        r"manoeuvre, queue and pumping) would change the reported penalty by "
        rf"{span}\,percentage points, so the tables assume none --- the "
        r"conservative direction.", ""]))


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
    for route, cust in COMBOS:
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
    _write_csv("additional_diesel_decomp.csv",
               ["component"] + [f"{r}_h" for r in routes],
               [[lbl] + [_fmt(_mean(per[r][k]), ".3f", "") for r in routes]
                for k, lbl in _DECOMP_LABEL]
               + [["Net penalty vs diesel"]
                  + [_fmt(_mean(per[r]["_gap"]), ".3f", "") for r in routes]])

    n = min(len(per[r]["_gap"]) for r in routes)
    tex = [
        r"\begin{table}[ht]\centering",
        r"\caption{Where the electrification penalty goes: mean per-instance "
        r"difference (h) between the hindsight-optimal electric and diesel "
        r"schedules, decomposed into the terms of the departure equations. "
        r"Driving is identical within each pair, so the rows sum to the net "
        r"penalty exactly. Both vehicles pay the same access manoeuvre to "
        r"pull off for a mandatory break, so only the EV's charge-only stops "
        r"survive that row. $n \geq " + str(n) + r"$ per class.}",
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
    tex += [r"\hline",
            r"\textbf{Net penalty vs.\ diesel} & "
            + " & ".join(rf"$\mathbf{{{_mean(per[r]['_gap']):+.2f}}}$"
                         for r in routes) + r" \\",
            r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_diesel_decomp.tex", "\n".join(tex))


# ══════════════════════════════════════════════════════════════════════════════
# §8.3 — SENSITIVITY (one-at-a-time)
# ══════════════════════════════════════════════════════════════════════════════

# (axis label, variant tag, planned?) — extend as new sweeps land
_SENS_ROWS = [
    ("CS spacing 30 km",        "cs30",   True),
    ("CS spacing 90 km",        "cs90",   True),
    ("Charger power 150 kW",    "kw150",  True),
    ("Charger power 350 kW",    "kw350",  True),
    ("Charger power 1000 kW",   "kw1000", True),
    ("No split break",          "nosplit", True),
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
        # per-route-class deltas (the figure) and pooled (the table)
        dg = {r: [] for r in _ROUTE_SPLIT}
        do = {r: [] for r in _ROUTE_SPLIT}
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
                    bo, vo = _oracle(st), _oracle(st, tag)
                    if bo and vo and bo["duration"] > 0:
                        do[route].append(
                            100 * (vo["duration"] / bo["duration"] - 1))

        all_g = [v for r in _ROUTE_SPLIT for v in dg[r]]
        all_o = [v for r in _ROUTE_SPLIT for v in do[r]]
        n_g, n_o = len(all_g), len(all_o)
        status = ("pending (needs code)" if not planned and n_g == 0 else
                  "pending" if n_g == 0 else
                  f"greedy n={n_g}" + (f", oracle n={n_o}" if n_o else
                                       ", oracle pending"))
        rows_out.append([label, tag,
                         _fmt(_mean(all_g), ".2f", ""), n_g,
                         _fmt(_mean(all_o), ".2f", ""), n_o,
                         _fmt(_mean(do["short"]), ".2f", ""), len(do["short"]),
                         _fmt(_mean(do["medium"]), ".2f", ""),
                         len(do["medium"]), status])
        fig_rows.append((label, planned,
                         {r: (_mean(do[r]), len(do[r]), _mean(dg[r]),
                              len(dg[r])) for r in _ROUTE_SPLIT}))

    _write_csv("additional_sens_stats.csv",
               ["axis", "tag", "greedy_delta_%", "n_greedy",
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
    series = [(m, r, off_i) for m, off_i in (("oracle", 0), ("greedy", 2))
              for r in _ROUTE_SPLIT]
    h = 0.19
    vals = [v for _l, _p, per in fig_rows for st in per.values()
            for v in (st[0], st[2]) if v is not None]

    fig, ax = plt.subplots(figsize=(6.6, 0.72 * len(fig_rows) + 1.5))
    drawn_m, drawn_r = set(), set()

    for yi, (label, planned, per) in zip(y, fig_rows):
        any_here = False
        for k, (meth, route, off_i) in enumerate(series):
            mean_v = per[route][off_i]
            if mean_v is None:
                continue
            any_here = True
            drawn_m.add(meth)
            drawn_r.add(route)
            col = ps.METHOD_COLOR[meth]
            face = ps.tint(col, 0.45) if route == "short" else col
            off = (1.5 - k) * h
            ax.barh(yi + off, mean_v, height=h, color=face,
                    edgecolor=col, linewidth=0.5)
            ax.text(mean_v + np.sign(mean_v) * 0.18, yi + off, f"{mean_v:+.1f}",
                    ha="left" if mean_v >= 0 else "right", va="center",
                    fontsize=6, color=INK)
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
    for meth, route, _off in series:
        if meth not in drawn_m or route not in drawn_r:
            continue
        col = ps.METHOD_COLOR[meth]
        handles.append(plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=ps.tint(col, 0.45) if route == "short" else col,
            edgecolor=col, linewidth=0.5))
        labels.append(f"{ps.METHOD_LBL[meth]} · {route}")
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=7,
                  loc="lower left", ncol=2,
                  title="hindsight optimum vs myopic policy",
                  title_fontsize=7, alignment="left")
    ax.set_title("Sensitivity of route duration to charging infrastructure",
                 loc="left")
    _save(fig, "additional_sens_effects")

    lines = [
        r"\begin{table}[ht]\centering",
        r"\caption{One-at-a-time sensitivity: mean change in route duration "
        r"vs the base case (\%). Preliminary cells use greedy; final values "
        r"use the hindsight optimum.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{lrrrr}", r"\hline",
        r"Axis & Greedy $\Delta$ (\%) & Oracle $\Delta$ (\%) "
        r"& Short & Medium \\",
        r"\hline",
    ]
    for (label, _tag, g, _ng, o, _no, o_s, _ns, o_m, _nm,
         _status) in rows_out:
        lines.append(f"{label} & {g or '--'} & {o or '--'} & "
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
                    f = (_latest(f"solutions/{st}__g{gam}_ROBU_*.json")
                         or _latest(f"solutions/{st}_g{gam}_ROBU_*.json"))
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
        with open("figures/paper_gap_stats.csv", encoding="utf-8") as fh:
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
    for f in glob.glob("results_vss/*_vss.json"):
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
              "'python additional_analysis.py vss')")


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = dict(diesel=section_diesel, sensitivity=section_sensitivity,
                 gamma=section_gamma, vss=section_vss)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real tables/figures for the "
                                             "additional analyses (8.3-8.5)")
    ap.add_argument("--section", default="all",
                    choices=["all", *_SECTIONS])
    args = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    for name, fn in _SECTIONS.items():
        if args.section in ("all", name):
            fn()
