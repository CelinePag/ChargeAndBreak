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
  figures/additional_la_all.png|pdf          §8.3 whole LA study on one plane
                                             (horizon + scenarios + policy)
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
import bisect
import fnmatch
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as _pe
import matplotlib.ticker as mticker
import numpy as np

from src.settings import (T_START, BETA_TW,
                          BATTERY_CAPACITY, CHARGER_POWER_BASE_KW)

# Shared palette + chrome (see paper_style.py): colour follows the entity, so
# Greedy is the same blue here as in the base-case figures.
from src.plot import paper_style as ps
from src import paths as _paths
from src.output_analysis import run_cache

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


_DIR_INDEX: dict[str, list[str]] = {}


def _dir_index(directory: str) -> list[str]:
    """Sorted entry names of ``directory``, scanned once per process.

    These are read-only reporting runs, so the listing cannot change underneath
    us; call ``_DIR_INDEX.clear()`` if that ever stops holding.
    """
    names = _DIR_INDEX.get(directory)
    if names is None:
        try:
            names = sorted(os.listdir(directory or "."))
        except OSError:
            names = []
        _DIR_INDEX[directory] = names
    return names


def _latest(pattern: str) -> str | None:
    """Newest file matching ``pattern`` (lexicographic, i.e. by run timestamp).

    NOT glob.glob: every call re-scanned the whole directory, and solutions/ now
    holds >10k files at ~54 ms a scan.  §8.3 alone makes ~14 400 of these calls
    (8 tags x 300 instances x greedy/LA x base/variant x two tag spellings), so
    the section spent ~13 minutes listing the same directory over and over while
    the JSON parsing it fed cost 7 seconds.

    The patterns all look like ``<stem>[__<tag>]_<ALG>_*.json`` — a literal
    prefix, then wildcards — so the sorted listing is bisected down to the
    matching prefix block and only that block is fnmatched.  Sorting names
    within one directory ranks them exactly as sorting full paths did, so the
    chosen file is unchanged.
    """
    directory, pat = os.path.split(pattern)
    names = _dir_index(directory)
    star = min((i for i in (pat.find("*"), pat.find("?"), pat.find("["))
                if i != -1), default=-1)
    if star == -1:                       # no wildcard: an exact name
        block = [pat] if pat in _DIR_INDEX_SET(directory) else []
    else:
        prefix = pat[:star]
        lo = bisect.bisect_left(names, prefix)
        hi = bisect.bisect_left(names, prefix + "￿")
        block = [n for n in names[lo:hi] if fnmatch.fnmatchcase(n, pat)]
    return os.path.join(directory, block[-1]) if block else None


_DIR_SETS: dict[str, set[str]] = {}


def _DIR_INDEX_SET(directory: str) -> set[str]:
    s = _DIR_SETS.get(directory)
    if s is None:
        s = _DIR_SETS[directory] = set(_dir_index(directory))
    return s


def _load(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


_RUNS: dict | None = None


def _RUNS_BY_NAME() -> dict[str, dict]:
    """Stripped run records keyed by file name, loaded on first use."""
    global _RUNS
    if _RUNS is None:
        _RUNS = run_cache.runs_by_name(_paths.solutions())
    return _RUNS


_ORACLE_JSON: dict[str, dict | None] = {}


def _load_oracle(path: str) -> dict | None:
    """``_load`` for oracle caches, memoised per process.

    Unlike the run files, an oracle's per-stop schedule IS needed here (§8.4
    decomposes the dwell, §8.3 reads tauc/g), so these cannot come from
    run_cache's stripped records.  They are, however, asked for repeatedly —
    _oracle() and _oracle_dwell() want the same file, and the diesel section
    walks the same instances several times — and a cache is ~70 KB, so parsing
    each one once turns the section's oracle I/O into a single pass.
    """
    if path not in _ORACLE_JSON:
        _ORACLE_JSON[path] = _load(path)
    return _ORACLE_JSON[path]


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
            # duration_h + metrics only, so the stripped run_cache record is
            # enough and the trajectory arrays are never read off disk.
            d = _RUNS_BY_NAME().get(os.path.basename(f))
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
        d = _load_oracle(n)
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
    #
    # Chrome follows the same grammar as §8.3 (additional_sens_effects) and the
    # base-case box plots: default frame, entity colour with a darker hairline
    # edge, major+minor grid on the response axis, one in-axes legend, and NO
    # title of any kind — the LaTeX \caption carries it (see results_section).
    routes = [r for r in DIESEL_ROUTES if r in per_class]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    w, x = 0.32, np.arange(len(routes), dtype=float)
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
               edgecolor=ps.shade(col, 0.35), linewidth=0.4, zorder=3,
               label=lbl)
        for p, v, h in zip(pos, vals, hrs):
            if v is None:
                continue
            ax.annotate(f"{v:.1f}%", (p, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom",
                        fontsize=7.5, color=INK, zorder=4)
            if h is not None:
                ax.annotate(f"{h:+.1f} h", (p, v), textcoords="offset points",
                            xytext=(0, -6), ha="center", va="top",
                            fontsize=6.5, color="white", zorder=4)
    # Coupling share only.  The sample size is not annotated here: it is the
    # same n for every bar in the base sweep, and where a class IS a partial
    # average the table caption names it (coverage is printed above and kept
    # per instance in the CSV), so the figure does not repeat it.
    span = ax.get_xaxis_transform()   # x in data coords, y in axes fraction
    for xi, r in enumerate(routes):
        c = _mean(per_class[r]["coup"])
        have = sum(1 for v in per_class[r]["pen_o"] if v is not None)
        note = f"{_fmt(c, '.0f')}% coupled" if have else "greedy only"
        # Fixed point offset, not an axes fraction, so the note clears the tick
        # labels by the same margin whatever the figure height ends up being.
        ax.annotate(note, xy=(float(xi), 0), xycoords=span,
                    xytext=(0, -18), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color=MUT)
    ax.set_xticks(x, [ps.ROUTE_LBL[r] for r in routes])
    ax.set_xlim(-0.6, len(routes) - 0.4)
    ax.set_ylabel("Route duration vs. diesel (%)")
    # Major + minor grid on the response axis only, as in §8.3: the reader
    # compares bar heights across groups, and the minor lines make the ~1 pp
    # differences between the two series legible.
    ax.yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.grid(True, which="major", color=GRID, lw=0.6)
    ax.yaxis.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.6)
    ax.tick_params(axis="y", which="minor", length=2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, colors=INK)
    # Headroom so the legend row clears the tallest bar's value label.
    _top = max((v for s in series for v in
                (_mean(per_class[r][s[1]]) for r in routes)
                if v is not None), default=1.0)
    ax.set_ylim(0, _top * 1.32)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white", fontsize=7.5, ncol=2, loc="upper center",
              handlelength=1.1, handletextpad=0.4, columnspacing=1.4)
    # Bottom reserve: the coupling notes hang below the axes and tight_layout
    # does not measure annotations drawn outside them.
    fig.tight_layout(rect=(0, 0.06, 1, 1))
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
    # The sweep is three distinct experiments (charger spacing / charger power /
    # battery capacity) and the levels are only comparable within one.  That
    # grouping used to be carried by SPACING alone, which cost ~0.85 of a column
    # per boundary and, with three blocks, stretched the figure to a 3.5:1 band
    # that had to be shrunk past legibility to fit \linewidth.  The gap is now
    # only wide enough to read as a break, and the separating work is done by an
    # alternating background band instead — which costs no width at all.
    _grp = [_sens_group(t) for _l, t, _p in _SENS_ROWS]
    bounds = [i for i in range(1, len(_grp)) if _grp[i] != _grp[i - 1]]
    _BLOCK_GAP = 0.30
    _BAND = "#f4f4f4"
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

    # Fixed page geometry, not derived from the column count.  The figure is
    # printed at \linewidth, so a width that grows with the number of levels is
    # a width that shrinks the type: the previous 13.8 in band rendered its
    # 7.5 pt ticks at about 3.5 pt on the page.  Sizing to the text block makes
    # the drawn point size the printed point size.
    _FIG_W, _FIG_H = 7.0, 4.3
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    drawn_m, drawn_r = set(), set()

    # Alternating band per experiment, drawn under the grid.  This replaces the
    # width the old block gap spent on the same job.
    half = 0.5 + _BLOCK_GAP / 2
    for bi, (s, e) in enumerate(zip([0] + bounds, bounds + [len(_grp)])):
        if bi % 2:
            ax.axvspan(x[s] - half, x[e - 1] + half,
                       facecolor=_BAND, edgecolor="none", zorder=0)

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
            # No per-bar value label.  Nine series x eight levels is 72 labels;
            # at \linewidth the bar pitch is ~5 pt, so they only fit rotated at
            # a size nobody can read, and they were the reason the figure had to
            # be drawn three times too wide.  The values live in
            # tex/tables/additional_sensitivity.tex, which the results section
            # now includes alongside this figure.
            ax.bar(xi + off, mean_v, width=w, color=face,
                   edgecolor=col, linewidth=0.35)
        if not any_here:
            note = "pending" if planned else "pending (needs a model flag)"
            ax.text(xi, 0.2, note, ha="center", va="bottom", rotation=90,
                    fontsize=7, color=MUT, style="italic")

    ax.axhline(0, color=INK, lw=0.9)
    lo = min(-2.0, (min(vals) if vals else 0) - 2.0)
    hi = max(3.0, (max(vals) if vals else 0) + 2.0)
    # Extra headroom above the tallest bar so the in-axes legend sits over empty
    # space instead of over data.  Less is needed now that no rotated value
    # labels stand on top of the bars.
    ax.set_ylim(lo, hi + 0.17 * (hi - lo))
    # Exactly the band extent, so the alternating blocks reach the axes edges
    # instead of stopping short of them.
    ax.set_xlim(x[0] - half, x[-1] + half)
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

    # One label per block, centred under its levels.  The band above already
    # separates the blocks; this names them.
    span = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for s, e in zip([0] + bounds, bounds + [len(_grp)]):
        # Fixed point offset (not an axes fraction) so the block label clears
        # the tick labels by the same margin whatever the figure height is.
        ax.annotate(_grp[s], xy=(float(x[s:e].mean()), 0), xycoords=span,
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=INK)

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
        ax.legend(handles, labels, frameon=True, framealpha=0.92,
                  edgecolor="none", facecolor="white", fontsize=7.5,
                  loc="upper center", ncol=min(3, len(handles)),
                  handlelength=1.1, handletextpad=0.4, columnspacing=1.4)
    # Bottom reserve: the block labels hang below the axes and tight_layout
    # does not measure annotations drawn outside them.
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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


def _la_stats(name: str = "additional_la_stats.csv") -> dict:
    """data_output/<name> -> {(cfg, route, tw): row}.

    Written by `additional_analysis.py la-report`.  Absent or partial file is
    normal while the sweep is running: missing cells render as "pending".
    """
    out: dict = {}
    path = _paths.data_output(name)
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
# §8.3 — THE WHOLE LOOK-AHEAD STUDY ON ONE PLANE
# ══════════════════════════════════════════════════════════════════════════════
# section_la and section_la_policy split the sweep because the two ladders have
# a position on an axis and the policy variants do not.  That split is right for
# a RESPONSE plot (x = |Xi| or x = L), but it hides the only question the reader
# actually has: given a per-decision compute budget, where does the next second
# go — horizon, scenarios, or a MIP tail?  On the cost/quality plane every cell
# has a position, ladder or not, so all three analyses collapse into one figure.
#
# Encoding, four channels on one set of axes:
#   x  decision time per CS stop        — what the configuration costs
#   y  gap to the hindsight optimum     — what it buys
#   line colour + marker  which analysis the cell belongs to (horizon ladder,
#                                        scenario ladder, policy variant)
#   marker FILL  infeasibility rate     — the correction the level needs
# The fill is not decoration: the gap median is taken over FEASIBLE runs only,
# so a configuration that strands the truck more often can post a better median
# for exactly the wrong reason (S25H12 strands up to 22%).  The ramp is the
# module-wide _INFEAS_CMAP (bluish green -> yellow -> vermillion), the same one
# the base-case box figure and both other LA figures use, so a reader who has
# learned "red = strands the truck" once carries it across every figure.  It
# does collide with the horizon ladder's green at the clean end; the marker EDGE
# keeps series identity and the fill is read against the colourbar, not against
# the line palette.
#
# Window classes are POOLED (the window_class = 'all' rows that
# additional_analysis.py's la-report writes from the raw runs).  Splitting them
# doubled the panel count to say something this figure is not asking: the
# configuration ranking is the same under both classes, and the level shift
# between them belongs to section_la's response plots.
#
# Series colour is deliberately OUTSIDE the method palette.  Everything drawn
# here is one method (LA), so a method hue would be a false cue: green would
# claim the horizon ladder is "the LA one" and blue would read as Greedy to
# anyone who has seen the base-case figures.  What varies across these series is
# the CONFIGURATION, which has no entry in paper_style, so the three take hues
# no method owns.
#
# The choice is constrained from two sides: clear of every method hue (blue,
# vermillion, orange, green, reddish purple, grey) AND clear of the
# green/yellow/vermillion the marker fills run through, which rules out the
# warm half of the wheel for a line that has to stay visible as a thin ring
# around a filled marker.  Violet and magenta are what survive, and they are
# separated by lightness as well as hue.
_LA_HUE_HORIZON = "#6A3D9A"    # violet   — horizon ladder L
_LA_HUE_SCEN    = "#C2007A"    # magenta  — scenario ladder |Xi|
# Every non-ladder cell shares ONE ink.  Splitting them grey-for-LP against
# charcoal-for-MILP was tried and abandoned twice: a grey light enough to read
# as "not black" is too light to read at all at a 1.4 pt marker edge, and a grey
# dark enough to see is no longer distinguishable from the charcoal it is
# supposed to contrast with.  Lightness is simply not a wide enough channel here,
# so the solver and the regime both move to the MARKER instead, where four
# shapes separate cleanly, and the ink stays at full strength for all of them.
_LA_HUE_CELL    = "#1A1A1A"
_LA_POLICY_MARK = {"MIPTAIL": "D", "TB0": "v"}

# ── the non-ladder cells ─────────────────────────────────────────────────────
# Everything that is not a rung on the horizon or scenario ladder, declared in
# one place so a cell that has not been run yet still has a name and a legend
# entry, and appears as a labelled absence rather than vanishing.
#
# The MARKER carries both distinctions among the LA cells: round/diamond for the
# split break as Art. 7 permits it, triangles for the regime that forbids it;
# circle/triangle-up for the LP subproblem, diamond/triangle-down for the MILP.
#
# Only LA configurations are plotted.  Other METHODS are deliberately absent:
# this figure answers "how should the look-ahead be configured", and a method
# that is not a configuration of it has no position on either ladder and no
# leader to the base cell.  Greedy in particular was tried here and removed —
# it lands at 0.0 s hard against the left spine, in the busiest corner of every
# panel.  la-report still writes its row, so the section text can quote it.
#
# Every LA variant is drawn as a displacement from the BASE cell — one leader
# each, all sharing a tail, so the panel reads as a fan out of the reference
# rather than a chain.  That reading is honest on the y-axis because la-report
# scores the no-split cells against the BASE-CASE oracle rather than the one
# that was itself denied the split break (see _regap_regimes_to_base_oracle):
# every point is a distance to the same unrestricted optimum, and a regime that
# removes an option can only move away from it.
#
# `above` sends the tag over or under its marker.  It is declared rather than
# derived because the two regimes land on top of each other by construction:
# the whole point of the no-split pair is that it sits beside the split pair, so
# their tags would collide on every panel unless the members of each pair are
# pushed to opposite sides once, here, instead of being nudged per panel.
#
# The CSV keys these by the launcher's tags (MIPTAIL, NOSPLIT, and the crossed
# MIPTAIL+NOSPLIT written by la-report); the reader sees the model names.  The
# base cells drop a ($|\Xi| = 25$, $L = 24$ h) qualifier: both ladder entries
# already state that centre point, so repeating it three times padded the legend
# without adding anything.
_LA_ALL_CELLS = [
    # cfg key         panel tag        legend label            mark colour   above leader
    ("base",            "base",          r"base case",
     "o", _LA_HUE_CELL, True,  False),
    ("MIPTAIL",         "MILP",          r"MILP subpr.",
     "D", _LA_HUE_CELL, True,  True),
    ("NOSPLIT",         "LP no-split",   r"base case, no split break",
     "^", _LA_HUE_CELL, True,  True),
    ("MIPTAIL+NOSPLIT", "MILP no-split", r"MILP subpr., no split break",
     "v", _LA_HUE_CELL, False, True),
]
_LA_ALL_BY_CFG = {c[0]: c for c in _LA_ALL_CELLS}
# Configs the CSV may carry that must never become a cell here: other methods,
# ad-hoc timing probes, and variants that were never launched.  Anything else
# unrecognised still draws, so a genuinely new cell is not silently dropped.
_LA_ALL_SKIP = {"GREEDY", "TB0"}
# Prefix-matched, for the same reason la-report matches variants that way: the
# LOCAL family arrives under several spellings (LOCAL, LOCAL_MIPTAIL, ...) and
# an exact-match list lets each new one through as a spurious cell.
_LA_ALL_SKIP_PREFIX = ("LOCAL",)


def section_la_all(csv_name: str = "additional_la_stats.csv",
                   cell_decl: list | None = None,
                   ladders: bool = True,
                   anchor: str = "base",
                   outname: str = "additional_la_all",
                   banner: str = "combined cost/quality plane",
                   min_n: int = 5):
    """§8.3 — one figure for the entire LA study (horizon, scenarios, policy).

    Reads the same data_output/additional_la_stats.csv as section_la and
    section_la_policy; it replaces neither, it re-reads both onto shared axes.
    Needs the pooled window_class = 'all' rows, so re-run
    `additional_analysis.py la-report` if the CSV predates them.

    Parameterised so a second experiment can reuse the whole plane rather than
    fork it: section_la_local draws the cab-hardware runs with the same axes,
    encoding and pending conventions, differing only in which CSV it reads, its
    cell list, the absence of the two ladders, and which cell is the anchor.
    """
    print(f"== Sec 8.3 look-ahead — {banner} ==")
    cells_all = cell_decl or _LA_ALL_CELLS
    stats = _la_stats(csv_name)
    if not stats:
        print("  additional_la_stats.csv missing — nothing drawn")
        return
    routes = ps.ROUTE_ORDER
    TW = "all"                       # pooled over window classes — see above
    if stats and not any(t == TW for (_c, _r, t) in stats):
        print("  additional_la_stats.csv has no pooled 'all' window rows — "
              "re-run: python -m src.output_analysis.additional_analysis "
              "la-report")
        return

    # The non-ladder cells: the declared ones first, in the order they are meant
    # to be read, then anything else the CSV happens to carry so a variant run
    # under a tag nobody added here still shows up.  TB0 stays excluded by name
    # — it is the one variant whose runs were never launched, and a permanent
    # "pending" note is noise, not information.  section_la_policy still has it.
    have = {c for (c, _r, _t) in stats}
    by_cfg = {c[0]: c for c in cells_all}
    extra = sorted(c for c in have
                   if c not in by_cfg and c not in _LA_ALL_SKIP
                   and not c.upper().startswith(_LA_ALL_SKIP_PREFIX)
                   and not re.fullmatch(r"S\d+H\d+(\.\d+)?", c))
    cells = ([c for c in cells_all if c[0] != anchor]
             + [(c, c, c, "*", _LA_HUE_CELL, True, True) for c in extra])

    # Below this a "median" is not an estimate of anything, and one such cell
    # can wreck the figure for every other: a 2-run long-route MILP no-split
    # cell landed at 13.8 % mid-sweep and stretched the shared y-axis to 15 %,
    # compressing the 2-6 % band where the entire study lives.  Such a cell is
    # reported as still pending, with its count, rather than plotted.  The
    # threshold is a parameter because it is a judgement about the experiment,
    # not a constant: the cab-hardware runs are a handful of instances BY
    # DESIGN, so there the same floor would hide the entire figure.
    _MIN_N = min_n

    def cell(cfg, route, tw=TW):
        """(cost, quality, infeasibility rate) for one cell, or None."""
        row = stats.get((cfg, route, tw))
        c = _la_num(row, "decision_cs_mean_s_median")
        q = _la_num(row, "gap_pen_median_pct")
        if c is None or q is None:
            return None
        n, i = _la_num(row, "n_runs"), _la_num(row, "n_infeasible")
        if (n or 0) < _MIN_N:
            return None
        return c, q, ((i / n) if (n and i is not None) else None)

    def n_runs(cfg, route, tw=TW):
        return _la_num(stats.get((cfg, route, tw)), "n_runs") or 0

    # A cell still filling up is not just noisier than the base cell, it is a
    # different POPULATION: the sweeps land combo by combo, so a half-finished
    # cell is typically all-Tnone or all-Cfew while the base cell it is read
    # against is pooled over every combo and window.  Tnone is the easy half, so
    # such a cell can plot BELOW the base cell while being WORSE than it on the
    # instances they share — which is exactly what the no-split arms did at
    # ~12 % coverage.  Rather than let the figure state that, an under-covered
    # cell carries its run count in its tag, so the reader sees the sample it is
    # looking at and the mark disappears by itself once the runs land.
    _PARTIAL_AT = 0.60

    def partial(cfg, route, tw=TW):
        base_n = n_runs(anchor, route, tw)
        n = n_runs(cfg, route, tw)
        return bool(base_n and n and n < _PARTIAL_AT * base_n)

    # A point is (tag, is_base, colour, marker, below, cost, gap, infeas).
    # The base cell is the last element of no ladder and the middle of two — it
    # is what both pivot on, so it is skipped inside the ladders and drawn once,
    # on top, together with the variants that fan out of it.
    def paths(route):
        out = []
        for ladder, is_h, col, pfx in ((
                (_LA_HORIZONS, True,  _LA_HUE_HORIZON, "L"),
                (_LA_SCENARIOS, False, _LA_HUE_SCEN, "S")) if ladders else ()):
            pts = []
            for v in ladder:
                cfg = (anchor if (v == _LA_BASE[1] if is_h else v == _LA_BASE[0])
                       else _la_cfg(_LA_BASE[0], float(v)) if is_h
                       else _la_cfg(int(v), _LA_BASE[1]))
                got = cell(cfg, route)
                if got:
                    # Scenario rungs label below, horizon rungs above: on the
                    # long routes the two ladders end within a marker's width of
                    # each other, and the vertical split is what stops those
                    # tags overprinting.
                    pts.append((f"{pfx}{v:g}", cfg == anchor, col,
                                "o" if is_h else "s", not is_h, *got))
            if pts:
                out.append(("ladder", None, pts))
        for cfg, tag, _lbl, mk, col, above, leader in cells:
            got = cell(cfg, route)
            if got:
                # Nothing printed on the point at all: the legend already binds
                # marker to cell, and the ladders are the only series whose
                # labels carry a VALUE the legend cannot state — it names the
                # ladder, not which rung is which.  Run counts belong in the
                # table, not scattered over the panels.
                out.append(("cell", anchor if leader else None,
                            [("", False, col, mk, not above, *got)]))
        return out

    allpts = [p for r in routes for _k, _a, pts in paths(r) for p in pts]
    # The anchor sits in no series — it is drawn on its own — so it has to be
    # added by hand here or it takes no part in the scaling.  In the sweep
    # figure that was invisible because the base cell is also a rung on both
    # ladders; strip the ladders and the anchor is the ONLY point, and the axes
    # would be computed from an empty list and place it off-panel.
    anch = [a for a in (cell(anchor, r) for r in routes) if a]
    fmax = max([p[7] for p in allpts if p[7]]
               + [a[2] for a in anch if a[2]] or [1.0])
    # Room for the tags, which are drawn OUTSIDE the data range: the costliest
    # cell (MIPTAIL on the medium routes, ~95 s) is a wide label anchored to the
    # right of its marker, and autoscale would clip it against the spine.
    # The axis is LINEAR: a log axis puts equal ratios at equal distances, which
    # flatters the cheap end and makes the base->MIPTAIL leader look like a
    # gentle slope instead of the 5-20x jump in compute it actually is.  Cost
    # here is a budget the reader spends in seconds, not a scale-free quantity,
    # so distance on the page should be seconds.
    # The left margin is negative on purpose: the cheapest rung of each ladder
    # labels to the LEFT of its marker, and on a linear axis that rung sits
    # within a few seconds of the origin, so a hard zero clips the tag.
    costs = [p[5] for p in allpts] + [a[0] for a in anch] or [1.0, 10.0]
    xlim  = (-0.055 * max(costs), max(costs) * 1.16)
    # Same reason on y: the tallest rung (L12 on the medium routes) carries its
    # tag ABOVE the marker, and matplotlib's autoscale margin is not deep enough
    # to hold it under the spine.
    quals = [p[6] for p in allpts] + [a[1] for a in anch] or [0.0, 1.0]
    # Floor on the span: early in an experiment one or two cells can sit within
    # a tenth of a point of each other, and an axis autoscaled to that reports
    # a rounding difference as if it were the finding.
    span  = max(max(quals) - min(quals), 2.0)
    mid   = 0.5 * (max(quals) + min(quals))
    ylim  = (mid - 0.66 * span, mid + 0.66 * span)

    fig, axs = plt.subplots(1, len(routes), figsize=(7.2, 2.9),
                            sharex=True, sharey=True)
    axs = np.atleast_1d(axs)

    # Tags are drawn outside their marker, so a cell close to either spine has
    # to lean inwards or it prints off the panel.  These two bands are where
    # that happens; everything between them keeps the side it asked for.
    _lo = xlim[0] + 0.10 * (xlim[1] - xlim[0])
    _hi = xlim[0] + 0.80 * (xlim[1] - xlim[0])

    for ri, route in enumerate(routes):
        ax = axs[ri]
        got = paths(route)
        base = cell(anchor, route)
        # A cell that exists but is under _MIN_N is reported here WITH its count,
        # so "too few runs to plot" is visibly different from "not launched".
        missing = [f"{tag or lbl} (n={n_runs(cfg, route):.0f})"
                   if n_runs(cfg, route) else (tag or lbl)
                   for cfg, tag, lbl, _m, _c, _b, _ld in cells
                   if not cell(cfg, route)]

        for kind, anchor, pts in got:
            col = pts[0][2]
            if kind == "ladder" and len(pts) > 1:
                ax.plot([p[5] for p in pts], [p[6] for p in pts], "-",
                        color=col, lw=1.2, zorder=2)
            elif kind == "cell" and anchor:
                # A variant is a DISPLACEMENT from its regime's LP arm, not a
                # rung on a ladder, so it is joined to that arm by a dotted
                # leader which cannot be mistaken for a sampled path.  Drawn
                # only when the arm exists: without it the displacement has no
                # meaning and a line to the wrong regime would invent one.
                a = cell(anchor, route)
                if a:
                    ax.plot([a[0], pts[0][5]], [a[1], pts[0][6]], ":",
                            color=col, lw=1.0, zorder=2)
            for tag, is_base, col, mk, below, c, q, f in pts:
                if is_base:
                    continue                     # drawn once, below
                ax.plot(c, q, mk, ms=5.6,
                        mfc=_INFEAS_CMAP(min(1.0, (f or 0.0) / fmax)),
                        mec=col, mew=1.4, zorder=4)
                # Two rules keep the tags apart without hand-placing them per
                # panel: horizontally each tag hugs the side its cell sits on
                # relative to base, and vertically it goes to the side its
                # series declared (`below`, set where the series is built).
                # Greedy sits at 0.0 s, hard against the left spine, so the
                # "cheaper than base -> label on the left" rule would push its
                # tag off the panel; inside either edge band the tag leans
                # inwards regardless of which side of base the cell is on.
                right = c < _lo or ((c >= (base[0] if base else c))
                                    and c <= _hi)
                # White stroke under the glyphs: S10 sits a hair below the base
                # cell on the short routes, which is exactly where the
                # base->MIPTAIL leader leaves the anchor, and a tag printed
                # straight onto a dotted line is unreadable.
                if not tag:
                    continue                     # keyed by the legend alone
                ax.annotate(tag, (c, q),
                            xytext=(4 if right else -4, -8 if below else 7),
                            textcoords="offset points",
                            ha="left" if right else "right",
                            va="top" if below else "bottom",
                            fontsize=5.4, color=ps.shade(col, 0.25),
                            zorder=5,
                            path_effects=[_pe.withStroke(linewidth=1.7,
                                                         foreground="white")])
        # The base cell last and on top: it is the reference every leader fans
        # out of, and it is the one point two ladders and every variant all
        # touch, so it must not be overdrawn by whichever series lands last.
        # It carries no tag — the legend names it, and a label on the busiest
        # point in the panel is where clutter starts.
        if base:
            c, q, f = base
            ax.plot(c, q, by_cfg.get(anchor, (None, None, None, "o"))[3], ms=5.6,
                    mfc=_INFEAS_CMAP(min(1.0, (f or 0.0) / fmax)),
                    mec=_LA_HUE_CELL, mew=1.4, zorder=6)
        if missing:
            # Top-right, not bottom-right: the policy variants land in the
            # cheap-and-good corner, so a note down there sits on the very
            # markers it is describing the absence of.
            ax.text(0.985, 0.965, "pending: " + ", ".join(missing),
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=5.8, color=MUT, style="italic")
        if not got and not base:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    fontsize=7.5, color=MUT, style="italic",
                    transform=ax.transAxes)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.6)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_axisbelow(True)
        ax.tick_params(which="minor", length=2)
        ax.set_title(ps.ROUTE_LBL[route], loc="left")
        ax.set_xlabel("Decision time per CS stop (s)")
        if ri == 0:
            ax.set_ylabel("Gap to hindsight optimum (%)")

    handles = [plt.Line2D([], [], color=_LA_HUE_HORIZON, lw=1.3, marker="o",
                          ms=5.0, mfc="white", mec=_LA_HUE_HORIZON, mew=1.1),
               plt.Line2D([], [], color=_LA_HUE_SCEN, lw=1.3, marker="s",
                          ms=5.0, mfc="white", mec=_LA_HUE_SCEN, mew=1.1)]
    labels  = [r"horizon ladder $L$ (h), $|\Xi| = 25$",
               r"scenario ladder $|\Xi|$, $L = 24$ h"]
    if not ladders:
        handles, labels = [], []
    # Every declared cell is listed whether or not it has runs yet, so the
    # legend states the intended design and the panels say how much of it has
    # landed.  The base cell shows as a bare marker, variants with their leader.
    for _cfg, _tag, lbl, mk, col, _ab, leader in cells_all:
        handles.append(plt.Line2D(
            # Cells without a leader key as a bare marker.  ls must be "none"
            # and not ":" at lw=0 — matplotlib rejects a dash pattern whose
            # segments are all zero-length.
            [], [], color=col, lw=1.0 if leader else 0,
            ls=":" if leader else "none",
            marker=mk, ms=5.2, mfc="white", mec=col, mew=1.4))
        labels.append(lbl)
    for cfg in extra:
        handles.append(plt.Line2D([], [], color=_LA_HUE_CELL, lw=1.0, ls=":",
                                  marker="*", ms=5.2, mfc="white",
                                  mec=_LA_HUE_CELL, mew=1.4))
        labels.append(_LA_POLICY_LBL.get(cfg, cfg))
    fig.subplots_adjust(left=0.088, right=0.895, top=0.735, bottom=0.175,
                        wspace=0.09)
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 0.998), handlelength=1.8,
               handletextpad=0.4, columnspacing=1.6)

    # Marker-fill key, scaled to the worst cell in the figure (as everywhere
    # else in this module) rather than to a hypothetical 100%.
    import matplotlib.cm as _cm
    from matplotlib.colors import Normalize as _Norm
    _sm = _cm.ScalarMappable(norm=_Norm(0, 100.0 * fmax), cmap=_INFEAS_CMAP)
    _sm.set_array([])
    _p = axs[-1].get_position()
    _cax = fig.add_axes([_p.x1 + 0.014, _p.y0, 0.010, _p.height])
    _cb = fig.colorbar(_sm, cax=_cax, orientation="vertical",
                       ticks=[0, 100.0 * fmax])
    _cb.ax.set_yticklabels(["0", f"{100.0 * fmax:.0f}"])
    _cb.outline.set_linewidth(0.3)
    _cb.set_label("Marker fill: infeasible runs (%)", fontsize=5.4, labelpad=2)
    _cb.ax.tick_params(labelsize=4.6, length=1.5, width=0.3, pad=1)
    _save(fig, outname)



# ── the cab-hardware experiment (LOCAL) ──────────────────────────────────────
# Same plane, same encoding, different question.  The sweep asks how the policy
# should be CONFIGURED and is run on the cluster; this one asks what the chosen
# configuration COSTS on the kind of machine a driver would actually have, and
# is run on a handful of instances on one local machine.  The two must not share
# an axis pair or a CSV — the numbers are not commensurable, because the whole
# point is that the hardware differs — so this reads its own file and draws its
# own figure, and the sweep can never be moved by a run landing here.
#
# No ladders: horizon and scenario count are settled by the sweep and held at
# the base cell throughout.  What varies is the subproblem solver and the break
# regime, which is the 2x2 the operational claim rests on.  The LP arm is the
# anchor the other three are read against, exactly as the base cell is in the
# sweep figure.
_LA_LOCAL_CELLS = [
    # cfg key             panel tag        legend label            mark colour   above leader
    ("LOCAL",             "LP",            r"LP subpr.",
     "o", _LA_HUE_CELL, True,  False),
    ("LOCAL+MIP",         "MILP",          r"MILP subpr.",
     "D", _LA_HUE_CELL, True,  True),
    ("LOCAL+NOSPLIT",     "LP no-split",   r"LP subpr., no split break",
     "^", _LA_HUE_CELL, True,  True),
    ("LOCAL+MIP+NOSPLIT", "MILP no-split", r"MILP subpr., no split break",
     "v", _LA_HUE_CELL, False, True),
]


def section_la_local():
    """§8.3 — the same cost/quality plane for the cab-hardware (LOCAL) runs.

    Mostly empty until those runs land; every declared cell still draws its
    legend entry and reports itself as pending, which is the point — the figure
    states the intended 2x2 before the evidence for it exists.
    """
    section_la_all(csv_name="additional_la_local_stats.csv",
                   cell_decl=_LA_LOCAL_CELLS,
                   ladders=False,
                   anchor="LOCAL",
                   outname="additional_la_local",
                   banner="cab-hardware cost/quality plane (LOCAL)",
                   # a timing probe is a few instances on purpose; one run is
                   # already the measurement, not a sample of one
                   min_n=1)
    section_la_local_table()


# ── the LOCAL timing table ───────────────────────────────────────────────────
# Read from the LOGS, not the solution JSONs, for two reasons.  The stored
# metrics keep only a mean and a max over ALL stops, so a CS-only figure has to
# be re-derived per stop anyway; and a timing measurement does not need the run
# to have finished, because every completed stop has already printed its own
# line.  Only a gap needs a finished route, and this table carries none — which
# is what lets a run still in progress contribute the stops it has done.
_LA_LOCAL_LOG_HDR = re.compile(r"^\[LA\] stop (\d+) \((\w+)\)")
_LA_LOCAL_LOG_CHO = re.compile(r"-> CHOSEN .*?([\d.]+)s\s*$")
# (solver, no_split) -> the two label columns, in the order the table reads.
# Grouped by solver rather than by regime: with the regime broken out into a
# column of its own, the pairs the reader compares are the two regimes under one
# solver, so they belong adjacent.
_LA_LOCAL_ROWS = [
    ("LP",   False, r"LP",   r"base"),
    ("LP",   True,  r"LP",   r"no split"),
    ("MILP", False, r"MILP", r"base"),
    ("MILP", True,  r"MILP", r"no split"),
]


def _la_local_logs() -> dict:
    """Per-run CS/all-stop timings for every LOCAL log on disk.

    -> {(solver, no_split): {"done": [...], "running": [...]}}.  A run counts
    as finished once it has a solution JSON.  Both are pooled into the table,
    but the split is kept so the count can be flagged: a partial route is not a
    small sample of a whole one, since the look-ahead is dearest early while the
    horizon still reaches far ahead.  Measured on the two finished MILP runs,
    their first 103 stops read 100.5 s and 164.3 s per CS stop against 84.1 s
    and 129.0 s over the full route — 20 to 27 % hot.
    """
    out: dict = {}
    for path in sorted(glob.glob(_paths.logs("*_LA_LOCAL*.txt"))):
        rid = os.path.basename(path)[:-4]
        by, cur, head, inst = {}, None, "", ""
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "Settings :" in line:
                    head = line
                elif "Instance :" in line:
                    inst = line
                m = _LA_LOCAL_LOG_HDR.match(line)
                if m:
                    cur = m.group(2)
                    continue
                if cur:
                    m = _LA_LOCAL_LOG_CHO.search(line)
                    if m:
                        by.setdefault(cur, []).append(float(m.group(1)))
        css = by.get("CS", [])
        if not css:
            continue
        key = ("MILP" if re.search(r"\bMIP\b", head) else "LP",
               "__nosplit" in inst)
        done = bool(glob.glob(_paths.solutions(f"{rid}.json")))
        out.setdefault(key, {"done": [], "running": []})
        out[key]["done" if done else "running"].append(
            dict(all=[t for v in by.values() for t in v], cs=css))
    return out


def section_la_local_table():
    """§8.3 — what the chosen configuration costs on cab-grade hardware.

    Every row of the 2x2 is printed whether or not it has runs, so the table
    states the experiment and shows how much of it has landed.
    """
    data = _la_local_logs()
    body, partial = [], []
    for solver, nosplit, c_solver, c_regime in _LA_LOCAL_ROWS:
        d = data.get((solver, nosplit), {})
        runs = d.get("done", []) + d.get("running", [])
        n_run = len(d.get("running", []))
        head = f"{c_solver} & {c_regime}"
        if not runs:
            body.append(f"{head} & \\multicolumn{{4}}{{c}}{{\\emph{{pending}}}}")
            continue
        allv = [t for r in runs for t in r["all"]]
        css = [t for r in runs for t in r["cs"]]
        n = f"{len(runs)}" + (f"$^{{\\dagger}}$" if n_run else "")
        body.append(f"{head} & {n} & {sum(allv) / len(allv):.1f} & "
                    f"{sum(css) / len(css):.1f} & {max(css):.1f}")
        if n_run:
            partial.append(f"{c_solver}/{c_regime} ({n_run})")

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Look-ahead decision cost on a single machine of the class "
        r"available in a vehicle, pooled over the runs made so far.  "
        r"Charging-station stops are reported separately because they carry the "
        r"branching part of the decision and are the wait a driver actually "
        r"experiences; the maximum is taken over individual stops, not over "
        r"runs.  The configuration is held at the base cell ($|\Xi| = 25$, "
        r"$L = 24$\,h) throughout, so the only quantities varying down the "
        r"table are the subproblem solver and the break regime.}",
        r"\label{tab:la_local}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Subproblem & Break regime & Runs & $\bar{t}$ / stop (s) & "
        r"$\bar{t}$ / CS stop (s) & $\max t$ / CS stop (s) \\",
        r"\midrule",
    ]
    lines += [b + r" \\" for b in body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if partial:
        # Flagged rather than dropped: the run in progress is real measurement,
        # but its stops are drawn from the expensive early part of a route, so a
        # reader comparing rows needs to know which ones carry one.
        lines.append(
            r"\\[2pt]{\footnotesize $^{\dagger}$ includes a run still in "
            r"progress (" + ", ".join(partial) + r"), whose completed stops are "
            r"pooled with the rest.  Such a run contributes only the early part "
            r"of a route, where the look-ahead is dearest because the horizon "
            r"still reaches far ahead, so the affected rows read slightly high.}")
    lines += [r"\end{table}", ""]
    _write_tex("additional_la_local.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# §8.3 — PACK x CHARGE POINT GRID
# ══════════════════════════════════════════════════════════════════════════════
# Why a grid and not two one-at-a-time rows: Emin = SOC_MIN_FRAC.Ecap and the
# tail acceptance is TAIL_C_RATE.Ecap, so resizing the pack moves BOTH the range
# and the point where the charge curve tapers.  A battery sweep at fixed charger
# power therefore measures range and taper avoidance together and cannot say
# which one moved the duration.  Crossing the two axes separates them: reading
# ACROSS a row isolates charge speed at a fixed pack, reading DOWN a column
# isolates pack at a fixed charge point, and curvature between the two is the
# interaction the one-at-a-time sweep folds away.
#
# The base row (350 kW) and base column (500 kWh) ARE the one-at-a-time sweeps —
# additional_analysis.cmd_grid gives those cells the single-axis tag on purpose,
# so they share instances and runs with §8.3 rather than duplicating them.

_GRID_BATTERY = [300, 500, 700, 900]      # kWh  (500 = base pack)
_GRID_POWER   = [150, 350, 700, 1000]     # kW   (350 = base charge point)
# The grid is run on short + medium only (agreed 2026-08-14): 9 crossed cells x
# 200 instances is already 1800 window-MILP generations, and long routes carry
# the slowest of those.  Long still has the one-at-a-time sweeps, so it appears
# in section_sensitivity — but here it would contribute nothing except the base
# row and column it shares with them, i.e. a panel that is empty by design.
# Widen this list if the crossed cells are ever run for long routes.
_GRID_ROUTES = ["short", "medium"]


def _grid_tag(batt, power) -> str | None:
    """Variant tag for one grid cell, or None for the base case.

    Mirrors additional_analysis._materialise_grid: only the axes that DIFFER
    from the base contribute, joined battery-first, so a cell on a base row or
    column collapses to exactly the one-at-a-time tag ("kwh300", "kw150") and
    finds the runs the sensitivity sweep already produced.
    """
    parts = []
    if float(batt) != float(BATTERY_CAPACITY):
        parts.append(f"kwh{batt:g}")
    if float(power) != float(CHARGER_POWER_BASE_KW):
        parts.append(f"kw{power:g}")
    return "_".join(parts) or None


def _grid_cell_values(route, combos_s, tws_s, seeds_s, tag) -> list:
    """Paired greedy deltas (%) vs the base instance for one cell.

    Paired exactly like section_sensitivity: base and variant must BOTH exist
    and BOTH be feasible, so a cell is never contaminated by an instance that
    stranded on one side only.  The base cell returns 0.0 per instance by
    construction (it is the denominator), which is what anchors the diverging
    colour scale at a real reference rather than at the data's midpoint.
    """
    out = []
    for r, c in combos_s:
        if r != route:
            continue
        for tw in tws_s:
            for seed in seeds_s:
                st = _stem(r, c, tw, seed)
                bg = _greedy(st)
                vg = bg if tag is None else _greedy(st, tag)
                if (bg and vg and not bg["infeasible"]
                        and not vg["infeasible"] and bg["duration"] > 0):
                    out.append(100.0 * (vg["duration"] / bg["duration"] - 1.0))
    return out


def section_grid():
    print("== Sec 8.3 grid: battery capacity x charger power ==")
    tags = [t for t in (_grid_tag(b, p)
                        for b in _GRID_BATTERY for p in _GRID_POWER) if t]
    found    = _discover_scope(tags)
    combos_s = found["combos"] or COMBOS
    tws_s    = found["tws"]    or TWS
    seeds_s  = found["seeds"]  or list(SEEDS)
    routes   = [r for r in (found["routes"] or _ROUTE_SPLIT)
                if r in _GRID_ROUTES] or _GRID_ROUTES
    combos_s = [(r, c) for r, c in combos_s if r in _GRID_ROUTES] or combos_s
    print(f"  scope     : combos {','.join(f'R{r}C{c}' for r, c in combos_s)}")
    print(f"              tw {','.join(tws_s)}  "
          f"seeds {min(seeds_s)}-{max(seeds_s)} (n={len(seeds_s)})  "
          f"routes {','.join(routes)}")

    nb, np_ = len(_GRID_BATTERY), len(_GRID_POWER)
    mats, cnts = {}, {}
    for route in routes:
        M = np.full((nb, np_), np.nan)
        C = np.zeros((nb, np_), dtype=int)
        for i, b in enumerate(_GRID_BATTERY):
            for j, p in enumerate(_GRID_POWER):
                v = _grid_cell_values(route, combos_s, tws_s, seeds_s,
                                      _grid_tag(b, p))
                if v:
                    M[i, j] = float(np.mean(v))
                    C[i, j] = len(v)
        mats[route], cnts[route] = M, C

    filled = sum(int(np.isfinite(m).sum()) for m in mats.values())
    print(f"  cells     : {filled} filled of {nb * np_ * len(routes)}")

    # ── colour: DIVERGING, because the value is signed (below/above the base
    # duration) and 0 is a real reference, not the data midpoint.  Two hues plus
    # a neutral grey centre — never a rainbow, never a hue at the midpoint.  The
    # poles are the Okabe-Ito blue/vermillion pair (the strongest CVD-separated
    # pair available); no method identity appears in this figure, so borrowing
    # the two hues here cannot collide with METHOD_COLOR.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "pack_power_div", [BLUE, "#f4f4f4", VERM])
    cmap.set_bad("#fbfbfb")                       # empty slots stay near-surface
    finite = np.concatenate([m[np.isfinite(m)].ravel() for m in mats.values()]
                            or [np.array([0.0])])
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 0.5)                         # keep a sane range when flat
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)

    # Height is driven by the square cells, not chosen: nb rows of squares at
    # the panel width, plus a fixed allowance for tick labels, axis titles and
    # the panel heading.  Keeps the figure a single-column-friendly band rather
    # than the near-square block "auto" aspect produced.
    _panel_w = 2.45
    fig, axes = plt.subplots(1, len(routes),
                             figsize=(_panel_w * len(routes) + 0.9,
                                      _panel_w * nb / np_ + 0.95),
                             squeeze=False)
    axes = axes[0]

    for ax, route in zip(axes, routes):
        M, C = mats[route], cnts[route]
        # aspect="equal": square cells.  A cell is a (pack, power) COMBINATION,
        # not a quantity, so stretching it to fill the axes gives the two axes
        # a visual weight they do not have and wastes column height in a paper.
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm,
                  origin="lower", aspect="equal")
        # 2 px surface gap between cells: adjacent fills must not touch
        ax.set_xticks(np.arange(-0.5, np_, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nb, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

        for i in range(nb):
            for j in range(np_):
                if not np.isfinite(M[i, j]):
                    ax.text(j, i, "--", ha="center", va="center",
                            color=MUT, fontsize=7)
                    continue
                # Ink colour by cell luminance so the value stays legible at
                # both poles; this is a contrast fix, not a second encoding.
                r_, g_, b_ = cmap(norm(M[i, j]))[:3]
                lum = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                        color=("white" if lum < 0.5 else INK),
                        fontsize=7.5,
                        fontweight=("bold" if _grid_tag(_GRID_BATTERY[i],
                                                        _GRID_POWER[j]) is None
                                    else "normal"))
        # mark the base cell — every number in the panel is relative to it
        bi = _GRID_BATTERY.index(BATTERY_CAPACITY) \
            if BATTERY_CAPACITY in _GRID_BATTERY else None
        bj = _GRID_POWER.index(CHARGER_POWER_BASE_KW) \
            if CHARGER_POWER_BASE_KW in _GRID_POWER else None
        if bi is not None and bj is not None:
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                       edgecolor=ps.BASELINE, linewidth=1.4,
                                       zorder=5))

        ax.set_xticks(range(np_)); ax.set_xticklabels(_GRID_POWER)
        ax.set_yticks(range(nb));  ax.set_yticklabels(_GRID_BATTERY)
        ax.set_xlabel("Charger power (kW)")
        if route == routes[0]:
            ax.set_ylabel("Battery capacity (kWh)")
        # Per-cell n is NOT on the figure — it lives in additional_grid_stats.csv,
        # one column per cell.  Keep an eye on it: the pairing rule drops any
        # instance that was infeasible on either side, so a partially-run grid
        # or a stranding-prone row (small packs) carries fewer pairs than the
        # nominal seed count, and cross-cell comparison is then uneven.
        ax.set_title(ps.ROUTE_LBL[route], color=INK)
        for s in ax.spines.values():
            s.set_edgecolor(MUT)

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=list(axes), fraction=0.045, pad=0.02)
    cb.set_label("Duration vs base case (%)", color=INK)
    cb.outline.set_edgecolor(MUT)
    cb.ax.tick_params(color=MUT)
    _save(fig, "additional_grid_battery_power")

    rows = []
    for route in routes:
        for i, b in enumerate(_GRID_BATTERY):
            for j, p in enumerate(_GRID_POWER):
                m = mats[route][i, j]
                rows.append([route, b, p, _grid_tag(b, p) or "base",
                             "" if not np.isfinite(m) else f"{m:.3f}",
                             int(cnts[route][i, j])])
    _write_csv("additional_grid_stats.csv",
               ["route_class", "battery_kwh", "charger_kw", "tag",
                "greedy_duration_vs_base_%", "n_paired"], rows)


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = dict(diesel=section_diesel, sensitivity=section_sensitivity,
                 grid=section_grid,
                 la=section_la,
                 la_all=section_la_all,
                 la_local=section_la_local,
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
