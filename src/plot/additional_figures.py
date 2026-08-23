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
  figures/additional_la_horizon.png|pdf      §8.3 horizon ladder, routes pooled
  figures/additional_la_scenarios.png|pdf    §8.3 scenario ladder, routes pooled
  figures/additional_la_plane.png|pdf        §8.3 cost/quality plane, one
                                             panel per ladder, routes pooled
  figures/additional_la_all.png|pdf          §8.3 whole LA study on one plane
                                             (horizon + scenarios + policy)
  tex/tables/additional_la.tex               (all read
  tex/tables/additional_la_policy[_pen].tex   additional_analysis.py's
  tex/tables/additional_la_local.tex          data_output/additional_la_stats.csv)

The LA study is four figures and four tables.  Dropped 2026-08-22 on request:
the cost/quality FRONTIER (its content is the la_all plane), the two POLICY bar
figures (dur and pen — the tables carry them), and the LOCAL plane (five runs on
axes built for twelve cells).  The policy and LOCAL TABLES are unchanged.

Every LA cost axis is the MEASURED decision time at charging-station stops
(la-report's decision_cs_mean_s_median), parsed per stop from the run logs.
The wall-clock-over-stops figure is still written to the CSV as
t_per_stop_s_median, for diagnosis only.
LA figures and tables are quoted over seeds 1-10 only; la-report prints what is
missing inside that window and writes data_output/additional_la_coverage.csv.
Every cell is reported on whatever runs it has (la-report --panel all, the
default since 2026-08-23); cells can therefore rest on different seeds, and a
cell too thin to read is drawn hollow and named in the footnote.
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


_DIR_WHERE: dict[str, dict[str, str]] = {}


def _dir_index(directory: str) -> list[str]:
    """Sorted entry NAMES of ``directory``, scanned once per process.

    solutions/ and logs/ are split into experiment buckets, so the index covers
    the tree root and each bucket and is keyed on the basename; _DIR_WHERE
    holds name -> full path so _matches can still hand back openable paths.
    Sorting basenames rather than paths is what keeps the order meaningful: a
    run_id ends in its timestamp, so lexicographic order is chronological order
    regardless of which bucket a run landed in.

    These are read-only reporting runs, so the listing cannot change underneath
    us; call ``_DIR_INDEX.clear()`` if that ever stops holding.
    """
    names = _DIR_INDEX.get(directory)
    if names is None:
        where = dict(_paths.scan_tree(directory or "."))
        _DIR_WHERE[directory] = where
        names = _DIR_INDEX[directory] = sorted(where)
    return names


def _matches(pattern: str) -> list[str]:
    """Every file matching ``pattern``, oldest first (lexicographic = by run ts).

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
    where = _DIR_WHERE[directory]
    return [where[n] for n in block]


def _latest(pattern: str) -> str | None:
    """Newest file matching ``pattern``, or None."""
    block = _matches(pattern)
    return block[-1] if block else None


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


# Preference order among the runs of one (instance, method).  The glob cannot
# express it: "<stem>_LA_*.json" also matches every --variant run of that
# instance, and since the sort is lexicographic a tag beats the bare timestamp
# ("S50H24…" > "MIPTAIL…" > "2026…"), so the newest match was routinely a sweep
# cell rather than the run the section means.  These sections report the METHOD,
# so they must take the method's standard configuration and never a sweep point.
#
# The LP tail is kept as a fallback because the §8.4/§8.5 batches predate the
# switch to the MIP tail and ran the policy under the old default: on those
# instances it is the only LA run there is, and dropping it would silently empty
# the LA series instead of reporting the runs that exist.
_POLICY_PREF = (None, _paths.LA_LEGACY_VARIANT)

# Policy parameters the variant tag does NOT capture, per method.  A paired
# delta divides two runs of "the same policy", and the tag alone does not
# establish that: LA's committed-charge energy guard is stored as a plain field
# and the base corpus holds both 0.5 and None runs, so an EV leg guarded at 0.5
# could be divided by an unguarded diesel one and the ratio would carry the
# guard as well as the drivetrain.  audit_runs polices the same parameters
# WITHIN a scope; this polices them ACROSS the two legs of a pair.
_PAIR_PARAMS = {"LA": ("la_energy_quantile",)}


def _policy(stem: str, alg: str, tag: str | None = None,
            cfg: str | None = ...) -> dict | None:
    """Standard-configuration simulated-policy solution for one instance.

    ``alg`` is the run-id algorithm token (GREEDY, LA, 2SP, RO, ROBU).  Accepts
    both the '__tag' stem (orchestrator batch) and the runner-normalised
    '_tag'.  Newest run wins within a configuration; configurations are ranked
    by _POLICY_PREF, and a run of any other one is not a candidate at all.

    ``cfg`` pins the configuration instead of ranking them — the ellipsis
    default means "rank", since None is itself a configuration (the standard
    one).  _policy_pair uses it to hold both legs of a paired delta on one
    configuration; nothing else should need it.
    """
    pats = ([_paths.solutions(f"{stem}_{alg}_*.json")] if tag is None else
            [_paths.solutions(f"{stem}__{tag}_{alg}_*.json"),
             _paths.solutions(f"{stem}_{tag}_{alg}_*.json")])
    order = _POLICY_PREF if cfg is ... else (cfg,)
    for p in pats:
        best, best_name = {}, {}
        for f in _matches(p):
            # duration_h + metrics only, so the stripped run_cache record is
            # enough and the trajectory arrays are never read off disk.
            name = os.path.basename(f)
            d = _RUNS_BY_NAME().get(name)
            if not d or d.get("duration_h") is None:
                continue
            got = _paths.effective_variant(d.get("method"), d.get("variant"),
                                           d.get("solve_mode"))
            if got in order:
                best[got] = d           # matches are ordered, so the last wins
                best_name[got] = name
        for want in order:
            d = best.get(want)
            if d is not None:
                infeas = bool((d.get("metrics") or {}).get("run_infeasible"))
                # A parameter that cannot affect a leg must not disqualify the
                # pair.  LA's energy guard sizes the committed charge against
                # predicted CONSUMPTION, and a diesel leg consumes nothing and
                # charges nothing, so the flag is inert there — None marks the
                # tuple "not applicable" rather than "set to nothing".
                inert = "diesel" in str(d.get("instance") or "").lower()
                return dict(duration=float(d["duration_h"]), infeasible=infeas,
                            file=best_name[want],
                            params=None if inert else
                            tuple(d.get(k) for k in
                                  _PAIR_PARAMS.get(alg.upper(), ())))
    return None


# Nominal length of each break block (h), for recovering the masked charge of a
# simulated run.  "0"/None is "no break" and is deliberately absent.
_BREAK_BLOCK_H = {"b45": 0.75, "b30": 0.5, "b15": 0.25}

_RUN_DWELL: dict = {}


def _run_dwell(fname: str | None) -> dict | None:
    """Per-component dwell totals (h) of a simulated run, from its own file.

    The same components _oracle_dwell reads off the hindsight schedule, taken
    from the run's per-stop durations_list: driving and customer service are
    identical within an EV/diesel pair, so differencing these accounts for the
    whole makespan gap — verified at 0.0000 h residual on every pair sampled.

    Read from the FILE, not from run_cache: durations_list is one of the arrays
    the cache drops (it is ~30 % of the corpus by bytes and nothing else needs
    it).  Memoised per process, since each run is asked for once per section.
    """
    if fname is None:
        return None
    if fname not in _RUN_DWELL:
        d = _load(_paths.solution_path(fname))
        if not d:
            _RUN_DWELL[fname] = None
        else:
            out = {k: 0.0 for k in _DWELL_ROWS}
            for e in (d.get("durations_list") or []):
                out["charging"]   += float(e.get("tauc") or 0.0)
                out["queue"]      += float(e.get("tauq") or 0.0)
                out["break"]      += float(e.get("taub") or 0.0)
                out["rest"]       += float(e.get("taur") or 0.0)
                out["manoeuvre"]  += (float(e.get("mstop") or 0.0)
                                      + float(e.get("mlay") or 0.0))
                out["reposition"] += float(e.get("mseq") or 0.0)
            # g is not stored for a simulated run, but it is recoverable by
            # the rule the MILP uses for it: at a stop that takes a break while
            # charging, the credited time is the charge capped at the break
            # block (g_i <= tauc_i, g_i <= the break's duration).  A policy
            # books concurrent break+charge as tauc with taub = 0, so without
            # this the run would look as if it masked nothing.
            acts = d.get("actions") or []
            out["_g"] = sum(
                min(float(e.get("tauc") or 0.0), _BREAK_BLOCK_H[bt])
                for e, a in zip(d.get("durations_list") or [], acts)
                for bt in [str(a.get("break_type") or "0")]
                if bt in _BREAK_BLOCK_H)
            out["_duration"] = float(d.get("duration_h") or 0.0)
            _RUN_DWELL[fname] = out
    return _RUN_DWELL[fname]


# The stack the §8.4 figure draws: where the electrification penalty goes, in
# terms that are each a positive cost rather than a signed accounting.
#
# Two hue families, because the components answer two different questions and
# the split is the point: WARM is time the charger costs (charging that no break
# was running to hide, queueing for the plug, repositioning off the bay); COOL
# is time that stopping more often costs (the breaks and rests the EV takes
# beyond the diesel's legal minimum, and the manoeuvring into those stops).
# Within a family the step is lightness, ordered by size, so the two blocks read
# as blocks.  Checked, not eyeballed: worst pair over all 15 is dE 16.5 in
# normal vision and 13.3 under simulated deuteranopia/protanopia.
_GAP_STACK = [
    ("charge_open", "Charging outside breaks",  "#A6461D"),
    ("queue",       "Charger queueing",         "#DE9350"),
    ("reposition",  "Bay repositioning",        "#F7DCC0"),
    ("extra_break", "Extra break",              "#1F4E6B"),
    ("manoeuvre",   "Extra stop manoeuvring",   "#3E7FA3"),
    ("rest",        "Extra rest",               "#86B6D0"),
]


# Which term absorbs a negative, per family: the charger-side costs collapse
# into the charging they are incurred for, the stopping-side ones into the extra
# break time they are part of.
_STACK_FAMILY = {"queue": "charge_open", "reposition": "charge_open",
                 "manoeuvre": "extra_break", "rest": "extra_break"}


def _stack_drawable(cell):
    """Cell means as a NON-NEGATIVE composition, summing to the same total.

    The components are EV-minus-diesel differences, so any one of them can come
    out negative in a cell -- most often `rest`, where the EV happens to take
    less rest than the diesel, occasionally `extra_break` or `charge_open`.  A
    stacked bar cannot draw that: matplotlib lays the negative segment ABOVE the
    running total, outside the outline, which reads as a floating block rather
    than as a credit.

    So each negative is netted into the principal term of its own family, and a
    principal that is itself negative into the largest term left.  Every drawn
    segment is then >= 0 and the bar height is still exactly the penalty.  The
    signed per-component values are what the CSV carries -- this is a drawing
    rule, not a change to the decomposition.
    """
    vals = {k: (_mean(cell[k]) or 0.0) for k, _l, _c in _GAP_STACK}
    for member, principal in _STACK_FAMILY.items():
        if vals.get(member, 0.0) < 0:
            vals[principal] += vals[member]
            vals[member] = 0.0
    for principal in ("charge_open", "extra_break"):
        if vals.get(principal, 0.0) < 0:
            target = max(vals, key=lambda k: vals[k])
            if target != principal:
                vals[target] += vals[principal]
                vals[principal] = 0.0
    return vals


def _gap_components(ev: dict | None, di: dict | None, fuel: float) -> dict | None:
    """EV-minus-diesel dwell components as a % of the diesel route duration.

    The split of the EV's charging is the model's own: `g` is the part credited
    inside a mandatory break, so it costs no makespan beyond the break that was
    running anyway, and the remainder is charging with nothing to hide behind.
    The break term is then everything the EV spends on breaks — the credited
    charge plus any standalone break — LESS the break time the diesel is legally
    obliged to take, i.e. the breaks it takes over and above the minimum.

    Why not (extra stops) x 45 min, which is the same idea with an assumed
    constant: a charge stop shorter than the block gets charged the whole block,
    which over-attributes.  Measured on the oracle it pushed the charging
    remainder negative in a quarter to a third of instances, and lifted the
    long-route break share from 41 % to 63 % — the headline would then rest on
    the constant rather than on the schedule.

    The percentages sum to exactly the penalty the run pair produces, so the
    stack height is the number the un-decomposed figure drew as a single bar.
    """
    if not (ev and di):
        return None
    base = di["_duration"] + fuel
    if base <= 0:
        return None
    pct = lambda h: 100.0 * h / base
    g = ev.get("_g") or 0.0
    d = {k: pct(ev[k] - di[k]) for k in _DWELL_ROWS}
    # charging with no break running to hide it; the diesel's post-hoc fuel stop
    # is credited here, being the one term that is not a dwell difference
    d["charge_open"] = pct(ev["charging"] - g - fuel)
    # break time over the diesel's legal minimum: the charge credited to a break
    # plus any break taken on its own, against what the diesel must take anyway
    d["extra_break"] = pct(g + ev["break"] - di["break"])
    # Coupling: the share of the EV charging that ran inside a mandatory break,
    # i.e. cost no makespan beyond the break that was happening anyway.  It is
    # an EV-side quantity and not part of the stack, but it comes off the same
    # dwell, and it is what says WHY the charging split falls where it does.
    d["_coupling"] = (100.0 * g / ev["charging"]
                      if ev["charging"] > 1e-9 else None)
    return d


def _policy_pair(stem: str, alg: str, tag: str | None):
    """(base run, tagged run) of one instance -- see _policy_pair_tags."""
    return _policy_pair_tags(stem, alg, None, tag)


def _policy_pair_tags(stem: str, alg: str, ev_tag, di_tag):
    """Two runs of one instance to divide, under ONE configuration.

    A paired delta divides two runs of the same policy, so both legs have to be
    the same configuration of it or the ratio carries the configuration as well
    as the perturbation.  Two independent _policy calls cannot guarantee that
    since the swap: the base instances hold both tails (the MILP one is
    standard, the LP one is the LPTAIL variant) while the §8.4/§8.5 variant
    instances were run under the old default and hold only the LP tail, so
    ranking each leg on its own pairs a MILP base against an LP variant.

    So: take the most-preferred configuration that BOTH legs have, and if there
    is none, report no pair at all.  A missing pair reads as pending, which the
    callers already handle; a mixed one would read as a measurement.

    The same rule covers the parameters no tag records (_PAIR_PARAMS): legs that
    disagree on one are not a pair either, however well their tags match.

    Both tags are arguments because the two figures pair different things: the
    headline 8.4 figure divides a base EV run by the diesel copy of the same
    instance, the OAT figure divides an axis-level EV run by that same diesel.
    They must nonetheless agree run for run, so they share this one rule -- when
    they did not, the two figures reported different numbers for the same cell.
    """
    for cfg in _POLICY_PREF:
        base = _policy(stem, alg, ev_tag, cfg=cfg)
        var  = _policy(stem, alg, di_tag, cfg=cfg)
        if not (base and var):
            continue
        # None on either side = the parameters do not apply to that leg
        if (base["params"] is None or var["params"] is None
                or base["params"] == var["params"]):
            return base, var
    return None, None


def _greedy(stem: str, tag: str | None = None) -> dict | None:
    return _policy(stem, "GREEDY", tag)


def _la(stem: str, tag: str | None = None) -> dict | None:
    return _policy(stem, "LA", tag)


def _oracle(stem: str, tag: str | None = None) -> dict | None:
    """Oracle cache -> duration (h), total/coupled charging time (h)."""
    names = ([_paths.solution_path(f"oracle_{stem}.json")] if tag is None else
             [_paths.solution_path(f"oracle_{stem}__{tag}.json"),
              _paths.solution_path(f"oracle_{stem}_{tag}.json")])
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
    names = ([_paths.solution_path(f"oracle_{stem}.json")] if tag is None else
             [_paths.solution_path(f"oracle_{stem}__{tag}.json"),
              _paths.solution_path(f"oracle_{stem}_{tag}.json")])
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

    # g = the charging time the model credits INSIDE a mandatory break, i.e.
    # charging that costs no extra makespan because a break was running anyway.
    out["_g"]        = sum(float(s.get("g") or 0.0) for s in sol)
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
        out = _paths.figure_out(f"{name}.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  Figure    : {os.path.relpath(out, _paths.ROOT)[:-4]}.png|pdf")


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
    stack: dict[tuple, dict[str, list]] = {}
    detail = []
    for route, cust in combos_d:
        for tw in tws_d:
            for seed in seeds_d:
                st       = _stem(route, cust, tw, seed)
                ev_o     = _oracle(st)
                di_o     = _oracle(st, "diesel")
                # Paired, so each policy is compared against ITSELF across the
                # two drivetrains: _policy_pair holds both legs on one
                # configuration, which for LA is what keeps a MILP-tail EV run
                # from being divided by an LP-tail diesel one.
                ev_g, di_g = _policy_pair(st, "GREEDY", "diesel")
                ev_l, di_l = _policy_pair(st, "LA", "diesel")

                fuel  = _refuel_h(route)         # post-hoc diesel fuel stop(s)
                dur_d = (di_o["duration"] + fuel) if di_o else None

                pen_o = pen_g = pen_l = coup = None
                # Absolute counterpart of the percentage: the same difference
                # in hours, so the figure states the penalty in the unit the
                # operator actually plans in.
                dt_o = dt_g = dt_l = None
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
                # LA exactly as greedy: both legs present, both feasible, so a
                # run that stranded on one drivetrain only never enters the mean.
                if (ev_l and di_l and not ev_l["infeasible"]
                        and not di_l["infeasible"] and di_l["duration"] > 0):
                    pen_l = 100 * (ev_l["duration"] / (di_l["duration"] + fuel) - 1)
                    dt_l  = ev_l["duration"] - (di_l["duration"] + fuel)

                # The EV oracle is only an incumbent where the solve hit its
                # wall budget (long routes stall on the DUAL bound), so the
                # penalty there is an upper bound on the true optimal one.
                ev_gap = 100 * ev_o["gap"] if ev_o else None

                detail.append([route, cust, tw, seed,
                               _fmt(dur_d, ".3f", ""),
                               _fmt(ev_o and ev_o["duration"], ".3f", ""),
                               _fmt(pen_o, ".2f", ""), _fmt(pen_g, ".2f", ""),
                               _fmt(pen_l, ".2f", ""),
                               _fmt(coup, ".1f", ""),
                               _fmt(fuel, ".3f", ""), _fmt(ev_gap, ".2f", "")])
                d = per_class.setdefault(route, dict(pen_o=[], pen_g=[],
                                                     pen_l=[],
                                                     dt_o=[], dt_g=[], dt_l=[],
                                                     coup=[],
                                                     dur_d=[], dur_e=[],
                                                     ev_gap=[]))
                d["pen_o"].append(pen_o); d["pen_g"].append(pen_g)
                d["pen_l"].append(pen_l)
                d["dt_o"].append(dt_o);   d["dt_g"].append(dt_g)
                d["dt_l"].append(dt_l)

                # Same three methods, decomposed.  The oracle reads its cached
                # schedule; the policies read the very run the penalty above was
                # computed from, so a bar and its decomposition can never come
                # from different runs.
                #
                # Gated on the penalty being defined, not merely on the two runs
                # existing: the penalties above drop a pair where either leg was
                # infeasible, and a stack built over a wider set would total to
                # something other than the penalty it is supposed to decompose
                # (greedy on the long routes: 13.2 % against a 13.5 % penalty).
                for meth, pen, legs in (
                        ("oracle", pen_o, (_oracle_dwell(st),
                                           _oracle_dwell(st, "diesel"))),
                        ("LA", pen_l, (_run_dwell(ev_l and ev_l["file"]),
                                       _run_dwell(di_l and di_l["file"]))),
                        ("greedy", pen_g, (_run_dwell(ev_g and ev_g["file"]),
                                           _run_dwell(di_g and di_g["file"])))):
                    if pen is None:
                        continue
                    comp = _gap_components(legs[0], legs[1], fuel)
                    if comp is None:
                        continue
                    cell = stack.setdefault((route, meth),
                                            {k: [] for k, _l, _c in _GAP_STACK})
                    for k, _l, _c in _GAP_STACK:
                        cell[k].append(comp[k])
                d["coup"].append(coup)
                d["dur_d"].append(dur_d)
                d["dur_e"].append(ev_o and ev_o["duration"])
                d["ev_gap"].append(ev_gap)

    _write_csv("additional_diesel_stats.csv",
               ["route", "cust", "tw", "seed", "diesel_oracle_h",
                "ev_oracle_h", "pen_oracle_%", "pen_greedy_%", "pen_la_%",
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
    la_cover = {}
    for r, d in per_class.items():
        have = sum(1 for v in d["pen_o"] if v is not None)
        want = len(tws_d) * len(seeds_d) * sum(1 for rr, _ in combos_d
                                               if rr == r)
        coverage[r] = (have, want)
        if have < want:
            print(f"  Oracle coverage {r}: {have}/{want} — partial average, "
                  f"reported with its n")
        # The LA leg needs a standard-configuration run on BOTH drivetrains, so
        # it fills in as the diesel batch lands and as the base-case MILP-tail
        # runs reach these instances.  Same treatment as the oracle: report the
        # partial mean and state what it rests on, never a silent average.
        have_l = sum(1 for v in d["pen_l"] if v is not None)
        la_cover[r] = (have_l, want)
        if have_l < want:
            print(f"  LA coverage     {r}: {have_l}/{want} — "
                  + ("partial average, reported with its n" if have_l
                     else "no paired LA runs yet, bar left empty"))
    oracle_ok = [r for r, (have, _w) in coverage.items() if have]

    # ── figure: what the penalty is MADE OF, per method and route class ──────
    # One stacked bar per (route class, method): the height is the same penalty
    # the bar used to carry on its own, and the segments say where it comes
    # from.  Colour is the COMPONENT (checked for CVD separation, see
    # _GAP_STACK); the method is read from the label under each bar, and the
    # route class from the label under each group.
    #
    # Every segment is non-negative by construction — the credits are netted
    # into "charging outside breaks", see _gap_components — so the bar is a
    # composition, never a signed accounting.  The signed version, with the
    # break and refuelling credits on their own rows, is tab:diesel-decomp.
    #
    # Chrome follows the same grammar as §8.3 (additional_sens_effects) and the
    # base-case box plots: default frame, major+minor grid on the response axis,
    # one legend, and NO title of any kind — the LaTeX \caption carries it.
    routes  = [r for r in DIESEL_ROUTES if r in per_class]
    # (method key, label, the per-class hour-difference series it is drawn from)
    methods = [("oracle", "Oracle", "dt_o"), ("LA", "LA", "dt_l"),
               ("greedy", "Greedy", "dt_g")]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(routes), dtype=float)
    # The group is sized to a FRACTION of the unit stride, not to fill it, so
    # the gap between route classes stays wider than the gaps inside a group.
    _GROUP, _PAD = 0.72, 0.03
    w = _GROUP / len(methods) - _PAD
    pos = {(r, m): x[ri] + (mi - (len(methods) - 1) / 2) * (w + _PAD)
           for ri, r in enumerate(routes)
           for mi, (m, _lbl, _dk) in enumerate(methods)}

    drawable = {k: _stack_drawable(c) for k, c in stack.items()}
    for key, lbl, col in _GAP_STACK:
        heights, bottoms, xs = [], [], []
        for r in routes:
            for m, _mlbl, _dk in methods:
                cell = stack.get((r, m))
                if not cell or not cell[key]:
                    continue
                vals = drawable[(r, m)]
                xs.append(pos[(r, m)])
                heights.append(vals[key])
                bottoms.append(sum(vals[k] for k, _l, _c in _GAP_STACK
                                   if _GAP_STACK.index((k, _l, _c))
                                   < _GAP_STACK.index((key, lbl, col))))
        if xs:
            # edgecolor="none": a stroke paints the left and right edges too,
            # which left a white halo between the fill and the method ring.  The
            # separators are drawn as lines across the bar below.
            ax.bar(xs, heights, w, bottom=bottoms, color=col, label=lbl,
                   edgecolor="none", zorder=3)
            for _x, _b, _h in zip(xs, bottoms, heights):
                if _b > 1e-9 and _h > 1e-9:
                    ax.plot([_x - w / 2, _x + w / 2], [_b, _b], color="white",
                            lw=0.7, solid_capstyle="butt", zorder=3.5)

    # Method identity is a SECOND encoding, kept off the fills: each bar is
    # ringed in its method's colour from paper_style (oracle grey, LA green,
    # greedy blue) and its label under the axis is printed in the same colour.
    # Without it the method was legible only from 6 pt grey text, and the figure
    # read as nine anonymous bars — the components are what the fills say, and
    # they are identical across the three, so the fills cannot carry it.
    from matplotlib.patches import Rectangle as _Rect
    for r in routes:
        for m, mlbl, dkey in methods:
            cell = stack.get((r, m))
            if not cell:
                continue
            tot = sum(_mean(cell[k]) or 0.0 for k, _l, _c in _GAP_STACK)
            mcol = ps.METHOD_COLOR["oracle" if m == "oracle" else m]
            ax.add_patch(_Rect((pos[(r, m)] - w / 2, 0), w, tot,
                               facecolor="none", edgecolor=mcol, lw=1.1,
                               zorder=4, joinstyle="miter"))
            # Both units on the mark: the percentage is what the axis measures,
            # and the hours underneath are what an operator actually plans in.
            # They come from the same pairs — dt_* is appended beside pen_* — so
            # the two lines can never describe different run sets.
            hrs = _mean(per_class[r][dkey])
            ax.annotate(f"{tot:.1f}%", (pos[(r, m)], tot),
                        textcoords="offset points", xytext=(0, 9.5),
                        ha="center", va="bottom", fontsize=6.8, color=mcol,
                        zorder=5)
            if hrs is not None:
                ax.annotate(f"{hrs:+.1f} h", (pos[(r, m)], tot),
                            textcoords="offset points", xytext=(0, 2.5),
                            ha="center", va="bottom", fontsize=6.0,
                            color=MUT, zorder=5)

    # Two label levels: the method under its own bar, the route class under the
    # group, so neither is inferred from the legend.
    ax.set_xticks([pos[(r, m)] for r in routes for m, _l, _d in methods],
                  [lbl for _r in routes for _m, lbl, _d in methods], fontsize=7)
    ax.tick_params(axis="x", length=0, pad=2.5)
    for lab, (m, _l, _d) in zip(ax.get_xticklabels(),
                                [mm for _r in routes for mm in methods]):
        lab.set_color(ps.METHOD_COLOR["oracle" if m == "oracle" else m])
    span = ax.get_xaxis_transform()
    for ri, r in enumerate(routes):
        ax.annotate(ps.ROUTE_LBL[r], xy=(x[ri], 0), xycoords=span,
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5, color=INK)
    ax.set_xlim(-0.55, len(routes) - 0.45)
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
    # Headroom so the legend row clears the tallest bar's total label.
    _top = max((sum(_mean(c[k]) or 0.0 for k, _l, _cl in _GAP_STACK)
                for c in stack.values()), default=1.0)
    # Headroom for the legend block, which sits INSIDE the axes: the tallest bar
    # plus its two label lines has to clear the bottom of it.
    ax.set_ylim(0, _top * 1.42)
    # Two columns, which is also the two families: matplotlib fills column-major
    # and _GAP_STACK is ordered warm-then-cool, so the left column is everything
    # the charger costs and the right everything stopping more often costs.
    # Small type and a white ground: it shares the panel with the bars, so it has
    # to stay out of their way and stay readable if a taller class ever reaches
    # it.
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none",
              facecolor="white", fontsize=5.8, ncol=2, loc="upper left",
              handlelength=0.9, handletextpad=0.35, columnspacing=0.9,
              labelspacing=0.35, borderpad=0.35)
    # Bottom reserve: the route labels hang below the axes and tight_layout does
    # not measure annotations drawn outside them.
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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
    # The LA column pairs two policy runs, so it has a coverage of its own and
    # states it separately from the oracle's n; the sentence goes away when the
    # pairing is complete.
    la_part = [f"{r} ({h}/{w})" for r in routes
               for h, w in [la_cover.get(r, (0, 0))] if h < w]
    if la_part:
        cov_note += (r"  The LA column rests on the instances where the "
                     r"look-ahead has been run under its standard "
                     r"configuration on both drivetrains: " +
                     ", ".join(la_part) + r".")

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
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        r"Route & Diesel (h) & EV (h) & Greedy (\%) & LA (\%) & "
        r"Oracle (\%) & Coupling (\%) & EV cert. (\%) & $n$ \\",
        r"\hline",
    ]
    for r in routes:
        d = per_class[r]
        have, want = coverage[r]
        lines.append(
            f"{r.capitalize()} & {_fmt(_mean(d['dur_d']))} & "
            f"{_fmt(_mean(d['dur_e']))} & "
            f"{_fmt(_mean(d['pen_g']))} & {_fmt(_mean(d['pen_l']))} & "
            f"{_fmt(_mean(d['pen_o']))} & "
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
                d   = per.setdefault(key, dict(pen_o=[], pen_g=[], pen_l=[],
                                               obj_o=[],
                                               d_ev=[], d_di=[], brk_di=[]))
                fuel = _refuel_h(route)
                ev_g, di_g = _policy_pair(st, "GREEDY", "diesel")
                if (ev_g and di_g and not ev_g["infeasible"]
                        and not di_g["infeasible"] and di_g["duration"] > 0):
                    d["pen_g"].append(
                        100 * (ev_g["duration"] / (di_g["duration"] + fuel) - 1))
                ev_l, di_l = _policy_pair(st, "LA", "diesel")
                if (ev_l and di_l and not ev_l["infeasible"]
                        and not di_l["infeasible"] and di_l["duration"] > 0):
                    d["pen_l"].append(
                        100 * (ev_l["duration"] / (di_l["duration"] + fuel) - 1))
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
    x = np.arange(len(tw_order))
    series = [("Oracle", "pen_o", INK), ("LA", "pen_l", GREEN),
              ("Greedy", "pen_g", BLUE)]
    # As in the headline figure: the group takes 0.72 of the unit stride so the
    # window classes stay visually separate as series are added.
    w = 0.72 / len(series)
    top = 0.0
    for ax, route in zip(axes, routes):
        for k, (lbl, key, col) in enumerate(series):
            vals = [_mean(per[(route, tw)][key]) if (route, tw) in per else None
                    for tw in tw_order]
            top  = max([top] + [v for v in vals if v is not None])
            ax.bar(x + (k - (len(series) - 1) / 2) * w,
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
                         len(d["pen_l"]),
                         _fmt(_mean(d["pen_g"]), ".2f", ""),
                         _fmt(_mean(d["pen_l"]), ".2f", ""),
                         _fmt(_mean(d["pen_o"]), ".2f", ""),
                         _fmt(_mean(d["obj_o"]), ".2f", ""),
                         _fmt(_mean(d["d_ev"]), ".2f", ""),
                         _fmt(_mean(d["d_di"]), ".2f", ""),
                         _fmt(_mean(d["brk_di"]), ".2f", "")])
    _write_csv("additional_diesel_tw.csv",
               ["route", "tw", "n_oracle", "n_greedy", "n_la",
                "pen_greedy_%", "pen_la_%",
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
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\hline",
        r"Route & Window & $n_G$ & Greedy (\%) & $n_{LA}$ & LA (\%) & "
        r"$n_O$ & Duration (\%) & "
        r"Objective (\%) & $\delta_{EV}$ & $\delta_{diesel}$ \\",
        r"\hline",
    ]
    # Column order of `rows`, which this indexes positionally:
    #   0 route  1 tw  2 n_O  3 n_G  4 n_LA  5 greedy  6 la  7 duration
    #   8 objective  9 delta_ev  10 delta_diesel  11 diesel_break_h
    for r in rows:
        # n differs by column: Greedy and LA drop instances they cannot
        # schedule feasibly (LA also the ones not yet paired on both
        # drivetrains), the oracle drops classes with incomplete coverage.
        tex.append(f"{r[0].capitalize()} & {r[1].capitalize()} & "
                   f"{r[3]} & {r[5] or '--'} & "
                   f"{r[4]} & {r[6] or '--'} & "
                   f"{r[2] or '--'} & {r[7] or '--'} & {r[8] or '--'} & "
                   f"{r[9] or '--'} & {r[10] or '--'} \\\\")
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


# ---- 8.4 crossed with the one-at-a-time axes -------------------------------
# One figure for the whole design: the base case and every OAT level side by
# side, each drawn as the same stack section_diesel draws, so a column is read
# against the diesel of its own instance rather than against the base case.
# That is the difference from additional_sens_effects, which reports a paired
# delta off the base case; here every column is an absolute penalty and the base
# case is simply one of the columns.
#
# The EV leg carries the axis tag and the diesel leg never does.  That is only
# legitimate where the axis leaves the instance untouched: charger power and
# pack size regenerate the same corridor (identical N, K, C, L, km and D_real)
# and neither quantity exists for a diesel, so the base diesel IS the same
# route.  CS SPACING re-lays the stops -- N, K and the total distance all move,
# and with them the driving time the decomposition relies on cancelling -- so
# pairing a cs30 EV against a base diesel would book a ~0.5 h driving difference
# as electrification cost.  That axis is excluded until it has diesel runs of
# its own.
_OAT_LEVELS = [
    (None,     "base",    "Base"),
    ("kw150",  "power",   "150"),
    ("kw700",  "power",   "700"),
    ("kw1000", "power",   "1000"),
    ("kwh300", "battery", "300"),
    ("kwh700", "battery", "700"),
    ("kwh900", "battery", "900"),
]
_OAT_BLOCKS = [("base", ""), ("power", "Charger power (kW)"),
               ("battery", "Battery capacity (kWh)")]
_OAT_EXCLUDED = ("CS spacing is excluded: it re-generates the corridor, so the "
                 "base diesel is not the same route")


def _oat_components(stem, tag, meth, fuel):
    """One instance stack at one axis level, EV(level) against diesel(base).

    Goes through the SAME pairing rule as the headline figure, with the axis tag
    on the EV leg: pairing each leg independently accepted runs the other figure
    refuses (a leg on the LP tail against one on the MILP tail, or two different
    energy guards), so the two figures disagreed on the base level, which is a
    cell they share and must report identically.
    """
    if meth == "oracle":
        ev, di = _oracle_dwell(stem, tag), _oracle_dwell(stem, "diesel")
        if not (ev and di):
            return None
        return _gap_components(ev, di, fuel)
    ev, di = _policy_pair_tags(stem, meth.upper(), tag, "diesel")
    if not (ev and di) or ev["infeasible"] or di["infeasible"]:
        return None
    return _gap_components(_run_dwell(ev["file"]), _run_dwell(di["file"]), fuel)


def _tot(cell):
    """Stack height of one cell (%), i.e. the penalty it decomposes."""
    return sum(_mean(cell[k]) or 0.0 for k, _l, _c in _GAP_STACK)


def section_diesel_oat():
    """8.4 x 8.3 -- how the electrification penalty RECOMPOSES along the axes.

    One bar per (design point, method), each the same composition
    section_diesel draws.  It answers what the paired-delta sensitivity figure
    cannot: not merely that a bigger pack or a faster charger shrinks the gap,
    but WHICH part of it shrinks.

    Route classes are POOLED.  They differ in level -- a long route pays more
    than a short one at every design point -- but the composition and the way it
    moves along an axis are the same in all three, so splitting them tripled the
    bar count to say one thing three times.  The per-route numbers stay in the
    CSV for anyone who wants to check that.
    """
    from matplotlib.patches import Patch as _Patch, Rectangle as _Rect
    print("== Sec 8.4 x one-at-a-time axes ==")
    print(f"  {_OAT_EXCLUDED}")
    scope  = _discover_scope(["diesel"])
    combos = scope["combos"] or DIESEL_COMBOS
    tws    = scope["tws"]    or TWS
    seeds  = scope["seeds"]  or list(SEEDS)

    methods = [("oracle", "Oracle"), ("LA", "LA"), ("greedy", "Greedy")]
    cells, hours, per_route, coup = {}, {}, {}, {}
    for route, cust in combos:
        for tw in tws:
            for seed in seeds:
                st, fuel = _stem(route, cust, tw, seed), _refuel_h(route)
                di = _oracle_dwell(st, "diesel")
                base_h = (di["_duration"] + fuel) if di else None
                for tag, _blk, _lbl in _OAT_LEVELS:
                    for meth, _m in methods:
                        c = _oat_components(st, tag, meth, fuel)
                        if c is None:
                            continue
                        if c.get("_coupling") is not None:
                            coup.setdefault((tag, meth), []).append(
                                c["_coupling"])
                            coup.setdefault((tag, meth, route), []).append(
                                c["_coupling"])
                        for key in ((tag, meth), (tag, meth, route)):
                            store = cells if len(key) == 2 else per_route
                            cell = store.setdefault(
                                key, {k: [] for k, _l, _c in _GAP_STACK})
                            for k, _l, _c in _GAP_STACK:
                                cell[k].append(c[k])
                        if base_h:
                            # the same total in hours, per instance, so the
                            # pooled mean is a mean of hours and not a
                            # percentage re-scaled by an average duration
                            hours.setdefault((tag, meth), []).append(
                                base_h * sum(c[k] for k, _l, _c in _GAP_STACK)
                                / 100.0)
    if not cells:
        print("  no paired runs yet -- nothing drawn")
        return
    for meth, _m in methods:
        got = sum(len(c["queue"]) for (t, m), c in cells.items() if m == meth)
        nc = len([1 for k in cells if k[1] == meth])
        per = "  ".join(
            f"{lbl}={len(cells[(tag, meth)]['queue'])}"
            for tag, _b, lbl in _OAT_LEVELS if (tag, meth) in cells)
        print(f"  {meth:<7}: {got:>5} paired instance(s) over {nc} level(s)"
              f"   [{per}]")

    x = np.arange(len(_OAT_LEVELS), dtype=float)
    # A gap between the experiment blocks, so "base" reads as the reference and
    # the two axes read as two experiments rather than one seven-point ladder.
    for i, (_t, blk, _l) in enumerate(_OAT_LEVELS):
        x[i] += 0.55 * sum(1 for j in range(i)
                           if _OAT_LEVELS[j][1] != _OAT_LEVELS[j + 1][1])
    # The two gaps have to be tellable apart or the hierarchy inverts: at
    # 0.82/0.30 the space between two methods was a third of the space between
    # two design points, close enough that the eye grouped a method with its
    # neighbour's level.  Methods now sit a hairline apart inside a group that
    # leaves a clear quarter-unit between design points -- roughly nine times
    # the within-group gap.
    _GROUP, _MGAP = 0.76, 0.10
    w = _GROUP / (len(methods) + _MGAP * (len(methods) - 1))
    _total = len(methods) * w + (len(methods) - 1) * _MGAP * w
    offs = [-_total / 2 + (k + 0.5) * w + k * _MGAP * w
            for k in range(len(methods))]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    half = 0.5 + 0.55 / 2
    for blk, _name in _OAT_BLOCKS[1::2]:
        xs = [x[i] for i, (_t, b, _l) in enumerate(_OAT_LEVELS) if b == blk]
        if xs:
            ax.axvspan(min(xs) - half, max(xs) + half, facecolor="#F5F5F5",
                       edgecolor="none", zorder=0)

    span = ax.get_xaxis_transform()
    for li, (tag, _blk, _lbl) in enumerate(_OAT_LEVELS):
        for mi, (meth, mlbl) in enumerate(methods):
            cell = cells.get((tag, meth))
            mcol = ps.METHOD_COLOR["oracle" if meth == "oracle" else meth]
            px = x[li] + offs[mi]
            # The method label goes under every bar, present or not: a missing
            # method then reads as a gap in a named slot rather than as a wider
            # space between the two that did run.
            # Sized to the bar pitch, which the tighter grouping shortened:
            # the colour already says which method this is, so the word only has
            # to confirm it, not carry it.
            ax.annotate(mlbl, xy=(px, 0), xycoords=span, xytext=(0, -4),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=5.2, color=mcol)
            if not cell:
                continue
            vals = _stack_drawable(cell)
            bottom, seams = 0.0, []
            for key, _l, col in _GAP_STACK:
                hgt = vals[key]
                ax.bar(px, hgt, width=w, bottom=bottom, color=col,
                       edgecolor="none", zorder=3)
                bottom += hgt
                if hgt > 1e-9:
                    seams.append(bottom)
            # Separators as lines ACROSS the bar, not as a stroke around each
            # segment: a stroke also paints the left and right edges, which is
            # what left a white halo inside the method ring.
            for yv in seams[:-1]:
                ax.plot([px - w / 2, px + w / 2], [yv, yv], color="white",
                        lw=0.3, solid_capstyle="butt", zorder=3.5)
            ax.add_patch(_Rect((px - w / 2, 0), w, bottom, facecolor="none",
                               edgecolor=mcol, lw=0.8, zorder=4,
                               joinstyle="miter"))
            # Hours only.  The axis already carries the percentage, and two
            # label lines per bar were wider than the bar pitch: wherever two
            # bars of a group landed within a point or two of each other the
            # lines of one ran into the next.
            # No value label.  Three per design point are each wider than the
            # bar they sit on, so wherever two bars of a group land within a
            # point or two of each other -- which is most of the low levels --
            # one label runs into the next.  The axis carries the percentage and
            # the CSV carries both it and the hours.
    ax.axhline(0, color=INK, lw=0.9)

    ax.set_xticks(x, [l for _t, _b, l in _OAT_LEVELS], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=13)
    ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)
    for blk, name in _OAT_BLOCKS:
        if not name:
            continue
        xs = [x[i] for i, (_t, b, _l) in enumerate(_OAT_LEVELS) if b == blk]
        ax.annotate(name, xy=(sum(xs) / len(xs), 0), xycoords=span,
                    xytext=(0, -33), textcoords="offset points", ha="center",
                    va="top", fontsize=8.5, color=ps.INK_PRIMARY)
    ax.set_ylabel("Route duration vs. diesel (%)", fontsize=8.5)
    # Labelled every 5 points and ruled every 1, with real ticks on both: the
    # differences the figure is read for are one to two points wide, but nine
    # unlabelled lines between majors is a haze rather than a scale.
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.yaxis.grid(True, which="major", color=GRID, lw=0.6)
    ax.yaxis.grid(True, which="minor", color="#F0F0F0", lw=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", which="major", length=3.2, width=0.7,
                   labelsize=8, color=ps.BASELINE)
    ax.tick_params(axis="y", which="minor", length=1.8, width=0.5,
                   color=ps.BASELINE)
    # Full frame, as the headline 8.4 figure has: this one sits beside it in the
    # same section, and a half-open panel next to a closed one reads as an
    # accident rather than a choice.
    for sp in ax.spines.values():
        sp.set_edgecolor(ps.BASELINE)
        sp.set_linewidth(0.7)

    handles = [_Patch(facecolor=c, edgecolor="white", lw=0.5, label=l)
               for _k, l, c in _GAP_STACK]
    # Rounded up to the next 10 rather than scaled off the tallest stack: a
    # fixed ceiling keeps the panel comparable when a level is re-run and the
    # tallest bar moves.
    ax.set_ylim(0, 40)
    ax.legend(handles=handles, frameon=True, framealpha=0.95,
              edgecolor="none", facecolor="white", fontsize=6.4, ncol=2,
              loc="upper right", handlelength=1.0, handletextpad=0.4,
              columnspacing=1.0, labelspacing=0.35, borderpad=0.45)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "additional_diesel_oat")

    # Pooled rows first, then the per-route detail the pooling hides.
    rows = []
    for tag, _blk, lbl in _OAT_LEVELS:
        for meth, _m in methods:
            if (tag, meth) not in cells:
                continue
            c = cells[(tag, meth)]
            rows.append(["all", lbl, meth, len(c["queue"])]
                        + [_fmt(_mean(c[k]), ".3f", "")
                           for k, _l, _c in _GAP_STACK]
                        + [_fmt(_tot(c), ".3f", ""),
                           _fmt(_mean(hours.get((tag, meth), [])), ".3f", "")])
    for route in [r for r in DIESEL_ROUTES
                  if any(k[2] == r for k in per_route)]:
        for tag, _blk, lbl in _OAT_LEVELS:
            for meth, _m in methods:
                c = per_route.get((tag, meth, route))
                if not c:
                    continue
                rows.append([route, lbl, meth, len(c["queue"])]
                            + [_fmt(_mean(c[k]), ".3f", "")
                               for k, _l, _c in _GAP_STACK]
                            + [_fmt(_tot(c), ".3f", ""), ""])
    _write_csv("additional_diesel_oat.csv",
               ["route", "level", "method", "n"]
               + [f"{k}_pct" for k, _l, _c in _GAP_STACK]
               + ["total_pct", "total_h"], rows)
    _diesel_oat_coupling(coup, methods)


def _diesel_oat_coupling(coup, methods) -> None:
    """Mean coupling per design point: how much charging hides inside a break.

    Read beside the stack, it says why the charging split moves: a level whose
    coupling falls is one where the charger can no longer be fed into the breaks
    the route already has, so the charging surfaces as makespan.

    Computed over the same paired instances the figure draws, so the two agree
    run for run.  The oracle's g is the model's own; a policy has no g in its
    record, so it is recovered by the rule the MILP uses for it (the charge at a
    stop, capped at the break block taken there).
    """
    print("  mean coupling, share of charging inside a mandatory break (%)")
    hdr = f"  {'level':<10}" + "".join(f"{lbl:>16}" for _m, lbl in methods)
    print(hdr)
    # The figure disambiguates the levels with the block label underneath it;
    # a table row has no such context, and "700" is both a charger power and a
    # pack size, so the unit travels with the level here.
    unit = {"power": " kW", "battery": " kWh", "base": ""}
    rows_csv, body = [], []
    for tag, _blk, lbl in _OAT_LEVELS:
        lbl = lbl + unit.get(_blk, "")
        line = f"  {lbl:<10}"
        tex_cells = []
        for meth, _mlbl in methods:
            v = coup.get((tag, meth), [])
            mv = _mean(v)
            line += (f"{mv:>11.1f} ({len(v):>3})" if mv is not None
                     else f"{'--':>16}")
            tex_cells.append(f"{mv:.1f}" if mv is not None else "--")
            rows_csv.append([lbl, meth, len(v), _fmt(mv, ".2f", "")])
        print(line)
        body.append(f"{lbl} & " + " & ".join(tex_cells) + r" \\")

    _write_csv("additional_diesel_oat_coupling.csv",
               ["level", "method", "n", "coupling_pct"], rows_csv)
    tex = [
        r"\begin{table}[ht]\centering",
        r"\caption{Coupling by design point: the share of electric charging "
        r"time that runs inside a mandatory break, and so costs no makespan "
        r"beyond the break itself ($\Sigma g_i / \Sigma \tau^c_i$).  Averaged "
        r"over the same paired instances as Figure~\ref{fig:diesel-oat}, route "
        r"classes pooled.  For the hindsight optimum $g$ is the model's own; a "
        r"simulated policy records none, so it is recovered by the rule the "
        r"model uses for it, the charge at a stop capped at the break taken "
        r"there.}",
        r"\label{tab:diesel-oat-coupling}",
        r"\begin{tabular}{l" + "r" * len(methods) + r"}",
        r"\hline",
        r"Design point & " + " & ".join(lbl for _m, lbl in methods) + r" \\",
        r"\hline",
    ] + body + [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write_tex("additional_diesel_oat_coupling.tex", "\n".join(tex))


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
        for path in _paths.glob_solutions(f"*__{tag}_*.json"):
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
                    bg, vg = _policy_pair(st, "GREEDY", tag)
                    if (bg and vg and not bg["infeasible"]
                            and not vg["infeasible"] and bg["duration"] > 0):
                        dg[route].append(
                            100 * (vg["duration"] / bg["duration"] - 1))
                    # LA is paired exactly like greedy: both legs must exist and
                    # both must be feasible, so the delta is never contaminated
                    # by a run that stranded on one side only — and both must be
                    # the same configuration, which is what _policy_pair adds.
                    bl, vl = _policy_pair(st, "LA", tag)
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
    # only wide enough to read as a break, and the separating work is done by
    # RULES — which cost no width at all.  Two weights, because there are two
    # nested groupings to show: a full-height rule between experiments (which
    # also runs down past the ticks to split the block labels) and a light
    # hairline between the levels inside one experiment.
    _grp = [_sens_group(t) for _l, t, _p in _SENS_ROWS]
    bounds = [i for i in range(1, len(_grp)) if _grp[i] != _grp[i - 1]]
    _BLOCK_GAP = 0.30
    _RULE_BLOCK = "#8c8c8c"   # between experiments
    _RULE_LEVEL = "#dcdcdc"   # between levels of one experiment
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
    # 7.5 pt ticks at about 3.5 pt on the page.  Sizing near the text block keeps
    # the drawn point size close to the printed point size.  Landscape band: the
    # 72 bars need horizontal room more than the response axis needs height, and
    # the legend parks in the top-right corner rather than over the middle.
    _FIG_W, _FIG_H = 9.0, 3.3
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    drawn_m, drawn_r = set(), set()

    # Separating rules, drawn under the bars.  Every gap between neighbouring
    # levels gets a line; the ones that straddle an experiment boundary get the
    # heavier colour and are extended below the axes so they also split the
    # block labels, which is what the alternating background band used to do.
    half = 0.5 + _BLOCK_GAP / 2
    span_tr = ax.get_xaxis_transform()   # x data, y axes-fraction
    for i in range(1, len(fig_rows)):
        xm = (x[i - 1] + x[i]) / 2
        if i in bounds:
            ax.plot([xm, xm], [-0.19, 1.0], transform=span_tr,
                    color=_RULE_BLOCK, lw=0.8, zorder=0.6, clip_on=False)
        else:
            ax.plot([xm, xm], [0.0, 1.0], transform=span_tr,
                    color=_RULE_LEVEL, lw=0.6, zorder=0.4)

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
    # Fixed top, not derived headroom: 28% clears the tallest bar (150 kW) with
    # room for the two legend rows in the top-right corner, and holding it fixed
    # keeps the bar heights comparable if a later run shifts the maximum.
    ax.set_ylim(lo, max(28.0, hi))
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
                  loc="upper right", ncol=min(3, len(handles)),
                  borderaxespad=0.5,
                  handlelength=1.1, handletextpad=0.4, columnspacing=1.4)
    # Bottom reserve: the block labels hang below the axes and tight_layout
    # does not measure annotations drawn outside them.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
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
_LA_POLICY_ORDER = ["TB0", "LPTAIL"]
_LA_POLICY_LBL = {
    "TB0":    "TB0 (no 5-min tie-break)",
    "LPTAIL": "LPTAIL (LP look-ahead)",
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


# Route scope of the LA figures.  All three classes — but only because
# la-report now reports every cell on a BALANCED PANEL (the instances carrying
# an OK run in EVERY cell), which is what makes pooling them legitimate again.
#
# Without that panel it was not.  A pooled cell's median is a median over
# whatever route mix the cell happens to have, and the mixes were not equal:
# S25H48/MIP was 53% short-route runs and 5% long, so its pooled cost landed
# inside the short group at 27 s/stop while the balanced base cell landed at
# 72 — which reads as "a longer horizon is cheaper", the opposite of the truth
# (455 s/stop on long routes alone).  On the panel every cell holds the same
# instances, so a difference between cells cannot be a difference in
# population.  Pass --panel all to la-report to go back to the unbalanced
# reading; these figures will then be comparing route mixes again.
# Short routes are left out (2026-08-22, on request).  A short route is about
# one duty cycle long, so the look-ahead has almost nothing to look ahead AT
# and the configuration axes flatten out there; the classes that carry the
# question are medium and long.  Dropping the class also halves the cost range
# the figures have to span, which is what lets the cost axes stay linear.
# The TABLE still carries all three classes.
_LA_FIG_ROUTES = ["medium", "long"]


# The two look-ahead TAIL SOLVERS, defined once and used by every LA figure.
# (lptail, label, dash, marker).  Same policy, same ladder; only the subproblem
# beyond the horizon differs, so they share a hue and separate on dash+marker.
_LA_TAIL_SERIES = (
    (False, "MILP subpr.", "-",  "o"),
    (True,  "LP subpr.",   "--", "s"),
)


def _la_tail_color(lptail: bool):
    """Hue for a tail solver.  LA's own colour, lightened for the LP tail.

    Used by the ladder and configuration figures, where every series is the LA
    policy and the tail is the only thing separating them, so the method hue is
    the right family.  `additional_la_plane` deliberately does NOT use this: it
    shares the plane with `la_all`, whose series colour already means "solver
    family" in black/purple, and one plane reading two colour schemes is worse
    than two figures reading one.
    """
    return (ps.tint(ps.METHOD_COLOR["LA"], 0.45) if lptail
            else ps.METHOD_COLOR["LA"])


def _la_solver_color(lptail: bool):
    """Solver-family hue, matching la_all: MILP near-black, LP purple."""
    return _LA_CLUSTER_HUES[1] if lptail else _LA_CLUSTER_HUES[0]


# Axis labels shared by the configuration figure and the ladder figures, so the
# same axis is never named two ways across the section.
_LBL_HORIZON      = r"Look-ahead horizon $L$ (h)"
_LBL_SCEN         = r"Scenarios $|\Xi|$"
_LBL_HORIZON_HELD = _LBL_HORIZON + r"   [$|\Xi| = 25$]"
_LBL_SCEN_HELD    = _LBL_SCEN + r"   [$L = 24$ h]"
_LBL_GAP          = "Gap to hindsight optimum (%)"
_LBL_COST         = "Decision time per CS stop (s)"

# When a cell is too thin to read as an estimate.  ABSOLUTE, and the same
# number in every figure of the section.  It used to be a relative rule (below
# 60% of the best-covered cell), which made sense while the cells held
# different instances — but on the balanced panel every cell of a given facet
# holds the SAME instances, so a relative rule either never fires or fires on
# all of them at once, depending on which rows happen to enter the reference.
# What is still worth flagging is a facet that is small in absolute terms: the
# long route class carries 3 of the panel's 70 instances.
_LA_THIN_N = 5


def _thin_note(seen: dict) -> str:
    """'  Hollow markers: long route, n = 3.' — or nothing when all cells are
    full.  One sentence for the whole figure: with the route class as the
    facet, every cell of a thin class carries the same n, so labelling each
    point restated one fact once per series."""
    if not seen:
        return ""
    parts = ", ".join(f"{ps.ROUTE_LBL[r].lower()} n = {n:g}"
                      for r, n in sorted(seen.items(),
                                         key=lambda kv: ps.ROUTE_ORDER.index(kv[0])))
    return f"  Hollow markers mark a thin cell ({parts})."


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
    """8.3 - the two configuration ladders, quality over cost, plus the risk
    strip underneath.

    Draws the SAME ten cells as `la_ladders` and `la_all`: five configurations
    crossed with the two look-ahead tail solvers (_LA_TAIL_SERIES).  It used to
    draw only the five MIP-tail ones while the other two figures of the section
    drew ten and twelve, so a configuration could appear in one figure and not
    in another.

    Route classes are POOLED, and the series dimension that frees up is spent
    on the tail solver instead.  Pooling is legitimate now and was not before:
    la-report reports every cell on a balanced panel, so the pooled cells hold
    the same instances (see _LA_FIG_ROUTES).  The per-route breakdown lives in
    `la_all` and in the table, which read the same rows at a finer grain.
    """
    print("== Sec 8.3 look-ahead configuration ==")
    stats = _la_stats()
    routes = _LA_FIG_ROUTES

    # Columns are the two axes of the one-at-a-time design, rows are the two
    # things a configuration trades off.  Sharing the y-axis WITHIN a row is
    # the whole point: the reader compares the shape of the horizon response
    # against the shape of the scenario response, and any difference in slope
    # is then real rather than an artefact of two independent scales.
    # A third, short row carries the infeasibility heat strip - the same device
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

    def _cell(v, is_h, lptail, route, col):
        """One rung of one ladder, for one tail solver and one route class.

        Window classes are POOLED into the route's 'all' row — the same row
        `la_all` reads — so a value shown here is the same number that figure
        shows for the same cell.
        """
        ns, hh = (_LA_BASE[0], float(v)) if is_h else (int(v), _LA_BASE[1])
        return _la_num(stats.get((_la_tail_cfg(ns, hh, lptail), route, "all")),
                       col)

    thin_seen = {}          # route class -> its run count, when below the floor
    for ax_q, ax_c, ladder, is_h in ((ax_hq, ax_hc, _LA_HORIZONS, True),
                                     (ax_sq, ax_sc, _LA_SCENARIOS, False)):
        drew = False
        for route in routes:
            col = _LA_ROUTE_SHADE[route]
            mk  = _LA_ROUTE_MARK[route]
            for lptail, _lbl, sty, _m in _LA_TAIL_SERIES:
                xs, gq, gc, nn = [], [], [], []
                for v in ladder:
                    q = _cell(v, is_h, lptail, route, "gap_pen_median_pct")
                    c = _cell(v, is_h, lptail, route, "decision_cs_mean_s_median")
                    if q is None and c is None:
                        continue
                    xs.append(v); gq.append(q); gc.append(c)
                    nn.append(_cell(v, is_h, lptail, route, "n_runs"))
                if not xs:
                    continue
                drew = True
                for ax, ys in ((ax_q, gq), (ax_c, gc)):
                    pts = [(x, y, n) for x, y, n in zip(xs, ys, nn)
                           if y is not None]
                    if not pts:
                        continue
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            sty, color=col, marker=mk, ms=3.8, lw=1.3,
                            mfc=col, mec=col)
                    # Thin cells keep the marker OUTLINE and lose the fill.
                    # How thin they are is stated once, in the footnote.
                    for x, y, n in pts:
                        if n and n < _LA_THIN_N:
                            thin_seen[route] = n
                            ax.plot(x, y, mk, ms=3.8, mfc="none", mec=col,
                                    mew=1.0, zorder=4)
                    # The base cell is the reference the sweep is quoted
                    # against, so it is marked rather than left as one dot
                    # among three.
                    bx = _LA_BASE[1] if is_h else _LA_BASE[0]
                    for x, y, _n in pts:
                        if x == bx:
                            ax.plot(x, y, mk, ms=7.5, mfc="none", mec=col,
                                    lw=0.9)
        if not drew:
            for ax in (ax_q, ax_c):
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                        fontsize=7.5, color=MUT, style="italic",
                        transform=ax.transAxes)

    # Both ladders are RATIO ladders (12/24/48, 10/25/50), so a log x-axis puts
    # the rungs at equal distances and the tick set is exactly the rungs - no
    # auto-ticks between them, no minor ticks to imply a continuum that was
    # never sampled.
    for ax_st, ladder in ((st_h, _LA_HORIZONS), (st_s, _LA_SCENARIOS)):
        rungs = [float(v) for v in ladder]
        ax_st.set_xscale("log")     # sharex carries this up the whole column
        ax_st.set_xticks(rungs, [f"{v:g}" for v in rungs])
        ax_st.xaxis.set_minor_locator(mticker.NullLocator())
        ax_st.set_xlim(min(rungs) / 1.22, max(rungs) * 1.22)

    st_h.set_xlabel(_LBL_HORIZON_HELD)
    st_s.set_xlabel(_LBL_SCEN_HELD)
    ax_hq.set_ylabel("Gap to hindsight\noptimum (%)")
    ax_hc.set_ylabel(_LBL_COST.replace(" per", "\nper"))

    # Both response axes are LINEAR and evenly ticked.  The cost axis used to be
    # log so that equal ratios read as equal distances, but a scale whose grid
    # spacing changes at 10 s invites the reader to measure it as if it were
    # linear and get the wrong slope; a uniform grid costs some resolution at
    # the cheap end and lies about nothing.
    # Both response rows are LINEAR and evenly ticked.  A log cost axis was
    # tried and dropped: a scale whose grid spacing changes partway invites the
    # reader to measure it as if it were linear and get the slope wrong, and
    # with the short routes gone the range no longer forces the issue.
    for ax in axes.ravel():
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.grid(True, which="minor", axis="y", color=GRID, lw=0.35, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", which="minor", length=2)
        ax.tick_params(axis="x", which="minor", length=0)
        # Anchored at ZERO, both rows.  A gap and a decision time are both
        # ratio quantities — the distance from zero is the quantity — so a
        # floating baseline exaggerates every difference between cells by
        # whatever the axis happens to crop.  The extra headroom on top is for
        # the n= labels on thin cells, which sit above their marker.
        _lo, hi = ax.get_ylim()
        ax.set_ylim(0, hi * 1.09 if hi > 0 else 1.0)

    # -- infeasibility heat strip --------------------------------------------
    # Same traffic-light ramp as the base-case figure and, as there, scaled to
    # the WORST rate actually observed so the red end marks a real cell rather
    # than a hypothetical 100%.  One ROW per tail solver (the series dimension
    # of the panels above) and, at each rung, one cell per window class:
    # solid-line class ("none") left, dashed ("tight") right.
    from matplotlib.patches import Rectangle as _Rect
    _reds = _INFEAS_CMAP

    def _infeas(v, is_h, lptail, route):
        n = _cell(v, is_h, lptail, route, "n_runs")
        i = _cell(v, is_h, lptail, route, "n_infeasible")
        return (i / n) if (n and i is not None) else None

    fracs = [f for _ax, ladder, is_h in ((st_h, _LA_HORIZONS, True),
                                         (st_s, _LA_SCENARIOS, False))
             for v in ladder for lp, _l, _s, _m in _LA_TAIL_SERIES
             for route in routes
             for f in [_infeas(v, is_h, lp, route)] if f]
    fmax = max(fracs) if fracs else 1.0

    # Cells are sized in DEX because the x-axis is log: a fixed multiplicative
    # half-width keeps every group the same visual width at every rung.
    _HALF_DEX = 0.085
    tails = list(_LA_TAIL_SERIES)
    for ax_st, ladder, is_h in ((st_h, _LA_HORIZONS, True),
                                (st_s, _LA_SCENARIOS, False)):
        ax_st.set_ylim(0, len(routes))
        ax_st.set_yticks([])
        ax_st.tick_params(axis="x", length=0)
        for sp in ax_st.spines.values():
            sp.set_visible(False)
        for v in ladder:
            for ri, route in enumerate(routes):
                y0 = len(routes) - 1 - ri + 0.10      # short on top
                for ti, (lptail, _l, _s, _m) in enumerate(tails):
                    f = _infeas(v, is_h, lptail, route)
                    if f is None:
                        continue                     # not run -> left blank
                    l0 = np.log10(float(v)) - _HALF_DEX + ti * _HALF_DEX
                    x0, x1 = 10 ** l0, 10 ** (l0 + _HALF_DEX)
                    ax_st.add_patch(_Rect(
                        (x0, y0), x1 - x0, 0.80,
                        facecolor=_reds(min(1.0, f / fmax)),
                        edgecolor="#8a8a8a", lw=0.35, zorder=3))

    # Row key on the left panel only, in the route shades the lines use.
    for ri, route in enumerate(routes):
        st_h.annotate(ps.ROUTE_LBL[route][0], xy=(0, len(routes) - 0.5 - ri),
                      xycoords=("axes fraction", "data"),
                      xytext=(-3, 0), textcoords="offset points",
                      ha="right", va="center", fontsize=5.6,
                      color=_LA_ROUTE_SHADE[route])
    # Which half of a pair is which tail solver, stated once.
    for ti, (_lp, lbl, _s, _m) in enumerate(tails):
        l0 = np.log10(float(_LA_HORIZONS[0])) - _HALF_DEX + (ti + 0.5) * _HALF_DEX
        st_h.annotate(lbl[0], xy=(10 ** l0, len(routes)),
                      xytext=(0, 1.5), textcoords="offset points",
                      ha="center", va="bottom", fontsize=5.0, color=MUT)
    st_h.set_ylabel("Infeas.\nrate", fontsize=5.8, color=MUT, labelpad=20,
                    linespacing=0.95)

    handles = [plt.Line2D([], [], color=_LA_ROUTE_SHADE[r], lw=1.4,
                          marker=_LA_ROUTE_MARK[r], ms=3.8) for r in routes]
    labels  = [ps.ROUTE_LBL[r] for r in routes]
    handles += [plt.Line2D([], [], color=MUT, lw=1.2, ls=sty)
                for _lp, _l, sty, _m in _LA_TAIL_SERIES]
    labels  += [l for _lp, l, _s, _m in _LA_TAIL_SERIES]
    # Explicit margins rather than tight_layout: the heat-strip axes carry only
    # patches and annotations, which tight_layout cannot measure (it warns and
    # guesses).  Fixed margins also make the colourbar placement below exact.
    fig.subplots_adjust(left=0.135, right=0.925, top=0.895, bottom=0.105,
                        hspace=0.16, wspace=0.12)
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 0.995), handlelength=1.6,
               handletextpad=0.4, columnspacing=1.4)

    # Compact colour key for the strip, tucked to its right (as in the
    # base-case figure).  Placed after subplots_adjust so positions are final.
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
    fig.text(0.005, 0.008,
             "Window classes pooled; short routes excluded.  Cost is the measured decision time at CS stops."
             + _thin_note(thin_seen),
             fontsize=4.6, color=MUT, ha="left")

    _save(fig, "additional_la_config")


    # The TABLE keeps the per-route breakdown the figure pools away, and the
    # MIP tail only: it is the standard configuration, and doubling every row
    # for the LP tail would restate the ladder figure at ten times the length.
    # The LP tail is reported by tab:la_policy, on the same rows of the same
    # CSV, so the two never disagree.
    routes = _LA_FIG_ROUTES
    tws    = ["none", "tight"]

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
    # Same thin-cell rule as the figure, in the form a table can carry: a cell
    # resting on well under the runs the design gives it is daggered rather
    # than printed as if it were one of the full ones.  Without it the L48
    # long-route column reads as a 455 s measurement instead of as four runs.
    n_max = max([n for cfg, _s, _h in order for tw in tws for route in routes
                 for n in [_la_num(stats.get((cfg, route, tw)), "n_runs")]
                 if n] or [0])
    thin = []
    for cfg, ns, hh in order:
        for tw in tws:
            cells = []
            for route in routes:
                row = stats.get((cfg, route, tw))
                gap = _la_num(row, "gap_pen_median_pct")
                dlt = _la_num(row, "delta_vs_base_pct")
                dec = _la_num(row, "decision_cs_mean_s_median")
                n   = _la_num(row, "n_runs")
                mark = ""
                if n and n_max and n < 0.6 * n_max:
                    mark = r"$^{\dagger}$"
                    thin.append(f"$|\Xi| = {ns}$, $L = {hh:g}$, "
                                f"{ps.TW_LBL[tw].lower()} windows, "
                                f"{ps.ROUTE_LBL[route].split()[0].lower()} "
                                f"($n = {n:g}$)")
                cells += [_fmt(gap, ".1f") + mark,
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
        r"is the median decision time at charging-station stops, measured per "
        r"stop in the run logs.  Every cell is quoted over seeds 1--10.  Cells "
        r"shown as ``--'' have no runs yet.}",
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
    lines += [r"\bottomrule", r"\end{tabular}"]
    if thin:
        lines.append(
            r"\\[2pt]{\footnotesize $^{\dagger}$ fewer than 60\% of the runs "
            r"the best-covered cell of the table has, so the median is thin: "
            + "; ".join(sorted(set(thin))) + r".}")
    lines += [r"\end{table}", ""]
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
    """§8.3 — LA POLICY variants (TB0, LPTAIL): one table, no figure.

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
        print("  no policy variants in additional_la_stats.csv — no table")
        return

    def cell(cfg, route, tw, col):
        return _la_num(stats.get((cfg, route, tw)), col)

    # No figure.  The bar panel this section used to draw was dropped
    # (2026-08-22): its six x-slots are the same route x window cells the
    # configuration figure already carries, and the policy variants have no
    # position on a ladder, so the picture added an axis of comparison the
    # table states more precisely.  The table below is the whole output.
    #
    # `drawn` is what the figure's first row used to decide: a variant is
    # reported when it has at least one paired effect value somewhere on the
    # grid.  A variant with none is named as pending instead of being printed
    # as a row of dashes.
    drawn = [c for c in cfgs
             if any(cell(c, r, t, eff["col"]) is not None
                    for r, t in [(r, t) for r in routes for t in tws])]
    pending = [c for c in cfgs if c not in drawn]
    if pending:
        print("  pending (no paired runs yet): " + ", ".join(pending))

    # ── table ────────────────────────────────────────────────────────────────
    body = []
    for cfg in ["base"] + drawn:
        for tw in tws:
            cells = []
            for route in routes:
                cells += [_fmt(cell(cfg, route, tw, "gap_pen_median_pct"), ".1f"),
                          "--" if cfg == "base"
                          else _fmt(cell(cfg, route, tw, eff["col"]), "+.2f"),
                          _fmt(cell(cfg, route, tw, "decision_cs_mean_s_median"),
                               ".0f")]
            body.append((f"{cfg} & {ps.TW_LBL[tw]}", cells))

    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Look-ahead POLICY variants, against the base cell "
        r"($|\Xi| = 25$, $L = 24$\,h).  TB0 removes the 5-minute tie-break that "
        r"buys opportunistic charging; LPTAIL solves the look-ahead tail as an "
        r"LP relaxation instead of the MIP the base cell uses.  Gap is the "
        r"median gap to the hindsight optimum, $\Delta$ the median paired "
        r"change in "
        + eff["tex"] +
        r" (positive = worse), $t_{\text{dec}}$ the median decision time at "
        r"charging-station stops, measured per stop in the run logs.  Every "
        r"cell is quoted over seeds 1--10.}",
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
_LA_POLICY_MARK = {"LPTAIL": "D", "TB0": "v"}

# ── the cost/quality plane, organised as SOLVER CLUSTERS ─────────────────────
# A cell id from la-report is a set of "+"-joined tokens: an optional ladder rung
# (S<n>H<h>), and flags for the subproblem solver (LPTAIL = the superseded LP
# tail; its absence = the MILP standard), the committed-charge energy guard
# (EQ<pp>), and the break regime (NOSPLIT).  Parsing the id rather than matching
# whole strings is what lets a cell nobody declared still land in the right
# place: a new flag opens a new cluster, a new rung joins an existing one, and
# nothing has to be enumerated here in advance.
#
# CLUSTER = the flags other than NOSPLIT.  Each cluster is one configuration
# family with its own base cell, and every variant in it — both ladders and the
# no-split regime — is drawn as a leader from THAT base, never across families.
# That is the point of the split: a horizon change measured against the LP base
# and one measured against the MILP base are different quantities, and a single
# shared anchor silently equated them.
#
# COLOUR distinguishes clusters; MARKER distinguishes the variant within one.
_LA_LADDER_RE = re.compile(r"^S(\d+)H(\d+(?:\.\d+)?)$", re.I)
_LA_VARIANTS = [
    (None,      "base",     "o"),
    ("S25H12",  "L12",      "v"),
    ("S25H48",  "L48",      "^"),
    ("S10H24",  "S10",      "<"),
    ("S50H24",  "S50",      ">"),
]
# Assigned in the order clusters are first seen.  Kept clear of every method hue
# (blue, vermillion, orange, green, reddish purple, grey) and of the
# green/yellow/vermillion the infeasibility fills run through.
_LA_CLUSTER_HUES = ["#1A1A1A", "#6A3D9A", "#C2007A", "#00688B", "#8C6D31"]
# Cell ids the CSV may carry that must never become a configuration cell here:
# other methods, and ad-hoc probes.  Prefix-matched for the LOCAL family, whose
# tags arrive under several spellings (LOCAL, LOCAL_MIPTAIL, LOCAL+LPTAIL, ...);
# an exact-match list let each new one through as a spurious cell.
# Cells no LA CONFIGURATION figure draws.  GREEDY is the do-nothing reference
# rather than a configuration; TB0 and LOCAL are other studies; the NOSPLIT
# arms belong to the break-regime axis, which `la_config` and the ladders do
# not carry either — leaving them in was the third way the three figures of
# this section disagreed about what they were showing.
_LA_ALL_SKIP = {"GREEDY", "TB0", "NOSPLIT"}
_LA_ALL_SKIP_PREFIX = ("LOCAL",)
_LA_ALL_SKIP_SUFFIX = ("+NOSPLIT",)


def _la_parse_cell(cfg):
    """'S25H12+LPTAIL' -> (cluster key, variant key).

    cluster : frozenset of flags that are neither a ladder rung nor the break
              regime — i.e. what makes this a separate configuration family.
    variant : the ladder rung, or "NOSPLIT", or None for the family's base.
    """
    ladder, flags = None, []
    for tok in (cfg or "").split("+"):
        if not tok or tok == "base":
            continue
        if _LA_LADDER_RE.match(tok):
            ladder = tok.upper()
        else:
            flags.append(tok.upper())
    nosplit = "NOSPLIT" in flags
    cluster = frozenset(f for f in flags if f != "NOSPLIT")
    return cluster, ("NOSPLIT" if (nosplit and ladder is None) else ladder)


def _la_cell_id(cluster, variant):
    """Inverse of _la_parse_cell: the CSV key for one (cluster, variant)."""
    toks = ([] if variant in (None, "NOSPLIT") else [variant])
    toks += sorted(cluster)
    if variant == "NOSPLIT":
        toks.append("NOSPLIT")
    return "+".join(toks) or "base"


def _la_cluster_label(cluster):
    """Reader-facing name for a configuration family."""
    solver = "LP subpr." if "LPTAIL" in cluster else "MILP subpr."
    extra = []
    for f in sorted(cluster):
        if f == "LPTAIL":
            continue
        if f.startswith("EQ"):
            extra.append(f"energy guard {int(f[2:]) / 100:g}")
        else:
            extra.append(f.title())
    return solver + (", " + ", ".join(extra) if extra else "")


def section_la_all(csv_name="additional_la_stats.csv",
                   outname="additional_la_all",
                   banner="cost/quality plane by solver cluster"):
    """8.3 - every LA configuration on one cost/quality plane, in solver clusters.

    Cost is the MEASURED DECISION TIME AT CS STOPS (la-report's
    decision_cs_mean_s_median), parsed per stop from the run logs.  Charging
    stations are the stops where the decision actually has branching structure
    — charge or not, how long, and the break/rest interaction on top — so their
    cost is the wait an operator would feel.  Averaging over every stop dilutes
    that with laybys and customers, which enumerate far fewer actions and run
    about half as long.

    One panel per route class, window classes pooled.  Each configuration family
    (LP subproblem, MILP subproblem, MILP + energy guard, ...) forms a cluster:
    its own base cell, with the two ladders and the no-split regime drawn as
    leaders out of it.  Cluster membership is PARSED from the cell ids that
    la-report writes, so a family nobody anticipated appears on its own.
    """
    print(f"== Sec 8.3 look-ahead - {banner} ==")
    stats = _la_stats(csv_name)
    if not stats:
        print(f"  data_output/{csv_name} missing - nothing drawn")
        return
    routes, TW = _LA_FIG_ROUTES, "all"
    if not any(t == TW for (_c, _r, t) in stats):
        print("  no pooled 'all' window rows - re-run: python -m "
              "src.output_analysis.additional_analysis la-report")
        return

    # A thin cell is FLAGGED, not dropped.  The hard minimum this used to apply
    # made the section inconsistent with itself: on the balanced panel the long
    # route class carries 3 instances, so a floor of 5 silently removed every
    # long-route panel here while `la_config` and the table went on reporting
    # those very cells.  A figure that hides what its neighbour shows is worse
    # than one that shows a small number and says so, which is what the hollow
    # marker and the n= label do (the same convention the ladder figure uses).
    def cell(cfg, route):
        row = stats.get((cfg, route, TW))
        c = _la_num(row, "decision_cs_mean_s_median")
        q = _la_num(row, "gap_pen_median_pct")
        if c is None or q is None:
            return None
        n, i = _la_num(row, "n_runs"), _la_num(row, "n_infeasible")
        thin = bool(n and n < _LA_THIN_N)
        return c, q, ((i / n) if (n and i is not None) else None), n, thin

    # discover the configuration families actually present
    have = {c for (c, _r, _t) in stats
            if c not in _LA_ALL_SKIP
            and not c.upper().startswith(_LA_ALL_SKIP_PREFIX)
            and not c.upper().endswith(_LA_ALL_SKIP_SUFFIX)}
    clusters = {}
    for cfg in have:
        cl, var = _la_parse_cell(cfg)
        clusters.setdefault(cl, {})[var] = cfg
    if not clusters:
        print("  no LA configuration cells found - nothing drawn")
        return
    # MILP standard first (no flags), then LP, then the rest by label, so the
    # colour a family gets does not shuffle when another one appears later.
    order = sorted(clusters, key=lambda c: (bool(c), "LPTAIL" not in c,
                                            _la_cluster_label(c)))
    hue = {cl: _LA_CLUSTER_HUES[i % len(_LA_CLUSTER_HUES)]
           for i, cl in enumerate(order)}
    mark = {v: m for v, _lbl, m in _LA_VARIANTS}

    vals = [got for cl, mem in clusters.items() for cfg in mem.values()
            for r in routes for got in [cell(cfg, r)] if got]
    if not vals:
        print("  no cell carries both a cost and a gap - nothing drawn")
        return
    quals = [v[1] for v in vals]
    fmax = max([v[2] for v in vals if v[2]] or [1.0])

    # x is scaled PER ROUTE, y is shared.  The cost axis spans an order of
    # magnitude between route classes — the long panel carries a 455 s cell
    # while the short one tops out near 40 — so one shared cost axis renders
    # two of the three panels as a vertical stripe against the spine.  The gap
    # axis stays shared, because comparing quality across route classes is the
    # thing the reader is actually here for.  Markers are drawn at the data
    # point but have width, so the left margin goes slightly negative rather
    # than clipping the cheapest cell against the spine.
    def _xlim(route):
        cs = [got[0] for cl, mem in clusters.items() for cfg in mem.values()
              for got in [cell(cfg, route)] if got]
        hi = max(cs) if cs else 1.0
        return (0.0, hi * 1.16)
    # From zero, like every other gap axis in the section.  This used to be a
    # window centred on the data, which magnified a two-point spread into the
    # full height of the panel.
    ylim = (0.0, max(max(quals) * 1.18, 1.0))

    fig, axs = plt.subplots(1, len(routes), figsize=(7.2, 3.3),
                            sharey=True)
    axs = np.atleast_1d(axs)

    thin_seen = {}          # route class -> its run count, when below the floor
    unknown = set()         # cells with no marker of their own -> drawn "*"
    for ri, route in enumerate(routes):
        ax = axs[ri]
        drew = False
        for cl in order:
            col, mem = hue[cl], clusters[cl]
            base = cell(mem[None], route) if None in mem else None
            # Leaders first, under the markers: every variant is a displacement
            # from its OWN family's base, so a family reads as a fan and two
            # families never share a line.
            if base:
                for var, cfg in mem.items():
                    got = cell(cfg, route) if var is not None else None
                    if got:
                        ax.plot([base[0], got[0]], [base[1], got[1]], ":",
                                color=col, lw=1.0, zorder=2)
            for var, cfg in mem.items():
                got = cell(cfg, route)
                if not got:
                    continue
                drew = True
                c, q, f, n, thin = got
                # Thin cells keep their marker OUTLINE but lose the
                # infeasibility fill, so they cannot be read off the colourbar
                # as if the rate were estimated from a full sample.
                if var not in mark:
                    # A cell whose variant this figure has no marker for.  It
                    # used to fall through to "*" in silence, which is how
                    # S25H24+LPTAIL — the base cell launched under its own tag —
                    # appeared as an unexplained star.  Name it instead.
                    unknown.add(f"{cfg} (variant {var})")
                ax.plot(c, q, mark.get(var, "*"), ms=6.0,
                        mfc=("none" if thin
                             else _INFEAS_CMAP(min(1.0, (f or 0.0) / fmax))),
                        mec=col, mew=1.4, zorder=5 if var is None else 4)
                if thin:
                    thin_seen[route] = n
        if not drew:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7.5, color=MUT,
                    style="italic")
        ax.set_xlim(*_xlim(route))
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
        ax.set_xlabel(_LBL_COST)
        if ri == 0:
            ax.set_ylabel(_LBL_GAP)

    # legend in two registers: colour = family, marker = variant within it
    h1 = [plt.Line2D([], [], color=hue[cl], lw=1.3, ls=":", marker="o", ms=5.4,
                     mfc="white", mec=hue[cl], mew=1.4) for cl in order]
    l1 = [_la_cluster_label(cl) for cl in order]
    seen = {v for cl in order for v in clusters[cl]}
    h2, l2 = [], []
    for var, lbl, mk in _LA_VARIANTS:
        if var in seen:
            h2.append(plt.Line2D([], [], color=MUT, lw=0, ls="none", marker=mk,
                                 ms=5.4, mfc="white", mec=MUT, mew=1.2))
            l2.append(lbl)
    fig.subplots_adjust(left=0.088, right=0.895, top=0.755, bottom=0.145,
                        wspace=0.09)
    leg1 = fig.legend(h1, l1, frameon=False, fontsize=6.6, loc="upper left",
                      bbox_to_anchor=(0.085, 0.995), ncol=1, handlelength=1.8,
                      handletextpad=0.4, labelspacing=0.22,
                      title="configuration family", title_fontsize=6.6)
    leg1._legend_box.align = "left"
    leg2 = fig.legend(h2, l2, frameon=False, fontsize=6.6, loc="upper right",
                      bbox_to_anchor=(0.895, 0.995), ncol=3, handlelength=1.0,
                      handletextpad=0.4, columnspacing=1.1, labelspacing=0.22,
                      title="variant", title_fontsize=6.6)
    leg2._legend_box.align = "left"

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
    fig.text(0.005, 0.008,
             "Cost is the measured decision time at CHARGING-STATION stops, "
             "from the run logs." + _thin_note(thin_seen),
             fontsize=4.6, color=MUT, ha="left")
    if unknown:
        print("  [!] no marker defined for: " + ", ".join(sorted(unknown))
              + "  -> drawn as '*'.  Add them to _LA_VARIANTS.")
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
def section_la_local():
    """§8.3 — the cab-hardware (LOCAL) timing probe: table only.

    The cost/quality plane this used to draw was dropped (2026-08-22).  It was
    the sweep's figure re-run on five runs, and a scatter of five points on a
    plane whose axes were built to separate twelve configurations reads as a
    finding it cannot support.  The measurement the probe exists for is a
    per-stop time on cab-grade hardware, and that is a table.
    """
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
    for path in _paths.glob_logs("*_LA_LOCAL*.txt"):
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
        done = _paths.find_solution(f"{rid}.json") is not None
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



def _la_tail_cfg(n_scen: int, horizon: float, lptail: bool) -> str:
    """CSV config tag for one ladder rung and one tail solver.

    The base cell predates the sweep and carries no S<n>H<h> tag: it is stored
    as 'base' (MILP tail) / 'LPTAIL' (LP tail).
    """
    if (n_scen, horizon) == _LA_BASE:
        return "LPTAIL" if lptail else "base"
    tag = f"S{n_scen}H{horizon:g}"
    return f"{tag}+LPTAIL" if lptail else tag


def _la_ladder_panels(ladder, is_h, outname, xlabel, held):
    """One ladder (horizon or scenarios), one series per route x tail solver.

    Three panels: what the configuration buys (gap), what it costs (decision
    time), and whether it strands the truck (infeasibility).  Two series per
    panel, one per tail solver.

    Cost is the MEASURED DECISION TIME AT CS STOPS, parsed per stop from the
    run logs rather than derived from the run's wall clock.  It is the quantity
    the policy actually spends at the stops where the decision branches, and it
    is immune to batch contention and to --jobs, which the wall-clock measure
    was not.

    Coverage is NOT uniform across the ladder — the MILP-tail rungs are still
    filling in — and pooling the route classes makes an uneven mix invisible
    in the median.  Points whose run count falls short of the best-covered
    rung on the same ladder are therefore drawn hollow with their n stated, so
    a thin cell cannot be read as a finding.
    """
    stats = _la_stats()
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6), sharex=True)
    ax_q, ax_c, ax_i = axs

    # Coverage: the run count of the pooled cell, against the section-wide
    # _LA_THIN_N floor.  On the balanced panel every cell holds the same
    # instances, so this stays silent — which is the point.  It speaks up when
    # a cell was excluded from constraining the panel (--panel-exclude) and
    # does not cover all of it.
    def _cover(cfg, route):
        return _la_num(stats.get((cfg, route, "all")), "n_runs") or 0

    drew = False
    thin_seen = {}          # route class -> its run count, when below the floor
    for route in _LA_FIG_ROUTES:
        col = _LA_ROUTE_SHADE[route]
        mk  = _LA_ROUTE_MARK[route]
        for lptail, lbl, ls, _m in _LA_TAIL_SERIES:
            xs, qs, cs_, is_, nn = [], [], [], [], []
            for v in ladder:
                ns, hh = (_LA_BASE[0], float(v)) if is_h else (int(v),
                                                               _LA_BASE[1])
                cfg = _la_tail_cfg(ns, hh, lptail)
                # The route's window-pooled row — the SAME row section_la and
                # la_all read, so a value here is the value there.
                row = stats.get((cfg, route, "all"))
                if row is None:
                    continue
                n = _la_num(row, "n_runs")
                inf = _la_num(row, "n_infeasible")
                xs.append(float(v))
                qs.append(_la_num(row, "gap_pen_median_pct"))
                cs_.append(_la_num(row, "decision_cs_mean_s_median"))
                is_.append(100.0 * inf / n if (n and inf is not None) else None)
                nn.append(_cover(cfg, route))
            if not xs:
                continue
            drew = True
            for ax, ys in ((ax_q, qs), (ax_c, cs_), (ax_i, is_)):
                pts = [(x, y, n) for x, y, n in zip(xs, ys, nn)
                       if y is not None]
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col,
                        lw=1.3, zorder=2)
                for x, y, n in pts:
                    thin = bool(n and n < _LA_THIN_N)
                    if thin:
                        thin_seen[route] = n
                    ax.plot(x, y, mk, ms=4.2, color=col, zorder=3,
                            mfc="none" if thin else col, mec=col,
                            mew=1.1 if thin else 0.8)

    rungs = [float(v) for v in ladder]
    base_x = float(_LA_BASE[1] if is_h else _LA_BASE[0])
    for ax, ttl, ylab in ((ax_q, "Quality", _LBL_GAP),
                          (ax_c, "Cost", _LBL_COST),
                          (ax_i, "Risk", "Infeasible runs (%)")):
        ax.set_title(ttl, loc="left", fontsize=8)
        ax.set_ylabel(ylab, fontsize=7)
        ax.set_xscale("log")
        ax.set_xticks(rungs, [f"{v:g}" for v in rungs])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlim(min(rungs) / 1.22, max(rungs) * 1.22)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.axvline(base_x, color=GRID, lw=0.9, zorder=0)
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_ylim(bottom=0)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
        if not drew:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    fontsize=7.5, color=MUT, style="italic",
                    transform=ax.transAxes)
    ax_q.annotate("base", xy=(base_x, 0.0), xycoords=("data", "axes fraction"),
                  xytext=(3, 3), textcoords="offset points", fontsize=5.2,
                  color=MUT, ha="left", va="bottom")

    handles = [plt.Line2D([], [], color=_LA_ROUTE_SHADE[r], lw=1.4,
                          marker=_LA_ROUTE_MARK[r], ms=4.2)
               for r in _LA_FIG_ROUTES]
    labels = [ps.ROUTE_LBL[r] for r in _LA_FIG_ROUTES]
    handles += [plt.Line2D([], [], color=MUT, lw=1.3, ls=ls)
                for _lp, _l, ls, _m in _LA_TAIL_SERIES]
    labels += [l for _lp, l, _ls, _m in _LA_TAIL_SERIES]
    handles.append(plt.Line2D([], [], color=MUT, lw=0, marker="o", ms=4.2,
                              mfc="none", mec=MUT))
    labels.append("thin cell")
    fig.tight_layout(rect=(0, 0.02, 1, 0.84))
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="upper center",
               ncol=6, bbox_to_anchor=(0.5, 0.995), handlelength=1.8,
               handletextpad=0.4, columnspacing=1.2)
    fig.text(0.005, 0.012,
             f"Window classes pooled; short routes excluded; {held}."
             + _thin_note(thin_seen),
             fontsize=5.4, color=MUT, ha="left")
    _save(fig, outname)


def section_la_plane():
    """8.3 - the cost/quality plane, one panel per LADDER, routes aggregated.

    The same plane as `la_all` and the same rows, re-cut along the axis the
    configuration question is actually asked on.  `la_all` gives a panel to each
    route class and puts every cell in all of them, which answers "what does
    this configuration cost on a route of that length"; this one gives a panel
    to each LADDER and pools the route classes, which answers "given a compute
    budget, does the next second go into horizon or into scenarios".  Reading
    that off `la_all` means tracing two interleaved fans through three panels.

    Both tail solvers appear in both panels, because the choice of subproblem
    is not a third ladder — it moves a cell along the same cost/quality plane,
    so it belongs beside the rung it modifies rather than in a figure of its
    own.

    Route classes are POOLED here, which `_LA_FIG_ROUTES` deliberately does not
    do elsewhere.  The reason it is safe is narrow: a ladder is read as a SHAPE
    -- which way the path bends -- and both series in a panel are pooled the
    same way, so a route mix that shifts a cell shifts its neighbour with it.
    The absolute level is not safe to quote from here; that is what the
    per-route panels are for.  n is printed against every point so a cell
    resting on a different population than its neighbour is visible.
    """
    print("== Sec 8.3 look-ahead - cost/quality plane by ladder ==")
    stats = _la_stats()
    if not stats:
        print("  data_output/additional_la_stats.csv missing - nothing drawn")
        return

    LADDERS = (("Horizon ladder", _LA_HORIZONS, True,
                rf"$|\Xi|$ held at {_LA_BASE[0]}", "L"),
               ("Scenario ladder", _LA_SCENARIOS, False,
                rf"$L$ held at {_LA_BASE[1]:g} h", "S"))

    def cell(v, is_h, lptail):
        ns, hh = (_LA_BASE[0], float(v)) if is_h else (int(v), _LA_BASE[1])
        row = stats.get((_la_tail_cfg(ns, hh, lptail), "all", "all"))
        c = _la_num(row, "decision_cs_mean_s_median")
        q = _la_num(row, "gap_pen_median_pct")
        if c is None or q is None:
            return None
        n = _la_num(row, "n_runs") or 0
        i = _la_num(row, "n_infeasible")
        return c, q, n, ((i / n) if (n and i is not None) else None)

    pts = [got for _t, lad, is_h, _h, _p in LADDERS
           for v in lad for lp, _l, _s, _m in _LA_TAIL_SERIES
           for got in [cell(v, is_h, lp)] if got]
    if not pts:
        print("  no cell carries both a cost and a gap - nothing drawn")
        return
    xmax = max(p[0] for p in pts)
    ymax = max(p[1] for p in pts)
    # The fill ramp is scaled to the WORST rate actually observed, as in every
    # other figure of the section, so the red end marks a real cell rather than
    # a hypothetical 100%.
    fmax = max([p[3] for p in pts if p[3]] or [1.0])

    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.0), sharex=True, sharey=True)
    thin_seen = {}

    for ax, (title, ladder, is_h, held, pfx) in zip(axs, LADDERS):
        drew = False
        for lptail, lbl, ls, mk in _LA_TAIL_SERIES:
            col = _la_solver_color(lptail)
            seq = [(v,) + got for v in ladder
                   for got in [cell(v, is_h, lptail)] if got]
            if not seq:
                continue
            drew = True
            # The path IS the finding: it runs along the ladder, so the reader
            # follows the rungs in order rather than matching labels to points.
            ax.plot([p[1] for p in seq], [p[2] for p in seq], ls, color=col,
                    lw=1.2, zorder=2)
            base_v = _LA_BASE[1] if is_h else _LA_BASE[0]
            for v, c, q, n, f in seq:
                # Marker FILL is the infeasibility rate, EDGE is the solver.
                # The fill is not decoration: the gap median is taken over
                # feasible runs only, so a configuration that strands the truck
                # more often can post a better gap for exactly the wrong
                # reason, and this is where that shows.  A thin cell keeps the
                # edge and loses the fill, so a rate estimated from two runs is
                # never read off the colourbar as if it were solid.
                thin = bool(n and n < _LA_THIN_N)
                if thin:
                    thin_seen[f"{pfx}{v:g}"] = n
                ax.plot(c, q, mk, ms=6.4, zorder=4,
                        mfc=("none" if thin
                             else _INFEAS_CMAP(min(1.0, (f or 0.0) / fmax))),
                        mec=col, mew=1.4)
                if v == base_v:      # the cell the sweep is quoted against
                    ax.plot(c, q, "o", ms=9.5, mfc="none", mec=col, lw=0.9,
                            zorder=3)
                ax.annotate(f"{pfx}{v:g}" + (f" (n={n:g})" if thin else ""),
                            (c, q), xytext=(4, 4),
                            textcoords="offset points", fontsize=5.6,
                            color=col)
        if not drew:
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7.5, color=MUT,
                    style="italic")
        ax.set_title(title, loc="left", fontsize=8)
        ax.annotate(held, xy=(1.0, 1.0), xycoords="axes fraction",
                    xytext=(-2, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=5.8, color=MUT)
        ax.set_xlim(0, xmax * 1.20)
        ax.set_ylim(0, ymax * 1.18)
        ax.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        ax.tick_params(which="minor", length=2)
        ax.set_xlabel(_LBL_COST, fontsize=7.5)
    axs[0].set_ylabel(_LBL_GAP, fontsize=7.5)

    # Compact legend: ONE row, on the figure's top edge, sharing its line with
    # nothing.  The per-panel titles carry the ladder names and the held
    # parameter sits inside each panel, so the legend only has to say which
    # series is which solver and what the ring means.
    handles = [plt.Line2D([], [], color=_la_solver_color(lp), lw=1.3, ls=ls,
                          marker=mk, ms=5.4, mfc="white",
                          mec=_la_solver_color(lp), mew=1.4)
               for lp, _l, ls, mk in _LA_TAIL_SERIES]
    labels = [l for _lp, l, _s, _m in _LA_TAIL_SERIES]
    handles.append(plt.Line2D([], [], color=MUT, lw=0, marker="o", ms=8.0,
                              mfc="none", mec=MUT))
    labels.append("base cell")
    if thin_seen:
        handles.append(plt.Line2D([], [], color=MUT, lw=0, marker="s", ms=5.4,
                                  mfc="none", mec=MUT))
        labels.append(f"thin (n < {_LA_THIN_N})")
    fig.subplots_adjust(left=0.085, right=0.885, top=0.845, bottom=0.155,
                        wspace=0.08)
    fig.legend(handles, labels, frameon=False, fontsize=6.8, loc="upper center",
               ncol=len(handles), bbox_to_anchor=(0.5, 1.005), handlelength=1.6,
               handletextpad=0.35, columnspacing=1.3, borderpad=0.0)

    import matplotlib.cm as _cm
    from matplotlib.colors import Normalize as _Norm
    _sm = _cm.ScalarMappable(norm=_Norm(0, 100.0 * fmax), cmap=_INFEAS_CMAP)
    _sm.set_array([])
    _p = axs[-1].get_position()
    _cax = fig.add_axes([_p.x1 + 0.014, _p.y0, 0.011, _p.height])
    _cb = fig.colorbar(_sm, cax=_cax, orientation="vertical",
                       ticks=[0, 100.0 * fmax])
    _cb.ax.set_yticklabels(["0", f"{100.0 * fmax:.0f}"])
    _cb.outline.set_linewidth(0.3)
    _cb.set_label("Marker fill: infeasible runs (%)", fontsize=5.4, labelpad=2)
    _cb.ax.tick_params(labelsize=4.8, length=1.5, width=0.3, pad=1)
    _save(fig, "additional_la_plane")


def section_la_ladders():
    """Sec 8.3 — the two configuration axes, one figure each.

    Splits the old two-column `additional_la_config` into a horizon figure and
    a scenario figure, pools the route classes (the corridor-length breakdown
    lives in `additional_la_config`), and adds the tail solver as the series
    dimension so MILP and LP subproblems are compared on the same ladder.
    """
    print("== Sec 8.3 look-ahead ladders (routes pooled) ==")
    _la_ladder_panels(_LA_HORIZONS, True, "additional_la_horizon",
                      _LBL_HORIZON,
                      rf"scenarios held at $|\Xi| = {_LA_BASE[0]}$")
    _la_ladder_panels(_LA_SCENARIOS, False, "additional_la_scenarios",
                      _LBL_SCEN,
                      rf"horizon held at $L = {_LA_BASE[1]:g}$ h")


_SECTIONS = dict(diesel=section_diesel, sensitivity=section_sensitivity,
                 grid=section_grid,
                 la=section_la,
                 la_ladders=section_la_ladders,
                 la_plane=section_la_plane,
                 la_all=section_la_all,
                 # tables only since 2026-08-22 — see each function
                 la_local=section_la_local,
                 # both effect measures: route duration and the penalised
                 # objective the model actually minimises
                 la_policy=lambda: (section_la_policy("dur"),
                                    section_la_policy("pen")),
                 diesel_oat=section_diesel_oat,
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
