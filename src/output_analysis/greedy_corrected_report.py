"""
greedy_corrected_report.py — the paper's tables and figures with the corrected greedy
=====================================================================================
Regenerates every table and figure of the submitted paper that shows the greedy
baseline, with the CORRECTED greedy (methods/greedy_corrected.py, runs labelled
with the variant "CORR") in its place, and adds a side-by-side comparison of the
two greedy versions.  The baseline greedy runs, the existing tables and the
existing figures are left untouched: every output below is a NEW file carrying
the suffix "_greedyC".

How the substitution works — no reporting module is edited:
  * tables and Figure 6: the corrected greedy rows are relabelled as the
    standard greedy configuration at load time and the baseline greedy rows are
    dropped, then the unchanged builders of compile_solutions / paper_figures
    run on that row set;
  * Figures 9-11: additional_figures._policy is wrapped so that a request for
    the greedy is answered by the CORR configuration, on BOTH legs of every
    paired delta (EV/variant and diesel/base);
  * the save/CSV/TeX writers of additional_figures are wrapped to add the suffix.

Scope of Tables 7-9: the 900 base-case instances only (route/customer/window
class present, no method variant).  Note that the submitted Table 9 pooled the
greedy runs of the sensitivity axes into its greedy row; this report does not.

Outputs
-------
  tex/tables/{gap,gap_nopen,gap_simplified,feasibility,runtime}_greedyC.tex
  tex/tables/greedy_corrected_comparison.tex
  data_output/paper_gap_stats_greedyC.csv
  data_output/greedy_corrected_comparison.csv
  data_output/additional_*_greedyC.csv
  figures/basecase/paper_gap_box_pooledtw_greedyC.{png,pdf}   (Fig. 6)
  figures/basecase/paper_gap_box_greedyC.{png,pdf}            (Fig. 6, by window)
  figures/sensitivity/additional_sens_effects_greedyC.*       (Fig. 9)
  figures/sensitivity/additional_grid_battery_power_greedyC.* (Fig. 10)
  figures/sensitivity/additional_diesel_oat_greedyC.*         (Fig. 11)
  figures/sensitivity/additional_diesel_gap_greedyC.*, additional_diesel_tw_greedyC.*

Usage
-----
  python -m src.output_analysis.greedy_corrected_report [--only tables|fig6|sens|compare]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
from collections import Counter, defaultdict

from src import paths as P
from src.output_analysis import compile_solutions as cs

VAR    = "CORR"          # variant label of methods/greedy_corrected.py
SUFFIX = "_greedyC"      # added to every output name
LABEL  = "Greedy (corrected)"   # table row label (Table 9 shows both versions)
FIG_LABEL = "Greedy"           # figures: plain label, no mention of the variant
ROUTES = ("short", "medium", "long")


# ══════════════════════════════════════════════════════════════════════════════
# Row substitution
# ══════════════════════════════════════════════════════════════════════════════

def _swap(rows: list[dict]) -> list[dict]:
    """Corrected greedy rows become the greedy; baseline greedy rows are dropped.

    Only the greedy is touched: every other method keeps its rows as they are,
    including its own variants (excluded later by the usual base-case filter).
    """
    out, n_new, n_old = [], 0, 0
    for r in rows:
        if r.get("method") == "greedy":
            v = r.get("variant")
            if v == VAR:
                r = dict(r)
                r["variant"] = None
                r["greedy_variant"] = VAR
                out.append(r)
                n_new += 1
            elif v:
                out.append(r)
            else:
                n_old += 1
            continue
        out.append(r)
    print(f"  [greedyC] substituted {n_new} corrected greedy row(s); "
          f"set aside {n_old} baseline greedy row(s)")
    return out


def _base_rows(swap: bool = True) -> list[dict]:
    """Annotated, deduplicated base-case rows (ROBU excluded, as in the paper)."""
    rows = cs.load_solutions(P.solutions())
    rows += cs.find_failed_runs(P.logs(), P.solutions())
    if swap:
        rows = _swap(rows)
    cs._annotate_instance_tags(rows)
    rows = [r for r in rows if r.get("route_class") and not r.get("variant")
            and r.get("method") != "ROBU"]
    cs._annotate_gap_to_oracle(rows, P.solutions())
    cs._annotate_outcome(rows)
    rows, _ = cs._dedup_latest(rows)
    cs._annotate_solver_gap(rows, P.logs())
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Tables 7-9
# ══════════════════════════════════════════════════════════════════════════════

def _write_tex(name: str, lines) -> str:
    path = P.tex_tables(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  LaTeX     : {path}")
    return path


def _note(lines: list[str]) -> list[str]:
    """Prefix a generated table with a comment saying which greedy it shows."""
    return (["% Generated by src/output_analysis/greedy_corrected_report.py.",
             "% Greedy column/row = CORRECTED greedy (variant CORR, MILP "
             "execution rules); base-case instances only."] + list(lines))


def tables() -> None:
    print("== Tables 7-9 with the corrected greedy ==")
    rows = _base_rows(swap=True)
    _write_tex(f"gap_simplified{SUFFIX}.tex", _note(
        cs.build_gap_simplified_latex(rows, metric="gap_pen",
                                      secondary="gap_nopen")))
    _write_tex(f"gap{SUFFIX}.tex", _note(
        cs.build_gap_latex(rows, metric="gap_pen", secondary="gap_nopen")))
    _write_tex(f"gap_nopen{SUFFIX}.tex", _note(
        cs.build_gap_latex(rows, metric="gap_nopen")))
    old_greedy = [r for r in _base_rows(swap=False) if r.get("method") == "greedy"]
    _write_tex(f"feasibility{SUFFIX}.tex", _note(
        _feasibility_table(rows, old_greedy)))
    _write_tex(f"runtime{SUFFIX}.tex", _note(cs.build_runtime_latex(rows)))


def _feasibility_table(rows: list[dict], old_greedy: list[dict]) -> list[str]:
    """Table 9 with explicit definitions, base case only.

    compile_solutions.build_feasibility_latex counts a run as failed only for
    an HoS breach or a stranding, so it leaves out LA runs that halt because
    no candidate action is feasible in the sampled scenarios; here every run
    whose outcome is "infeasible" counts, which is the 1.2% the paper reports
    for LA.  The window hit rate is given twice: as in the paper (mean per-run
    hit rate over all runs, the no-window class counting as hits) and over
    windowed instances only.  The submitted greedy is shown for reference.
    """
    def _stats(u, has_opt):
        solved = [r for r in u if not cs._is_unsolved(r)]
        n = len(solved)
        inf = [r for r in solved if r.get("outcome") == "infeasible"]
        hos = sum(1 for r in inf if cs._hos_viol_count(r) > 0)
        soc = sum(1 for r in inf if (cs._mval(r, "n_stranding") or 0) > 0)
        twh = cs._mean([cs._mval(r, "tw_hit_rate") for r in solved])
        win = [r for r in solved if r.get("outcome") == "feasible"
               and r.get("window_class") != "none"]
        nc = sum(int(cs._mval(r, "tw_n_customers") or 0) for r in win)
        nm = sum(int(cs._mval(r, "tw_n_misses") or 0) for r in win)
        nopt = [r for r in u if r.get("solve_status") == "time_limit"]
        meas = [r.get("solver_gap") for r in nopt if r.get("solver_gap") is not None]
        lo, hi = _wilson(len(inf), n)
        pct = lambda x: "--" if x is None else f"{100 * x:.1f}"
        return (f"{pct(len(inf) / n if n else None)} & [{lo:.1f}, {hi:.1f}] "
                f"& {pct(hos / n if n else None)} & {pct(soc / n if n else None)} "
                f"& {pct(twh)} & {pct(1 - nm / nc if nc else None)} "
                f"& {pct(len(nopt) / len(u)) if (has_opt and u) else '--'} "
                f"& {pct(cs._mean(meas)) if (has_opt and meas) else '--'}")

    body = [
        rf"{LABEL} & " + _stats([r for r in rows if r.get("method") == "greedy"], False) + r" \\",
        r"\textit{Greedy (submitted)} & " + _stats(old_greedy, False) + r" \\",
        r"RO & " + _stats([r for r in rows if r.get("method") == "RO"], True) + r" \\",
        r"LA & " + _stats([r for r in rows if r.get("method") == "LA"], False) + r" \\",
        r"2SP & " + _stats([r for r in rows if r.get("method") == "2SP"], True) + r" \\",
    ]
    return [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Feasibility and robustness by method, base-case instances. "
        r"Infeasible: share of runs that ended with any violation (HoS breach, "
        r"reserve breach, or no feasible action for LA), with its 95\% Wilson "
        r"interval and the HoS/reserve split.  TW hit: mean per-run share of "
        r"customers served inside their window over all runs (the no-window "
        r"class counts as hits, as in the submitted table) and over feasible "
        r"runs on windowed instances only.  Not opt.\ and Opt.\ gap: offline "
        r"solves stopped at the time limit and their mean final MIP gap.  "
        r"The corrected greedy executes every stop under the MILP's rules; "
        r"the submitted greedy is shown for reference.}",
        r"\label{tab:feasibility-greedyC}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}", r"\toprule",
        r"\textbf{Method} & \textbf{Infeasible (\%)} & \textbf{95\% CI} "
        r"& \textbf{HoS} & \textbf{Reserve} & \textbf{TW hit (\%)} "
        r"& \textbf{TW hit, windowed (\%)} & \textbf{Not opt. (\%)} "
        r"& \textbf{Opt. gap (\%)}\\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}%", r"}",
        r"\end{table}"]


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6
# ══════════════════════════════════════════════════════════════════════════════

def fig6() -> None:
    print("== Figure 6 with the corrected greedy ==")
    from src.plot import paper_figures as pf
    from src.plot import paper_style as ps
    orig_load = cs.load_solutions
    orig_lbl  = ps.METHOD_LBL.get("greedy")
    cs.load_solutions = lambda d: _swap(orig_load(d))
    ps.METHOD_LBL["greedy"] = FIG_LABEL
    try:
        gaps, n_infe, n_unsl, n_feas = pf.collect_gaps(P.solutions())
        extra = set(ps.METHOD_ORDER_EXTRA)
        gaps, n_infe, n_unsl, n_feas = (
            {k: v for k, v in d.items() if k[3] not in extra}
            for d in (gaps, n_infe, n_unsl, n_feas))
        pf.write_stats_csv(gaps, n_infe, n_unsl, n_feas,
                           P.data_output(f"paper_gap_stats{SUFFIX}.csv"))
        for inner in ("pooled", "tw"):
            for p in pf.plot_gap_figure(
                    gaps, n_infe, n_unsl, n_feas, kind="box",
                    metric="gap_pen", full_grid=True, layout="row",
                    inner=inner, line_band=True, drop_methods=["ROBU"],
                    name_suffix=SUFFIX, with_det=False):
                print(f"  Figure    : {p}")
    finally:
        cs.load_solutions = orig_load
        ps.METHOD_LBL["greedy"] = orig_lbl


# ══════════════════════════════════════════════════════════════════════════════
# Figures 9-11 (and the diesel tables that go with Fig. 11)
# ══════════════════════════════════════════════════════════════════════════════

def sens() -> None:
    print("== Figures 9-11 with the corrected greedy ==")
    from src.plot import additional_figures as af
    from src.plot import paper_style as ps

    orig_policy, orig_save = af._policy, af._save
    orig_csv, orig_tex     = af._write_csv, af._write_tex
    orig_lbl               = ps.METHOD_LBL.get("greedy")

    def _policy(stem, alg, tag=None, cfg=...):
        if str(alg).upper() == "GREEDY":
            # the standard greedy configuration is answered by CORR; every
            # other configuration (e.g. LPTAIL, an LA-only tag) has no greedy
            if cfg is ... or cfg is None:
                return orig_policy(stem, alg, tag, cfg=VAR)
            return None
        return orig_policy(stem, alg, tag, cfg=cfg)

    def _save(fig, name):
        return orig_save(fig, f"{name}{SUFFIX}")

    def _write_csv(name, header, rows):
        stem, ext = os.path.splitext(name)
        return orig_csv(f"{stem}{SUFFIX}{ext}", header, rows)

    def _write_tex(name, text):
        stem, ext = os.path.splitext(name)
        text = ("% Greedy = CORRECTED greedy (variant CORR); generated by "
                "greedy_corrected_report.py\n" + text)
        return orig_tex(f"{stem}{SUFFIX}{ext}", text)

    af._policy, af._save = _policy, _save
    af._write_csv, af._write_tex = _write_csv, _write_tex
    ps.METHOD_LBL["greedy"] = FIG_LABEL
    try:
        af._paths.ensure_dirs()
        af.section_sensitivity()      # Fig. 9
        af.section_grid()             # Fig. 10
        af.section_diesel()           # 8.3 headline + tables
        af.section_diesel_oat()       # Fig. 11
    finally:
        af._policy, af._save = orig_policy, orig_save
        af._write_csv, af._write_tex = orig_csv, orig_tex
        ps.METHOD_LBL["greedy"] = orig_lbl


# ══════════════════════════════════════════════════════════════════════════════
# Side-by-side comparison of the two greedy versions
# ══════════════════════════════════════════════════════════════════════════════

def _wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def _sigma_counts(fname: str) -> tuple[int, int, int]:
    """(charge+rest sequential, charge+break sequential, break idled at bay)
    stops in one stored run, read from its durations_list."""
    path = P.find_solution(fname)
    if not path:
        return (0, 0, 0)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    tb = {"b45": 0.75, "b30": 0.5, "b15": 0.25}
    seq_r = seq_b = idle = 0
    for a, du in zip(d.get("actions") or [], d.get("durations_list") or []):
        if not du.get("tauc"):
            continue
        brk, rst = a.get("break_type"), a.get("rest_type")
        if du.get("sigma"):
            seq_r += bool(rst)
            seq_b += bool(brk and not rst)
        elif brk in tb and du.get("taub", 0) > 1e-6 and du["tauc"] < tb[brk]:
            idle += 1
    return seq_r, seq_b, idle


def compare() -> None:
    print("== Baseline vs corrected greedy ==")
    rows = cs.load_solutions(P.solutions())
    cs._annotate_instance_tags(rows)
    rows = [r for r in rows if r.get("route_class") and r.get("method") == "greedy"
            and r.get("variant") in (None, VAR) and not r.get("supervised")]
    cs._annotate_gap_to_oracle(rows, P.solutions())
    cs._annotate_outcome(rows)
    rows, _ = cs._dedup_latest(rows)
    by = {None: {}, VAR: {}}
    for r in rows:
        by[r.get("variant")][r["instance"]] = r

    out = []
    for rc in ROUTES + ("all",):
        rec = {}
        for v, tag in ((None, "old"), (VAR, "new")):
            rs = [r for i, r in by[v].items() if rc == "all" or r["route_class"] == rc]
            n = len(rs)
            k = sum(1 for r in rs if r["outcome"] == "infeasible")
            feas = [r for r in rs if r["outcome"] == "feasible"]
            gp = [100 * r["gap_pen"] for r in feas if r.get("gap_pen") is not None]
            gn = [100 * r["gap_nopen"] for r in feas if r.get("gap_nopen") is not None]
            win = [r for r in feas if r["window_class"] != "none"]
            nc = sum(int(cs._get(r, ("metrics", "tw_n_customers")) or 0) for r in win)
            nm = sum(int(cs._get(r, ("metrics", "tw_n_misses")) or 0) for r in win)
            dirc = [bool(cs._get(r, ("metrics", "directive_compliance", "compliant")))
                    for r in feas]
            types = Counter()
            for r in rs:
                if r["outcome"] == "infeasible":
                    for t in (cs._get(r, ("metrics", "violations_by_type")) or {}):
                        if t not in cs._NONFATAL_VIOL:
                            types[t] += 1
            lo, hi = _wilson(k, n)
            rec[tag] = dict(n=n, infeas=100 * k / n if n else float("nan"),
                            ci=f"{lo:.1f}-{hi:.1f}",
                            gap=st.mean(gp) if gp else float("nan"),
                            gap_nopen=st.mean(gn) if gn else float("nan"),
                            tw_hit=100 * (1 - nm / nc) if nc else float("nan"),
                            directive=100 * st.mean(dirc) if dirc else float("nan"),
                            types=";".join(f"{a}:{b}" for a, b in sorted(types.items())))
        both = [i for i in by[None] if i in by[VAR]
                and (rc == "all" or by[None][i]["route_class"] == rc)
                and by[None][i]["outcome"] == "feasible"
                and by[VAR][i]["outcome"] == "feasible"]
        dd = [100 * (by[VAR][i]["duration_h"] / by[None][i]["duration_h"] - 1)
              for i in both]
        sc_new = [_sigma_counts(by[VAR][i]["_file"]) for i in both]
        sc_old = [_sigma_counts(by[None][i]["_file"]) for i in both]
        out.append(dict(
            route=rc,
            n_old=rec["old"]["n"], n_new=rec["new"]["n"],
            infeas_old=rec["old"]["infeas"], ci_old=rec["old"]["ci"],
            infeas_new=rec["new"]["infeas"], ci_new=rec["new"]["ci"],
            types_old=rec["old"]["types"], types_new=rec["new"]["types"],
            gap_old=rec["old"]["gap"], gap_new=rec["new"]["gap"],
            gap_nopen_old=rec["old"]["gap_nopen"], gap_nopen_new=rec["new"]["gap_nopen"],
            tw_hit_old=rec["old"]["tw_hit"], tw_hit_new=rec["new"]["tw_hit"],
            directive_old=rec["old"]["directive"], directive_new=rec["new"]["directive"],
            n_paired=len(dd),
            dur_change_mean=st.mean(dd) if dd else float("nan"),
            dur_change_max=max(dd) if dd else float("nan"),
            dur_unchanged=sum(1 for x in dd if abs(x) < 1e-9),
            seq_rest_per_run_new=st.mean(s[0] for s in sc_new) if sc_new else 0,
            seq_break_per_run_new=st.mean(s[1] for s in sc_new) if sc_new else 0,
            idle_bay_per_run_old=st.mean(s[2] for s in sc_old) if sc_old else 0,
        ))

    path = P.data_output(f"greedy_corrected_comparison{_cmp_sfx()}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"  Stats CSV : {path}")

    f = lambda x, nd=1: "--" if x != x else f"{x:.{nd}f}"
    lines = [
        "% Generated by src/output_analysis/greedy_corrected_report.py",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Baseline greedy (submitted paper) against the corrected "
        r"greedy, which executes every stop under the MILP's rules (a charge "
        r"before a daily rest is sequential; a charge is credited as break "
        r"only when it lasts at least the break; no charge shorter than "
        r"15~min).  Base-case instances.  Gaps over each version's feasible "
        r"runs; duration change paired on instances where both are feasible.}",
        r"\label{tab:greedy-corrected}",
        r"\begin{tabular}{l rr rr rr rr r}", r"\toprule",
        r" & \multicolumn{2}{c}{Infeasible (\%)} & \multicolumn{2}{c}{Gap (\%)}"
        r" & \multicolumn{2}{c}{TW hit (\%)} & \multicolumn{2}{c}{Directive ok (\%)}"
        r" & $\Delta$ dur. (\%) \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Route & old & new & old & new & old & new & old & new & mean \\",
        r"\midrule"]
    for o in out:
        lines.append(
            f"{o['route'].capitalize()} & {f(o['infeas_old'])} & {f(o['infeas_new'])}"
            f" & {f(o['gap_old'])} & {f(o['gap_new'])}"
            f" & {f(o['tw_hit_old'])} & {f(o['tw_hit_new'])}"
            f" & {f(o['directive_old'])} & {f(o['directive_new'])}"
            f" & {f(o['dur_change_mean'], 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex(f"greedy_corrected_comparison{_cmp_sfx()}.tex", lines)
    for o in out:
        print(f"  {o['route']:<6} infeas {f(o['infeas_old'])}->{f(o['infeas_new'])}%"
              f"  gap {f(o['gap_old'],2)}->{f(o['gap_new'],2)}%"
              f"  dur {f(o['dur_change_mean'],2)}% (n={o['n_paired']})"
              f"  seq rest/brk per run {o['seq_rest_per_run_new']:.2f}/"
              f"{o['seq_break_per_run_new']:.2f}"
              f"  old idle-at-bay {o['idle_bay_per_run_old']:.3f}")


def _cmp_sfx() -> str:
    """The first corrected greedy (CORR) wrote the comparison without a suffix;
    later variants carry theirs so the files never overwrite each other."""
    return "" if VAR == "CORR" else SUFFIX


# ══════════════════════════════════════════════════════════════════════════════
# Monotonicity in charger power, all greedy versions side by side
# ══════════════════════════════════════════════════════════════════════════════

# (battery kWh, [(charger kW, instance tag)]) — the power ladder of each row of
# Fig. 10; the 500 kWh row is the one-at-a-time charger-power axis.
_LADDERS = [
    (500, [(150, "kw150"), (350, None), (700, "kw700"), (1000, "kw1000")]),
    (300, [(150, "kwh300_kw150"), (350, "kwh300"), (700, "kwh300_kw700"),
           (1000, "kwh300_kw1000")]),
    (700, [(150, "kwh700_kw150"), (350, "kwh700"), (700, "kwh700_kw700"),
           (1000, "kwh700_kw1000")]),
    (900, [(150, "kwh900_kw150"), (350, "kwh900"), (700, "kwh900_kw700"),
           (1000, "kwh900_kw1000")]),
]
_VERSIONS = [(None, "submitted"), ("CORR", "corrected v1 (MILP rules)"),
             ("CORR2", "corrected v2 (stay at bay)")]


def mono() -> None:
    """Per instance: does the greedy route get longer when only the charger
    power increases?  Pairs of consecutive powers, both runs feasible."""
    print("== Greedy route duration vs charger power ==")
    rows = cs.load_solutions(P.solutions())
    rows = [r for r in rows if r.get("method") == "greedy"
            and not r.get("supervised")
            and r.get("variant") in [v for v, _ in _VERSIONS]]
    cs._annotate_outcome(rows)
    rows, _ = cs._dedup_latest(rows)
    dur = {}
    for r in rows:
        inst = r.get("instance") or ""
        stem, _, tag = inst.partition("__")
        if r["outcome"] == "feasible" and r.get("duration_h"):
            dur[(r.get("variant"), stem, tag or None)] = float(r["duration_h"])
    out = []
    for var, vlabel in _VERSIONS:
        for bat, ladder in _LADDERS:
            for (p_lo, t_lo), (p_hi, t_hi) in zip(ladder, ladder[1:]):
                d = []
                stems = {s for (v, s, t) in dur if v == var and t == t_hi}
                for s in stems:
                    a, b = dur.get((var, s, t_lo)), dur.get((var, s, t_hi))
                    if a and b:
                        d.append(100 * (b / a - 1))
                if not d:
                    continue
                worse = [x for x in d if x > 1e-6]
                out.append(dict(
                    version=vlabel, variant=var or "", battery_kwh=bat,
                    from_kw=p_lo, to_kw=p_hi, n_pairs=len(d),
                    n_worse=len(worse),
                    pct_worse=100 * len(worse) / len(d),
                    n_worse_gt_0_5pct=sum(1 for x in d if x > 0.5),
                    max_worse_pct=max(worse) if worse else 0.0,
                    mean_change_pct=st.mean(d)))
    path = P.data_output("greedy_power_monotonicity.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"  Stats CSV : {path}")
    lines = [
        "% Generated by src/output_analysis/greedy_corrected_report.py (mono)",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Greedy route duration when only the charger power increases. "
        r"Instances where the run at the higher power is longer than at the "
        r"lower power, out of the instances where both runs are feasible "
        r"(500 kWh: all routes; other packs: short and medium routes).}",
        r"\label{tab:greedy-monotonicity}",
        r"\begin{tabular}{ll rrr}", r"\toprule",
        r"Version & Battery & 150$\to$350 kW & 350$\to$700 kW & 700$\to$1000 kW \\",
        r"\midrule"]
    for var, vlabel in _VERSIONS:
        for bat, _ in _LADDERS:
            cells = []
            for lo, hi in ((150, 350), (350, 700), (700, 1000)):
                o = next((x for x in out if x["variant"] == (var or "")
                          and x["battery_kwh"] == bat and x["from_kw"] == lo), None)
                cells.append("--" if o is None else f"{o['n_worse']}/{o['n_pairs']}")
            lines.append(f"{vlabel} & {bat} kWh & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    _write_tex("greedy_power_monotonicity.tex", lines)
    for o in out:
        print(f"  {o['version']:<28} {o['battery_kwh']} kWh {o['from_kw']:>4}->"
              f"{o['to_kw']:<4} worse {o['n_worse']:>3}/{o['n_pairs']:<3}"
              f"  max {o['max_worse_pct']:5.2f}%  mean {o['mean_change_pct']:+6.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", choices=["tables", "fig6", "sens", "compare",
                                       "mono"], default=None)
    ap.add_argument("--variant", default=VAR,
                    help="greedy variant to substitute (CORR = MILP rules, "
                         "CORR2 = stay-at-bay rules)")
    ap.add_argument("--suffix", default=None,
                    help="output-name suffix (default: _greedyC for CORR, "
                         "_greedyC2 for CORR2)")
    a = ap.parse_args()
    VAR    = a.variant
    SUFFIX = a.suffix or {"CORR": "_greedyC", "CORR2": "_greedyC2"}.get(
        VAR, f"_greedy{VAR}")
    print(f"  [greedyC] variant {VAR} -> outputs suffixed '{SUFFIX}'")
    P.ensure_dirs()
    steps = dict(tables=tables, fig6=fig6, sens=sens, compare=compare,
                 mono=mono)
    for name, fn in steps.items():
        if a.only in (None, name):
            fn()
