"""
plots.py — Visualisation for the BET scheduling system
=======================================================
All matplotlib figure code.  No solver, simulation, or optimisation logic.

Public interface
----------------
    plot_solution(sol, data, title="solution")
        Three-panel figure for a full-route MILP solution:
          1. Gantt  — activity timeline per stop
          2. SOC    — battery state of charge trajectory
          3. HoS    — consecutive driving / shift driving / shift working

    plot_simulation_results(results, full_data, title="simulation", save=True)
        Five-panel figure for a completed simulation run:
          1. Gantt (simulation realisation)
          2. Gantt (oracle hindsight-optimal schedule)
          3. SOC trajectory
          4. HoS accumulator trajectories
          5. Look-ahead scenario scatter (decision quality across stops)

Dependencies
------------
    matplotlib, numpy — plotting only; no pyomo, no HiGHS.
    MILP              — INFEASIBLE_PENALTY constant (scalar, no solver import).
    oracle            — check_simulation_feasibility (feasibility check helper).

Import chain
------------
    plots.py → MILP (INFEASIBLE_PENALTY), oracle (check_simulation_feasibility)
    No circular imports: MILP.py and oracle.py do not import from plots.py
    (except MILP.py re-exports plot_solution for callers that use
     `from MILP import plot_solution`).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Shared colour palette ─────────────────────────────────────────────────────
COL = dict(
    drive   = "#2C6FAC",
    service = "#27AE60",
    queue   = "#C0392B",
    charge  = "#E67E22",
    brk     = "#F1C40F",
    rest    = "#8E44AD",
    mstop   = "#95A5A6",   # maneuver/setup overhead at CS stop (v·Mstop)
    mseq    = "#5D6D7E",   # sequential reposition overhead (sigma·Mseq)
    window  = "#17A2B8",   # customer arrival-time window [Wha, Whf]
)
EPS = 1e-3

# Window widths above this (hours) are treated as unconstrained ("none" window
# class, Whf ≈ Wha + 2e7) and are not drawn.
WINDOW_MAX_DRAW_WIDTH_H = 100.0

FIGURES_DIR = "figures"


def _ensure_fig_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Low-level drawing primitives ─────────────────────────────────────────────

def _bar(ax, start, dur, y, h, color, label=None, fontsize=7, text_color="white"):
    """Draw one horizontal Gantt bar; skip if duration is negligible.

    The label is drawn rotated 90° so it stays readable inside narrow bars
    (adjacent short activities used to smear their horizontal labels into
    each other, e.g. "→24→2526→27→setup").
    """
    if dur < EPS:
        return
    ax.barh(y, dur, left=start, height=h, color=color,
            edgecolor="white", linewidth=0.3)
    if dur > 0.08 and label:
        ax.text(start + dur / 2, y, label,
                ha="center", va="center", rotation=90,
                fontsize=fontsize, color=text_color,
                fontweight="bold", clip_on=True)


def _draw_time_window(ax, wha, whf, y, h, color=None):
    """
    Visualise a customer's feasible arrival window [wha, whf] on a Gantt row:
    a light shaded rectangle spanning the row behind its bars, plus a thin
    bracket with end-ticks just below the row marking the exact bounds.

    Skipped when the window is effectively unconstrained (width exceeds
    WINDOW_MAX_DRAW_WIDTH_H, i.e. window_class="none").
    """
    if wha is None or whf is None:
        return
    color = color or COL["window"]
    width = whf - wha
    if width <= 0 or width > WINDOW_MAX_DRAW_WIDTH_H:
        return
    ax.add_patch(mpatches.Rectangle((wha, y - h / 2), width, h,
                                    facecolor=color, edgecolor="none",
                                    alpha=0.12, zorder=0.4))
    y_line = y - h / 2 - 0.07
    tick   = h * 0.16
    ax.plot([wha, whf], [y_line, y_line], color=color, lw=1.4, alpha=0.85, zorder=6)
    ax.plot([wha, wha], [y_line - tick, y_line + tick], color=color, lw=1.4, alpha=0.85, zorder=6)
    ax.plot([whf, whf], [y_line - tick, y_line + tick], color=color, lw=1.4, alpha=0.85, zorder=6)


def _shade_tod(ax, t_start, t_end):
    """Shade time-of-day bands (night / day / evening) across the full x-axis."""
    bands = [(0, 6, "#D6EAF8"), (6, 20, "#FEF9E7"), (20, 24, "#E8DAEF")]
    t = 0
    while t < t_end:
        day = int(t) // 24
        for h0, h1, col in bands:
            s = max(day * 24 + h0, t_start)
            e = min(day * 24 + h1, t_end)
            if e > s:
                ax.axvspan(s, e, color=col, alpha=0.25, zorder=0, lw=0)
        t += 24


def _draw_vlines(ax, vlines):
    """Draw a deduplicated set of vertical reference lines.

    vlines : iterable of (t, color, linewidth, alpha, linestyle) tuples.
    """
    seen = set()
    for (t, col, lw, alpha, ls) in vlines:
        key = round(t, 4)
        if key in seen:
            continue
        seen.add(key)
        ax.axvline(t, color=col, lw=lw, alpha=alpha, ls=ls, zorder=1)


# ═════════════════════════════════════════════════════════════════════════════
# plot_solution  (full-route MILP result)
# ═════════════════════════════════════════════════════════════════════════════

def plot_solution(sol, data, title="solution"):
    """
    Three-panel figure for a full-route MILP solution.

    Parameters
    ----------
    sol   : list of per-stop dicts (from MILP.extract_solution)
    data  : instance data dict (from instances.make_data())
    title : string used in the figure suptitle and the saved filename
    """
    N    = data["N"]
    tend = sol[-1]["ta"]

    fig, axes = plt.subplots(3, 1, figsize=(17, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.suptitle(f"{title}  —  {data['label']}", fontsize=12, fontweight="bold")

    # Pre-compute vertical event lines
    vlines = []
    for s in sol:
        t = ta = s["ta"]
        vlines.append((ta, "gray", 0.5, 0.30, "--"))
        if s["is_C"]:
            t += data["S"].get(s["i"], 0)
        if s["is_K"]:
            sigma_v  = int(s.get("sigma", 0))
            brk_v    = bool(s["b45"] or s["b15"] or s["b30"])
            v_v      = bool(s["y"] or brk_v or s["rho1"] or s["rho2"])
            mstop_t  = float(v_v)     * data.get("M_stop", {}).get(s["i"], 0.0)
            mseq_t   = float(sigma_v) * data.get("M_seq",  {}).get(s["i"], 0.0)
            if mstop_t > EPS:
                t += mstop_t
                vlines.append((t, COL["mstop"], 0.5, 0.25, ":"))
            if s["y"]:
                if s["tauq"] > EPS:
                    t += s["tauq"]
                    vlines.append((t, COL["queue"], 0.6, 0.28, ":"))
                if s["tauc"] > EPS:
                    if sigma_v == 0 and brk_v:
                        t += s["tauc"] + s["taub"]  # concurrent window
                    else:
                        t += s["tauc"]
                    vlines.append((t, COL["charge"], 0.7, 0.33, ":"))
        if s["taub"] > EPS:
            already = s["is_K"] and s["y"] and int(s.get("sigma", 0)) == 0 and \
                      bool(s["b45"] or s["b15"] or s["b30"])
            if not already:
                vlines.append((t, COL["brk"], 0.8, 0.48, "--"))
                t += s["taub"]
        if s["taur"] > EPS:
            vlines.append((t, COL["rest"], 1.0, 0.52, "--"))

    # ── Panel 1: Gantt ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Activity timeline", fontsize=10)
    _shade_tod(ax, 0, tend)
    Y, H = 0.5, 0.38

    for s in sol:
        i         = s["i"]
        is_K      = s["is_K"]; is_C = s["is_C"]
        brk_type  = ("b45" if s["b45"] else "b15" if s["b15"] else
                     "b30" if s["b30"] else None)
        rst_type  = "r1" if s["rho1"] else ("r2" if s["rho2"] else None)
        sigma_val = int(s.get("sigma", 0))

        if i > 0:
            _bar(ax, sol[i - 1]["td"], s["ta"] - sol[i - 1]["td"],
                 Y, H, COL["drive"], label=f"drv→{i}", fontsize=6.5)
        t        = s["ta"]
        brk_drew = False
        mseq_t   = 0.0

        if is_C:
            _draw_time_window(ax, data.get("Wha", {}).get(i),
                              data.get("Whf", {}).get(i), Y, H)
            svc = data["S"].get(i, 0)
            _bar(ax, t, svc, Y, H, COL["service"], label=f"C{i}", fontsize=7)
            t += svc

        if is_K:
            v_val   = bool(s["y"] or s["b45"] or s["b15"] or s["b30"] or
                           s["rho1"] or s["rho2"])
            mstop_t = float(v_val)     * data.get("M_stop", {}).get(i, 0.0)
            mseq_t  = float(sigma_val) * data.get("M_seq",  {}).get(i, 0.0)

            if mstop_t > EPS:
                _bar(ax, t, mstop_t, Y, H, COL["mstop"],
                     label="setup", fontsize=7, text_color="#333")
                t += mstop_t

            if s["y"] and s["tauq"] > EPS:
                _bar(ax, t, s["tauq"], Y, H, COL["queue"], label="Q", fontsize=7)
                t += s["tauq"]

            if s["y"] and s["tauc"] > EPS:
                tauc = s["tauc"]; taub = s["taub"]
                if sigma_val == 0 and brk_type:
                    # concurrent: break underlaid, charge on top
                    _bar(ax, t, tauc + taub, Y, H, COL["brk"],
                         label=None, fontsize=7, text_color="#333")
                    _bar(ax, t, tauc, Y, H, COL["charge"],
                         label=f"CHG\n{s['ea']:.0f}→{s['ed']:.0f}", fontsize=6.5)
                    t += tauc + taub
                    brk_drew = True
                else:
                    # sequential or charge-only: charge first
                    _bar(ax, t, tauc, Y, H, COL["charge"],
                         label=f"CHG\n{s['ea']:.0f}→{s['ed']:.0f}", fontsize=6.5)
                    t += tauc

        if brk_type and s["taub"] > EPS and not brk_drew:
            _bar(ax, t, s["taub"], Y, H, COL["brk"],
                 label=brk_type.upper(), fontsize=7, text_color="#333")
            t += s["taub"]

        if rst_type and s["taur"] > EPS:
            _bar(ax, t, s["taur"], Y, H, COL["rest"],
                 label=f"RST-{rst_type}", fontsize=7)
            t += s["taur"]

        if is_K and mseq_t > EPS:
            _bar(ax, t, mseq_t, Y, H, COL["mseq"], label="repos", fontsize=7)

        typ = "●C" if is_C else ("▲K" if is_K else ("O" if i == 0 else "D"))
        ax.text(s["ta"], Y + H / 2 + 0.06, f"{typ}{i}",
                ha="center", va="bottom", fontsize=6,
                color="#444", rotation=90, clip_on=True)

    _draw_vlines(ax, vlines)
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.0)   # room for stop labels above bars, windows below
    ax.set_xlim(-0.2, tend * 1.02)
    patches = [mpatches.Patch(color=v, label=k.replace("_", "").title())
               for k, v in COL.items()]
    patches += [
        mpatches.Patch(color="#D6EAF8", alpha=0.6, label="night 0-6h"),
        mpatches.Patch(color="#FEF9E7", alpha=0.6, label="day 6-20h"),
        mpatches.Patch(color="#E8DAEF", alpha=0.6, label="evening 20-24h"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=7, ncol=5)

    # ── Panel 2: SOC ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Battery state of charge", fontsize=10)
    _shade_tod(ax2, 0, tend)
    tpts, spts = [], []
    for s in sol:
        ta, td, ea, ed = s["ta"], s["td"], s["ea"], s["ed"]
        is_K  = s["is_K"]
        tauq  = s["tauq"] if is_K else 0
        tauc  = s["tauc"] if is_K else 0
        v_val = bool(s.get("y") or s.get("b45") or s.get("b15") or s.get("b30") or
                     s.get("rho1") or s.get("rho2")) if is_K else False
        mstop_t = float(v_val) * data.get("M_stop", {}).get(s["i"], 0.0) if is_K else 0.0
        tpts.append(ta); spts.append(ea)
        if td - ta > EPS:
            tcs = ta + mstop_t + tauq; tce = tcs + tauc
            if mstop_t + tauq > EPS:
                tpts.append(tcs); spts.append(ea)
            if tauc > EPS:
                tpts.append(tce); spts.append(ed)
            tpts.append(td); spts.append(ed)
    ax2.plot(tpts, spts, color=COL["drive"], lw=2, label="SOC", zorder=2)
    ax2.fill_between(tpts, spts, alpha=0.10, color=COL["drive"])
    for s in sol:
        if s["is_K"] and s["y"] and s["ed"] - s["ea"] > 0.5:
            ts = s["ta"] + s["tauq"]
            te = ts + s["tauc"]
            ax2.annotate("", xy=(te, s["ed"]), xytext=(ts, s["ea"]),
                         arrowprops=dict(arrowstyle="->",
                                         color=COL["charge"], lw=1.5),
                         zorder=3)
            ax2.text((ts + te) / 2, (s["ea"] + s["ed"]) / 2,
                     f"+{s['ed'] - s['ea']:.0f}",
                     ha="center", fontsize=7, color=COL["charge"])
    ax2.axhline(data["Emin"], color="red", ls=":", lw=1.2,
                label=f"E_min={data['Emin']} kWh")
    ax2.axhline(data["Ecap"], color="gray", ls=":", lw=1.2,
                label=f"E_cap={data['Ecap']} kWh")
    _draw_vlines(ax2, vlines)
    ax2.set_ylabel("kWh")
    ax2.set_ylim(0, data["Ecap"] * 1.15)
    ax2.legend(fontsize=8, ncol=3, loc="upper right")

    # ── Panel 3: HoS accumulators ─────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_title("HoS accumulators (at arrival)", fontsize=10)
    _shade_tod(ax3, 0, tend)
    cdt, cdv, sdt, sdv, swt, swv = [], [], [], [], [], []
    for s in sol:
        ta, td = s["ta"], s["td"]
        r_cd  = s["b45"] or s["b30"] or s["rho1"] or s["rho2"]
        r_rho = s["rho1"] or s["rho2"]
        cdt.append(ta); cdv.append(s["cd"])
        sdt.append(ta); sdv.append(s["sd"])
        swt.append(ta); swv.append(s["sw"])
        if td - ta > EPS:
            cdt.append(td); cdv.append(0.0 if r_cd  else s["cd"])
            sdt.append(td); sdv.append(0.0 if r_rho else s["sd"])
            swt.append(td); swv.append(0.0 if r_rho else s["sw"])
    ax3.plot(cdt, cdv, "o-", color="#E74C3C", lw=1.5, ms=3, label="Consec. driving")
    ax3.plot(sdt, sdv, "s-", color="#3498DB", lw=1.5, ms=3, label="Shift driving")
    ax3.plot(swt, swv, "^-", color="#1ABC9C", lw=1.5, ms=3, label="Shift working")
    ax3.axhline(data["Tdrv_cons"], color="#E74C3C", ls=":", lw=1.2, alpha=0.7,
                label=f"max consec {data['Tdrv_cons']}h")
    ax3.axhline(data["Tdrv_sh1"],  color="#3498DB", ls=":", lw=1.2, alpha=0.7,
                label=f"max shift drv {data['Tdrv_sh1']}h")
    ax3.axhline(data["Twrk_sh"],   color="#1ABC9C", ls=":", lw=1.2, alpha=0.7,
                label=f"max shift wk {data['Twrk_sh']}h")
    _draw_vlines(ax3, vlines)
    ax3.set_ylabel("Hours")
    ax3.set_xlabel("Time (h)")
    ax3.legend(fontsize=7, ncol=3, loc="upper left")

    plt.tight_layout()
    _ensure_fig_dir()
    fname = os.path.join(FIGURES_DIR, f"solution_{title}_{int(time.time())}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {fname}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# plot_simulation_results  (stochastic simulation run)
# ═════════════════════════════════════════════════════════════════════════════

def plot_simulation_results(results, full_data, title="simulation", save=True, show=False):
    """
    Five-panel figure for a complete simulation run.

    Parameters
    ----------
    results   : dict returned by Simulation.run_simulation
    full_data : instance data dict from instances.make_data()
    title     : string for suptitle and filename
    save      : bool — save PNG to figures/ directory

    Panels
    ------
    1. Simulation Gantt  — actual realised activity timeline
    2. Oracle Gantt      — hindsight-optimal schedule (same realised travel times)
    3. SOC               — battery state of charge
    4. HoS               — cd / sd / sw accumulators
    5. Look-ahead        — scenario objective scatter by decision stop
    """
    # Late imports — avoid circular dependency at module level.
    # INFEASIBLE_PENALTY is a plain scalar; importing it does not load HiGHS.
    from MILP import INFEASIBLE_PENALTY
    from oracle import check_simulation_feasibility

    states         = results["states"]
    actions        = results["actions"]
    td_list        = results["td_list"]
    D_actual_list  = results["D_actual_list"]
    durations_list = results["durations_list"]

    N     = full_data["N"]
    C_set = set(full_data["C"])
    K_set = set(full_data["K"])

    tend        = states[-1].t_arr
    oracle      = results.get("oracle", {})
    oracle_sol  = oracle.get("sol",      [])
    oracle_obj  = oracle.get("obj",      None)
    oracle_feas = oracle.get("feasible", False)
    tend_all    = max(tend, oracle_obj) if (oracle_feas and oracle_obj) else tend

    # ── Figure setup ─────────────────────────────────────────────────────
    gap_str = ""
    if oracle_feas and oracle_obj:
        gap = tend - oracle_obj
        gap_str = (f"  |  oracle {oracle_obj:.2f}h"
                   f"  |  gap {gap:+.2f}h ({100 * gap / oracle_obj:.1f}%)")

    fig, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 2.5, 2, 2, 2.5]})
    fig.suptitle(f"Simulation — {title}  (arrival {tend:.2f}h){gap_str}",
                 fontsize=11, fontweight="bold")

    # ── Panel 1: Simulation Gantt ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_title("Simulation — actual realisation", fontsize=10)
    _shade_tod(ax1, 0, tend)
    Y, H = 0.5, 0.40

    for i in range(N):
        st    = states[i]
        ta_i  = st.t_arr
        act   = actions[i]
        dur   = durations_list[i] if i < len(durations_list) else {}
        td_i  = td_list[i]        if i < len(td_list)        else ta_i
        is_K  = (i in K_set)
        is_C  = (i in C_set)
        y_val = int(act.get("y", 0))
        brk   = act.get("break_type")
        rst   = act.get("rest_type")
        sigma = int(dur.get("sigma", 0))
        mstop = dur.get("mstop", 0.0)
        mseq  = dur.get("mseq",  0.0)
        t     = ta_i
        brk_drew = False

        if is_C:
            _draw_time_window(ax1, full_data.get("Wha", {}).get(i),
                              full_data.get("Whf", {}).get(i), Y, H)
            svc = full_data["S"].get(i, 0.0)
            _bar(ax1, t, svc, Y, H, COL["service"], f"C{i}", fontsize=7)
            t += svc

        if is_K and mstop > EPS:
            _bar(ax1, t, mstop, Y, H, COL["mstop"], "setup", fontsize=7, text_color="#333")
            t += mstop

        if is_K and y_val:
            tauq = dur.get("tauq", 0.0)
            _bar(ax1, t, tauq, Y, H, COL["queue"], "Q", fontsize=7)
            t += tauq
            tauc = dur.get("tauc", 0.0)
            taub = dur.get("taub", 0.0)
            ea_v = st.e_arr
            ed_v = (states[i + 1].e_arr + full_data["E"].get(i, 0.0)
                    if i + 1 < len(states) else ea_v)
            if sigma == 0 and brk and tauc + taub > EPS:
                # concurrent: break underlaid across full window, charge on top
                _bar(ax1, t, tauc + taub, Y, H, COL["brk"],
                     label=None, fontsize=7, text_color="#333")
                _bar(ax1, t, tauc, Y, H, COL["charge"],
                     f"CHG\n{ea_v:.0f}→{ed_v:.0f}", fontsize=6)
                t += tauc + taub
                brk_drew = True
            else:
                # sequential or charge-only: charge first
                _bar(ax1, t, tauc, Y, H, COL["charge"],
                     f"CHG\n{ea_v:.0f}→{ed_v:.0f}", fontsize=6)
                t += tauc

        taub = dur.get("taub", 0.0)
        if brk and taub > EPS and not brk_drew:
            lbl = {"b45": "B45", "b15": "B15", "b30": "B30"}.get(brk, brk)
            _bar(ax1, t, taub, Y, H, COL["brk"], lbl, fontsize=7, text_color="#333")
            t += taub

        taur = dur.get("taur", 0.0)
        if rst and taur > EPS:
            lbl = "RST-r1" if rst == "r1" else "RST-r2"
            _bar(ax1, t, taur, Y, H, COL["rest"], lbl, fontsize=7)
            t += taur

        if is_K and mseq > EPS:
            _bar(ax1, t, mseq, Y, H, COL["mseq"], "repos", fontsize=7)
            t += mseq

        if i < N and i < len(D_actual_list):
            _bar(ax1, td_i, D_actual_list[i], Y, H, COL["drive"],
                 f"→{i + 1}", fontsize=6)

        typ = "●" if is_C else ("▲" if is_K else "O")
        ax1.text(ta_i, Y + H / 2 + 0.06, f"{typ}{i}",
                 ha="center", va="bottom", fontsize=6,
                 color="#444", rotation=90, clip_on=True)

    # Shade look-ahead windows
    LA_PENALTY = INFEASIBLE_PENALTY / 2
    for stp, sc_list in enumerate(results["scores_log"]):
        if stp == 0 or not sc_list:
            continue
        mean_horizon = sc_list[0][1]
        if mean_horizon < LA_PENALTY:
            t_la = states[stp].t_arr
            ax1.axvspan(t_la, mean_horizon, alpha=0.055,
                        color="navy", zorder=0, lw=0)

    ax1.set_yticks([])
    ax1.set_ylim(0.0, 1.0)   # room for stop labels above bars, windows below
    ax1.set_xlim(-0.1, tend_all * 1.04)
    patches = [mpatches.Patch(color=v, label=k.replace("_", " ").title())
               for k, v in COL.items()]
    patches += [
        mpatches.Patch(color="#D6EAF8", alpha=0.6, label="night 0-6h"),
        mpatches.Patch(color="#FEF9E7", alpha=0.6, label="day 6-20h"),
        mpatches.Patch(color="#E8DAEF", alpha=0.6, label="eve 20-24h"),
    ]
    ax1.legend(handles=patches, loc="upper left", fontsize=7, ncol=5)

    # ── Panel 2: Oracle Gantt ─────────────────────────────────────────────
    ax_or = axes[1]
    ax_or.set_title(
        "Oracle — hindsight-optimal schedule (same realised travel times)",
        fontsize=10)
    _shade_tod(ax_or, 0, tend_all)
    Yo, Ho = 0.5, 0.40

    if oracle_feas and oracle_sol:
        orsol = {s["i"]: s for s in oracle_sol}
        for i in range(N):
            s_or = orsol.get(i, {})
            if not s_or:
                continue
            ta_or    = s_or.get("ta", 0.0)
            td_or    = s_or.get("td", ta_or)
            t        = ta_or
            is_K     = i in K_set
            is_C     = i in C_set
            y_or     = s_or.get("y", 0)
            sigma_or = int(s_or.get("sigma", 0))
            brk_or   = ("b45" if s_or.get("b45") else
                        "b15" if s_or.get("b15") else
                        "b30" if s_or.get("b30") else None)
            rst_or   = "r1" if s_or.get("rho1") else ("r2" if s_or.get("rho2") else None)
            brk_drew_or = False
            mseq_or  = 0.0

            if is_C:
                _draw_time_window(ax_or, full_data.get("Wha", {}).get(i),
                                  full_data.get("Whf", {}).get(i), Yo, Ho)
                svc = full_data["S"].get(i, 0.0)
                _bar(ax_or, t, svc, Yo, Ho, COL["service"], f"C{i}", fontsize=7)
                t += svc

            if is_K:
                v_or    = bool(y_or or s_or.get("b45") or s_or.get("b15") or
                               s_or.get("b30") or s_or.get("rho1") or s_or.get("rho2"))
                mstop_or = float(v_or)    * full_data.get("M_stop", {}).get(i, 0.0)
                mseq_or  = float(sigma_or) * full_data.get("M_seq",  {}).get(i, 0.0)
                if mstop_or > EPS:
                    _bar(ax_or, t, mstop_or, Yo, Ho, COL["mstop"],
                         "setup", fontsize=7, text_color="#333")
                    t += mstop_or

            if is_K and y_or:
                tauq_or = s_or.get("tauq", 0.0)
                _bar(ax_or, t, tauq_or, Yo, Ho, COL["queue"], "Q", fontsize=7)
                t += tauq_or
                tauc_or = s_or.get("tauc", 0.0)
                taub_or = s_or.get("taub", 0.0)
                ea_or   = s_or.get("ea", 0.0)
                ed_or   = s_or.get("ed", 0.0)
                if sigma_or == 0 and brk_or:
                    # concurrent: break underlaid, charge on top
                    _bar(ax_or, t, tauc_or + taub_or, Yo, Ho, COL["brk"],
                         label=None, fontsize=7, text_color="#333")
                    _bar(ax_or, t, tauc_or, Yo, Ho, COL["charge"],
                         f"CHG\n{ea_or:.0f}→{ed_or:.0f}", fontsize=6)
                    t += tauc_or + taub_or
                    brk_drew_or = True
                else:
                    _bar(ax_or, t, tauc_or, Yo, Ho, COL["charge"],
                         f"CHG\n{ea_or:.0f}→{ed_or:.0f}", fontsize=6)
                    t += tauc_or

            taub_or_val = s_or.get("taub", 0.0)
            if brk_or and taub_or_val > EPS and not brk_drew_or:
                _bar(ax_or, t, taub_or_val, Yo, Ho,
                     COL["brk"], brk_or.upper(), fontsize=7, text_color="#333")
                t += taub_or_val

            if rst_or:
                taur_or = s_or.get("taur", 0.0)
                if taur_or > EPS:
                    lbl = f"RST-{rst_or}"
                    _bar(ax_or, t, taur_or, Yo, Ho, COL["rest"], lbl, fontsize=7)
                    t += taur_or

            if is_K and mseq_or > EPS:
                _bar(ax_or, t, mseq_or, Yo, Ho, COL["mseq"], "repos", fontsize=7)
                t += mseq_or

            if i < len(D_actual_list):
                _bar(ax_or, td_or, D_actual_list[i], Yo, Ho,
                     COL["drive"], f"→{i + 1}", fontsize=6)

            typ = "●" if is_C else ("▲" if is_K else "O")
            ax_or.text(ta_or, Yo + Ho / 2 + 0.06, f"{typ}{i}",
                       ha="center", va="bottom", fontsize=6,
                       color="#444", rotation=90, clip_on=True)

        ax_or.axvline(oracle_obj, color="green",  lw=2, ls="-",  alpha=0.9,
                      label=f"oracle arrival {oracle_obj:.2f}h")
        ax_or.axvline(tend,       color="crimson", lw=2, ls="--", alpha=0.9,
                      label=f"simulation arrival {tend:.2f}h")
    else:
        ax_or.text(0.5, 0.5, "Oracle infeasible or not run",
                   ha="center", va="center", transform=ax_or.transAxes,
                   fontsize=12, color="grey")

    ax_or.set_yticks([])
    ax_or.set_ylim(0.0, 1.0)   # room for stop labels above bars, windows below
    ax_or.legend(fontsize=8, loc="upper right")

    # ── Panel 3: SOC ─────────────────────────────────────────────────────
    ax2 = axes[2]
    ax2.set_title("Battery state of charge (at arrival)", fontsize=10)
    _shade_tod(ax2, 0, tend)

    t_pts = [s.t_arr for s in states]
    e_pts = [s.e_arr for s in states]

    t_full, e_full = [], []
    for i, (t, e) in enumerate(zip(t_pts, e_pts)):
        t_full.append(t); e_full.append(e)
        if i < N and int(actions[i].get("y", 0)):
            dur   = durations_list[i] if i < len(durations_list) else {}
            td_i  = td_list[i]        if i < len(td_list)        else t
            mstop = dur.get("mstop", 0.0)
            tauq  = dur.get("tauq",  0.0)
            tauc  = dur.get("tauc",  0.0)
            t_cs  = t + mstop + tauq   # charging starts after setup + queue
            t_ce  = t_cs + tauc
            e_dep = (e_pts[i + 1] + full_data["E"].get(i, 0.0)
                     if i + 1 < len(e_pts) else e)
            if mstop + tauq > EPS:
                t_full.append(t_cs); e_full.append(e)
            t_full.append(t_ce); e_full.append(e_dep)
            t_full.append(td_i); e_full.append(e_dep)

    ax2.plot(t_full, e_full, color=COL["drive"], lw=2, label="SOC", zorder=2)
    ax2.fill_between(t_full, e_full, alpha=0.10, color=COL["drive"])
    ax2.axhline(full_data["Emin"], color="red",  ls=":", lw=1.2,
                label=f"E_min={full_data['Emin']} kWh")
    ax2.axhline(full_data["Ecap"], color="gray", ls=":", lw=1.2,
                label=f"E_cap={full_data['Ecap']} kWh")
    ax2.set_ylabel("kWh")
    ax2.set_ylim(0, full_data["Ecap"] * 1.15)
    ax2.legend(fontsize=8, loc="upper right")

    # ── Panel 4: HoS accumulators ─────────────────────────────────────────
    ax3 = axes[3]
    ax3.set_title("HoS accumulators at arrival", fontsize=10)
    _shade_tod(ax3, 0, tend)

    ta_vals = [s.t_arr for s in states]
    cd_vals = [s.cd    for s in states]
    sd_vals = [s.sd    for s in states]
    sw_vals = [s.sw    for s in states]

    ax3.plot(ta_vals, cd_vals, "o-", color="#E74C3C", lw=1.5, ms=4,
             label="Consec. driving")
    ax3.plot(ta_vals, sd_vals, "s-", color="#3498DB", lw=1.5, ms=4,
             label="Shift driving")
    ax3.plot(ta_vals, sw_vals, "^-", color="#1ABC9C", lw=1.5, ms=4,
             label="Shift working")

    ax3.axhline(full_data["Tdrv_cons"], color="#E74C3C", ls=":", lw=1.2, alpha=0.7,
                label=f"max consec {full_data['Tdrv_cons']}h")
    ax3.axhline(full_data["Tdrv_sh1"],  color="#3498DB", ls=":", lw=1.2, alpha=0.7,
                label=f"max shift drv {full_data['Tdrv_sh1']}h")
    ax3.axhline(full_data["Twrk_sh"],   color="#1ABC9C", ls=":", lw=1.2, alpha=0.7,
                label=f"max shift wk {full_data['Twrk_sh']}h")

    for i, act in enumerate(actions):
        brk = act.get("break_type")
        rst = act.get("rest_type")
        if brk or rst:
            t_ev = states[i].t_arr
            col  = COL["rest"] if rst else COL["brk"]
            ax3.axvline(t_ev, color=col, lw=1.2, alpha=0.55, ls="--")

    ax3.set_ylabel("Hours")
    ax3.legend(fontsize=7, ncol=3, loc="upper left")

    # ── Panel 5: Look-ahead decision quality ─────────────────────────────
    ax4 = axes[4]
    ax4.set_title(
        "Look-ahead: scenario objectives by decision stop  "
        "(●=chosen ±σ,  ×=2nd-best,  dots=raw scenarios)",
        fontsize=10)

    PENALTY = INFEASIBLE_PENALTY / 2

    def _action_label(act):
        rst = act.get("rest_type")
        brk = act.get("break_type")
        y   = int(act.get("y", 0))
        if rst:  return f"REST-{rst}"
        if brk:  return f"BRK-{brk}"
        if y:    return "CHARGE"
        return "pass"

    def _action_color(act):
        rst = act.get("rest_type")
        brk = act.get("break_type")
        y   = int(act.get("y", 0))
        if rst == "r1":   return "#8E44AD"
        if rst == "r2":   return "#6C3483"
        if brk == "b45":  return "#E67E22"
        if brk == "b30":  return "#D68910"
        if brk == "b15":  return "#F1C40F"
        if y:             return "#E74C3C"
        return "#2C6FAC"

    rng_jit = np.random.default_rng(0)
    _shade_tod(ax4, 0, tend)
    line_x, line_y = [], []

    for stp, sc_list in enumerate(results["scores_log"]):
        if stp == 0 or not sc_list:
            continue
        t_x = states[stp].t_arr
        b_act, b_mean, b_std, b_feas, b_raw = sc_list[0]
        b_col = _action_color(b_act)

        feas_raw = [o for o in b_raw if o < PENALTY]
        if feas_raw:
            n   = len(feas_raw)
            jit = rng_jit.uniform(-0.12, 0.12, n)
            ax4.scatter(t_x + jit, feas_raw,
                        color=b_col, alpha=0.30, s=18, zorder=3, lw=0)

        if b_mean < PENALTY:
            ax4.errorbar(t_x, b_mean, yerr=b_std,
                         fmt="o", color=b_col, ms=9,
                         elinewidth=1.8, capsize=5, zorder=6,
                         label=_action_label(b_act) if stp == 1 else "")
            line_x.append(t_x); line_y.append(b_mean)

        if len(sc_list) > 1:
            _, s_mean, _, _, _ = sc_list[1]
            if s_mean < PENALTY:
                ax4.scatter(t_x, s_mean, color="grey", marker="x",
                            s=55, lw=2, zorder=5, alpha=0.65)

    if line_x:
        ax4.plot(line_x, line_y, color="dimgrey", lw=1.2, ls="--",
                 alpha=0.6, zorder=4, label="chosen action mean")

    ax4.axhline(states[-1].t_arr, color="crimson", ls="-", lw=1.8,
                label=f"simulation arrival {states[-1].t_arr:.2f}h", zorder=7)
    if oracle_feas and oracle_obj:
        ax4.axhline(oracle_obj, color="green", ls="-", lw=1.8,
                    label=f"oracle {oracle_obj:.2f}h", zorder=7)
        ax4.fill_between([0, tend_all], oracle_obj, states[-1].t_arr,
                         alpha=0.08, color="red", label="suboptimality gap")

    ax4.set_xlabel("Time (h)")
    ax4.set_ylabel("Horizon arrival time (h)")
    ax4.legend(fontsize=7, loc="upper left", ncol=3)

    plt.tight_layout()

    # ── Summary block ─────────────────────────────────────────────────────
    if oracle_feas and oracle_obj:
        gap = tend - oracle_obj
        feas_ok, feas_iss = check_simulation_feasibility(results, full_data)
        feas_tag = ("✓ feasible" if feas_ok
                    else f"✗ INFEASIBLE ({len(feas_iss)} HoS violations)")
        ora_opt = oracle.get("optimal", False)
        ora_gap = oracle.get("gap", float("nan"))
        ora_tag = ("optimal" if ora_opt
                   else f"feasible (gap≈{ora_gap:.1%})"
                        if not np.isnan(ora_gap) else "feasible")
        print(f"\n  ┌────────────────────────────────────────────────────┐")
        print(f"  │  Simulation arrival :   {tend:>8.3f} h                    │")
        print(f"  │  Simulation status  :   {feas_tag:<32}│")
        print(f"  │  Oracle  arrival    :   {oracle_obj:>8.3f} h  [{ora_tag}]  │")
        print(f"  │  Gap (sim − oracle) :   {gap:>+8.3f} h ({100 * gap / oracle_obj:.1f}%)              │")
        if not feas_ok:
            print(f"  │  ⚠  Gap meaningless — trajectory violates HoS.     │")
            print(f"  │     Increase horizon (H ≥ 6h recommended).          │")
        print(f"  └────────────────────────────────────────────────────┘")

    if save:
        _ensure_fig_dir()
        if results.get("fig_path"):
            fname = results["fig_path"]
        else:
            fname = os.path.join(FIGURES_DIR,
                                 f"simulation_{title}_{int(time.time())}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {fname}")

    if show:
        plt.show()
        plt.close()