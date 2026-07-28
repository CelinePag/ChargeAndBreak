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

    replot_run(run_ref) / load_run(run_ref)
        Rebuild the results dict from a saved solutions/<run_id>.json and
        re-render the five-panel figure — runs no longer plot automatically.

CLI
---
    python plots.py <run_id | instance-prefix | glob | json-path> [--show]

    Examples:
      python plots.py RshortCfewTlarge_10_RO_20260716_092310_001
      python plots.py RshortCfewTlarge_10            # every saved run of it
      python plots.py "solutions/Rmedium*_LA_*.json" --show

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
from matplotlib.ticker import FuncFormatter

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
    layby   = "#935116",   # layby / rest-area node (M8, break/rest-only stop)
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


# Night-only shading: a clean, saturated blue at full strength across the core
# night (within ``_NIGHT_CORE_H`` of midnight) that fades out with soft
# shoulders by ``_NIGHT_EDGE_H`` (dawn/dusk).  Daytime is left clear.  A
# saturated mid-navy (not near-black) is used so it reads as blue rather than
# desaturating to grey at partial opacity.  Keyed on absolute hour-of-day so
# it tiles seamlessly across days.
_NIGHT_COLOR     = "#3a5c92"   # clean, slightly muted deep blue
_NIGHT_ALPHA_MAX = 0.30        # opacity across the core night
_NIGHT_CORE_H    = 3.0         # full-strength within this many hours of midnight
_NIGHT_EDGE_H    = 5.5         # faded fully out this many hours from midnight


def _shade_tod(ax, t_start, t_end, step=0.25):
    """Shade the night hours across the full x-axis.

    Only the night is shaded — a clean deep blue at full strength across the
    core night, fading out with soft cosine shoulders by dawn/dusk; daytime is
    left clear.  Drawn as adjacent (non-overlapping) strips of width ``step``
    hours, so the alpha never accumulates.
    """
    t = t_start
    while t < t_end:
        e   = min(t + step, t_end)
        hod = ((t + e) / 2.0) % 24.0
        m   = min(hod, 24.0 - hod)          # hours from midnight, in [0, 12]
        if m < _NIGHT_CORE_H:
            night = 1.0
        elif m < _NIGHT_EDGE_H:
            # soft cosine shoulder from full strength → 0
            night = (1.0 + np.cos(np.pi * (m - _NIGHT_CORE_H)
                                  / (_NIGHT_EDGE_H - _NIGHT_CORE_H))) / 2.0
        else:
            night = 0.0
        if night > 0.0:
            ax.axvspan(t, e, color=_NIGHT_COLOR,
                       alpha=night * _NIGHT_ALPHA_MAX, zorder=0, lw=0)
        t += step


def _dual_time_axis(axes_list, t_end, T_START=8.0, minor_step=1.0):
    """Label the shared x-axis of every panel with hour-of-day AND elapsed time.

    Minor ticks (and a faint grid) every ``minor_step`` hours; labelled major
    ticks roughly every 3 h — widened to 6/12 h only if 3 h spacing would crowd
    the labels.  Each labelled tick shows two lines: the clock hour-of-day
    (absolute time mod 24) on top and the time elapsed since the route start
    (absolute − T_START) below.  Applied to *every* axis in ``axes_list``.
    """
    # Label spacing: a divisor of 24 (so ticks fall on clean clock hours),
    # aiming for ~3 h but backing off if that would produce too many labels.
    label_step = 24.0
    for cand in (3, 6, 12, 24):
        if t_end / cand <= 28:
            label_step = float(cand)
            break
    major = np.arange(0.0, t_end + 1e-6, label_step)
    minor = np.arange(0.0, t_end + 1e-6, minor_step)

    def _fmt(x, _pos):
        hod = int(round(x)) % 24
        el  = x - T_START
        return f"{hod:02d}:00\n{el:+.0f}h"

    for ax in axes_list:
        ax.set_xticks(major)
        ax.set_xticks(minor, minor=True)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
        ax.tick_params(axis="x", which="major", labelbottom=True,
                       labelsize=6.5, length=4)
        ax.tick_params(axis="x", which="minor", length=2)
        ax.grid(axis="x", which="major", color="0.5",  alpha=0.28, lw=0.6)
        ax.grid(axis="x", which="minor", color="0.6",  alpha=0.12, lw=0.4)
        ax.set_axisbelow(True)


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
    N     = data["N"]
    tend  = sol[-1]["ta"]
    L_set = set(data.get("L", []))          # layby / rest-area nodes (M8)

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
        is_K      = s["is_K"]; is_C = s["is_C"]; is_L = i in L_set
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

        if is_L:
            mlay_t = data.get("M_lay", {}).get(i, 0.0) * bool(brk_type or rst_type)
            if mlay_t > EPS:
                _bar(ax, t, mlay_t, Y, H, COL["layby"],
                     label="park", fontsize=7, text_color="#fff")
                t += mlay_t

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

        is_L = i in L_set
        typ  = ("●C" if is_C else "▲K" if is_K else "◆L" if is_L
                else "O" if i == 0 else "D")
        ax.text(s["ta"], Y + H / 2 + 0.06, f"{typ}{i}",
                ha="center", va="bottom", fontsize=6,
                color=COL["layby"] if is_L else "#444",
                rotation=90, clip_on=True)

    _draw_vlines(ax, vlines)
    ax.set_yticks([])
    ax.set_ylim(0.0, 1.0)   # room for stop labels above bars, windows below
    ax.set_xlim(-0.2, tend * 1.02)
    patches = [mpatches.Patch(color=v, label=k.replace("_", "").title())
               for k, v in COL.items()]
    patches += [
        mpatches.Patch(color=_NIGHT_COLOR, alpha=_NIGHT_ALPHA_MAX, label="night"),
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
    ax3.legend(fontsize=7, ncol=3, loc="upper left")

    _dual_time_axis(axes, tend, T_START=data.get("T_START", 8.0))
    ax3.set_xlabel("hour of day  /  +elapsed since start")

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
    L_set = set(full_data.get("L", []))     # layby / rest-area nodes (M8)

    tend        = states[-1].t_arr          # physical arrival at N, NO penalty
    # oracle is decoupled from method runs: prefer any embedded block (old
    # runs), else load the shared cache solutions/oracle_<instance>.json
    oracle      = results.get("oracle")
    if not oracle:
        from oracle import load_oracle_cache
        oracle = load_oracle_cache(full_data.get("title", "")) or {}
    oracle_sol  = oracle.get("sol",      [])
    oracle_obj  = oracle.get("obj",      None)   # includes beta·Σdelta (MILP obj)
    oracle_feas = oracle.get("feasible", False)

    # ── Window-penalty accounting (TW2) ──────────────────────────────────
    # The objective everywhere is  arrival + beta·(#missed windows).  tend is
    # the penalty-FREE physical arrival; the oracle obj already bakes the
    # penalty in — so a like-for-like gap needs both split out consistently.
    beta   = float(full_data.get("beta", 2.0))
    Wha    = full_data.get("Wha", {})
    Whf    = full_data.get("Whf", {})
    _n_miss_sim = 0                              # same rule as BEHDV.step (1e-3 tol)
    for i in C_set:
        ta_i = states[i].t_arr
        whf_i, wha_i = Whf.get(i), Wha.get(i)
        if   whf_i is not None and ta_i > whf_i + 1e-3: _n_miss_sim += 1
        elif wha_i is not None and ta_i < wha_i - 1e-3: _n_miss_sim += 1
    sim_pen  = beta * _n_miss_sim
    tend_pen = tend + sim_pen                    # arrival WITH penalty

    oracle_pen = beta * sum(int(s.get("delta", 0)) for s in oracle_sol
                            if s.get("is_C")) if (oracle_feas and oracle_sol) else 0.0

    # xlim reference uses penalty-free arrivals (physical timeline)
    oracle_arr = (oracle_obj - oracle_pen) if (oracle_feas and oracle_obj) else None
    tend_all   = max(tend, oracle_arr) if oracle_arr is not None else tend

    # ── Feasibility breaches (recorded by BEHDV, even with the supervisor off) ─
    # Each violation is {type, stop, amount, detail}; `stop` is the node where
    # the breach materialised.  We flag the run and mark the earliest breach.
    _vio = results.get("metrics", {}).get("violations")
    if _vio is None:
        _vio = list(getattr(results.get("vehicle", None), "violations", []) or [])
    _vio = [v for v in _vio if isinstance(v, dict)
            and v.get("stop") is not None
            and 0 <= int(v["stop"]) < len(states)]
    first_break_stop = min((int(v["stop"]) for v in _vio), default=None)
    t_break = states[first_break_stop].t_arr if first_break_stop is not None else None

    # ── Figure setup ─────────────────────────────────────────────────────
    arr_str = f"arrival {tend:.2f}h"
    if sim_pen > 0:
        arr_str += f" / {tend_pen:.2f}h +pen"

    gap_str = ""
    if oracle_feas and oracle_obj:
        gap     = tend - oracle_arr              # penalty-free vs penalty-free
        gap_pen = tend_pen - oracle_obj          # penalty-incl vs penalty-incl
        ora_txt = f"oracle {oracle_arr:.2f}h"
        if oracle_pen > 0:
            ora_txt += f" / {oracle_obj:.2f}h +pen"
        gap_txt = f"gap {gap:+.2f}h ({100 * gap / oracle_arr:.1f}%)"
        if sim_pen > 0 or oracle_pen > 0:
            gap_txt += f"  /  +pen {gap_pen:+.2f}h"
        gap_str = f"  |  {ora_txt}  |  {gap_txt}"

    infeas_str = ""
    if _vio:
        _vtypes = ", ".join(sorted({v.get("type", "?") for v in _vio}))
        infeas_str = (f"    ⚠ INFEASIBLE — {len(_vio)} violation(s) "
                      f"from stop {first_break_stop} [{_vtypes}]")

    fig, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 2.5, 2, 2, 2.5]})
    fig.suptitle(f"Simulation — {title}  ({arr_str}){gap_str}{infeas_str}",
                 fontsize=11, fontweight="bold",
                 color=("crimson" if _vio else "black"))

    # ── Panel 1: Simulation Gantt ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_title("Simulation — actual realisation", fontsize=10)
    _shade_tod(ax1, 0, tend_all)
    Y, H = 0.5, 0.40

    for i in range(N):
        st    = states[i]
        ta_i  = st.t_arr
        act   = actions[i]
        dur   = durations_list[i] if i < len(durations_list) else {}
        td_i  = td_list[i]        if i < len(td_list)        else ta_i
        is_K  = (i in K_set)
        is_C  = (i in C_set)
        is_L  = (i in L_set)
        y_val = int(act.get("y", 0))
        brk   = act.get("break_type")
        rst   = act.get("rest_type")
        sigma = int(dur.get("sigma", 0))
        mstop = dur.get("mstop", 0.0)
        mseq  = dur.get("mseq",  0.0)
        mlay  = dur.get("mlay",  0.0)
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

        if is_L and mlay > EPS:
            _bar(ax1, t, mlay, Y, H, COL["layby"], "park", fontsize=7, text_color="#fff")
            t += mlay

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

        typ  = "●" if is_C else ("▲" if is_K else ("◆" if is_L else "O"))
        ax1.text(ta_i, Y + H / 2 + 0.06, f"{typ}{i}",
                 ha="center", va="bottom", fontsize=6,
                 color=COL["layby"] if is_L else "#444",
                 rotation=90, clip_on=True)

    # (Removed the per-stop navy look-ahead-window shading: one filled axvspan
    # per decision stop accumulated into an opaque wash that buried the
    # day/night gradient on long routes.  The look-ahead horizon arrivals are
    # already visualised in Panel 5.)

    ax1.set_yticks([])
    ax1.set_ylim(0.0, 1.0)   # room for stop labels above bars, windows below
    ax1.set_xlim(-0.1, tend_all * 1.04)
    patches = [mpatches.Patch(color=v, label=k.replace("_", " ").title())
               for k, v in COL.items()]
    patches += [
        mpatches.Patch(color=_NIGHT_COLOR, alpha=_NIGHT_ALPHA_MAX, label="night"),
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
            is_L     = i in L_set
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

            if is_L:
                mlay_or = (full_data.get("M_lay", {}).get(i, 0.0)
                           * bool(brk_or or rst_or))
                if mlay_or > EPS:
                    _bar(ax_or, t, mlay_or, Yo, Ho, COL["layby"],
                         "park", fontsize=7, text_color="#fff")
                    t += mlay_or

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

            typ  = "●" if is_C else ("▲" if is_K else ("◆" if is_L else "O"))
            ax_or.text(ta_or, Yo + Ho / 2 + 0.06, f"{typ}{i}",
                       ha="center", va="bottom", fontsize=6,
                       color=COL["layby"] if is_L else "#444",
                       rotation=90, clip_on=True)

        ax_or.axvline(oracle_arr, color="green",  lw=2, ls="-",  alpha=0.9,
                      label=f"oracle arrival {oracle_arr:.2f}h")
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
    _shade_tod(ax2, 0, tend_all)

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
    _shade_tod(ax3, 0, tend_all)

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
    _shade_tod(ax4, 0, tend_all)
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

    ax4.set_ylabel("Horizon arrival time (h)")
    ax4.legend(fontsize=7, loc="upper left", ncol=3)

    # ── Infeasibility overlay: show WHERE and from which point it broke ────
    # A red curtain from the first breach to the end of the timeline (drawn on
    # every time-aligned panel), plus a per-breach ✗ marker with its criterion.
    if _vio and t_break is not None:
        from collections import defaultdict as _dd
        x_hi = tend_all * 1.04
        for _ax in (ax1, ax_or, ax2, ax3, ax4):
            _ax.axvspan(t_break, x_hi, color="red", alpha=0.06, zorder=0.5)
            _ax.axvline(t_break, color="crimson", lw=1.6, alpha=0.85, zorder=8)
        ax1.text(t_break, 0.99, f" ✗ INFEASIBLE from stop {first_break_stop}",
                 ha="left", va="top", fontsize=8.5, fontweight="bold",
                 color="crimson", zorder=9, clip_on=True)

        # Group breaches by stop so co-located criteria share one marker.
        _by_stop = _dd(list)
        for v in _vio:
            _by_stop[int(v["stop"])].append(v.get("type", "?"))
        for s, types in _by_stop.items():
            t_v = states[s].t_arr
            ax1.scatter([t_v], [0.15], marker="X", s=80, color="crimson",
                        edgecolor="black", lw=0.6, zorder=9, clip_on=True)
            ax1.text(t_v, 0.02, "\n".join(sorted(set(types))),
                     ha="center", va="bottom", fontsize=6, color="crimson",
                     zorder=9, clip_on=True)

    _dual_time_axis((ax1, ax_or, ax2, ax3, ax4), tend_all,
                    T_START=full_data.get("T_START", 8.0))
    ax4.set_xlabel("hour of day  /  +elapsed since start")

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


# ═════════════════════════════════════════════════════════════════════════════
# Replot from saved run files  (CLI: python plots.py <run_id>)
# ═════════════════════════════════════════════════════════════════════════════
# finalize_run (runner.py) no longer renders figures during a run; it saves
# everything needed into solutions/<run_id>.json instead.  The functions below
# rebuild the canonical results dict from that JSON + the instance JSON and
# re-render the standard five-panel figure at any later time.
#
#   python plots.py RshortCfewTlarge_10_RO_20260716_092310_001
#   python plots.py RshortCfewTlarge_10          # all saved runs of instance
#   python plots.py "solutions/Rmedium*_LA_*.json" --show
#
# Panel 5 (look-ahead scenario scatter) is empty on replots: per-stop scenario
# scores are not serialised to the solution JSON.

SOLUTIONS_DIR = "solutions"
INSTANCES_DIR = "instances"


def _resolve_solution_paths(run_ref: str, solutions_dir: str = SOLUTIONS_DIR):
    """
    Expand one CLI reference into a list of solution JSON paths.

    Accepts, in order of precedence:
      - a literal file path                 solutions/<run_id>.json
      - a glob pattern                      solutions/Rshort*_RO_*.json
      - a run_id (with or without .json)    <run_id>
      - a run_id / instance-name prefix     RshortCfewTlarge_10
        (matches every saved run of that instance)
    """
    import glob as _glob

    if os.path.isfile(run_ref):
        return [run_ref]
    if any(ch in run_ref for ch in "*?["):
        hits = sorted(_glob.glob(run_ref))
        if not hits:
            raise FileNotFoundError(f"no files matched pattern '{run_ref}'")
        return hits
    for cand in (os.path.join(solutions_dir, run_ref),
                 os.path.join(solutions_dir, run_ref + ".json")):
        if os.path.isfile(cand):
            return [cand]
    if os.path.isdir(solutions_dir):
        hits = sorted(
            os.path.join(solutions_dir, f)
            for f in os.listdir(solutions_dir)
            if f.startswith(run_ref) and f.endswith(".json")
            and not f.startswith("oracle_")
        )
        if hits:
            return hits
    raise FileNotFoundError(
        f"no solution file found for '{run_ref}' "
        f"(looked in '{solutions_dir}/'; pass a run_id, an instance-name "
        f"prefix, a glob pattern, or a JSON path)")


def _pwl_e2t(full_data: dict):
    """Return e→t interpolator along the instance's PWL charging curve."""
    Ebar = full_data["Ebar"]
    Tbar = full_data["Tbar"]
    rs   = sorted(Ebar)
    Es   = [Ebar[r] for r in rs]
    Ts   = [Tbar[r] for r in rs]

    def e2t(e):
        e = max(Es[0], min(Es[-1], e))
        for k in range(len(Es) - 1):
            if Es[k] <= e <= Es[k + 1]:
                span = Es[k + 1] - Es[k]
                return (Ts[k] + (e - Es[k]) / span * (Ts[k + 1] - Ts[k])
                        if span else Ts[k])
        return Ts[-1]

    return e2t


def _reconstruct_durations(traj, actions, full_data, E_real, td_list):
    """
    Legacy fallback: rebuild per-stop {taub, tauc, taur, tauq, sigma, v,
    mstop, mseq, mlay} for solution JSONs saved before durations_list was
    serialised.  Fixed components (setup, queue, charge via the PWL curve,
    reposition, rest) are recomputed from the instance data; taub absorbs
    the remaining dwell so the timeline is consistent by construction.
    """
    e2t    = _pwl_e2t(full_data)
    K_set  = set(full_data["K"])
    C_set  = set(full_data["C"])
    L_set  = set(full_data.get("L", []))
    Q      = full_data.get("Q", {})
    S      = full_data.get("S", {})
    M_stop = full_data.get("M_stop", {})
    M_seq  = full_data.get("M_seq", {})
    M_lay  = full_data.get("M_lay", {})
    Tr1    = full_data.get("Tr1", 11.0)
    Tr2    = full_data.get("Tr2", 9.0)
    N      = full_data["N"]

    durs = []
    for i in range(min(N, len(traj) - 1)):
        s, s1  = traj[i], traj[i + 1]
        a      = actions[i] if i < len(actions) else {}
        dwell  = max(0.0, td_list[i] - s["t_arr"])
        y      = int(a.get("y", 0))
        brk    = a.get("break_type")
        rst    = a.get("rest_type")
        e_dep  = s1["e_arr"] + E_real[i]
        tauc   = (max(0.0, e2t(e_dep) - e2t(s["e_arr"]))
                  if (y and e_dep - s["e_arr"] > 1e-6) else 0.0)
        tauq   = float(Q.get(i, 0.0)) if y else 0.0
        sigma  = 1 if (y and rst) else 0
        v      = 1 if (i in K_set and dwell > EPS) else 0
        mstop  = float(M_stop.get(i, 0.0)) * v
        mseq   = float(M_seq.get(i, 0.0)) * sigma
        mlay   = (float(M_lay.get(i, 0.0))
                  if (i in L_set and (brk or rst)) else 0.0)
        taur   = Tr2 if rst == "r2" else (Tr1 if rst == "r1" else 0.0)
        svc    = float(S.get(i, 0.0)) if i in C_set else 0.0
        taub   = (max(0.0, dwell - svc - mstop - tauq - tauc - mseq
                      - mlay - taur) if brk else 0.0)
        durs.append(dict(taub=taub, tauc=tauc, taur=taur, tauq=tauq,
                         sigma=sigma, v=v, mstop=mstop, mseq=mseq, mlay=mlay))
    return durs


def load_run(run_ref: str,
             solutions_dir: str = SOLUTIONS_DIR,
             instances_dir: str = INSTANCES_DIR):
    """
    Rebuild (results, full_data, run_id) from a saved solution JSON so that
    plot_simulation_results can be called long after the run finished.

    `run_ref` must resolve to exactly one solution file (run_id or path);
    use _resolve_solution_paths / the CLI for prefix and glob expansion.
    """
    import json
    from types import SimpleNamespace

    paths = _resolve_solution_paths(run_ref, solutions_dir)
    if len(paths) > 1:
        raise ValueError(
            f"'{run_ref}' matches {len(paths)} solution files; "
            f"pass a unique run_id (e.g. {os.path.basename(paths[0])})")
    sol_path = paths[0]

    with open(sol_path, "r", encoding="utf-8") as fh:
        sol = json.load(fh)

    run_id = sol.get("run_id") or os.path.splitext(os.path.basename(sol_path))[0]
    title  = sol.get("instance", "")
    diesel = title.endswith("_diesel")
    stem   = title[:-len("_diesel")] if diesel else title

    inst_path = os.path.join(instances_dir, stem + ".json")
    if not os.path.isfile(inst_path):
        raise FileNotFoundError(
            f"instance file '{inst_path}' not found for run '{run_id}'")

    # Late imports — instance_io/runner_dispatch import chains lead back to
    # plots.py; importing them lazily avoids a circular import at load time.
    from instance_io import load_instance_json
    full_data, D_real, E_real, _cv = load_instance_json(inst_path)
    if diesel:
        from runner_dispatch import _apply_diesel_mode
        full_data, D_real, E_real = _apply_diesel_mode(full_data, D_real, E_real)

    traj    = sol["sim_trajectory"]
    actions = sol["actions"]
    states  = [SimpleNamespace(**s) for s in traj]
    N       = full_data["N"]
    from oracle import load_oracle_cache as _load_oracle_cache

    D_actual = sol.get("D_actual_list") or list(D_real)
    td_list  = sol.get("td_list") or [
        traj[i + 1]["t_arr"] - D_actual[i]
        for i in range(min(N, len(traj) - 1))
    ]
    durations = sol.get("durations_list") or _reconstruct_durations(
        traj, actions, full_data, E_real, td_list)

    results = dict(
        states           = states,
        actions          = actions,
        scores_log       = [],          # not serialised — panel 5 stays empty
        td_list          = td_list,
        D_actual_list    = D_actual,
        durations_list   = durations,
        total_time       = traj[-1]["t_arr"],
        wall_clock       = sol.get("wall_clock_s", 0.0),
        oracle           = (sol.get("oracle")
                            or _load_oracle_cache(sol.get("instance", ""))
                            or {}),
        metrics          = sol.get("metrics", {}),
        sol_path         = sol_path,
        run_id           = run_id,
        fig_path         = os.path.join(FIGURES_DIR, f"{run_id}.png"),
    )
    return results, full_data, run_id


def replot_run(run_ref: str,
               show: bool = False,
               save: bool = True,
               solutions_dir: str = SOLUTIONS_DIR,
               instances_dir: str = INSTANCES_DIR) -> str:
    """Render the five-panel figure for one saved run; returns the PNG path."""
    results, full_data, run_id = load_run(run_ref, solutions_dir, instances_dir)
    plot_simulation_results(results, full_data,
                            title=run_id, save=save, show=show)
    fig_path = results["fig_path"]
    plt.close("all")
    return fig_path


if __name__ == "__main__":
    import argparse
    import sys

    # The summary block prints Unicode box-drawing characters, which the
    # default Windows console encoding (cp1252) cannot represent.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Render the five-panel simulation figure from saved "
                    "solution JSON files, without re-running any algorithm. "
                    "Each reference may be a run_id, an instance-name prefix "
                    "(plots every saved run of that instance), a glob "
                    "pattern, or a path to a solution JSON.")
    parser.add_argument("runs", nargs="+",
                        help="run_id(s), instance prefix(es), glob(s), or "
                             "solution JSON path(s)")
    parser.add_argument("--show", action="store_true", default=False,
                        help="open an interactive window for each figure")
    parser.add_argument("--solutions_dir", default=SOLUTIONS_DIR)
    parser.add_argument("--instances_dir", default=INSTANCES_DIR)
    args = parser.parse_args()

    targets = []
    for ref in args.runs:
        targets.extend(_resolve_solution_paths(ref, args.solutions_dir))
    # de-duplicate while preserving order
    seen = set()
    targets = [p for p in targets if not (p in seen or seen.add(p))]

    print(f"  Replotting {len(targets)} run(s)")
    n_fail = 0
    for p in targets:
        try:
            replot_run(p, show=args.show,
                       solutions_dir=args.solutions_dir,
                       instances_dir=args.instances_dir)
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {os.path.basename(p)}: {type(e).__name__}: {e}")
    if n_fail:
        raise SystemExit(f"  {n_fail}/{len(targets)} replot(s) failed")