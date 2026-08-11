# -*- coding: utf-8 -*-
"""
diesel_vs_ev_figure.py -- side-by-side execution of ONE instance under the EV
and the diesel (HoS-only) model, on a shared time axis.

The aggregate diesel comparison (additional_figures.py) shows the ~11% duration
gap but not where it comes from.  This figure shows the mechanism on a single
route: identical driving, identical HoS rules, and a stack of EV-only dwell --
charging, queueing and the extra pull-offs the charging stops require -- of
which only the part that fits inside a mandatory break is free.

Everything is reconstructed from the instance JSON + the two solution JSONs;
nothing is hand-placed, so any paired instance can be passed on the CLI.

Block order inside a dwell mirrors BEHDV.advance's departure equation:
    CS       ta + M_stop + Q + tau_c + M_seq + tau_b + tau_r
    customer ta + S + tau_b + tau_r
    layby    ta + M_lay + tau_b + tau_r

Run:
    python -m src.plot.diesel_vs_ev_figure                       # default instance
    python -m src.plot.diesel_vs_ev_figure RshortCfewTnone_10
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from src import paths as _paths

# Median-delta instance of the RmediumCfewTnone family (EV 59.03 h, diesel
# 52.13 h, delta 6.90 h = the family median): representative, and small enough
# that every charging stop is still individually legible on the time axis.
DEFAULT_INSTANCE = "RmediumCfewTnone_11"

# Oracle mode has a much smaller pool: only the RlongCfewTnone family was
# solved for both vehicles.  This one sits closest to the paired mean gap
# (9.93 h vs 9.99 h) and its diesel side is proven optimal.
DEFAULT_ORACLE_INSTANCE = "RlongCfewTnone_2"

# ── activity palette (shared with concept_solution_figure.py) ────────────────
INK, MUT = "#222222", "#666666"
C_DRIVE  = "#d9d9d9"
C_MAN    = "#141414"    # M_stop / M_seq / M_lay -- pulling off and back on
C_QUEUE  = "#e02b2b"
C_CHARGE = "#1e7a3c"
C_SERV   = "#e8822d"
C_BRK_F, C_BRK_E = "#ffffff", "#2b7bba"   # break: outline only  (short)
C_RST_F, C_RST_E = "#b9cfe6", "#1f4e79"   # rest:  filled       (long)
C_EV, C_DIESEL = "#1e7a3c", "#8a5a2b"

KIND_STYLE = {
    "man":     dict(fc=C_MAN,    ec=C_MAN,    hatch=None),
    "queue":   dict(fc=C_QUEUE,  ec=C_QUEUE,  hatch=None),
    "charge":  dict(fc=C_CHARGE, ec=C_CHARGE, hatch=None),
    "service": dict(fc=C_SERV,   ec="#a85a17", hatch=None),
    "break":   dict(fc=C_BRK_F,  ec=C_BRK_E,  hatch="//////"),
    "rest":    dict(fc=C_RST_F,  ec=C_RST_E,  hatch="\\\\\\\\"),
    "wait":    dict(fc="#f7f7f7", ec="#999999", hatch=".."),
}


# ══════════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _latest_greedy(instance: str) -> dict:
    """Newest finished, non-variant greedy solution for `instance`."""
    best = None
    for path in sorted(glob.glob(_paths.solutions(f"{instance}_GREEDY_*.json"))):
        sol = json.load(open(path, encoding="utf-8"))
        if sol.get("sim_arrival_h") is None or sol.get("variant"):
            continue
        # `RmediumCfewTnone_1_GREEDY_*` also globs `..._11_GREEDY_*`? no --
        # the underscore after the seed makes the stem exact.  But the EV glob
        # would still catch the diesel copy if it were named `<inst>_diesel`;
        # it is `<inst>__diesel`, so filter on the stored instance name.
        if sol.get("instance") != instance:
            continue
        best = path                       # sorted -> last is the newest stamp
    if best is None:
        raise SystemExit(f"no finished greedy run found for '{instance}'")
    return json.load(open(best, encoding="utf-8"))


def _oracle_run(instance: str, fd: dict) -> dict:
    """
    Load the hindsight-oracle cache and adapt it to the solution-JSON shape
    that `reconstruct` consumes.

    The cache stores the raw MILP solution (one record per stop) rather than a
    simulated trajectory, so the per-stop overheads have to be rebuilt from the
    indicators exactly as BEHDV.advance does -- v.Mstop, sigma.Mseq, Mlay.xsum.
    They are not stored because in the model they are coefficients, not
    variables.
    """
    path = _paths.solutions(f"oracle_{instance}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"no oracle cache at {path}")
    cache = json.load(open(path, encoding="utf-8"))
    if not cache.get("feasible") or not cache.get("sol"):
        raise SystemExit(f"oracle cache for '{instance}' is not a feasible solution")

    sol   = cache["sol"]
    N     = fd["N"]
    K, C, L = set(fd["K"]), set(fd["C"]), set(fd["L"])
    D_act = [float(cache["D_actual"][str(i)]) for i in range(N)]

    traj, acts, durs, td_list = [], [], [], []
    for i, s in enumerate(sol):
        traj.append(dict(stop=i, t_arr=s["ta"], e_arr=s["ea"],
                         cd=s["cd"], sd=s["sd"], sw=s["sw"]))
        if i >= N:
            break
        brk = ("b45" if s["b45"] else "b15" if s["b15"] else "b30" if s["b30"] else None)
        rst = ("r1" if s["rho1"] else "r2" if s["rho2"] else None)
        y, sigma = int(s["y"]), int(s["sigma"])
        v = int(i in K and bool(y or brk or rst))
        acts.append(dict(y=y, break_type=brk, rest_type=rst))
        durs.append(dict(
            taub=s["taub"], tauc=s["tauc"], taur=s["taur"], tauq=s["tauq"],
            sigma=sigma, v=v,
            mstop=v * fd["M_stop"].get(str(i), 0.0) if i in K else 0.0,
            mseq=sigma * fd["M_seq"].get(str(i), 0.0) if i in K else 0.0,
            mlay=(fd["M_lay"].get(str(i), 0.0) if (i in L and (brk or rst)) else 0.0),
            wait=s.get("wait", 0.0)))
        td_list.append(s["td"])

    return dict(instance=instance, method="oracle",
                sim_trajectory=traj, actions=acts, durations_list=durs,
                td_list=td_list, D_actual_list=D_act,
                duration_h=traj[-1]["t_arr"] - traj[0]["t_arr"],
                mip_gap=cache.get("gap"), optimal=cache.get("optimal"))


def _pwl(fd: dict):
    """Charging curve interpolators (energy <-> charging time)."""
    Ebar = {int(k): v for k, v in fd["Ebar"].items()}
    Tbar = {int(k): v for k, v in fd["Tbar"].items()}
    rs = sorted(Ebar)
    Es = [Ebar[r] for r in rs]
    Ts = [Tbar[r] for r in rs]

    def e2t(e: float) -> float:
        e = max(Es[0], min(Es[-1], e))
        for k in range(len(Es) - 1):
            if Es[k] <= e <= Es[k + 1]:
                span = Es[k + 1] - Es[k]
                return Ts[k] + (e - Es[k]) / span * (Ts[k + 1] - Ts[k]) if span else Ts[k]
        return Ts[-1]

    return Es, Ts, e2t


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULE RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct(sol: dict, fd: dict, E_real: list) -> dict:
    """
    Turn a solution JSON into drawable primitives.

    Returns dict with:
      blocks   [(kind, t0, t1)]      dwell activity blocks, absolute hours
      drive    [(t0, t1)]            merged driving legs
      soc      ndarray (t, kWh)      SOC polyline (flat for diesel: E == 0)
      cd, sd   ndarray (t, h)        HoS accumulators read off the trajectory
      used     {stop: kind}          stops where something happened
      budget   {component: hours}    time-budget decomposition
    """
    N    = fd["N"]
    K    = set(fd["K"]); C = set(fd["C"]); L = set(fd["L"])
    traj = sol["sim_trajectory"]
    acts = sol["actions"]
    durs = sol["durations_list"]
    td_l = sol["td_list"]
    D_act = sol["D_actual_list"]
    Es, Ts, e2t = _pwl(fd)

    blocks, drive, used = [], [], {}
    soc = [(traj[0]["t_arr"], traj[0]["e_arr"])]
    budget = dict(drive=0.0, charge=0.0, queue=0.0, man=0.0,
                  service=0.0, brk=0.0, rest=0.0, wait=0.0)

    for i in range(N):
        ta, td = traj[i]["t_arr"], td_l[i]
        du, a = durs[i], acts[i]
        taub = du.get("taub") or 0.0
        tauc = du.get("tauc") or 0.0
        taur = du.get("taur") or 0.0
        tauq = du.get("tauq") or 0.0
        man  = (du.get("mstop") or 0.0) + (du.get("mseq") or 0.0) + (du.get("mlay") or 0.0)
        serv = fd["S"].get(str(i), 0.0) if i in C else 0.0
        wait = du.get("wait") or 0.0        # oracle only; the simulation never waits

        budget["charge"] += tauc; budget["queue"] += tauq; budget["man"] += man
        budget["brk"] += taub; budget["rest"] += taur; budget["service"] += serv
        budget["wait"] += wait

        if td - ta > 1e-3:
            t = ta
            seq = []
            if i in K:
                seq = [("man", du.get("mstop") or 0.0), ("queue", tauq),
                       ("charge", tauc), ("man", du.get("mseq") or 0.0),
                       ("break", taub), ("rest", taur)]
            elif i in C:
                seq = [("service", serv), ("break", taub), ("rest", taur),
                       ("wait", wait)]
            elif i in L:
                seq = [("man", du.get("mlay") or 0.0), ("break", taub),
                       ("rest", taur), ("wait", wait)]
            for kind, dt in seq:
                if dt <= 1e-6:
                    continue
                if kind == "charge":
                    # SOC rises along the PWL curve; insert the knot crossings
                    e_arr = traj[i]["e_arr"]
                    e_dep = traj[i + 1]["e_arr"] + E_real[i]
                    t0 = e2t(e_arr)
                    soc.append((t, e_arr))
                    for Ek, Tk in zip(Es, Ts):
                        if e_arr < Ek < e_dep:
                            soc.append((t + (Tk - t0), Ek))
                    soc.append((t + dt, e_dep))
                blocks.append((kind, t, t + dt))
                t += dt
            used[i] = ("charge" if a.get("y") else
                       "rest" if a.get("rest_type") else
                       "break" if a.get("break_type") else "service")

        drive.append([td, traj[i + 1]["t_arr"]])
        budget["drive"] += D_act[i]
        soc.append((td, traj[i + 1]["e_arr"] + E_real[i]))
        soc.append((traj[i + 1]["t_arr"], traj[i + 1]["e_arr"]))

    # merge legs across drive-through stops (dwell 0)
    merged = []
    for t0, t1 in drive:
        if t1 - t0 <= 1e-9:
            continue
        if merged and abs(t0 - merged[-1][1]) < 2.5e-3:
            merged[-1][1] = t1
        else:
            merged.append([t0, t1])

    # HoS accumulators: read the value off the trajectory rather than replaying
    # the reset rules, so the panel can never drift from what was simulated.
    # cd after the action at i = cd on arrival at i+1 minus the leg just driven.
    def counter(key):
        pts = [(traj[0]["t_arr"], traj[0][key])]
        for i in range(N):
            post = traj[i + 1][key] - (D_act[i] if key in ("cd", "sd") else 0.0)
            pts.append((td_l[i], traj[i][key]))     # flat through the dwell
            pts.append((td_l[i], max(0.0, post)))   # reset (if any) on departure
            pts.append((traj[i + 1]["t_arr"], traj[i + 1][key]))
        return np.array(pts)

    return dict(blocks=blocks, drive=[(a, b) for a, b in merged],
                soc=np.array(soc), cd=counter("cd"), sd=counter("sd"),
                used=used, budget=budget,
                t0=traj[0]["t_arr"], t1=traj[-1]["t_arr"])


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def build(instance: str, out_stem: str | None = None,
          method: str = "greedy") -> str:
    inst_json = json.load(open(_paths.instances(f"{instance}.json"), encoding="utf-8"))
    fd     = inst_json["instance"]
    E_real = inst_json["E_real"]

    # The diesel transform zeroes E and Q in memory; rebuild the same view here
    # so the reconstruction uses the parameters the diesel run actually saw.
    fd_di = dict(fd)
    fd_di["Q"] = {k: 0.0 for k in fd["Q"]}

    if method == "oracle":
        ev_sol = _oracle_run(instance, fd)
        di_sol = _oracle_run(f"{instance}__diesel", fd_di)
        gaps = " / ".join(f"{100*(s.get('mip_gap') or 0.0):.2f} %" for s in (ev_sol, di_sol))
        run_lbl = f"hindsight oracle (MILP optimum, gap {gaps})"
    else:
        ev_sol = _latest_greedy(instance)
        di_sol = _latest_greedy(f"{instance}__diesel")
        run_lbl = "greedy policy"

    ev = reconstruct(ev_sol, fd,    E_real)
    di = reconstruct(di_sol, fd_di, [0.0] * len(E_real))

    N = fd["N"]
    K, C, L = set(fd["K"]), set(fd["C"]), set(fd["L"])
    km = fd["km"]
    cumkm = [0.0]
    for i in range(N):
        cumkm.append(cumkm[-1] + km[str(i)])
    TOTKM = cumkm[-1]

    T0 = ev["t0"]
    ev_end = ev["t1"] - T0
    di_end = di["t1"] - T0
    TMAX = max(ev_end, di_end)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.edgecolor": "#aaaaaa", "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(13.0, 9.8), dpi=200)
    gs = fig.add_gridspec(
        8, 1, height_ratios=[0.85, 0.48, 0.92, 0.80, 0.20, 0.48, 0.80, 0.66],
        left=0.085, right=0.945, top=0.880, bottom=0.115, hspace=0.46)
    ax_route = fig.add_subplot(gs[0])
    ax_ev    = fig.add_subplot(gs[1])
    ax_soc   = fig.add_subplot(gs[2], sharex=ax_ev)
    ax_evh   = fig.add_subplot(gs[3], sharex=ax_ev)
    ax_gap   = fig.add_subplot(gs[4]); ax_gap.axis("off")
    ax_di    = fig.add_subplot(gs[5], sharex=ax_ev)
    ax_dih   = fig.add_subplot(gs[6], sharex=ax_ev)
    ax_bud   = fig.add_subplot(gs[7])

    # ── title ───────────────────────────────────────────────────────────────
    delta = ev_end - di_end
    fig.text(0.085, 0.977, "Same route, same driving, same HoS rules -- "
             "why the EV finishes later", fontsize=15, fontweight="bold",
             color=INK, va="top")
    fig.text(0.085, 0.943,
             f"Instance {instance}  ({TOTKM:.0f} km, {len(K)} charging stations, "
             f"{len(L)} rest areas, {len(C)} customers)  |  {run_lbl}, "
             f"identical realised travel times",
             fontsize=9.5, color=MUT, va="top")
    fig.text(0.085, 0.917,
             f"EV {ev_end:.2f} h  vs  diesel {di_end:.2f} h   =   "
             f"+{delta:.2f} h ({100*delta/ev_end:.1f} %), all of it dwell",
             fontsize=9.5, color=INK, va="top", fontweight="bold")

    # ── route strip ─────────────────────────────────────────────────────────
    ax_route.set_xlim(-TOTKM * 0.012, TOTKM * 1.012)
    ax_route.set_ylim(-1.85, 1.75)
    ax_route.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax_route.spines[s].set_visible(False)
    ax_route.plot([0, TOTKM], [0, 0], color="#444444", lw=1.3, zorder=1)
    for i in range(N + 1):
        x = cumkm[i]
        if i in L:
            ax_route.plot([x, x], [-0.10, 0.10], color="#bbbbbb", lw=0.7, zorder=2)
        if i in K:
            ax_route.scatter([x], [0], marker="^", s=42, c=C_CHARGE,
                             edgecolors="#0d3c1d", linewidths=0.4, alpha=0.45, zorder=3)
        if i in C:
            ax_route.scatter([x], [0], marker="D", s=42, c=C_SERV,
                             edgecolors="#7a3d0d", linewidths=0.4, alpha=0.9, zorder=3)
    for i in sorted(ev["used"]):
        ax_route.plot([cumkm[i], cumkm[i]], [0.16, 0.86], color=C_EV, lw=1.0, zorder=4)
        ax_route.scatter([cumkm[i]], [0.95], marker="v", s=46, c=C_EV,
                         edgecolors="none", zorder=5)
    for i in sorted(di["used"]):
        ax_route.plot([cumkm[i], cumkm[i]], [-0.16, -0.86], color=C_DIESEL, lw=1.0, zorder=4)
        ax_route.scatter([cumkm[i]], [-0.95], marker="^", s=46, c=C_DIESEL,
                         edgecolors="none", zorder=5)
    ax_route.text(-TOTKM * 0.008, 1.32, f"EV stops ({len(ev['used'])})",
                  fontsize=8.6, color=C_EV, fontweight="bold", ha="left", va="center")
    ax_route.text(-TOTKM * 0.008, -1.32, f"diesel stops ({len(di['used'])})",
                  fontsize=8.6, color=C_DIESEL, fontweight="bold", ha="left", va="center")
    for x, lab in [(0.0, "O"), (TOTKM, "D")]:
        ax_route.scatter([x], [0], marker="s", s=70, c="#111111", zorder=6)
        ax_route.text(x, 0.16, lab, ha="center", va="bottom", fontsize=10,
                      fontweight="bold", color=INK)
    kmticks = np.linspace(0, TOTKM, 5)
    ax_route.set_xticks(kmticks)
    ax_route.set_xticklabels([f"{v:.0f}" + (" km" if j == len(kmticks) - 1 else "")
                              for j, v in enumerate(kmticks)],
                             fontsize=7.5, color="#555555")
    ax_route.tick_params(axis="x", length=2, color="#aaaaaa", pad=1)
    ax_route.legend(handles=[
        Line2D([], [], marker="^", color=C_CHARGE, markeredgecolor="#0d3c1d",
               markersize=7, linestyle="none", label="Charging station"),
        Line2D([], [], marker="D", color=C_SERV, markeredgecolor="#7a3d0d",
               markersize=6, linestyle="none", label="Customer"),
        Line2D([], [], color="#bbbbbb", lw=1.0, marker="|", markersize=9,
               linestyle="none", label="Rest area (no charger)")],
        loc="upper right", ncol=3, frameon=True, framealpha=1.0,
        edgecolor="#cccccc", fontsize=8.2, bbox_to_anchor=(1.0, 1.30),
        borderpad=0.4, handletextpad=0.35)
    fig.text(0.012, ax_route.get_position().y0 + ax_route.get_position().height / 2,
             "Route", fontsize=10.5, fontweight="bold", color=INK, ha="left", va="center")

    # ── activity rows ───────────────────────────────────────────────────────
    def draw_gantt(ax, rec, label, color):
        ax.set_ylim(0, 1); ax.set_yticks([])
        for t0, t1 in rec["drive"]:
            ax.add_patch(Rectangle((t0 - T0, 0.10), t1 - t0, 0.80,
                                   facecolor=C_DRIVE, edgecolor="#9a9a9a", lw=0.4))
        for kind, t0, t1 in rec["blocks"]:
            st = KIND_STYLE[kind]
            ax.add_patch(Rectangle((t0 - T0, 0.10), max(t1 - t0, 0.035), 0.80,
                                   facecolor=st["fc"], edgecolor=st["ec"],
                                   lw=0.5, hatch=st["hatch"]))
        end = rec["t1"] - T0
        ax.plot([end, end], [0.0, 1.0], color=color, lw=1.8, zorder=6)
        ax.scatter([end], [1.0], marker="v", s=52, c=color, zorder=7, clip_on=False)
        ax.text(end + TMAX * 0.008, 0.5, f"{end:.2f} h", ha="left", va="center",
                fontsize=8.6, fontweight="bold", color=color, zorder=8)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis="x", length=0)
        fig.text(0.012, ax.get_position().y0 + ax.get_position().height / 2,
                 label, fontsize=10.5, fontweight="bold", color=color,
                 ha="left", va="center")

    draw_gantt(ax_ev, ev, "EV", C_EV)
    draw_gantt(ax_di, di, "Diesel", C_DIESEL)
    ax_ev.set_xlim(-TMAX * 0.012, TMAX * 1.045)

    # ── SOC ─────────────────────────────────────────────────────────────────
    sp = ev["soc"]
    for kind, t0, t1 in ev["blocks"]:
        if kind == "charge":
            ax_soc.axvspan(t0 - T0, t1 - T0, color=C_CHARGE, alpha=0.15, lw=0)
    ax_soc.plot(sp[:, 0] - T0, sp[:, 1], color=C_EV, lw=1.5, zorder=3, label="EV state of charge")
    ax_soc.axhline(fd["Ecap"], color="#888888", ls=(0, (5, 4)), lw=0.9)
    ax_soc.axhline(fd["Emin"], color="#d62728", ls=(0, (5, 4)), lw=0.9)
    ax_soc.text(TMAX * 1.05, fd["Ecap"], "$E^{cap}$", fontsize=8, color="#666666", va="center")
    ax_soc.text(TMAX * 1.05, fd["Emin"], "$E^{min}$", fontsize=8, color="#d62728", va="center")
    ax_soc.plot([0, di_end], [fd["Ecap"] * 1.06] * 2, color=C_DIESEL, lw=1.5,
                ls=(0, (4, 2)), zorder=3, label="Diesel: no energy constraint (tank never binds)")
    ax_soc.set_ylim(0, fd["Ecap"] * 1.22)
    ax_soc.set_yticks([fd["Emin"], fd["Ecap"]])
    ax_soc.grid(axis="y", color="#eeeeee", lw=0.7); ax_soc.set_axisbelow(True)
    ax_soc.legend(loc="lower left", fontsize=8, frameon=True, ncol=2,
                  facecolor="white", edgecolor="none", framealpha=0.92,
                  handlelength=2.0, borderaxespad=0.25)
    plt.setp(ax_soc.get_xticklabels(), visible=False)
    ax_soc.tick_params(axis="x", length=0, labelsize=7.5)
    fig.text(0.012, ax_soc.get_position().y0 + ax_soc.get_position().height / 2,
             "SOC\n[kWh]", fontsize=10.5, fontweight="bold", color=INK, ha="left", va="center")

    # ── HoS accumulators ────────────────────────────────────────────────────
    def draw_hos(ax, rec, color, xlabels):
        ax.plot(rec["cd"][:, 0] - T0, rec["cd"][:, 1], color="#2b6cb0", lw=1.3,
                label="consecutive driving $t^{drv}$", zorder=3)
        ax.plot(rec["sd"][:, 0] - T0, rec["sd"][:, 1], color="#8a63b8", lw=1.1,
                ls=(0, (4, 2)), label="shift driving $t^{sd}$", zorder=3)
        ax.axhline(fd["Tdrv_cons"], color="#2b6cb0", ls=(0, (2, 3)), lw=0.9, alpha=0.85)
        ax.axhline(fd["Tdrv_sh1"], color="#8a63b8", ls=(0, (2, 3)), lw=0.9, alpha=0.85)
        ax.text(TMAX * 1.05, fd["Tdrv_cons"], "4.5 h", fontsize=7.5, color="#2b6cb0", va="center")
        ax.text(TMAX * 1.05, fd["Tdrv_sh1"], "9 h", fontsize=7.5, color="#8a63b8", va="center")
        ax.set_ylim(0, 10.6); ax.set_yticks([0, 4.5, 9])
        ax.grid(axis="y", color="#eeeeee", lw=0.7); ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)
        end = rec["t1"] - T0
        ax.axvline(end, color=color, lw=1.0, ls=(0, (3, 2)), zorder=2)
        if xlabels:
            ticks = np.arange(0, TMAX + 0.01, 6.0)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.0f}" for t in ticks], fontsize=7.5, color="#555555")
            ax.set_xlabel("hours since departure", fontsize=8.5, color="#555555", labelpad=1)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(axis="x", length=0)
        fig.text(0.012, ax.get_position().y0 + ax.get_position().height / 2,
                 "HoS\n[h]", fontsize=10.5, fontweight="bold", color=color,
                 ha="left", va="center")

    draw_hos(ax_evh, ev, C_EV, xlabels=False)
    draw_hos(ax_dih, di, C_DIESEL, xlabels=True)
    ax_evh.legend(loc="upper left", fontsize=7.8, frameon=True, ncol=2,
                  handlelength=1.8, borderaxespad=0.2, facecolor="white",
                  edgecolor="none", framealpha=0.92)

    # ── the gap, called out between the two vehicles ────────────────────────
    ax_gap.set_xlim(ax_ev.get_xlim()); ax_gap.set_ylim(0, 1)
    ax_gap.annotate("", xy=(ev_end, 0.5), xytext=(di_end, 0.5),
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.4))
    ax_gap.text((ev_end + di_end) / 2, 0.80, f"+{delta:.2f} h",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#d62728")

    # ── time budget ─────────────────────────────────────────────────────────
    order = [("drive", "Driving", C_DRIVE, None),
             ("charge", "Charging", C_CHARGE, None),
             ("queue", "Queueing", C_QUEUE, None),
             ("man", "Manoeuvring (pull off / on)", C_MAN, None),
             ("service", "Service", C_SERV, None),
             ("brk", "Break (beyond any charge)", C_BRK_F, "//////"),
             ("rest", "Daily rest", C_RST_F, "\\\\\\\\"),
             ("wait", "Idle wait", "#f7f7f7", "..")]
    edge = {"drive": "#9a9a9a", "charge": C_CHARGE, "queue": C_QUEUE, "man": C_MAN,
            "service": "#a85a17", "brk": C_BRK_E, "rest": C_RST_E, "wait": "#999999"}
    # drop components that are zero for BOTH vehicles, so the legend only ever
    # advertises something the reader can actually find in the bars
    order = [o for o in order
             if ev["budget"][o[0]] > 1e-6 or di["budget"][o[0]] > 1e-6]
    maskable = min(ev["budget"]["charge"], di["budget"]["brk"])
    for row, (rec, lab, col) in enumerate([(di, "Diesel", C_DIESEL), (ev, "EV", C_EV)]):
        x = 0.0
        for key, _, fc, hatch in order:
            w = rec["budget"][key]
            if w <= 1e-6:
                continue
            ax_bud.add_patch(Rectangle((x, row - 0.28), w, 0.56, facecolor=fc,
                                       edgecolor=edge[key], lw=0.5, hatch=hatch))
            if w > TMAX * 0.035:
                ax_bud.text(x + w / 2, row, f"{w:.1f}",
                            ha="center", va="center", fontsize=7.6,
                            color="white" if key in ("man",) else "#333333")
            # Split the EV's charging block at the point where the mandatory
            # break runs out: everything to the left would have been spent
            # standing still anyway, everything to the right is pure EV cost.
            if key == "charge" and rec is ev and maskable < w:
                ax_bud.plot([x + maskable] * 2, [row - 0.28, row + 0.28],
                            color="white", lw=1.2, ls=(0, (2, 1.5)), zorder=5)
                ax_bud.annotate(
                    f"{maskable:.2f} h hides inside a break the driver owed anyway\n"
                    f"{w - maskable:.2f} h is pure EV time",
                    xy=(x + maskable, row + 0.28), xytext=(TMAX * 0.02, row + 0.42),
                    ha="left", va="bottom", fontsize=7.6, color="#d62728",
                    linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.7,
                                    shrinkA=1, shrinkB=1))
            x += w
        ax_bud.text(x + TMAX * 0.008, row, f"{x:.2f} h", ha="left", va="center",
                    fontsize=8.6, fontweight="bold", color=col)
    ax_bud.set_ylim(-0.62, 2.05)
    ax_bud.set_yticks([0, 1])
    ax_bud.set_yticklabels(["Diesel", "EV"], fontsize=9)
    ax_bud.tick_params(axis="y", length=0, pad=2)
    ax_bud.set_xlim(-TMAX * 0.012, TMAX * 1.045)
    ticks = np.arange(0, TMAX + 0.01, 6.0)
    ax_bud.set_xticks(ticks)
    ax_bud.set_xticklabels([f"{t:.0f}" for t in ticks], fontsize=7.5, color="#555555")
    ax_bud.set_xlabel("hours of the total duration", fontsize=8.5, color="#555555", labelpad=1)
    ax_bud.grid(axis="x", color="#eeeeee", lw=0.7); ax_bud.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax_bud.spines[s].set_visible(False)
    ax_bud.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=fc, edgecolor=edge[k],
                                     lw=0.5, hatch=h, label=lb)
                           for k, lb, fc, h in order],
                  loc="upper center", ncol=len(order), frameon=True, framealpha=1.0,
                  edgecolor="#cccccc", fontsize=7.8, bbox_to_anchor=(0.5, -0.50),
                  handlelength=1.5, handletextpad=0.4, columnspacing=1.0)
    fig.text(0.012, ax_bud.get_position().y0 + ax_bud.get_position().height / 2,
             "Time\nbudget", fontsize=10.5, fontweight="bold", color=INK,
             ha="left", va="center")

    # ── caption ─────────────────────────────────────────────────────────────
    b_ev, b_di = ev["budget"], di["budget"]
    fig.text(0.5, 0.010,
             f"The EV charges {b_ev['charge']:.2f} h, but HoS only mandates "
             f"{b_di['brk']:.2f} h of break on this route -- so at most "
             f"{maskable:.2f} h of charging can hide behind a break.\n"
             f"On top of the {b_ev['charge']-maskable:.2f} h of exposed charging, the "
             f"{len(ev['used'])-len(di['used'])} extra pull-offs cost "
             f"+{b_ev['man']-b_di['man']:.2f} h and the chargers "
             f"{b_ev['queue']:.2f} h of queueing.",
             fontsize=7.8, color="#888888", ha="center", va="bottom",
             linespacing=1.5)

    _paths.ensure_dirs()
    sfx = "" if method == "greedy" else f"_{method}"
    stem = out_stem or _paths.figures(f"diesel_vs_ev_{instance}{sfx}")
    fig.savefig(stem + ".png", dpi=200)
    fig.savefig(stem + ".pdf")
    plt.close(fig)

    print(f"saved {stem}.png / .pdf")
    print(f"  EV {ev_end:.2f} h  diesel {di_end:.2f} h  delta {delta:.2f} h")
    print("  budget      EV      diesel")
    for key, lab, _, _ in order:
        print(f"  {lab:<28s} {b_ev[key]:7.2f} {b_di[key]:7.2f}")
    print(f"  stops used  EV {len(ev['used'])}   diesel {len(di['used'])}")
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instance", nargs="?", default=None,
                    help=f"instance stem (default: {DEFAULT_INSTANCE}, or "
                         f"{DEFAULT_ORACLE_INSTANCE} with --method oracle); "
                         "a matching <instance>__diesel run must exist")
    ap.add_argument("--method", choices=("greedy", "oracle"), default="greedy",
                    help="which pair of runs to draw (default: greedy)")
    ap.add_argument("--out", default=None, help="output path stem (no extension)")
    args = ap.parse_args()
    inst = args.instance or (DEFAULT_ORACLE_INSTANCE if args.method == "oracle"
                             else DEFAULT_INSTANCE)
    build(inst, args.out, method=args.method)


if __name__ == "__main__":
    main()
