# -*- coding: utf-8 -*-
"""
Conceptual solution figure for one executed RO run.

Everything (positions, times, block widths, SOC, HoS counters) is computed
from the instance JSON + solution JSON -- nothing hand-placed.

Run:  RshortCfewTmedium_19_RO_20260716_092310_060
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

import os
from src import paths as _paths
BASE = str(_paths.ROOT)
SOL  = "RshortCfewTmedium_19_RO_20260716_092310_060"
INST = "RshortCfewTmedium_19"
OUT  = _paths.figure_out("solution_concept_" + SOL + ".png")

# ---------------------------------------------------------------- load ------
sol  = json.load(open(_paths.solution_path(f"{SOL}.json")))
inst = json.load(open(_paths.instances(f"{INST}.json")))
fd     = inst["instance"]
D_real = inst["D_real"]
E_real = inst["E_real"]
traj   = sol["sim_trajectory"]
acts   = sol["actions"]

N     = fd["N"]
K     = set(fd["K"]); C = set(fd["C"]); L = set(fd["L"])
km    = fd["km"]
cumkm = [0.0]
for i in range(N):
    cumkm.append(cumkm[-1] + km[str(i)])
TOTKM = cumkm[-1]

Ebar = {int(k): v for k, v in fd["Ebar"].items()}
Tbar = {int(k): v for k, v in fd["Tbar"].items()}
rs   = sorted(Ebar); Es = [Ebar[r] for r in rs]; Ts = [Tbar[r] for r in rs]

def e2t(e):
    e = max(Es[0], min(Es[-1], e))
    for k in range(len(Es) - 1):
        if Es[k] <= e <= Es[k + 1]:
            span = Es[k + 1] - Es[k]
            return Ts[k] + (e - Es[k]) / span * (Ts[k + 1] - Ts[k]) if span else Ts[k]
    return Ts[-1]

def t2e(t):
    t = max(Ts[0], min(Ts[-1], t))
    for k in range(len(Ts) - 1):
        if Ts[k] <= t <= Ts[k + 1]:
            span = Ts[k + 1] - Ts[k]
            return Es[k] + (t - Ts[k]) / span * (Es[k + 1] - Es[k]) if span else Es[k]
    return Es[-1]

# ------------------------------------------------- reconstruct schedule -----
# Per stop: arrival, block sequence, departure.  Blocks carry (kind, t0, t1).
# kinds: mstop, queue, charge, mseq, mlay, service, break, rest, idle
events = []          # list of dicts per visited stop
soc_pts = []         # (t, e) polyline points for the SOC panel
drive_blocks = []    # (t0, t1) driving legs
charge_bands = []    # (t0, t1) for SOC shading
cd_resets = []       # times where cd resets (end of qualifying break / rest)
rest_resets = []     # times where sd/sw reset (end of rest)

t = traj[0]["t_arr"]
soc_pts.append((t, traj[0]["e_arr"]))

for i in range(N):
    s, s1 = traj[i], traj[i + 1]
    a  = acts[i]
    td = s1["t_arr"] - D_real[i]
    dwell = td - s["t_arr"]
    y, brk, rst = a.get("y"), a.get("break_type"), a.get("rest_type")
    e_arr = s["e_arr"]
    e_dep = s1["e_arr"] + E_real[i]
    chg   = e_dep - e_arr
    blocks = []
    tt = s["t_arr"]
    if dwell > 1e-3:
        if i in K:
            ms = fd["M_stop"][str(i)]
            qv = fd["Q"][str(i)] if y else 0.0
            tauc = e2t(e_dep) - e2t(e_arr) if chg > 1e-6 else 0.0
            sigma = 1 if (y and rst) else 0
            mseq = fd["M_seq"][str(i)] * sigma
            taur = fd["Tr2"] if rst == "r2" else (fd["Tr1"] if rst == "r1" else 0.0)
            extra = dwell - (ms + qv + tauc + mseq + taur)   # extra break beyond charge
            blocks.append(("mstop", tt, tt + ms)); tt += ms
            if qv > 1e-6:
                blocks.append(("queue", tt, tt + qv)); tt += qv
            if tauc > 1e-6:
                blocks.append(("charge", tt, tt + tauc))
                charge_bands.append((tt, tt + tauc, e_arr, e_dep))
                # SOC rise along PWL, inserting knot crossings
                t0 = e2t(e_arr)
                soc_pts.append((tt, e_arr))
                for Ek, Tk in zip(Es, Ts):
                    if e_arr < Ek < e_dep:
                        soc_pts.append((tt + (Tk - t0), Ek))
                soc_pts.append((tt + tauc, e_dep))
                tt += tauc
            if mseq > 1e-6:
                blocks.append(("mseq", tt, tt + mseq)); tt += mseq
            if brk and extra > 1e-6:
                blocks.append(("break", tt, tt + extra)); tt += extra
            if taur > 1e-6:
                blocks.append(("rest", tt, tt + taur)); tt += taur
        elif i in C:
            sv = fd["S"][str(i)]
            blocks.append(("service", tt, tt + sv)); tt += sv
            rem = dwell - sv
            if rst and rem > 1e-6:
                blocks.append(("rest", tt, tt + rem)); tt += rem
            elif brk and rem > 1e-6:
                blocks.append(("break", tt, tt + rem)); tt += rem
        elif i in L:
            ml = fd["M_lay"][str(i)] if (brk or rst) else 0.0
            if ml > 1e-6:
                blocks.append(("mlay", tt, tt + ml)); tt += ml
            rem = td - tt
            kind = "rest" if rst else ("break" if brk else "idle")
            if rem > 1e-6:
                blocks.append((kind, tt, tt + rem)); tt += rem
        events.append(dict(stop=i, t_arr=s["t_arr"], td=td, blocks=blocks,
                           y=y, brk=brk, rst=rst,
                           tauc=(e2t(e_dep) - e2t(e_arr)) if chg > 1e-6 else 0.0))
    # qualifying-break / rest reset times (accumulators reset when activity ends)
    if brk in ("b45", "b30") or rst:
        cd_resets.append(td)
    if rst:
        rest_resets.append(td)
    # leg
    drive_blocks.append((td, s1["t_arr"]))
    soc_pts.append((td, e_dep))
    soc_pts.append((s1["t_arr"], s1["e_arr"]))

T0   = traj[0]["t_arr"]
TEND = traj[-1]["t_arr"]

# merge consecutive legs (drive-through stops have ~0 dwell from rounding)
_md = []
for (t0, t1) in drive_blocks:
    if _md and abs(t0 - _md[-1][1]) < 2.5e-3:
        _md[-1][1] = t1
    else:
        _md.append([t0, t1])
drive_blocks = [(a, b) for a, b in _md]

# ------------------------------------------------- HoS counters cd, sd ------
# piecewise: rate 1 while driving, flat during dwell, drop to 0 at reset times
def counter_path(reset_times):
    pts = [(T0, 0.0)]
    val = 0.0
    segs = []
    for (t0, t1) in drive_blocks:
        segs.append(("drive", t0, t1))
    for rt in reset_times:
        segs.append(("reset", rt, rt))
    segs.sort(key=lambda x: (x[1], 0 if x[0] == "reset" else 1))
    cur_t = T0
    for kind, t0, t1 in segs:
        if t0 > cur_t:
            pts.append((t0, val))          # flat gap
        if kind == "reset":
            pts.append((t0, val)); val = 0.0; pts.append((t0, 0.0))
            cur_t = t0
        else:
            val += (t1 - t0); pts.append((t1, val)); cur_t = t1
    pts.append((TEND, val))
    return np.array(pts)

cd_path = counter_path(cd_resets)
sd_path = counter_path(rest_resets)

# ------------------------------------------------------------- styling ------
INK, MUT = "#222222", "#666666"
c_drive  = "#d9d9d9"
c_mstop  = "#141414"
c_queue  = "#e02b2b"
c_charge = "#1e7a3c"
c_work   = "#f2c7ee"; c_work_e = "#c583be"
c_serv   = "#e8822d"
c_brk_f, c_brk_e = "#eef6fd", "#2b7bba"
c_rst_f, c_rst_e = "#e4edf7", "#1f4e79"
c_ring   = "#cc2222"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.edgecolor": "#aaaaaa", "axes.linewidth": 0.8,
})

fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
gs = fig.add_gridspec(6, 1, height_ratios=[0.95, 0.42, 0.52, 1.05, 1.05, 0.52],
                      left=0.065, right=0.968, top=0.855, bottom=0.150, hspace=0.42)
ax_route = fig.add_subplot(gs[0])
ax_call  = fig.add_subplot(gs[1]); ax_call.axis("off")
ax_truck = fig.add_subplot(gs[2])
ax_soc   = fig.add_subplot(gs[3], sharex=ax_truck)
ax_hos   = fig.add_subplot(gs[4], sharex=ax_truck)
ax_drv   = fig.add_subplot(gs[5], sharex=ax_truck)
ax_call.set_xlim(T0 - 0.35, TEND + 0.35)

# ------------------------------------------------------------ title ---------
arr_clock = f"{int(TEND % 24):02d}:{int(round((TEND % 1) * 60)):02d}"
fig.text(0.065, 0.965, "What a solution looks like",
         fontsize=16, fontweight="bold", color=INK, va="top")
fig.text(0.065, 0.922,
         f"Instance {INST}  ({TOTKM:.0f} km, {len(K)} charging stations, "
         f"{len(L)} rest areas, {len(C)} customer)",
         fontsize=10, color=MUT, va="top")
fig.text(0.065, 0.895,
         f"RO plan, executed as-is  •  depart 08:00, arrive {arr_clock} next day "
         f"({sol['duration_h']:.1f} h)  •  feasible under realised travel times, "
         f"customer window met",
         fontsize=10, color=MUT, va="top")

# ------------------------------------------------------------ route ---------
chosen = {e["stop"] for e in events}
ax_route.set_xlim(-18, TOTKM + 18)
ax_route.set_ylim(-1.75, 1.85)
ax_route.axis("off")
ax_route.plot([0, TOTKM], [0, 0], color="#444444", lw=1.4, zorder=1)
for i in range(N + 1):
    x = cumkm[i]
    if i in L:
        ax_route.plot([x, x], [-0.16, 0.16], color="#999999", lw=1.0, zorder=2)
    if i in K:
        sel = i in chosen
        ax_route.scatter([x], [0.34], marker="^",
                         s=150 if sel else 95,
                         c=c_charge, edgecolors="#0d3c1d",
                         linewidths=0.7, alpha=1.0 if sel else 0.50, zorder=3)
    if i in C:
        sel = i in chosen
        ax_route.scatter([x], [0.34], marker="D",
                         s=120 if sel else 80, c=c_serv,
                         edgecolors="#7a3d0d", linewidths=0.7,
                         alpha=1.0 if sel else 0.55, zorder=3)
prev_x = -1e9
row2 = False
for i in sorted(chosen):
    x = cumkm[i]
    ax_route.scatter([x], [0.34], marker="o", s=440, facecolors="none",
                     edgecolors=c_ring, linewidths=1.7, zorder=4)
    row2 = (x - prev_x < 45) and not row2      # stagger labels of close stops
    ax_route.text(x, -1.05 if row2 else -0.52, f"{cumkm[i]:.0f} km",
                  ha="center", va="top",
                  fontsize=8.5, color=c_ring, fontweight="bold")
    prev_x = x
# O / D
for x, lab in [(0.0, "O"), (TOTKM, "D")]:
    ax_route.scatter([x], [0], marker="s", s=190, c="#111111", zorder=5)
    ax_route.text(x, 0.62, lab, ha="center", va="bottom",
                  fontsize=11, fontweight="bold", color=INK)
    ax_route.text(x, -0.52, f"{x:.0f} km", ha="center", va="top",
                  fontsize=8.5, color=INK)
ax_route.text(cumkm[sorted(C)[0]] + 14, 0.92, "C1", ha="left", va="bottom",
              fontsize=10, fontweight="bold", color="#7a3d0d")
# legend (top right, like the template)
leg = [Line2D([], [], color="#999999", lw=1.2, marker="|", markersize=11,
              linestyle="none", label="Rest area"),
       Line2D([], [], marker="^", color=c_charge, markeredgecolor="#0d3c1d",
              markersize=9, linestyle="none", label="Charging station"),
       Line2D([], [], marker="D", color=c_serv, markeredgecolor="#7a3d0d",
              markersize=8, linestyle="none", label="Customer"),
       Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=c_ring,
              markersize=11, markeredgewidth=1.6, linestyle="none",
              label="Stop used by the solution")]
ax_route.legend(handles=leg, loc="upper right", ncol=4, frameon=True,
                framealpha=1.0, edgecolor="#33415c", fontsize=8.8,
                bbox_to_anchor=(1.0, 1.9), borderpad=0.55, handletextpad=0.35)
ax_route.set_ylabel("")
fig.text(0.012, ax_route.get_position().y0 + ax_route.get_position().height / 2,
         "Route", fontsize=11, fontweight="bold", color=INK,
         ha="left", va="center")

# ---------------------------------------------------- callout labels --------
# stop: (text, text-x)   -- one top row, arrows point at the dwell midpoint
callouts = {
    10: ("Charge & 15 min break\n(1st split, $g_i=\\tau_i^c$)", 8.15),
    11: ("Service + 30 min break\n(2nd split)", 12.85),
    20: ("Charge, then 9 h rest\n(sequential: $\\sigma_i=1$, $g_i=0$)", 20.9),
    27: ("Charge & 45 min break\n($g_i=\\tau_i^c$)", 27.1),
    35: ("Charge & 45 min break", 31.0),
}
for e in events:
    i = e["stop"]
    txt, tx = callouts[i]
    xm = 0.5 * (e["t_arr"] + e["td"])
    ax_call.annotate(txt, xy=(xm, -0.06), xytext=(tx, 1.00),
                     ha="center", va="top", fontsize=8.6, color="#444444",
                     arrowprops=dict(arrowstyle="-", color="#b8b8b8", lw=0.8,
                                     shrinkA=1, shrinkB=0),
                     annotation_clip=False)
ax_call.set_ylim(-0.06, 1.0)

# ------------------------------------------------------------ truck ---------
def block_colors(kind):
    if kind == "mstop" or kind == "mseq":
        return dict(fc=c_mstop, ec=c_mstop)
    if kind == "queue":
        return dict(fc=c_queue, ec=c_queue)
    if kind == "charge":
        return dict(fc=c_charge, ec=c_charge)
    return dict(fc="white", ec="#aaaaaa")

ax_truck.set_ylim(0, 1)
ax_truck.set_yticks([])
for (t0, t1) in drive_blocks:
    if t1 - t0 <= 1e-9:
        continue
    ax_truck.add_patch(Rectangle((t0, 0.07), t1 - t0, 0.86,
                                 facecolor=c_drive, edgecolor="#9a9a9a", lw=0.5))
    if t1 - t0 > 1.55:
        ax_truck.text(0.5 * (t0 + t1), 0.5, "Driving", ha="center", va="center",
                      fontsize=8.4, color="#333333")
for e in events:
    white_run = None   # merge consecutive idle blocks into one patch
    for kind, t0, t1 in e["blocks"] + [("_end", None, None)]:
        if kind in ("mstop", "queue", "charge", "mseq"):
            if white_run:
                ax_truck.add_patch(Rectangle((white_run[0], 0.07),
                                             white_run[1] - white_run[0], 0.86,
                                             facecolor="white", edgecolor="#9a9a9a", lw=0.5))
                white_run = None
            cc = block_colors(kind)
            ax_truck.add_patch(Rectangle((t0, 0.07), t1 - t0, 0.86,
                                         facecolor=cc["fc"], edgecolor=cc["ec"], lw=0.4))
        elif kind == "_end":
            if white_run:
                ax_truck.add_patch(Rectangle((white_run[0], 0.07),
                                             white_run[1] - white_run[0], 0.86,
                                             facecolor="white", edgecolor="#9a9a9a", lw=0.5))
        else:  # truck idle during service / break / rest
            white_run = [white_run[0], t1] if white_run else [t0, t1]
ax_truck.set_xlim(T0 - 0.35, TEND + 0.35)
plt.setp(ax_truck.get_xticklabels(), visible=False)
ax_truck.tick_params(axis="x", length=0)
fig.text(0.012, ax_truck.get_position().y0 + ax_truck.get_position().height / 2,
         "Truck", fontsize=11, fontweight="bold", color=INK, ha="left", va="center")

# micro annotations on the truck row (template style): fan-out at the first
# charge stop (10) plus M_seq at the charge-then-rest stop (20)
e10  = next(e for e in events if e["stop"] == 10)
bl10 = {k: (t0, t1) for k, t0, t1 in e10["blocks"]}
e20  = next(e for e in events if e["stop"] == 20)
bl20 = {k: (t0, t1) for k, t0, t1 in e20["blocks"]}
micro = [(bl10["mstop"],  "$M_i^{stop}$", 9.70, 1.48),
         (bl10["queue"],  "$Q_i$",        10.60, 1.85),
         (bl10["charge"], "$\\tau_i^c$",  11.30, 1.42),
         (bl20["mseq"],   "$M_i^{seq}$",  16.60, 1.75)]
for (t0, t1), lab, tx, ty in micro:
    ax_truck.annotate(lab, xy=(0.5 * (t0 + t1), 0.97), xytext=(tx, ty),
                      fontsize=9, color=INK, ha="center",
                      arrowprops=dict(arrowstyle="-", color="#555555", lw=0.7,
                                      shrinkA=2, shrinkB=1),
                      annotation_clip=False)
t0, t1 = bl20["rest"]
ax_truck.text(0.5 * (t0 + t1), 0.5, "truck parked  ($\\tau_i^r$)",
              fontsize=8.4, color="#777777", ha="center", va="center",
              style="italic")

# ------------------------------------------------------------- SOC ----------
sp = np.array(soc_pts)
for (t0, t1, e0, e1) in charge_bands:
    ax_soc.axvspan(t0, t1, color=c_charge, alpha=0.16, lw=0)
ax_soc.plot(sp[:, 0], sp[:, 1], color=c_charge, lw=1.8, zorder=3)
ax_soc.axhline(fd["Ecap"], color="#888888", ls=(0, (5, 4)), lw=1.1)
ax_soc.axhline(fd["Emin"], color="#d62728", ls=(0, (5, 4)), lw=1.1)
ax_soc.text(TEND + 0.25, fd["Ecap"], "$E^{cap}$", fontsize=8.5, color="#666666",
            va="center", ha="left")
ax_soc.text(TEND + 0.25, fd["Emin"], "$E^{min}$", fontsize=8.5, color="#d62728",
            va="center", ha="left")
ax_soc.set_ylim(0, 560)
ax_soc.set_yticks([100, 300, 500])
ax_soc.tick_params(labelsize=8)
plt.setp(ax_soc.get_xticklabels(), visible=False)
ax_soc.tick_params(axis="x", length=0)
ax_soc.grid(axis="y", color="#eeeeee", lw=0.7)
ax_soc.set_axisbelow(True)
fig.text(0.012, ax_soc.get_position().y0 + ax_soc.get_position().height / 2,
         "SOC\n[kWh]", fontsize=11, fontweight="bold", color=INK, ha="left", va="center")

# ------------------------------------------------------------- HoS ----------
ax_hos.plot(cd_path[:, 0], cd_path[:, 1], color="#2b6cb0", lw=1.7,
            label="consecutive driving $t^{drv}$", zorder=3)
ax_hos.plot(sd_path[:, 0], sd_path[:, 1], color="#8a63b8", lw=1.4, ls=(0, (4, 2)),
            label="shift driving $t^{sd}$", zorder=3)
ax_hos.axhline(fd["Tdrv_cons"], color="#2b6cb0", ls=(0, (2, 3)), lw=1.0, alpha=0.85)
ax_hos.axhline(fd["Tdrv_sh1"], color="#8a63b8", ls=(0, (2, 3)), lw=1.0, alpha=0.85)
ax_hos.text(TEND + 0.25, fd["Tdrv_cons"], "4.5 h", fontsize=8, color="#2b6cb0", va="center")
ax_hos.text(TEND + 0.25, fd["Tdrv_sh1"], "9 h", fontsize=8, color="#8a63b8", va="center")
ax_hos.set_ylim(0, 10.3)
ax_hos.set_yticks([0, 4.5, 9])
ax_hos.tick_params(labelsize=8)
ax_hos.grid(axis="y", color="#eeeeee", lw=0.7)
ax_hos.set_axisbelow(True)
ax_hos.legend(loc="upper left", fontsize=8, frameon=True, ncol=2,
              handlelength=1.8, borderaxespad=0.2, facecolor="white",
              edgecolor="none", framealpha=0.95,
              bbox_to_anchor=(0.005, 0.99))
fig.text(0.012, ax_hos.get_position().y0 + ax_hos.get_position().height / 2,
         "HoS\n[h]", fontsize=11, fontweight="bold", color=INK, ha="left", va="center")
# clock ticks on HoS bottom
tick_h = np.arange(np.ceil(T0), TEND + 0.01, 2.0)
ax_hos.set_xticks(tick_h)
ax_hos.set_xticklabels([f"{int(h % 24):02d}:00" for h in tick_h], fontsize=7.5,
                       color="#555555")
ax_hos.tick_params(axis="x", length=2)

# midnight marker across time panels
for ax in (ax_truck, ax_soc, ax_hos, ax_drv):
    ax.axvline(24.0, color="#bbbbbb", ls=(0, (1, 2)), lw=0.9, zorder=0)
ax_soc.text(24.15, 430, "midnight", fontsize=7.3, color="#999999",
            ha="left", va="center")

# ------------------------------------------------------------ driver --------
ax_drv.set_ylim(0, 1)
ax_drv.set_yticks([])
plt.setp(ax_drv.get_xticklabels(), visible=False)
ax_drv.tick_params(axis="x", length=0)
rot_labels = []   # (x, text)
for (t0, t1) in drive_blocks:
    if t1 - t0 <= 1e-9:
        continue
    ax_drv.add_patch(Rectangle((t0, 0.07), t1 - t0, 0.86,
                               facecolor=c_drive, edgecolor="#9a9a9a", lw=0.5))
    if t1 - t0 > 1.55:
        ax_drv.text(0.5 * (t0 + t1), 0.5, "Driving", ha="center", va="center",
                    fontsize=8.4, color="#333333")

def drv_block(kind, t0, t1, stop_e):
    """map a stop block to the driver's activity"""
    sigma = 1 if (stop_e["y"] and stop_e["rst"]) else 0
    if kind in ("mstop", "queue", "mseq", "mlay"):
        return "work"
    if kind == "charge":
        return "work" if sigma else "break"     # concurrent break credit g
    if kind == "service":
        return "service"
    return kind                                  # break / rest / idle

lab_txt = {10: [("work", "Working time"), ("break", "1st split break (15$'$)")],
           11: [("service", "Service"), ("break", "2nd split break (30$'$)")],
           20: [("work", "Working time"), ("rest", "9 h rest (reduced)")],
           27: [("work", "Working time"), ("break", "45$'$ break")],
           35: [("work", "Working time"), ("break", "45$'$ break")]}

for e in events:
    # merge consecutive same-kind driver blocks
    merged = []
    for kind, t0, t1 in e["blocks"]:
        dk = drv_block(kind, t0, t1, e)
        if merged and merged[-1][0] == dk:
            merged[-1][2] = t1
        else:
            merged.append([dk, t0, t1])
    for dk, t0, t1 in merged:
        if dk == "work":
            ax_drv.add_patch(Rectangle((t0, 0.07), t1 - t0, 0.86,
                                       facecolor=c_work, edgecolor=c_work_e, lw=0.5))
        elif dk == "service":
            ax_drv.add_patch(Rectangle((t0, 0.07), t1 - t0, 0.86,
                                       facecolor=c_serv, edgecolor="#a85a17", lw=0.5))
        elif dk == "break":
            ax_drv.add_patch(Rectangle((t0, 0.07), t1 - t0, 0.86,
                                       facecolor=c_brk_f, edgecolor=c_brk_e,
                                       lw=0.6, hatch="//////"))
        elif dk == "rest":
            ax_drv.add_patch(Rectangle((t0, 0.07), t1 - t0, 0.86,
                                       facecolor=c_rst_f, edgecolor=c_rst_e,
                                       lw=0.6, hatch="\\\\\\\\"))
    # rotated labels below
    kinds_here = [m[0] for m in merged]
    for want, txt in lab_txt[e["stop"]]:
        for dk, t0, t1 in merged:
            if dk == want:
                rot_labels.append((0.5 * (t0 + t1), txt))
                break

for x, txt in rot_labels:
    ax_drv.annotate(txt, xy=(x, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -7), textcoords="offset points",
                    rotation=38, ha="right", va="top", rotation_mode="anchor",
                    fontsize=8.6, color="#555555", annotation_clip=False)
fig.text(0.012, ax_drv.get_position().y0 + ax_drv.get_position().height / 2,
         "Driver", fontsize=11, fontweight="bold", color=INK, ha="left", va="center")

# ------------------------------------------------- bottom-right caption -----
fig.text(0.985, 0.012,
         "run " + SOL + "  •  break rules: 45$'$ after ≤4.5 h driving, splittable 15$'$+30$'$; "
         "daily rest 11 h (9 h reduced)  •  charge counts as break when concurrent ($g_i$), "
         "as work when followed by rest ($\\sigma_i$=1)",
         fontsize=7.6, color="#888888", ha="right", va="bottom")

fig.savefig(OUT, dpi=200)
print("saved", OUT)
print("events:")
for e in events:
    print(e["stop"], [(k, round(t0, 3), round(t1, 3)) for k, t0, t1 in e["blocks"]])
print("cd max %.2f  sd max %.2f" % (cd_path[:, 1].max(), sd_path[:, 1].max()))
