

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    Binary, NonNegativeReals, minimize, value, SolverFactory
)


# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────
C_DRIVE   = "#4C9BE8"   # blue  – travel between locations
C_CHARGE  = "#F5A623"   # amber – charging at a station
C_BREAK   = "#7ED321"   # green – mandatory HOS break (at customer)
C_BAC     = "#BD10E0"   # purple – break-and-charge (break taken while charging)
C_IDLE    = "#D0D0D0"   # light grey – at customer location (no break / service)
C_ENERGY  = "#E85454"   # red   – SOC curve
C_LIMIT   = "#E85454"   # red dashed – HOS limit line
C_HOS     = "#4C9BE8"   # blue  – accumulated drive time


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 – build a flat, ordered list of timeline events
# ──────────────────────────────────────────────────────────────────────────────

def build_timeline(model, successor, data):
    """
    Walk the solved route chronologically and return a list of *events*.

    Each event is a dict with the keys:

        t_start  : float  – clock time at start of event [h]
        t_end    : float  – clock time at end   of event [h]
        location : str    – location label (customer or charger id)
        kind     : str    – one of 'travel' | 'at_customer' | 'break' |
                            'travel_to_charger' | 'charging' | 'bac' |
                            'travel_from_charger'
        soc_start: float  – battery SOC [kWh] at t_start
        soc_end  : float  – battery SOC [kWh] at t_end
        hos_start: float  – accumulated drive since last break at t_start [h]
        hos_end  : float  – accumulated drive since last break at t_end   [h]
        label    : str    – short description for annotation

    The timeline drives all three plots so that they share a common time axis.
    """
    N_sorted = sorted(data["N"])
    h        = data["h"]           # kWh/km

    events   = []
    clock    = 0.0                 # absolute clock [h]
    soc      = data["y0"]          # current SOC [kWh]
    hos      = value(model.W_break) # remaining drive time before break [h]
    #   hos = W_break means "fully rested, can drive W_break more hours"
    #   We store it as "accumulated drive since last break" = W_break - t_b
    #   t_b[i] = remaining drive time budget on arrival at i
    #   => accumulated = W_break - t_b[i]

    W_break  = value(model.W_break)
    B_break  = value(model.B_break)

    def accum_drive(i):
        """Drive accumulated since last break on arrival at customer i."""
        return W_break - value(model.t_b[i])

    def accum_drive_f(i, f):
        """Drive accumulated since last break on arrival at charger f."""
        return W_break - value(model.t_b_prime[i])

    for idx, i in enumerate(N_sorted):

        # ── Arrival at customer i ──────────────────────────────────────────
        soc_here = value(model.y[i])
        hos_here = accum_drive(i)

        # Determine what happens *at* this customer location
        w_i = value(model.w[i]) > 0.5

        print(f"Customer {i}: arrival at t={clock:.2f} h, SOC={soc_here:.1f} kWh, "
              f"HOS since last break={hos_here:.2f} h, "
              f"{'break taken' if w_i else 'no break'}")

        if w_i:
            # Mandatory break taken at this customer
            events.append(dict(
                t_start=clock, t_end=clock + B_break,
                location=str(i), kind="break",
                soc_start=soc_here, soc_end=soc_here,
                hos_start=hos_here, hos_end=0.0,
                label=f"Break {B_break*60:.0f} min",
            ))
            clock += B_break
        else:
            # Brief service stop (zero duration in model, show as thin bar)
            events.append(dict(
                t_start=clock, t_end=clock,
                location=str(i), kind="at_customer",
                soc_start=soc_here, soc_end=soc_here,
                hos_start=hos_here, hos_end=hos_here,
                label="",
            ))

        if i not in successor:
            break   # last customer: done

        ip1 = successor[i]

        # ── Leg from customer i to customer i+1 ───────────────────────────
        # Check if a detour via a charger is taken on this leg
        detour_taken = False
        for f in data["F"]:
            if (i, f) not in [tuple(z) for z in data["Z"]]:
                continue
            if value(model.z[i, f]) < 0.5:
                continue

            detour_taken = True
            T_if  = value(model.T_travel[i, f])
            T_fi1 = value(model.T_travel[f, ip1])
            D_if  = value(model.D_dist[i, f])
            D_fi1 = value(model.D_dist[f, ip1])
            y_arr = value(model.y_prime[i, f])
            y_chg = value(model.y_dbl_prime[i, f])
            t_chg = value(model.t_dbl_prime[i, f])
            w_pr  = value(model.w_prime[i]) > 0.5
            hos_at_f = accum_drive_f(i, f)

            # Travel i → f
            soc_drive_to_f = soc_here - D_if * h
            hos_after_to_f = hos_here + T_if   # more driving = more accumulated
            print(f"Detour via charger {f}: i → f takes {T_if:.2f} h, SOC on arrival = {soc_drive_to_f:.1f} kWh, ")
            events.append(dict(
                t_start=clock, t_end=clock + T_if,
                location=str(i),          # leaving from i, arriving at f
                kind="travel_to_charger",
                soc_start=soc_here, soc_end=soc_drive_to_f,
                hos_start=hos_here if not w_i else 0, hos_end=hos_at_f,
                label=f"→ {f}  ({T_if*60:.0f} min)",
            ))
            clock   += T_if
            soc_now  = y_arr
            hos_now  = hos_at_f

            # Charging at f  (may also count as break-and-charge)
            soc_after_chg = y_arr + y_chg
            kind_chg = "bac" if w_pr else "charging"
            hos_after_chg = 0.0 if w_pr else hos_now   # break resets counter
            label_chg = (
                f"Charge {y_chg:.0f} kWh  ({t_chg*60:.0f} min)"
                + (" + Break" if w_pr else "")
            )
            print(f"Charging at {f} for {t_chg:.2f} h, SOC after = {soc_after_chg:.1f} kWh, "
                  f"HOS since last break = {hos_after_chg:.2f} h")
            events.append(dict(
                t_start=clock, t_end=clock + t_chg,
                location=str(f), kind=kind_chg,
                soc_start=soc_now, soc_end=soc_after_chg,
                hos_start=hos_now, hos_end=hos_after_chg,
                label=label_chg,
            ))
            clock   += t_chg
            soc_now  = soc_after_chg
            hos_now  = hos_after_chg

            # Travel f → i+1
            soc_after_drive = soc_now - D_fi1 * h
            hos_at_ip1 = accum_drive(ip1)
            print(f"Travel from {f} to {ip1}: {T_fi1:.2f} h, SOC after = {soc_after_drive:.1f} kWh, ")

            events.append(dict(
                t_start=clock, t_end=clock + T_fi1,
                location=str(f),          # leaving charger f
                kind="travel_from_charger",
                soc_start=soc_now, soc_end=soc_after_drive,
                hos_start=hos_now, hos_end=hos_at_ip1,
                label=f"→ {ip1}  ({T_fi1*60:.0f} min)",
            ))
            clock += T_fi1
            break   # only one charger detour per leg

        if not detour_taken:
            # Direct arc i → i+1
            T_dir = value(model.T_travel[i, ip1])
            D_dir = value(model.D_dist[i, ip1])
            soc_after = soc_here - D_dir * h
            hos_at_ip1 = accum_drive(ip1)

            print(f"Direct travel {i} → {ip1}: {T_dir:.2f} h, SOC after = {soc_after:.1f} kWh, ")
            events.append(dict(
                t_start=clock, t_end=clock + T_dir,
                location=str(i),
                kind="travel",
                soc_start=soc_here, soc_end=soc_after,
                hos_start=hos_here if not w_i else 0, hos_end=hos_at_ip1,
                label=f"→ {ip1}  ({T_dir*60:.0f} min)",
            ))
            clock += T_dir

    return events


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 – helper: all unique locations in route order
# ──────────────────────────────────────────────────────────────────────────────

def _ordered_locations(timeline, data):
    """Return ordered list of location labels as they first appear."""
    seen  = []
    order = []
    for ev in timeline:
        loc = ev["location"]
        if loc not in seen:
            seen.append(loc)
            order.append(loc)
    return order


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1 – Route Gantt chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_route_gantt(timeline, data, ax=None, show=True):
    """
    Gantt chart: x = clock time [h], y = location.

    Bar colours:
      Blue   – driving (travel between locations)
      Amber  – charging only
      Green  – HOS break at customer
      Purple – break-and-charge at charging station
      Grey   – brief customer stop (no break)
    Arrows connect consecutive bars to show order of movement.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 5))

    locs   = _ordered_locations(timeline, data)
    y_pos  = {loc: idx for idx, loc in enumerate(locs)}

    COLOUR = {
        "travel":               C_DRIVE,
        "travel_to_charger":    C_DRIVE,
        "travel_from_charger":  C_DRIVE,
        "at_customer":          C_IDLE,
        "break":                C_BREAK,
        "charging":             C_CHARGE,
        "bac":                  C_BAC,
    }

    bar_height = 0.55

    for ev in timeline:
        loc   = ev["location"]
        y     = y_pos[loc]
        dur   = ev["t_end"] - ev["t_start"]
        color = COLOUR.get(ev["kind"], C_IDLE)

        # For travel events show a thin diagonal arrow instead of a fat bar
        if ev["kind"] in ("travel", "travel_to_charger", "travel_from_charger"):
            # Determine destination location
            if ev["label"]:
                dest_raw = ev["label"].split("→")[1].strip().split(" ")[0]
                dest = dest_raw
            else:
                dest = loc
            y_dest = y_pos.get(dest, y)

            # Draw thin arrow from (t_start, y_src) to (t_end, y_dest)
            ax.annotate(
                "",
                xy     =(ev["t_end"],   y_dest),
                xytext =(ev["t_start"], y),
                arrowprops=dict(arrowstyle="-|>", color=C_DRIVE,
                                lw=1.8, mutation_scale=12),
            )
            # Annotate duration in the middle of the arrow
            mid_t = (ev["t_start"] + ev["t_end"]) / 2
            mid_y = (y + y_dest) / 2
            ax.text(mid_t, mid_y + 0.25, ev["label"].split("(")[1].rstrip(")"),
                    ha="center", va="bottom", fontsize=7, color=C_DRIVE,
                    fontstyle="italic")
        else:
            # Non-travel: a horizontal bar
            if dur < 1e-6:
                dur_plot = 0.04   # tiny marker so zero-duration stops are visible
            else:
                dur_plot = dur

            rect = mpatches.FancyBboxPatch(
                (ev["t_start"], y - bar_height / 2),
                dur_plot, bar_height,
                boxstyle="round,pad=0.01",
                linewidth=0.5,
                edgecolor="white",
                facecolor=color,
                alpha=0.90,
                zorder=3,
            )
            ax.add_patch(rect)

            # Label inside bar if wide enough
            if dur > 0.08:
                ax.text(
                    ev["t_start"] + dur_plot / 2, y,
                    ev["label"],
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold",
                    zorder=4,
                )

    # Y-axis: location labels
    ax.set_yticks(range(len(locs)))
    ax.set_yticklabels(
        [f"Customer {l}" if l.isdigit() else f"Charger {l}" for l in locs],
        fontsize=9,
    )
    ax.set_ylim(-0.7, len(locs) - 0.3)

    # X-axis
    t_end_total = max(ev["t_end"] for ev in timeline)
    ax.set_xlim(-0.05, t_end_total * 1.05)
    ax.set_xlabel("Time from departure [h]", fontsize=10)
    ax.set_title("Route schedule", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_DRIVE,  label="Driving"),
        mpatches.Patch(facecolor=C_BREAK,  label="HOS break (at customer)"),
        mpatches.Patch(facecolor=C_CHARGE, label="Charging"),
        mpatches.Patch(facecolor=C_BAC,    label="Break-and-Charge"),
        mpatches.Patch(facecolor=C_IDLE,   label="Customer stop"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              framealpha=0.9)

    if standalone:
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2 – Battery SOC over time
# ──────────────────────────────────────────────────────────────────────────────

def plot_battery(timeline, data, ax=None, show=True):
    """
    Step-line plot of battery SOC [kWh] as a function of clock time.

    Driving segments slope downward (energy consumed).
    Charging segments slope upward.
    Break / customer-stop segments are horizontal.
    Safety threshold and Y_max are shown as horizontal reference lines.
    The charging phase itself is coloured amber under the curve.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 4))

    Y_max    = data["Y_max"]
    D_safety = data["D_safety"]
    h        = data["h"]
    soc_min  = D_safety * h   # minimum allowed SOC [kWh]

    # Build (time, soc) point list
    pts_t   = []
    pts_soc = []

    for ev in timeline:
        pts_t.append(ev["t_start"])
        pts_soc.append(ev["soc_start"])
        pts_t.append(ev["t_end"])
        pts_soc.append(ev["soc_end"])

    pts_t   = np.array(pts_t)
    pts_soc = np.array(pts_soc)

    # Main SOC curve
    ax.plot(pts_t, pts_soc, color=C_ENERGY, lw=2.0, zorder=4, label="SOC")

    # Shade charging intervals in amber
    for ev in timeline:
        if ev["kind"] in ("charging", "bac") and ev["t_end"] > ev["t_start"]:
            ax.fill_betweenx(
                [ev["soc_start"], ev["soc_end"]],
                ev["t_start"], ev["t_end"],
                color=C_CHARGE, alpha=0.20, zorder=2,
            )
            # Actually we want to shade between t_start–t_end at the soc values
            t_range = np.linspace(ev["t_start"], ev["t_end"], 50)
            soc_range = np.linspace(ev["soc_start"], ev["soc_end"], 50)
            ax.fill_between(t_range, soc_min, soc_range,
                            color=C_CHARGE, alpha=0.18, zorder=2)

    # Shade driving intervals in blue
    for ev in timeline:
        if ev["kind"] in ("travel", "travel_to_charger", "travel_from_charger") \
                and ev["t_end"] > ev["t_start"]:
            t_range = np.linspace(ev["t_start"], ev["t_end"], 50)
            soc_range = np.linspace(ev["soc_start"], ev["soc_end"], 50)
            ax.fill_between(t_range, soc_min, soc_range,
                            color=C_DRIVE, alpha=0.10, zorder=2)

    # Reference lines
    ax.axhline(Y_max, color="grey", lw=1.2, linestyle="--",
               label=f"Y_max = {Y_max:.0f} kWh")
    ax.axhline(soc_min, color=C_LIMIT, lw=1.2, linestyle=":",
               label=f"Safety min = {soc_min:.0f} kWh  ({D_safety} km reserve)")

    # Annotate each customer arrival with the SOC value
    customer_ids = [str(n) for n in sorted(data["N"])]
    annotated = set()
    for ev in timeline:
        if ev["location"] in customer_ids and ev["location"] not in annotated:
            t_arr = ev["t_start"]
            soc_arr = ev["soc_start"]
            ax.scatter(t_arr, soc_arr, color=C_ENERGY, s=40, zorder=5)
            ax.annotate(
                f" C{ev['location']}\n {soc_arr:.0f} kWh",
                xy=(t_arr, soc_arr),
                fontsize=7, color=C_ENERGY,
                va="bottom",
            )
            annotated.add(ev["location"])

    t_end_total = max(ev["t_end"] for ev in timeline)
    ax.set_xlim(-0.05, t_end_total * 1.05)
    ax.set_ylim(0, Y_max * 1.08)
    ax.set_xlabel("Time from departure [h]", fontsize=10)
    ax.set_ylabel("Battery SOC [kWh]", fontsize=10)
    ax.set_title("Battery state of charge", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3 – HOS break-time counter
# ──────────────────────────────────────────────────────────────────────────────

def plot_hos_status(timeline, data, ax=None, show=True):
    """
    Plot the cumulative driving time accumulated since the last break [h]
    as a function of clock time.

    The EU legal limit (W_break = 4.5 h) is shown as a red dashed line.
    Break events (where the counter resets to 0) are shaded green.
    Break-and-charge events are shaded purple.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 3.5))

    W_break = data["W_break"]   # legal max [h]

    pts_t   = []
    pts_hos = []

    for ev in timeline:
        pts_t.append(ev["t_start"])
        pts_hos.append(ev["hos_start"])
        pts_t.append(ev["t_end"])
        pts_hos.append(ev["hos_end"])

    pts_t   = np.array(pts_t)
    pts_hos = np.array(pts_hos)

    # Main HOS curve
    ax.plot(pts_t, pts_hos, color=C_HOS, lw=2.0, zorder=4,
            label="Drive since last break")

    # Shade breaks green / purple
    for ev in timeline:
        if ev["kind"] in ("break", "bac") and ev["t_end"] > ev["t_start"]:
            color = C_BREAK if ev["kind"] == "break" else C_BAC
            ax.axvspan(ev["t_start"], ev["t_end"], color=color,
                       alpha=0.25, zorder=2,
                       label=("HOS break" if ev["kind"] == "break"
                              else "Break-and-Charge"))

    # Legal limit line
    ax.axhline(W_break, color=C_LIMIT, lw=1.5, linestyle="--",
               label=f"EU limit = {W_break} h (EC No 561/2006)")

    t_end_total = max(ev["t_end"] for ev in timeline)
    ax.set_xlim(-0.05, t_end_total * 1.05)
    ax.set_ylim(-0.1, W_break * 1.2)
    ax.set_xlabel("Time from departure [h]", fontsize=10)
    ax.set_ylabel("Drive since last break [h]", fontsize=10)
    ax.set_title("HOS break status", fontsize=12, fontweight="bold")

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left",
              fontsize=8, framealpha=0.9)

    ax.grid(linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE – all three plots stacked vertically
# ──────────────────────────────────────────────────────────────────────────────

def plot_all(model, successor, data, save_path=None, show=True):
    """
    Build the timeline once and render all three plots in a single figure.

    Parameters
    ----------
    model     : solved Pyomo ConcreteModel
    successor : dict {i: i+1} from build_successor()
    data      : the original data dict passed to build_bet_tdsp_model()
    save_path : optional str – if given, save figure to this path
    show      : bool – call plt.show() at the end
    """
    timeline = build_timeline(model, successor, data)

    fig, axes = plt.subplots(3, 1, figsize=(15, 11),
                             gridspec_kw={"height_ratios": [2.5, 1.8, 1.4]})
    plt.subplots_adjust(hspace=0.45)

    plot_route_gantt(timeline, data, ax=axes[0], show=False)
    plot_battery    (timeline, data, ax=axes[1], show=False)
    plot_hos_status (timeline, data, ax=axes[2], show=False)

    # Shared x-axis label only on bottom panel
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")

    # Vertical time-marker lines across all panels at break events
    for ev in timeline:
        if ev["kind"] in ("break", "bac"):
            for ax in axes:
                ax.axvline(ev["t_start"], color="grey", lw=0.7,
                           linestyle=":", alpha=0.5, zorder=1)
                ax.axvline(ev["t_end"],   color="grey", lw=0.7,
                           linestyle=":", alpha=0.5, zorder=1)

    fig.suptitle(
        f"BET Route Analysis  |  "
        f"Total: {value(model.T_total):.2f} h  "
        f"(Drive: {value(model.T_drive):.2f} h, "
        f"Stop: {value(model.T_stop):.2f} h)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig, axes, timeline
