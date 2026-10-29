"""
fig_schedule.py — the schedule of one test route under each policy (paper figure)
=================================================================================
One timeline per policy on one base-case test route, on its stored travel-time
realisation: driving, charging, breaks, daily rests and customer service, the
customers' time windows, missed windows, and the stops where the rollout
changed the learned policy's action.

The oracle and the LA are read from their stored runs; the PyTorch student
(seed 0), its rollout, and the expert-iteration student's rollout are re-driven
here and checked against the stored test runs (rl_search_dsr_test_*_s0,
rl_search_exit_test_r2).  The route is the one whose gaps are closest to every
policy's test average (see choose()).  Writes fig_schedule.json (the
schedules) and fig_schedule.pdf/.png.

    python ML/paper/fig_schedule.py            # re-drive, cache, draw
    python ML/paper/fig_schedule.py --draw     # draw from fig_schedule.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE = os.path.join(HERE, "..", "code")
RES = os.path.join(HERE, "..", "results")
SOL = os.path.join(ROOT, "solutions", "basecase")
CACHE = os.path.join(HERE, "fig_schedule.json")
T0 = 8.0
INST = "RlongCmediumTmedium_25"
TORCH = "tmlp_F95_split_list_s0"
EXIT = "tmlp_F95_split_list_s0_exit2"


def _stops_from(fd, t_arr, td, durs):
    out = []
    for k in range(len(td)):
        d = durs[k] if k < len(durs) else {}
        out.append(dict(stop=k, ta=float(t_arr[k]) - T0, td=float(td[k]) - T0,
                        tauc=float(d.get("tauc", 0.0)), taub=float(d.get("taub", 0.0)),
                        taur=float(d.get("taur", 0.0))))
    return out


def _windows(fd, stops):
    """Mark customer visits and whether the arrival is outside [Wha, Whf]."""
    C = set(fd["C"])
    miss = 0
    for s in stops:
        s["cust"] = s["stop"] in C
        s["miss"] = False
        if s["cust"]:
            t = s["ta"] + T0
            s["miss"] = bool(t < fd["Wha"][s["stop"]] - 1e-3 or t > fd["Whf"][s["stop"]] + 1e-3)
            miss += int(s["miss"])
    return miss


def compute(inst):
    sys.path.insert(0, CODE)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from paper_link import gaps, oracle_facts
    from rl_env import final_student, load_route, realised, run_batch
    from rl_policy import RolloutChooser

    fd, D, E, cv = load_route(inst)
    orc = oracle_facts(inst)
    rows = []

    o = json.load(open(os.path.join(SOL, f"oracle_{inst}.json")))
    sol = sorted(o["sol"], key=lambda s: s["i"])
    st = [dict(stop=s["i"], ta=s["ta"] - T0, td=(s["td"] if s.get("td") is not None else s["ta"]) - T0,
               tauc=s["tauc"], taub=s["taub"], taur=s["taur"]) for s in sol]
    m = _windows(fd, st)
    rows.append(dict(name="Oracle (hindsight)", stops=st, arrival=st[-1]["ta"], misses=m,
                     gap=0.0, changed=[]))

    f = sorted(glob.glob(os.path.join(SOL, f"{inst}_LA_MIPTAIL_*.json")))[-1]
    s = json.load(open(f))
    st = _stops_from(fd, [x["t_arr"] for x in s["sim_trajectory"]], s["td_list"], s["durations_list"])
    m = _windows(fd, st)
    assert m == int(s["metrics"]["tw_n_misses"]), (m, s["metrics"]["tw_n_misses"])
    rows.append(dict(name="LA", stops=st, arrival=s["duration_h"], misses=m,
                     gap=gaps(s["duration_h"], m, s["duration_h"] + T0, orc)[1], changed=[]))

    def stored(run, k):
        for line in open(os.path.join(RES, f"rl_search_{run}.jsonl")):
            r = json.loads(line)
            if r["instance"] == inst:
                return r["res"][k][0]
        raise SystemExit(f"{inst} not in {run}")

    for name, tag, run, k, roll in (("Learned network", TORCH, f"dsr_test_{TORCH}", 0, False),
                                    ("Network + rollout", TORCH, f"dsr_test_{TORCH}", 1, True),
                                    ("Expert iteration + rollout", EXIT, "exit_test_r2", 1, True)):
        base = final_student("torch", tag)
        rec = []
        ch = None
        if roll:
            ch = RolloutChooser(base, futures=32, race=8, record=rec)
            ch.route, ch.cv = inst, cv
        res, vehs = run_batch(base, fd, cv, [(ch, realised(D, E))], return_vehicles=True)
        r, v = res[0], vehs[0]
        ref = stored(run, k)
        assert abs(r["duration"] - ref["duration"]) < 1e-6 and r["misses"] == ref["misses"], (name, r, ref)
        st = _stops_from(fd, v.t_arr_history, v.td_list, v.durations)
        m = _windows(fd, st)
        assert m == r["misses"], (name, m, r["misses"])
        changed = [int(x["stop"]) for x in rec if int(x["choice"]) != 0]
        rows.append(dict(name=name, stops=st, arrival=r["duration"], misses=m,
                         gap=gaps(r["duration"], m, r["duration"] + T0, orc)[1], changed=changed))
        print(f"{name}: {r['duration']:.2f} h, {m} missed, rollout checked {len(rec)} and changed {len(changed)}")

    win = [(int(c), fd["Wha"][c] - T0, fd["Whf"][c] - T0) for c in fd["C"]]
    km = float(fd.get("H", 0.0))
    out = dict(instance=inst, km=km, customers=len(fd["C"]), windows=win, rows=rows)
    with open(CACHE, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


# Okabe-Ito hues (fixed per activity), dark grey for rests, light grey driving
COL = dict(drive="#D0D0D0", charge="#0072B2", brk="#E69F00", rest="#3A3A3A", service="#009E73",
           window="#009E73", miss="#D55E00")


def draw(data, rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.0, "pdf.fonttype": 42})
    R = [r for r in data["rows"] if rows is None or r["name"] in rows]
    n = len(R)
    fig, ax = plt.subplots(figsize=(4.8, 0.34 * n + 0.72))
    xmax = max(r["arrival"] for r in R)
    for _c, a, b in data["windows"]:
        ax.axvspan(a, b, color=COL["window"], alpha=0.13, lw=0, zorder=0)
    for y, r in zip(range(n - 1, -1, -1), R):
        ax.barh(y, r["arrival"], height=0.16, color=COL["drive"], lw=0, zorder=1)
        for s in r["stops"]:
            dw = s["td"] - s["ta"]
            if dw <= 1e-4:
                continue
            if s["taur"] > 0.5:
                c = COL["rest"]
            elif s["tauc"] > 1e-6:
                c = COL["charge"]
            elif s["taub"] > 1e-6:
                c = COL["brk"]
            elif s["cust"]:
                c = COL["service"]
            else:
                continue
            ax.barh(y, dw, left=s["ta"], height=0.46, color=c, lw=0, zorder=2)
        for s in r["stops"]:
            if s["cust"] and s["miss"]:
                ax.plot(s["ta"], y + 0.36, marker="x", ms=4.2, mew=1.1, color=COL["miss"], zorder=4,
                        ls="none")
        tchg = {s["stop"]: s["ta"] for s in r["stops"]}
        for k in r["changed"]:
            ax.plot(tchg[k], y - 0.36, marker="^", ms=3.2, color="#000000", zorder=4, ls="none")
        lab = f"{r['arrival']:.1f} h, gap {r['gap']:.2f}%"
        lab += f", {r['misses']} missed" if r["misses"] else ", no miss"
        ax.text(-2.0, y + 0.13, r["name"], va="center", ha="right", fontsize=8, color="#111111")
        ax.text(-2.0, y - 0.22, lab, va="center", ha="right", fontsize=6.5, color="#444444")
    ax.set_yticks([])
    ax.set_ylim(-0.7, n - 0.4)
    ax.set_xlim(0, xmax + 1)
    step = 12 if xmax <= 130 else 24
    ax.set_xticks(np.arange(0, xmax + 1, step))
    ax.set_xlabel("Hours since departure")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [Patch(color=COL["drive"], label="Driving"), Patch(color=COL["charge"], label="Charging"),
               Patch(color=COL["brk"], label="Break"), Patch(color=COL["rest"], label="Daily rest"),
               Patch(color=COL["service"], label="Customer service"),
               Patch(color=COL["window"], alpha=0.13, label="Time window"),
               Line2D([], [], marker="x", ls="none", color=COL["miss"], mew=1.1, ms=4.2, label="Missed window"),
               Line2D([], [], marker="^", ls="none", color="#000000", ms=3.2, label="Changed by the rollout")]
    ax.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              fontsize=7, handlelength=1.2, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(left=0.25, right=0.99)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_schedule.{ext}"), dpi=300)
    print("wrote fig_schedule.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", action="store_true", help="draw from the cached schedules")
    ap.add_argument("--inst", default=INST)
    ap.add_argument("--rows", default="Oracle (hindsight),LA,Learned network,Network + rollout",
                    help="comma-separated row names to keep ('' = all)")
    a = ap.parse_args()
    data = json.load(open(CACHE)) if a.draw else compute(a.inst)
    draw(data, a.rows.split(",") if a.rows else None)


if __name__ == "__main__":
    main()
