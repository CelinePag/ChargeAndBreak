"""
fig_la_time.py — time per decision: the LA teacher vs the student, by stop type
==============================================================================
The LA's times come from its own logs: every base-case run (MIP tail, 25
scenarios, 24 h, 8 parallel workers on a 4-core laptop) prints the wall time
of each decision.  The student's are measured here: the trees (F95, base case,
spread-room check, 0.99 guard) drive 24 base-case test routes in one process,
timing each call of decide().  Marker = median, bar = 10th-90th percentile;
the axis is logarithmic.  Cached in ML/results/decision_times.json
(--refresh re-parses and re-times).

    python ML/code/fig_la_time.py [--refresh]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

from ml_style import (COLOR, INK_MUTED, INK_PRIMARY, apply_rc,   # noqa: E402
                      shade, style_axes)

FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
CACHE = os.path.abspath(os.path.join(HERE, "..", "results", "decision_times.json"))
TYPES = [("CS", "charging station"), ("LAYBY", "layby"), ("CUST", "customer")]
RE_STOP = re.compile(r"^\[LA\] stop (\d+) \((\w+)\)")
RE_CHOSEN = re.compile(r"^  -> CHOSEN .*\s([\d.]+)s\s*$")


def _type(kind):
    return "CUST" if kind.startswith("CUST") else kind


def la_times():
    out = {t: [] for t, _ in TYPES}
    for f in sorted(glob.glob(os.path.join(_ROOT, "logs", "basecase", "*LA_MIPTAIL*.txt"))):
        kind = None
        for line in open(f, encoding="utf-8", errors="replace"):
            m = RE_STOP.match(line)
            if m:
                kind = _type(m.group(2))
                continue
            m = RE_CHOSEN.match(line)
            if m and kind in out:
                out[kind].append(float(m.group(1)))
                kind = None
    return out


def student_times(n_routes=24):
    from phys_eval import instances
    from policy_core import load_policy, run_student
    from src.instance_gen.instance_io import load_instance_json
    pol = load_policy("gbt", "gbt_F95_base_s1", guard_q=0.99, spread_room=True)
    out = {t: [] for t, _ in TYPES}

    class Timed:
        def __init__(self, p, fd):
            self.p, self.K, self.C = p, set(fd["K"]), set(fd["C"])

        def decide(self, fd, pre, stop, veh, cv):
            t0 = time.perf_counter()
            r = self.p.decide(fd, pre, stop, veh, cv)
            k = "CS" if stop in self.K else "CUST" if stop in self.C else "LAYBY"
            out[k].append(time.perf_counter() - t0)
            return r

    routes = instances("base")
    for name, path in routes[:: max(1, len(routes) // n_routes)][:n_routes]:
        fd, D, E, cv = load_instance_json(path)
        fd["_horizon_h"] = 24.0
        run_student(fd, D, E, Timed(pol, fd), cv=cv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if os.path.exists(CACHE) and not args.refresh:
        with open(CACHE) as fh:
            c = json.load(fh)
    else:
        c = dict(LA=la_times(), student=student_times())
        with open(CACHE, "w") as fh:
            json.dump(c, fh)

    apply_rc()
    fig, ax = plt.subplots(figsize=(6.6, 2.5), dpi=200)
    style_axes(ax)
    rows = []
    y = 0
    for who, lab, ck, mk in (("LA", "LA (the teacher)", "LA", "D"),
                             ("student", "Trees (the student)", "gbt", "o")):
        for t, tl in TYPES:
            v = np.array(c[who][t])
            if len(v):
                rows.append((y, f"{lab} — {tl}", ck, mk, v))
                y += 1
        y += 0.4
    for yy, lab, ck, mk, v in rows:
        p10, med, p90 = np.percentile(v, [10, 50, 90])
        col = COLOR[ck]
        ax.plot([p10, p90], [yy, yy], color=shade(col, 0.1), lw=2.0,
                solid_capstyle="round", zorder=2)
        ax.plot([med], [yy], marker=mk, ms=5.0, color=shade(col, 0.35), mfc=col,
                mew=1.2, zorder=3)
        txt = f"{med:.0f} s" if med >= 1 else f"{1000 * med:.1f} ms"
        ax.text(p90 * 1.35, yy, f"{txt}  (n={len(v):,})", va="center",
                fontsize=6.0, color=INK_MUTED)
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 3e3)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=6.4, color=INK_MUTED)
    ax.tick_params(axis="y", pad=2, length=0)
    ax.invert_yaxis()
    ax.set_xlabel("wall time per decision (seconds, log scale)   "
                  "marker = median, bar = 10th-90th percentile")
    la_cs = np.median(c["LA"]["CS"])
    st_cs = np.median(c["student"]["CS"])
    ax.set_title(f"Time per decision — at a charging station the student is "
                 f"~{la_cs / st_cs:,.0f}x faster", fontsize=8.6,
                 color=INK_PRIMARY, loc="left", pad=6)
    ax.annotate("LA: every decision of the 840 stored base-case runs (25 scenario "
                "MILPs per candidate action, 8 workers, 4-core laptop). Student: "
                "24 base-case test routes, one process, same laptop -- timed "
                "while an LA batch was running, so an upper bound (re-time idle "
                "with --refresh).",
                xy=(0, 0), xycoords="axes fraction", xytext=(-120, -34),
                textcoords="offset points", va="top", fontsize=5.8,
                color=INK_MUTED, annotation_clip=False)
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "fig_la_time.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
