"""
fig_results.py -- the three result figures of the CPAIOR draft
==============================================================
Redrawn 2026-10-20 after "ugly and unreadable": one message per figure, one
measure everywhere, plain words, horizontal bars with their values written on
them, no hatching, no error bars (the paired statistics go in the captions).

  fig_base.pdf      the learned policy is as good as the look-ahead, and it
                    stays legal only because of the shield
  fig_mixed.pdf     chargers of different power: with LA data from such routes
                    the learned policy charges where the look-ahead does, but
                    is still slower
  fig_residual.pdf  the rest of the gap is a few routes with one extra daily
                    rest; asking the solver only on doubtful rest decisions
                    recovers part of it

Measure: "extra time over the best schedule in hindsight" = route duration
minus the hindsight oracle's, in % of the oracle's (the manuscript's gap,
ML/code/paper_link.py for the base case, mixed_eval._gap for mixed routes,
certified oracles only).  Learned policies: each route averaged over the 3
training seeds first, then averaged over routes.

Both objectives are drawn (2026-10-20): every bar comes in a pair, the light
one "time only" (the measure above) and the dark one "with missed windows",
the paper's objective: duration + 0.5 h per missed time window, against the
oracle's duration + its own penalty (paper_link.gaps' second value).

    python ML/paper/fig_results.py              # render from fig_data.json
    python ML/paper/fig_results.py --recompute  # rebuild the numbers first
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ML, "code"))
RES = os.path.join(ML, "results")
CACHE = os.path.join(HERE, "fig_data.json")

# manuscript colours (src/plot/paper_style.py): one colour per kind of method
C_LA, C_GREEDY, C_BEST, C_LEARN = "#009E73", "#0072B2", "#3A3A3A", "#56B4E9"
INK, MUTED, LIGHT = "#1a1a1a", "#666666", "#d9d9d9"
BETA = 0.5          # h per missed time window, the paper's objective
TIME_ONLY, WITH_TW = "time only", "with missed windows"

UNIFORM = ["torch split+list, T inputs, all phys"]           # trained on uniform routes
MIXED = ["torch T, + pmix + mix routes", "torch T, + pmix + mix routes (s1)",
         "torch T, + pmix + mix routes (s2)"]               # + LA data on mixed routes


# ═════════════════════════════════════════════════════════════════════════════
#  numbers
# ═════════════════════════════════════════════════════════════════════════════

def base_case():
    import paper_link as pl
    tag = "tmlp_F95_split_list_s{}"
    per_seed, per_seed_p, la, greedy, ms, la_s, dec = [], [], None, None, [], 0.0, 0
    for s in range(3):
        f = f"eval_{tag.format(s)}_g99sr_test.json"
        g = pl.collect_gaps(f, "student")
        per_seed.append(float(np.mean(g["student"][0])))
        per_seed_p.append(float(np.mean(g["student"][1])))
        if s == 0:
            gl, gg = g[pl.LABEL["LA_MIPTAIL"]], g[pl.LABEL["GREEDY"]]
            la, la_p = float(np.mean(gl[0])), float(np.mean(gl[1]))
            greedy, greedy_p = float(np.mean(gg[0])), float(np.mean(gg[1]))
        rows = json.load(open(os.path.join(RES, f)))
        ms += [r["ms_per_decision"] for r in rows]
        if s == 0:
            la_s = sum(r["LA_wall_s"] for r in rows if r.get("LA_wall_s"))
            dec = sum(r["decisions"] for r in rows)

    def fails(v):
        rows = [r for s in range(3)
                for r in json.load(open(os.path.join(RES, f"eval_{tag.format(s)}_{v}_test.json")))]
        return sum(not r["route_completed"] for r in rows), len(rows)

    return dict(gap=dict(la=la, learned=float(np.mean(per_seed)), greedy=greedy),
                gap_pen=dict(la=la_p, learned=float(np.mean(per_seed_p)), greedy=greedy_p),
                ms_learned=float(np.mean(ms)), s_la=la_s / dec,
                fails_shield=fails("g99sr"), fails_noshield=fails("nsN"))


def gap_pen(r, o):
    """mixed_eval._gap with the window penalty on both sides: duration + 0.5 h
    per missed window, against the oracle's duration + its own penalty."""
    if r["duration_h"] is None or o is None:
        return None
    od = o["ta_N"] - 8.0
    return (100.0 * (r["duration_h"] + BETA * r["tw"] - od - o["pen"]) / (od + o["pen"])
            if od > 0 else None)


def mixed_gaps():
    """G / P[(method, variant, base)] = gap (%) without / with the window
    penalty; D / DP[...] = duration (h) without / with it; R[...] = rests."""
    import mixed_eval as me
    rows = me.read_rows(me.store_path(0.99, True))
    orc, G, P, D, DP, R = {}, {}, {}, {}, {}, {}
    for r in rows:
        k = (r["variant"], r["instance"])
        if k not in orc:
            orc[k] = me.read_oracle(*k)
        if not r["completed"]:
            continue
        b = r["instance"].split("__")[0]
        key = (r["method"], r["variant"], b)
        D[key], R[key] = r["duration_h"], r["rests"]
        DP[key] = r["duration_h"] + BETA * r["tw"]
        g = me._gap(r, orc[k])
        if g is not None and orc[k] and (orc[k]["gap"] or 0) <= 0.01:
            G[key], P[key] = g, gap_pen(r, orc[k])
    for v in ("pmix", "mix"):
        for b in sorted({b for (_m, vv, b) in D if vv == v}):
            nm = f"{b}__{v}"
            la, o = me.la_row(v, nm), me.read_oracle(v, nm)
            if not la or not la["completed"]:
                continue
            D[("LA", v, b)], R[("LA", v, b)] = la["duration_h"], la["rests"]
            DP[("LA", v, b)] = la["duration_h"] + BETA * la["tw"]
            if o and (o["gap"] or 0) <= 0.01:
                G[("LA", v, b)], P[("LA", v, b)] = me._gap(la, o), gap_pen(la, o)
    return me, G, P, D, DP, R


def route_mean(T, labels, v, b):
    x = [T.get((m, v, b)) for m in labels]
    return None if any(y is None for y in x) else float(np.mean(x))


def mixed(me, G, P):
    """Mean gap, without and with the window penalty, on the routes every
    policy below has (paired)."""
    pols = {"la": ["LA"], "greedy": ["Greedy"], "uniform": UNIFORM, "mixed": MIXED}
    per = {k: [] for k in pols}
    per_p = {k: [] for k in pols}
    n = 0
    for v in ("pmix", "mix"):
        for b in sorted({b for (_m, vv, b) in G if vv == v}):
            x = {k: route_mean(G, labs, v, b) for k, labs in pols.items()}
            xp = {k: route_mean(P, labs, v, b) for k, labs in pols.items()}
            if all(y is not None for y in list(x.values()) + list(xp.values())):
                n += 1
                for k in pols:
                    per[k].append(x[k])
                    per_p[k].append(xp[k])
    return dict(gap={k: float(np.mean(v)) for k, v in per.items()},
                gap_pen={k: float(np.mean(v)) for k, v in per_p.items()}, routes=n)


def charging_share(me):
    """Share of charged energy at >= 700 kW chargers, mixed test routes."""
    keep = set(UNIFORM + MIXED)
    full = me.MODELS
    me.MODELS = [m for m in full if m[2] in keep]
    E = {}
    try:
        for v in ("pmix", "mix"):
            E_by, _n, _r = me.energy_by_power(v, la_only=True)
            for lab, c in E_by.items():
                a = E.setdefault(lab, {})
                for k, e in c.items():
                    a[int(k)] = a.get(int(k), 0.0) + float(e)
    finally:
        me.MODELS = full

    def share(labs):
        tot = sum(sum(E[x].values()) for x in labs)
        return sum(e for x in labs for k, e in E[x].items() if k >= 700) / tot
    return dict(best=share(["oracle"]), la=share(["LA"]), greedy=share(["Greedy"]),
                uniform=share(UNIFORM), mixed=share(MIXED))


def residual(D, DP, R):
    """Mixed-trained learned policy minus the look-ahead, hours, per run;
    `_pen` counts 0.5 h per missed window on both sides."""
    out = dict(extra=[], other=[], extra_pen=[], other_pen=[])
    for (m, v, b), d in D.items():
        if m not in MIXED or ("LA", v, b) not in D:
            continue
        grp = "extra" if R[(m, v, b)] > R[("LA", v, b)] else "other"
        out[grp].append(d - D[("LA", v, b)])
        out[grp + "_pen"].append(DP[(m, v, b)] - DP[("LA", v, b)])
    return out


def hybrid(me, G, P, cfg="m60"):
    """Learned alone / learned + solver on doubtful rests / look-ahead, paired."""
    tags = {f"tmlp_T144_physmix_split_list_s{s}": s for s in range(3)}
    per, calls = {}, []
    for r in (json.loads(x) for x in open(os.path.join(RES, f"hybrid_{cfg}.jsonl")) if x.strip()):
        calls.append(r["n_calls"])
        if not r["completed"]:
            continue
        o = me.read_oracle(r["variant"], r["instance"])
        if o and (o["gap"] or 0) <= 0.01:
            per.setdefault((r["variant"], r["instance"].split("__")[0]), {})[tags[r["method"]]] = \
                (me._gap(r, o), gap_pen(r, o))
    alone, hyb, la, alone_p, hyb_p, la_p = [], [], [], [], [], []
    for (v, b), x in per.items():
        s, sp = route_mean(G, MIXED, v, b), route_mean(P, MIXED, v, b)
        if len(x) == 3 and s is not None and sp is not None and ("LA", v, b) in G:
            hyb.append(float(np.mean([g for g, _p in x.values()])))
            hyb_p.append(float(np.mean([p for _g, p in x.values()])))
            alone.append(s)
            alone_p.append(sp)
            la.append(G[("LA", v, b)])
            la_p.append(P[("LA", v, b)])
    return dict(alone=float(np.mean(alone)), hybrid=float(np.mean(hyb)), la=float(np.mean(la)),
                alone_pen=float(np.mean(alone_p)), hybrid_pen=float(np.mean(hyb_p)),
                la_pen=float(np.mean(la_p)), calls=float(np.mean(calls)), routes=len(hyb))


# ── the paper's mixed-route figure, final protocol (2026-10-26) ──────────────
# Re-driven the way every table of the paper runs the learned policies: inputs
# at nominal travel times and the repositioning time paid (run_hybrid --mseq
# --nominal-features).  Seed 0 is checked against hybrid_alone_ms_N.jsonl,
# which run_hybrid wrote under the same protocol.
MIXED_TAGS = [f"tmlp_T144_physmix_split_list_s{s}" for s in range(3)]
UNIFORM_TAG = "tmlp_T144_phys_split_list_s0"


class _Fwd:
    """Keeps the vehicle for its charging history; every other attribute
    (pay_mseq, ...) is the policy's own."""

    def __init__(self, pol):
        self.pol, self.veh = pol, None

    def __getattr__(self, k):
        return getattr(self.pol, k)

    def decide(self, fd, pre, stop, veh, cv):
        self.veh = veh
        return self.pol.decide(fd, pre, stop, veh, cv)


def mixed_final():
    """Energy taken at >= 700 kW chargers on the 64 mixed test routes the LA
    ran, and the gap to the oracle (time only / with the window penalty) on
    the routes with a certified oracle that every policy completed."""
    import collections
    import mixed_eval as me
    from features import charger_kw
    from policy_core import load_policy, run_student
    me.ROUTE_SET = "test"
    pols = {}
    for tag in MIXED_TAGS + [UNIFORM_TAG]:
        p = load_policy("torch", tag, guard_q=0.99, spread_room=True)
        p.feature_q, p.pay_mseq = None, True
        pols[tag] = p
    groups = {"best": ["oracle"], "la": ["LA"], "greedy": ["Greedy"],
              "uniform": [UNIFORM_TAG], "mixed": MIXED_TAGS}
    stored = {}
    for x in open(os.path.join(RES, "hybrid_alone_ms_N.jsonl")):
        if x.strip():
            r = json.loads(x)
            stored[(r["variant"], r["instance"])] = r
    E = collections.defaultdict(lambda: collections.defaultdict(float))
    G, P, dev, n_routes = {}, {}, [], 0
    for v in ("pmix", "mix"):
        keep = set(me.la_routes(v))
        for _v, name, path in me.instances([v], uniform=False):
            if name not in keep:
                continue
            n_routes += 1
            fd, D, Ereal, cv = me.load(path)
            kw = {k: int(round(charger_kw(fd, k))) for k in fd["K"]}
            o = me.read_oracle(v, name)
            cert = o is not None and (o["gap"] or 0) <= 0.01
            rows = {}
            with open(me.oracle_path(v, name)) as fh:
                for s in json.load(fh).get("sol") or []:
                    if s["i"] in kw and s["ed"] - s["ea"] > 1e-3:
                        E["oracle"][kw[s["i"]]] += s["ed"] - s["ea"]
            la = me.la_row(v, name)
            tr = la["trajectory"]
            for k in range(len(tr) - 1):
                st = int(tr[k]["stop"])
                c = float(tr[k + 1]["e_arr"]) + float(Ereal[st]) - float(tr[k]["e_arr"])
                if st in kw and c > 1e-3:
                    E["LA"][kw[st]] += c
            rows["LA"] = la
            veh = me.run_greedy_ml(fd, D, Ereal, cv, return_vehicle=True)
            done = (not veh.is_halted) and veh.stop >= int(fd["N"])
            rows["Greedy"] = dict(duration_h=(veh.t_arr - float(fd.get("T_START", 8.0)))
                                  if done else None, tw=len(veh.tw_misses))
            for st, c in me._charged(veh, Ereal).items():
                if st in kw and c > 1e-3:
                    E["Greedy"][kw[st]] += c
            for tag, pol in pols.items():
                fd, D, Ereal, cv = me.load(path)
                spy = _Fwd(pol)
                r = run_student(fd, D, Ereal, spy, cv=cv)
                rows[tag] = dict(duration_h=r["duration_h"], tw=r["tw_misses"])
                for st, c in me._charged(spy.veh, Ereal).items():
                    if st in kw and c > 1e-3:
                        E[tag][kw[st]] += c
                if tag == MIXED_TAGS[0] and (v, name) in stored:
                    a, b = r["duration_h"], stored[(v, name)]["duration_h"]
                    dev.append(abs(a - b) if a is not None and b is not None
                               else (0.0 if a is b else float("inf")))
            if cert:
                for lab, r in rows.items():
                    if r.get("duration_h") is not None:
                        G[(lab, v, name)], P[(lab, v, name)] = me._gap(r, o), gap_pen(r, o)
    per = {k: [] for k in groups}
    per_p = {k: [] for k in groups if k != "best"}
    n_gap = 0
    for (lab, v, name) in [k for k in G if k[0] == "LA"]:
        x = {k: [G.get((m, v, name)) for m in labs] for k, labs in groups.items() if k != "best"}
        xp = {k: [P.get((m, v, name)) for m in labs] for k, labs in groups.items() if k != "best"}
        if any(y is None for vals in list(x.values()) + list(xp.values()) for y in vals):
            continue
        n_gap += 1
        for k in x:
            per[k].append(float(np.mean(x[k])))
            per_p[k].append(float(np.mean(xp[k])))

    def share(labs):
        tot = sum(sum(E[x].values()) for x in labs)
        return sum(e for x in labs for k, e in E[x].items() if k >= 700) / tot
    return dict(share={k: share(labs) for k, labs in groups.items()},
                gap={k: float(np.mean(per[k])) for k in per_p},
                gap_pen={k: float(np.mean(per_p[k])) for k in per_p},
                routes_energy=n_routes, routes_gap=n_gap,
                check_s0_max_abs_h=max(dev) if dev else None, check_s0_n=len(dev))


def recompute():
    data = {"base": base_case()}
    with contextlib.redirect_stdout(io.StringIO()):
        me, G, P, D, DP, R = mixed_gaps()
    data["mixed"] = mixed(me, G, P)
    old = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if "share_v2" in old:
        data["share_v2"] = old["share_v2"]
    else:
        print("[fig] re-driving the learned policies for the charging shares ...", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            data["share_v2"] = charging_share(me)
    data["residual"] = residual(D, DP, R)
    data["hybrid"] = hybrid(me, G, P)
    with open(CACHE, "w") as fh:
        json.dump(data, fh, indent=1)
    return data


# ═════════════════════════════════════════════════════════════════════════════
#  drawing: horizontal bars, value at the end of each bar
# ═════════════════════════════════════════════════════════════════════════════

def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType in the PDF (LNCS)
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
    return plt


def bars(ax, rows, title, unit, xmax, fmt="{:.1f}", note=None):
    """rows: [(label, value or None, colour, sublabel or None)], top to bottom."""
    n = len(rows)
    for i, (lab, val, col, sub) in enumerate(rows):
        y = n - 1 - i
        if val is not None:
            ax.barh(y, val, height=0.62, color=col, lw=0)
            ax.text(val + xmax * 0.015, y, fmt.format(val) + unit, va="center",
                    fontsize=8.5, color=INK, fontweight="bold")
        elif note:
            ax.text(xmax * 0.015, y, note, va="center", fontsize=8, color=MUTED,
                    style="italic")
        txt = lab if not sub else f"{lab}\n"
        ax.text(-xmax * 0.03, y + (0.13 if sub else 0), lab, ha="right", va="center",
                fontsize=8.5, color=INK)
        if sub:
            ax.text(-xmax * 0.03, y - 0.24, sub, ha="right", va="center", fontsize=7.3,
                    color=MUTED)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(0, color=MUTED, lw=0.8)
    title_(ax, title)


def title_(ax, text):
    ax.text(0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom",
            multialignment="left", fontsize=9, color=INK, fontweight="bold")


def save(plt, fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[fig] {name}.pdf / .png")


def tint(colour, a=0.55):
    """The method's colour mixed with white: the "time only" bar of a pair."""
    h = colour.lstrip("#")
    c = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(x + (255 - x) * a):02x}" for x in c)


def pair_bars(ax, rows, title, unit, xmax, fmt="{:.1f}", ref_note=None, labels=True):
    """rows: [(label, time only, with missed windows, colour, sublabel)], top
    to bottom.  Each row is a pair: the light bar is time only, the full-colour
    bar adds 0.5 h per missed window; the first pair names both in words.  A
    row whose values are None is the reference (ref_note).  labels=False
    leaves the row labels to a panel on the left."""
    n, h, off, named = len(rows), 0.34, 0.19, False
    for i, (lab, v0, v1, col, sub) in enumerate(rows):
        y = n - 1 - i
        if v0 is None:
            if ref_note:
                ax.text(xmax * 0.015, y, ref_note, va="center", fontsize=8, color=MUTED,
                        style="italic")
        else:
            for val, c, dy, word in ((v0, tint(col), off, TIME_ONLY),
                                     (v1, col, -off, WITH_TW)):
                ax.barh(y + dy, val, height=h, color=c, lw=0)
                t = ax.text(max(val, 0) + xmax * 0.015, y + dy, fmt.format(val) + unit,
                            va="center", fontsize=7.8, color=INK, fontweight="bold")
                if not named:
                    ax.annotate(word, xy=(1, 0.5), xycoords=t, xytext=(4, 0),
                                textcoords="offset points", va="center", fontsize=7.3,
                                color=MUTED)
            named = True
        if labels:
            ax.text(-xmax * 0.03, y + (0.13 if sub else 0), lab, ha="right", va="center",
                    fontsize=8.5, color=INK)
            if sub:
                ax.text(-xmax * 0.03, y - 0.24, sub, ha="right", va="center",
                        fontsize=7.3, color=MUTED)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(0, color=MUTED, lw=0.8)
    title_(ax, title)


def fig_base(plt, d):
    b, g, gp = d["base"], d["base"]["gap"], d["base"]["gap_pen"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.6, 3.0),
                                 gridspec_kw=dict(height_ratios=[3, 2], hspace=0.55))
    fig.subplots_adjust(left=0.42)
    pair_bars(ax, [("Look-ahead", g["la"], gp["la"], C_LA,
                    f"solver at every stop, {b['s_la']:.0f} s per decision"),
                   ("Learned policy", g["learned"], gp["learned"], C_LEARN,
                    f"neural network, {b['ms_learned']:.0f} ms per decision"),
                   ("Greedy rule", g["greedy"], gp["greedy"], C_GREEDY, None)],
              "(a) Extra time over the best schedule in hindsight", " %",
              1.45 * max(gp.values()))
    f0, n0 = b["fails_shield"]
    f1, n1 = b["fails_noshield"]
    bars(bx, [("Learned policy, with shield", 100 * f0 / n0, C_LEARN, None),
              ("Learned policy, no shield", 100 * f1 / n1, C_LEARN, None)],
         "(b) Runs that broke a driving-time rule", " %", 14, fmt="{:.0f}")
    save(plt, fig, "fig_base")


def fig_mixed(plt, d):
    """The paper's figure (final protocol, mixed_final): where the energy is
    taken, and what it costs.  Terms of the companion paper: oracle, LA,
    greedy heuristic, gap with / without the window penalty."""
    f = d["mixed_final"]
    m, mp, s = f["gap"], f["gap_pen"], f["share"]
    rows = [("Oracle", C_BEST, s["best"], None, "perfect hindsight"),
            ("Look-ahead (LA)", C_LA, s["la"], "la", "solver at every stop"),
            ("Learned policy", C_LEARN, s["uniform"], "uniform", "trained on equal-power routes"),
            ("Learned policy", C_LEARN, s["mixed"], "mixed", "+ LA runs on mixed routes"),
            ("Greedy heuristic", C_GREEDY, s["greedy"], "greedy", None)]
    global TIME_ONLY, WITH_TW
    TIME_ONLY, WITH_TW = "duration only", "with window penalty"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.4, 2.3),
                                 gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.42))
    fig.subplots_adjust(left=0.33)
    bars(ax, [(lab, 100 * sh, col, sub) for lab, col, sh, _k, sub in rows],
         "(a) Share of energy\ncharged at ≥ 700 kW", " %", 125, fmt="{:.0f}")
    # same rows on the right, labels already on the left
    pair_bars(bx, [(lab, m[k] if k else None, mp[k] if k else None, col, sub)
                   for lab, col, _sh, k, sub in rows],
              "(b) Gap to\nthe oracle", " %", 1.3 * max(mp.values()),
              ref_note="reference", labels=False)
    save(plt, fig, "fig_mixed")


def fig_residual(plt, d):
    r, h = d["residual"], d["hybrid"]
    ne, no = len(r["extra"]), len(r["other"])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.6, 3.05),
                                 gridspec_kw=dict(height_ratios=[2, 3], hspace=0.6))
    fig.subplots_adjust(left=0.42)
    ext = [float(np.mean(r[k])) for k in ("extra", "extra_pen", "other", "other_pen")]
    pair_bars(ax, [("Routes with one extra daily rest", ext[0], ext[1], C_LEARN,
                    f"{ne} of {ne + no} runs"),
                   ("All other routes", ext[2], ext[3], C_LEARN, f"{no} of {ne + no} runs")],
              "(a) Extra hours compared with the look-ahead", " h", 1.45 * max(ext),
              fmt="{:+.1f}")
    pair_bars(bx, [("Learned policy alone", h["alone"], h["alone_pen"], C_LEARN,
                    "no solver calls"),
                   ("Learned policy + solver", h["hybrid"], h["hybrid_pen"], C_LEARN,
                    f"solver only on doubtful rests, {h['calls']:.0f} calls per route"),
                   ("Look-ahead", h["la"], h["la_pen"], C_LA,
                    "solver at every stop, about 72 calls per route")],
              "(b) Extra time over the best schedule in hindsight", " %",
              1.45 * max(h["alone_pen"], h["hybrid_pen"], h["la_pen"]))
    save(plt, fig, "fig_residual")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    data = recompute() if args.recompute or not os.path.exists(CACHE) else json.load(open(CACHE))
    if "share_v2" not in data or "gap_pen" not in data.get("base", {}):
        data = recompute()
    if "mixed_final" not in data or args.recompute:
        print("[fig] re-driving the policies on the mixed test routes (final protocol) ...",
              flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            data["mixed_final"] = mixed_final()
        with open(CACHE, "w") as fh:
            json.dump(data, fh, indent=1)
        print("[fig] mixed_final:", json.dumps({k: v for k, v in data["mixed_final"].items()}),
              flush=True)
    plt = setup()
    fig_base(plt, data)
    fig_mixed(plt, data)
    fig_residual(plt, data)


if __name__ == "__main__":
    main()
