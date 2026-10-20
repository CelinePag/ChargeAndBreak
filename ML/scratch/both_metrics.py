"""
both_metrics.py -- every result of RESULTS_OVERVIEW.md, with and without the
time-window penalty (2026-10-20)
============================================================================
Measure: the manuscript's gap to the hindsight oracle, in %.
  no penalty : (duration - oracle duration) / oracle duration
  penalty    : (duration + 0.5 h * missed windows - oracle's) / oracle's
"vs LA" = model gap minus LA gap, in percentage points, paired by route:
each (training seed, route) run is paired with the LA's run on that route,
the seeds are averaged per route, then mean +/- se over routes.

Gap code is the project's own: paper_link.gaps (base, length), ood_eval.gap
via phys_eval (physics, learning curve), and mixed_eval's oracle with t0 = 8 h
(mixed, hybrid; certified oracles only, as fig_results.py).

    python ML/scratch/both_metrics.py            # prints markdown tables
"""
from __future__ import annotations

import collections
import contextlib
import glob
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ML, "code"))
RES = os.path.join(ML, "results")
OUT = os.path.join(RES, "both_metrics.json")

with contextlib.redirect_stdout(io.StringIO()):
    import paper_link as pl
    import phys_eval as pe
    import mixed_eval as me

BETA = 0.5
ALL = {}


# ── shared summary ───────────────────────────────────────────────────────────

def summarise(runs, la):
    """runs: {route: [run, ...]} over training seeds, run = dict(ok, g, gp, tw).
    la: {route: dict(ok, g, gp, tw)}.  Paired on routes where the LA completed
    and at least one seed did."""
    d, dp, g, gp, tw, tw_la = [], [], [], [], 0.0, 0
    broken = sum(1 for rs in runs.values() for r in rs if not r["ok"])
    nruns = sum(len(rs) for rs in runs.values())
    for k, rs in runs.items():
        l = la.get(k)
        ok = [r for r in rs if r["ok"] and r["g"] is not None]
        if not ok or not l or not l["ok"] or l["g"] is None:
            continue
        d.append(np.mean([r["g"] - l["g"] for r in ok]))
        dp.append(np.mean([r["gp"] - l["gp"] for r in ok]))
        g.append(np.mean([r["g"] for r in ok]))
        gp.append(np.mean([r["gp"] for r in ok]))
        tw += np.mean([r["tw"] for r in ok])
        tw_la += l["tw"]
    if not d:
        return None
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return dict(n=len(d), gap=float(np.mean(g)), gap_pen=float(np.mean(gp)),
                vs=float(np.mean(d)), vs_se=se(d), vs_pen=float(np.mean(dp)),
                vs_pen_se=se(dp), tw=float(tw), tw_la=int(tw_la),
                broken=broken, runs=nruns)


def la_summary(la, keys):
    v = [la[k] for k in keys if k in la and la[k]["ok"] and la[k]["g"] is not None]
    return dict(n=len(v), gap=float(np.mean([x["g"] for x in v])),
                gap_pen=float(np.mean([x["gp"] for x in v])),
                tw=int(sum(x["tw"] for x in v)))


def table(title, rows, la_line=None, note=None):
    print(f"\n### {title}\n")
    if la_line:
        print(la_line + "\n")
    print("| Model | Trained on | Routes | vs LA, no penalty | vs LA, with penalty "
          "| Missed windows (model / LA) | Rules broken |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for name, trained, s in rows:
        if s is None:
            print(f"| {name} | {trained} | – | – | – | – | – |")
            continue
        print(f"| {name} | {trained} | {s['n']} | {s['vs']:+.2f} ± {s['vs_se']:.2f} "
              f"| {s['vs_pen']:+.2f} ± {s['vs_pen_se']:.2f} | {s['tw']:.0f} / {s['tw_la']} "
              f"| {s['broken']} / {s['runs']} |")
    if note:
        print(f"\n{note}")
    ALL[title] = {"la": la_line, "rows": [(n, t, s) for n, t, s in rows]}


# ── 1. base case ─────────────────────────────────────────────────────────────

def base_runs(tags):
    runs = collections.defaultdict(list)
    for t in tags:
        p = os.path.join(RES, f"eval_{t}_test.json")
        if not os.path.exists(p):
            return None
        for r in json.load(open(p)):
            orc = pl.oracle_facts(r["instance"])
            if orc is None:
                continue
            t0 = _t0(r["instance"])
            if r.get("duration_h") is None:
                runs[r["instance"]].append(dict(ok=False, g=None, gp=None, tw=0))
                continue
            g, gp = pl.gaps(r["duration_h"], r.get("tw_misses", 0), r["duration_h"] + t0, orc)
            runs[r["instance"]].append(dict(ok=True, g=g, gp=gp, tw=r.get("tw_misses", 0)))
    return runs


_T0 = {}


def _t0(inst):
    if inst not in _T0:
        f = pl._latest(os.path.join(pl.SOL, f"{inst}_LA_MIPTAIL_*.json"))
        t0 = 8.0
        if f:
            s = json.load(open(f))
            if s.get("duration_h") is not None and s.get("sim_arrival_h"):
                t0 = s["sim_arrival_h"] - s["duration_h"]
        _T0[inst] = t0
    return _T0[inst]


def base_baselines(insts):
    out = {"LA_MIPTAIL": {}, "GREEDY": {}}
    for i in insts:
        orc = pl.oracle_facts(i)
        if orc is None:
            continue
        for m in out:
            f = pl._latest(os.path.join(pl.SOL, f"{i}_{m}_*.json"))
            if not f:
                continue
            s = json.load(open(f))
            met = s.get("metrics", {})
            tw = int(met.get("tw_n_misses", 0))
            if met.get("run_infeasible") or s.get("duration_h") is None:
                out[m][i] = dict(ok=False, g=None, gp=None, tw=tw)
                continue
            g, gp = pl.gaps(s["duration_h"], tw, s.get("sim_arrival_h"), orc)
            out[m][i] = dict(ok=True, g=g, gp=gp, tw=tw)
    return out


def seeds(fmt, n=3):
    return [fmt.format(s) for s in range(n)]


def section_base():
    groups = [
        ("PyTorch network", "base routes", seeds("tmlp_F95_split_list_s{}_g99sr")),
        ("Trees", "base routes", seeds("gbt_F95_base_s{}_g99sr")),
        ("Classifier", "base routes", seeds("clf_F77_base_s{}_g99sr")),
        ("scikit-learn network", "base routes", seeds("mlp_F95_base_s{}_g99sr")),
        ("Greedy rule (reference)", "–", None),
        ("PyTorch network, no shield", "base routes", seeds("tmlp_F95_split_list_s{}_nsN")),
        ("Trees, no shield", "base routes", seeds("gbt_F95_base_s{}_nsN")),
        ("PyTorch network, nominal features at driving time*", "base routes",
         seeds("tmlp_F95_split_list_s{}_g99srN")),
        ("Trees, nominal features at driving time*", "base routes", seeds("gbt_F95_base_s{}_g99srN")),
    ]
    runs = {nm: base_runs(t) for nm, _, t in groups if t}
    insts = sorted(set().union(*[set(r) for r in runs.values() if r]))
    bl = base_baselines(insts)
    la = bl["LA_MIPTAIL"]
    rows = []
    for nm, tr, t in groups:
        if t is None:
            rows.append((nm, tr, summarise({k: [v] for k, v in bl["GREEDY"].items() if k in insts}, la)))
        else:
            rows.append((nm, tr, summarise(runs[nm], la) if runs[nm] else None))
    L = la_summary(la, insts)
    gr = summarise({k: [v] for k, v in bl["GREEDY"].items() if k in insts}, la)
    s0 = rows[0][2]
    line = (f"Gap to the hindsight oracle, mean over the {L['n']} routes: LA {L['gap']:.2f} % "
            f"without penalty, {L['gap_pen']:.2f} % with it; PyTorch network {s0['gap']:.2f} / "
            f"{s0['gap_pen']:.2f} %; Greedy {gr['gap']:.2f} / {gr['gap_pen']:.2f} %.")
    table("1. Base case, 125 test routes, 3 training seeds", rows, line,
          "\\* the train/serve fix from the review: evaluated on test only, never validated.")

    # single-model ablations (guard 0.95, no spread room, as they were run)
    abl = [("Trees (reference for the next row)", "base routes, F91", ["gbt_F91_base_g95"]),
           ("Trees, raw costs instead of per-stop differences", "base routes, F91", ["gbt_F91_rawcost_g95"]),
           ("scikit-learn network (reference)", "base routes, F91", ["mlp_F91_base_g95"]),
           ("scikit-learn network, raw costs", "base routes, F91", ["mlp_F91_rawcost_g95"]),
           ("PyTorch, first version (no ranking loss)", "base routes", ["tmlp_F95_base_s0_g95"]),
           ("PyTorch with ranking loss (reference)", "base routes", ["tmlp_F95_split_list_s0_g95"])]
    rows = [(nm, tr, summarise(base_runs(t), la) if base_runs(t) else None) for nm, tr, t in abl]
    table("1b. Base-case ablations, one model each, older shield (guard 0.95, no spread-room check)", rows)

    # learning curve (trees, seed 1, guard 0.99 + spread room)
    lc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in pe.read_store(os.path.join(RES, "lc_g99sr.jsonl")):
        lc[r["tag"]][r["instance"]].append(
            dict(ok=r["completed"], g=r["gap"], gp=r["gap_pen"], tw=r["tw"]))
    rows = []
    for n in (2, 4, 7, 12, 19):
        tag = f"gbt_F95_lc{n}_s1"
        rows.append((f"Trees, {n} route seeds per family", f"{n} seeds", summarise(lc[tag], la)))
    table("1c. Learning curve (trees, one training seed)", rows)


# ── 2. route length ──────────────────────────────────────────────────────────

def section_length():
    groups = [("Trees", "short + medium", seeds("gbt_F95_base_SM_s{}_g99sr")),
              ("Classifier", "short + medium", seeds("clf_F77_base_SM_s{}_g99sr")),
              ("scikit-learn network", "short + medium", seeds("mlp_F95_base_SM_s{}_g99sr"))]
    for scope, keep in (("30 long test routes (seeds 22-25)", lambda i: int(i.rsplit("_", 1)[1]) >= 22),
                        ("all 239 long routes with an LA run (none seen in training)", lambda i: True)):
        rows, la = [], None
        for nm, tr, tags in groups:
            runs = collections.defaultdict(list)
            for t in tags:
                for r in json.load(open(os.path.join(RES, f"eval_{t}_longall.json"))):
                    i = r["instance"]
                    if not keep(i) or pl.oracle_facts(i) is None:
                        continue
                    orc, t0 = pl.oracle_facts(i), _t0(i)
                    if r.get("duration_h") is None:
                        runs[i].append(dict(ok=False, g=None, gp=None, tw=0))
                    else:
                        g, gp = pl.gaps(r["duration_h"], r["tw_misses"], r["duration_h"] + t0, orc)
                        runs[i].append(dict(ok=True, g=g, gp=gp, tw=r["tw_misses"]))
            if la is None:
                bl = base_baselines(sorted(runs))
                la, gr = bl["LA_MIPTAIL"], bl["GREEDY"]
            rows.append((nm, tr, summarise(runs, la)))
        rows.append(("Greedy rule (reference)", "–", summarise({k: [v] for k, v in gr.items()}, la)))
        L = la_summary(la, list(la))
        table(f"2. Trained on short + medium routes, tested on long routes: {scope}", rows,
              f"LA gap to the oracle on these routes: {L['gap']:.2f} % without penalty, "
              f"{L['gap_pen']:.2f} % with it.")


# ── 3. physics, uniform along the route ──────────────────────────────────────

def section_physics():
    rows_all = pe.read_store(os.path.join(RES, "phys_g99sr.jsonl"))
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in rows_all:
        by[r["axis"]][r["tag"]][r["instance"]].append(
            dict(ok=r["completed"], g=r["gap"], gp=r["gap_pen"], tw=r["tw"]))
    groups = [("Trees", "base routes", ["gbt_F95_base_s1"]),
              ("Trees", "all uniform variants", ["gbt_F95_phys_s1"]),
              ("Trees", "all uniform variants but the tested one", ["gbt_F95_phys_LO{ax}_s1"]),
              ("PyTorch network", "base routes", seeds("tmlp_F95_split_list_s{}")),
              ("PyTorch network", "all uniform variants", ["tmlp_F95_phys_split_list_s0"]),
              ("PyTorch network", "all uniform variants but the tested one",
               ["tmlp_F95_phys_LO{ax}_split_list_s0"]),
              ("Greedy rule (reference)", "–", None)]
    for ax in ("base", "kw150", "kw700", "kw1000", "kwh300", "kwh700", "kwh900", "cs30", "cs100"):
        if ax not in by:
            continue
        names = sorted({i for t in by[ax].values() for i in t})
        with contextlib.redirect_stdout(io.StringIO()):
            bl = pe.baselines(ax, names)
        conv = lambda d: {k: dict(ok=v["completed"], g=v["gap"], gp=v["gap_pen"], tw=v["tw"])
                          for k, v in d.items()}
        la, gr = conv(bl["LA"]), conv(bl["Greedy"])
        rows = []
        for nm, tr, tags in groups:
            if tags is None:
                rows.append((nm, tr, summarise({k: [v] for k, v in gr.items()}, la)))
                continue
            if ax == "base" and "{ax}" in tags[0]:
                continue
            tags = [t.format(ax=ax) for t in tags]
            if not all(t in by[ax] for t in tags):
                continue
            runs = collections.defaultdict(list)
            for t in tags:
                for k, v in by[ax][t].items():
                    runs[k] += v
            rows.append((nm, tr, summarise(runs, la)))
        L = la_summary(la, names)
        table(f"3. Uniform physics: {pe.AXES[ax][2]}", rows,
              f"LA gap to the oracle: {L['gap']:.2f} % without penalty, {L['gap_pen']:.2f} % with it "
              f"({L['n']} routes).")


# ── 4. mixed chargers along the route, and the hybrid ────────────────────────

_ORC = {}


def mixed_orc(v, n):
    if (v, n) not in _ORC:
        _ORC[(v, n)] = me.read_oracle(v, n)
    return _ORC[(v, n)]


def mgap(dur, tw, orc):
    if dur is None or orc is None or not ((orc["gap"] or 0) <= 0.01):
        return None, None
    od = orc["ta_N"] - 8.0
    return (100.0 * (dur - od) / od,
            100.0 * (dur + BETA * tw - od - orc["pen"]) / (od + orc["pen"]))


def mrun(v, n, ok, dur, tw):
    if not ok:
        return dict(ok=False, g=None, gp=None, tw=tw)
    g, gp = mgap(dur, tw, mixed_orc(v, n))
    return dict(ok=g is not None, g=g, gp=gp, tw=tw) if g is not None else None


def mixed_la(variants):
    la = {}
    for v in variants:
        for p in glob.glob(os.path.join(ML, "instances_mixed", v, "*.json")):
            n = os.path.splitext(os.path.basename(p))[0]
            if int(n.split("__")[0].rsplit("_", 1)[1]) < 22:
                continue
            r = me.la_row(v, n)
            if r is None:
                continue
            x = mrun(v, n, r["completed"], r["duration_h"], r["tw"])
            if x is not None:
                la[(v, n)] = x
    return la


def section_mixed():
    rows_all = me.read_rows(me.store_path(0.99, True))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows_all:
        if r["variant"] not in ("pmix", "mix"):
            continue
        x = mrun(r["variant"], r["instance"], r["completed"], r["duration_h"], r["tw"])
        if x is not None:
            by[r["method"]][(r["variant"], r["instance"])].append(x)
    s3 = lambda lab: [lab, f"{lab} (s1)", f"{lab} (s2)"]
    groups = [
        ("Greedy rule (reference)", "–", ["Greedy"]),
        ("Trees", "base routes", ["trees, base only"]),
        ("Trees", "all uniform variants", ["trees, all physics"]),
        ("Trees + power input", "all uniform variants", ["trees, all physics + power"]),
        ("Trees + power input", "all uniform + 121 power-mixed routes", ["trees + power, + 121 mixed routes"]),
        ("Trees + power input", "all uniform + DAgger corrections", ["trees + power, + DAgger"]),
        ("PyTorch network", "base routes", ["torch split+list, base only"]),
        ("PyTorch network", "all uniform variants", ["torch split+list, all physics"]),
        ("PyTorch + charger inputs", "all uniform variants", ["torch split+list, T inputs, all phys"]),
        ("ChargerNet", "all uniform variants", ["ChargerNet, all physics"]),
        ("ChargerNet", "all uniform + 47 power-mixed routes", ["ChargerNet, + pilot"]
         + ["ChargerNet, + pilot (s1)", "ChargerNet, + pilot (s2)"]),
        ("PyTorch + charger inputs", "all uniform + 47 power-mixed routes",
         ["torch T inputs, + pilot", "torch T, + pilot (s1)", "torch T, + pilot (s2)"]),
        ("PyTorch + charger inputs", "all uniform + 121 power-mixed routes", s3("torch T, + 121 mixed routes")),
        ("PyTorch + charger inputs", "same, mixed routes weighted ×5", s3("torch T, + 121 mixed x5")),
        ("PyTorch + charger inputs", "121 power-mixed routes only", s3("torch T, mixed only")),
        ("PyTorch + charger inputs", "all uniform + DAgger, 1 round", s3("torch T, + DAgger")),
        ("PyTorch + charger inputs", "all uniform + DAgger, 2 rounds", s3("torch T, + DAgger x2")),
        ("PyTorch + charger inputs", "all uniform + 238 mixed routes", s3("torch T, + pmix + mix routes")),
        ("PyTorch + charger inputs", "those 238 mixed routes only", s3("torch T, pmix + mix only")),
        ("PyTorch + charger inputs", "all uniform + 238 mixed routes ×5", s3("torch T, + pmix + mix x5")),
    ]
    for vs, title in ((("pmix",), "power varies along the route"),
                      (("mix",), "power and spacing vary along the route"),
                      (("pmix", "mix"), "both kinds together")):
        la = mixed_la(vs)
        rows = []
        for nm, tr, labs in groups:
            runs = collections.defaultdict(list)
            for lab in labs:
                for k, v in by.get(lab, {}).items():
                    if k[0] in vs:
                        runs[k] += v
            rows.append((nm, tr, summarise(runs, la) if runs else None))
        if vs == ("pmix", "mix"):
            for cfg, desc in (("m10", "rest vs no rest within 10 min"),
                              ("m30", "within 30 min"), ("m60", "within 60 min"),
                              ("f5", "every stop of the last 5 h")):
                runs = collections.defaultdict(list)
                calls = []
                for line in open(os.path.join(RES, f"hybrid_{cfg}.jsonl")):
                    r = json.loads(line)
                    x = mrun(r["variant"], r["instance"], r["completed"], r["duration_h"], r["tw"])
                    if x is not None:
                        runs[(r["variant"], r["instance"])].append(x)
                        calls.append(r["n_calls"])
                rows.append((f"Hybrid {cfg}: LA called on {desc} ({np.mean(calls):.1f} calls/route)",
                             "238 mixed routes model", summarise(runs, la)))
        L = la_summary(la, list(la))
        table(f"4. Mixed chargers, {title}", rows,
              f"LA gap to the oracle: {L['gap']:.2f} % without penalty, {L['gap_pen']:.2f} % with it "
              f"({L['n']} routes with a certified oracle).")


if __name__ == "__main__":
    section_base()
    section_length()
    section_physics()
    section_mixed()
    with open(OUT, "w") as fh:
        json.dump(ALL, fh, indent=1)
    print(f"\n[wrote {OUT}]")
