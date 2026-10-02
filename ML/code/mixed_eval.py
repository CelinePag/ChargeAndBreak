"""
mixed_eval.py — trained on one charger type per route, tested on routes that mix them
====================================================================================
The routes come from mixed_instances.py (pmix: a power per charger; dmix:
spacing changes along the route; mix: both), each paired with its uniform
original (route seeds 22-25, short and medium routes).

src reads per-charger curves natively since 2026-10-01 (instance field
"TbarK"; MILP.py "Per-charger charging curves", BEHDV.charging_curve_at),
so every part below runs src's own code, unpatched:

  oracle   src's oracle_solve, with ONE CHARGING CURVE PER CHARGER through
           the MILP's pwl_tc.  Solutions go to ML/solutions_mixed/, never to
           solutions/.  (The first 96 were solved on 2026-09-30 with a
           runtime patch of that one constraint, before src supported it; the
           native oracle reproduces them — same objectives within the 0.5 %
           MIP gap — so they are kept.)
  greedy   src's greedy_decision in the same loop as run_greedy (run_greedy
           itself writes logs into the main tree).
  drive    the students, through policy_core.run_student.
  la       not here: run_la_mixed.py runs src's LA on these routes.

    python ML/code/mixed_eval.py oracle [--time-limit 600]
    python ML/code/mixed_eval.py drive [--jobs 4]
    python ML/code/mixed_eval.py report
    python ML/code/mixed_eval.py charging --variants pmix   # where energy is taken
    python ML/code/mixed_eval.py la --variants pmix         # vs the LA (run_la_mixed.py)
    python ML/code/mixed_eval.py charging --variants pmix --la-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.instance_gen.instance_io import load_instance_json      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.abspath(os.path.join(HERE, ".."))
INST = os.path.join(ML, "instances_mixed")
SOL = os.path.join(ML, "solutions_mixed")
RESULTS = os.path.join(ML, "results")
VARIANTS = ("pmix", "dmix", "mix")
BETA = 0.5
# what every stored Greedy run used (its solution JSON records them)
GREEDY_GUARD_Q, GREEDY_SAFETY = 0.95, 0.1

# (kind, tag, label)
MODELS = [("gbt", "gbt_F95_base_s1", "trees, base only"),
          ("gbt", "gbt_F95_phys_s1", "trees, all physics"),
          ("gbt", "gbt_P102_phys_s1", "trees, all physics + power"),
          # the PyTorch model chosen on the stop split (2026-10-01), base only
          ("torch", "tmlp_F95_split_list_s0", "torch split+list, base only"),
          # the same recipe on every physics value (2026-10-02): charger powers
          # 150-1000 kW are then inside the training range
          ("torch", "tmlp_F95_phys_split_list_s0", "torch split+list, all physics"),
          # direction B (2026-10-02), all physics, the charger-token inputs (set T):
          # no structure / ChargerNet / ChargerNet whose g also sees this
          # charger's speed.  Chosen on --set val; the test routes are run once.
          ("torch", "tmlp_T144_phys_split_list_s0", "torch split+list, T inputs, all phys"),
          ("torch", "tmlp_T144_phys_charger_s0", "ChargerNet, all physics"),
          ("torch", "tmlp_T144_phys_chargerG_s0", "ChargerNet, g sees speed, all phys"),
          # the data answer (2026-10-03): the 47 pmix pilot routes added to training
          ("gbt", "gbt_P102_physpmix_s1", "trees + power, + pilot"),
          ("gbt", "gbt_P102_physpmix_w5_s1", "trees + power, + pilot x5"),
          ("torch", "tmlp_T144_physpmix_split_list_s0", "torch T inputs, + pilot"),
          ("torch", "tmlp_T144_physpmix_charger_s0", "ChargerNet, + pilot"),
          # chosen on the validation routes (2026-10-03): two more seeds for the test
          ("torch", "tmlp_T144_physpmix_charger_s1", "ChargerNet, + pilot (s1)"),
          ("torch", "tmlp_T144_physpmix_charger_s2", "ChargerNet, + pilot (s2)"),
          # more mixed data vs DAgger at equal LA time (2026-10-03): the PyTorch
          # model with the per-charger inputs, 3 seeds per configuration
          ("torch", "tmlp_T144_physpmix_split_list_s1", "torch T, + pilot (s1)"),
          ("torch", "tmlp_T144_physpmix_split_list_s2", "torch T, + pilot (s2)"),
          ("torch", "tmlp_T144_physpmixall_split_list_s0", "torch T, + all mixed data"),
          ("torch", "tmlp_T144_physpmixall_split_list_s1", "torch T, + all mixed data (s1)"),
          ("torch", "tmlp_T144_physpmixall_split_list_s2", "torch T, + all mixed data (s2)"),
          ("torch", "tmlp_T144_physdg_split_list_s0", "torch T, + DAgger"),
          ("torch", "tmlp_T144_physdg_split_list_s1", "torch T, + DAgger (s1)"),
          ("torch", "tmlp_T144_physdg_split_list_s2", "torch T, + DAgger (s2)"),
          ("gbt", "gbt_P102_physpmixall_s1", "trees + power, + all mixed data"),
          ("gbt", "gbt_P102_physdg_s1", "trees + power, + DAgger")]


# which route set: "test" (seeds 22-25, ML/instances_mixed/<v>/) or "val"
# (seeds 20-21, ML/instances_mixed/val/<v>/, to choose models on); set by --set
ROUTE_SET = "test"


def instances(variants=VARIANTS, uniform=True):
    """(variant, name, path); 'uniform' = the originals they were built from."""
    out = []
    root = INST if ROUTE_SET == "test" else os.path.join(INST, ROUTE_SET)
    for v in variants:
        for p in sorted(glob.glob(os.path.join(root, v, "*.json"))):
            out.append((v, os.path.splitext(os.path.basename(p))[0], p))
    if uniform:
        seen = sorted({n.split("__")[0] for _, n, _ in out})
        out += [("uniform", n, os.path.join(_ROOT, "instances", n + ".json")) for n in seen]
    return out


def load(path):
    fd, D, E, cv = load_instance_json(path)
    fd["_horizon_h"] = 24.0
    return fd, D, E, cv


# ── oracle ───────────────────────────────────────────────────────────────────

def oracle_path(variant, name):
    return os.path.join(SOL, variant, f"oracle_{name}.json")


def solve_oracle(variant, name, path, time_limit):
    from src.methods.oracle import oracle_solve
    fd, D, _E, _cv = load(path)
    t0 = time.time()
    o = oracle_solve(fd, D, sim_results=None, time_limit=time_limit,
                     tee=False, verbose=False)
    o["wall_clock"] = time.time() - t0
    o.pop("D_actual", None)
    return o


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def cmd_oracle(args):
    todo = [(v, n, p) for v, n, p in instances(args.variants.split(","), uniform=False)
            if not os.path.exists(oracle_path(v, n))]
    print(f"[oracle] {len(todo)} to solve (time limit {args.time_limit}s)", flush=True)
    for i, (v, n, p) in enumerate(todo):
        o = solve_oracle(v, n, p, args.time_limit)
        os.makedirs(os.path.join(SOL, v), exist_ok=True)
        with open(oracle_path(v, n), "w") as fh:
            json.dump(_json_safe(o), fh)
        print(f"  {i+1}/{len(todo)} {n}: {o.get('stop_reason')} obj={o.get('obj')} "
              f"gap={o.get('gap')} {o['wall_clock']:.0f}s", flush=True)


def read_oracle(variant, name):
    """Stored oracle as ood_eval.oracle reads it: arrival and window penalty."""
    p = (oracle_path(variant, name) if variant != "uniform"
         else os.path.join(_ROOT, "solutions", "basecase", f"oracle_{name}.json"))
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        o = json.load(fh)
    sol = o.get("sol") or []
    if not o.get("feasible") or not sol or o.get("obj") is None:
        return None
    ta_N = float(sol[-1]["ta"])
    rests = int(sum(round(s.get("rho1", 0)) + round(s.get("rho2", 0)) for s in sol))
    return dict(ta_N=ta_N, pen=float(o["obj"]) - ta_N, gap=o.get("gap"),
                rests=rests, stop_reason=o.get("stop_reason"))


# ── greedy, with the charger's own curve ─────────────────────────────────────

def run_greedy_ml(fd, D_real, E_real, cv, return_vehicle=False):
    """run_greedy's loop (unsupervised, as every stored Greedy run), minus its
    file output.  greedy and BEHDV read each charger's own curve themselves."""
    from src.methods.greedy import _greedy_durations, greedy_decision
    from src.simulation.BEHDV import BEHDV
    from features import Precomp
    pre, N = Precomp(fd), int(fd["N"])
    veh = BEHDV(fd, strict_spread=True)
    rests = 0
    for stop in range(N):
        action, _ = greedy_decision(fd, stop, veh, cv=cv,
                                    guard_quantile=GREEDY_GUARD_Q,
                                    safety_buffer_frac=GREEDY_SAFETY)
        y = action.get("y", 0)
        brk = action.get("break_type") or "---"
        rst = action.get("rest_type") or "---"
        rests += int(rst in ("r1", "r2"))
        dur = _greedy_durations(fd, stop, action, veh)
        mock = dict(feasible=True, sol=[dict(
            i=0, taub=dur["taub"], tauc=dur["tauc"], taur=dur["taur"],
            tauq=dur["tauq"], y=y, b45=int(brk == "b45"), b15=int(brk == "b15"),
            b30=int(brk == "b30"), rho1=int(rst == "r1"), rho2=int(rst == "r2"),
            is_C=(stop in pre.C), is_K=(stop in pre.K))])
        veh.advance(action=action, D_next=float(D_real[stop]),
                    E_next=float(E_real[stop]), milp_sol=mock)
        if veh.is_halted:
            break
    if return_vehicle:
        return veh
    done = (not veh.is_halted) and veh.stop >= N
    T0 = float(fd.get("T_START", 8.0))
    return dict(duration_h=(veh.t_arr - T0) if done else None,
                route_completed=bool(done), tw_misses=len(veh.tw_misses),
                rests=rests, halt_reason=veh.halt_reason)


# ── driving ──────────────────────────────────────────────────────────────────

_POL = {}


def _drive(job):
    variant, name, path, method, kind, tag, guard_q, spread_room = job
    fd, D, E, cv = load(path)
    if method == "Greedy":
        r = run_greedy_ml(fd, D, E, cv)
    else:
        from policy_core import load_policy, run_student
        if tag not in _POL:
            _POL[tag] = load_policy(kind, tag, guard_q=guard_q, spread_room=spread_room)
        r = run_student(fd, D, E, _POL[tag], cv=cv)
        r["rests"] = sum(1 for k in r["actions"] if k.endswith(("_r1", "_r2")))
    return dict(variant=variant, instance=name, method=method,
                completed=r["route_completed"], duration_h=r["duration_h"],
                tw=r["tw_misses"], rests=r["rests"],
                violation=r.get("halt_reason"))


def store_path(guard_q, spread_room):
    tail = "" if ROUTE_SET == "test" else f"_{ROUTE_SET}"
    return os.path.join(RESULTS, f"mixed{tail}_g{round(100 * guard_q)}"
                                 f"{'sr' if spread_room else ''}.jsonl")


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def cmd_drive(args):
    store = store_path(args.guard_q, args.spread_room)
    done = {(r["method"], r["instance"]) for r in read_rows(store)}
    methods = [("Greedy", None, None)] + [
        (lab, kind, tag) for kind, tag, lab in MODELS
        if os.path.exists(os.path.join(ML, "models", f"{tag}_meta.json"))]
    jobs = [(v, n, p, m, kind, tag, args.guard_q, args.spread_room)
            for v, n, p in instances(args.variants.split(","))
            for m, kind, tag in methods if (m, n) not in done]
    print(f"[drive] {len(jobs)} runs -> {store}", flush=True)
    t0 = time.time()
    with open(store, "a") as fh:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for i, row in enumerate(pool.imap_unordered(_drive, jobs, 2)):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    cmd_report(args)


# ── report ───────────────────────────────────────────────────────────────────

def _gap(r, orc):
    if r["duration_h"] is None or orc is None:
        return None
    od = orc["ta_N"] - 8.0
    return 100.0 * (r["duration_h"] - od) / od if od > 0 else None


def cmd_report(args):
    rows = read_rows(store_path(args.guard_q, args.spread_room))
    orc = {(r["variant"], r["instance"]): None for r in rows}
    for k in orc:
        orc[k] = read_oracle(*k)
    methods = ["Greedy"] + [lab for _k, _t, lab in MODELS]
    print("\n" + "=" * 104)
    print("MIXED CHARGERS - gap to the oracle (%), short+medium routes, seeds 22-25; "
          "paired = mean +/- se of (method - Greedy) in pp")
    print("=" * 104)
    print(f"{'routes':10s} {'method':30s} {'n':>4s} {'inf':>4s} {'median':>8s} {'mean':>8s} "
          f"{'TW':>5s} {'+rest':>6s}   {'vs Greedy':>18s}   {'orc gap':>8s}")
    for v in ("uniform",) + VARIANTS:
        sub = [r for r in rows if r["variant"] == v]
        if not sub:
            continue
        print("-" * 104)
        per = {m: {r["instance"]: r for r in sub if r["method"] == m} for m in methods}
        og = [o["gap"] for (vv, _), o in orc.items() if vv == v and o and o.get("gap") is not None]
        for m in methods:
            runs = per[m]
            if not runs:
                continue
            g = {k: _gap(r, orc[(v, k)]) for k, r in runs.items() if r["completed"]}
            g = {k: x for k, x in g.items() if x is not None}
            extra = sum(1 for k, r in runs.items() if r["completed"] and orc[(v, k)]
                        and r["rests"] > orc[(v, k)]["rests"])
            gr = per["Greedy"]
            d = [g[k] - _gap(gr[k], orc[(v, k)]) for k in g
                 if m != "Greedy" and k in gr and gr[k]["completed"]
                 and _gap(gr[k], orc[(v, k)]) is not None]
            pl = (f"{np.mean(d):+.2f} +/- {np.std(d, ddof=1)/np.sqrt(len(d)):.2f}"
                  if len(d) > 1 else "")
            vals = list(g.values())
            print(f"{v if m == 'Greedy' else '':10s} {m:30s} {len(runs):4d} "
                  f"{sum(1 for r in runs.values() if not r['completed']):4d} "
                  f"{np.median(vals) if vals else float('nan'):+8.2f} "
                  f"{np.mean(vals) if vals else float('nan'):+8.2f} "
                  f"{sum(r['tw'] for r in runs.values()):5d} {extra:6d}   {pl:>18s}   "
                  f"{(100*np.median(og)) if (og and m == 'Greedy') else float('nan'):8.2f}")


# ── the LA teacher (runs made by run_la_mixed.py) ────────────────────────────

LA_ROOT = os.path.join(ML, "la_mixed")
REST_DWELL_H = 8.9          # a stop dwell this long is a daily rest


def _latest(paths_):
    return sorted(paths_)[-1] if paths_ else None


def la_solution(variant, name):
    """The LA's solution JSON on one route: run_la_mixed.py's output for a
    mixed route, the stored base-case teacher run for its uniform original."""
    if variant == "uniform":
        return _latest(glob.glob(os.path.join(_ROOT, "solutions", "basecase",
                                              f"{name}_LA_MIPTAIL_*.json")))
    return _latest(glob.glob(os.path.join(LA_ROOT, "solutions", "**",
                                          f"{name}_LA_*.json"), recursive=True))


def la_row(variant, name):
    """The LA's run as a mixed_eval row, or None if it has not run.  Rests are
    read from the dwells: its stored actions are what the look-ahead SELECTED,
    which the nominal re-solve may have moved."""
    p = la_solution(variant, name)
    if p is None:
        return None
    with open(p) as fh:
        s = json.load(fh)
    m = s.get("metrics", {})
    done = (not m.get("run_infeasible")) and s.get("duration_h") is not None
    tr, td = s.get("sim_trajectory") or [], s.get("td_list") or []
    rests = sum(1 for k in range(min(len(tr), len(td)))
                if float(td[k]) - float(tr[k]["t_arr"]) >= REST_DWELL_H)
    return dict(variant=variant, instance=name, method="LA", completed=done,
                duration_h=s["duration_h"] if done else None,
                tw=int(m.get("tw_n_misses", 0)), rests=rests,
                wall_h=(s.get("wall_clock_s_full_route") or s.get("wall_clock_s") or 0) / 3600,
                trajectory=tr)


def la_routes(variant):
    """Mixed routes of `variant` the LA has finished."""
    return sorted(n for _, n, _ in instances([variant], uniform=False)
                  if la_solution(variant, n))


def cmd_la(args):
    """Every method on exactly the routes the LA has run: the uniform original
    and the mixed route, and the change between them, paired by route."""
    v = args.variants.split(",")[0]
    names = la_routes(v)
    if not names:
        print(f"[la] no LA run on {v} yet")
        return
    rows = read_rows(store_path(args.guard_q, args.spread_room))
    methods = ["LA", "Greedy"] + [lab for _k, _t, lab in MODELS]
    G = {}                                          # (method, variant, base) -> gap
    for name in names:
        base = name.split("__")[0]
        for vv, nm in (("uniform", base), (v, name)):
            orc = read_oracle(vv, nm)
            cand = [r for r in rows if r["variant"] == vv and r["instance"] == nm]
            la = la_row(vv, nm)
            if la:
                cand.append(la)
            for r in cand:
                g = _gap(r, orc) if r["completed"] else None
                if g is not None and orc and (orc["gap"] or 0) <= 0.01:
                    G[(r["method"], vv, base)] = g
    bases = [n.split("__")[0] for n in names]
    print(f"\n{v}: {len(names)} routes with an LA run; gap to the oracle (%), "
          f"certified oracles only; change = {v} minus uniform, same route")
    print(f"{'method':30s} {'uniform':>9s} {v:>9s} {'change (pp)':>18s} {'n':>3s}")
    for m in methods:
        u = [G[(m, 'uniform', b)] for b in bases if (m, 'uniform', b) in G]
        x = [G[(m, v, b)] for b in bases if (m, v, b) in G]
        d = [G[(m, v, b)] - G[(m, 'uniform', b)] for b in bases
             if (m, v, b) in G and (m, 'uniform', b) in G]
        se = np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"{m:30s} {np.mean(u) if u else float('nan'):+9.2f} "
              f"{np.mean(x) if x else float('nan'):+9.2f} "
              f"{(np.mean(d) if d else float('nan')):+10.2f} +/- {se:.2f} {len(d):3d}")
    walls = [la_row(v, n)["wall_h"] for n in names]
    print(f"LA wall clock on {v}: {sum(walls):.1f} h for {len(names)} routes")


# ── where the energy is taken ────────────────────────────────────────────────

class _Spy:
    """A policy wrapper that keeps the vehicle, to read its history after."""
    def __init__(self, pol):
        self.pol, self.veh = pol, None

    def decide(self, fd, pre, stop, veh, cv):
        self.veh = veh
        return self.pol.decide(fd, pre, stop, veh, cv)


def _charged(veh, E_real):
    """{stop: kWh charged there}: departure energy (next arrival + the leg's
    realised energy) minus arrival energy."""
    st, ea = veh.stop_history, veh.e_arr_history
    return {st[k]: ea[k + 1] + float(E_real[st[k]]) - ea[k] for k in range(len(st) - 1)}


def energy_by_power(v, la_only=False, guard_q=0.99, spread_room=True):
    """{method: {kW: kWh charged}}, {method: {kW: charging stops}}, n routes.

    The oracle with hindsight chooses WHERE to charge; this shows whether an
    online method does too.  la_only restricts to the routes the LA has run
    and adds the LA.  Greedy and the students are re-driven (deterministic)."""
    import collections
    from features import charger_kw
    from policy_core import load_policy, run_student
    pols = [(lab, load_policy(kind, tag, guard_q=guard_q, spread_room=spread_room))
            for kind, tag, lab in MODELS
            if os.path.exists(os.path.join(ML, "models", f"{tag}_meta.json"))]
    E_by = collections.defaultdict(collections.Counter)
    n_by = collections.defaultdict(collections.Counter)
    routes = instances([v], uniform=False)
    if la_only:                           # the routes the LA has run, + the LA
        keep = set(la_routes(v))
        routes = [r for r in routes if r[1] in keep]
    for _, name, path in routes:
        fd, D, E, cv = load(path)
        kw = {k: int(round(charger_kw(fd, k))) for k in fd["K"]}
        la = la_row(v, name) if la_only else None
        if la:
            tr = la["trajectory"]
            for k in range(len(tr) - 1):
                st = int(tr[k]["stop"])
                c = float(tr[k + 1]["e_arr"]) + float(E[st]) - float(tr[k]["e_arr"])
                if st in kw and c > 1e-3:
                    E_by["LA"][kw[st]] += c
                    n_by["LA"][kw[st]] += 1
        with open(oracle_path(v, name)) as fh:
            o = json.load(fh)
        for s in o.get("sol") or []:
            if s["i"] in kw and s["ed"] - s["ea"] > 1e-3:
                E_by["oracle"][kw[s["i"]]] += s["ed"] - s["ea"]
                n_by["oracle"][kw[s["i"]]] += 1
        vehs = [("Greedy", run_greedy_ml(fd, D, E, cv, return_vehicle=True))]
        for lab, pol in pols:
            fd, D, E, cv = load(path)
            spy = _Spy(pol)
            run_student(fd, D, E, spy, cv=cv)
            vehs.append((lab, spy.veh))
        for lab, veh in vehs:
            for st, c in _charged(veh, E).items():
                if st in kw and c > 1e-3:
                    E_by[lab][kw[st]] += c
                    n_by[lab][kw[st]] += 1
    return ({m: dict(c) for m, c in E_by.items() if c},
            {m: dict(c) for m, c in n_by.items() if c}, len(routes))


def cmd_charging(args):
    """Share of the charged energy taken at each charger power, per method."""
    v = args.variants.split(",")[0]
    E_by, n_by, n_routes = energy_by_power(v, args.la_only, args.guard_q,
                                           args.spread_room)
    levels = sorted({k for c in E_by.values() for k in c})
    print()
    print(f"{v}: share of charged energy by charger power (charging stops), "
          f"{n_routes} routes")
    print(f"{'method':30s}" + "".join(f"{k:>13d} kW" for k in levels) + f"{'kWh':>9s}")
    labs = ["oracle"] + (["LA"] if "LA" in E_by else []) + ["Greedy"] + [
        lab for _k, _t, lab in MODELS if lab in E_by]
    for lab in labs:
        n_by.setdefault(lab, {})
        tot = sum(E_by[lab].values()) or 1.0
        print(f"{lab:30s}" + "".join(
            f"{100*E_by[lab].get(k, 0)/tot:9.1f}% ({n_by[lab].get(k, 0):3d})"
            for k in levels) + f"{tot:9.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["oracle", "drive", "report", "charging", "la"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--la-only", action="store_true",
                    help="charging: only the routes the LA has run, LA included")
    ap.add_argument("--time-limit", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--guard-q", type=float, default=0.99)
    ap.add_argument("--no-spread-room", dest="spread_room", action="store_false")
    ap.add_argument("--set", default="test", choices=["test", "val"],
                    help="val: the seed 20-21 routes (mixed_instances.py --split val), "
                         "results in their own store, to choose models on")
    ap.set_defaults(spread_room=True)
    args = ap.parse_args()
    global ROUTE_SET
    ROUTE_SET = args.set
    {"oracle": cmd_oracle, "drive": cmd_drive, "report": cmd_report,
     "charging": cmd_charging, "la": cmd_la}[args.cmd](args)


if __name__ == "__main__":
    main()
