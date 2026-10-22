"""
review_checks.py -- ex-post checks for the TRD referee report (round 1).

Reads STORED results only (no solver call, no simulation): the base-case runs
in solutions/, their oracle caches, the instance files, and the derived CSVs in
data_output/.  Every number quoted in the response document comes from here.

    python -m review.TRD_R1.review_checks            (from the repo root)

Writes review/TRD_R1/review_checks.json and prints a readable summary.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src import paths as P                                    # noqa: E402
from src import settings as S                                 # noqa: E402
from src.output_analysis import compile_solutions as cs       # noqa: E402

OUT = {}
METHODS = ["greedy", "RO", "2SP", "LA"]
ROUTES = ["short", "medium", "long"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def q(xs, p):
    return float(np.quantile(np.asarray(xs, float), p)) if xs else None


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return (None, None, None)
    m = st.mean(xs)
    se = st.stdev(xs) / math.sqrt(len(xs))
    return (m, m - 1.96 * se, m + 1.96 * se)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Base-case corpus, exactly as the reporting layer builds it
# ─────────────────────────────────────────────────────────────────────────────
rows = cs.load_solutions(P.solutions())
rows = [r for r in rows if "__" not in (r.get("instance") or "")
        and not r.get("supervised")
        and (r.get("variant") or None) is None
        and r.get("method") in METHODS]
rows, n_dup = cs._dedup_latest(rows)
cs._annotate_instance_tags(rows)
cs._annotate_gap_to_oracle(rows)
cs._annotate_outcome(rows)
base = [r for r in rows if r.get("route_class") in ROUTES]
by_m = defaultdict(list)
for r in base:
    by_m[r["method"]].append(r)

OUT["counts"] = {m: len(by_m[m]) for m in METHODS}
inst_all = sorted({r["instance"] for r in base})
OUT["n_instances"] = len(inst_all)
missing = {m: sorted(set(inst_all) - {r["instance"] for r in by_m[m]})
           for m in METHODS}
OUT["missing_runs"] = {m: (len(v), v[:12]) for m, v in missing.items()}
OUT["missing_LA_by_class"] = dict(Counter(
    i.split("_")[0] for i in missing["LA"]))

# parameters actually used
def _pc(m, keys):
    return {k: dict(Counter(str(r.get(k)) for r in by_m[m])) for k in keys}
OUT["params"] = {
    "greedy": _pc("greedy", ["guard_quantile", "safety_buffer", "prune_quantile"]),
    "LA": _pc("LA", ["n_scenarios", "horizon_hours", "solve_mode",
                     "la_energy_quantile", "tiebreak_min", "criterion",
                     "prune_quantile"]),
    "2SP": _pc("2SP", ["n_scenarios", "twosp_heuristics"]),
    "RO": _pc("RO", ["gamma"]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feasibility with Wilson intervals + violation types
# ─────────────────────────────────────────────────────────────────────────────
feas = {}
for m in METHODS:
    for rc in ROUTES + ["all"]:
        recs = [r for r in by_m[m] if rc == "all" or r["route_class"] == rc]
        k = sum(1 for r in recs if r["outcome"] == "infeasible")
        n = len(recs)
        types = Counter()
        for r in recs:
            if r["outcome"] == "infeasible":
                for t, c in (cs._get(r, ("metrics", "violations_by_type")) or {}).items():
                    if t not in cs._NONFATAL_VIOL:
                        types[t] += 1
        feas[f"{m}|{rc}"] = dict(n=n, k=k, pct=100 * k / n if n else None,
                                 ci=wilson(k, n), types=dict(types))
OUT["feasibility"] = feas

# violation severity (first fatal violation of each infeasible run)
sev = {}
for m in METHODS:
    strand, hos = [], defaultdict(list)
    for r in by_m[m]:
        if r["outcome"] != "infeasible":
            continue
        vs = cs._get(r, ("metrics", "violations")) or []
        vs = [v for v in vs if v.get("type") not in cs._NONFATAL_VIOL]
        if not vs:
            continue
        v = vs[0]
        if v["type"] == "stranding":
            strand.append(float(v.get("amount") or 0.0))
        elif v["type"].startswith("hos"):
            hos[v["type"]].append(60 * float(v.get("amount") or 0.0))
    sev[m] = dict(
        n_strand=len(strand),
        strand_kwh_q=[q(strand, p) for p in (0.25, 0.5, 0.75, 0.9)] if strand else None,
        strand_gt5=sum(1 for x in strand if x > 5),
        hos={t: dict(n=len(v), min_q=[q(v, p) for p in (0.5, 0.9)],
                     gt15=sum(1 for x in v if x > 15),
                     gt30=sum(1 for x in v if x > 30)) for t, v in hos.items()},
        no_feasible_action=sum(1 for r in by_m[m] if r["outcome"] == "infeasible"
                               and "no_feasible_action" in (cs._get(r, ("metrics", "violations_by_type")) or {})),
    )
OUT["severity"] = sev

# ─────────────────────────────────────────────────────────────────────────────
# 3. Gaps: as published (feasible-only) and on the common-feasible set
# ─────────────────────────────────────────────────────────────────────────────
gap = {}
inst_gap = defaultdict(dict)       # instance -> method -> (gap_pen, gap_nopen)
for r in base:
    if r["outcome"] == "feasible" and r.get("gap_pen") is not None:
        inst_gap[r["instance"]][r["method"]] = (100 * r["gap_pen"],
                                                100 * r["gap_nopen"])
common = [i for i, d in inst_gap.items() if all(m in d for m in METHODS)]
common_LG = [i for i, d in inst_gap.items() if all(m in d for m in ("LA", "greedy", "2SP"))]
for rc in ROUTES:
    for cc in ["few", "medium", "many"]:
        key = f"{rc}|{cc}"
        cell = {}
        for m in METHODS:
            pub = [inst_gap[i][m][0] for i in inst_gap
                   if m in inst_gap[i] and i.startswith(f"R{rc}C{cc}T")]
            com = [inst_gap[i][m][0] for i in common if i.startswith(f"R{rc}C{cc}T")]
            cell[m] = dict(pub=st.mean(pub) if pub else None, n_pub=len(pub),
                           common=st.mean(com) if com else None, n_common=len(com))
        gap[key] = cell
OUT["gap_cells"] = gap
OUT["n_common_all4"] = len(common)
OUT["n_common_LA_greedy_2SP"] = len(common_LG)

from scipy.stats import wilcoxon                                # noqa: E402
paired = {}
for rc in ROUTES + ["all"]:
    ids = [i for i in common if rc == "all" or i.startswith(f"R{rc}")]
    out = {}
    for m in ["greedy", "2SP", "RO"]:
        d = [inst_gap[i]["LA"][0] - inst_gap[i][m][0] for i in ids]
        if len(d) > 5:
            try:
                p = wilcoxon(d).pvalue
            except ValueError:
                p = None
            out[m] = dict(n=len(d), mean_LA_minus=st.mean(d),
                          ci=mean_ci(d)[1:], median=st.median(d), p=p,
                          LA_better_share=100 * sum(1 for x in d if x < 0) / len(d))
    out["LA_mean_gap_ci"] = mean_ci([inst_gap[i]["LA"][0] for i in ids])
    paired[rc] = out
OUT["paired_LA"] = paired

# LA mean gap over Table-8 cells and overall
la_pub = [inst_gap[i]["LA"][0] for i in inst_gap if "LA" in inst_gap[i]]
OUT["LA_overall_gap_pub"] = mean_ci(la_pub)
OUT["LA_cells_mean_of_means"] = st.mean([gap[k]["LA"]["pub"] for k in gap])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Directive 2002/15/EC (ex post) and weekly working time
# ─────────────────────────────────────────────────────────────────────────────
wtd = {}
for m in METHODS:
    for rc in ROUTES:
        recs = [r for r in by_m[m] if r["route_class"] == rc and r["outcome"] == "feasible"]
        dc = [cs._get(r, ("metrics", "directive_compliance")) or {} for r in recs]
        n_ok = sum(1 for d in dc if d.get("compliant"))
        cons = [d.get("max_consec_work") for d in dc if d.get("max_consec_work") is not None]
        kinds = Counter()
        for d in dc:
            for iss in d.get("issues", []):
                kinds["consec>6h" if "consecutive" in iss else "shift-break"] += 1
        ww = [cs._safe_float(cs._get(r, ("metrics", "weekly_work_h"))) for r in recs]
        ww = [x for x in ww if x is not None]
        over = [x - 60 for x in ww if x > 60 + 1e-3]
        wtd[f"{m}|{rc}"] = dict(
            n=len(recs), compliant_pct=100 * n_ok / len(recs) if recs else None,
            issues=dict(kinds),
            max_consec_q=[q(cons, p) for p in (0.5, 0.9, 1.0)] if cons else None,
            weekly_mean=st.mean(ww) if ww else None, weekly_max=max(ww) if ww else None,
            weekly_over60_pct=100 * len(over) / len(ww) if ww else None,
            weekly_over_mean_min=60 * st.mean(over) if over else None,
            weekly_over_max_min=60 * max(over) if over else None)
OUT["directive"] = wtd

# ─────────────────────────────────────────────────────────────────────────────
# 5. Time windows: hit rate, early/late, lateness magnitude
# ─────────────────────────────────────────────────────────────────────────────
tw = {}
for m in METHODS:
    recs = [r for r in by_m[m] if r["window_class"] != "none" and r["outcome"] == "feasible"]
    n_c = sum(int(cs._get(r, ("metrics", "tw_n_customers")) or 0) for r in recs)
    n_miss = sum(int(cs._get(r, ("metrics", "tw_n_misses")) or 0) for r in recs)
    early = sum(int(cs._get(r, ("metrics", "tw_n_early")) or 0) for r in recs)
    late = sum(int(cs._get(r, ("metrics", "tw_n_late")) or 0) for r in recs)
    amts_e, amts_l = [], []
    for r in recs:
        for s_, mm in (cs._get(r, ("metrics", "tw_misses_by_stop")) or {}).items():
            (amts_e if mm.get("type") == "early" else amts_l).append(60 * float(mm.get("amount") or 0))
    tw[m] = dict(n_runs=len(recs), hit_pct=100 * (1 - n_miss / n_c) if n_c else None,
                 early=early, late=late,
                 late_min_q=[q(amts_l, p) for p in (0.5, 0.9)] if amts_l else None,
                 early_min_q=[q(amts_e, p) for p in (0.5, 0.9)] if amts_e else None)
OUT["tw"] = tw

# ─────────────────────────────────────────────────────────────────────────────
# 6. Offline solves: time limits, repairs; oracle certification
# ─────────────────────────────────────────────────────────────────────────────
off = {}
for m, flag in (("RO", "ro_optimal"), ("2SP", "twosp_optimal")):
    for rc in ROUTES:
        recs = [r for r in by_m[m] if r["route_class"] == rc]
        off[f"{m}|{rc}"] = dict(
            n=len(recs),
            time_limit_pct=100 * sum(1 for r in recs if not r.get(flag)) / len(recs),
            repair_runs_pct=100 * sum(1 for r in recs if (cs._get(r, ("metrics", "n_repairs")) or 0) > 0) / len(recs),
            repairs_mean=st.mean([(cs._get(r, ("metrics", "n_repairs")) or 0) for r in recs]))
OUT["offline"] = off

from src.output_analysis import run_cache                       # noqa: E402
orc = run_cache.load_oracles(P.solutions())
ora = {}
for rc in ROUTES:
    recs = [v for k, v in orc.items() if k.startswith(f"R{rc}") and "__" not in k
            and "_error" not in v]
    gaps = [100 * float(v.get("gap") or 0) for v in recs]
    ora[rc] = dict(n=len(recs),
                   above_tol=sum(1 for g in gaps if g > 0.5 + 1e-6),
                   above_1=sum(1 for g in gaps if g > 1.0),
                   gap_q=[q(gaps, p) for p in (0.5, 0.9, 1.0)])
OUT["oracle"] = ora

# ─────────────────────────────────────────────────────────────────────────────
# 7. Trajectory pass (opens the files): driving totals, elapsed time,
#    bay idling, manoeuvres-as-driving what-if, common realisation
# ─────────────────────────────────────────────────────────────────────────────
inst_cache = {}


def inst(name):
    if name not in inst_cache:
        f = P.find_in(P.instances(), f"{name}.json") if hasattr(P, "instances") else None
        if f is None:
            f = os.path.join(ROOT, "instances", f"{name}.json")
        with open(f) as fh:
            raw = json.load(fh)
        fd = raw["instance"]
        fd["_n_real"] = (1 if raw.get("D_real") and not isinstance(raw["D_real"][0], list)
                         else len(raw.get("D_real") or []))
        inst_cache[name] = fd
    return inst_cache[name]


traj = defaultdict(lambda: defaultdict(list))
realisation_ok = Counter()
for r in base:
    m = r["method"]
    f = P.find_solution(r["_file"]) if hasattr(P, "find_solution") else None
    if f is None:
        hits = glob.glob(os.path.join(ROOT, "solutions", "**", r["_file"]), recursive=True)
        f = hits[0] if hits else None
    if f is None:
        continue
    with open(f) as fh:
        d = json.load(fh)
    rc = r["route_class"]
    Dl = d.get("D_actual_list") or []
    durs = d.get("durations_list") or []
    acts = d.get("actions") or []
    trj = d.get("sim_trajectory") or []
    done = r["outcome"] == "feasible"
    if done:
        traj[f"{m}|{rc}"]["drive_h"].append(sum(Dl))
        traj[f"{m}|{rc}"]["elapsed_h"].append(float(d.get("duration_h") or 0))
    # same realisation as the oracle cache?
    o = orc.get(r["instance"]) or {}
    oD = o.get("D_actual") if isinstance(o.get("D_actual"), dict) else None
    if oD and Dl:
        k = min(len(Dl), 5)
        same = all(abs(Dl[j] - float(oD.get(str(j), oD.get(j, -1)))) < 1e-5 for j in range(1, k))
        realisation_ok[(m, same)] += 1

    # bay idling: break declared at a CS with a charge, parallel mode, residual break time
    try:
        fd = inst(r["instance"])
    except Exception:
        fd = None
    K = set(fd["K"]) if fd else set()
    Tb = {"b45": 0.75, "b15": 0.25, "b30": 0.5}
    n_brk_cs, n_idle_short, n_idle_long, idle_min = 0, 0, 0, 0.0
    n_chg_only, n_chg = 0, 0
    for idx, (a, du) in enumerate(zip(acts, durs)):
        stop = trj[idx]["stop"] if idx < len(trj) else idx
        if stop not in K:
            continue
        brk = a.get("break_type")
        y = int(du.get("tauc", 0) > 1e-6)
        if y:
            n_chg += 1
            if brk in (None, "0") and a.get("rest_type") in (None, "0"):
                n_chg_only += 1
        if brk in Tb and y:
            n_brk_cs += 1
            if int(du.get("sigma", 0)) == 0 and du.get("taub", 0) > 1 / 60:
                idle_min += 60 * du["taub"]
                if du["tauc"] + 1e-6 < Tb[brk]:
                    n_idle_short += 1      # charge shorter than break, truck idles at bay
                else:
                    n_idle_long += 1       # charge covers break, break extended at bay
    t = traj[f"{m}|{rc}"]
    t["brk_cs"].append(n_brk_cs)
    t["idle_short"].append(n_idle_short)
    t["idle_long"].append(n_idle_long)
    t["idle_min"].append(idle_min)
    t["chg_only"].append(n_chg_only)
    t["chg"].append(n_chg)

    # manoeuvres-as-driving what-if (feasible runs only)
    if done and fd is not None:
        L = set(fd.get("L", []))
        cd = 0.0
        sd = 0.0
        worst_cd = 0.0
        worst_sd_excess = 0.0
        ext_used = 0
        for idx, (a, du) in enumerate(zip(acts, durs)):
            man = float(du.get("mstop", 0)) + float(du.get("mlay", 0)) + float(du.get("mseq", 0))
            brk, rst = a.get("break_type"), a.get("rest_type")
            reset_cd = brk in ("b45", "b30") or rst in ("r1", "r2")
            reset_sd = rst in ("r1", "r2")
            cd += man / 2
            sd += man / 2
            worst_cd = max(worst_cd, cd)
            if reset_sd:
                lim = 10.0 if ext_used < 2 else 9.0
                if sd > 9.0 + 1e-3:
                    ext_used += 1
                worst_sd_excess = max(worst_sd_excess, sd - lim)
            if reset_cd:
                cd = 0.0
            if reset_sd:
                sd = 0.0
            cd += man / 2
            sd += man / 2
            if idx < len(Dl):
                cd += Dl[idx]
                sd += Dl[idx]
            worst_cd = max(worst_cd, cd)
        t["cd_whatif_max"].append(worst_cd)
        t["sd_whatif_excess"].append(worst_sd_excess)

trj_out = {}
for k, t in traj.items():
    o = {}
    if t["drive_h"]:
        o["drive_max_h"] = max(t["drive_h"])
        o["elapsed_max_h"] = max(t["elapsed_h"])
        o["elapsed_gt144"] = sum(1 for x in t["elapsed_h"] if x > 144)
        o["elapsed_gt136"] = sum(1 for x in t["elapsed_h"] if x > 136)
    n = len(t["brk_cs"])
    o["n_runs"] = n
    o["brk_cs_per_run"] = st.mean(t["brk_cs"]) if n else None
    o["idle_short_per_run"] = st.mean(t["idle_short"]) if n else None
    o["idle_long_per_run"] = st.mean(t["idle_long"]) if n else None
    o["runs_with_idle_short_pct"] = 100 * sum(1 for x in t["idle_short"] if x) / n if n else None
    o["runs_with_idle_long_pct"] = 100 * sum(1 for x in t["idle_long"] if x) / n if n else None
    o["idle_min_per_run"] = st.mean(t["idle_min"]) if n else None
    o["chg_only_per_run"] = st.mean(t["chg_only"]) if n else None
    o["chg_per_run"] = st.mean(t["chg"]) if n else None
    if t["cd_whatif_max"]:
        cdw = t["cd_whatif_max"]
        o["cd_whatif_over_pct"] = 100 * sum(1 for x in cdw if x > 4.5 + 1e-3) / len(cdw)
        o["cd_whatif_over_q90_min"] = 60 * max(0.0, q(cdw, 0.9) - 4.5)
        o["cd_whatif_over_max_min"] = 60 * max(0.0, max(cdw) - 4.5)
        sdw = t["sd_whatif_excess"]
        o["sd_whatif_over_pct"] = 100 * sum(1 for x in sdw if x > 1e-3) / len(sdw)
    trj_out[k] = o
OUT["trajectory"] = trj_out
OUT["same_realisation_as_oracle"] = {f"{m}|{s}": c for (m, s), c in realisation_ok.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 8. Uncertainty model facts (closed form + Monte Carlo on the law itself)
# ─────────────────────────────────────────────────────────────────────────────
mu, sig = S.lognormal_params(S.TRAVEL_TIME_CV_TARGET)
rng = np.random.default_rng(1)
xi = S.sample_multipliers(2_000_000, rng)
legs = defaultdict(list)
for name in inst_all:
    try:
        fd = inst(name)
    except Exception:
        continue
    rc = name.split("C")[0][1:]
    Dn = [float(v) for v in fd["D"].values() if float(v) > 0]
    legs[rc].append(st.mean(Dn))
leg_min = {rc: 60 * st.mean(v) for rc, v in legs.items()}
sd_xi = S.xi_realised_sd()
per = {}
for rc, lm in leg_min.items():
    n = 270 / lm
    rho = 0.4
    var_fac = (1 + rho) / (1 - rho)          # large-n AR(1) variance inflation
    per[rc] = dict(leg_min=lm, legs_per_4h30=n,
                   cv_period_iid=100 * sd_xi / math.sqrt(n),
                   cv_period_ar04=100 * sd_xi * math.sqrt(var_fac) / math.sqrt(n))
ecr80 = S.ecr(80)
e_ratio = float(np.mean([S.ecr(80 / x) for x in xi[:400_000]])) / ecr80
OUT["uncertainty"] = dict(
    mu_eta=mu, sigma_eta=sig, mean_xi=float(xi.mean()), median_xi=float(np.median(xi)),
    sd_xi_closed=sd_xi, sd_xi_mc=float(xi.std()), p_xi_lt1=float((xi < 1).mean()),
    q05=float(np.quantile(xi, 0.05)), q95=float(np.quantile(xi, 0.95)),
    cap_mass=S.xi_atom_mass(), per_period=per,
    energy_jensen_bias_pct=100 * (e_ratio - 1),
    p_energy_above_nominal=float(np.mean([S.ecr(80 / x) > ecr80 for x in xi[:400_000]])))

# ─────────────────────────────────────────────────────────────────────────────
# 9. Vehicle / charging facts
# ─────────────────────────────────────────────────────────────────────────────
veh = dict(ecr50=S.ecr(50), ecr80=ecr80, ecr90=S.ecr(90),
           mass_kg=S.VEH_MASS_KG, cd=S.C_D,
           e_period_80=270 / 60 * 80 * ecr80,
           p_needed_45min=270 / 60 * 80 * ecr80 / 0.75)
curves = {}
for P_kw in (150, 350, 700, 1000):
    for ecap in (300, 500, 700, 900):
        tb = S.charging_curve(P_kw, ecap)
        t20_80 = (0.6 * ecap) / P_kw
        tail = tb[2] - tb[1]
        curves[f"{P_kw}kW|{ecap}kWh"] = dict(
            T1_h=tb[1], T2_h=tb[2], tail_min=60 * tail, t20_80_min=60 * t20_80,
            t20_100_min=60 * (t20_80 + tail), c_rate_bulk=P_kw / ecap,
            p_tail_kw=min(P_kw, S.TAIL_C_RATE * ecap))
veh["curves"] = curves
qmu = math.log(S.QUEUE_WAIT_MEAN_MIN ** 2 / math.sqrt(S.QUEUE_WAIT_STD_MIN ** 2 + S.QUEUE_WAIT_MEAN_MIN ** 2))
qsg = math.sqrt(math.log(1 + (S.QUEUE_WAIT_STD_MIN / S.QUEUE_WAIT_MEAN_MIN) ** 2))
veh["queue_log_params"] = (qmu, qsg)
veh["t_start"] = S.T_START
veh["tmin_charge_h"] = 0.25
OUT["vehicle"] = veh

# ─────────────────────────────────────────────────────────────────────────────
# 10. Sensitivity runs: which LA configuration, how many pairs
# ─────────────────────────────────────────────────────────────────────────────
allrows = cs.load_solutions(P.solutions())
sens = defaultdict(Counter)
for r in allrows:
    inst_ = r.get("instance") or ""
    if "__" in inst_ and r.get("method") == "LA":
        tag = inst_.split("__")[1]
        sens[tag][(r.get("solve_mode"), r.get("la_energy_quantile"),
                   r.get("n_scenarios"), r.get("horizon_hours"))] += 1
OUT["sens_LA_configs"] = {k: {str(kk): vv for kk, vv in v.items()} for k, v in sens.items()}
base_la_300 = Counter((r.get("solve_mode"), r.get("la_energy_quantile"))
                      for r in allrows if r.get("method") == "LA" and "__" not in (r.get("instance") or "")
                      and r.get("route_class") is None)
OUT["all_base_LA_modes"] = {str(k): v for k, v in Counter(
    (r.get("solve_mode"), r.get("la_energy_quantile"), r.get("variant"))
    for r in allrows if r.get("method") == "LA" and "__" not in (r.get("instance") or "")).items()}

with open(os.path.join(os.path.dirname(__file__), "review_checks.json"), "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str)[:60000])
