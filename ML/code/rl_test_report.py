"""
rl_test_report.py — the decisive-state rollout against the LA, oracle and Greedy
================================================================================
Reads rl_search runs made with --realised (first draw = the stored
realisation) for one or more training seeds of a student, and reports on the
stored realisations, as the paper does:

  gap to the hindsight oracle, with the window penalty and without
    base routes : paper_link.gaps (compile_solutions' definition), every
                  route with a feasible oracle;
    mixed routes: mixed_eval.read_oracle, oracles certified within 1 % only
                  (fig_results' rule);
  the student alone and student + rollout, each route averaged over the
  training seeds first; the LA (stored runs: solutions/basecase or
  ML/la_mixed) and Greedy (stored base runs; mixed_g99sr.jsonl on mixed
  routes); paired differences in pp, +- one standard error over routes;
  routes where any policy broke a rule are counted and left out of the means;
  daily rests above the LA's; wall time per route of the rl_search run.

Writes results/rl_test_report_<out>.json and prints the table.

    python ML/code/rl_test_report.py --routes base \
        --runs dsr_test_tmlp_F95_split_list_s0,dsr_test_tmlp_F95_split_list_s1 --out torch_base
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:                    # `src` on machines without the editable install
    sys.path.insert(0, _ROOT)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BETA, T0 = 0.5, 8.0


def _gaps_mixed(dur, tw, o):
    """(gap, gap with penalty) in %, mixed_eval._gap / fig_results.gap_pen."""
    if dur is None or o is None:
        return None, None
    od = o["ta_N"] - T0
    if od <= 0:
        return None, None
    return (100.0 * (dur - od) / od,
            100.0 * (dur + BETA * tw - od - o["pen"]) / (od + o["pen"]))


def references(routes, variant, inst):
    """oracle, (LA gap, LA gap pen), (Greedy gaps), LA rests -- or None if the
    route has no usable oracle."""
    if routes == "base":
        from evaluate import baselines
        from paper_link import baseline_gaps, gaps, oracle_facts
        orc = oracle_facts(inst)
        if orc is None:
            return None
        bl = baseline_gaps(inst, orc)
        return dict(gaps=lambda d, tw: gaps(d, tw, None if d is None else d + T0, orc),
                    la=bl.get("LA_MIPTAIL", (None, None)),
                    greedy=bl.get("GREEDY", (None, None)),
                    la_rests=baselines(inst).get("LA_rests"))
    import mixed_eval as me
    o = me.read_oracle(variant, inst)
    if o is None or (o["gap"] or 0) > 0.01:
        return None
    la = me.la_row(variant, inst)
    g = _GREEDY.get((variant, inst))
    return dict(gaps=lambda d, tw: _gaps_mixed(d, tw, o),
                la=_gaps_mixed(la["duration_h"], la["tw"], o) if la and la["completed"] else (None, None),
                greedy=_gaps_mixed(g["duration_h"], g["tw"], o) if g and g["completed"] else (None, None),
                la_rests=la["rests"] if la else None)


_GREEDY = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="base", choices=["base", "mixed"])
    ap.add_argument("--runs", required=True, help="rl_search run names, comma-separated "
                                                  "(one per training seed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.routes == "mixed":
        p = os.path.join(RESULTS, "mixed_g99sr.jsonl")
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                if r["method"] == "Greedy":
                    _GREEDY[(r["variant"], r["instance"])] = r

    per = defaultdict(lambda: dict(student=[], rollout=[], broken=0, secs=[], rests_s=[],
                                   rests_r=[], changed=[]))
    names = [n for n in a.runs.split(",") if n]
    for name in names:
        with open(os.path.join(RESULTS, f"rl_search_{name}.jsonl")) as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("realised_first"):
                    raise SystemExit(f"{name}: needs rl_search --realised")
                k = (r.get("variant", "base"), r["instance"])
                s, d = r["res"][0][0], r["res"][1][0]
                x = per[k]
                if s["failed"] or d["failed"]:
                    x["broken"] += 1
                    continue
                x["student"].append((s["duration"], s["misses"]))
                x["rollout"].append((d["duration"], d["misses"]))
                x["rests_s"].append(s["rests"])
                x["rests_r"].append(d["rests"])
                x["changed"].append(d["changed"])
                x["secs"].append(r["seconds"])

    rows, no_orc, broken = [], 0, 0
    for (v, inst), x in sorted(per.items()):
        if x["broken"]:
            broken += 1
            continue
        if len(x["student"]) != len(names):
            continue                            # not every seed has finished this route
        ref = references(a.routes, v, inst)
        if ref is None:
            no_orc += 1
            continue
        gs = np.mean([ref["gaps"](dur, tw) for dur, tw in x["student"]], axis=0)
        gr = np.mean([ref["gaps"](dur, tw) for dur, tw in x["rollout"]], axis=0)
        rows.append(dict(variant=v, inst=inst, length=("short" if inst.startswith("Rshort") else
                                                       "medium" if inst.startswith("Rmedium") else "long"),
                         student=gs, rollout=gr, la=ref["la"], greedy=ref["greedy"],
                         la_rests=ref["la_rests"], rests_s=np.mean(x["rests_s"]),
                         rests_r=np.mean(x["rests_r"]), changed=np.mean(x["changed"]),
                         secs=np.mean(x["secs"])))
    ok = [r for r in rows if r["la"][1] is not None]
    print(f"{a.out}: {len(per)} routes run x {len(names)} seed(s); {broken} with a broken run "
          f"(left out), {no_orc} without a usable oracle, {len(ok)} compared with the LA")
    if not ok:
        return

    def ms(v):
        v = np.asarray(v, dtype=float)
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    out = dict(runs=names, routes=a.routes, n=len(ok), broken=broken, no_oracle=no_orc)
    print(f"{'policy':22s} {'gap pen %':>9s} {'gap %':>7s} {'vs LA pen (pp)':>16s} "
          f"{'vs student pen':>16s} {'rests > LA':>10s}")
    for lab, key in (("LA", "la"), ("student", "student"), ("student + rollout", "rollout"),
                     ("Greedy", "greedy")):
        sub = [r for r in ok if r[key][1] is not None]
        gp = [r[key][1] for r in sub]
        g = [r[key][0] for r in sub]
        dl = ms([r[key][1] - r["la"][1] for r in sub])
        line = f"{lab:22s} {np.mean(gp):9.2f} {np.mean(g):7.2f} {dl[0]:+8.2f} ± {dl[1]:4.2f}  "
        res = dict(n=len(sub), gap_pen=float(np.mean(gp)), gap=float(np.mean(g)),
                   vs_la_pen=dl)
        if key == "rollout":
            dsd = ms([r["rollout"][1] - r["student"][1] for r in sub])
            line += f"{dsd[0]:+8.2f} ± {dsd[1]:4.2f}"
            res["vs_student_pen"] = dsd
            res["better_than_student"] = int(sum(r["rollout"][1] < r["student"][1] - 1e-9 for r in sub))
        else:
            line += " " * 16
        if key in ("student", "rollout"):
            rk = "rests_s" if key == "student" else "rests_r"
            more = sum(1 for r in sub if r["la_rests"] is not None and r[rk] > r["la_rests"] + 1e-9)
            line += f" {more:10d}"
            res["routes_more_rests_than_la"] = more
        print(line)
        out[key] = res
    for grp, keyf in (("length", lambda r: r["length"]), ("variant", lambda r: r["variant"])):
        vals = sorted({keyf(r) for r in ok})
        if len(vals) < 2:
            continue
        for gv in vals:
            sub = [r for r in ok if keyf(r) == gv]
            s = ms([r["student"][1] - r["la"][1] for r in sub])
            ro = ms([r["rollout"][1] - r["la"][1] for r in sub])
            print(f"  {gv:8s} (n={len(sub):3d}) vs LA pen: student {s[0]:+.2f} ± {s[1]:.2f}, "
                  f"student + rollout {ro[0]:+.2f} ± {ro[1]:.2f}")
            out.setdefault(f"by_{grp}", {})[gv] = dict(n=len(sub), student=s, rollout=ro)
    for ln in ("short", "medium", "long"):
        sec = [r["secs"] for r in ok if r["length"] == ln]
        if sec:
            print(f"  {ln:8s} wall time per route (student + rollout, one core): "
                  f"median {np.median(sec):.0f} s; changed decisions {np.mean([r['changed'] for r in ok if r['length'] == ln]):.1f}")
    with open(os.path.join(RESULTS, f"rl_test_report_{a.out}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
