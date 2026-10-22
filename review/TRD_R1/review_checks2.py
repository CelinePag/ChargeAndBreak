"""
review_checks2.py -- follow-up ex-post checks (stored results only).

  (a) manoeuvre minutes per shift against the shift-driving slack left by each
      policy (referee 8.2: manoeuvres recorded as driving on a tachograph);
  (b) LA base runs split by the committed-charge energy guard (q0.5 vs none);
  (c) nominal-plan (EV / DET) violation severity, raw and supervised.

    python -m review.TRD_R1.review_checks2
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src import paths as P                                    # noqa: E402
from src.output_analysis import compile_solutions as cs       # noqa: E402

OUT = {}


def q(xs, p):
    return float(np.quantile(np.asarray(xs, float), p)) if xs else None


rows = cs.load_solutions(P.solutions())
rows = [r for r in rows if "__" not in (r.get("instance") or "")
        and not r.get("supervised") and (r.get("variant") or None) is None
        and r.get("method") in ("greedy", "RO", "2SP", "LA")]
rows, _ = cs._dedup_latest(rows)
cs._annotate_instance_tags(rows)
cs._annotate_gap_to_oracle(rows)
cs._annotate_outcome(rows)
rows = [r for r in rows if r.get("route_class")]

# (a) manoeuvres vs shift-driving slack ------------------------------------------------
man = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["outcome"] != "feasible":
        continue
    hits = glob.glob(os.path.join(ROOT, "solutions", "**", r["_file"]), recursive=True)
    if not hits:
        continue
    d = json.load(open(hits[0]))
    acts, durs, Dl = d["actions"], d["durations_list"], d["D_actual_list"]
    sd, mm = 0.0, 0.0
    key = f"{r['method']}|{r['route_class']}"
    for idx, (a, du) in enumerate(zip(acts, durs)):
        m_here = float(du.get("mstop", 0)) + float(du.get("mlay", 0)) + float(du.get("mseq", 0))
        if a.get("rest_type") in ("r1", "r2"):
            lim = 10.0 if sd > 9.0 + 1e-6 else 9.0
            man[key]["slack_min"].append(60 * (lim - sd))
            man[key]["man_min"].append(60 * (mm + m_here / 2))
            man[key]["exceeds"].append(int(mm + m_here / 2 > lim - sd + 1e-6))
            sd, mm = 0.0, m_here / 2
        else:
            mm += m_here
        if idx < len(Dl):
            sd += Dl[idx]
mo = {}
for k, v in man.items():
    mo[k] = dict(n_shifts=len(v["slack_min"]),
                 slack_med=q(v["slack_min"], 0.5), man_med=q(v["man_min"], 0.5),
                 man_q90=q(v["man_min"], 0.9),
                 exceed_pct=100 * st.mean(v["exceeds"]) if v["exceeds"] else None)
OUT["manoeuvre_vs_slack"] = mo

# (b) LA by energy guard -------------------------------------------------------------
la = [r for r in rows if r["method"] == "LA"]
grp = defaultdict(list)
for r in la:
    grp[str(r.get("la_energy_quantile"))].append(r)
OUT["LA_by_energy_guard"] = {
    g: dict(n=len(v),
            by_route={rc: sum(1 for r in v if r["route_class"] == rc) for rc in ("short", "medium", "long")},
            infeasible=sum(1 for r in v if r["outcome"] == "infeasible"),
            strand=sum(1 for r in v if r["outcome"] == "infeasible"
                       and "stranding" in (cs._get(r, ("metrics", "violations_by_type")) or {})),
            gap_mean=st.mean([100 * r["gap_pen"] for r in v if r["outcome"] == "feasible"
                              and r.get("gap_pen") is not None]))
    for g, v in grp.items()}

# (c) nominal (EV/DET) plan severity -------------------------------------------------
det = defaultdict(list)
for f in glob.glob(os.path.join(ROOT, "solutions", "VSS", "*_DET_*.json")):
    d = json.load(open(f))
    sup = bool(d.get("supervised")) or bool(d.get("guarded")) or "guard" in str(d.get("variant") or "")
    tag = "sup" if sup else "raw"
    vs = (d.get("metrics") or {}).get("violations") or []
    vs = [v for v in vs if v.get("type") != "hos_weekly"]
    det[tag + "|n"].append(1)
    if vs:
        v = vs[0]
        if v["type"] == "stranding":
            det[tag + "|kwh"].append(float(v.get("amount") or 0))
        else:
            det[tag + "|hos_min"].append(60 * float(v.get("amount") or 0))
do = {}
for tag in ("raw", "sup"):
    k = det[tag + "|kwh"]
    h = det[tag + "|hos_min"]
    do[tag] = dict(n_files=len(det[tag + "|n"]), n_strand=len(k),
                   kwh_q=[q(k, p) for p in (0.25, 0.5, 0.75, 0.9)] if k else None,
                   kwh_gt5=sum(1 for x in k if x > 5), kwh_gt20=sum(1 for x in k if x > 20),
                   n_hos=len(h), hos_q=[q(h, p) for p in (0.5, 0.9)] if h else None,
                   hos_gt15=sum(1 for x in h if x > 15))
OUT["det_severity"] = do
sample = glob.glob(os.path.join(ROOT, "solutions", "VSS", "*_DET_*.json"))[:1]
if sample:
    d = json.load(open(sample[0]))
    OUT["det_keys"] = {k: (v if not isinstance(v, (list, dict)) else type(v).__name__)
                       for k, v in d.items()}

json.dump(OUT, open(os.path.join(os.path.dirname(__file__), "review_checks2.json"), "w"), indent=1)
print(json.dumps(OUT, indent=1))
