"""Student + LA on close rest calls (run_hybrid.py), mixed test routes.
    python ML/scratch/cmp_hybrid.py [m10 m30 f5 m60 ...]
Per configuration: LA calls per route, gap to the certified oracle, the
paired change against the student alone (same model, same route), the
paired difference to the LA, and the runs that rest more than the LA."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import mixed_eval as me                                          # noqa: E402

R = os.path.join(HERE, "..", "results")
CFGS = sys.argv[1:] or ["m10", "m30", "f5", "m60"]
TAGS = {f"tmlp_T144_physmix_split_list_s{s}":
        "torch T, + pmix + mix routes" + ("" if s == 0 else f" (s{s})") for s in range(3)}

stu = {(r["variant"], r["instance"], r["method"]): r for r in me.read_rows(me.store_path(0.99, True))}
orc, la = {}, {}


def gap(r, v, n):
    if (v, n) not in orc:
        orc[(v, n)] = me.read_oracle(v, n)
    o = orc[(v, n)]
    if not r or not r["completed"] or not o or (o["gap"] or 0) > 0.01:
        return None
    return me._gap(r, o)


def la_of(v, n):
    if (v, n) not in la:
        la[(v, n)] = me.la_row(v, n)
    return la[(v, n)]


def se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


print(f"{'config':8s} {'routes':>6s} {'calls/run':>9s} {'LA s/run':>8s} {'fails':>5s} "
      f"{'gap':>6s} {'vs student':>14s} {'vs LA':>14s} {'extra-rest runs':>16s}")
first = True
for cfg in CFGS:
    p = os.path.join(R, f"hybrid_{cfg}.jsonl")
    if not os.path.exists(p):
        print(f"{cfg:8s} (no results)")
        continue
    rows = [json.loads(x) for x in open(p) if x.strip()]
    g, d_st, d_la, calls, las, xr, xs = [], [], [], [], [], 0, 0
    for r in rows:
        v, n = r["variant"], r["instance"]
        s = stu.get((v, n, TAGS[r["method"]]))
        L = la_of(v, n)
        gh, gs, gl = gap(r, v, n), gap(s, v, n), gap(L, v, n)
        calls.append(r["n_calls"]); las.append(r["la_s"])
        if gh is None:
            continue
        g.append(gh)
        if gs is not None:
            d_st.append(gh - gs)
        if gl is not None:
            d_la.append(gh - gl)
            xr += r["rests"] > L["rests"]
            if first and s:
                xs += s["rests"] > L["rests"]
    fails = sum(not r["completed"] for r in rows)
    print(f"{cfg:8s} {len(rows):6d} {np.mean(calls):9.1f} {np.mean(las):8.0f} {fails:5d} "
          f"{np.mean(g):+6.2f} {np.mean(d_st):+7.2f} ± {se(d_st):4.2f} "
          f"{np.mean(d_la):+7.2f} ± {se(d_la):4.2f} {xr:9d} of {len(d_la)}")
print("\n(student alone on the same runs: gap vs LA and extra-rest runs are in cmp_mix.py test)")
