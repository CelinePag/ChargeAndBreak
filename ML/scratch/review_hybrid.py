"""Hybrid configs vs LA and vs the plain student (external review, 2026-10-20)."""
import os, sys, json, collections, contextlib, io
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
with contextlib.redirect_stdout(io.StringIO()):
    import mixed_eval as me
RES = os.path.join(HERE, "..", "results")
plain = {}
for r in me.read_rows(me.store_path(0.99, True)):
    if r["method"].startswith("torch T, + pmix + mix routes"):
        plain.setdefault((r["variant"], r["instance"]), []).append(r)
for cfg in ("m10", "m30", "m60", "f5"):
    rows = [json.loads(x) for x in open(os.path.join(RES, f"hybrid_{cfg}.jsonl"))]
    by = collections.defaultdict(list)
    for r in rows: by[(r["variant"], r["instance"])].append(r)
    dl, dp, calls, dec, fail = [], [], [], [], 0
    for k, rs in by.items():
        fail += sum(not r["completed"] for r in rs)
        la = me.la_row(*k)
        if not la or not la["completed"] or any(not r["completed"] for r in rs): continue
        f = lambda r: r["duration_h"] + 0.5 * r["tw"]
        h = np.mean([f(r) for r in rs]); l = f(la)
        p = plain.get(k)
        dl.append(100 * (h - l) / l)
        if p and all(x["completed"] for x in p): dp.append(100 * (h - np.mean([f(x) for x in p])) / l)
        calls += [r["n_calls"] for r in rs]; dec += [r["decisions"] for r in rs]
    dl, dp = np.array(dl), np.array(dp)
    print(f"{cfg}: routes {len(dl)} fails {fail}/{len(rows)}  vs LA (objective) {dl.mean():+.2f}±{dl.std(ddof=1)/np.sqrt(len(dl)):.2f}"
          f"  vs plain student {dp.mean():+.2f}±{dp.std(ddof=1)/np.sqrt(len(dp)):.2f}  calls/route {np.mean(calls):.1f} of {np.mean(dec):.0f} decisions")
print("hybrid val files:", [f for f in os.listdir(RES) if f.startswith("hybrid_val")])
