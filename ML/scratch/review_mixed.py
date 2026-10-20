"""Independent re-check of the mixed-charger claims (external review, 2026-10-20)."""
import os, sys, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import mixed_eval as me
rows = me.read_rows(me.store_path(0.99, True))
print("rows", len(rows), "methods", len({r['method'] for r in rows}),
      "variants", collections.Counter(r['variant'] for r in rows))
by = collections.defaultdict(dict)
for r in rows:
    by[r["method"]][(r["variant"], r["instance"])] = r
keys = sorted({(r["variant"], r["instance"]) for r in rows})
la = {}
orc = {}
for k in keys:
    v, n = k
    orc[k] = me.read_oracle(v, n)
    l = me.la_row(v, n) if v in ("pmix", "mix", "dmix") else None
    if l: la[k] = l
print("routes by variant:", collections.Counter(v for v, _ in keys), " LA rows:", collections.Counter(v for v, _ in la))
ocert = {k: (o is not None and (o['gap'] or 0) <= 0.01) for k, o in orc.items()}
print("certified oracle (<=1%):", collections.Counter((k[0], ocert[k]) for k in keys))

def obj(r): return r["duration_h"] + 0.5 * r["tw"]
def paired(labels, variants=("pmix", "mix"), pen=False):
    d = []
    for k in keys:
        if k[0] not in variants or k not in la or not la[k]["completed"]: continue
        xs = [by[m].get(k) for m in labels]
        if any(x is None or not x["completed"] for x in xs): continue
        f = obj if pen else (lambda r: r["duration_h"])
        s = np.mean([f(x) for x in xs]); l = f(la[k])
        d.append(100 * (s - l) / l)
    d = np.array(d)
    return len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d
groups = {
 "uniform T (s0 only)": ["torch split+list, T inputs, all phys"],
 "+pilot47 (3s)": ["torch T inputs, + pilot", "torch T, + pilot (s1)", "torch T, + pilot (s2)"],
 "+121 (3s)": ["torch T, + 121 mixed routes", "torch T, + 121 mixed routes (s1)", "torch T, + 121 mixed routes (s2)"],
 "+DAgger1 (3s)": ["torch T, + DAgger", "torch T, + DAgger (s1)", "torch T, + DAgger (s2)"],
 "+DAgger2 (3s)": ["torch T, + DAgger x2", "torch T, + DAgger x2 (s1)", "torch T, + DAgger x2 (s2)"],
 "+238 pmix+mix (3s)": ["torch T, + pmix + mix routes", "torch T, + pmix + mix routes (s1)", "torch T, + pmix + mix routes (s2)"],
 "Greedy": ["Greedy"],
}
for v in (("pmix",), ("mix",), ("pmix", "mix")):
    print(f"\n-- vs LA, % of LA, variants {v}")
    D = {}
    for g, labs in groups.items():
        n, m, se, d = paired(labs, v); n2, m2, se2, d2 = paired(labs, v, pen=True)
        D[g] = d
        print(f"  {g:22s} n={n:3d} duration {m:+.2f}±{se:.2f}   objective {m2:+.2f}±{se2:.2f}")
# DAgger1 vs +121, paired on the same routes (pmix)
def paired_between(a, b, variants):
    d = []
    for k in keys:
        if k[0] not in variants: continue
        xa = [by[m].get(k) for m in groups[a]]; xb = [by[m].get(k) for m in groups[b]]
        if any(x is None or not x["completed"] for x in xa + xb): continue
        sa = np.mean([x["duration_h"] for x in xa]); sb = np.mean([x["duration_h"] for x in xb])
        d.append(100 * (sa - sb) / sb)
    d = np.array(d); return len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
for a, b in [("+DAgger1 (3s)", "+121 (3s)"), ("+DAgger1 (3s)", "+pilot47 (3s)"), ("+DAgger2 (3s)", "+DAgger1 (3s)"), ("+238 pmix+mix (3s)", "+121 (3s)")]:
    for v in (("pmix",), ("mix",)):
        n, m, se = paired_between(a, b, v)
        print(f"  {a} minus {b} on {v}: n={n} {m:+.2f}±{se:.2f} pp")
print("\nconfigurations driven on the mixed TEST routes:", len(by))
