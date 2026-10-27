"""
dsr_vs_la.py — an rl_search run on the STORED realisations, against the LA
==========================================================================
For an rl_search output made with --realised (its first draw per route is
the stored realisation), compute the manuscript's gap to the hindsight
oracle with the window penalty (paper_link.gaps, compile_solutions'
definition) for every variant, and pair it with the stored LA and Greedy runs
on the same routes.  Differences in pp, ± one standard error over routes;
also daily rests against the LA's, and the fresh-draw difference to the
student for reference.  Prints only.

    python ML/scratch/dsr_vs_la.py dsr_val_gbt
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from evaluate import baselines                                       # noqa: E402
from paper_link import baseline_gaps, gaps, oracle_facts             # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
T0 = 8.0


def main(name):
    recs = [json.loads(l) for l in open(os.path.join(RESULTS, f"rl_search_{name}.jsonl"))
            if l.strip()]
    variants = recs[0]["variants"]
    rows = []
    for r in recs:
        if not r.get("realised_first"):
            raise SystemExit("needs an rl_search run made with --realised")
        orc = oracle_facts(r["instance"])
        if orc is None:
            continue
        bl = baseline_gaps(r["instance"], orc)
        la = bl.get("LA_MIPTAIL", (None, None))[1]
        gr = bl.get("GREEDY", (None, None))[1]
        b = baselines(r["instance"])
        g = []
        for v in range(len(variants)):
            x = r["res"][v][0]
            g.append(None if x["failed"] else
                     gaps(x["duration"], x["misses"], x["duration"] + T0, orc)[1])
        fresh = [np.mean([(r["res"][v][k]["cost"] or np.nan) - (r["res"][0][k]["cost"] or np.nan)
                          for k in range(1, r["n_draws"])]) for v in range(len(variants))]
        rows.append(dict(inst=r["instance"], la=la, greedy=gr, g=g,
                         rests=[r["res"][v][0]["rests"] for v in range(len(variants))],
                         la_rests=b.get("LA_rests"), fresh=fresh,
                         changed=[r["res"][v][0]["changed"] for v in range(len(variants))]))
    print(f"{name}: {len(rows)} routes with an oracle (stored realisation)")
    ok = [x for x in rows if x["la"] is not None and all(v is not None for v in x["g"])]
    la = np.array([x["la"] for x in ok])
    print(f"LA gap to oracle (pen): mean {la.mean():.2f} %, n={len(ok)}")
    for v, vn in enumerate(variants):
        gv = np.array([x["g"][v] for x in ok])
        d = gv - la
        extra = sum(1 for x in ok if x["la_rests"] is not None and x["rests"][v] > x["la_rests"])
        fewer = sum(1 for x in ok if x["la_rests"] is not None and x["rests"][v] < x["la_rests"])
        line = (f"  {vn:10s} gap {gv.mean():5.2f} %   vs LA {d.mean():+.2f} ± "
                f"{d.std(ddof=1) / np.sqrt(len(d)):.2f} pp   routes resting more/less than "
                f"the LA {extra}/{fewer}")
        if v:
            g0 = np.array([x["g"][0] for x in ok])
            dd = gv - g0
            fr = np.array([x["fresh"][v] for x in rows])
            line += (f"   vs student {dd.mean():+.2f} ± {dd.std(ddof=1) / np.sqrt(len(dd)):.2f} pp;"
                     f" fresh draws {np.nanmean(fr):+.3f} ± "
                     f"{np.nanstd(fr, ddof=1) / np.sqrt(np.isfinite(fr).sum()):.3f} h;"
                     f" decisions changed {np.mean([x['changed'][v] for x in ok]):.1f}/route")
        print(line)
    for ln in ("Rshort", "Rmedium", "Rlong"):
        sub = [x for x in ok if x["inst"].startswith(ln)]
        if sub:
            parts = []
            for v, vn in enumerate(variants):
                d = np.array([x["g"][v] - x["la"] for x in sub])
                parts.append(f"{vn} {d.mean():+.2f}±{d.std(ddof=1) / np.sqrt(len(d)):.2f}")
            print(f"  {ln[1:]:7s} (n={len(sub)}) vs LA: " + ", ".join(parts))
    gr = [x["greedy"] - x["la"] for x in ok if x["greedy"] is not None]
    print(f"  Greedy vs LA {np.mean(gr):+.2f} pp (n={len(gr)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dsr_val_gbt")
