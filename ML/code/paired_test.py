"""
paired_test.py — the statistics every comparative claim in the paper needs
==========================================================================
WHY THIS EXISTS
---------------
Reporting "mean +/- sd over 3 training seeds" answers the question *how
reproducible is this estimate?* — NOT *is this policy actually better?*.  The
two can point in opposite directions, and did: PPO v2's duration figure was
-0.238 +/- 0.049 across seeds, which looked decisive, while the paired
per-instance test put the 95% CI of the median at [-0.173, +0.053] with the
policy ahead on only 54% of instances (sign test p = 0.19).  Seed spread was
an order of magnitude smaller than instance spread, so the tight-looking
error bar was measuring the wrong thing entirely.

WHAT IT COMPUTES, per method
----------------------------
* paired per-instance differences against the TEACHER and against the
  hindsight ORACLE, for duration and for the penalised objective
  (duration + beta x window misses);
* an exact binomial SIGN TEST (no scipy dependency, and it needs no normality
  assumption — route deltas are heavily skewed);
* a bootstrap 95% CI of the median;
* halts, counted and never averaged into a duration.

Seeds belonging to one configuration are AVERAGED PER INSTANCE first, so the
unit of analysis is the instance, not the (instance, seed) pair — otherwise
three seeds of the same policy would be counted as three independent
observations of the same route and the p-values would be inflated.

COMMON SET
----------
With `--common`, only instances completed by EVERY compared method enter the
statistics.  Without it a policy that abandons its hardest routes looks better,
because those routes silently leave the median.

Usage
-----
  python ML/code/paired_test.py --split val --methods STUDENTSPRs STUDENTPPO2s
  python ML/code/paired_test.py --split test_id --methods TESTBASEs TESTDAGs
"""
from __future__ import annotations
import argparse, glob, json, math, os, random, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ML", "code"))
from src import paths as _paths                                   # noqa: E402

SOLS = os.path.join(ROOT, "ML", "solutions")


def _latest(pattern):
    fs = [f for f in _paths.in_tree(str(_paths.ROOT / "solutions"), pattern)
          if "nosplit" not in f and "LOCAL" not in f]
    return sorted(fs, key=os.path.basename)[-1] if fs else None


_ref = {}
def refs(inst):
    """Teacher duration, oracle objective and beta for one instance."""
    if inst not in _ref:
        g = _latest(f"{inst}_LA_MIPTAIL*.json")
        t = json.load(open(g)) if g else None
        p = os.path.join(ROOT, "solutions", "basecase", f"oracle_{inst}.json")
        if not os.path.exists(p):
            p = _latest(f"oracle_{inst}.json")
        try:
            o = json.load(open(p)) if p else None
        except Exception:
            o = None
        raw = json.load(open(os.path.join(ROOT, "instances", inst + ".json")))["instance"]
        _ref[inst] = dict(
            t_dur=t.get("duration_h") if t else None,
            t_pen=(t["duration_h"] + float(raw.get("beta", .5))
                   * int(t["metrics"].get("tw_n_misses") or 0))
                  if (t and t.get("duration_h") is not None) else None,
            o_obj=((o["obj"] - float(raw.get("T_START", 8.0)))
                   if (o and o.get("feasible") and o.get("obj")) else None),
            beta=float(raw.get("beta", .5)))
    return _ref[inst]


def collect(prefix):
    """-> {instance: (duration, penalised)} averaged over seeds, plus halts."""
    per = {}
    halts = {}
    for f in glob.glob(os.path.join(SOLS, f"*{prefix}*_S*.json")):
        d = json.load(open(f))
        i = d["instance"]
        if d.get("duration_h") is None:
            halts[i] = halts.get(i, 0) + 1
            continue
        b = refs(i)["beta"]
        pen = d["duration_h"] + b * int(d["metrics"].get("tw_n_misses") or 0)
        per.setdefault(i, []).append((d["duration_h"], pen))
    # average the seeds of one configuration -> one observation per instance
    return ({i: (float(np.mean([x[0] for x in v])),
                 float(np.mean([x[1] for x in v]))) for i, v in per.items()},
            halts)


def sign_test(v):
    """Exact binomial sign test, H1: median < 0.  No normality assumption —
    per-instance route deltas are strongly skewed."""
    n = sum(1 for x in v if abs(x) > 1e-12)
    k = sum(1 for x in v if x < -1e-12)
    if n == 0:
        return 0, 0, 1.0
    p = sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n
    return k, n, p


def boot_ci(v, B=20000, seed=0):
    rng = random.Random(seed)
    n = len(v)
    med = [np.median([v[rng.randrange(n)] for _ in range(n)]) for _ in range(B)]
    return np.percentile(med, [2.5, 97.5])


def main(a):
    from extract_dataset import split_of
    data = {m: collect(m) for m in a.methods}
    keep = None
    for m, (per, _) in data.items():
        sel = {i for i in per
               if (not a.split or split_of(i.rsplit("_", 1)[0],
                                           int(i.rsplit("_", 1)[1]), a.mode) == a.split)}
        keep = sel if keep is None else (keep & sel if a.common else keep | sel)
    keep = sorted(keep)
    print(f"instances: {len(keep)}   ({'COMMON completed set' if a.common else 'union'})"
          f"   split={a.split}\n")

    for m in a.methods:
        per, halts = data[m]
        rows = [i for i in keep if i in per]
        print(f"=== {m}   n={len(rows)}   halts={sum(halts.values())} "
              f"(on {len(halts)} instances) ===")
        for lab, j, ref_key in (("duration  vs teacher", 0, "t_dur"),
                                ("penalised vs teacher", 1, "t_pen")):
            v = [per[i][j] - refs(i)[ref_key] for i in rows
                 if refs(i)[ref_key] is not None]
            if not v:
                continue
            k, n, p = sign_test(v)
            lo, hi = boot_ci(v)
            sig = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"  {lab}:  mean {np.mean(v):+7.4f} h   median {np.median(v):+7.4f} h"
                  f"   95%CI [{lo:+.4f},{hi:+.4f}]")
            print(f"      better on {k}/{n} ({100*k/max(n,1):.0f}%)   "
                  f"sign test p={p:.3g}  -> {sig}")
        # absolute yardstick: gap to the hindsight oracle
        g = [100 * (per[i][1] - refs(i)["o_obj"]) / refs(i)["o_obj"]
             for i in rows if refs(i)["o_obj"]]
        if g:
            print(f"  gap to ORACLE (penalised): median {np.median(g):+.2f}%"
                  f"   IQR [{np.percentile(g,25):+.2f},{np.percentile(g,75):+.2f}]")
        print()

    # teacher / greedy reference gaps on the same instances
    for lab, tag in (("TEACHER", "LA_MIPTAIL"), ("GREEDY", "GREEDY")):
        g = []
        for i in keep:
            o = refs(i)["o_obj"]
            f = _latest(f"{i}_{tag}*.json")
            if not (o and f):
                continue
            d = json.load(open(f))
            if d.get("duration_h") is None:
                continue
            pen = d["duration_h"] + refs(i)["beta"] * int(d["metrics"].get("tw_n_misses") or 0)
            g.append(100 * (pen - o) / o)
        if g:
            print(f"{lab}: gap to ORACLE (penalised) median {np.median(g):+.2f}%  (n={len(g)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True,
                   help="method-label prefixes, e.g. STUDENTSPRs STUDENTPPO2s")
    p.add_argument("--split", default="val")
    p.add_argument("--split-mode", dest="mode", default="seed",
                   choices=["family", "seed"])
    p.add_argument("--common", action="store_true", default=True,
                   help="restrict to instances every method completed (default)")
    p.add_argument("--union", dest="common", action="store_false")
    main(p.parse_args())
