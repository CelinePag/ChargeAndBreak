"""Oracle-trained vs LA-trained students on the 125 base test routes
(2026-10-20, option A).  Same measure as both_metrics.py: gap to the oracle,
model minus LA in points, paired by route, seeds averaged per route; with and
without the 0.5 h window penalty.  Also rests, charges and windows per run,
and the students compared with each other directly."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import both_metrics as bm                                         # noqa: E402

RES = bm.RES
ARMS = [("Trees, LA labels", "gbt_F95_base_s{}"),
        ("Trees, oracle labels", "gbt_F95_oracle_s{}"),
        ("PyTorch, LA labels", "tmlp_F95_split_list_s{}"),
        ("PyTorch, oracle labels", "tmlp_F95_oracle_split_list_s{}")]


def tags(fmt, var):
    t = [f"{fmt.format(s)}_{var}" for s in range(3)]
    return t if all(os.path.exists(os.path.join(RES, f"eval_{x}_test.json")) for x in t) else None


def counts(tg):
    rows = [r for x in tg for r in json.load(open(os.path.join(RES, f"eval_{x}_test.json")))]
    done = [r for r in rows if r["route_completed"]]
    la = {r["instance"]: r for r in rows}
    return dict(rests=np.mean([r["n_rests"] for r in done]),
                la_rests=np.mean([r["LA_rests"] for r in la.values()]),
                charges=np.mean([r["n_charges"] for r in done]),
                la_charges=np.mean([r["LA_charges"] for r in la.values()]),
                more_rests=np.mean([r["n_rests"] > r["LA_rests"] for r in done]) * 100,
                fewer_rests=np.mean([r["n_rests"] < r["LA_rests"] for r in done]) * 100)


def main():
    for var in ("g99sr", "g99srN"):
        runs = {}
        for name, fmt in ARMS:
            t = tags(fmt, var)
            if t:
                runs[name] = (t, bm.base_runs(t))
        if not runs:
            continue
        insts = sorted(set().union(*[set(r) for _, r in runs.values()]))
        bl = bm.base_baselines(insts)
        la = bl["LA_MIPTAIL"]
        L = bm.la_summary(la, insts)
        print(f"\n### {var}: {'features at the 0.99 quantile when driving (as the headline)' if var == 'g99sr' else 'nominal features when driving (the train/serve fix)'}")
        print(f"LA gap to the oracle: {L['gap']:.2f} % without penalty, {L['gap_pen']:.2f} % with it\n")
        print("| Model | Routes | Gap, no penalty | Gap, with penalty | vs LA, no penalty | vs LA, with penalty "
              "| Missed windows (model / LA) | Rests per run (LA) | Charges per run (LA) | Rules broken |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, (t, r) in runs.items():
            s = bm.summarise(r, la)
            c = counts(t)
            print(f"| {name} | {s['n']} | {s['gap']:.2f} % | {s['gap_pen']:.2f} % | "
                  f"{s['vs']:+.2f} ± {s['vs_se']:.2f} | {s['vs_pen']:+.2f} ± {s['vs_pen_se']:.2f} | "
                  f"{s['tw']:.0f} / {s['tw_la']} | {c['rests']:.2f} ({c['la_rests']:.2f}) | "
                  f"{c['charges']:.1f} ({c['la_charges']:.1f}) | {s['broken']} / {s['runs']} |")
        # oracle labels minus LA labels, same arm, paired by route
        for arm in ("Trees", "PyTorch"):
            a, b = runs.get(f"{arm}, oracle labels"), runs.get(f"{arm}, LA labels")
            if not (a and b):
                continue
            d, dp = [], []
            for k in set(a[1]) & set(b[1]):
                ra = [x for x in a[1][k] if x["ok"]]
                rb = [x for x in b[1][k] if x["ok"]]
                if ra and rb:
                    d.append(np.mean([x["g"] for x in ra]) - np.mean([x["g"] for x in rb]))
                    dp.append(np.mean([x["gp"] for x in ra]) - np.mean([x["gp"] for x in rb]))
            se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
            print(f"\n{arm}: oracle labels minus LA labels, {len(d)} routes: "
                  f"{np.mean(d):+.2f} ± {se(d):.2f} points without penalty, "
                  f"{np.mean(dp):+.2f} ± {se(dp):.2f} with it")


if __name__ == "__main__":
    main()
