"""LA decision time per stop, by stop type, from the logs."""
import collections
import glob
import os
import re

import numpy as np

ROOT = r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak"
RE_STOP = re.compile(r"^\[LA\] stop (\d+) \((\w+)\)")
RE_ACT = re.compile(r"^  y=\d.*\(([\d.]+)s\)\s*$")
RE_CHOSEN = re.compile(r"^  -> CHOSEN .*\s([\d.]+)s\s*$")


def parse(path):
    out, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = RE_STOP.match(line)
        if m:
            cur = dict(kind=m.group(2), acts=0, t=None)
            continue
        if cur is None:
            continue
        if RE_ACT.match(line):
            cur["acts"] += 1
        m = RE_CHOSEN.match(line)
        if m:
            cur["t"] = float(m.group(1))
            out.append(cur)
            cur = None
    return out


def length(name):
    return "long" if "Rlong" in name else "medium" if "Rmedium" in name else "short"


def table(label, files):
    by = collections.defaultdict(list)
    acts = collections.defaultdict(list)
    by_len = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        L = length(os.path.basename(f))
        for s in parse(f):
            k = "CS" if s["kind"] == "CS" else ("CUSTOMER" if s["kind"] in ("CUST", "CUSTOMER") else s["kind"])
            by[k].append(s["t"]); acts[k].append(s["acts"]); by_len[L][k].append(s["t"])
            by["ALL"].append(s["t"]); acts["ALL"].append(s["acts"]); by_len[L]["ALL"].append(s["t"])
    print(f"\n{label}: {len(files)} runs, {len(by['ALL'])} decisions")
    print(f"  {'stop type':10s} {'share':>6s} {'mean s':>8s} {'median s':>9s} {'p90 s':>7s} {'actions':>8s}")
    for k in sorted(by, key=lambda x: (x != "ALL", x)):
        t = np.array(by[k])
        print(f"  {k:10s} {100*len(t)/len(by['ALL']):5.1f}% {t.mean():8.1f} {np.median(t):9.1f} "
              f"{np.percentile(t, 90):7.1f} {np.mean(acts[k]):8.1f}")
    print("  by route length (mean s): " + "  ".join(
        f"{L}: all {np.mean(v['ALL']):.0f}, CS {np.mean(v['CS']):.0f}" for L, v in sorted(by_len.items()) if v.get("CS")))


table("BASE CASE, standard LA (MIP tail)",
      sorted(glob.glob(os.path.join(ROOT, "logs", "basecase", "*LA_MIPTAIL*.txt"))))
mixed = sorted(glob.glob(os.path.join(ROOT, "ML", "la_mixed", "logs", "**", "*__pmix_LA_*.txt"), recursive=True))
table("MIXED-POWER test routes (pmix), finished + running", mixed)
base_same = []
for f in mixed:
    n = os.path.basename(f).split("__")[0]
    base_same += sorted(glob.glob(os.path.join(ROOT, "logs", "basecase", f"{n}_LA_MIPTAIL_*.txt")))[-1:]
table("same routes, uniform chargers (stored)", base_same)
