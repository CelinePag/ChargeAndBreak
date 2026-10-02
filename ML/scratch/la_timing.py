"""Per-stop / per-action LA timings: mixed-power route vs its uniform original."""
import glob
import os
import re
import sys

import numpy as np

ROOT = r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak"
RE_STOP = re.compile(r"^\[LA\] stop (\d+) \((\w+)\)")
RE_ACT = re.compile(r"^  y=\d.*\(([\d.]+)s\)\s*$")
RE_CHOSEN = re.compile(r"^  -> CHOSEN .*\s([\d.]+)s\s*$")
RE_FREE = re.compile(r"\[FREE-MIP\]\s+obj=[\d.]+h\s+([\d.]+)s.*?(\d+)v/(\d+)c")


def parse(path):
    stops, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = RE_STOP.match(line)
        if m:
            cur = int(m.group(1))
            stops[cur] = dict(kind=m.group(2), acts=[], chosen=None, free=None)
            continue
        if cur is None:
            continue
        m = RE_ACT.match(line)
        if m:
            stops[cur]["acts"].append(float(m.group(1)))
        m = RE_FREE.search(line)
        if m:
            stops[cur]["free"] = (float(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = RE_CHOSEN.match(line)
        if m:
            stops[cur]["chosen"] = float(m.group(1))
    return {k: v for k, v in stops.items() if v["chosen"] is not None}


name = sys.argv[1] if len(sys.argv) > 1 else "RshortCfewTnone_22"
mixed = sorted(glob.glob(os.path.join(ROOT, "ML", "la_mixed", "logs", "**", f"{name}__pmix_LA_*.txt"), recursive=True))[-1]
uni = sorted(glob.glob(os.path.join(ROOT, "logs", "basecase", f"{name}_LA_MIPTAIL_*.txt")))[-1]
M, U = parse(mixed), parse(uni)
common = sorted(set(M) & set(U))
print(f"{name}: {len(M)} decisions done on pmix; {len(common)} stops in common with the uniform run")
for lab, S in (("uniform (stored)", U), ("pmix (running) ", M)):
    s = [S[k] for k in common]
    dec = np.array([x["chosen"] for x in s])
    act = np.array([a for x in s for a in x["acts"]])
    nact = np.array([len(x["acts"]) for x in s])
    free = [x["free"] for x in s if x["free"]]
    print(f"  {lab}: per decision median {np.median(dec):6.1f}s  mean {dec.mean():6.1f}s  total {dec.sum()/60:5.1f} min | "
          f"per action (25 scenario MIPs in parallel) median {np.median(act):5.1f}s | actions/stop {nact.mean():.1f} | "
          f"FREE-MIP median {np.median([f[0] for f in free]) if free else float('nan'):.1f}s, "
          f"size {free[0][1] if free else '-'}v/{free[0][2] if free else '-'}c")
ratio = np.array([M[k]["chosen"] / U[k]["chosen"] for k in common])
print(f"  paired per-decision ratio pmix/uniform: median {np.median(ratio):.2f}  IQR {np.percentile(ratio,25):.2f}-{np.percentile(ratio,75):.2f}")
for k in common[:12]:
    print(f"    stop {k:3d} {M[k]['kind']:6s} uniform {U[k]['chosen']:6.1f}s ({len(U[k]['acts'])} act)   pmix {M[k]['chosen']:6.1f}s ({len(M[k]['acts'])} act)"
          f"   sizes {U[k]['free'][1:] if U[k]['free'] else '-'} vs {M[k]['free'][1:] if M[k]['free'] else '-'}")
