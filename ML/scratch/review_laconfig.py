"""Cheaper LA configurations as a compute/quality baseline (external review, 2026-10-20).
Paired with the base LA_MIPTAIL run of the same instance (same realisation)."""
import glob, json, os, re, collections
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
base = {}
for f in sorted(glob.glob(os.path.join(ROOT, "solutions", "basecase", "*_LA_MIPTAIL_*.json"))):
    inst = re.match(r"(.+?)_LA_MIPTAIL", os.path.basename(f)).group(1)
    base[inst] = f
out = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(ROOT, "solutions", "LAconfig", "*.json"))):
    m = re.match(r"(R\w+?_\d+)_(LA_S\d+H\d+)", os.path.basename(f))
    if not m or m.group(1) not in base: continue
    a, b = json.load(open(f)), json.load(open(base[m.group(1)]))
    if a.get("duration_h") is None or b.get("duration_h") is None: continue
    if a.get("metrics", {}).get("run_infeasible") or b.get("metrics", {}).get("run_infeasible"): continue
    wa = a.get("wall_clock_s_full_route") or a.get("wall_clock_s") or np.nan
    wb = b.get("wall_clock_s_full_route") or b.get("wall_clock_s") or np.nan
    pa = a["duration_h"] + 0.5 * a.get("metrics", {}).get("tw_n_misses", 0)
    pb = b["duration_h"] + 0.5 * b.get("metrics", {}).get("tw_n_misses", 0)
    out[m.group(2)].append((100 * (pa - pb) / pb, wa / wb))
for k, v in sorted(out.items()):
    d = np.array([x[0] for x in v]); w = np.array([x[1] for x in v])
    print(f"{k}: n={len(d)} vs LA_MIPTAIL objective {d.mean():+.2f}±{d.std(ddof=1)/np.sqrt(len(d)):.2f}%  median wall ratio {np.nanmedian(w):.2f}")
