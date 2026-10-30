"""
new_routes.py — brand-new routes of the 36 base-case instance classes
=====================================================================
For expert iteration on routes the network has never seen (2026-10-30): new
route seeds (default 26-63, i.e. 38 per class; the base case uses 1-25), so
the geometry, customers, windows AND the stored travel times are all new.
Nothing is solved on them except the short nominal MILP that centres the
time windows, exactly as for the base routes.

They must come from the SAME generator as the 900 base routes.  src/ was
refactored on 2026-08-22 and no longer reproduces them (same seeds, other
routes), so this script runs the generator of commit c11ba157b (2026-08-11,
the commit that added the base instances), extracted with

    git archive c11ba157b src | tar -x -C <dir>

and passed as --gen-src.  Checked on 2026-10-30: that generator reproduces
RshortCfewTnone_1, RlongCmanyTnone_7 and RmediumCfewTtight_3 exactly
(geometry, D_real, E_real); the window centres differ by up to 0.1 h, since
they come from a time-limited MILP solve.

The main process never imports `src`; each worker puts --gen-src first on
sys.path and checks that it got that tree.  One Gurobi thread per worker.
Resumable: existing files are skipped and each file is written under a
temporary directory first.  Writes ML/instances_new/<class>_<seed>.json and
prints, per route length, the new routes next to the base ones.

    git archive c11ba157b src | tar -x -C /tmp/cab_gen
    python ML/code/new_routes.py --gen-src /tmp/cab_gen --workers 32
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

ML = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(ML)
OUT = os.path.join(ML, "instances_new")

ROUTE = {"short": "Rshort", "medium": "Rmedium", "long": "Rlong"}
CUST = {"few": "Cfew", "medium": "Cmedium", "many": "Cmany"}
WIN = {"none": "Tnone", "tight": "Ttight", "medium": "Tmedium", "large": "Tlarge"}
_G = {}


def _init(gen_src, out):
    os.environ["CB_GRB_THREADS"] = "1"
    sys.path.insert(0, gen_src)
    import src
    if not os.path.abspath(src.__file__).startswith(os.path.abspath(gen_src)):
        raise SystemExit(f"imported {src.__file__}, not the generator in {gen_src}")
    import pyomo.environ as pyo              # one Gurobi thread per worker
    factory = pyo.SolverFactory

    def one_thread(*a, **k):
        s = factory(*a, **k)
        s.options["Threads"] = 1
        return s
    pyo.SolverFactory = one_thread
    from src.instance_gen import instance_io
    _G.update(io=instance_io, out=out, tmp=os.path.join(out, ".tmp", str(os.getpid())))
    os.makedirs(_G["tmp"], exist_ok=True)


def one(job):
    rc, cc, wc, seed = job
    t0 = time.perf_counter()
    p = _G["io"].generate_instance_file(rc, cc, wc, seed, output_dir=_G["tmp"], verbose=False)
    with open(p) as fh:
        src_w = (json.load(fh)["meta"].get("window_half_widths") or {}).get("_source", "-")
    os.replace(p, os.path.join(_G["out"], os.path.basename(p)))
    return os.path.basename(p), time.perf_counter() - t0, src_w


def summary(out):
    """Mean stops, km and customers per route length: new routes vs the base ones."""
    def stats(files):
        acc = {}
        for f in files:
            with open(f) as fh:
                i = json.load(fh)["instance"]
            ln = os.path.basename(f)[1:].split("C")[0]
            acc.setdefault(ln, []).append((i["N"], sum(i["km"].values()), len(i["C"])))
        return acc
    new = stats(glob.glob(os.path.join(out, "*.json")))
    base = stats(glob.glob(os.path.join(ROOT, "instances", "*.json")))
    print(f"{'length':8s} {'routes':>12s} {'stops':>13s} {'km':>15s} {'customers':>11s}   (new | base)")
    for ln in ("short", "medium", "long"):
        n, b = new.get(ln, []), base.get(ln, [])
        if not n or not b:
            continue
        m = [[sum(r[k] for r in x) / len(x) for k in range(3)] for x in (n, b)]
        print(f"{ln:8s} {len(n):5d} | {len(b):4d} {m[0][0]:5.0f} | {m[1][0]:5.0f} "
              f"{m[0][1]:6.0f} | {m[1][1]:6.0f} {m[0][2]:4.1f} | {m[1][2]:4.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-src", required=True,
                    help="directory holding the generator's `src` (git archive c11ba157b src)")
    ap.add_argument("--first-seed", type=int, default=26)
    ap.add_argument("--n-seeds", type=int, default=38)
    ap.add_argument("--classes", default="", help="comma list of class prefixes (default: all 36)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.first_seed <= 25:
        raise SystemExit("seeds 1-25 are the base routes; new routes start at 26")
    os.makedirs(a.out, exist_ok=True)
    keep = set(a.classes.split(",")) - {""}
    jobs = [(rc, cc, wc, s) for rc in ROUTE for cc in CUST for wc in WIN
            for s in range(a.first_seed, a.first_seed + a.n_seeds)
            if not keep or ROUTE[rc] + CUST[cc] + WIN[wc] in keep]
    todo = [j for j in jobs if not os.path.exists(
        os.path.join(a.out, f"{ROUTE[j[0]]}{CUST[j[1]]}{WIN[j[2]]}_{j[3]}.json"))]
    print(f"[new_routes] {len(jobs)} routes (seeds {a.first_seed}-{a.first_seed + a.n_seeds - 1}), "
          f"{len(todo)} to generate -> {a.out}", flush=True)
    t0, fallback = time.time(), 0
    with Pool(a.workers, initializer=_init, initargs=(a.gen_src, a.out)) as pool:
        for n, (name, _sec, src_w) in enumerate(pool.imap_unordered(one, todo), 1):
            fallback += src_w == "greedy"
            if n % 50 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} routes, {el / 60:.1f} min "
                      f"(~{el / n * (len(todo) - n) / 60:.0f} min left); windows centred on the "
                      f"greedy schedule (MILP fallback): {fallback}", flush=True)
    shutil.rmtree(os.path.join(a.out, ".tmp"), ignore_errors=True)
    missing = [j for j in jobs if not os.path.exists(
        os.path.join(a.out, f"{ROUTE[j[0]]}{CUST[j[1]]}{WIN[j[2]]}_{j[3]}.json"))]
    if missing:
        raise SystemExit(f"{len(missing)} routes missing, e.g. {missing[0]}")
    summary(a.out)


if __name__ == "__main__":
    main()
