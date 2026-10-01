"""
mixed_instances.py — routes whose chargers change ALONG the route
=================================================================
Every training route has one charger type (one power, one spacing) for its
whole length.  These test routes do not:

    pmix   each charger gets its own power, drawn from the four training
           values {150, 350, 700, 1000} kW
    dmix   charger spacing changes by thirds of the route: ~30 km, ~60 km
           (as generated) and ~120 km, in a random order
    mix    both

Each is built FROM a base test instance (route seeds 22-25) and keeps its
geometry, customers, windows and realisation (D_real, E_real) exactly, so a
mixed route is paired with its uniform original:

    denser   a layby near the midpoint of a long gap becomes a charger (it
             gets a queue drawn like the generator's, and the charger
             overheads); legs are unchanged, so D_real / E_real still apply
    sparser  every other charger becomes a layby (layby overhead, no queue)

Per-charger curves go in instance["TbarK"] {charger: Tbar}; instance["Tbar"]
becomes the SLOWEST charger's curve, which keeps the MILP's charging big-M
(TK) and the time bounds valid.  lb_t / ub_t and the horizon big-M H are
recomputed for the new chargers with src's own functions.

The draws are seeded from the instance name and the variant, so a rebuild
is identical.  Test routes go to ML/instances_mixed/<variant>/; TRAINING
routes (--split train: the LA labels them for the students, seeds 1-19 only)
go to ML/instances_mixed/train/<variant>/, so no evaluator can pick one up as
a test route.

    python ML/code/mixed_instances.py [--variants pmix,dmix,mix]
    python ML/code/mixed_instances.py --split train --variants pmix \\
        --lengths short --seeds 1-12          # the 2026-10-01 pilot
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.instance_gen.instances import (compute_horizon_bigM,     # noqa: E402
                                        compute_time_bounds)
from src.settings import (M_LAYBY_H, M_SEQ_H, M_STOP_H,          # noqa: E402
                          QUEUE_WAIT_MEAN_MIN, QUEUE_WAIT_STD_MIN,
                          charging_curve)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "instances_mixed"))
POWERS_KW = (150.0, 350.0, 700.0, 1000.0)
TEST_SEEDS = (22, 23, 24, 25)
# the sensitivity grid, minus long routes (their oracle does not certify
# within 2 h): the routes the physics-trained models were tested on
FAMILIES = [f"R{r}C{c}T{t}" for r in ("short", "medium")
            for c in ("few", "many") for t in ("none", "tight")]
DENSE_KM, SPARSE_EVERY = 30.0, 2


def _rng(name, variant):
    return np.random.default_rng(zlib.crc32(f"{name}|{variant}".encode()))


def _positions(inst):
    """Cumulative km of every node from the origin."""
    km = inst["km"]
    pos = [0.0]
    for i in range(int(inst["N"])):
        pos.append(pos[-1] + float(km[str(i)] if str(i) in km else km[i]))
    return pos


def _queue(rng):
    mu = np.log(QUEUE_WAIT_MEAN_MIN ** 2
                / np.sqrt(QUEUE_WAIT_STD_MIN ** 2 + QUEUE_WAIT_MEAN_MIN ** 2))
    sigma = np.sqrt(np.log(1 + (QUEUE_WAIT_STD_MIN / QUEUE_WAIT_MEAN_MIN) ** 2))
    return float(rng.lognormal(mu, sigma) / 60)


def _make_cs(inst, j, rng):
    inst["L"] = [x for x in inst["L"] if x != j]
    inst["M_lay"].pop(str(j), None)
    inst["K"] = sorted(inst["K"] + [j])
    inst["Q"][str(j)] = _queue(rng)
    inst["M_stop"][str(j)] = M_STOP_H
    inst["M_seq"][str(j)] = M_SEQ_H


def _make_layby(inst, j):
    inst["K"] = [x for x in inst["K"] if x != j]
    for d in ("Q", "M_stop", "M_seq"):
        inst[d].pop(str(j), None)
    inst["L"] = sorted(inst["L"] + [j])
    inst["M_lay"][str(j)] = M_LAYBY_H


def densities(inst, rng):
    """Thirds of the route at ~30 / ~60 / ~120 km spacing, random order.
    Returns the per-third labels for the metadata."""
    pos = _positions(inst)
    total = pos[-1]
    order = list(rng.permutation(["dense", "as-built", "sparse"]))
    third = lambda x: min(int(3 * x / total), 2)                   # noqa: E731
    # sparser: drop every other charger inside the sparse third
    sparse = [k for k in sorted(inst["K"]) if order[third(pos[k])] == "sparse"]
    for k in sparse[1::SPARSE_EVERY]:
        _make_layby(inst, k)
    # denser: while a gap inside the dense third is > 1.5x the target, turn
    # the layby nearest its midpoint into a charger
    changed = True
    while changed:
        changed = False
        anchors = [0] + sorted(inst["K"]) + [int(inst["N"])]
        for a, b in zip(anchors[:-1], anchors[1:]):
            mid = 0.5 * (pos[a] + pos[b])
            if order[third(mid)] != "dense" or pos[b] - pos[a] <= 1.5 * DENSE_KM:
                continue
            lay = [x for x in inst["L"] if a < x < b
                   and pos[x] - pos[a] >= 10 and pos[b] - pos[x] >= 10]
            if not lay:
                continue
            _make_cs(inst, min(lay, key=lambda x: abs(pos[x] - mid)), rng)
            changed = True
            break
    return order


def powers(inst, rng):
    """One power per charger, uniform over the training values."""
    return {int(k): float(rng.choice(POWERS_KW)) for k in inst["K"]}


def build(src_path, variant):
    with open(src_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    inst = payload["instance"]
    name = inst["title"]
    rng = _rng(name, variant)
    ecap = float(inst["Ecap"])
    base_kw = 350.0
    meta = dict(source=os.path.relpath(src_path, _ROOT), variant=variant)
    for d in ("L", "K"):
        inst[d] = [int(x) for x in inst[d]]
    inst.setdefault("M_lay", {})

    if variant in ("dmix", "mix"):
        meta["thirds"] = densities(inst, rng)
    kw = (powers(inst, rng) if variant in ("pmix", "mix")
          else {int(k): base_kw for k in inst["K"]})
    curves = {k: charging_curve(p, ecap) for k, p in kw.items()}
    slowest = charging_curve(min(kw.values()), ecap)
    inst["TbarK"] = {str(k): {str(r): t for r, t in c.items()} for k, c in curves.items()}
    inst["Tbar"] = {str(r): t for r, t in slowest.items()}
    meta["kw"] = {str(k): p for k, p in kw.items()}

    # bounds and big-M for the new chargers, with src's own functions
    ik = lambda d: {int(k): float(v) for k, v in d.items()}        # noqa: E731
    I, C, K = [int(x) for x in inst["I"]], [int(x) for x in inst["C"]], inst["K"]
    D, S, Q = ik(inst["D"]), ik(inst["S"]), ik(inst["Q"])
    man = float(next(iter(inst["M"].values())))
    lb, ub = compute_time_bounds(I, C, K, D, S, Q, ik(inst["Tbar"]),
                                 float(inst["T_hor"]),
                                 t0=float(inst.get("T_START", 8.0)),
                                 Man_default=man)
    inst["lb_t"] = {str(k): v for k, v in lb.items()}
    inst["ub_t"] = {str(k): v for k, v in ub.items()}
    inst["H"] = compute_horizon_bigM(int(inst["N"]), D, S, Q,
                                     ik(inst["M_stop"]), float(inst["Tr1"]))

    title = f"{name}__{variant}"
    inst["title"] = title
    inst["label"] = f"{title} (chargers mixed along the route: {variant})"
    payload["meta"] = dict(payload.get("meta", {}), mixed=meta)
    return title, payload


def _seeds(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="pmix,dmix,mix")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--seeds", default=None,
                    help="test: 22-25 (default); train: must lie in 1-19")
    ap.add_argument("--lengths", default="short,medium")
    args = ap.parse_args()
    seeds = _seeds(args.seeds) if args.seeds else list(TEST_SEEDS)
    if args.split == "train" and any(s > 19 for s in seeds):
        raise SystemExit("training routes must use the fitting seeds 1-19")
    lengths = args.lengths.split(",")
    fams = [f for f in FAMILIES if f[1:].split("C")[0] in lengths]
    for v in args.variants.split(","):
        out_dir = os.path.join(OUT, "train", v) if args.split == "train" else os.path.join(OUT, v)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for fam in fams:
            for s in seeds:
                src = os.path.join(_ROOT, "instances", f"{fam}_{s}.json")
                if not os.path.exists(src):
                    print(f"  missing {src}")
                    continue
                title, payload = build(src, v)
                with open(os.path.join(out_dir, title + ".json"), "w",
                          encoding="utf-8") as fh:
                    json.dump(payload, fh)
                n += 1
        print(f"[mixed] {v}: {n} routes -> {out_dir}")


if __name__ == "__main__":
    main()
