"""
dagger_io.py — what the DAgger scripts share: route sets, paths, naming
======================================================================
DAgger (Ross, Gordon & Bagnell 2011) runs in two steps because no single
Python here has everything: the students need torch / lightgbm (the project
.venv), the teacher needs gurobipy (anaconda).

    step A  dagger_rollout.py  (.venv)     students drive training routes; the
                                           vehicle state at sampled stops is
                                           saved with BEHDV.to_checkpoint()
    step B  dagger_label.py    (anaconda)  each state is restored with
                                           BEHDV.load_checkpoint() and the LA
                                           teacher is asked about it
    use     dataset.add_dagger, --dagger in every trainer; dagger_report.py

This module imports nothing heavy, so both interpreters can load it.

Layout, one directory per LABEL (a round "r1", "r2", ... or a "probe"):

    ML/data/dagger/<label>/queries/<model tag>__<instance>.json   step A
    ML/data/dagger/<label>/logs/<model tag>__<instance>__s<stop>.txt
    ML/data/dagger/<label>/labels/<model tag>__<instance>.npz      step B

Route sets.  "base": the base-case instances, teacher runs in
solutions/basecase.  "pmix": the mixed-power TRAINING routes of
mixed_instances.py --split train (seeds 1-12), teacher runs (the pilot) in
ML/la_mixed.  Mixed-power TEST routes are never a DAgger route set.
"""
from __future__ import annotations

import glob
import json
import os
import re
import zlib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ML = os.path.join(_ROOT, "ML")
DAGGER = os.path.join(ML, "data", "dagger")

ROUTE_SETS = {
    "base": dict(inst_dir=os.path.join(_ROOT, "instances"), physics="base",
                 sol=os.path.join(_ROOT, "solutions", "basecase", "{inst}_LA_MIPTAIL_*.json"),
                 log=os.path.join(_ROOT, "logs", "basecase", "{run_id}.txt")),
    "pmix": dict(inst_dir=os.path.join(ML, "instances_mixed", "train", "pmix"), physics="pmix",
                 sol=os.path.join(ML, "la_mixed", "solutions", "**", "{inst}_LA_*.json"),
                 log=os.path.join(ML, "la_mixed", "logs", "**", "{run_id}.txt")),
}
_RE_ROUTE = re.compile(r"^R(short|medium|long)C\w+?T\w+?_\d+(__\w+)?$")


def label_dir(label, sub=None):
    d = os.path.join(DAGGER, label)
    return os.path.join(d, sub) if sub else d


def family_seed(inst):
    """'RshortCfewTnone_5__pmix' -> ('RshortCfewTnone', 5) (as extract.py)."""
    fam, seed = inst.split("__")[0].rsplit("_", 1)
    return fam, int(seed)


def routes(route_set, seeds):
    """{family: [(instance, path), ...]} of the route set, restricted to seeds."""
    out = {}
    for p in sorted(glob.glob(os.path.join(ROUTE_SETS[route_set]["inst_dir"], "*.json"))):
        inst = os.path.splitext(os.path.basename(p))[0]
        if not _RE_ROUTE.match(inst):
            continue
        fam, seed = family_seed(inst)
        if seed in seeds:
            out.setdefault(fam, []).append((inst, p))
    return out


def query_name(tag, inst):
    return f"{tag}__{inst}"


def dataset_instance(inst, label, tag):
    """The instance name a DAgger row carries in the training set: unique per
    (route, label, visiting model), so decision ids never collide with the
    teacher's own decisions on the same route."""
    return f"{inst}@{label}:{tag}"


def stable_seed(*parts):
    """A reproducible integer from strings / ints (rng seeds, scenario seeds)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def teacher_run(route_set, inst):
    """(solution dict, log path or None) of the latest stored LA run, or None."""
    rs = ROUTE_SETS[route_set]
    sols = sorted(glob.glob(rs["sol"].format(inst=inst), recursive=True))
    # a mixed route's pattern "{inst}_LA_*" would also match a longer instance
    # name that merely starts with this one; keep exact stems only
    sols = [s for s in sols if os.path.basename(s).startswith(f"{inst}_LA_")]
    if not sols:
        return None
    with open(sols[-1], encoding="utf-8") as fh:
        sol = json.load(fh)
    run_id = os.path.splitext(os.path.basename(sols[-1]))[0]
    logs = glob.glob(rs["log"].format(run_id=run_id), recursive=True)
    return sol, (logs[0] if logs else None)


def teacher_states(route_set, inst):
    """{stop: (t_arr, e_arr)} along the teacher's own run, or None."""
    run = teacher_run(route_set, inst)
    if run is None:
        return None
    traj = run[0].get("sim_trajectory") or []
    return {int(s["stop"]): (float(s["t_arr"]), float(s["e_arr"])) for s in traj}


def teacher_logged_stops(route_set, inst):
    """Stops with a decision block in the teacher's log.  ~20% of stored LA
    runs were resumed mid-route, and their logs start there: a teacher check
    is only possible where the teacher's own answer was written down."""
    run = teacher_run(route_set, inst)
    if run is None or run[1] is None:
        return set()
    from parse_logs import parse_log
    return {d.stop for d in parse_log(run[1])[0] if d.chosen is not None}


def jsonable(o):
    """json.dump default= for the numpy scalars a checkpoint may hold."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")
