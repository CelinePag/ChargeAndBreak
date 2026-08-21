"""
dagger.py — Dataset Aggregation: label the states the STUDENT actually visits
=============================================================================
WHY (the failure this fixes)
----------------------------
Behaviour cloning trains on situations the TEACHER created.  A policy that
acts creates its own situations: one small error puts the truck somewhere the
teacher never went, where nobody ever showed the student what to do, so it
errs again — errors COMPOUND (Ross, Gordon & Bagnell, AISTATS 2011, show the
cost can grow with the square of the horizon).  This mismatch between the
training distribution and the deployment distribution is COVARIATE SHIFT.

DAgger's fix, in one image: the apprentice drives, the master rides along, and
at every stop the master says what HE would have done — including in the
messes the apprentice created.  Those corrections join the training deck, the
student is retrained, repeat.  The training distribution converges to the
deployment distribution.

We are unusually well placed for this because our "master" is a solver we can
query at ANY state (robotics people imitating humans cannot).  The price is
that one query costs a full 25-scenario MIP look-ahead: ~45 s (short routes),
~73 s (medium), ~89 s (long).  Hence `--prob`, which labels a random subset of
visited stops instead of all of them; the sampling is UNIFORM over the
student's trajectory on purpose — labelling only the crashes would bias the
aggregated set toward disasters and teach the student that rare states are
common.

WHAT IT WRITES
--------------
One shard per instance: ML/data/dagger/round{R}/<instance>.npz holding
(X, y, tauc) for the sampled stops, plus the class mapping used.  Shards make
the job resumable and trivially parallel — run several processes over
disjoint --slice ranges.  merge_dagger.py folds them into a training set.

IMPORTANT: only TRAIN-split instances are eligible.  Aggregating states from
val/test instances would leak the evaluation sets into training.

Usage (repo root):
    python ML/code/dagger.py --model ML/models/policy_K20_seedsplit_seed0_cw0.pt \\
        --round 1 --prob 0.3 --limit 24
    # parallel: run N processes with --slice i/N
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ML", "code"))

from extract_dataset import (build_instance_context, features_at, action_key,
                             tag_for)
from rollout import make_policy, charge_time_to           # reuse the policy

from src import paths as _paths
_paths.redirect_outputs(os.path.join(ROOT, "ML"))         # keep runs inside ML/

from src.instance_gen.instance_io import load_instance_json
from src.simulation.Simulation import run_simulation_precomputed, select_best_action

INST_DIR = os.path.join(ROOT, "instances")
OUT_ROOT = os.path.join(ROOT, "ML", "data", "dagger")


def teacher_label(full_data, stop, vehicle, cv):
    """Query the exact-tail look-ahead at an arbitrary state.

    Same configuration the staged teacher runs used (S=25, H=24 h,
    solve_mode='mip', no flag pruning), so the labels are drawn from exactly
    the policy we are cloning — a cheaper teacher here would silently change
    what the student is imitating between rounds.
    """
    action, _scores, nom = select_best_action(
        full_data=full_data, stop=stop, state=vehicle,
        n_scenarios=25, horizon_hours=24.0, cv=cv,
        time_limit=20, verbose=False, n_workers=1,
        solve_mode="mip", criterion="mean",
        tracker=None, log_fh=None,
        ext_shift_used=vehicle.ext_shift_used,
        prune_quantile=None, tiebreak_min=5.0)
    tauc = 0.0
    if nom and nom.get("sol"):
        tauc = float(nom["sol"][0].get("tauc") or 0.0)
    return action, tauc


def main(a):
    meta = json.load(open(os.path.join(ROOT, "ML", "data",
                                       f"meta_{tag_for(a.k, a.mode)}.json")))
    classes = [tuple(c) for c in meta["classes"]]
    cls_of = {c: j for j, c in enumerate(classes)}
    k_look = meta["k_lookahead"]

    # eligible instances: TRAIN split only (never val/test — that would leak)
    z = np.load(os.path.join(ROOT, "ML", "data",
                             f"dataset_{tag_for(a.k, a.mode)}.npz"), allow_pickle=False)
    names = sorted(set(z["instance"][z["split"] == "train"].tolist()))
    rng = random.Random(a.seed)
    rng.shuffle(names)                       # mix route classes across shards
    if a.limit:
        names = names[:a.limit]
    if a.slice:
        i, n = (int(x) for x in a.slice.split("/"))
        names = names[i::n]

    out_dir = os.path.join(OUT_ROOT, f"round{a.round}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"DAgger round {a.round}: {len(names)} instances, "
          f"label prob {a.prob}, model {os.path.basename(a.model)}")

    for idx, name in enumerate(names):
        shard = os.path.join(out_dir, name + ".npz")
        if os.path.exists(shard) and not a.overwrite:
            print(f"[{idx+1}/{len(names)}] {name}: shard exists, skip")
            continue
        raw = json.load(open(os.path.join(INST_DIR, name + ".json")))
        full_data, D_real, E_real, cv = load_instance_json(
            os.path.join(INST_DIR, name + ".json"))
        ctx = build_instance_context(raw["instance"])
        stats = dict(clamp_up=0, tauc_pred_h=[], forced=0, forced_fallback=0,
                     rest_forced=0, rest_free=0, must_rest_fired=0,
                     must_rest_unmet=0)
        student = make_policy(a.model, raw["instance"], full_data, stats,
                              guard_q=(None if a.guard_q is not None
                                       and a.guard_q < 0 else a.guard_q),
                              cv=cv, no_split=a.no_split)
        X, Y, T = [], [], []
        local_rng = random.Random(hash(name) & 0xFFFF)
        t0 = time.perf_counter()

        def policy(fd, stop, vehicle):
            # THE STUDENT ACTS — this is what makes the visited states its own
            action, scores, nom = student(fd, stop, vehicle)
            if local_rng.random() < a.prob:
                st = dict(stop=stop, t_arr=vehicle.t_arr, e_arr=vehicle.e_arr,
                          cd=vehicle.cd, sd=vehicle.sd, sw=vehicle.sw,
                          phi=vehicle.phi, rho2_used=vehicle.rho2_used,
                          ext_shift_used=vehicle.ext_shift_used)
                try:
                    t_act, t_tauc = teacher_label(fd, stop, vehicle, cv)
                    key = action_key(t_act)
                    if key in cls_of:              # unseen combo -> skip, do
                        X.append(features_at(ctx, st, k_look))   # not invent a
                        Y.append(cls_of[key])                    # new class
                        T.append(t_tauc)
                except Exception as e:             # never let one bad solve
                    print(f"    [warn] teacher query failed @{stop}: {e}")
            return action, scores, nom

        run_simulation_precomputed(
            full_data, D_real, E_real, n_scenarios=1, horizon_hours=24.0,
            solve_mode="mip", verbose=False, supervised=False,
            external_policy=policy, alg_label=f"DAGGER{a.round}")
        np.savez_compressed(shard,
                            X=np.asarray(X, dtype=np.float32),
                            y=np.asarray(Y, dtype=np.int64),
                            tauc=np.asarray(T, dtype=np.float32),
                            instance=np.array([name] * len(Y)))
        print(f"[{idx+1}/{len(names)}] {name:28s} labelled {len(Y):4d} states "
              f"in {(time.perf_counter()-t0)/60:5.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="student that will drive")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--prob", type=float, default=0.3,
                   help="probability of querying the teacher at a stop")
    p.add_argument("--limit", type=int, default=24, help="0 = all train instances")
    p.add_argument("--slice", default=None, help="i/N for parallel shards")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--split-mode", dest="mode", default="seed",
                   choices=["family", "seed"])
    p.add_argument("--guard-q", dest="guard_q", type=float, default=0.95)
    p.add_argument("--no-split", dest="no_split", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
