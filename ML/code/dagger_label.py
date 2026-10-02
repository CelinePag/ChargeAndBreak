"""
dagger_label.py — DAgger step B: ask the teacher about the students' states
==========================================================================
Run with the ANACONDA python (gurobipy); see dagger_io.py for why two steps.

For every query written by dagger_rollout.py:

  1. RESTORE  a fresh BEHDV + load_checkpoint() -- the mechanism the LA itself
     uses to resume a crashed run, so the restored vehicle is the visited one.
     Refused unless the stop, arrival time and charge match the record AND the
     state features recomputed here equal the ones recorded at the stop.
  2. ASK      Simulation.select_best_action, src unchanged, with the stored
     teacher's configuration (run_la_mixed.LA_CONFIG: mip tail, 25 scenarios,
     24 h, mean, tiebreak 5 min, energy quantile 0.5, no pruning, 8 workers).
     Two differences from a stored LA run, both unavoidable: no previous
     nominal plan to warm-start from (the student made none -- the same as at
     a run's first decision), and a fixed scenario seed per query so a label
     can be reproduced.  The LA writes its usual decision block to a log.
  3. ROWS     parse_logs.parse_log reads that block and extract.decision_rows
     turns it into rows -- the same two functions that built the dataset, so
     a DAgger label cannot mean something different from a teacher-run label.

One label file per (model, route); a route's queries are resumable at the stop
level (a complete log is reused, never re-solved).  --slice i/n splits the
query files across processes; --dry-run does steps 1 and 3's feature check
only and asks nothing, which is how the pipeline is tested without Gurobi
time.

    C:\\Users\\celinep\\AppData\\Local\\anaconda3\\python.exe ML/code/dagger_label.py --label probe
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.instance_gen.instance_io import load_instance_json      # noqa: E402
from src.simulation.BEHDV import BEHDV                            # noqa: E402

from dagger_io import (dataset_instance, label_dir, stable_seed)  # noqa: E402
from extract import (GUARD_Q, _column_order, decision_rows,       # noqa: E402
                     rows_to_arrays)
from features import ACTION_VOCAB, Precomp, state_features        # noqa: E402
from parse_logs import parse_log                                  # noqa: E402
from run_la_mixed import LA_CONFIG                                # noqa: E402

FEAT_TOL = 1e-4


def teacher_data(path):
    """The instance as run_algorithm hands it to the LA (title from the file
    stem, the energy guard on full_data)."""
    fd, D_real, E_real, cv = load_instance_json(path)
    fd["title"] = os.path.splitext(os.path.basename(path))[0]
    if LA_CONFIG.get("la_energy_quantile"):
        fd["la_energy_quantile"] = float(LA_CONFIG["la_energy_quantile"])
        fd["la_energy_cv"] = cv
    return fd, cv


def restore(fd, q):
    veh = BEHDV(fd)
    veh.load_checkpoint(q["ck"])
    if (veh.stop != q["stop"] or abs(veh.t_arr - q["t_arr"]) > 1e-9
            or abs(veh.e_arr - q["e_arr"]) > 1e-9):
        raise RuntimeError(f"stop {q['stop']}: restored state differs from the record "
                           f"(stop {veh.stop}, t {veh.t_arr}, e {veh.e_arr})")
    return veh


def feature_gap(fd_ft, pre, q, veh, cv):
    sf, _ = state_features(fd_ft, pre, q["stop"], veh, cv, GUARD_Q)
    rec = q["state_feats"]
    return max((abs(float(sf[k]) - rec[k]) for k in rec), default=0.0)


def log_complete(path, stop):
    if not os.path.exists(path):
        return None
    decs, _ = parse_log(path)
    hit = [d for d in decs if d.stop == stop and d.chosen is not None]
    return hit[0] if hit else None


def ask_teacher(fd, stop, veh, cv, log_path, seed, workers, time_limit):
    from src.simulation.Simulation import select_best_action
    with open(log_path, "w", buffering=1, encoding="utf-8") as fh:
        # the header line parse_log reads its meta from (as run_simulation_precomputed)
        print(f"  Settings : N_scen={LA_CONFIG['n_scenarios']}  "
              f"H={LA_CONFIG['horizon_hours']}h  cv={cv:.2f}  workers={workers}  "
              f"MIP  [{LA_CONFIG['criterion']}]   DAgger query, scenario seed {seed}",
              file=fh)
        select_best_action(
            full_data=fd, stop=stop, state=veh,
            n_scenarios=LA_CONFIG["n_scenarios"],
            horizon_hours=LA_CONFIG["horizon_hours"], cv=cv,
            scenario_seed=seed, time_limit=time_limit, verbose=False,
            n_workers=workers, solve_mode=LA_CONFIG["solve_mode"],
            charge_only=False, criterion=LA_CONFIG["criterion"],
            include_best=False, include_worst=False, prev_nom_sol=None,
            log_fh=fh, tracker=None, ext_shift_used=veh.ext_shift_used,
            prune_quantile=LA_CONFIG["prune_quantile"],
            tiebreak_min=LA_CONFIG["tiebreak_min"])
    return log_complete(log_path, stop)


def label_file(qpath, args, state_names, action_names):
    with open(qpath, encoding="utf-8") as fh:
        Q = json.load(fh)
    name = os.path.splitext(os.path.basename(qpath))[0]
    path = os.path.join(_ROOT, Q["path"])
    fd_la, cv = teacher_data(path)
    fd_ft, _, _, _ = load_instance_json(path)        # features: as extract.py builds them
    fd_ft["_horizon_h"] = 24.0
    pre = Precomp(fd_ft)

    rows, meta, n_asked = [], [], 0
    for q in Q["queries"]:
        veh = restore(fd_la, q)
        gap = feature_gap(fd_ft, pre, q, veh, cv)
        if gap > FEAT_TOL:
            raise RuntimeError(f"{name} stop {q['stop']}: state features differ from "
                               f"the rollout's by {gap:.2e}")
        if args.dry_run:
            continue
        log_path = os.path.join(label_dir(Q["label"], "logs"), f"{name}__s{q['stop']}.txt")
        dec = log_complete(log_path, q["stop"])
        if dec is None:
            seed = stable_seed("scen", Q["label"], name, q["stop"])
            dec = ask_teacher(fd_la, q["stop"], veh, cv, log_path, seed,
                              args.workers, args.time_limit)
            n_asked += 1
        if dec is None or not dec.actions:
            print(f"  [!] {name} stop {q['stop']}: no decision in the log, skipped")
            continue
        r = decision_rows(fd_ft, pre, veh, cv, dec, state_names, action_names)
        rows.extend(r)
        meta.extend([(q["stop"], q["key"], int(q["check"]))] * len(r))
    return Q, name, rows, meta, n_asked


def save(Q, name, rows, meta, state_names, action_names):
    arr = rows_to_arrays(rows)
    stop, key, check = zip(*meta)
    out = os.path.join(label_dir(Q["label"], "labels"), name + ".npz")
    np.savez_compressed(
        out, **arr,
        feature_names=np.array(state_names + action_names), n_state=len(state_names),
        action_vocab=np.array(ACTION_VOCAB),
        instance=np.array(dataset_instance(Q["instance"], Q["label"], Q["model"]["tag"])),
        route=np.array(Q["instance"]), family=np.array(Q["family"]),
        seed=np.array(int(Q["seed"])), physics=np.array(Q["physics"]),
        label=np.array(Q["label"]), model_tag=np.array(Q["model"]["tag"]),
        student_key=np.array(key), is_check=np.array(check, dtype=np.int8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--slice", default="0/1", help="i/n: this process's share of files")
    ap.add_argument("--limit", type=int, default=0, help="query files (0 = all)")
    ap.add_argument("--workers", type=int, default=LA_CONFIG["n_workers"])
    ap.add_argument("--time-limit", type=int, default=LA_CONFIG["time_limit"])
    ap.add_argument("--dry-run", action="store_true",
                    help="restore and check every state; ask the teacher nothing")
    args = ap.parse_args()

    i, n = (int(x) for x in args.slice.split("/"))
    files = sorted(glob.glob(os.path.join(label_dir(args.label, "queries"), "*.json")))[i::n]
    if args.limit:
        files = files[: args.limit]
    os.makedirs(label_dir(args.label, "logs"), exist_ok=True)
    os.makedirs(label_dir(args.label, "labels"), exist_ok=True)
    state_names, action_names = _column_order()
    print(f"[label] {args.label}: {len(files)} query files (slice {args.slice})"
          f"{'  DRY RUN' if args.dry_run else ''}", flush=True)

    t0, done, asked, n_q = time.time(), 0, 0, 0
    for k, qpath in enumerate(files):
        name = os.path.splitext(os.path.basename(qpath))[0]
        if not args.dry_run and os.path.exists(
                os.path.join(label_dir(args.label, "labels"), name + ".npz")):
            continue
        Q, name, rows, meta, n_asked = label_file(qpath, args, state_names, action_names)
        n_q += len(Q["queries"])
        asked += n_asked
        if not args.dry_run and rows:
            save(Q, name, rows, meta, state_names, action_names)
        done += 1
        print(f"  {k + 1}/{len(files)} {name}: {len(Q['queries'])} queries, "
              f"{n_asked} asked, {len(rows)} rows  {time.time() - t0:.0f}s", flush=True)
    print(f"[label] {done} files, {n_q} queries checked, {asked} teacher calls, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
