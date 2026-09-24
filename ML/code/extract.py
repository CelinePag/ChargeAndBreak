"""
extract.py — build the training matrix from the teacher's stored runs
=====================================================================
Labels come from the look-ahead's LOGS (per-action scenario-mean costs);
inputs come from its stored SOLUTION (the state at each stop).  This module
joins the two on the stop index and writes ML/data/dataset.npz.

Why the state is rebuilt rather than re-simulated
-------------------------------------------------
The obvious approach -- drive the recorded actions back through BEHDV -- does
NOT reproduce the run.  BEHDV.advance takes the executed break/rest from the
nominal MIP's flags (milp_sol["sol"][0]["b45"|"b15"|"b30"|"rho1"|"rho2"]),
but `vehicle.actions` stores the action the look-ahead SELECTED, and the two
disagree whenever the nominal re-solve placed the break elsewhere.  The
executed flags are never written to disk, so a naive replay silently drifts:
measured here as phi off by one and cd off by up to 2.7 h on the first three
routes tried.

Everything needed is stored directly, so we use it directly.  Eight of the
nine state fields sit in `sim_trajectory`.  The ninth, the shift spread h, is
not stored -- but it is exactly reconstructible, because BEHDV computes

    o_dwell = td[k] - t_arr[k] - taur[k]
    h[k+1]  = (0 if rest at k else h[k] + o_dwell) + D_actual[k]

and td_list, durations_list and D_actual_list are all in the solution JSON.

Gates (all cheap, all catch silent join corruption)
---------------------------------------------------
  G1  the LOG's state line agrees with the SOLUTION's trajectory -- two
      independently written sources of the same numbers, compared at the
      precision each was printed with
  G2  t_arr[k+1] == td[k] + D_actual[k]  (internal consistency of the run)
  G3  t_arr[last] - T_START == duration_h (the run's headline number)
      G2/G3 are checked to 1e-3 h (3.6 s), not to machine precision: the
      solution JSON stores td_list and D_actual_list rounded, which alone
      accounts for ~5e-5 h of disagreement.
  G4  the reconstructed spread is non-negative, and never exceeds the 15 h
      ceiling on a run that was recorded as feasible
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

from features import (ACTION_VOCAB, Precomp, StaticState,        # noqa: E402
                      action_features, key_to_action, state_features)
from parse_logs import parse_log                                  # noqa: E402

LOG_DIR = os.path.join(_ROOT, "logs", "basecase")
SOL_DIR = os.path.join(_ROOT, "solutions", "basecase")
INST_DIR = os.path.join(_ROOT, "instances")
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

GUARD_Q = None      # the teacher ran prune_quantile=None on every base run

# Tolerances for G1, set by the precision each source was PRINTED with:
# the log writes soc as a whole kWh, cd/sd/sw to 2 dp, t to 3 dp; the
# trajectory stores t/cd to 4 dp and e_arr to 2 dp.
G1_TOL = dict(t_arr=2e-3, e_arr=0.51, cd=6e-3, sd=6e-3, sw=6e-3,
              phi=0.5, rho2_used=0.5)
# G2/G3: the JSON rounds td_list / D_actual_list, which is worth ~5e-5 h.
TIME_TOL = 1e-3


def reconstruct_spread(sol: dict, traj: list) -> list:
    """Rebuild the shift spread h at every stop (BEHDV formula, exactly)."""
    td = sol.get("td_list") or []
    durs = sol.get("durations_list") or []
    dact = sol.get("D_actual_list") or []
    h = [0.0]
    for k in range(len(traj) - 1):
        if k >= len(td) or k >= len(dact):
            break
        taur = float(durs[k].get("taur", 0.0)) if k < len(durs) else 0.0
        o_dwell = max(0.0, float(td[k]) - float(traj[k]["t_arr"]) - taur)
        h.append((0.0 if taur > 0 else h[k] + o_dwell) + float(dact[k]))
    return h


def _solution_for(log_path: str):
    run_id = os.path.basename(log_path)[:-4]
    p = os.path.join(SOL_DIR, run_id + ".json")
    return p if os.path.exists(p) else None


def extract_one(log_path: str, state_names, action_names):
    """Join one route's logged costs onto its stored states."""
    run_id = os.path.basename(log_path)[:-4]
    inst = run_id.split("_LA_MIPTAIL")[0]
    rep = dict(instance=inst, run_id=run_id, status="ok", n_dec=0, n_rows=0,
               g1=dict(), g2_max=0.0, g3_delta=None, g4_bad=0, n_log_dec=0,
               halted=False)

    sol_path = _solution_for(log_path)
    if sol_path is None:
        rep["status"] = "no_solution"
        return None, rep
    inst_path = os.path.join(INST_DIR, inst + ".json")
    if not os.path.exists(inst_path):
        rep["status"] = "no_instance"
        return None, rep

    with open(sol_path, "r", encoding="utf-8") as fh:
        sol = json.load(fh)
    if sol.get("metrics", {}).get("run_infeasible"):
        rep["status"] = "run_infeasible"
        return None, rep

    decisions, meta = parse_log(log_path)
    rep["n_log_dec"] = len(decisions)
    if not decisions:
        rep["status"] = "empty_log"
        return None, rep

    traj = sol.get("sim_trajectory") or []
    if len(traj) < 2:
        rep["status"] = "no_trajectory"
        return None, rep

    fd, D_real, E_real, cv_file = load_instance_json(inst_path)
    fd["_horizon_h"] = meta.get("horizon_h", 24.0)
    cv = float(sol.get("cv", meta.get("cv", cv_file)))
    pre = Precomp(fd)

    spread = reconstruct_spread(sol, traj)
    traj_by_stop = {int(s["stop"]): (i, s) for i, s in enumerate(traj)}

    # -- G2 / G3: is the stored run internally consistent? --------------------
    td = sol.get("td_list") or []
    dact = sol.get("D_actual_list") or []
    g2 = 0.0
    for k in range(min(len(td), len(dact), len(traj) - 1)):
        g2 = max(g2, abs(float(traj[k + 1]["t_arr"])
                         - (float(td[k]) + float(dact[k]))))
    rep["g2_max"] = g2
    stored_dur = sol.get("duration_h")
    if stored_dur is not None:
        rep["g3_delta"] = float(traj[-1]["t_arr"]
                                - float(fd.get("T_START", 8.0))
                                - float(stored_dur))

    # -- G4: the reconstructed spread must be legal on a feasible run ---------
    Tspr2 = float(fd.get("Tspr2", 15.0))
    rep["g4_bad"] = int(sum(1 for x in spread if x < -1e-6 or x > Tspr2 + 1e-2))

    rows = []
    g1 = {k: 0.0 for k in G1_TOL}
    for dec in decisions:
        hit = traj_by_stop.get(dec.stop)
        if hit is None:
            continue
        i, tr = hit
        # -- G1: the two stored sources must agree --------------------------
        for fld, got in (("t_arr", dec.t_arr), ("e_arr", dec.soc),
                         ("cd", dec.cd), ("sd", dec.sd), ("sw", dec.sw),
                         ("phi", dec.phi), ("rho2_used", dec.rho2_used)):
            if fld in tr:
                g1[fld] = max(g1[fld], abs(float(tr[fld]) - float(got)))

        st = StaticState(
            t_arr=float(tr["t_arr"]), e_arr=float(tr["e_arr"]),
            cd=float(tr["cd"]), sd=float(tr["sd"]), sw=float(tr["sw"]),
            h=float(spread[i]) if i < len(spread) else 0.0,
            phi=int(tr.get("phi", 0)), rho2_used=int(tr.get("rho2_used", 0)),
            ext_shift_used=int(tr.get("ext_shift_used", 0)),
            stop=dec.stop,
        )
        sf, _flags = state_features(fd, pre, dec.stop, st, cv, GUARD_Q)
        regrets = dec.regrets()
        best = min((a.cost_h for a in dec.clean_actions), default=np.nan)
        sv = np.array([sf[n] for n in state_names], dtype=np.float32)

        for a in dec.actions:
            af = action_features(fd, pre, dec.stop, st, key_to_action(a.key), sf)
            rows.append((
                sv,
                np.array([af[n] for n in action_names], dtype=np.float32),
                ACTION_VOCAB.index(a.key),
                float(regrets.get(a.key, np.nan)),
                float(a.cost_h), float(a.std_h),
                a.ok, a.n, float(a.tauc_h), float(a.taub_h),
                int(a.clean), dec.stop,
                int(a.key == dec.chosen), int(dec.tiebreak),
                float(best), len(dec.actions), len(dec.clean_actions),
            ))
        rep["n_dec"] += 1

    rep["g1"] = g1
    rep["n_rows"] = len(rows)
    return rows, rep


def _worker(args):
    log_path, state_names, action_names = args
    try:
        return extract_one(log_path, state_names, action_names)
    except Exception as exc:               # one bad run must not kill the batch
        return None, dict(instance=os.path.basename(log_path), run_id="",
                          status=f"error:{type(exc).__name__}:{exc}",
                          n_dec=0, n_rows=0, g1={}, g2_max=0.0,
                          g3_delta=None, g4_bad=0, n_log_dec=0, halted=False)


def _column_order():
    """Fix the feature order once, from a real call."""
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "*LA_MIPTAIL*.txt")))
    inst = os.path.basename(logs[0]).split("_LA_MIPTAIL")[0]
    fd, _, _, _ = load_instance_json(os.path.join(INST_DIR, inst + ".json"))
    fd["_horizon_h"] = 24.0
    pre = Precomp(fd)
    st = StaticState(t_arr=float(fd.get("T_START", 8.0)), e_arr=float(fd["E0"]),
                     cd=0.0, sd=0.0, sw=0.0, h=0.0, phi=0, rho2_used=0,
                     ext_shift_used=0, stop=0)
    sf, _ = state_features(fd, pre, 0, st, 0.15, GUARD_Q)
    af = action_features(fd, pre, 0, st,
                         dict(y=0, break_type=None, rest_type=None), sf)
    return list(sf.keys()), list(af.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all logs")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="dataset.npz")
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(LOG_DIR, "*LA_MIPTAIL*.txt")))
    if args.limit:
        logs = logs[: args.limit]
    state_names, action_names = _column_order()
    print(f"[extract] {len(logs)} LA logs | {len(state_names)} state + "
          f"{len(action_names)} action = {len(state_names)+len(action_names)} features")

    t0 = time.time()
    all_rows, reps = [], []
    payload = [(lg, state_names, action_names) for lg in logs]
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            it = pool.imap(_worker, payload, 4)
            for i, (rows, rep) in enumerate(it):
                reps.append(rep)
                if rows:
                    all_rows.extend(rows)
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(logs)} rows={len(all_rows)} "
                          f"{time.time()-t0:.0f}s", flush=True)
    else:
        for i, p in enumerate(payload):
            rows, rep = _worker(p)
            reps.append(rep)
            if rows:
                all_rows.extend(rows)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(logs)} rows={len(all_rows)} "
                      f"{time.time()-t0:.0f}s", flush=True)

    print(f"[extract] done in {time.time()-t0:.0f}s")
    _report_gates(reps)
    if not all_rows:
        print("[extract] no rows — aborting")
        return

    good = [r for r in reps if r["status"] == "ok" and r["n_rows"] > 0]
    insts = sorted({r["instance"] for r in good})
    inst_ix = {n: i for i, n in enumerate(insts)}
    fams = sorted({n.rsplit("_", 1)[0] for n in insts})
    fam_ix = {n: i for i, n in enumerate(fams)}

    X = np.concatenate([np.stack([r[0] for r in all_rows]),
                        np.stack([r[1] for r in all_rows])], axis=1)
    rest = np.array([r[2:] for r in all_rows], dtype=np.float64)

    inst_col, fam_col, seed_col = [], [], []
    for rep in good:
        nm = rep["instance"]
        inst_col += [inst_ix[nm]] * rep["n_rows"]
        fam_col += [fam_ix[nm.rsplit("_", 1)[0]]] * rep["n_rows"]
        seed_col += [int(nm.rsplit("_", 1)[1])] * rep["n_rows"]

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, args.out)
    np.savez_compressed(
        out,
        X=X.astype(np.float32),
        feature_names=np.array(state_names + action_names),
        n_state=len(state_names),
        action_ix=rest[:, 0].astype(np.int16),
        regret=rest[:, 1].astype(np.float32),
        cost=rest[:, 2].astype(np.float64),
        std=rest[:, 3].astype(np.float32),
        ok=rest[:, 4].astype(np.int16),
        n_scen=rest[:, 5].astype(np.int16),
        tauc=rest[:, 6].astype(np.float32),
        taub=rest[:, 7].astype(np.float32),
        clean=rest[:, 8].astype(np.int8),
        stop=rest[:, 9].astype(np.int32),
        chosen=rest[:, 10].astype(np.int8),
        tiebreak=rest[:, 11].astype(np.int8),
        best_cost=rest[:, 12].astype(np.float64),
        n_actions=rest[:, 13].astype(np.int8),
        n_clean=rest[:, 14].astype(np.int8),
        instance_ix=np.array(inst_col, dtype=np.int32),
        family_ix=np.array(fam_col, dtype=np.int16),
        seed=np.array(seed_col, dtype=np.int16),
        instances=np.array(insts),
        families=np.array(fams),
        action_vocab=np.array(ACTION_VOCAB),
    )
    print(f"[extract] wrote {out}  X={X.shape}  instances={len(insts)}  "
          f"families={len(fams)}")


def _report_gates(reps):
    ok = [r for r in reps if r["status"] == "ok"]
    bad = [r for r in reps if r["status"] != "ok"]
    print(f"\n{'='*72}\nVALIDATION GATES\n{'='*72}")
    print(f"usable runs: {len(ok)}/{len(reps)}   excluded: {len(bad)}")
    for st in sorted({r["status"] for r in bad}):
        print(f"    {st}: {sum(1 for r in bad if r['status'] == st)}")
    if not ok:
        return
    print("\nG1  log state line vs solution trajectory (max abs diff over runs)")
    for fld, tol in G1_TOL.items():
        v = np.array([r["g1"].get(fld, 0.0) for r in ok])
        n_bad = int((v > tol).sum())
        flag = "OK " if n_bad == 0 else "FAIL"
        print(f"    [{flag}] {fld:10s} max={v.max():.4f}  tol={tol:<6.3f} "
              f"runs over tol: {n_bad}")
    g2 = np.array([r["g2_max"] for r in ok])
    n2 = int((g2 > TIME_TOL).sum())
    print(f"\nG2  [{'OK ' if n2 == 0 else 'FAIL'}] t_arr[k+1] == td[k]+D_actual[k]"
          f"   max={g2.max():.2e} h  runs over {TIME_TOL:g}: {n2}")
    g3 = np.array([r["g3_delta"] for r in ok if r["g3_delta"] is not None])
    if len(g3):
        n3 = int((np.abs(g3) > TIME_TOL).sum())
        print(f"G3  [{'OK ' if n3 == 0 else 'FAIL'}] arrival - T_START == duration_h"
              f"  max|d|={np.abs(g3).max():.2e} h  over {len(g3)} runs, "
              f"violations: {n3}")
    g4 = np.array([r["g4_bad"] for r in ok])
    n4 = int((g4 > 0).sum())
    print(f"G4  [{'OK ' if n4 == 0 else 'FAIL'}] reconstructed spread in "
          f"[0, 15] h   runs with any bad stop: {n4}")
    nd = sum(r["n_dec"] for r in ok)
    nl = sum(r["n_log_dec"] for r in ok)
    print(f"\ndecisions joined: {nd} / {nl} logged ({100*nd/max(nl,1):.1f}%)   "
          f"rows: {sum(r['n_rows'] for r in ok)}")


if __name__ == "__main__":
    main()
