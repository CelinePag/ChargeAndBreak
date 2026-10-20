"""
extract_oracle.py — a training set whose teacher is the hindsight ORACLE
=======================================================================
Option A of the 2026-10-20 review: behaviour cloning of the stored oracle
schedules, no new solver call.  Same routes, features, splits and file format
as dataset.npz (the LA teacher), so every trainer runs on it unchanged except
for one optional column (`tauc_ok`, below).

Labels
------
The oracle gives ONE action per stop, not a cost for every action, so:
  regret  0 for the oracle's action, M (= 1 h) for every other clean action
          -> the cost head becomes a score of "is this the oracle's action";
             with the listwise term (tau 0.25 h) PyTorch is close to plain
             cross-entropy on the oracle's choice
  clean   the action passes the safety rules at the 0.99 travel-time quantile
          (supervisor.action_passes + the spread-room check), OR it is the
          oracle's action.  The oracle knows the realised travel times, so its
          action can fail the 0.99 guard; it is kept as a positive anyway.
          The feasibility head therefore learns the rule layer, which the
          policy applies again at driving time -- it carries no teacher signal.
  tauc    the oracle's charge, on the oracle's action only (`tauc_ok` = 1);
          the charge head trains on those rows alone.
  std 0, n_scen 1 -> every decision has weight 1 (margin M, no sampling noise).

States
------
The oracle solution stores the state at every stop (ta, ea, cd, sd, sw, h) and
the executed durations and flags, and it never waits (checked: wait = 0 on
every sampled oracle; BEHDV never inserts waiting either).  The schedule is
replayed through BEHDV with exactly those durations, so the remaining three
state fields (phi, rho2_used, ext_shift_used) come from the simulator's own
accounting, and the replayed state is checked against the stored one at every
stop (gate G1, max deviation per route).  A route whose replay drifts, halts,
or meets an oracle action outside enumerate_actions is reported; its rows are
kept only up to the first such stop.

    python ML/code/extract_oracle.py                 # -> ML/data/dataset_oracle.npz
    python ML/code/extract_oracle.py --limit 20      # quick check of the gates
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.instance_gen.instance_io import load_instance_json      # noqa: E402
from src.simulation.BEHDV import BEHDV                            # noqa: E402
from src.simulation.Simulation import enumerate_actions           # noqa: E402
from src.simulation.supervisor import action_passes, compute_flags  # noqa: E402

from features import (ACTION_VOCAB, Precomp, action_features,     # noqa: E402
                      action_key, state_features)
from extract import rows_to_arrays                                # noqa: E402
from policy_core import charge_needed_to_reach, spread_room       # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SOL = os.path.join(_ROOT, "solutions", "basecase")
INST = os.path.join(_ROOT, "instances")

M_REGRET = 1.0        # h, pseudo-regret of any action the oracle did not take
GUARD_Q = 0.99        # the shield's quantile, for the `clean` label
FEATURE_Q = None      # nominal features, as the LA dataset (extract.GUARD_Q)
TOL = dict(t=0.02, e=1.0, cd=0.02, sd=0.02, sw=0.02, h=0.02)   # h / kWh


def _flag(s, k):
    return int(round(float(s.get(k, 0) or 0)))


def oracle_key(s):
    brk = "b45" if _flag(s, "b45") else "b15" if _flag(s, "b15") else \
          "b30" if _flag(s, "b30") else None
    rst = "r1" if _flag(s, "rho1") else "r2" if _flag(s, "rho2") else None
    return action_key(_flag(s, "y"), brk, rst), brk, rst


def _clean(fd, stop, veh, a, flags):
    if not action_passes(fd, stop, veh, a, flags):
        return False
    r = spread_room(fd, stop, a, flags)
    if r is None:
        return True
    need = charge_needed_to_reach(fd, veh.e_arr, flags) if int(a.get("y", 0)) else 0.0
    return r >= need - 1e-9


def extract_route(job):
    inst, state_names, action_names = job
    rep = dict(instance=inst, status="ok", n_dec=0, n_rows=0, n_stops=0,
               dev={k: 0.0 for k in TOL}, first_bad=None, chosen_dirty=0)
    try:
        with open(os.path.join(SOL, f"oracle_{inst}.json")) as fh:
            o = json.load(fh)
        if not o.get("feasible") or not o.get("sol"):
            rep["status"] = "no_oracle"
            return [], [], rep
        rep["oracle_gap"] = o.get("gap")
        fd, D, E, cv = load_instance_json(os.path.join(INST, inst + ".json"))
        fd["_horizon_h"] = 24.0
        pre = Precomp(fd)
        veh = BEHDV(fd)
        sol = {int(s["i"]): s for s in o["sol"]}
        N = int(fd["N"])
        L_set = {int(x) for x in fd.get("L", [])}
        rows, tauc_ok = [], []
        for k in range(N):
            s = sol[k]
            if veh.stop != k:
                rep.update(status="desync", first_bad=k)
                break
            # -- G1: the replayed state against the oracle's stored one -------
            # sw is compared only where no work happens at the stop: the MILP
            # books the work done AT stop k into sw[k] on arrival, BEHDV adds
            # it on departure, so the two differ by that work at a charger,
            # customer or layby break and agree again at the next stop.
            no_work = not (s.get("is_C") or s.get("is_K") or k in L_set)
            for fld, got, want in (("t", veh.t_arr, s["ta"]), ("e", veh.e_arr, s["ea"]),
                                   ("cd", veh.cd, s["cd"]), ("sd", veh.sd, s["sd"]),
                                   ("sw", veh.sw if no_work else s["sw"], s["sw"]),
                                   ("h", veh.h, s["h"])):
                rep["dev"][fld] = max(rep["dev"][fld], abs(float(got) - float(want)))
            if any(rep["dev"][f] > TOL[f] for f in TOL):
                rep.update(status="drift", first_bad=k)
                break
            key, brk, rst = oracle_key(s)
            acts = enumerate_actions(k, veh, fd, charge_only=False)
            keys = [action_key(a.get("y", 0), a.get("break_type"), a.get("rest_type"))
                    for a in acts]
            if key not in keys:
                rep.update(status=f"illegal:{key}", first_bad=k)
                break
            rep["n_stops"] += 1
            if len(acts) >= 2:
                sf, _ = state_features(fd, pre, k, veh, cv, FEATURE_Q)
                flags = compute_flags(fd, k, veh, cv, GUARD_Q)
                sv = np.array([sf[n] for n in state_names], dtype=np.float32)
                cl = [_clean(fd, k, veh, a, flags) or kk == key for a, kk in zip(acts, keys)]
                rep["chosen_dirty"] += int(not _clean(fd, k, veh, acts[keys.index(key)], flags))
                n_clean = int(sum(cl))
                for a, kk, c in zip(acts, keys, cl):
                    af = action_features(fd, pre, k, veh, a, sf)
                    ch = kk == key
                    reg = 0.0 if ch else (M_REGRET if c else np.nan)
                    tc = float(s["tauc"]) if ch and int(a.get("y", 0)) else 0.0
                    rows.append((sv, np.array([af[n] for n in action_names], dtype=np.float32),
                                 ACTION_VOCAB.index(kk), reg, reg, 0.0, int(c), 1,
                                 tc, float(s["taub"]) if ch else 0.0, int(c), k,
                                 int(ch), 0, 0.0, len(acts), n_clean))
                    tauc_ok.append(int(ch and int(a.get("y", 0)) == 1))
                rep["n_dec"] += 1
            # -- execute exactly what the oracle executed ------------------------
            a_exec = acts[keys.index(key)]
            mock = dict(feasible=True, sol=[dict(
                i=0, taub=float(s["taub"]), tauc=float(s["tauc"]), taur=float(s["taur"]),
                tauq=float(s["tauq"]), y=_flag(s, "y"), b45=_flag(s, "b45"),
                b15=_flag(s, "b15"), b30=_flag(s, "b30"), rho1=_flag(s, "rho1"),
                rho2=_flag(s, "rho2"), sigma=_flag(s, "sigma"),     # charge-then-rest
                is_C=bool(s.get("is_C")), is_K=bool(s.get("is_K")))])  # bay overhead M_seq
            veh.advance(action=a_exec, D_next=float(D[k]), E_next=float(E[k]), milp_sol=mock)
            if veh.is_halted:
                rep.update(status=f"halted:{veh.halt_reason}", first_bad=k)
                break
        else:
            rep["dev_end"] = abs(float(veh.t_arr) - float(sol[N]["ta"]))
        rep["n_rows"] = len(rows)
        return rows, tauc_ok, rep
    except Exception as exc:                 # one bad route must not kill the batch
        rep["status"] = f"error:{type(exc).__name__}:{exc}"
        return [], [], rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(DATA, "dataset_oracle.npz"))
    args = ap.parse_args()

    ref = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    names = [str(x) for x in ref["feature_names"]]
    n_state = int(ref["n_state"])
    state_names, action_names = names[:n_state], names[n_state:]
    # the SAME routes as the LA teacher's dataset, so only the teacher differs
    insts = [str(x) for x in ref["instances"]]
    if args.limit:
        insts = insts[: args.limit]
    print(f"[oracle] {len(insts)} routes (those of dataset.npz), {args.jobs} jobs", flush=True)

    t0 = time.time()
    all_rows, all_tok, inst_rows, reps = [], [], [], []
    with Pool(args.jobs) as pool:
        for i, (rows, tok, rep) in enumerate(pool.imap(
                extract_route, [(n, state_names, action_names) for n in insts], 2)):
            reps.append(rep)
            if rows:
                all_rows += rows
                all_tok += tok
                inst_rows.append((rep["instance"], len(rows)))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(insts)}  {time.time()-t0:.0f}s", flush=True)

    st = {}
    for r in reps:
        k = r["status"].split(":")[0]
        st[k] = st.get(k, 0) + 1
    dev = {f: max(r["dev"][f] for r in reps) for f in TOL}
    ends = [r["dev_end"] for r in reps if "dev_end" in r]
    print(f"\n[gates] status: {st}")
    print(f"[gates] max replay deviation over routes: "
          + "  ".join(f"{f} {v:.4f}" for f, v in dev.items())
          + f"   | arrival at destination: max {max(ends) if ends else float('nan'):.4f} h")
    print(f"[labels] oracle actions that fail the 0.99 safety rules: "
          f"{sum(r['chosen_dirty'] for r in reps)} of {sum(r['n_dec'] for r in reps)} decisions")
    for r in reps:
        if r["status"] != "ok":
            print(f"   {r['instance']}: {r['status']} at stop {r['first_bad']} "
                  f"(kept {r['n_dec']} decisions)")

    arr = rows_to_arrays(all_rows)
    inst_names = [n for n, _ in inst_rows]
    arr["instance_ix"] = np.concatenate([np.full(c, j, dtype=np.int32)
                                         for j, (_, c) in enumerate(inst_rows)])
    fams = sorted({n.rsplit("_", 1)[0] for n in inst_names})
    fam_of = {n: fams.index(n.rsplit("_", 1)[0]) for n in inst_names}
    arr["family_ix"] = np.concatenate([np.full(c, fam_of[n], dtype=np.int16)
                                       for n, c in inst_rows])
    arr["seed"] = np.concatenate([np.full(c, int(n.rsplit("_", 1)[1]), dtype=np.int16)
                                  for n, c in inst_rows])
    np.savez_compressed(
        args.out, physics="base", teacher="oracle", feature_names=np.array(names),
        n_state=n_state, instances=np.array(inst_names), families=np.array(fams),
        action_vocab=np.array(ACTION_VOCAB), tauc_ok=np.array(all_tok, dtype=np.int8),
        **arr)
    with open(args.out.replace(".npz", "_report.json"), "w") as fh:
        json.dump(reps, fh, indent=1)
    print(f"\n[saved] {args.out}: {len(all_rows)} rows, "
          f"{sum(r['n_dec'] for r in reps)} decisions, {len(inst_names)} routes, "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
