"""
rollout.py — the ROAD TEST: the student policy drives in the real simulator
===========================================================================
EASY MODE
---------
train.py graded flashcards ("would you have answered like the teacher?").
This script is the driving exam: the student is put in the truck, the world
rolls its dice (the SAME recorded travel-time realisations every benchmark
method faced), and we measure what actually matters — total route duration,
window misses, strandings, HoS breaches — with the exact machinery that
grades every other policy.  A policy can match the teacher on 90% of
flashcards and still fail here (its own small mistakes create situations it
never studied) — which is precisely why closed-loop numbers, not accuracy,
go in the paper.

HOW IT PLUGS IN (and why this is the honest way)
------------------------------------------------
run_simulation_precomputed() in src/simulation/Simulation.py accepts an
`external_policy` callback that replaces ONLY the decision step of the LA
loop.  Vehicle physics, HoS clock updates, ferry handling, TW misses,
metrics, and the saved JSON are byte-identical to every other run — so the
student's numbers are comparable by construction, and `decision_times`
records the student's true online latency (a forward pass, ~ms).

The student outputs a FULL decision:
  * discrete class -> (y, break_type, rest_type)  [feasibility-masked]
  * charge duration tau_c from the regression head, clamped to
      [ time to reach (Emin + energy-to-next-charger + safety buffer),
        time to charge to full ]
    The lower clamp is the same worst-case reachability idea every benchmark
    policy uses; if the net proposes an unsafe short charge we extend it and
    LOG the event (clamp activations are reported — a silent safety net
    would hide the very failure mode we claim to study).

Durations are handed to vehicle.advance() through a mock `milp_sol` with the
same duration semantics as the simulator's own fallback path (parallel
break-in-charge crediting: taub = max(0, T_break - tauc)).

Run (repo root), examples:
  python ML/code/rollout.py --split test_id --limit 5      # quick check
  python ML/code/rollout.py --split test_id ood_route ood_tw ood_both
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ML", "code"))

from model import StudentPolicy, build_mask                      # ML/code
from extract_dataset import build_instance_context, features_at, split_of

# Sandbox EVERY output of this experiment inside ML/ before importing anything
# that writes.  The reporting pipeline globs <root>/solutions by method name,
# so a stray STUDENT run outside ML/ would silently enter the manuscript's
# tables; redirecting here makes that impossible rather than merely unlikely.
from src import paths as _paths
_paths.redirect_outputs(os.path.join(ROOT, "ML"))

from src.instance_gen.instance_io import load_instance_json
from src.simulation.Simulation import run_simulation_precomputed
from src.simulation.supervisor import compute_flags, action_passes
from src.settings import energy_at_quantile, TRAVEL_TIME_CV_TARGET

INST_DIR = os.path.join(ROOT, "instances")   # inputs stay shared (read-only)

# ── Guard levels: mirror what the TEACHER actually ran ───────────────────────
# The staged MIPTAIL runs carry la_energy_quantile = 0.5 (730/772) and
# prune_quantile = None (772/772).  So the teacher sized its COMMITTED charge
# to cover the legs up to the next CS at the 0.5-quantile of CONSUMPTION, and
# applied no flag-based pruning on the time side — it let the 25-scenario
# ensemble expose time-infeasible actions instead.
#
# The student has no ensemble, so it needs explicit forcing rules; they are
# the same checks (supervisor.compute_flags / action_passes) that greedy and
# the S1 supervisor use, at a configurable quantile.
#
# NOTE ON DIRECTION (subtle, matters): a quantile q means opposite things on
# the two sides.  Energy rises when a leg is driven FAST, so the q-quantile of
# consumption is driven by the (1-q)-quantile of xi.  Time rises when a leg is
# SLOW.  Because the shifted-lognormal xi has median 0.957 < 1, q = 0.5 gives
# a mild energy margin but a time factor BELOW nominal — i.e. q = 0.5 on the
# time side is anti-conservative.  We therefore keep the two knobs separate:
#   ENERGY_Q = 0.5   (teacher-matched, sizes the charge)
#   GUARD_Q  = None  (teacher-matched default: nominal time checks)
# and sweep GUARD_Q in {None, 0.95, 1.0} as an ablation, since the student's
# observed failures are marginal HoS overruns on slower-than-nominal legs.
ENERGY_Q_DEFAULT = 0.5
GUARD_Q_DEFAULT  = None


# ── PWL charging curve helpers ───────────────────────────────────────────────
def charge_time_to(ea: float, target: float, Ebar, Tbar) -> float:
    """Hours on the piecewise-linear curve to charge from `ea` to `target`.

    The curve is given as breakpoints (Ebar[k], Tbar[k]) measured from an
    empty battery; charging from ea to target is the horizontal distance
    between their time-coordinates.  Concave curve => charging is slower
    near the top — exactly why "always charge to full" (greedy) wastes time
    and why the teacher's partial charges are worth imitating.
    """
    def t_of(e):
        e = min(max(e, Ebar[0]), Ebar[-1])
        for k in range(1, len(Ebar)):
            if e <= Ebar[k] + 1e-9:
                f = (e - Ebar[k-1]) / (Ebar[k] - Ebar[k-1])
                return Tbar[k-1] + f * (Tbar[k] - Tbar[k-1])
        return Tbar[-1]
    return max(0.0, t_of(target) - t_of(ea))


# ── the student as an external_policy callback ───────────────────────────────
def energy_to_next_cs_at_q(full_data: dict, stop: int, q: float | None,
                           cv: float) -> float:
    """kWh to reach the next CS, sizing each leg at the q-quantile of
    CONSUMPTION — the same rule MILP._build_sub_data applies to the teacher's
    committed charge (settings.energy_at_quantile over the legs up to the
    next station).  q=None falls back to the instance's nominal energies."""
    N, K_set = full_data["N"], set(full_data["K"])
    km, D, E = full_data.get("km", {}) or {}, full_data["D"], full_data["E"]
    total, cur = 0.0, stop
    while cur < N:
        e_nom = float(E.get(cur, 0.0))
        L, d = km.get(cur), D.get(cur)
        total += (max(e_nom, energy_at_quantile(L, d, q, cv))
                  if (q and L and d) else e_nom)
        cur += 1
        if cur in K_set or cur == N:
            break
    return total


def make_policy(ckpt_path: str, raw_inst: dict, full_data: dict, stats: dict,
                energy_q: float | None = ENERGY_Q_DEFAULT,
                guard_q: float | None = GUARD_Q_DEFAULT,
                cv: float = TRAVEL_TIME_CV_TARGET,
                no_split: bool = False):
    ck  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = [tuple(c) for c in ck["classes"]]
    k_look  = int(ck.get("k_look", 10))    # features must be built the same
    net = StudentPolicy(len(ck["mu"]), len(classes), hidden=ck["hidden"])
    net.load_state_dict(ck["state"]); net.eval()
    mu, sd = torch.as_tensor(ck["mu"]), torch.as_tensor(ck["sd"])
    ctx  = build_instance_context(raw_inst)
    Ebar = [float(raw_inst["Ebar"][str(k)]) for k in range(len(raw_inst["Ebar"]))]
    Tbar = [float(raw_inst["Tbar"][str(k)]) for k in range(len(raw_inst["Tbar"]))]
    Emin, Ecap = float(raw_inst["Emin"]), float(raw_inst["Ecap"])
    Q = full_data["Q"]

    def policy(fd, stop, vehicle):
        # 1. state dict, exactly the fields extract_dataset trained on
        st = dict(stop=stop, t_arr=vehicle.t_arr, e_arr=vehicle.e_arr,
                  cd=vehicle.cd, sd=vehicle.sd, sw=vehicle.sw,
                  phi=vehicle.phi, rho2_used=vehicle.rho2_used,
                  ext_shift_used=vehicle.ext_shift_used)
        x_raw = torch.tensor([features_at(ctx, st, k_look)], dtype=torch.float32)
        mask  = build_mask(x_raw, classes)

        # Optional: forbid split breaks at DEPLOYMENT while keeping the model
        # trained on the teacher's full action set.  Restricting the policy is
        # legitimate (it is our choice of what the truck may do); restricting
        # the TRAINING TARGET would be relabelling the teacher, which is not.
        # Note b15 is not a cd-reset (supervisor.action_passes counts only
        # b45/b30/rest), so dropping b30 removes a genuine reset option and
        # the forcing rules will fall back to b45 or a rest.
        if no_split:
            for j, (_y, bb, _r) in enumerate(classes):
                if bb in ("b15", "b30"):
                    mask[0, j] = False

        # 2. FORCING rules — the structural mask says what is forbidden; these
        #    say what is MANDATORY.  Same checks greedy and the S1 supervisor
        #    use (supervisor.compute_flags / action_passes) at guard level
        #    `guard_q`.  Applied as a further mask restriction rather than an
        #    override, so the network still chooses freely among the actions
        #    that comply — the rules constrain, they do not decide.
        flags = compute_flags(fd, stop, vehicle, cv=cv, quantile=guard_q)
        if flags["must_charge"] or flags["must_reset_cd"] or flags["must_rest"]:
            allow = torch.zeros_like(mask)
            for j, (yy, bb, rr) in enumerate(classes):
                if not mask[0, j]:
                    continue
                cand = dict(y=int(yy),
                            break_type=None if bb == "none" else bb,
                            rest_type=None if rr == "none" else rr)
                allow[0, j] = action_passes(fd, stop, vehicle, cand, flags)
            if allow.any():
                mask = allow
                stats["forced"] += 1
            else:
                # nothing complies (e.g. must_rest at a node where the only
                # legal rest is already excluded): fall back to greedy's
                # priority — rest beats break beats charge — and record it.
                stats["forced_fallback"] += 1
                rst_t = "r2" if vehicle.rho2_used < int(fd.get("rho_bar", 3)) else "r1"
                for j, (yy, bb, rr) in enumerate(classes):
                    if rr == rst_t:
                        allow[0, j] = True
                mask = allow if allow.any() else mask

        cls, tauc = net.act((x_raw - mu) / sd, mask)
        y, brk, rst = classes[int(cls)]
        y = int(y)
        tauc = float(tauc)

        # Attribution: was a rest COMPELLED by the guard, or chosen freely by
        # the network?  A daily rest costs 9-11 h, so one spurious rest
        # dominates a whole route; this counter is what separates "our rule
        # over-fires" from "the network over-predicts r2".
        if rst != "none":
            stats["rest_forced" if flags["must_rest"] else "rest_free"] += 1
        if flags["must_rest"]:
            stats["must_rest_fired"] += 1
            if rst == "none":
                stats["must_rest_unmet"] += 1   # should be impossible

        # 3. charge sizing — cover the legs to the next CS at the q-quantile
        #    of consumption, exactly the teacher's committed-charge rule
        if y == 1:
            need    = Emin + energy_to_next_cs_at_q(fd, stop, energy_q, cv)
            t_need  = charge_time_to(vehicle.e_arr, min(need, Ecap), Ebar, Tbar)
            t_full  = charge_time_to(vehicle.e_arr, Ecap, Ebar, Tbar)
            clamped = min(max(tauc, t_need), t_full)
            if clamped > tauc + 1e-6:
                stats["clamp_up"] += 1        # net proposed an unsafe charge
            stats["tauc_pred_h"].append(tauc)
            tauc = clamped
        else:
            tauc = 0.0

        # 4. durations, mirroring vehicle.advance()'s fallback semantics
        Tb = dict(b45=fd["Tb45"], b15=fd["Tb15"], b30=fd["Tb30"])
        taub = max(0.0, Tb[brk] - tauc) if brk in Tb else 0.0   # break rides
        taur = (fd["Tr1"] if rst == "r1" else                    # inside the
                fd["Tr2"] if rst == "r2" else 0.0)               # charge dwell
        tauq = float(Q.get(stop, 0.0)) * y
        sol0 = dict(taub=taub, tauc=tauc, taur=taur, tauq=tauq, y=y,
                    b45=int(brk == "b45"), b15=int(brk == "b15"),
                    b30=int(brk == "b30"),
                    rho1=int(rst == "r1"), rho2=int(rst == "r2"))
        action = dict(y=y,
                      break_type=None if brk == "none" else brk,
                      rest_type=None if rst == "none" else rst)
        # score 0.0 keeps the LA forced-rest net dormant: the student is
        # graded on its own behaviour, rescue would mask failures.
        return action, [(action, 0.0, 0.0, 0, [])], dict(feasible=True, sol=[sol0])

    return policy


# ── instance selection by split ──────────────────────────────────────────────
def instances_for(splits: list[str], k_look: int, mode: str) -> list[str]:
    """Instances belonging to `splits`, read from the extracted dataset.

    Taking the assignment from the dataset rather than recomputing it keeps a
    single source of truth: a model can only ever be evaluated on the split
    labels it was actually trained under, even if split_of() later changes.
    """
    import numpy as _np
    from extract_dataset import tag_for
    p = os.path.join(ROOT, "ML", "data", f"dataset_{tag_for(k_look, mode)}.npz")
    if os.path.exists(p):
        z = _np.load(p, allow_pickle=False)
        sel = _np.isin(z["split"], splits)
        return sorted(set(z["instance"][sel].tolist()))
    # fallback: recompute (dataset not extracted for this K/mode)
    out = []
    for f in sorted(glob.glob(os.path.join(INST_DIR, "R*.json"))):
        name = os.path.basename(f)[:-5]
        m = re.match(r"(R[a-z]+C[a-z]+T[a-z]+)_(\d+)$", name)
        if m and split_of(m.group(1), int(m.group(2)), mode) in splits:
            out.append(name)
    return out


def main(a):
    stats = dict(clamp_up=0, tauc_pred_h=[], forced=0, forced_fallback=0,
                 rest_forced=0, rest_free=0, must_rest_fired=0,
                 must_rest_unmet=0)
    _ck = torch.load(a.model, map_location="cpu", weights_only=False)
    mode = _ck.get("split_mode", "family")
    names = instances_for(a.split, int(_ck.get("k_look", 10)), mode)
    if a.limit: names = names[:a.limit]
    guard_q = None if a.guard_q is not None and a.guard_q < 0 else a.guard_q
    print(f"rolling out {len(names)} instances from splits {a.split}  "
          f"(split_mode={mode}, energy_q={a.energy_q}, guard_q={guard_q}, "
          f"no_split={a.no_split})")
    for k, name in enumerate(names):
        raw = json.load(open(os.path.join(INST_DIR, name + ".json")))
        full_data, D_real, E_real, cv = load_instance_json(
            os.path.join(INST_DIR, name + ".json"))
        pol = make_policy(a.model, raw["instance"], full_data, stats,
                          energy_q=a.energy_q, guard_q=guard_q, cv=cv,
                          no_split=a.no_split)
        t0 = time.perf_counter()
        res = run_simulation_precomputed(
            full_data, D_real, E_real,
            n_scenarios=1, horizon_hours=24.0,       # metadata only: the
            solve_mode="mip",                        # policy never solves
            verbose=False, supervised=False,
            external_policy=pol, alg_label=a.alg)
        m = res.get("metrics", {})
        print(f"[{k+1}/{len(names)}] {name:28s} dur={res.get('duration_h', -1):7.2f}h"
              f"  infeas={m.get('run_infeasible')}  tw_miss={m.get('tw_n_misses')}"
              f"  dec_mean={1000*np.mean(res.get('decision_times', [0])):.1f}ms"
              f"  wall={time.perf_counter()-t0:.1f}s")
    n_chg = len(stats["tauc_pred_h"])
    print(f"\ncharge decisions: {n_chg} | charge-size clamp raised tauc on "
          f"{stats['clamp_up']} ({100*stats['clamp_up']/max(n_chg,1):.1f}%)"
          f" | forcing active at {stats['forced']} stops"
          f" (fallback {stats['forced_fallback']})")
    print(f"rests: {stats['rest_forced']} FORCED by must_rest, "
          f"{stats['rest_free']} chosen freely by the network"
          f"  (must_rest fired {stats['must_rest_fired']}x, "
          f"unmet {stats['must_rest_unmet']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(ROOT, "ML", "models",
                                                   "policy_v1_seed0.pt"))
    p.add_argument("--split", nargs="+", default=["test_id"],
                   choices=["train", "val", "test_id",
                            "ood_route", "ood_tw", "ood_both"])
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--alg", default="STUDENT",
                   help="method label written into run_id and metadata")
    p.add_argument("--energy-q", dest="energy_q", type=float,
                   default=ENERGY_Q_DEFAULT,
                   help="quantile of CONSUMPTION for charge sizing "
                        "(teacher used 0.5); 0 = nominal energies")
    p.add_argument("--no-split", dest="no_split", action="store_true",
                   help="forbid b15/b30 at deployment (policy restriction; "
                        "the model is still trained on the teacher's full set)")
    p.add_argument("--guard-q", dest="guard_q", type=float,
                   default=GUARD_Q_DEFAULT,
                   help="quantile for the forcing checks on the TIME side; "
                        "negative = None = nominal (teacher-matched)")
    main(p.parse_args())
