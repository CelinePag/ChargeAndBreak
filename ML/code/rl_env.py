"""
rl_env.py — the environment for the reinforcement-learning work (ML/RL_THEORY.md)
=================================================================================
One route is one episode.  The vehicle is src's BEHDV with its default rules
(the user's call, 2026-10-26: keep the simulator as it is), the decision is
taken behind the students' shield exactly as policy_core.StudentPolicy.decide
takes it, and the travel times are either the instance's stored realisation
(validation / test, where the LA and the oracle exist) or FRESH draws from the
instance's own law (training): generate_scenarios over legs [0, N) uses the
same sample_multipliers + ECR energy as instance_io used to draw D_real/E_real.

What an episode reports (RL_THEORY.md §3):

  duration  arrival - T_START, the manuscript's route duration
  cost      duration + BETA * window misses (BETA = 0.5 h), the objective
  dwell     duration - sum of the realised leg times.  The leg times are
            exogenous -- indexed by leg, never by clock or action -- so on one
            draw they are the same for every policy, and any paired difference
            in cost is a difference in dwell (+ BETA * misses).  It is the
            reward RL should use: same optimum, less variance.
  rests, r1, r2, charges, misses, failed / halt_reason, decisions
  changed   decisions where the chooser overruled the student's argmin
  overrun   diagnostic only, NOT a rule: rests that began after 13 h (r1) /
            15 h (r2) of spread, plus a final shift longer than 13 h.  BEHDV
            does not test these by default and they stay untested here; the
            count is kept so a gain bought by overrunning can be seen.

run_batch drives many vehicles on ONE route in lockstep (one stop per tick,
one batched prediction per tick, as fpi_policy.drive_fleet does), each with
its own travel times and its own CHOOSER.  A chooser sees the student's
shielded candidates and scores and returns the index to execute; None is the
student itself.  So one batch can hold the student and every variant of it
under the SAME draws (common random numbers): their differences are then due
to the decisions alone.

Nothing here writes to disk; src/ is imported read-only.
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.simulation.BEHDV import BEHDV                               # noqa: E402
from src.simulation.scenarios import generate_scenarios              # noqa: E402

from features import Precomp, action_key, charger_curve              # noqa: E402
from policy_core import _norm                                        # noqa: E402
from rollout_policy import step                                      # noqa: E402

BETA = 0.5          # h per missed window (settings.BETA_TW)


def final_student(kind, tag):
    """A student under the paper's final protocol (eval suffix g99srNm):
    0.99 drive guard, spread-room check, nominal input features, charge-then-
    rest pays M_seq."""
    from policy_core import load_policy
    pol = load_policy(kind, tag, guard_q=0.99, spread_room=True)
    pol.feature_q = None
    pol.pay_mseq = True
    return pol


def load_route(inst, root=None, path=None):
    """(fd, D_real, E_real, cv) of an instance: a base-case one by name, or
    any instance file by `path` (e.g. a mixed-charger route)."""
    from src.instance_gen.instance_io import load_instance_json
    if path is None:
        path = os.path.join(root or os.path.join(_ROOT, "instances"), inst + ".json")
    fd, D_real, E_real, cv = load_instance_json(path)
    fd["_horizon_h"] = 24.0
    return fd, D_real, E_real, cv


def route_set(routes, split, variants="pmix,mix"):
    """[(variant, name, path)].  routes "base": the base-case routes with an LA
    run, split fit / stop (validation) / test.  routes "mixed": the mixed-
    charger routes of mixed_eval (variants pmix / dmix / mix), split test
    (seeds 22-25) or stop (= mixed_eval's "val", seeds 20-21)."""
    if routes == "base":
        root = os.path.join(_ROOT, "instances")
        return [("base", n, os.path.join(root, n + ".json")) for n in split_routes(split)]
    import mixed_eval as me
    if split not in ("test", "stop"):
        raise ValueError("mixed routes: split test or stop (validation)")
    me.ROUTE_SET = "test" if split == "test" else "val"
    return list(me.instances(variants.split(","), uniform=False))


def split_routes(split):
    """Base-case routes with an LA run (dataset.npz), by route-seed split:
    fit 1-19, stop (validation) 20-21, test 22-25."""
    rng = dict(fit=range(1, 20), stop=range(20, 22), test=range(22, 26))[split]
    d = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "dataset.npz"),
                allow_pickle=True)
    insts = sorted({str(x) for x in d["instances"]})
    return [i for i in insts if int(i.rsplit("_", 1)[1]) in rng]


def rests_needed(fd, drive_left, sd, ext_left):
    """Fewest further daily rests that let the nominal remaining driving fit,
    counting the 10 h days: with n more rests there are n + 1 shifts (this one
    and n new ones), each allowed 9 h of driving, and up to `ext_left` of them
    1 h more (Art. 6(1)).  Returns (n, slack): slack = driving allowance left
    over in the last shift.  Driving only (breaks, charging and the spread are
    dwell, not counted); fpi_policy.walk_to_end ignores the 10 h days, which
    is why it miscounts the rests at the end of a route."""
    t9 = float(fd["Tdrv_sh1"])
    ext_h = float(fd.get("Tdrv_sh2", 10.0)) - t9
    ext_left = max(0, int(ext_left))
    n = 0
    while True:
        cap = (t9 - sd) + t9 * n + ext_h * min(ext_left, n + 1)
        if cap >= drive_left - 1e-9 or n > 50:
            return n, cap - drive_left
        n += 1


def crc(text: str) -> int:
    """A seed from a string: reproducible, independent of run order."""
    return zlib.crc32(text.encode())


def fresh_draws(fd, cv, n, tag):
    """`n` full-route travel-time draws from the instance's own law, seeded by
    `tag` (e.g. "<instance>|rl|<round>").  Each is {"D": {leg: h}, "E": {leg: kWh}}."""
    return generate_scenarios(fd, 0, int(fd["N"]), n_scenarios=int(n), cv=cv,
                              seed=crc(tag))


def realised(D_real, E_real):
    """The stored realisation in the same shape as a draw."""
    return {"D": {i: float(d) for i, d in enumerate(D_real)},
            "E": {i: float(e) for i, e in enumerate(E_real)}}


def scores(base, cost, feas):
    """The student's own decision score (StudentPolicy.decide)."""
    return cost + 1e6 * (feas < base.feas_thr)


def _summary(fd, veh, st, n_dec):
    N = int(fd["N"])
    T0 = float(fd.get("T_START", 8.0))
    done = (not veh.is_halted) and veh.stop >= N
    dur = veh.t_arr - T0
    acts = veh.actions
    rests = [_norm(a.get("rest_type")) for a in acts]
    out = dict(
        failed=not done,
        halt_reason=veh.halt_reason,
        duration=dur if done else None,
        cost=(dur + BETA * len(veh.tw_misses)) if done else None,
        dwell=(dur - float(sum(veh.D_actual_list))) if done else None,
        misses=len(veh.tw_misses),
        rests=sum(r is not None for r in rests),
        r1=sum(r == "r1" for r in rests),
        r2=sum(r == "r2" for r in rests),
        charges=sum(int(a.get("y", 0)) == 1 for a in acts),
        decisions=n_dec,
        changed=st["changed"],
        overrun=st["overrun"] + int(done and veh.h > float(fd.get("Tspr1", 13.0)) + 1e-3),
    )
    return out


class ForceFirst:
    """A chooser that executes the action with key `key` at its first decision
    (raises if it is not among the shielded candidates), then hands over to
    `then` (None: the student).  One instance per job: it keeps state."""

    def __init__(self, key, then=None):
        self.key, self.then, self.used = key, then, False

    def choose(self, fd, pre, veh, legal, rows, score, flags):
        if not self.used:
            self.used = True
            for j, a in enumerate(legal):
                if action_key(a.get("y", 0), a.get("break_type"),
                              a.get("rest_type")) == self.key:
                    return j
            raise ValueError(f"{self.key} not among the candidates at stop {veh.stop}")
        if self.then is not None:
            return self.then.choose(fd, pre, veh, legal, rows, score, flags)
        return int(np.argmin(score))


def run_batch(base, fd, cv, jobs, pre=None, on_decision=None, start=None,
              return_vehicles=False):
    """Drive len(jobs) vehicles to the destination on route `fd`.

    jobs: [(chooser, draw), ...].  chooser is None (the student) or an object
    with choose(fd, pre, veh, legal, rows, score, flags) -> index into legal.
    on_decision(i, veh, legal, rows, score, flags, j): optional hook, called
    for every decision before it is executed (data collection).
    start: optional list of BEHDV, one per job (clones, e.g. rollout_policy.
    clone of a mid-route vehicle), to continue from instead of the origin;
    a draw's legs before the vehicle's stop are then simply unused.
    Returns one _summary dict per job (for the whole route, so paired
    differences between jobs that share a start are differences from there).

    With chooser None and the stored realisation this makes exactly the
    decisions of policy_core.run_student (checked by rl_check.py).
    """
    pre = pre or Precomp(fd)
    fd.setdefault("_horizon_h", 24.0)
    N = int(fd["N"])
    vehs = list(start) if start is not None else [BEHDV(fd) for _ in jobs]
    st = [dict(changed=0, overrun=0) for _ in jobs]
    n_dec = [0] * len(jobs)
    Tspr1 = float(fd.get("Tspr1", 13.0))
    Tspr2 = float(fd.get("Tspr2", 15.0))
    pay = bool(getattr(base, "pay_mseq", False))
    # a route whose chargers differ: the policy and the clamp read fd["Tbar"],
    # set to the curve of the stop's charger (or the next one), as run_student
    curves, tbar = fd.get("TbarK"), fd["Tbar"]

    def at(s):
        if curves:
            fd["Tbar"] = charger_curve(fd, s if s in pre.K else int(pre.next_cs[s]))

    live = [i for i, v in enumerate(vehs) if not v.is_halted and v.stop < N]
    try:
        while live:
            built = []
            for i in live:
                at(vehs[i].stop)
                built.append(base._rows(fd, pre, vehs[i].stop, vehs[i], cv))
            cost, feas = base._predict(np.vstack([b[1] for b in built]))
            sc_all = scores(base, np.asarray(cost, dtype=float),
                            np.asarray(feas, dtype=float))
            picks, o = [], 0
            for i, (legal, rows, flags) in zip(live, built):
                n = len(legal)
                sc = sc_all[o:o + n]
                o += n
                j0 = int(np.argmin(sc))
                ch = jobs[i][0]
                if ch is None:
                    j = j0
                else:
                    at(vehs[i].stop)
                    j = int(ch.choose(fd, pre, vehs[i], legal, rows, sc, flags))
                if on_decision is not None:
                    on_decision(i, vehs[i], legal, rows, sc, flags, j)
                st[i]["changed"] += int(j != j0)
                n_dec[i] += 1
                picks.append((i, legal[j], rows[j:j + 1], flags))
            charging = [k for k, p in enumerate(picks) if int(p[1].get("y", 0)) == 1]
            raw = {}
            if charging:
                if getattr(base, "kind", "") == "gbt":
                    vals = base.tauc.predict(np.vstack([picks[k][2] for k in charging]))
                    raw = {k: float(x) for k, x in zip(charging, vals)}
                else:
                    raw = {k: float(base._predict_tauc(picks[k][2])) for k in charging}
            for k, (i, act, row, flags) in enumerate(picks):
                v = vehs[i]
                s = v.stop
                at(s)
                tc = base._charge_hours(fd, s, v, act, row, flags, raw=raw.get(k))
                h0, ta0 = float(v.h), float(v.t_arr)
                draw = jobs[i][1]
                step(v, fd, act, tc, draw["D"][s], draw["E"][s], pay)
                rst = _norm(act.get("rest_type"))
                if rst in ("r1", "r2"):
                    h_pre = h0 + (v.td_list[-1] - ta0 - float(v.durations[-1]["taur"]))
                    cap = Tspr1 if rst == "r1" else Tspr2
                    st[i]["overrun"] += int(h_pre > cap + 1e-3)
            live = [i for i in live if not vehs[i].is_halted and vehs[i].stop < N]
    finally:
        fd["Tbar"] = tbar
    out = [_summary(fd, v, st[i], n_dec[i]) for i, v in enumerate(vehs)]
    return (out, vehs) if return_vehicles else out
