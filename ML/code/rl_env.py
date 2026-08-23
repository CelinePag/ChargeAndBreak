"""
rl_env.py — lightweight episode runner for reinforcement learning
=================================================================
WHY A SEPARATE ENV
------------------
`run_simulation_precomputed` writes a solution JSON, a log, a scenario file and
consults the oracle cache on every call.  That is right for evaluation and
ruinous for training, where we need tens of thousands of episodes.  This module
drives the same `BEHDV` vehicle through the same `advance()` transition with no
disk I/O at all, so an episode costs milliseconds rather than a tenth of a
second.  The physics, the HoS clocks and the halt rule are identical, because
they live in `BEHDV`, not here.

TWO DESIGN POINTS THAT MATTER
-----------------------------
1. **Fresh realisations every episode.**  Each instance ships ONE recorded
   travel-time realisation, and that is what evaluation uses.  Training on it
   repeatedly would let the policy memorise that particular sequence of legs —
   it would learn a *plan*, not a *policy*, and would look brilliant in
   training and useless out of sample.  We therefore draw a new realisation
   from the same shifted-lognormal law for every training episode, via the
   project's own `generate_scenarios`.  The recorded realisation is reserved
   for evaluation and never trained on.

2. **Halting must never be attractive.**  Reward is negative elapsed time, so a
   policy that ends the route early stops accumulating cost.  Left alone, the
   optimal policy would be to breach a regulation immediately.  The v1 fixed
   surcharge proved too weak in practice (see HALT_MULT below); the penalty is
   now MULTIPLICATIVE in an estimate of what completing would have cost.

REWARD
------
    r_i = -(t_arr_{i+1} - t_arr_i)              per step: elapsed hours
        - beta                                   for each window missed at i
    on halt: the episode return becomes -HALT_MULT x (estimated completion),
    so abandoning a route is always about HALT_MULT times worse than finishing
    it, whatever the route length.

Summed over a completed episode this is exactly -(duration + beta * misses),
i.e. the negative of the benchmark's own objective.  Training and evaluation
therefore optimise the same quantity, which is the single most important
property of the whole setup.
"""
from __future__ import annotations
import numpy as np
import torch

from src.simulation.BEHDV import BEHDV
from src.simulation.scenarios import generate_scenarios
from src.simulation.supervisor import compute_flags, action_passes

from extract_dataset import build_instance_context, features_at
from model import build_mask
from rollout import charge_time_to, energy_to_next_cs_at_q

# ── Halt penalty ────────────────────────────────────────────────────────────
# v1 used a FIXED surcharge (24 h) on top of the unfinished remainder.  That
# was measurably too weak: halting a 100 h route at 50 h cost 50 + 24 + ~40 =
# 114 against ~100 for completing, i.e. only ~14% worse, and PPO duly bought
# duration with feasibility (halts 5.7 -> 9.7 per 129 routes, 27 of 29 of them
# spread-ceiling breaches).
#
# v2 is MULTIPLICATIVE: a halt costs HALT_MULT times an ESTIMATE of what
# completing would have cost, so the deterrent scales with the route instead
# of being a constant that long routes can absorb.  The estimate converts the
# remaining nominal drive time into a full cost using DWELL_INFLATION, which
# is measured, not guessed: over completed routes,
# duration / nominal-drive-hours has median 2.44 (p10 2.23, p90 2.56).
DWELL_INFLATION = 2.44    # measured: realised duration per nominal drive hour
HALT_MULT       = 2.0     # a halt costs 2x an estimated completion


class RouteEnv:
    """One instance; `reset()` draws a fresh realisation, `step()` executes one
    stop.  Deliberately not a gym.Env — there is no benefit here and the extra
    indirection would only obscure the transition."""

    def __init__(self, full_data, raw_inst, classes, k_look,
                 energy_q=0.5, guard_q=0.95, spread_q=None, cv=0.15, seed=None):
        self.fd, self.classes, self.k = full_data, classes, k_look
        self.ctx = build_instance_context(raw_inst)
        self.Ebar = [float(raw_inst["Ebar"][str(i)]) for i in range(len(raw_inst["Ebar"]))]
        self.Tbar = [float(raw_inst["Tbar"][str(i)]) for i in range(len(raw_inst["Tbar"]))]
        self.Emin, self.Ecap = float(raw_inst["Emin"]), float(raw_inst["Ecap"])
        self.beta = float(raw_inst.get("beta", 0.5))
        self.energy_q, self.guard_q, self.cv = energy_q, guard_q, cv
        # A STRICTER quantile applied to the spread ceiling only.  27 of 29
        # halts in the v1 PPO run were `hos_spread`, so that one constraint —
        # not the guard in general — is what the policy keeps walking into.
        # None = use guard_q for everything (v1 behaviour).
        self.spread_q = spread_q
        self.rng = np.random.default_rng(seed)
        self.N = int(full_data["N"])
        self.C = set(full_data["C"])
        self.Wha = full_data.get("Wha", {}) or {}
        self.Whf = full_data.get("Whf", {}) or {}
        # nominal drive time still to come, for the halt penalty
        D = full_data["D"]
        suf = np.zeros(self.N + 1)
        for i in range(self.N - 1, -1, -1):
            suf[i] = suf[i + 1] + float(D.get(i, 0.0))
        self.suf_D = suf

    # ── episode control ──────────────────────────────────────────────────
    def reset(self, use_recorded=None):
        """Draw a fresh realisation (or use a supplied recorded one)."""
        if use_recorded is not None:
            self.D_real, self.E_real = use_recorded
        else:
            sc = generate_scenarios(self.fd, 0, self.N, n_scenarios=1,
                                    cv=self.cv,
                                    seed=int(self.rng.integers(1 << 31)))[0]
            self.D_real = [float(sc["D"][i]) for i in range(self.N)]
            self.E_real = [float(sc["E"][i]) for i in range(self.N)]
        self.v = BEHDV(self.fd)
        self.t0 = float(self.fd.get("T_START", 8.0))
        self.done = False
        self.halted = False
        return self._obs()

    def _obs(self):
        st = dict(stop=self.v.stop, t_arr=self.v.t_arr, e_arr=self.v.e_arr,
                  cd=self.v.cd, sd=self.v.sd, sw=self.v.sw, phi=self.v.phi,
                  rho2_used=self.v.rho2_used,
                  ext_shift_used=self.v.ext_shift_used,
                  h=self.v.h)           # live spread clock
        return np.asarray(features_at(self.ctx, st, self.k), dtype=np.float32)

    def action_mask(self):
        """Structural mask AND the forcing restriction, identical to the rules
        the evaluated policy obeys — exploration is confined to legal, compliant
        actions so RL cannot 'discover' a regulatory breach as a shortcut."""
        x = torch.from_numpy(self._obs()).unsqueeze(0)
        m = build_mask(x, self.classes)[0].numpy().copy()
        flags = compute_flags(self.fd, self.v.stop, self.v, cv=self.cv,
                              quantile=self.guard_q)
        if self.spread_q is not None and self.spread_q != self.guard_q:
            # compute_flags folds the sd limit and the spread ceiling into one
            # must_rest; evaluating it again at a stricter quantile and OR-ing
            # tightens the spread test without touching the energy guard.  The
            # stricter D_next_wc also feeds the action-level spread check in
            # action_passes.
            f2 = compute_flags(self.fd, self.v.stop, self.v, cv=self.cv,
                               quantile=self.spread_q)
            flags = dict(flags)
            flags["must_rest"] = bool(flags["must_rest"] or f2["must_rest"])
            flags["D_next_wc"] = max(flags["D_next_wc"], f2["D_next_wc"])
        if flags["must_charge"] or flags["must_reset_cd"] or flags["must_rest"]:
            allow = np.zeros_like(m)
            for j, (y, brk, rst) in enumerate(self.classes):
                if not m[j]:
                    continue
                cand = dict(y=int(y),
                            break_type=None if brk == "none" else brk,
                            rest_type=None if rst == "none" else rst)
                allow[j] = action_passes(self.fd, self.v.stop, self.v, cand, flags)
            if allow.any():
                m = allow
        if not m.any():                       # never hand back an empty mask
            m[0] = True
        return m

    def step(self, cls_idx: int, tauc_pred: float):
        """Execute one stop.  Returns (obs, reward, done, info)."""
        i = self.v.stop
        y, brk, rst = self.classes[int(cls_idx)]
        y = int(y)
        tauc = 0.0
        if y == 1:
            need = self.Emin + energy_to_next_cs_at_q(self.fd, i, self.energy_q, self.cv)
            lo = charge_time_to(self.v.e_arr, min(need, self.Ecap), self.Ebar, self.Tbar)
            hi = charge_time_to(self.v.e_arr, self.Ecap, self.Ebar, self.Tbar)
            tauc = float(min(max(tauc_pred, lo), hi))
        Tb = dict(b45=self.fd["Tb45"], b15=self.fd["Tb15"], b30=self.fd["Tb30"])
        taub = max(0.0, Tb[brk] - tauc) if brk in Tb else 0.0
        taur = (self.fd["Tr1"] if rst == "r1" else
                self.fd["Tr2"] if rst == "r2" else 0.0)
        tauq = float(self.fd["Q"].get(i, 0.0)) * y
        sol0 = dict(taub=taub, tauc=tauc, taur=taur, tauq=tauq, y=y,
                    b45=int(brk == "b45"), b15=int(brk == "b15"),
                    b30=int(brk == "b30"),
                    rho1=int(rst == "r1"), rho2=int(rst == "r2"))
        action = dict(y=y, break_type=None if brk == "none" else brk,
                      rest_type=None if rst == "none" else rst)

        t_before = self.v.t_arr
        n_miss_before = len(getattr(self.v, "tw_misses", {}) or {})
        self.v.advance(action=action, D_next=float(self.D_real[i]),
                       E_next=float(self.E_real[i]),
                       milp_sol=dict(feasible=True, sol=[sol0]))
        elapsed = self.v.t_arr - t_before
        n_miss_after = len(getattr(self.v, "tw_misses", {}) or {})
        reward = -elapsed - self.beta * (n_miss_after - n_miss_before)

        if self.v.is_halted:
            self.done = self.halted = True
            # Total episode return becomes -HALT_MULT * est_total, where
            # est_total is what a completed route would plausibly have cost.
            elapsed_total = self.v.t_arr - self.t0
            est_rem   = float(self.suf_D[min(self.v.stop, self.N)]) * DWELL_INFLATION
            est_total = elapsed_total + est_rem
            reward -= est_rem + (HALT_MULT - 1.0) * est_total
        elif self.v.stop >= self.N:
            self.done = True
        info = dict(halted=self.halted, stop=self.v.stop,
                    duration=(self.v.t_arr - self.t0) if self.done else None,
                    misses=n_miss_after)
        return (None if self.done else self._obs()), reward, self.done, info

    # ── convenience: full greedy/argmax rollout, used for evaluation ─────
    def rollout(self, net, mu, sd, deterministic=True, recorded=None):
        obs = self.reset(use_recorded=recorded)
        total = 0.0
        while not self.done:
            m = self.action_mask()
            x = (torch.from_numpy(obs).unsqueeze(0) - mu) / sd
            with torch.no_grad():
                logits, tauc = net(x, torch.from_numpy(m).unsqueeze(0))
            if deterministic:
                a = int(logits.argmax(-1))
            else:
                a = int(torch.distributions.Categorical(logits=logits).sample())
            obs, r, done, info = self.step(a, float(tauc))
            total += r
        return total, info
