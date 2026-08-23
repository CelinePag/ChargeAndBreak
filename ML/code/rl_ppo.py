"""
rl_ppo.py — PPO fine-tuning of the cloned policy
================================================
GOAL
----
Behaviour cloning is bounded by its teacher by construction: it can match the
exact-tail look-ahead, never beat it.  Reinforcement learning optimises the
REALISED objective directly, so it can in principle exceed the teacher — the
teacher is itself only ~2.2% from the hindsight oracle, and it is myopic in
specific ways (24 h horizon, mean-over-scenarios criterion) that a policy
trained on outcomes need not inherit.

WHY PPO, AND WHY FROM THE CLONE
-------------------------------
* From the clone, always.  RL from scratch on a 12-action masked problem with
  hard regulatory infeasibility would spend its entire budget rediscovering
  behaviour we can copy for free.  We initialise actor AND critic trunk from
  the BC checkpoint.
* PPO rather than REINFORCE: the clipped surrogate bounds how far one update
  can move the policy, which matters enormously here because a single bad
  update can push the policy into halting territory and the return signal
  becomes uninformative.
* A KL penalty back to the FROZEN clone on top of PPO's clipping.  This is the
  standard RL-from-a-good-initialisation stabiliser: it lets us take many small
  steps without drifting into a regime where the (already good) prior is lost.
  `--kl-coef 0` disables it.

WHAT IS AND IS NOT TRAINED
--------------------------
Only the DISCRETE head is fine-tuned.  The charge-duration head is frozen at
its cloned values because (a) it is already within 0.9% of the teacher's total
charging time, so there is little to gain, and (b) adding a continuous action
distribution doubles the variance of the gradient for a second-order effect.
This is a deliberate scope choice, not an oversight; `--train-tauc` is left as
future work.

VARIANCE CONTROL
----------------
Routes differ by an order of magnitude in length, so raw returns are wildly
heteroscedastic.  Three defences:
  1. A learned value function (GAE) as the baseline.
  2. Returns are scaled per instance by the instance's own BC return, so the
     advantage is dimensionless and comparable across route classes.
  3. Advantages are normalised within each batch.

Run (HPC):
    python ML/code/rl_ppo.py --model ML/models/policy_K20_seedsplit_seed0_cw0.pt \\
        --iters 400 --episodes-per-iter 64 --seed 0
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, collections
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ML", "code"))

from src import paths as _paths
_paths.redirect_outputs(os.path.join(ROOT, "ML"))

from src.instance_gen.instance_io import load_instance_json
from extract_dataset import tag_for
from model import StudentPolicy
from rl_env import RouteEnv

INST_DIR = os.path.join(ROOT, "instances")
OUT      = os.path.join(ROOT, "ML", "models")


class ActorCritic(nn.Module):
    """The cloned policy plus a value head on the same trunk.

    The trunk is shared with the actor rather than separate: the features that
    predict what the teacher does are the same features that predict how much
    time remains, and a separate critic would have to relearn them from a much
    weaker signal.
    """
    def __init__(self, ck):
        super().__init__()
        self.classes = [tuple(c) for c in ck["classes"]]
        self.net = StudentPolicy(len(ck["mu"]), len(self.classes), hidden=ck["hidden"])
        self.net.load_state_dict(ck["state"])
        self.value = nn.Linear(ck["hidden"], 1)
        nn.init.zeros_(self.value.weight); nn.init.zeros_(self.value.bias)

    def forward(self, x, mask):
        h = self.net.trunk(x)
        logits = self.net.head_cls(h).masked_fill(~mask, float("-inf"))
        tauc = self.net.head_tauc(h).squeeze(-1)
        return logits, tauc, self.value(h).squeeze(-1)


def gae(rewards, values, gamma, lam):
    """Generalised advantage estimation on one episode (no bootstrap: every
    episode terminates, either by finishing the route or by halting)."""
    adv, last = np.zeros(len(rewards), dtype=np.float32), 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


def _teacher_duration(inst, _cache={}):
    """Stored exact-tail teacher duration for an instance (None if it halted
    or is missing).  Read from the bucketed solutions tree."""
    if inst not in _cache:
        fs = [f for f in _paths.in_tree(str(_paths.ROOT / "solutions"),
                                        f"{inst}_LA_MIPTAIL*.json")
              if "nosplit" not in f and "LOCAL" not in f]
        d = json.load(open(sorted(fs, key=os.path.basename)[-1])) if fs else None
        _cache[inst] = d.get("duration_h") if d else None
    return _cache[inst]


def evaluate(ac, mu, sd, names, classes, k_look, get_env):
    """THE metric that matters: deterministic rollouts on the RECORDED
    realisations, paired against the stored teacher on the same instance.
    Training return (on fresh draws) is not comparable to anything published,
    so without this we would be flying blind about the actual goal."""
    deltas, halts = [], 0
    for n in names:
        t = _teacher_duration(n)
        env, rec = get_env(n)
        _R, info = env.rollout(ac.net, mu, sd, deterministic=True, recorded=rec)
        if info["halted"]:
            halts += 1
            continue
        if t:
            deltas.append(100 * (info["duration"] - t) / t)
    return (float(np.median(deltas)) if deltas else float("nan")), halts, len(deltas)


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    classes, k_look = [tuple(c) for c in ck["classes"]], int(ck.get("k_look", 20))
    mu, sd = torch.as_tensor(ck["mu"]), torch.as_tensor(ck["sd"])

    ac = ActorCritic(ck)
    frozen = ActorCritic(ck)                      # KL anchor, never updated
    for p in frozen.parameters():
        p.requires_grad_(False)
    ac.net.head_tauc.requires_grad_(False)        # duration head frozen

    # ── training instances: the TRAIN split only ────────────────────────
    z = np.load(os.path.join(ROOT, "ML", "data",
                             f"dataset_{tag_for(k_look, a.mode)}.npz"), allow_pickle=False)
    names = sorted(set(z["instance"][z["split"] == "train"].tolist()))
    if a.limit:
        names = names[:a.limit]
    val_names = sorted(set(z["instance"][z["split"] == "val"].tolist()))
    if a.eval_n:
        val_names = val_names[:a.eval_n]
    print(f"PPO on {len(names)} training instances, "
          f"{a.episodes_per_iter} episodes/iter x {a.iters} iters")

    envs, base_ret = {}, {}
    def get_env(n):
        if n not in envs:
            raw = json.load(open(os.path.join(INST_DIR, n + ".json")))["instance"]
            fd, D0, E0, cv = load_instance_json(os.path.join(INST_DIR, n + ".json"))
            envs[n] = (RouteEnv(fd, raw, classes, k_look, cv=cv, seed=a.seed),
                       (D0, E0))
        return envs[n]

    opt = torch.optim.Adam([p for p in ac.parameters() if p.requires_grad], lr=a.lr)
    rng = np.random.default_rng(a.seed)
    hist = []

    for it in range(a.iters):
        # ── collect ──────────────────────────────────────────────────────
        OBS, MASK, ACT, LOGP, ADV, RET = [], [], [], [], [], []
        ep_returns, ep_halts = [], 0
        t0 = time.time()
        for _ in range(a.episodes_per_iter):
            n = names[int(rng.integers(len(names)))]
            env, _rec = get_env(n)
            obs = env.reset()
            o_l, m_l, a_l, lp_l, v_l, r_l = [], [], [], [], [], []
            while not env.done:
                m = env.action_mask()
                xt = (torch.from_numpy(obs).unsqueeze(0) - mu) / sd
                mt = torch.from_numpy(m).unsqueeze(0)
                with torch.no_grad():
                    logits, tauc, v = ac(xt, mt)
                    dist = torch.distributions.Categorical(logits=logits)
                    act = dist.sample()
                    lp = dist.log_prob(act)
                o_l.append(obs); m_l.append(m); a_l.append(int(act))
                lp_l.append(float(lp)); v_l.append(float(v))
                obs, r, done, info = env.step(int(act), float(tauc))
                r_l.append(r)
            ep_returns.append(sum(r_l)); ep_halts += bool(info["halted"])
            # scale by this instance's typical magnitude -> dimensionless
            if n not in base_ret:
                base_ret[n] = max(abs(sum(r_l)), 1.0)
            s = base_ret[n]
            adv = gae(np.array(r_l) / s, np.array(v_l) / s, a.gamma, a.lam)
            ret = adv + np.array(v_l) / s
            OBS += o_l; MASK += m_l; ACT += a_l; LOGP += lp_l
            ADV += list(adv); RET += list(ret)

        OBS = torch.as_tensor(np.array(OBS, dtype=np.float32))
        MASK = torch.as_tensor(np.array(MASK))
        ACT = torch.as_tensor(np.array(ACT, dtype=np.int64))
        LOGP = torch.as_tensor(np.array(LOGP, dtype=np.float32))
        ADV = torch.as_tensor(np.array(ADV, dtype=np.float32))
        RET = torch.as_tensor(np.array(RET, dtype=np.float32))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        X = (OBS - mu) / sd

        # ── update ───────────────────────────────────────────────────────
        idx = np.arange(len(ACT))
        for _ in range(a.epochs):
            rng.shuffle(idx)
            for s0 in range(0, len(idx), a.batch):
                b = idx[s0:s0 + a.batch]
                logits, _t, v = ac(X[b], MASK[b])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(ACT[b])
                ratio = torch.exp(lp - LOGP[b])
                s1 = ratio * ADV[b]
                s2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * ADV[b]
                pol_loss = -torch.min(s1, s2).mean()
                v_loss = ((v - RET[b]) ** 2).mean()
                ent = dist.entropy().mean()
                with torch.no_grad():
                    f_logits, _, _ = frozen(X[b], MASK[b])
                kl = torch.distributions.kl_divergence(
                    torch.distributions.Categorical(logits=f_logits), dist).mean()
                loss = pol_loss + a.vf_coef * v_loss - a.ent_coef * ent + a.kl_coef * kl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()

        rec = dict(iter=it, ret=float(np.mean(ep_returns)), halts=ep_halts,
                   kl=float(kl.detach()), ent=float(ent.detach()))
        msg = (f"it {it:4d}  return {np.mean(ep_returns):9.2f}  "
               f"halts {ep_halts:3d}/{a.episodes_per_iter}  "
               f"KL {rec['kl']:.4f}  H {rec['ent']:.3f}")
        if a.eval_every and (it % a.eval_every == 0 or it + 1 == a.iters):
            med, vh, nv = evaluate(ac, mu, sd, val_names, classes, k_look, get_env)
            rec.update(val_median=med, val_halts=vh, val_n=nv)
            msg += (f"  | VAL vs teacher {med:+.2f}% (n={nv}, halts {vh})"
                    f" {'<-- BEATS TEACHER' if med < 0 else ''}")
        hist.append(rec)
        print(msg + f"  ({time.time()-t0:.0f}s)")

        if (it + 1) % a.save_every == 0 or it + 1 == a.iters:
            tag = f"policy_ppo_seed{a.seed}"
            torch.save(dict(state=ac.net.state_dict(), mu=ck["mu"], sd=ck["sd"],
                            classes=ck["classes"], feature_names=ck["feature_names"],
                            hidden=ck["hidden"], seed=a.seed, k_look=k_look,
                            split_mode=a.mode),
                       os.path.join(OUT, tag + ".pt"))
            json.dump(hist, open(os.path.join(OUT, f"history_{tag}.json"), "w"))
            print(f"  saved {tag}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(
        ROOT, "ML", "models", "policy_K20_seedsplit_seed0_cw0.pt"))
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--episodes-per-iter", dest="episodes_per_iter", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="1.0: the objective is undiscounted total time")
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--vf-coef", dest="vf_coef", type=float, default=0.5)
    p.add_argument("--ent-coef", dest="ent_coef", type=float, default=0.005)
    p.add_argument("--kl-coef", dest="kl_coef", type=float, default=0.05,
                   help="penalty against drifting from the cloned policy")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--split-mode", dest="mode", default="seed",
                   choices=["family", "seed"])
    p.add_argument("--save-every", dest="save_every", type=int, default=25)
    p.add_argument("--eval-every", dest="eval_every", type=int, default=10,
                   help="iterations between validation evaluations (0 = off)")
    p.add_argument("--eval-n", dest="eval_n", type=int, default=40,
                   help="validation instances per evaluation (0 = all 129)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
