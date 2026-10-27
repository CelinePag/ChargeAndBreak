"""
rl_torch_ft.py — step 3a of ML/RL_THEORY.md: policy iteration for the PyTorch student
=====================================================================================
Fine-tunes ONLY the cost network of a SplitNet student on its own rollout
labels (rl_api.py run with --kind torch): at a labelled state with candidates
j (the student's pick first) the rollouts measured d_j = cost(j) - cost(pick)
in hours, under shared futures, with the student driving on.  The loss is

    sum_states  sum_j pi_theta(j | s) d_j            expected measured advantage
              + beta * KL( pi_theta(.|s) || pi_0(.|s) )       stay near the student
              + alpha * mean_rows (c_theta - c_0)^2            leave other states alone

with pi_theta(j | s) = softmax_j(-c_theta(x_j) / tau) over the candidates and
pi_0 the same for the original network.  The first term is linear in the
labels, so their noise averages out over states instead of biasing the fit.
The policy deployed is the student's usual argmin of c_theta; the feasibility
and charge-duration heads are untouched.  This is policy iteration with a
soft-greedy, KL-regularised improvement step (AggreVaTeD-style, with the
learner's own roll-outs rather than an expert's).

Early stopping on route seeds `--hold` (default 17-19): the measured one-step
gain of the argmin over each held-out state's candidates (as in rl_api_fit).
Saves models/<out>_torch.pt + <out>_meta.json (a drop-in TorchPolicy tag).

    python ML/code/rl_torch_ft.py --data r1_torch --base tmlp_F95_split_list_s0 \
        --out tmlp_F95_split_list_s0_rl1
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:                    # `src` on machines without the editable install
    sys.path.insert(0, _ROOT)

import torch                                                         # noqa: E402

from torch_models import build                                       # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def load_states(name):
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, f"rl_api_{name}", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        seed = int(str(z["instance"]).rsplit("_", 1)[1])
        for r in z["recs"]:
            if len(r["keys"]) >= 2:
                out.append((seed, r["rows"].astype(np.float32),
                            r["d_mean"].astype(np.float32), r["d_se"].astype(np.float32)))
    return out


def one_step_gain(net_cost, states, mean, std):
    """Measured gain (h) of the argmin of c over each state's candidates."""
    g = []
    with torch.no_grad():
        for _s, rows, d, _se in states:
            c = net_cost(torch.from_numpy((rows - mean) / std)).squeeze(-1).numpy()
            g.append(float(d[int(np.argmin(c))]))
    g = np.array(g)
    return g.mean(), g.std(ddof=1) / np.sqrt(len(g)), float((g != 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", default="17,18,19")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    with open(os.path.join(MODELS, f"{a.base}_meta.json")) as fh:
        meta = json.load(fh)
    ck = torch.load(os.path.join(MODELS, f"{a.base}_torch.pt"), map_location="cpu",
                    weights_only=True)
    net = build(ck["config"])
    net.load_state_dict(ck["state_dict"])
    if not hasattr(net, "cost"):
        raise SystemExit("needs a SplitNet student (a separate cost network)")
    mean, std = ck["mean"].numpy(), ck["std"].numpy()
    c0 = copy.deepcopy(net.cost).eval()
    for p in c0.parameters():
        p.requires_grad_(False)

    states = load_states(a.data)
    hold = {int(s) for s in a.hold.split(",")}
    tr = [s for s in states if s[0] not in hold]
    va = [s for s in states if s[0] in hold]
    print(f"{len(states)} labelled states: {len(tr)} fit, {len(va)} held out")
    g0 = one_step_gain(net.cost, va, mean, std)
    print(f"held-out one-step gain before: {g0[0]:+.4f} ± {g0[1]:.4f} h "
          f"(the student's own pick is the reference, so this is 0 by construction "
          f"unless ties)")

    opt = torch.optim.Adam(net.cost.parameters(), lr=a.lr)
    best = (np.inf, None, -1)
    for ep in range(a.epochs):
        net.cost.train()
        order = rng.permutation(len(tr))
        tot = 0.0
        for b0 in range(0, len(order), a.batch):
            loss = 0.0
            for i in order[b0:b0 + a.batch]:
                _s, rows, d, _se = tr[i]
                x = torch.from_numpy((rows - mean) / std)
                c = net.cost(x).squeeze(-1)
                with torch.no_grad():
                    cz = c0(x).squeeze(-1)
                logp = torch.log_softmax(-c / a.tau, dim=0)
                logp0 = torch.log_softmax(-cz / a.tau, dim=0)
                p = logp.exp()
                loss = loss + (p * torch.from_numpy(d)).sum() \
                    + a.beta * (p * (logp - logp0)).sum() \
                    + a.alpha * ((c - cz) ** 2).mean()
            loss = loss / max(1, len(order[b0:b0 + a.batch]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        net.cost.eval()
        gt = one_step_gain(net.cost, tr, mean, std)
        gv = one_step_gain(net.cost, va, mean, std)
        print(f"epoch {ep + 1:2d}  loss {tot:9.3f}  one-step gain fit {gt[0]:+.4f}  "
              f"held out {gv[0]:+.4f} ± {gv[1]:.4f} h  (changes {100 * gv[2]:.1f} %)")
        if gv[0] < best[0]:
            best = (gv[0], copy.deepcopy(net.state_dict()), ep + 1)
    if best[1] is None:
        raise SystemExit("no epoch improved on the held-out states")
    torch.save(dict(config=ck["config"], state_dict=best[1], mean=ck["mean"], std=ck["std"]),
               os.path.join(MODELS, f"{a.out}_torch.pt"))
    meta = dict(meta, tag=a.out, rl=dict(base=a.base, data=a.data, epoch=best[2],
                                         held_out_gain=best[0], tau=a.tau, beta=a.beta,
                                         alpha=a.alpha, lr=a.lr))
    with open(os.path.join(MODELS, f"{a.out}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"saved {a.out} (epoch {best[2]}, held-out gain {best[0]:+.4f} h)")


if __name__ == "__main__":
    main()
