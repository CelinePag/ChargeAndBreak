"""
rl_torch_ft.py — expert-iteration training step for the PyTorch student
=======================================================================
Fine-tunes ONLY the cost network of a SplitNet student on rollout labels
(rl_exit_collect.py, or rl_api.py run with --kind torch): at a labelled
state with candidates j the rollouts measured d_j = cost(j) - cost(student's
pick) in hours, under shared futures, the student driving on.  The loss is

    mean_states  sum_j pi_theta(j | s) d_j                expected measured advantage
               + beta  * KL( pi_theta(.|s) || pi_0(.|s) )          stay near the student
               + alpha * mean_rows (c_theta - c_0)^2                on the candidates
               + gamma * mean_rows (c_theta - c_0)^2                on imitation rows

with pi_theta(j | s) = softmax_j(-c_theta(x_j) / tau) over the candidates and
pi_0 the same for the original network.  The first term is linear in the
labels, so their noise averages out over states.  The last term anchors the
network on a sample of the imitation dataset (data/dataset.npz, fit route
seeds), i.e. on the states where no label says anything, so the student does
not drift there.  The policy deployed is the usual argmin of c_theta; the
feasibility and charge-duration heads are untouched.

Several label sets can be given (--data r1,r2: all rounds so far, the
DAgger aggregation).  Early stopping on the states of route seeds `--hold`
(default 17-19): the measured one-step gain of the argmin of c_theta over
each held-out state's candidates (the student's own pick is 0).

Saves models/<out>_torch.pt + <out>_meta.json, a drop-in TorchPolicy tag.

    python ML/code/rl_torch_ft.py --data exit_r1 --base tmlp_F95_split_list_s0 \
        --out tmlp_F95_split_list_s0_exit1
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
K = 4                                         # candidates per labelled state (rl_api.candidates)


def label_dir(name):
    """rl_exit_<name> (expert iteration) or rl_api_<name> (policy-iteration labels)."""
    for pre in ("rl_exit_", "rl_api_"):
        p = os.path.join(DATA, pre + name)
        if os.path.isdir(p):
            return p
    raise SystemExit(f"no label set {name} in {DATA}")


def load_states(names):
    """Padded arrays over all labelled states with >= 2 candidates:
    rows [S, K, F], d [S, K] (pad 0), mask [S, K], route seed [S]."""
    rows, d, mask, seeds = [], [], [], []
    for name in names:
        for f in sorted(glob.glob(os.path.join(label_dir(name), "*.npz"))):
            z = np.load(f, allow_pickle=True)
            seed = int(str(z["instance"]).rsplit("_", 1)[1])
            for r in z["recs"]:
                n = len(r["keys"])
                if n < 2:
                    continue
                n = min(n, K)
                x = np.zeros((K, r["rows"].shape[1]), dtype=np.float32)
                x[:n] = r["rows"][:n]
                dd = np.zeros(K, dtype=np.float32)
                dd[:n] = r["d_mean"][:n]
                m = np.zeros(K, dtype=bool)
                m[:n] = True
                rows.append(x)
                d.append(dd)
                mask.append(m)
                seeds.append(seed)
    return (np.stack(rows), np.stack(d), np.stack(mask), np.array(seeds))


def anchor_rows(features, n, rng, fit_seeds=range(1, 20)):
    """`n` random imitation rows (data/dataset.npz, fit route seeds), with the
    student's own feature columns in its order."""
    if n <= 0:
        return None
    z = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    names = [str(x) for x in z["feature_names"]]
    cols = [names.index(f) for f in features]
    ok = np.where(np.isin(z["seed"], list(fit_seeds)))[0]
    pick = rng.choice(ok, size=min(n, len(ok)), replace=False)
    return np.asarray(z["X"][np.sort(pick)][:, cols], dtype=np.float32)


def costs(net_cost, x, mean, std):
    """c for rows x [..., F] (raw features), standardized as TorchPolicy does."""
    shape = x.shape[:-1]
    xs = torch.from_numpy(((x.reshape(-1, x.shape[-1]) - mean) / std).astype(np.float32))
    return net_cost(xs).squeeze(-1).reshape(shape)


def one_step_gain(net_cost, rows, d, mask, mean, std):
    """Measured gain (h) per state of the argmin of c over the candidates;
    (mean, se, share of states changed)."""
    with torch.no_grad():
        c = costs(net_cost, rows, mean, std).numpy()
    c = np.where(mask, c, np.inf)
    j = np.argmin(c, axis=1)
    g = d[np.arange(len(d)), j]
    return float(g.mean()), float(g.std(ddof=1) / np.sqrt(len(g))), float((j != 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="label sets, comma-separated")
    ap.add_argument("--base", required=True, help="the student to fine-tune (torch tag)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", default="17,18,19")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--anchor-n", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
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

    rows, d, mask, seeds = load_states(a.data.split(","))
    hold = np.isin(seeds, [int(s) for s in a.hold.split(",")])
    tr, va = np.where(~hold)[0], np.where(hold)[0]
    A = anchor_rows(meta["features"], a.anchor_n, rng)
    print(f"{len(rows)} labelled states ({len(tr)} fit, {len(va)} held out, seeds {a.hold}); "
          f"{0 if A is None else len(A)} anchor rows")
    g0 = one_step_gain(net.cost, rows[va], d[va], mask[va], mean, std)
    best_possible = float(np.where(mask[va], d[va], np.inf).min(axis=1).mean())
    print(f"held out, student as is: {g0[0]:+.4f} h per state (changes {100 * g0[2]:.1f} %); "
          f"labels' hindsight best {best_possible:+.4f} h")

    opt = torch.optim.Adam(net.cost.parameters(), lr=a.lr)
    dt, mt = torch.from_numpy(d), torch.from_numpy(mask)
    best = (g0[0], copy.deepcopy(net.state_dict()), 0)
    for ep in range(1, a.epochs + 1):
        net.cost.train()
        order = rng.permutation(tr)
        tot = 0.0
        for b0 in range(0, len(order), a.batch):
            ib = order[b0:b0 + a.batch]
            x = rows[ib]
            c = costs(net.cost, x, mean, std)
            with torch.no_grad():
                cz = costs(c0, x, mean, std)
            m = mt[ib]
            neg = torch.finfo(c.dtype).min / 4
            logp = torch.log_softmax(torch.where(m, -c / a.tau, torch.full_like(c, neg)), dim=1)
            logp0 = torch.log_softmax(torch.where(m, -cz / a.tau, torch.full_like(cz, neg)), dim=1)
            p = logp.exp() * m
            loss = (p * dt[ib]).sum(1).mean() \
                + a.beta * (p * (logp - logp0)).sum(1).mean() \
                + a.alpha * (((c - cz) ** 2) * m).sum() / m.sum()
            if A is not None:
                ia = rng.integers(0, len(A), size=4 * len(ib))
                ca = costs(net.cost, A[ia], mean, std)
                with torch.no_grad():
                    ca0 = costs(c0, A[ia], mean, std)
                loss = loss + a.gamma * ((ca - ca0) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(ib)
        net.cost.eval()
        gt = one_step_gain(net.cost, rows[tr], d[tr], mask[tr], mean, std)
        gv = one_step_gain(net.cost, rows[va], d[va], mask[va], mean, std)
        flag = ""
        if gv[0] < best[0]:
            best = (gv[0], copy.deepcopy(net.state_dict()), ep)
            flag = "  *"
        print(f"epoch {ep:2d}  loss {tot / len(tr):+.4f}  one-step gain fit {gt[0]:+.4f}  "
              f"held out {gv[0]:+.4f} ± {gv[1]:.4f} h (changes {100 * gv[2]:.1f} %){flag}",
              flush=True)
    if best[2] == 0:
        print("no epoch beat the student on the held-out states: saving it unchanged")
    torch.save(dict(config=ck["config"], state_dict=best[1], mean=ck["mean"], std=ck["std"]),
               os.path.join(MODELS, f"{a.out}_torch.pt"))
    meta = dict(meta, tag=a.out, rl=dict(base=a.base, data=a.data, epoch=best[2],
                                         held_out_gain=best[0], tau=a.tau, beta=a.beta,
                                         alpha=a.alpha, gamma=a.gamma, anchor_n=a.anchor_n,
                                         lr=a.lr))
    with open(os.path.join(MODELS, f"{a.out}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"saved {a.out} (epoch {best[2]}, held-out one-step gain {best[0]:+.4f} h)")


if __name__ == "__main__":
    main()
