"""Synthetic checks of torch_models.ChargerNet (no dataset needed)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from features import N_TOK, TOKEN_ATTRS                          # noqa: E402
from torch_models import build                                   # noqa: E402
from torch_train import charger_config                           # noqa: E402

rng = np.random.default_rng(0)
state = ["soc_frac", "cd_slack", "charge_time_to_full"] + \
        [f"tok{t}_{a}" for t in range(N_TOK + 1) for a in TOKEN_ATTRS]
action = ["a_y", "a_b45", "a_tauc_full"]
names = state + action
n = 400
X = rng.normal(size=(n, len(names))).astype(np.float32)
for t in range(N_TOK + 1):                         # plausible raw token values
    p = f"tok{t}_"
    X[:, names.index(p + "exists")] = rng.random(n) > 0.1
    X[:, names.index(p + "reach")] = rng.random(n) > 0.2
    X[:, names.index(p + "kw")] = rng.choice([150, 350, 700, 1000], n)
    X[:, names.index(p + "tfull")] = rng.uniform(0.2, 3.0, n)
X[:, names.index("a_y")] = rng.random(n) > 0.5
tr = np.ones(n, bool)
mean, std = X.mean(0), X.std(0)
std[std < 1e-6] = 1.0
cfg = dict(arch="charger", n_in=len(names), hidden=[32, 32], hidden_aux=[16], dropout=0.0)
cfg.update(charger_config(names, len(state), X, mean, std, tr, "power"))
print("g reads", len(cfg["g_ix"]), "of", len(names), "| excluded:",
      [names[i] for i in range(len(names)) if i not in cfg["g_ix"] and not names[i].startswith("tok")])
net = build(cfg)
Xs = torch.from_numpy((X - mean) / std)
c, f, t = net(Xs)
assert c.shape == f.shape == t.shape == (n,)
c.sum().backward()
print("forward/backward ok; phi grad norm",
      float(sum(p.grad.norm() for p in net.cost.phi.parameters())))

# no reachable charger ahead -> m must be 0 and finite
Xn = X.copy()
for t in range(1, N_TOK + 1):
    Xn[:, names.index(f"tok{t}_reach")] = 0
g, vh, m, y = net.cost.parts(torch.from_numpy((Xn - mean) / std))
assert torch.isfinite(m).all() and (m == 0).all()
print("no reachable charger ahead: m = 0, finite")

# weight sharing: the same token in position 0 and position 1 gets the same value
x1 = X[:1].copy()
for a in TOKEN_ATTRS:
    x1[0, names.index(f"tok1_{a}")] = x1[0, names.index(f"tok0_{a}")]
xs = torch.from_numpy((x1 - mean) / std)
raw = xs[:, net.cost.tok_ix] * net.cost.tok_std + net.cost.tok_mean
tok = (raw - net.cost.pool_mean) / net.cost.pool_std
assert torch.allclose(tok[0, 0], tok[0, 1], atol=1e-5)
print("same charger in two positions -> identical phi input")

# save / reload as torch_policy does
sd = net.state_dict()
net2 = build(cfg)
net2.load_state_dict(sd)
assert torch.allclose(net2(Xs)[0], net(Xs)[0])
print("state dict round trip ok;", sum(p.numel() for p in net.parameters()), "parameters")
