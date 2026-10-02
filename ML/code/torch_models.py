"""
torch_models.py — the PyTorch networks of the neural arm
========================================================
One module for the network definitions, imported by the trainer and by the
serving policy, so the two can never build different architectures from the
same checkpoint.

RowNet (stage 1, "parity")
    The same problem as the trees and the scikit-learn MLPs: one (state,
    action) row in, the action's predicted cost out.  One shared trunk with
    three heads instead of three separate networks:

        cost   regret in hours            Huber, clean rows, margin-weighted
        feas   logit P(action feasible)   binary cross-entropy, all rows
        tauc   charge duration in hours   Huber, clean y=1 rows

    What it adds over the scikit-learn MLPs (nn_train.py), all of which that
    library could not do: a Huber loss on the raw regret (no log1p detour),
    the trees' margin sample weights, and one trunk shared by the three
    heads.  The policy is the same argmin over legal actions.

SplitNet (stage 1b)
    The same three heads, each on its OWN trunk -- as the trees have one
    booster per head -- so the trainer can keep each at its own best epoch.
    With one shared trunk the epoch that ranks actions best (epoch 3-5 once
    selection is by regret) left the feasibility head undertrained: it called
    16-22% of the stop split's infeasible rows feasible (trees 2.6%) and the
    policy broke HOS limits on 2-5 of 66 stop routes.  The auxiliary heads
    get a smaller trunk (--hidden-aux).

ChargerNet (direction B, 2026-10-02)
    Trained on routes with one charger type, every student loses 5-9 pp when
    charger power varies along the route: with uniform chargers the inputs
    describing "this charger" and "the chargers ahead" are always equal, so no
    model can learn how to weigh one against the other.  ChargerNet builds the
    comparison into the cost head instead of hoping it is learned:

        v_j   = phi(token_j, context)        ONE network for every charger
        m     = softmin over reachable chargers ahead of v_j   (temperature tau)
        cost  = g(row) + v_here   if the action charges here (y = 1)
                g(row) + m        otherwise

    A token is the same seven numbers for the charger here and each of the
    next six (features.py section J): exists, drive to it, charge on arrival,
    queue, reachable, kW, hours to charge to full there.  Hours-to-full varies
    within a uniform route, because the arrival charge does, so phi can learn
    its effect from uniform data and apply it unchanged where chargers differ.
    Token attributes are put back in physical units and scaled with ONE
    statistic per attribute pooled over all tokens, so phi sees the same
    number for the same charger wherever it sits.  `g` can be denied the
    inputs that already describe this charger's speed (--g-exclude power), so
    that speed reaches the cost only through the comparison.  Feasibility and
    charge duration keep their own networks (as SplitNet), on every input.

Inputs are standardised with the mean / std of the TRAINING rows, which the
checkpoint carries (as tensors, so torch.load(weights_only=True) reads it).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class RowNet(nn.Module):
    def __init__(self, n_in: int, hidden=(256, 256, 128), dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.trunk = nn.Sequential(*layers)
        self.cost = nn.Linear(d, 1)
        self.feas = nn.Linear(d, 1)
        self.tauc = nn.Linear(d, 1)

    def forward(self, x):
        z = self.trunk(x)
        return (self.cost(z).squeeze(-1), self.feas(z).squeeze(-1),
                self.tauc(z).squeeze(-1))


def _mlp(n_in, hidden, dropout):
    """Linear+SiLU stack ending in one output."""
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class SplitNet(nn.Module):
    HEADS = ("cost", "feas", "tauc")

    def __init__(self, n_in: int, hidden=(256, 256, 128), hidden_aux=(128, 128),
                 dropout: float = 0.0):
        super().__init__()
        self.cost = _mlp(n_in, hidden, dropout)
        self.feas = _mlp(n_in, hidden_aux, dropout)
        self.tauc = _mlp(n_in, hidden_aux, dropout)

    def forward(self, x):
        return (self.cost(x).squeeze(-1), self.feas(x).squeeze(-1),
                self.tauc(x).squeeze(-1))


class ChargerCost(nn.Module):
    """The cost head of ChargerNet (see the module docstring)."""

    def __init__(self, cfg: dict):
        super().__init__()
        hidden, dropout = tuple(cfg["hidden"]), cfg.get("dropout", 0.0)
        t = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt)       # noqa: E731
        # indices into the (standardised) row, and the numbers that undo the
        # standardisation of the token and y columns -- rebuilt from the
        # config, so they are not part of the state dict
        self.register_buffer("g_ix", t(cfg["g_ix"], torch.long), persistent=False)
        self.register_buffer("ctx_ix", t(cfg["ctx_ix"], torch.long), persistent=False)
        self.register_buffer("tok_ix", t(cfg["tok_ix"], torch.long), persistent=False)
        self.register_buffer("tok_mean", t(cfg["tok_mean"]), persistent=False)
        self.register_buffer("tok_std", t(cfg["tok_std"]), persistent=False)
        self.register_buffer("pool_mean", t(cfg["pool_mean"]), persistent=False)
        self.register_buffer("pool_std", t(cfg["pool_std"]), persistent=False)
        self.y_ix, self.y_mean, self.y_std = cfg["y_ix"], cfg["y_mean"], cfg["y_std"]
        self.a_exists, self.a_reach = cfg["attr_exists"], cfg["attr_reach"]
        n_attr, n_ctx = len(cfg["pool_mean"]), cfg.get("ctx_dim", 32)
        self.g = _mlp(len(cfg["g_ix"]), hidden, dropout)
        self.ctx = nn.Sequential(nn.Linear(len(cfg["ctx_ix"]), 64), nn.SiLU(),
                                 nn.Linear(64, n_ctx), nn.SiLU())
        self.phi = _mlp(n_attr + n_ctx, tuple(cfg.get("phi_hidden", (64, 64))), dropout)
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.get("tau0", 0.1))))

    def parts(self, x):
        """(g, v_here, m, charges_here): the pieces, for inspection."""
        raw = x[:, self.tok_ix] * self.tok_std + self.tok_mean          # (B, T, A)
        tok = (raw - self.pool_mean) / self.pool_std
        c = self.ctx(x[:, self.ctx_ix])                                 # (B, C)
        v = self.phi(torch.cat([tok, c.unsqueeze(1).expand(-1, tok.shape[1], -1)],
                               dim=-1)).squeeze(-1)                     # (B, T)
        ok = (raw[:, 1:, self.a_exists] > 0.5) & (raw[:, 1:, self.a_reach] > 0.5)
        tau = self.log_tau.exp()
        lse = torch.logsumexp((-v[:, 1:] / tau).masked_fill(~ok, -1e9), dim=1)
        m = torch.where(ok.any(1), -tau * lse, torch.zeros_like(lse))
        y = (x[:, self.y_ix] * self.y_std + self.y_mean) > 0.5
        return self.g(x[:, self.g_ix]).squeeze(-1), v[:, 0], m, y

    def forward(self, x):
        g, v_here, m, y = self.parts(x)
        return g + torch.where(y, v_here, m)


class ChargerNet(nn.Module):
    HEADS = ("cost", "feas", "tauc")

    def __init__(self, cfg: dict):
        super().__init__()
        aux, dropout = tuple(cfg["hidden_aux"]), cfg.get("dropout", 0.0)
        self.cost = ChargerCost(cfg)
        self.feas = _mlp(cfg["n_in"], aux, dropout)
        self.tauc = _mlp(cfg["n_in"], aux, dropout)

    def forward(self, x):
        return self.cost(x), self.feas(x).squeeze(-1), self.tauc(x).squeeze(-1)


def build(config: dict) -> nn.Module:
    """The network a checkpoint's config describes."""
    if config["arch"] == "row":
        return RowNet(config["n_in"], tuple(config["hidden"]), config.get("dropout", 0.0))
    if config["arch"] == "split":
        return SplitNet(config["n_in"], tuple(config["hidden"]),
                        tuple(config["hidden_aux"]), config.get("dropout", 0.0))
    if config["arch"] == "charger":
        return ChargerNet(config)
    raise ValueError(f"unknown architecture {config['arch']!r}")
