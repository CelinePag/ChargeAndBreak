"""
torch_train.py — the neural arm in PyTorch, stage 1: per-row parity
===================================================================
Trains torch_models.RowNet on the SAME rows, targets, splits and feature sets
as gbt_train.py (the trees) and nn_train.py (the scikit-learn MLPs), so the
comparison measures the model and nothing else.  Stage 1 exists to validate
the PyTorch pipeline -- data, scaling, losses, early stopping, serving -- on
the formulation every other arm uses, before the decision-level network
(stage 2) changes the formulation itself.

    loss = Huber(cost; delta) · margin weights      clean rows
         + lambda_feas · BCE(feas)                  all rows
         + lambda_tauc · Huber(tauc)                clean y=1 rows
         + lambda_list · listwise CE (see below)    decisions with >= 2 clean rows

Early stopping is GROUPED: the stop split is route seeds 20-21, never rows, for
the reason nn_train.py gives (rows of a route are a Markov chain).  The best
epoch's weights are kept.  CPU only (no GPU on this machine); --threads keeps
it from starving a concurrent LA run.

Two additions after the first run (tmlp_F95_base_s0, 2026-10-01), whose offline
regret was 1.8x the trees' although its cost MAE on rest rows was LOWER -- the
errors were in the ORDER of the actions within a decision, not their level:

  --select regret   the best epoch is the one with the lowest argmin regret on
                    the stop split (the decision metric), not the lowest loss.
  --lambda-list     a listwise term per decision: cross-entropy between
                    softmax(-regret/tau) and softmax(-pred/tau) over the clean
                    actions.  It is minimised by pred = regret + a per-decision
                    constant, so it agrees with the regression and only adds
                    pressure where two actions are within a few tau.

Both need decision batches (--batch-by decisions: --batch counts decisions);
--arch row --batch-by rows --select loss --lambda-list 0 is the first run's
recipe.

Selecting the shared trunk by regret (tmlp_F95_{sel,list,list01,list3}_s0,
--arch row) picked epoch 3-5, where the feasibility head was still poor and
the policy broke HOS limits on 2-5 of 66 stop routes.  Hence --arch split (the
default since): each head has its own trunk, optimiser and best epoch -- cost
by clean-only stop regret, feasibility by stop log-loss, charge duration by
stop Huber -- exactly as the trees early-stop each booster on its own metric.

    python ML/code/torch_train.py --tag tmlp_F95_base_s0 --seed 0 \
        --arch row --batch-by rows --batch 1024 --select loss
    python ML/code/torch_train.py --tag tmlp_F95_list_s0 --arch row --lambda-list 1
    python ML/code/torch_train.py --tag tmlp_F95_split_list_s0 --lambda-list 1
    python ML/code/torch_train.py --physics base,kwh300,... --fset P ...
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the repo root, for `src` (features.py needs it for the charger arch); every
# other script does the same -- without it the import only worked where the
# project happened to be pip-installed
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch                                                       # noqa: E402
import torch.nn.functional as F                                    # noqa: E402

from dataset import (MODELS, argmin_policy_regret, decision_id,    # noqa: E402
                     decision_margins, load, load_multi, physics_file,
                     print_offline, sample_weights, split_report)
from torch_models import SplitNet, build                           # noqa: E402

ARM = "mlp"          # feature-set resolution: scores (state, action) rows


def _huber(pred, y, w, delta):
    """Weighted mean Huber loss; rows with w = 0 do not count."""
    l = F.huber_loss(pred, y, reduction="none", delta=delta)
    return (l * w).sum() / w.sum().clamp_min(1e-9)


# Inputs that already carry THIS charger's speed (via its charging curve): with
# --g-exclude power the charger arch keeps them out of g, so speed reaches the
# cost only through the comparison of charger tokens.
SPEED_INPUTS = ("charge_time_to_full", "charge_rate_now_kw", "a_tauc_full",
                "a_break_absorbable", "a_residual_break")


def charger_config(names, n_state, X, mean, std, tr, g_exclude):
    """Index lists and statistics ChargerNet needs, from the training rows.

    Token attributes are scaled with ONE mean / std per attribute pooled over
    every token that exists, so phi sees a charger the same way whichever
    position it holds.
    """
    from features import N_TOK, TOKEN_ATTRS
    tok_names = [[f"tok{t}_{a}" for a in TOKEN_ATTRS] for t in range(N_TOK + 1)]
    missing = [n for row in tok_names for n in row if n not in names]
    if missing:
        raise SystemExit(f"charger arch needs --fset T (missing e.g. {missing[:3]})")
    tok_ix = [[names.index(n) for n in row] for row in tok_names]
    is_tok = set(i for row in tok_ix for i in row)
    drop = set(SPEED_INPUTS) if g_exclude == "power" else set()
    g_ix = [i for i, n in enumerate(names) if i not in is_tok and n not in drop]
    ctx_ix = [i for i in g_ix if i < n_state]
    a_ex = TOKEN_ATTRS.index("exists")
    raw = X[np.ix_(np.flatnonzero(tr), np.array(tok_ix).ravel())].reshape(
        -1, len(tok_ix), len(TOKEN_ATTRS))          # no copy of the other columns
    exist = raw[:, :, a_ex] > 0.5
    pool = raw[exist]                                  # (n tokens, A)
    pm, ps = pool.mean(0), pool.std(0)
    ps[ps < 1e-6] = 1.0
    ti = np.array(tok_ix)
    y_ix = names.index("a_y")
    return dict(g_ix=g_ix, ctx_ix=ctx_ix, tok_ix=tok_ix,
                tok_mean=mean[ti].tolist(), tok_std=std[ti].tolist(),
                pool_mean=pm.tolist(), pool_std=ps.tolist(),
                y_ix=y_ix, y_mean=float(mean[y_ix]), y_std=float(std[y_ix]),
                attr_exists=a_ex, attr_reach=TOKEN_ATTRS.index("reach"),
                g_exclude=g_exclude, token_attrs=list(TOKEN_ATTRS))


def decision_table(d):
    """(n_decisions, n_actions) row index per (decision, action); -1 = not legal.

    The action vocabulary is fixed, so a decision is one row of this table and
    a batch of decisions is a dense block -- no ragged segment ops needed.
    """
    _, inv = np.unique(decision_id(d), return_inverse=True)
    tab = np.full((inv.max() + 1, len(d["action_vocab"])), -1, np.int64)
    tab[inv, d["action_ix"]] = np.arange(len(inv))
    assert (tab >= 0).sum() == len(inv), "two rows share a (decision, action)"
    return tab


def _listwise(pred, regret, ok, w, tau):
    """Weighted mean CE between softmax(-regret/tau) and softmax(-pred/tau).

    pred, regret, ok, w: (B, A) blocks; `ok` marks clean actions with a finite
    regret.  Decisions with fewer than two such actions carry no ranking signal
    and are dropped; a decision's weight is the mean row weight over `ok`.
    """
    use = ok.sum(1) >= 2
    if not bool(use.any()):
        return pred.sum() * 0.0
    p, r, m = pred[use], regret[use], ok[use]
    tgt = torch.softmax((-r / tau).masked_fill(~m, -torch.inf), dim=1)
    logp = torch.log_softmax((-p / tau).masked_fill(~m, -torch.inf), dim=1)
    ce = -(tgt * logp.masked_fill(~m, 0.0)).sum(1)
    wd = (w[use] * m).sum(1) / m.sum(1)
    return (ce * wd).sum() / wd.sum().clamp_min(1e-9)


def main():
    from fsets import FSET_IDS, resolve
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fset", default="F", choices=list(FSET_IDS))
    ap.add_argument("--scope", default="all", choices=["all", "SM"])
    ap.add_argument("--physics", default=None,
                    help="comma list of physics values to join (see gbt_train)")
    ap.add_argument("--hold-out", default="")
    ap.add_argument("--arch", default="split", choices=["split", "row", "charger"],
                    help="split: one trunk per head, each at its own best epoch "
                         "(torch_models.SplitNet); row: one shared trunk; charger: "
                         "the charger-comparison cost head (ChargerNet, needs --fset T)")
    ap.add_argument("--g-exclude", default="power", choices=["power", "none"],
                    help="charger arch: keep the inputs that already describe THIS "
                         "charger's speed out of g, so speed reaches the cost only "
                         "through the comparison (power) -- or not (none)")
    ap.add_argument("--hidden", default="256,256,128")
    ap.add_argument("--hidden-aux", default="128,128",
                    help="split arch: the feasibility and charge-duration trunks")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--batch-by", default="decisions", choices=["decisions", "rows"])
    ap.add_argument("--batch", type=int, default=192,
                    help="decisions (or rows, with --batch-by rows) per step")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--select", default="regret", choices=["regret", "loss"],
                    help="best epoch by stop-split argmin regret, or by loss")
    ap.add_argument("--delta", type=float, default=1.0,
                    help="Huber delta for the cost head, in hours")
    ap.add_argument("--lambda-feas", type=float, default=1.0)
    ap.add_argument("--lambda-tauc", type=float, default=1.0)
    ap.add_argument("--lambda-list", type=float, default=0.0,
                    help="weight of the listwise ranking term (decision batches)")
    ap.add_argument("--tau", type=float, default=0.25,
                    help="listwise temperature, in hours")
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="margin weighting exponent; 0 disables it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dagger", default="",
                    help="comma list of DAgger labels (dagger_label.py) whose "
                         "rows join the data, e.g. 'r1,r2'")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(MODELS, exist_ok=True)

    hold_out = tuple(t for t in args.hold_out.split(",") if t)
    if args.physics:
        tags = [t for t in args.physics.split(",") if t]
        with np.load(physics_file(tags[0]), allow_pickle=True) as z:
            sup = [str(x) for x in z["feature_names"]]
            sup_ns = int(z["n_state"])
        want, _ = resolve(args.fset, sup, sup_ns, ARM)
        d = load_multi(tags, columns=set(want) | {"a_y"})
    else:
        d = load()
    if args.dagger:
        from dataset import add_dagger
        d = add_dagger(d, [t for t in args.dagger.split(",") if t])
    _all = [str(x) for x in d["feature_names"]]
    is_y1 = d["X"][:, _all.index("a_y")] > 0.5
    names, n_state = resolve(args.fset, _all, int(d["n_state"]), ARM)
    X = d["X"][:, [_all.index(n) for n in names]].astype(np.float32)
    print(f"[fset] {args.fset}: {len(names)} features ({n_state} state)")
    tr, va, te = split_report(d, args.scope, hold_out)

    # -- standardisation from the TRAINING rows only --------------------------
    mean = X[tr].mean(0)
    std = X[tr].std(0)
    std[std < 1e-6] = 1.0                       # constant columns pass through
    Xs = (X - mean) / std

    clean = d["clean"].astype(bool)
    regret = d["regret"].astype(np.float32)
    m_cost = clean & np.isfinite(regret)
    w = sample_weights(d, decision_margins(d), args.weight_power).astype(np.float32)
    w_cost = np.where(m_cost, w, 0.0).astype(np.float32)
    m_tauc = (clean & is_y1).astype(np.float32)
    y_cost = np.where(m_cost, regret, 0.0).astype(np.float32)
    y_feas = clean.astype(np.float32)
    y_tauc = np.where(clean & is_y1, d["tauc"], 0.0).astype(np.float32)

    T = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in dict(
        X=Xs, w=w_cost, yc=y_cost, yf=y_feas, yt=y_tauc, mt=m_tauc).items()}
    T["ok"] = torch.from_numpy(m_cost)
    i_tr, i_va = np.flatnonzero(tr), np.flatnonzero(va)
    if args.lambda_list > 0 and args.batch_by != "decisions":
        ap.error("--lambda-list needs --batch-by decisions")
    tab = decision_table(d)
    first = tab.max(1)                          # any row of each decision
    dec_tr = np.flatnonzero(tr[first])

    config = dict(arch=args.arch, n_in=len(names),
                  hidden=[int(h) for h in args.hidden.split(",")],
                  dropout=args.dropout)
    if args.arch in ("split", "charger"):
        config["hidden_aux"] = [int(h) for h in args.hidden_aux.split(",")]
    if args.arch == "charger":
        config.update(charger_config(names, n_state, X, mean, std, tr, args.g_exclude))
    net = build(config)
    # One optimiser, scheduler and best checkpoint per independently selected
    # part: the whole net for "row", each head's own trunk otherwise.
    parts = ({h: getattr(net, h) for h in SplitNet.HEADS}
             if args.arch in ("split", "charger") else {"all": net})
    opts = {k: torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
            for k, m in parts.items()}
    scheds = {k: torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.5, patience=3)
              for k, o in opts.items()}
    n_par = sum(p.numel() for p in net.parameters())
    aux = f" aux {config['hidden_aux']}" if "hidden_aux" in config else ""
    if args.arch == "charger":
        aux += f", g reads {len(config['g_ix'])} of {len(names)} inputs"
    print(f"[net] {args.arch} {config['hidden']}{aux}"
          f" -> 3 heads, {n_par:,} parameters, "
          f"{len(i_tr):,} training rows / {len(dec_tr):,} decisions, "
          f"{args.threads} threads")
    print(f"[loss] batch {args.batch} {args.batch_by}, select by {args.select}, "
          f"lambda_list {args.lambda_list:g} (tau {args.tau:g} h)")

    def losses(idx):
        x = T["X"][idx]
        c, f, t = net(x)
        lc = _huber(c, T["yc"][idx], T["w"][idx], args.delta)
        lf = F.binary_cross_entropy_with_logits(f, T["yf"][idx])
        lt = _huber(t, T["yt"][idx], T["mt"][idx], 0.25)
        return lc, lf, lt

    def decision_losses(dec):
        """The row losses over a batch of decisions, plus the listwise term."""
        rows = torch.from_numpy(tab[dec])
        present = rows >= 0
        idx = rows[present]
        c, f, t = net(T["X"][idx])
        lc = _huber(c, T["yc"][idx], T["w"][idx], args.delta)
        lf = F.binary_cross_entropy_with_logits(f, T["yf"][idx])
        lt = _huber(t, T["yt"][idx], T["mt"][idx], 0.25)
        if args.lambda_list <= 0:
            return lc, lf, lt, c.sum() * 0.0

        def block(v, fill=0.0):
            out = torch.full(rows.shape, fill, dtype=v.dtype)
            out[present] = v
            return out
        ll = _listwise(block(c), block(T["yc"][idx]),
                       block(T["ok"][idx], False), block(T["w"][idx]), args.tau)
        return lc, lf, lt, ll

    def stop_regret(use_feas=True):
        """Argmin regret (hours) on the stop split -- the selection metric.

        use_feas=False ranks the clean actions by cost alone: the split arch
        selects its cost head on that, so the feasibility head's epoch cannot
        move the cost head's.
        """
        net.eval()
        pc = np.zeros(len(Xs))
        pf = np.ones(len(Xs), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(i_va), 65536):
                b = torch.from_numpy(i_va[s:s + 65536])
                c, f, _t = net(T["X"][b])
                pc[b.numpy()] = c.numpy()
                pf[b.numpy()] = torch.sigmoid(f).numpy()
        net.train()
        return argmin_policy_regret(d, va, pc, pf if use_feas else None)[0]

    def evaluate(idx):
        net.eval()
        tot = np.zeros(3)
        n = 0
        with torch.no_grad():
            for s in range(0, len(idx), 8192):
                b = torch.from_numpy(idx[s:s + 8192])
                lc, lf, lt = losses(b)
                tot += np.array([lc.item(), lf.item(), lt.item()]) * len(b)
                n += len(b)
        net.train()
        return tot / max(n, 1)

    def step(loss):
        for o in opts.values():
            o.zero_grad()
        loss.backward()
        for o in opts.values():
            o.step()

    best = {k: (np.inf, -1, None) for k in parts}     # (metric, epoch, state)
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        tl = 0.0
        if args.batch_by == "rows":
            perm = rng.permutation(i_tr)
            for s in range(0, len(perm), args.batch):
                b = torch.from_numpy(perm[s:s + args.batch])
                lc, lf, lt = losses(b)
                step(lc + args.lambda_feas * lf + args.lambda_tauc * lt)
        else:
            perm = rng.permutation(dec_tr)
            for s in range(0, len(perm), args.batch):
                lc, lf, lt, ll = decision_losses(perm[s:s + args.batch])
                step(lc + args.lambda_feas * lf + args.lambda_tauc * lt
                     + args.lambda_list * ll)
                tl += float(ll.detach())
        vc, vf, vt = evaluate(i_va)
        if args.arch in ("split", "charger"):
            vr = stop_regret(use_feas=False)
            metric = dict(cost=vr if args.select == "regret" else vc, feas=vf, tauc=vt)
        else:
            vr = stop_regret()
            metric = dict(all=vr if args.select == "regret"
                          else vc + args.lambda_feas * vf + args.lambda_tauc * vt)
        hist.append([float(vc), float(vf), float(vt), float(vr)])
        flag = ""
        for k, m in parts.items():
            scheds[k].step(metric[k])
            if metric[k] < best[k][0] - 1e-6:
                best[k] = (metric[k], ep, copy.deepcopy(m.state_dict()))
                flag += " *" if k == "all" else f" {k}*"
        lst = f"  list {tl / max(1, -(-len(dec_tr) // args.batch)):.4f}" if args.lambda_list > 0 else ""
        lr = next(iter(opts.values())).param_groups[0]["lr"]
        print(f"  ep {ep + 1:3d}  val cost {vc:.4f}  feas {vf:.4f}  tauc {vt:.4f}"
              f"  regret {vr * 60:.2f} min{lst}"
              f"  lr {lr:.1e}  {time.time() - t0:.0f}s{flag}",
              flush=True)
        # a part is done after `patience` epochs without a new best, or once its
        # learning rate has decayed 100-fold (split runs otherwise crawl on: the
        # feasibility trunk kept finding 1e-3 log-loss gains at lr 2e-6)
        if all(ep - best[k][1] >= args.patience
               or opts[k].param_groups[0]["lr"] < args.lr * 1e-2 for k in parts):
            break
    for k, m in parts.items():
        m.load_state_dict(best[k][2])
    net.eval()
    best_ep = {k: b[1] + 1 for k, b in best.items()}
    best_val = {k: float(b[0]) for k, b in best.items()}
    print(f"[train] best epoch {best_ep} of {len(hist)}, val {best_val} "
          f"(cost by {args.select}), {time.time() - t0:.0f}s")

    # -- offline metrics, identical code to the other arms --------------------
    preds, feas = [], []
    with torch.no_grad():
        for s in range(0, len(Xs), 65536):
            c, f, _t = net(T["X"][s:s + 65536])
            preds.append(c.numpy())
            feas.append(torch.sigmoid(f).numpy())
    print_offline(d, np.concatenate(preds).astype(np.float64),
                  np.concatenate(feas), (tr, va, te))

    torch.save(dict(config=config, state_dict=net.state_dict(),
                    mean=torch.from_numpy(mean), std=torch.from_numpy(std)),
               os.path.join(MODELS, f"{args.tag}_torch.pt"))
    with open(os.path.join(MODELS, f"{args.tag}_meta.json"), "w") as fh:
        json.dump(dict(tag=args.tag, kind="torch", args=vars(args),
                       features=names, n_state=n_state, fset=args.fset,
                       config=config, best_epoch=best_ep, epochs=len(hist),
                       val=best_val, history=hist), fh, indent=1)
    print(f"[saved] {MODELS}/{args.tag}_torch.pt")


if __name__ == "__main__":
    main()
