"""
train.py — behaviour cloning of the MIP-tail lookahead policy
=============================================================
EASY MODE — what "training" actually is
---------------------------------------
We have ~15k flashcards: (situation -> what the teacher did).  One training
step: show the network a random batch of cards, measure how wrong its answers
are (the LOSS), compute for every knob which direction reduces the loss
(BACKPROPAGATION — an automated chain rule), nudge every knob a tiny bit that
way (the OPTIMISER).  Repeat until answers on cards the network has NEVER
seen (the validation set) stop improving — then stop (EARLY STOPPING),
because further training only memorises the training cards (OVERFITTING).

The three defences against fooling ourselves, all implemented here:
  1. Splits are BY INSTANCE (routes never straddle train/val/test) and the
     out-of-distribution families (long routes, unseen TW classes) are never
     touched during training — they are the paper's real exam.
  2. Normalisation statistics come from the TRAIN split only.  Using val or
     test rows to compute means/stds would leak information about the exam
     into the study session (subtle, classic mistake).
  3. Class weighting: 88% of decisions are "just drive on".  Unweighted, the
     network gets 88% accuracy by always saying that, having learned nothing.
     We weight rare classes up (tempered by sqrt so a 2-example class does
     not dominate) and we report BALANCED accuracy = mean per-class recall,
     which an always-say-pass policy cannot fake (it scores 1/12).

Loss = weighted cross-entropy (discrete decision)
       + SmoothL1 on charge duration, counted only where the teacher charged.
Cross-entropy = -log(probability the model gave to the teacher's choice):
confidently right -> ~0; confidently wrong -> large.  SmoothL1 is squared
error near zero, absolute error for large mistakes (robust to outliers).

Run (repo root):   python ML/code/train.py [--epochs 200] [--seed 0]
Output:            ML/models/policy_v1.pt  (weights + everything needed to
                   deploy: normalisation stats, class list, feature names)
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import StudentPolicy, build_mask

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "ML", "data")
OUT  = os.path.join(ROOT, "ML", "models")


def load_split(which: tuple[str, ...], k: int, mode: str, dagger: int = 0):
    """Rows for the requested splits, plus a per-row weight.

    DAgger rows are ~2% of the deck by count but are the only labels drawn
    from the state distribution the student actually produces, so they are
    upweighted (see merge_dagger.py).  Base rows get weight 1.
    """
    from extract_dataset import tag_for
    suffix = f"_dagger{dagger}" if dagger else ""
    z = np.load(os.path.join(DATA, f"dataset_{tag_for(k, mode)}{suffix}.npz"),
                allow_pickle=False)
    sel = np.isin(z["split"], which)
    src = z["source"][sel] if "source" in z.files else None
    return z["X"][sel], z["y"][sel], z["tauc"][sel], src


def main(a):
    from extract_dataset import tag_for
    dtag = tag_for(a.k, a.mode)
    tag = (f"{dtag}_seed{a.seed}"
           + ("" if a.cw_power == 0.5 else f"_cw{a.cw_power:g}")
           + (f"_dag{a.dagger}" if a.dagger else "")
           + ("" if (a.hidden == 128 and a.depth == 2)
              else f"_d{a.depth}w{a.hidden}"))
    torch.manual_seed(a.seed); np.random.seed(a.seed)   # reproducibility:
    # neural training is stochastic (init + batch order); fixing seeds makes
    # one run repeatable, and the paper reports spread over several seeds.
    meta = json.load(open(os.path.join(DATA, f"meta_{dtag}.json")))
    classes = [tuple(c) for c in meta["classes"]]
    # the mask reads features by position — fail loudly if the layout moved.
    # These indices sit in the K-independent "dashboard" block, so the same
    # mask code is valid for every K in the ablation.
    fn = meta["feature_names"]
    assert fn[5] == "phi" and fn[6] == "rho2_used" and fn[8] == "at_cs" \
        and fn[19] == "allow_split", "feature order changed; update model.py"

    Xtr, ytr, ttr, str_ = load_split(("train",), a.k, a.mode, a.dagger)
    Xva, yva, tva, _ = load_split(("val",), a.k, a.mode, a.dagger)
    # per-row weights: aggregated (DAgger) rows count for more
    rw = np.ones(len(ytr), dtype=np.float32)
    if str_ is not None and a.dagger:
        rw[str_ != "base"] = a.dagger_weight
        print(f"DAgger rows: {int((str_ != 'base').sum()):,} "
              f"weighted x{a.dagger_weight}")
    print(f"train {len(ytr)} rows | val {len(yva)} rows | "
          f"{Xtr.shape[1]} features | {len(classes)} classes")

    # ── normalisation (train stats ONLY — see docstring, defence #2) ────────
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6      # +eps: constant features
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    Xtr, ytr, ttr = map(torch.as_tensor, (Xtr, ytr, ttr))
    rw = torch.as_tensor(rw)
    Xva, yva, tva = map(torch.as_tensor, (Xva, yva, tva))

    # ── masks + sanity: the teacher's own action must always be legal ───────
    mtr, mva = build_mask(Xtr * sd + mu, classes), build_mask(Xva * sd + mu, classes)
    bad = (~mtr[torch.arange(len(ytr)), ytr]).sum().item()
    if bad:
        print(f"WARNING: {bad} teacher actions violate the structural mask "
              f"-> unmasking those rows' true class (data wins over rules).")
        mtr[torch.arange(len(ytr)), ytr] = True
        mva[torch.arange(len(yva)), yva] = True

    # ── class weights (defence #3): tempered inverse frequency ─────────────
    # weight_c ∝ (N / n_c) ** cw_power.  power 0 = unweighted, 0.5 = sqrt.
    #
    # CALIBRATION MATTERS MORE THAN IT LOOKS.  At power 0.5 the dominant
    # "just drive on" class ends up at weight 0.035 while a 2-example class
    # gets 5.3 — a 150x ratio.  Predicting a REST when the truth is PASS then
    # costs almost nothing, and a daily rest costs 9-11 h of route time, so
    # the loss quietly stops caring about the single most expensive mistake
    # the policy can make.  Closed-loop diagnosis traced the entire heavy tail
    # to exactly one spurious rest per bad route, with 344 of 364 rests chosen
    # FREELY by the network rather than compelled by the guard.
    cnt = torch.bincount(ytr, minlength=len(classes)).float().clamp(min=1)
    w = (cnt.sum() / cnt) ** a.cw_power
    w = w / w.mean()
    print(f"class weights (power {a.cw_power}):", [f"{x:.2f}" for x in w])

    # charging classes: rows whose label has y=1 also supervise the tauc head
    is_charge = torch.tensor([int(c[0]) == 1 for c in classes])

    net = StudentPolicy(Xtr.shape[1], len(classes), hidden=a.hidden,
                        depth=a.depth)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)   # Adam: gradient
    # descent with per-knob adaptive step sizes — the boring, robust default.
    # reduction="none" so each row's loss can be scaled by its own weight
    # (class weight x row weight) before averaging
    ce_raw = nn.CrossEntropyLoss(weight=w, reduction="none")
    ce  = nn.CrossEntropyLoss(weight=w)          # validation (unweighted rows)
    l1  = nn.SmoothL1Loss()

    dl = DataLoader(TensorDataset(Xtr, ytr, ttr, mtr, rw), batch_size=a.batch,
                    shuffle=True)                       # fresh random batches
    best, best_state, patience = float("inf"), None, 0
    hist = dict(train_loss=[], val_loss=[], acc=[], bal_acc=[], tauc_mae=[])
    for epoch in range(a.epochs):
        net.train(); t0 = time.time(); ep_loss = []
        for xb, yb, tb, mb, wb in dl:
            logits, tauc = net(xb, mb)
            chg = is_charge[yb]                          # rows where teacher charged
            loss = (ce_raw(logits, yb) * wb).sum() / wb.sum()
            if chg.any():
                loss = loss + a.lam * l1(tauc[chg], tb[chg])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss.append(loss.item())

        # ── validation: the honest number ───────────────────────────────────
        net.eval()
        with torch.no_grad():
            lo, ta = net(Xva, mva)
            vloss = ce(lo, yva).item()
            chg = is_charge[yva]
            if chg.any():
                vloss += a.lam * l1(ta[chg], tva[chg]).item()
            pred = lo.argmax(1)
            acc  = (pred == yva).float().mean().item()
            recalls = [(pred[yva == c] == c).float().mean().item()
                       for c in range(len(classes)) if (yva == c).any()]
            bacc = float(np.mean(recalls))
            mae_min = ((ta[chg] - tva[chg]).abs().mean().item() * 60
                       if chg.any() else 0.0)
        for k_, v_ in (("train_loss", float(np.mean(ep_loss))),
                       ("val_loss", vloss), ("acc", acc),
                       ("bal_acc", bacc), ("tauc_mae", mae_min)):
            hist[k_].append(v_)
        if a.verbose:
            print(f"ep {epoch:3d}  val_loss {vloss:.4f}  acc {acc:.3f}  "
                  f"bal_acc {bacc:.3f}  tauc_mae {mae_min:.1f}min  "
                  f"({time.time()-t0:.1f}s)")
        if vloss < best - 1e-4:
            best, best_state, patience = vloss, net.state_dict(), 0
        else:
            patience += 1
            if patience >= a.patience:
                print(f"early stop: no val improvement for {a.patience} epochs")
                break

    os.makedirs(OUT, exist_ok=True)
    torch.save(dict(state=best_state, mu=mu, sd=sd, classes=classes,
                    feature_names=fn, hidden=a.hidden, seed=a.seed,
                    k_look=a.k,           # rollout re-builds features with this
                    split_mode=a.mode, depth=a.depth),
               os.path.join(OUT, f"policy_{tag}.pt"))
    json.dump(hist, open(os.path.join(OUT, f"history_{tag}.json"), "w"))
    best_ep = int(np.argmin(hist["val_loss"]))
    print(f"[K={a.k} seed={a.seed}] best epoch {best_ep}: "
          f"val_loss {hist['val_loss'][best_ep]:.4f}  "
          f"acc {hist['acc'][best_ep]:.3f}  bal_acc {hist['bal_acc'][best_ep]:.3f}  "
          f"tauc_mae {hist['tauc_mae'][best_ep]:.1f}min  -> policy_{tag}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10,
                   help="lookahead nodes; must match an extracted dataset_K*.npz")
    p.add_argument("--split-mode", dest="mode", default="family",
                   choices=["family", "seed"],
                   help="which extracted dataset to train on")
    p.add_argument("--dagger", type=int, default=0,
                   help="train on dataset_*_dagger{N}.npz (0 = base only)")
    p.add_argument("--dagger-weight", dest="dagger_weight", type=float, default=5.0,
                   help="per-row loss weight for aggregated DAgger rows")
    p.add_argument("--cw-power", dest="cw_power", type=float, default=0.5,
                   help="class-weight exponent: 0 = unweighted, 0.5 = sqrt "
                        "inverse frequency (see note in the code)")
    p.add_argument("--verbose", action="store_true", help="per-epoch lines")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=2,
                   help="number of hidden layers in the shared trunk")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1.0,
                   help="weight of the charge-duration loss term")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
