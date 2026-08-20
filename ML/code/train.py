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


def load_split(which: tuple[str, ...]):
    z = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=False)
    sel = np.isin(z["split"], which)
    return z["X"][sel], z["y"][sel], z["tauc"][sel]


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)   # reproducibility:
    # neural training is stochastic (init + batch order); fixing seeds makes
    # one run repeatable, and the paper reports spread over several seeds.
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    classes = [tuple(c) for c in meta["classes"]]
    # the mask reads features by position — fail loudly if the layout moved:
    fn = meta["feature_names"]
    assert fn[5] == "phi" and fn[6] == "rho2_used" and fn[8] == "at_cs" \
        and fn[19] == "allow_split", "feature order changed; update model.py"

    Xtr, ytr, ttr = load_split(("train",))
    Xva, yva, tva = load_split(("val",))
    print(f"train {len(ytr)} rows | val {len(yva)} rows | "
          f"{Xtr.shape[1]} features | {len(classes)} classes")

    # ── normalisation (train stats ONLY — see docstring, defence #2) ────────
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6      # +eps: constant features
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    Xtr, ytr, ttr = map(torch.as_tensor, (Xtr, ytr, ttr))
    Xva, yva, tva = map(torch.as_tensor, (Xva, yva, tva))

    # ── masks + sanity: the teacher's own action must always be legal ───────
    mtr, mva = build_mask(Xtr * sd + mu, classes), build_mask(Xva * sd + mu, classes)
    bad = (~mtr[torch.arange(len(ytr)), ytr]).sum().item()
    if bad:
        print(f"WARNING: {bad} teacher actions violate the structural mask "
              f"-> unmasking those rows' true class (data wins over rules).")
        mtr[torch.arange(len(ytr)), ytr] = True
        mva[torch.arange(len(yva)), yva] = True

    # ── class weights (defence #3): sqrt-tempered inverse frequency ─────────
    cnt = torch.bincount(ytr, minlength=len(classes)).float().clamp(min=1)
    w = (cnt.sum() / cnt).sqrt(); w = w / w.mean()
    print("class weights:", [f"{x:.1f}" for x in w])

    # charging classes: rows whose label has y=1 also supervise the tauc head
    is_charge = torch.tensor([int(c[0]) == 1 for c in classes])

    net = StudentPolicy(Xtr.shape[1], len(classes), hidden=a.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)   # Adam: gradient
    # descent with per-knob adaptive step sizes — the boring, robust default.
    ce  = nn.CrossEntropyLoss(weight=w)
    l1  = nn.SmoothL1Loss()

    dl = DataLoader(TensorDataset(Xtr, ytr, ttr, mtr), batch_size=a.batch,
                    shuffle=True)                       # fresh random batches
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(a.epochs):
        net.train(); t0 = time.time()
        for xb, yb, tb, mb in dl:
            logits, tauc = net(xb, mb)
            chg = is_charge[yb]                          # rows where teacher charged
            loss = ce(logits, yb)
            if chg.any():
                loss = loss + a.lam * l1(tauc[chg], tb[chg])
            opt.zero_grad(); loss.backward(); opt.step()

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
                    feature_names=fn, hidden=a.hidden, seed=a.seed),
               os.path.join(OUT, f"policy_v1_seed{a.seed}.pt"))
    print("saved", os.path.join(OUT, f"policy_v1_seed{a.seed}.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1.0,
                   help="weight of the charge-duration loss term")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
