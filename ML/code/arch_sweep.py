"""
arch_sweep.py — how small can the policy get?
=============================================
Sweeps the two things that set model size — the look-ahead window K (which
fixes the input width, 23 + 6K) and the trunk (depth x width) — and reports
supervised fit for each.

USE THIS ONLY TO SHORTLIST.  Supervised metrics have pointed the wrong way
four times in this project (class weighting, K, all-families data, the RL
"win"), so nothing here decides anything: the finalists go to a closed-loop
rollout, and that is what picks the model.

Prints one line per configuration as it finishes, so a long sweep can be read
while it runs and killed early without losing the results already in hand.

  python ML/code/arch_sweep.py --epochs 25 --seeds 1
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "ML", "data")


def load(k):
    z = np.load(os.path.join(DATA, f"dataset_K{k}_seedsplit.npz"), allow_pickle=False)
    tr, va = z["split"] == "train", z["split"] == "val"
    Xtr, ytr = z["X"][tr], z["y"][tr].astype(np.int64)
    Xva, yva = z["X"][va], z["y"][va].astype(np.int64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (torch.tensor((Xtr - mu) / sd), torch.tensor(ytr),
            torch.tensor((Xva - mu) / sd), torch.tensor(yva))


def mlp(nf, w, depth, n_cls=12):
    """depth = number of HIDDEN layers; 0 is plain multinomial logistic
    regression, which is the honest floor for 'is the trunk earning its keep'."""
    if depth == 0:
        return nn.Linear(nf, n_cls)
    layers = [nn.Linear(nf, w), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(w, w), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(w, n_cls))


def fit(model, data, epochs, seed):
    torch.manual_seed(seed)
    Xtr, ytr, Xva, yva = data
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, ytr), batch_size=512, shuffle=True)
    best = (9e9, 0.0, 0)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            loss = ce(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = ce(model(Xva), yva).item()
            pv = model(Xva).argmax(1)
            bal = float(np.mean([(pv[yva == c] == c).float().mean().item()
                                 for c in range(12) if (yva == c).any()]))
        if vl < best[0]:
            best = (vl, bal, ep)
    return best


def main(a):
    ks = [int(x) for x in a.k]
    cache = {k: load(k) for k in ks}
    n_rows = len(cache[ks[0]][1])
    print(f"train rows {n_rows:,}\n")
    hdr = (f"{'K':>3s} {'feat':>5s} {'depth':>6s} {'width':>6s} {'params':>8s} "
           f"{'rows/p':>7s} {'val_loss':>9s} {'bal_acc':>8s} {'ep':>3s}")
    print(hdr); print("-" * len(hdr)); sys.stdout.flush()
    rows = []
    for k in ks:
        d = cache[k]; nf = d[0].shape[1]
        for depth in [int(x) for x in a.depth]:
            widths = [0] if depth == 0 else [int(x) for x in a.width]
            for w in widths:
                vls, bals = [], []
                for s in range(a.seeds):
                    m = mlp(nf, w, depth)
                    vl, bal, ep = fit(m, d, a.epochs, s)
                    vls.append(vl); bals.append(bal)
                n = sum(p.numel() for p in mlp(nf, w, depth).parameters())
                rows.append(dict(k=k, feat=nf, depth=depth, width=w, params=n,
                                 val_loss=float(np.mean(vls)),
                                 bal_acc=float(np.mean(bals))))
                print(f"{k:3d} {nf:5d} {depth:6d} {w:6d} {n:8,} "
                      f"{n_rows/n:7.2f} {np.mean(vls):9.4f} {np.mean(bals):8.3f} {ep:3d}")
                sys.stdout.flush()
    json.dump(rows, open(os.path.join(DATA, "arch_sweep.json"), "w"), indent=1)
    ref = next((r for r in rows if r["k"] == 20 and r["depth"] == 2
                and r["width"] == 128), None)
    if ref:
        print(f"\nreference (K=20, 2x128): val_loss {ref['val_loss']:.4f}, "
              f"{ref['params']:,} params")
        print("configs within 2% of it, smallest first:")
        ok = [r for r in rows if r["val_loss"] <= ref["val_loss"] * 1.02]
        for r in sorted(ok, key=lambda r: r["params"])[:8]:
            print(f"   K={r['k']:<3d} depth={r['depth']} width={r['width']:<4d} "
                  f"{r['params']:7,} params  val_loss {r['val_loss']:.4f} "
                  f"({100*(r['val_loss']-ref['val_loss'])/ref['val_loss']:+.1f}%)  "
                  f"bal_acc {r['bal_acc']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", nargs="+", default=["0", "3", "5", "10", "20"])
    p.add_argument("--depth", nargs="+", default=["0", "1", "2"])
    p.add_argument("--width", nargs="+", default=["8", "16", "32", "64", "128"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seeds", type=int, default=1)
    main(p.parse_args())
