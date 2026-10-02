"""Where does the torch RowNet lose vs the trees offline?  Rest decisions?"""
import sys, json, os
import numpy as np
sys.path.insert(0, r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak\ML\code")
from dataset import load, split_masks, decision_id, route_class
from gbt_policy import GBTPolicy
from torch_policy import TorchPolicy
import torch
torch.set_num_threads(2)

d = load()
names_all = [str(x) for x in d["feature_names"]]
vocab = [str(a) for a in d["action_vocab"]]
act = np.array(vocab)[d["action_ix"]]
is_rest = np.char.endswith(act.astype(str), "_r1") | np.char.endswith(act.astype(str), "_r2")
tr, va, te = split_masks(d)
clean = d["clean"].astype(bool)

def preds(pol):
    cols = [names_all.index(n) for n in pol.state_names + pol.action_names]
    X = d["X"][:, cols].astype(np.float64 if pol.kind == "gbt" else np.float32)
    out_c, out_f = [], []
    for s in range(0, len(X), 65536):
        c, f = pol._predict(X[s:s + 65536])
        out_c.append(np.asarray(c, dtype=np.float64)); out_f.append(np.asarray(f))
    return np.concatenate(out_c), np.concatenate(out_f)

def choose(mask, pred, feas):
    did = decision_id(d)[mask]
    p = pred[mask].copy() + 1e6 * (feas[mask] < 0.5)
    p[~clean[mask]] = np.inf
    order = np.lexsort((p, did))
    did_s = did[order]
    start = np.flatnonzero(np.r_[True, did_s[1:] != did_s[:-1]])
    rows = np.flatnonzero(mask)[order][start]
    return rows   # chosen row per decision

def teacher_rows(mask):
    did = decision_id(d)[mask]
    r = d["regret"][mask].copy(); r[~clean[mask]] = np.inf
    order = np.lexsort((r, did))
    did_s = did[order]
    start = np.flatnonzero(np.r_[True, did_s[1:] != did_s[:-1]])
    return np.flatnonzero(mask)[order][start]

pols = {"trees": GBTPolicy("gbt_F95_base_s0"), "torch": TorchPolicy("tmlp_F95_base_s0")}
P = {k: preds(p) for k, p in pols.items()}
cls = route_class(d)
for split, m in (("stop", va), ("test", te)):
    tch = teacher_rows(m)
    print(f"\n=== {split}: {len(tch)} decisions, teacher rests at {is_rest[tch].sum()}")
    for k, (pc, pf) in P.items():
        ch = choose(m, pc, pf)
        reg = d["regret"][ch] * 60
        ok = np.isfinite(reg)
        fr = is_rest[ch] & ~is_rest[tch]     # rests the teacher did not take
        mr = ~is_rest[ch] & is_rest[tch]     # rests the teacher took, model skipped
        line = f"  {k:6s} regret {np.nanmean(reg[ok]):5.2f} min | false rests {fr.sum():3d} (mean regret {np.nanmean(reg[fr & ok]) if (fr&ok).any() else 0:6.1f} min, sum {np.nansum(reg[fr & ok]):7.0f})"
        line += f" | missed rests {mr.sum():3d} (mean {np.nanmean(reg[mr & ok]) if (mr&ok).any() else 0:6.1f}, sum {np.nansum(reg[mr & ok]):6.0f}) | other sum {np.nansum(reg[~fr & ~mr & ok]):6.0f}"
        print(line)
        for c in ("short", "medium", "long"):
            cm = cls[ch] == c
            print(f"      {c:6s} regret {np.nanmean(reg[cm & ok]):5.2f} min  false rests {(fr & cm).sum():3d}  sum regret {np.nansum(reg[fr & cm & ok]):6.0f} min")

# rest-row cost prediction error on clean rows, test split
for k, (pc, pf) in P.items():
    for nm, sel in (("rest rows", is_rest), ("other rows", ~is_rest)):
        mm = te & clean & sel & np.isfinite(d["regret"])
        err = pc[mm] - d["regret"][mm]
        print(f"{k:6s} {nm:10s} n={mm.sum():6d}  bias {err.mean()*60:+7.1f} min  MAE {np.abs(err).mean()*60:6.1f} min  "
              f"true mean {d['regret'][mm].mean()*60:6.1f} min")
