"""Offline: how often does argmin(pred | feas >= 0.5) pick a DIRTY action?"""
import sys, json
import numpy as np
sys.path.insert(0, r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak\ML\code")
from dataset import load, split_masks, decision_id, route_class
from gbt_policy import GBTPolicy
from torch_policy import TorchPolicy
import torch
torch.set_num_threads(2)

d = load()
names_all = [str(x) for x in d["feature_names"]]
tr, va, te = split_masks(d)
clean = d["clean"].astype(bool)
cls = route_class(d)

def preds(pol, mask):
    cols = [names_all.index(n) for n in pol.state_names + pol.action_names]
    ix = np.flatnonzero(mask)
    X = d["X"][np.ix_(ix, cols)].astype(np.float64 if pol.kind == "gbt" else np.float32)
    c, f = pol._predict(X)
    pc = np.zeros(len(clean)); pf = np.zeros(len(clean))
    pc[ix] = c; pf[ix] = f
    return pc, pf

def pick(mask, pc, pf):
    did = decision_id(d)[mask]
    p = pc[mask] + 1e6 * (pf[mask] < 0.5)
    order = np.lexsort((p, did))
    did_s = did[order]
    start = np.flatnonzero(np.r_[True, did_s[1:] != did_s[:-1]])
    return np.flatnonzero(mask)[order][start]

tags = [("gbt", "gbt_F95_base_s0"), ("torch", "tmlp_F95_base_s0"), ("torch", "tmlp_F95_sel_s0"),
        ("torch", "tmlp_F95_list_s0"), ("torch", "tmlp_F95_list01_s0"), ("torch", "tmlp_F95_list3_s0")]
n_dirty_dec = None
for kind, tag in tags:
    pol = GBTPolicy(tag) if kind == "gbt" else TorchPolicy(tag)
    pc, pf = preds(pol, va | tr)
    out = []
    for nm, m in (("fit", tr), ("stop", va)):
        ch = pick(m, pc, pf)
        dirty = ~clean[ch]
        lng = cls[ch] == "long"
        # feasibility head quality on rows: false 'feasible' on dirty rows
        mm = m & ~clean
        fp = (pf[mm] >= 0.5).mean()
        out.append(f"{nm}: dirty picks {dirty.sum():4d}/{len(ch)} (long {(dirty & lng).sum():3d})  dirty rows called feasible {fp*100:5.1f}%")
    print(f"{tag:22s} " + " | ".join(out))
