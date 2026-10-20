"""How far apart are the two teachers?  Trees trained on LA labels vs on oracle
labels, each scored on the TEST rows of both datasets (2026-10-20, option A):
  on the LA dataset   -> mean LA regret (min) of the pick, and top-1 with the LA
  on the oracle data  -> agreement with the oracle's action"""
import os, sys
import numpy as np
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from dataset import load, split_masks, argmin_policy_regret, teacher_top1
from fsets import resolve
M = os.path.join(HERE, "..", "models")
DATA = os.path.join(HERE, "..", "data")


def preds(d, tag):
    names = [str(x) for x in d["feature_names"]]
    cols, _ = resolve("F", names, int(d["n_state"]), "gbt")
    X = d["X"][:, [names.index(c) for c in cols]]
    c = lgb.Booster(model_file=os.path.join(M, f"{tag}_cost.txt")).predict(X)
    f = lgb.Booster(model_file=os.path.join(M, f"{tag}_feas.txt")).predict(X)
    return c, f


for dname, path in (("LA labels", None), ("oracle labels", os.path.join(DATA, "dataset_oracle.npz"))):
    d = load(path)
    _, _, te = split_masks(d)
    go_ix = list(d["action_vocab"]).index("y0_go")
    r0, _ = argmin_policy_regret(d, te, np.where(d["action_ix"] == go_ix, 0.0, 1.0))
    print(f"\n== test rows scored against {dname} (always continue: {r0*60:.2f} min, "
          f"top-1 {teacher_top1(d, te, np.where(d['action_ix'] == go_ix, 0.0, 1.0))*100:.1f}%)")
    for tag in ("gbt_F95_base_s0", "gbt_F95_oracle_s0"):
        c, f = preds(d, tag)
        r, n = argmin_policy_regret(d, te, c, f)
        print(f"  {tag:20s} regret {r*60:6.2f} min   top-1 {teacher_top1(d, te, c)*100:5.1f}%   ({n} decisions)")
