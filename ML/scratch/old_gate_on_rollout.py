"""The one-step gate (gate_<arm>.txt) scored on the ROLLOUT labels of the
validation decisions (2026-10-21), so old and new gate compare on one measure:
net share of the positive value recovered at 2/5/10/20 % of decisions called."""
import os, sys, json
import numpy as np
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from gate_train import curve, features, BUDGETS
from dataset import DATA, MODELS
arm = sys.argv[1] if len(sys.argv) > 1 else "gbt"
g = dict(np.load(os.path.join(DATA, f"gate_{arm}.npz"), allow_pickle=True))
r = np.load(os.path.join(DATA, f"gate_{arm}_rollout.npz"), allow_pickle=True)
keep = np.asarray(r["row"])
for k in ("signals", "state", "seed"):
    g[k] = g[k][keep]
lab = np.asarray(r["value"])
va = (g["seed"] >= 20) & (g["seed"] <= 21)
out = {}
for model in (f"gate_{arm}", f"gate_{arm}_ro"):
    p = os.path.join(MODELS, f"{model}.txt")
    if not os.path.exists(p):
        continue
    meta = json.load(open(os.path.join(MODELS, f"{model}_meta.json")))
    X, _ = features(g, meta["block"])
    c = curve(lab[va], lgb.Booster(model_file=p).predict(X[va]))
    out[model] = c
    print(f"{model:14s} " + " ".join(f"{x:7.1f}%" for x in c) + f"   mean {np.mean(c):5.1f}")
print(f"(validation decisions {va.sum()}, of which the call helps on {np.sum(lab[va] > 0)}, hurts on {np.sum(lab[va] < 0)}; budgets {BUDGETS})")
