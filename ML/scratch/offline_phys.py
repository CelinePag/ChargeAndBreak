import sys, os, json, numpy as np, lightgbm as lgb
sys.path.insert(0, r"c:/Users/celinep/Documents/GitHub/ChargeAndBreak/ML/code")
from dataset import load_multi, split_masks, argmin_policy_regret, teacher_top1, MODELS
tags = sys.argv[1].split(",")
models = {}
for t in tags:
    meta = json.load(open(os.path.join(MODELS, f"{t}_meta.json")))
    models[t] = (lgb.Booster(model_file=os.path.join(MODELS, f"{t}_cost.txt")),
                 lgb.Booster(model_file=os.path.join(MODELS, f"{t}_feas.txt")), meta["features"])
feats = sorted(set().union(*[set(m[2]) for m in models.values()]))
print(f"{'physics':8s} {'split':5s} " + "".join(f"{t[8:]:>22s}" for t in tags) + "   (mean regret min | top-1)")
for ph in ["base","kwh300","kwh700","kwh900","kw150","kw700","kw1000","cs30","cs100"]:
    d = load_multi([ph], columns=set(feats) | {"a_y"})
    names = [str(x) for x in d["feature_names"]]
    fit, stop, test = split_masks(d)
    for nm, m in (("stop", stop), ("test", test)):
        if m.sum() == 0: continue
        cells = []
        for t, (cm, fm, fl) in models.items():
            X = d["X"][:, [names.index(n) for n in fl]]
            pred = cm.predict(X, num_threads=2); feas = fm.predict(X, num_threads=2)
            r, n = argmin_policy_regret(d, m, pred, feas); t1 = teacher_top1(d, m, pred)
            cells.append(f"{r*60:8.2f} | {100*t1:5.1f}% ")
        print(f"{ph:8s} {nm:5s} " + "".join(f"{c:>22s}" for c in cells) + f"  dec={n}")
    del d
