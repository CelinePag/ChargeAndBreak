"""Is the student's loss concentrated in few decisions? (2026-10-20, prelude to
'when is calling the solver worth it').  Trees gbt_F95_base_s0 on the TEST rows
of the LA dataset: at each LA-visited decision, the regret (LA horizon cost,
hours) of the action the student would pick.  One-step and on the LA's states,
so it ignores compounding -- a first look, not the answer."""
import os, sys, json
import numpy as np
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from dataset import load, split_masks, decision_id
from fsets import resolve

d = load()
names = [str(x) for x in d["feature_names"]]
cols, _ = resolve("F", names, int(d["n_state"]), "gbt")
X = d["X"][:, [names.index(c) for c in cols]]
M = os.path.join(HERE, "..", "models")
cost = lgb.Booster(model_file=os.path.join(M, "gbt_F95_base_s0_cost.txt")).predict(X)
feas = lgb.Booster(model_file=os.path.join(M, "gbt_F95_base_s0_feas.txt")).predict(X)
_, _, te = split_masks(d)
did = decision_id(d)
clean = d["clean"].astype(bool)
rows = np.flatnonzero(te & clean & np.isfinite(d["regret"]))
score = cost + 1e6 * (feas < 0.5)
per = {}
for i in rows:
    per.setdefault(did[i], []).append(i)
reg, margin, rest_gap, n_act = [], [], [], []
vocab = [str(x) for x in d["action_vocab"]]
for k, ix in per.items():
    if len(ix) < 2:
        continue
    ix = np.array(ix)
    o = ix[np.argsort(score[ix])]
    reg.append(float(d["regret"][o[0]]))
    margin.append(float(score[o[1]] - score[o[0]]))
    isrest = np.array([vocab[d["action_ix"][j]].endswith(("_r1", "_r2")) for j in ix])
    if isrest.any() and (~isrest).any():
        rest_gap.append(abs(score[ix][isrest].min() - score[ix][~isrest].min()))
    else:
        rest_gap.append(np.inf)
    n_act.append(len(ix))
reg, margin, rest_gap = np.array(reg), np.array(margin), np.array(rest_gap)
tot = reg.sum()
srt = np.sort(reg)[::-1]
print(f"test decisions with >=2 clean actions: {len(reg)}; mean student regret {reg.mean()*60:.2f} min; "
      f"zero-regret decisions {np.mean(reg < 1e-9)*100:.1f}%")
for q in (0.01, 0.02, 0.05, 0.10, 0.20):
    k = int(round(q * len(reg)))
    print(f"  top {q*100:4.0f}% of decisions hold {srt[:k].sum()/tot*100:5.1f}% of the regret")
def recovered(signal, frac):
    """call the LA on the `frac` decisions with the SMALLEST signal (most doubtful)"""
    k = int(round(frac * len(reg)))
    pick = np.argsort(signal)[:k]
    return reg[pick].sum() / tot * 100
rng = np.random.default_rng(0)
print("regret recovered by calling the LA on a share of decisions, by trigger:")
print(f"  {'share':>6s} {'oracle-ranked':>14s} {'pred. margin':>13s} {'rest margin':>12s} {'random':>8s}")
for f in (0.02, 0.05, 0.10, 0.20):
    print(f"  {f*100:5.0f}% {srt[:int(round(f*len(reg)))].sum()/tot*100:13.1f}% "
          f"{recovered(margin, f):12.1f}% {recovered(rest_gap, f):11.1f}% "
          f"{recovered(rng.random(len(reg)), f):7.1f}%")
