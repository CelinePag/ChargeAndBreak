"""Preview (2026-10-20): can a learned gate find the decisions worth a solver
call?  Student = trees gbt_F95_base_s0 (LA labels).  At every LA-visited
decision of the STOP split (seeds 20-21, never fitted) and TEST split (22-25):
  target  = LA regret of the student's pick (one-step, LA cost, hours)
  signals = student margins, seed disagreement (s1, s2), disagreement with the
            oracle-trained trees, and the 253 state features
Gate = LightGBM regressor fitted on STOP decisions, scored on TEST decisions:
call the LA on the top x% by predicted regret, report the share of regret
recovered.  One-step, on the LA's own states: a preview, not the closed loop."""
import os, sys
import numpy as np
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from dataset import load, split_masks, decision_id
from fsets import resolve

d = load()
names = [str(x) for x in d["feature_names"]]
ns = int(d["n_state"])
cols, _ = resolve("F", names, ns, "gbt")
XF = d["X"][:, [names.index(c) for c in cols]]
M = os.path.join(HERE, "..", "models")


def pred(tag):
    c = lgb.Booster(model_file=os.path.join(M, f"{tag}_cost.txt")).predict(XF)
    f = lgb.Booster(model_file=os.path.join(M, f"{tag}_feas.txt")).predict(XF)
    return c + 1e6 * (f < 0.5)


S = {t: pred(t) for t in ("gbt_F95_base_s0", "gbt_F95_base_s1", "gbt_F95_base_s2", "gbt_F95_oracle_s0")}
_, va, te = split_masks(d)
did = decision_id(d)
clean = d["clean"].astype(bool) & np.isfinite(d["regret"])
vocab = [str(x) for x in d["action_vocab"]]
isrest = np.array([v.endswith(("_r1", "_r2")) for v in vocab])[d["action_ix"]]


def table(mask):
    per = {}
    for i in np.flatnonzero(mask & clean):
        per.setdefault(did[i], []).append(i)
    R, F = [], []
    for k, ix in per.items():
        if len(ix) < 2:
            continue
        ix = np.array(ix)
        picks = {t: ix[np.argmin(s[ix])] for t, s in S.items()}
        s0 = S["gbt_F95_base_s0"][ix]
        o = np.argsort(s0)
        p = picks["gbt_F95_base_s0"]
        rg = (abs(s0[isrest[ix]].min() - s0[~isrest[ix]].min())
              if isrest[ix].any() and (~isrest[ix]).any() else 99.0)
        R.append(float(d["regret"][p]))
        F.append(np.r_[min(s0[o[1]] - s0[o[0]], 99.0), rg, len(ix), float(isrest[p]),
                       int(picks["gbt_F95_base_s1"] != p) + int(picks["gbt_F95_base_s2"] != p),
                       int(picks["gbt_F95_oracle_s0"] != p), d["X"][p, :ns]])
    return np.array(R), np.array(F, dtype=np.float32)


Rv, Fv = table(va)
Rt, Ft = table(te)
gate = lgb.train(dict(objective="huber", alpha=0.1, learning_rate=0.05, num_leaves=15,
                      min_child_samples=30, feature_fraction=0.7, verbose=-1, seed=0),
                 lgb.Dataset(Fv, label=Rv * 60), num_boost_round=300)
g = gate.predict(Ft)
tot = Rt.sum()


def rec(score, f):                       # call on the top-f share by score (higher = call)
    k = int(round(f * len(Rt)))
    return Rt[np.argsort(-score)[:k]].sum() / tot * 100


rng = np.random.default_rng(0)
print(f"gate fitted on {len(Rv)} stop-split decisions, scored on {len(Rt)} test decisions "
      f"(student regret {Rt.mean()*60:.2f} min per decision)")
print(f"{'calls':>6s} {'perfect':>8s} {'learned gate':>13s} {'margin':>7s} {'rest margin':>12s} "
      f"{'seeds disagree':>15s} {'teachers disagree':>18s} {'random':>7s}")
for f in (0.02, 0.05, 0.10, 0.20):
    print(f"{f*100:5.0f}% {rec(Rt, f):7.1f}% {rec(g, f):12.1f}% {rec(-Ft[:, 0], f):6.1f}% "
          f"{rec(-Ft[:, 1], f):11.1f}% {rec(Ft[:, 4] + 1e-6 * rng.random(len(Rt)), f):14.1f}% "
          f"{rec(Ft[:, 5] + 1e-6 * rng.random(len(Rt)), f):17.1f}% {rec(rng.random(len(Rt)), f):6.1f}%")
print(f"seed disagreement on {np.mean(Ft[:, 4] > 0)*100:.1f}% of decisions; "
      f"teachers' students disagree on {np.mean(Ft[:, 5] > 0)*100:.1f}%")
