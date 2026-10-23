"""
gate_train.py — learn where a call to the LA is worth it (offline, one-step)
===========================================================================
Fits a LightGBM gate on gate_<arm>.npz (gate_data.py): training decisions of
seeds 1-19 (honest out-of-fold picks), chosen on seeds 20-21.  The score is
"value of calling the LA here"; the deployed rule calls the LA when the score
is in the top b % of decisions, with the threshold for each budget b read off
the validation decisions.

Measure: share of the student's one-step loss recovered when the LA is called
on a budget of b % of decisions (2, 5, 10, 20 %); the selection score is the
mean over the four budgets.  Compared at equal budgets with the triggers that
need no learning: the student's margin, its rest/no-rest margin (the current
hybrid trigger), softmax entropy, the two disagreement signals, and random.

    python ML/code/gate_train.py --arm gbt            # choose on validation
    python ML/code/gate_train.py --arm gbt --test     # the chosen gate, test once
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import DATA, MODELS                                   # noqa: E402

BUDGETS = (0.02, 0.05, 0.10, 0.20)


def recovered(label, score, b, rng=None):
    """Share (%) of the total positive label recovered by calling on the top-b
    share, NET of the calls whose label is negative (rollout labels: the LA's
    action hurts once the student drives on)."""
    k = int(round(b * len(label)))
    if rng is not None:                       # random tie-break for 0/1 signals
        score = score + 1e-6 * rng.random(len(score))
    return label[np.argsort(-score)[:k]].sum() / np.clip(label, 0, None).sum() * 100


def curve(label, score, rng=None):
    return [recovered(label, score, b, rng) for b in BUDGETS]


def baselines(g, m, rng):
    s = g["signals"][m]
    n = list(g["signal_names"])
    col = lambda name: s[:, n.index(name)]
    return {
        "perfect (hindsight)": g["label"][m],
        "student margin": -col("margin"),
        "rest/no-rest margin (current hybrid)": -col("rest_margin"),
        "entropy": col("entropy"),
        "other model disagrees":  # base: trees vs PyTorch; mixed: seed 1
            col("other_arm") - 1e-3 * col("margin"),
        **({"LA- and oracle-trained disagree": col("oracle_dis") - 1e-3 * col("margin")}
           if "oracle_dis" in n else {}),
        "random": rng.random(m.sum()),
    }


CONFIGS = {
    # name: (feature block, target)
    "signals, regression": ("signals", "reg"),
    "signals+state, regression": ("both", "reg"),
    "signals+state, log target": ("both", "log"),
    "signals+state, classifier >1 min": ("both", "clf"),
}


def features(g, block):
    if block == "signals":
        return g["signals"], list(g["signal_names"])
    return (np.concatenate([g["signals"], g["state"]], axis=1),
            list(g["signal_names"]) + list(g["state_names"]))


def fit(g, tr, va, block, target, seed=0):
    import lightgbm as lgb
    X, names = features(g, block)
    y = g["label"] * 60.0                                         # minutes
    if target == "log":
        y = np.sign(y) * np.log1p(np.abs(y))                      # signed: rollout labels can be < 0
    params = dict(learning_rate=0.03, num_leaves=31, min_child_samples=50,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
                  verbose=-1, seed=seed, num_threads=0)
    if target == "clf":
        params.update(objective="binary", metric="auc")
        y = (y > 1.0).astype(float)
        if "label_se" in g:                   # rollout labels: helps by more than 2 se
            y = ((g["label"] > 2 * g["label_se"]) & (g["label"] * 60 > 1.0)).astype(float)
    else:
        params.update(objective="huber", alpha=1.0, metric="huber")
    dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(X[va], label=y[va], reference=dtr, free_raw_data=False)
    b = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return b, X


def table(rows, title):
    print(f"\n{title}")
    print(f"  {'trigger':42s} " + " ".join(f"{int(b*100):>3d}% calls" for b in BUDGETS) + "   mean")
    for name, c in rows:
        print(f"  {name:42s} " + " ".join(f"{x:8.1f}%" for x in c) + f"  {np.mean(c):5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gbt", choices=["gbt", "torch"])
    ap.add_argument("--setting", default="base", choices=["base", "mixed"],
                    help="mixed: the mixed-route student's gate (gate_data.py --setting mixed); "
                         "fitted on seeds 1-16, chosen on 17-19")
    ap.add_argument("--label", default="onestep", choices=["onestep", "rollout"],
                    help="onestep: the LA's own regret of the student's pick (gate_data.py); "
                         "rollout: the value of the call with the student driving on (gate_rollout.py)")
    ap.add_argument("--test", action="store_true",
                    help="score the gate chosen on validation on the test decisions (once)")
    args = ap.parse_args()
    prefix = "gate_mixed_" if args.setting == "mixed" else "gate_"
    g = dict(np.load(os.path.join(DATA, f"{prefix}{args.arm}.npz"), allow_pickle=True))
    if args.label == "rollout":
        r = np.load(os.path.join(DATA, f"gate_{args.arm}_rollout.npz"), allow_pickle=True)
        keep = np.asarray(r["row"])
        for key in ("signals", "state", "seed", "instance_ix", "stop"):
            g[key] = g[key][keep]
        g["label"], g["label_se"] = np.asarray(r["value"]), np.asarray(r["se"])
        print(f"[label] rollout values (K = {int(r['k'])} futures, student driving on) "
              f"on {len(keep)} decisions; the call hurts on {np.mean(g['label'] < 0)*100:.1f} %")
    seed = g["seed"]
    tr, va, te = seed <= 19, (seed >= 20) & (seed <= 21), seed >= 22
    if args.setting == "mixed":          # no LA on the mixed validation routes: fold 4
        tr, va, te = seed <= 16, (seed >= 17) & (seed <= 19), np.zeros_like(tr)
    lab = g["label"]
    print(f"[gate] {args.arm}: fit {tr.sum()} / validation {va.sum()} / test {te.sum()} decisions; "
          f"value of a call, mean {lab[tr].mean()*60:.2f} / {lab[va].mean()*60:.2f} / "
          f"{lab[te].mean()*60:.2f} min")
    rng = np.random.default_rng(0)

    tag = f"{prefix}{args.arm}" + ("_ro" if args.label == "rollout" else "")
    meta_path = os.path.join(MODELS, f"{tag}_meta.json")
    if not args.test:
        rows = [(k, curve(lab[va], v, rng)) for k, v in baselines(g, va, rng).items()]
        best = None
        for name, (block, target) in CONFIGS.items():
            b, X = fit(g, tr, va, block, target)
            c = curve(lab[va], b.predict(X[va], num_iteration=b.best_iteration))
            rows.append((f"learned: {name}", c))
            if best is None or np.mean(c) > best[1]:
                best = (name, np.mean(c), b, X)
        table(rows, ("SELECTION FOLD (seeds 17-19)" if args.setting == "mixed" else "VALIDATION (seeds 20-21)") + ": share of the " + ("value of calling (student driving on)" if args.label == "rollout" else "one-step loss") + " recovered, net")
        name, _, b, X = best
        sva = b.predict(X[va], num_iteration=b.best_iteration)
        thr = {f"{int(x*100)}": float(np.sort(sva)[::-1][max(int(round(x * len(sva))) - 1, 0)])
               for x in BUDGETS}
        b.save_model(os.path.join(MODELS, f"{tag}.txt"), num_iteration=b.best_iteration)
        block, target = CONFIGS[name]
        with open(meta_path, "w") as fh:
            extra = (dict(student="tmlp_T144_physmix_split_list_s0",
                          other="tmlp_T144_physmix_split_list_s1", oracle=None)
                     if args.setting == "mixed" else {})
            json.dump(dict(tag=tag, arm=args.arm, config=name, block=block, target=target, **extra,
                           features=features(g, block)[1], thresholds=thr,
                           trees=b.best_iteration), fh, indent=1)
        imp = sorted(zip(b.feature_importance("gain"), features(g, block)[1]), reverse=True)[:12]
        print(f"\n[chosen] {name}; thresholds per budget (score): "
              + ", ".join(f"{k}% {v:.3g}" for k, v in thr.items()))
        print("[importance] " + ", ".join(f"{n} {v/sum(x for x, _ in imp)*100:.0f}%" for v, n in imp))
        print(f"[saved] {os.path.join(MODELS, tag)}.txt / _meta.json")
    else:
        import lightgbm as lgb
        meta = json.load(open(meta_path))
        b = lgb.Booster(model_file=os.path.join(MODELS, f"{tag}.txt"))
        X, _ = features(g, meta["block"])
        rows = [(k, curve(lab[te], v, rng)) for k, v in baselines(g, te, rng).items()]
        rows.append((f"learned: {meta['config']} (chosen)", curve(lab[te], b.predict(X[te]))))
        table(rows, "TEST (seeds 22-25), chosen gate, looked at once")


if __name__ == "__main__":
    main()
