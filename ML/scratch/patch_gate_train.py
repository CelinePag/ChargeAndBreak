"""One-off patch (2026-10-21): gate_train.py --label rollout (labels from
gate_rollout.py: the value of a call with the student driving on, which can be
negative).  The share recovered becomes NET: calls on decisions where the LA's
action hurts count against the gate."""
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "gate_train.py")
s = open(P, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:60], s.count(old))
    s = s.replace(old, new)


sub('''def recovered(label, score, b, rng=None):
    """Share (%) of the total label recovered by calling on the top-b share."""
    k = int(round(b * len(label)))
    if rng is not None:                       # random tie-break for 0/1 signals
        score = score + 1e-6 * rng.random(len(score))
    return label[np.argsort(-score)[:k]].sum() / label.sum() * 100''',
'''def recovered(label, score, b, rng=None):
    """Share (%) of the total positive label recovered by calling on the top-b
    share, NET of the calls whose label is negative (rollout labels: the LA's
    action hurts once the student drives on)."""
    k = int(round(b * len(label)))
    if rng is not None:                       # random tie-break for 0/1 signals
        score = score + 1e-6 * rng.random(len(score))
    return label[np.argsort(-score)[:k]].sum() / np.clip(label, 0, None).sum() * 100''')

sub('''    y = g["label"] * 60.0                                         # minutes
    if target == "log":
        y = np.log1p(y)''',
'''    y = g["label"] * 60.0                                         # minutes
    if target == "log":
        y = np.sign(y) * np.log1p(np.abs(y))                      # signed: rollout labels can be < 0''')

sub('''    if target == "clf":
        params.update(objective="binary", metric="auc")
        y = (y > 1.0).astype(float)''',
'''    if target == "clf":
        params.update(objective="binary", metric="auc")
        y = (y > 1.0).astype(float)
        if "label_se" in g:                   # rollout labels: helps by more than 2 se
            y = ((g["label"] > 2 * g["label_se"]) & (g["label"] * 60 > 1.0)).astype(float)''')

sub('''    ap.add_argument("--test", action="store_true",''',
'''    ap.add_argument("--label", default="onestep", choices=["onestep", "rollout"],
                    help="onestep: the LA's own regret of the student's pick (gate_data.py); "
                         "rollout: the value of the call with the student driving on (gate_rollout.py)")
    ap.add_argument("--test", action="store_true",''')

sub('''    g = dict(np.load(os.path.join(DATA, f"gate_{args.arm}.npz"), allow_pickle=True))
    seed = g["seed"]''',
'''    g = dict(np.load(os.path.join(DATA, f"gate_{args.arm}.npz"), allow_pickle=True))
    if args.label == "rollout":
        r = np.load(os.path.join(DATA, f"gate_{args.arm}_rollout.npz"), allow_pickle=True)
        keep = np.asarray(r["row"])
        for key in ("signals", "state", "seed", "instance_ix", "stop"):
            g[key] = g[key][keep]
        g["label"], g["label_se"] = np.asarray(r["value"]), np.asarray(r["se"])
        print(f"[label] rollout values (K = {int(r['k'])} futures, student driving on) "
              f"on {len(keep)} decisions; the call hurts on {np.mean(g['label'] < 0)*100:.1f} %")
    seed = g["seed"]''')

sub('''    tag = f"gate_{args.arm}"''', '''    tag = f"gate_{args.arm}" + ("_ro" if args.label == "rollout" else "")''')

open(P, "w", encoding="utf-8").write(s)
print("patched", P)
