"""One-off patch (2026-10-23): gate_data.py / gate_train.py get --setting mixed.
mixed: decisions of the LA on the 238 mixed training routes (dataset_phys_pmix,
dataset_phys_mix; seeds 1-19 only), student = tmlp_T144_physmix_split_list
(folds f0-f4 by seed, trained on the cluster: stage mixed-train), other arm =
gbt_P102_physmix (same data), no oracle signal (no oracle on those routes).
The gate is chosen on fold 4 (seeds 17-19) and fitted on folds 0-3."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def patch(path, subs):
    s = open(path, encoding="utf-8").read()
    for old, new in subs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)
    print("patched", path)


patch(os.path.join(HERE, "..", "code", "gate_data.py"), [
    ('''ARMS = {"gbt": ("gbt_F95_base_f{k}_s0", "gbt_F95_base_s0"),
        "torch": ("tmlp_F95_split_list_f{k}_s0", "tmlp_F95_split_list_s0")}
ORACLE = ("gbt_F95_oracle_f{k}_s0", "gbt_F95_oracle_s0")''',
     '''ARMS = {"gbt": ("gbt_F95_base_f{k}_s0", "gbt_F95_base_s0"),
        "torch": ("tmlp_F95_split_list_f{k}_s0", "tmlp_F95_split_list_s0")}
ORACLE = ("gbt_F95_oracle_f{k}_s0", "gbt_F95_oracle_s0")
# --setting mixed (2026-10-23): the mixed-route student and its trees twin,
# on the LA's mixed training routes; no oracle there, so no oracle signal
MIXED_ARMS = {"torch": ("tmlp_T144_physmix_split_list_f{k}_s0", "tmlp_T144_physmix_split_list_s0"),
              "gbt": ("gbt_P102_physmix_f{k}_s0", "gbt_P102_physmix_s0")}'''),
    ('''        po, _ = pick(*P[other], ix)
        pr, _ = pick(*P["oracle"], ix)''',
     '''        po, _ = pick(*P[other], ix)
        pr = pick(*P["oracle"], ix)[0] if "oracle" in P else p'''),
    ('''        sig.append([margin, rm, ent, feas[p], feas[top2].min(), len(cand), len(ix),
                    a_rest[p], a_brk[p], a_chg[p], float(po != p), cost[po] - cost[p],
                    float(pr != p), cost[pr] - cost[p]])''',
     '''        row = [margin, rm, ent, feas[p], feas[top2].min(), len(cand), len(ix),
               a_rest[p], a_brk[p], a_chg[p], float(po != p), cost[po] - cost[p]]
        if "oracle" in P:
            row += [float(pr != p), cost[pr] - cost[p]]
        sig.append(row)'''),
    ('''    return dict(label=np.array(lab), signals=np.array(sig, dtype=np.float32),
                signal_names=np.array(SIGNALS), state=np.array(st, dtype=np.float32),''',
     '''    names = SIGNALS if "oracle" in P else SIGNALS[:-2]
    return dict(label=np.array(lab), signals=np.array(sig, dtype=np.float32),
                signal_names=np.array(names), state=np.array(st, dtype=np.float32),'''),
    ('''def main():
    d = load()
    P = {arm: predictions(d, pair) for arm, pair in ARMS.items()}
    P["oracle"] = predictions(d, ORACLE)
    for arm in ARMS:
        g = build(d, arm, P)
        out = os.path.join(DATA, f"gate_{arm}.npz")''',
     '''def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="base", choices=["base", "mixed"])
    args = ap.parse_args()
    if args.setting == "mixed":
        from dataset import load_multi
        d = load_multi(["pmix", "mix"])
        P = {arm: predictions(d, pair) for arm, pair in MIXED_ARMS.items()}
        builds, prefix = ["torch"], "gate_mixed_"
    else:
        d = load()
        P = {arm: predictions(d, pair) for arm, pair in ARMS.items()}
        P["oracle"] = predictions(d, ORACLE)
        builds, prefix = list(ARMS), "gate_"
    for arm in builds:
        g = build(d, arm, P)
        out = os.path.join(DATA, f"{prefix}{arm}.npz")'''),
])

patch(os.path.join(HERE, "..", "code", "gate_train.py"), [
    ('''        "trees and PyTorch disagree": col("other_arm") - 1e-3 * col("margin"),
        "LA- and oracle-trained disagree": col("oracle_dis") - 1e-3 * col("margin"),
        "random": rng.random(m.sum()),
    }''',
     '''        "trees and PyTorch disagree": col("other_arm") - 1e-3 * col("margin"),
        **({"LA- and oracle-trained disagree": col("oracle_dis") - 1e-3 * col("margin")}
           if "oracle_dis" in n else {}),
        "random": rng.random(m.sum()),
    }'''),
    ('''    ap.add_argument("--label", default="onestep", choices=["onestep", "rollout"],''',
     '''    ap.add_argument("--setting", default="base", choices=["base", "mixed"],
                    help="mixed: the mixed-route student's gate (gate_data.py --setting mixed); "
                         "fitted on seeds 1-16, chosen on 17-19")
    ap.add_argument("--label", default="onestep", choices=["onestep", "rollout"],'''),
    ('''    g = dict(np.load(os.path.join(DATA, f"gate_{args.arm}.npz"), allow_pickle=True))''',
     '''    prefix = "gate_mixed_" if args.setting == "mixed" else "gate_"
    g = dict(np.load(os.path.join(DATA, f"{prefix}{args.arm}.npz"), allow_pickle=True))'''),
    ('''    tr, va, te = seed <= 19, (seed >= 20) & (seed <= 21), seed >= 22''',
     '''    tr, va, te = seed <= 19, (seed >= 20) & (seed <= 21), seed >= 22
    if args.setting == "mixed":          # no LA on the mixed validation routes: fold 4
        tr, va, te = seed <= 16, (seed >= 17) & (seed <= 19), np.zeros_like(tr)'''),
    ('''    tag = f"gate_{args.arm}" + ("_ro" if args.label == "rollout" else "")''',
     '''    tag = f"{prefix}{args.arm}" + ("_ro" if args.label == "rollout" else "")'''),
    ('''            json.dump(dict(tag=tag, arm=args.arm, config=name, block=block, target=target,''',
     '''            extra = (dict(student="tmlp_T144_physmix_split_list_s0", other="gbt_P102_physmix_s0",
                          oracle=None) if args.setting == "mixed" else {})
            json.dump(dict(tag=tag, arm=args.arm, config=name, block=block, target=target, **extra,'''),
])
