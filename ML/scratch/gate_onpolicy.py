"""Step 2 of the solver-call gate (2026-10-20): does the gate still rank well at
states the STUDENT reached itself?  The DAgger probe (data/dagger/probe) holds
LA labels at 3 sampled stops of each of 36 validation routes, per student
(gbt_F95_base_s0, tmlp_F95_split_list_s0) -- the teacher-check rows dropped.
Same signals and gate as gate_data.py / gate_train.py.  108 decisions per
student, so this is a sanity check, not a measurement."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import gate_data as gd                                            # noqa: E402
import lightgbm as lgb                                            # noqa: E402

ML = os.path.join(HERE, "..")
COLS = ["X", "action_ix", "regret", "clean", "stop"]

for arm, tag in (("gbt", "gbt_F95_base_s0"), ("torch", "tmlp_F95_split_list_s0")):
    fs = sorted(glob.glob(os.path.join(ML, "data", "dagger", "probe", "labels", f"{tag}__*.npz")))
    parts, inst, seeds = {c: [] for c in COLS}, [], []
    for k, f in enumerate(fs):
        z = np.load(f, allow_pickle=True)
        keep = z["is_check"] == 0
        for c in COLS:
            parts[c].append(z[c][keep])
        inst.append(np.full(keep.sum(), k))
        seeds.append(np.full(keep.sum(), int(z["seed"])))
        names, ns, vocab = z["feature_names"], int(z["n_state"]), z["action_vocab"]
    d = {c: np.concatenate(v) for c, v in parts.items()}
    d.update(instance_ix=np.concatenate(inst), seed=np.concatenate(seeds), feature_names=names,
             n_state=ns, action_vocab=vocab, instances=np.array([os.path.basename(f) for f in fs]))
    rows = np.arange(len(d["stop"]))
    P = {"gbt": gd.predict(d, "gbt_F95_base_s0", rows),
         "torch": gd.predict(d, "tmlp_F95_split_list_s0", rows),
         "oracle": gd.predict(d, "gbt_F95_oracle_s0", rows)}
    g = gd.build(d, arm, P)
    meta = json.load(open(os.path.join(ML, "models", f"gate_{arm}_meta.json")))
    b = lgb.Booster(model_file=os.path.join(ML, "models", f"gate_{arm}.txt"))
    X = np.concatenate([g["signals"], g["state"]], axis=1) if meta["block"] == "both" else g["signals"]
    sc = b.predict(X)
    lab = g["label"]
    print(f"\n{tag}: {len(lab)} student-visited decisions, value of a call {lab.mean()*60:.2f} min "
          f"(LA-visited validation decisions: see gate_train.py), nonzero {int((lab > 1e-9).sum())}")
    for bud, thr in meta["thresholds"].items():
        fire = sc >= thr
        print(f"  gate at the {bud:>2s} % threshold: fires on {fire.mean()*100:5.1f} % of these decisions, "
              f"recovers {lab[fire].sum() / max(lab.sum(), 1e-12) * 100:5.1f} % of their loss")
