"""Synthetic test of dataset.add_dagger: a fake label shard copied from real rows."""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from dataset import add_dagger, decision_id, load, split_masks  # noqa: E402
from torch_train import decision_table                          # noqa: E402

COLS = ["X", "action_ix", "regret", "cost", "std", "ok", "n_scen", "tauc", "taub",
        "clean", "stop", "chosen", "tiebreak", "best_cost", "n_actions", "n_clean"]
ROOT = os.path.join(os.path.dirname(__file__), "_dagger_test")

d = load()
inst = [str(x) for x in d["instances"]]


def shard(fn, route, seed):
    m = d["instance_ix"] == inst.index(route)
    n = int(m.sum())
    os.makedirs(os.path.join(ROOT, "t1", "labels"), exist_ok=True)
    np.savez(os.path.join(ROOT, "t1", "labels", fn + ".npz"),
             **{c: d[c][m] for c in COLS},
             feature_names=d["feature_names"], n_state=d["n_state"],
             action_vocab=d["action_vocab"], instance=np.array(route + "@t1:fake"),
             route=np.array(route), family=np.array(route.rsplit("_", 1)[0]),
             seed=np.array(seed), physics=np.array("base"), label=np.array("t1"),
             model_tag=np.array("fake"), student_key=np.array(["y0_go"] * n),
             is_check=np.zeros(n, np.int8))
    return n


try:
    n = shard("a", "RlongCfewTlarge_5", 5)
    d2 = add_dagger(d, ["t1"], root=ROOT)
    fit, stop, test = split_masks(d2)
    new = d2["source"] == 1
    print("rows added", int(new.sum()), "expected", n)
    print("all new rows in fit:", bool(fit[new].all()),
          "| none in stop/test:", not (stop[new].any() or test[new].any()))
    did = decision_id(d2)
    print("decision ids distinct from the teacher's:",
          len(np.intersect1d(did[new], did[~new])) == 0)
    print("decision table:", decision_table(d2).shape)

    os.remove(os.path.join(ROOT, "t1", "labels", "a.npz"))
    shard("b", "RlongCfewTlarge_20", 20)
    try:
        add_dagger(d, ["t1"], root=ROOT)
        print("ERROR: a stop-seed shard was accepted")
    except ValueError as e:
        print("stop-seed shard refused:", str(e)[-55:])
finally:
    shutil.rmtree(ROOT, ignore_errors=True)
