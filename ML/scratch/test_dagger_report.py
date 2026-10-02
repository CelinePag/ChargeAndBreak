"""Synthetic test of dagger_report: fake labels = the teacher's own decisions on
one stop-split route, half flagged as checks.  The check must report perfect
agreement (same chosen action everywhere, zero cost difference)."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from dagger_io import label_dir                                  # noqa: E402
from dataset import load                                         # noqa: E402

COLS = ["X", "action_ix", "regret", "cost", "std", "ok", "n_scen", "tauc", "taub",
        "clean", "stop", "chosen", "tiebreak", "best_cost", "n_actions", "n_clean"]
LABEL, ROUTE, TAG = "_selftest", "RlongCfewTlarge_20", "tmlp_F95_split_list_s0"

d = load()
m = d["instance_ix"] == [str(x) for x in d["instances"]].index(ROUTE)
stops = d["stop"][m]
vocab = np.array([str(x) for x in d["action_vocab"]])
chosen_key = {int(s): str(vocab[a]) for s, a, c in zip(stops, d["action_ix"][m], d["chosen"][m]) if c}
try:
    os.makedirs(label_dir(LABEL, "labels"), exist_ok=True)
    os.makedirs(label_dir(LABEL, "queries"), exist_ok=True)
    us = np.unique(stops)
    is_check = np.isin(stops, us[::2]).astype(np.int8)
    np.savez(os.path.join(label_dir(LABEL, "labels"), f"{TAG}__{ROUTE}.npz"),
             **{c: d[c][m] for c in COLS}, feature_names=d["feature_names"],
             n_state=d["n_state"], action_vocab=d["action_vocab"],
             instance=np.array(f"{ROUTE}@{LABEL}:{TAG}"), route=np.array(ROUTE),
             family=np.array(ROUTE.rsplit("_", 1)[0]), seed=np.array(20),
             physics=np.array("base"), label=np.array(LABEL), model_tag=np.array(TAG),
             student_key=np.array([chosen_key.get(int(s), "y0_go") for s in stops]),
             is_check=is_check)
    with open(os.path.join(label_dir(LABEL, "queries"), f"{TAG}__{ROUTE}.json"), "w") as fh:
        json.dump(dict(model=dict(kind="torch", tag=TAG), split="stop"), fh)
    subprocess.run([sys.executable, os.path.join(HERE, "..", "code", "dagger_report.py"),
                    "--label", LABEL, "--reference", "--check"], check=True)
finally:
    shutil.rmtree(label_dir(LABEL), ignore_errors=True)
