"""Synthetic end-to-end test: torch_train --physics cs30 --dagger _t, where the
fake label file holds real pmix rows (one pilot route) as if DAgger made them."""
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LAB = os.path.join(ROOT, "ML", "data", "dagger", "_t", "labels")
COLS = ["X", "action_ix", "regret", "cost", "std", "ok", "n_scen", "tauc", "taub",
        "clean", "stop", "chosen", "tiebreak", "best_cost", "n_actions", "n_clean"]

z = np.load(os.path.join(ROOT, "ML", "data", "dataset_phys_pmix47.npz"), allow_pickle=True)
inst = [str(x) for x in z["instances"]]
route = inst[0]
m = z["instance_ix"] == 0
n = int(m.sum())
os.makedirs(LAB, exist_ok=True)
try:
    np.savez(os.path.join(LAB, "fake.npz"), **{c: z[c][m] for c in COLS},
             feature_names=z["feature_names"], n_state=z["n_state"],
             action_vocab=z["action_vocab"], instance=np.array(route + "@_t:fake"),
             route=np.array(route), family=np.array(route.split("__")[0].rsplit("_", 1)[0]),
             seed=np.array(int(route.split("__")[0].rsplit("_", 1)[1])),
             physics=np.array("pmix"), label=np.array("_t"), model_tag=np.array("fake"),
             student_key=np.array(["y0_go"] * n), is_check=np.zeros(n, np.int8))
    out = subprocess.run([sys.executable, "-u", os.path.join(ROOT, "ML", "code", "torch_train.py"),
                          "--tag", "_t_dg", "--fset", "T", "--physics", "cs30", "--dagger", "_t",
                          "--epochs", "1", "--threads", "1", "--lambda-list", "1"],
                         capture_output=True, text=True, cwd=ROOT)
    lines = [l for l in out.stdout.splitlines() + out.stderr.splitlines()
             if l.startswith(("[dagger]", "[physics]", "[data]", "  fit", "[saved]", "Traceback"))
             or "Error" in l]
    print("\n".join(lines))
    print("return code", out.returncode)
finally:
    shutil.rmtree(os.path.join(ROOT, "ML", "data", "dagger", "_t"), ignore_errors=True)
    for f in ("_t_dg_torch.pt", "_t_dg_meta.json"):
        p = os.path.join(ROOT, "ML", "models", f)
        if os.path.exists(p):
            os.remove(p)
