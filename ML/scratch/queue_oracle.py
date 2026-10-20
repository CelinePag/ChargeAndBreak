"""Option A (2026-10-20, user: "do the oracle training with option A"):
trees and PyTorch trained on the hindsight oracle's actions
(ML/data/dataset_oracle.npz, extract_oracle.py) with the recipes of the
LA-trained headline models (gbt_F95_base_s*, tmlp_F95_split_list_s*), then
each driven on the 125 base test routes with the full shield (g99sr) and with
nominal features (g99srN), the same two variants the LA-trained models have.
Log: ML/logs/queue_oracle.log"""
import datetime
import os
import subprocess

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_oracle.log")
DATA = os.path.join(ROOT, "ML", "data", "dataset_oracle.npz")


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def run(args, log):
    with open(os.path.join(ROOT, "ML", "logs", log), "w", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-u"] + args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    say(f"rc={rc}  {' '.join(args[:3])}")


def have(tag):
    return os.path.exists(os.path.join(ROOT, "ML", "models", f"{tag}_meta.json"))


say(f"oracle chain started (pid {os.getpid()})")
models = []
for s in range(3):                                   # trees first: minutes each
    tag = f"gbt_F95_oracle_s{s}"
    if not have(tag):
        run(["ML/code/gbt_train.py", "--data", DATA, "--fset", "F", "--tag", tag,
             "--seed", str(s)], f"train_{tag}.log")
    models.append(("gbt", tag))
for s in range(3):
    tag = f"tmlp_F95_oracle_split_list_s{s}"
    if not have(tag):
        run(["ML/code/torch_train.py", "--data", DATA, "--tag", tag, "--fset", "F",
             "--arch", "split", "--lambda-list", "1", "--tau", "0.25",
             "--seed", str(s), "--threads", "4"], f"train_{tag}.log")
    models.append(("torch", tag))
for kind, tag in models:
    if not have(tag):
        say(f"missing model {tag}, not evaluated")
        continue
    for var, extra in (("g99sr", []), ("g99srN", ["--nominal-features"])):
        out = f"eval_{tag}_{var}_test.json"
        if os.path.exists(os.path.join(ROOT, "ML", "results", out)):
            continue
        run(["ML/code/evaluate.py", "--kind", kind, "--tag", tag, "--split", "test",
             "--guard-q", "0.99", "--spread-room", "--out", out] + extra,
            f"eval_{tag}_{var}_test.log")
say("oracle chain done")
