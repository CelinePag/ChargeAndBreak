"""Detached chain (2026-10-02): direction B, baseline B0 -- the chosen torch
recipe (split + listwise) trained on EVERY physics value, then driven on the
mixed routes and compared with the LA.  Log: ML/logs/queue_torch.log"""
import datetime
import os
import subprocess

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_torch.log")
PHYS = "base,kwh300,kwh700,kwh900,kw150,kw700,kw1000,cs30,cs100"
TAG = "tmlp_F95_phys_split_list_s0"


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def run(args, log):
    with open(os.path.join(ROOT, "ML", "logs", log), "w", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-u"] + args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    say(f"rc={rc}  {' '.join(args)}")
    return rc


say(f"B0 chain started (pid {os.getpid()})")
if run(["ML/code/torch_train.py", "--tag", TAG, "--physics", PHYS, "--seed", "0",
        "--threads", "2", "--lambda-list", "1", "--tau", "0.25"], f"train_{TAG}.log") == 0:
    run(["ML/code/mixed_eval.py", "drive", "--jobs", "2"], "mixed_drive_B0.log")
    run(["ML/code/mixed_eval.py", "la", "--variants", "pmix"], "mixed_la_B0.log")
say("B0 chain done")
