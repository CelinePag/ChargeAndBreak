"""Rollout labels for the solver-call gate (2026-10-21, user: "okay we can try
that"): the value of a call when the STUDENT drives on, K = 32 shared futures,
at every LA-visited decision of seeds 1-21 where the student would act
differently.  Trees first, then PyTorch.  Log: ML/logs/queue_gate_rollout.log"""
import datetime
import os
import subprocess

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_gate_rollout.log")


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


say(f"gate rollout labels started (pid {os.getpid()})")
for arm in ("gbt", "torch"):
    out = os.path.join(ROOT, "ML", "data", f"gate_{arm}_rollout.npz")
    if os.path.exists(out):
        continue
    with open(os.path.join(ROOT, "ML", "logs", f"gate_rollout_{arm}.log"), "w", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-u", "ML/code/gate_rollout.py", "--arm", arm, "--k", "32",
                              "--jobs", "6"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    say(f"rc={rc}  {arm}")
say("gate rollout labels done")
