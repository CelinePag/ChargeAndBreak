"""Detached queue (survives the Claude Code session): finish the 8-route pmix
LA test batch (solved routes are skipped), then the trees learning curve, then
the 48-route pilot.  Keeps Windows awake while it runs.  Log: ML/logs/queue.log

Each step names its interpreter: the LA needs gurobipy (anaconda), the
learning curve needs lightgbm (.venv)."""
import ctypes
import datetime
import os
import subprocess

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY_LA = r"C:\Users\celinep\AppData\Local\anaconda3\python.exe"
PY_ML = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue.log")

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def run(py, args, log):
    with open(os.path.join(ROOT, "ML", "logs", log), "a", encoding="utf-8") as fh:
        return subprocess.call([py, "-u"] + args, cwd=ROOT, stdout=fh,
                               stderr=subprocess.STDOUT)


say(f"queue restarted (pid {os.getpid()}): LA test batch remainder")
rc = run(PY_LA, ["ML/code/run_la_mixed.py"], "la_mixed_pmix_resume.log")
say(f"LA test batch rc={rc}; learning curve")
rc = run(PY_ML, ["ML/code/learning_curve.py", "--jobs", "4"], "learning_curve.log")
say(f"learning curve rc={rc}; pilot LA on 48 short pmix training routes")
rc = run(PY_LA, ["ML/code/run_la_mixed.py", "--split", "train"], "la_mixed_train.log")
say(f"pilot rc={rc}; queue done")
