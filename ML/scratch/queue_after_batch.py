"""Queue: wait for tonight's LA batch, then the learning curve, then the
48-route pilot.  Keeps Windows from sleeping while it runs (released when
this process exits).  Log: ML/logs/queue.log"""
import ctypes
import datetime
import os
import subprocess
import sys
import time

ROOT = r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak"
WAIT_PID = int(sys.argv[1])
LOG = os.path.join(ROOT, "ML", "logs", "queue.log")

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


say(f"queue started; waiting for LA batch pid {WAIT_PID}")
while alive(WAIT_PID):
    time.sleep(120)
say("LA batch finished; learning curve")
rc = subprocess.call([sys.executable, "-u", "ML/code/learning_curve.py", "--jobs", "4"],
                     cwd=ROOT, stdout=open(os.path.join(ROOT, "ML", "logs", "learning_curve.log"), "w"),
                     stderr=subprocess.STDOUT)
say(f"learning curve rc={rc}; pilot LA on 48 short pmix training routes")
rc = subprocess.call([sys.executable, "-u", "ML/code/run_la_mixed.py", "--split", "train"],
                     cwd=ROOT, stdout=open(os.path.join(ROOT, "ML", "logs", "la_mixed_train.log"), "w"),
                     stderr=subprocess.STDOUT)
say(f"pilot rc={rc}; queue done")
