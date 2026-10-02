"""Detached queue (2026-10-02): wait for the LA queue (pilot) to finish, then
the DAgger probe on the stop split -- PyTorch + trees, 3 stops per route, one
teacher check per route where the teacher's log allows -- labelled by the LA
(anaconda), then the report.  Log: ML/logs/queue_dagger.log; report:
ML/logs/dagger_probe_report.txt.  The pmix round is NOT queued: it waits for
the user to see the probe."""
import datetime
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY_ML = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
PY_LA = r"C:\Users\celinep\AppData\Local\anaconda3\python.exe"
LOG = os.path.join(ROOT, "ML", "logs", "queue_dagger.log")
WAIT_PID = int(sys.argv[1])


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def run(py, args, log, mode="a"):
    with open(os.path.join(ROOT, "ML", "logs", log), mode, encoding="utf-8") as fh:
        rc = subprocess.call([py, "-u"] + args, cwd=ROOT, stdout=fh,
                             stderr=subprocess.STDOUT)
    say(f"rc={rc}  {os.path.basename(args[0])} {' '.join(args[1:])}")
    return rc


say(f"dagger probe queue started (pid {os.getpid()}); waiting for LA queue pid {WAIT_PID}")
while alive(WAIT_PID):
    time.sleep(120)
say("LA queue finished; probe rollouts")
common = ["--label", "probe", "--routes", "base", "--split", "stop", "--stops", "3"]
run(PY_ML, ["ML/code/dagger_rollout.py"] + common
    + ["--models", "torch:tmlp_F95_split_list_s0", "--check-stops", "1"], "dagger_probe.log")
run(PY_ML, ["ML/code/dagger_rollout.py"] + common
    + ["--models", "gbt:gbt_F95_base_s0"], "dagger_probe.log")
say("labelling (LA teacher)")
run(PY_LA, ["ML/code/dagger_label.py", "--label", "probe"], "dagger_probe.log")
run(PY_ML, ["ML/code/dagger_report.py", "--label", "probe", "--reference", "--check"],
    "dagger_probe_report.txt", mode="w")
say("probe done; report in ML/logs/dagger_probe_report.txt")
