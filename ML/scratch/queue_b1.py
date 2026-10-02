"""Detached chain (2026-10-02), direction B: once the token re-extraction has
passed its check, train three all-physics torch models on the T inputs
(no structure / ChargerNet / ChargerNet with speed in g), then evaluate them
on the VALIDATION routes only: uniform stop split (evaluate.py) and the 16
mixed-power validation routes (mixed_eval --set val).  The test routes are
not touched.  Log: ML/logs/queue_b1.log"""
import datetime
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_b1.log")
RX_LOG = os.path.join(ROOT, "ML", "logs", "reextract_tokens.log")
PHYS = "base,kwh300,kwh700,kwh900,kw150,kw700,kw1000,cs30,cs100"
WAIT_PID = int(sys.argv[1])
COMMON = ["--fset", "T", "--physics", PHYS, "--seed", "0", "--threads", "2",
          "--lambda-list", "1", "--tau", "0.25"]
RUNS = [("tmlp_T144_phys_split_list_s0", ["--arch", "split"]),
        ("tmlp_T144_phys_charger_s0", ["--arch", "charger", "--g-exclude", "power"]),
        ("tmlp_T144_phys_chargerG_s0", ["--arch", "charger", "--g-exclude", "none"])]


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def run(args, log):
    with open(os.path.join(ROOT, "ML", "logs", log), "w", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-u"] + args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    say(f"rc={rc}  {' '.join(args)}")
    return rc


say(f"B1 chain started (pid {os.getpid()}); waiting for re-extraction pid {WAIT_PID}")
while alive(WAIT_PID):
    time.sleep(60)
with open(RX_LOG, encoding="utf-8") as fh:
    if "all datasets re-extracted" not in fh.read():
        say("re-extraction did not pass its check -- stopping")
        sys.exit(1)
for tag, extra in RUNS:
    run(["ML/code/torch_train.py", "--tag", tag] + COMMON + extra, f"train_{tag}.log")
for tag, _ in RUNS:
    run(["ML/code/evaluate.py", "--kind", "torch", "--tag", tag, "--split", "stop",
         "--guard-q", "0.99", "--spread-room", "--out", f"eval_{tag}_g99sr_stop.json"],
        f"eval_{tag}_g99sr_stop.log")
# the two references on the same uniform validation routes
for kind, tag in (("torch", "tmlp_F95_phys_split_list_s0"), ("gbt", "gbt_F95_phys_s1")):
    run(["ML/code/evaluate.py", "--kind", kind, "--tag", tag, "--split", "stop",
         "--guard-q", "0.99", "--spread-room", "--out", f"eval_{tag}_g99sr_stop.json"],
        f"eval_{tag}_g99sr_stop.log")
run(["ML/code/mixed_eval.py", "drive", "--set", "val", "--variants", "pmix", "--jobs", "2"],
    "mixed_val_drive_b1.log")
say("B1 chain done")
