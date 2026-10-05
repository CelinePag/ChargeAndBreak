"""The validation winner (pool + 121 mixed routes x5, chosen 2026-10-04 on the
15 mixed-power validation routes) gets seeds 1-2, then all three seeds run
ONCE on the test routes: the 125 base-case routes and the mixed test routes.
Log: ML/logs/queue_w5_test.log"""
import datetime
import os
import subprocess

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_w5_test.log")
PHYS = "base,kwh300,kwh700,kwh900,kw150,kw700,kw1000,cs30,cs100"
TC = ["--fset", "T", "--lambda-list", "1", "--tau", "0.25", "--arch", "split"]


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def run(args, log):
    with open(os.path.join(ROOT, "ML", "logs", log), "w", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-u"] + args, cwd=ROOT, stdout=fh,
                             stderr=subprocess.STDOUT)
    say(f"rc={rc}  {' '.join(args[:3])}")


say(f"w5 test chain started (pid {os.getpid()})")
for s in (1, 2):
    tag = f"tmlp_T144_physpmix121w5_split_list_s{s}"
    if not os.path.exists(os.path.join(ROOT, "ML", "models", f"{tag}_meta.json")):
        run(["ML/code/torch_train.py", "--tag", tag, "--threads", "4", "--physics",
             f"{PHYS},pmix", "--weight-physics", "pmix=5", "--seed", str(s)] + TC,
            f"train_{tag}.log")
for s in (0, 1, 2):
    tag = f"tmlp_T144_physpmix121w5_split_list_s{s}"
    out = f"eval_{tag}_g99sr_test.json"
    run(["ML/code/evaluate.py", "--kind", "torch", "--tag", tag, "--split", "test",
         "--guard-q", "0.99", "--spread-room", "--out", out], out.replace(".json", ".log"))
run(["ML/code/mixed_eval.py", "drive", "--tags", "physpmix121w5", "--jobs", "6"],
    "mixed_test_drive_w5.log")
say("w5 test chain done")
