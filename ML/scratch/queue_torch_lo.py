"""PyTorch, all charger powers but one (2026-10-20, user: "yes and report").
Same recipe as tmlp_F95_phys_split_list_s0 (B0: F95 inputs, split heads,
listwise term, seed 0, every physics value) with one charger power held out,
then each model drives its held-out axis and the base case.
Log: ML/logs/queue_torch_lo.log"""
import datetime
import os
import subprocess

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_torch_lo.log")
PHYS = "base,kwh300,kwh700,kwh900,kw150,kw700,kw1000,cs30,cs100"


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def run(args, log):
    with open(os.path.join(ROOT, "ML", "logs", log), "w", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-u"] + args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    say(f"rc={rc}  {' '.join(args[:3])}")


say(f"torch hold-out chain started (pid {os.getpid()})")
extra = []
for v in ("kw150", "kw700", "kw1000"):
    tag = f"tmlp_F95_phys_LO{v}_split_list_s0"
    if not os.path.exists(os.path.join(ROOT, "ML", "models", f"{tag}_meta.json")):
        run(["ML/code/torch_train.py", "--tag", tag, "--fset", "F", "--physics", PHYS,
             "--hold-out", v, "--arch", "split", "--lambda-list", "1", "--tau", "0.25",
             "--seed", "0", "--threads", "4"], f"train_{tag}.log")
    extra += [f"{tag}@{v}", f"{tag}@base"]
run(["ML/code/phys_eval.py", "--extra", ",".join(extra),
     "--models", ",".join(sorted({e.split("@")[0] for e in extra})), "--jobs", "6"],
    "phys_eval_torch_lo.log")
say("torch hold-out chain done")
