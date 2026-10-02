"""Detached chain: train the two split-arch torch variants, then evaluate them
(and references) closed-loop on the stop split.  Log: ML/logs/queue_torch.log"""
import datetime
import os
import subprocess

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_torch.log")


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def run(args, log):
    with open(os.path.join(ROOT, "ML", "logs", log), "w", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-u"] + args, cwd=ROOT, stdout=fh,
                             stderr=subprocess.STDOUT)
    say(f"rc={rc}  {' '.join(args)}")


say(f"torch chain started (pid {os.getpid()})")
for tag, extra in (("tmlp_F95_split_s0", ["--lambda-list", "0"]),
                   ("tmlp_F95_split_list_s0", ["--lambda-list", "1", "--tau", "0.25"])):
    run(["ML/code/torch_train.py", "--tag", tag, "--seed", "0", "--threads", "2"] + extra,
        f"train_{tag}.log")
for tag in ("tmlp_F95_split_s0", "tmlp_F95_split_list_s0"):
    run(["ML/code/evaluate.py", "--kind", "torch", "--tag", tag, "--split", "stop",
         "--guard-q", "0.95", "--out", f"eval_{tag}_g95_stop.json"], f"eval_{tag}_stop.log")
for kind, tag in (("gbt", "gbt_F95_base_s0"), ("torch", "tmlp_F95_list_s0"),
                  ("torch", "tmlp_F95_split_s0"), ("torch", "tmlp_F95_split_list_s0")):
    run(["ML/code/evaluate.py", "--kind", kind, "--tag", tag, "--split", "stop",
         "--guard-q", "0.99", "--spread-room", "--out", f"eval_{tag}_g99sr_stop.json"],
        f"eval_{tag}_g99sr_stop.log")
say("torch chain done")
