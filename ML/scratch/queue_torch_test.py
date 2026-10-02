"""Detached chain: the model chosen on the stop split (tmlp_F95_split_list,
2026-10-01) gets seeds 1-2, then all three seeds are evaluated ONCE on the
test split, headline shield (g99sr) and the g95 setting.
Log: ML/logs/queue_torch.log"""
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


say(f"torch test chain started (pid {os.getpid()})")
for s in (1, 2):
    tag = f"tmlp_F95_split_list_s{s}"
    run(["ML/code/torch_train.py", "--tag", tag, "--seed", str(s), "--threads", "2",
         "--lambda-list", "1", "--tau", "0.25"], f"train_{tag}.log")
for s in (0, 1, 2):
    tag = f"tmlp_F95_split_list_s{s}"
    run(["ML/code/evaluate.py", "--kind", "torch", "--tag", tag, "--split", "test",
         "--guard-q", "0.99", "--spread-room", "--out", f"eval_{tag}_g99sr_test.json"],
        f"eval_{tag}_g99sr_test.log")
    run(["ML/code/evaluate.py", "--kind", "torch", "--tag", tag, "--split", "test",
         "--guard-q", "0.95", "--out", f"eval_{tag}_g95_test.json"], f"eval_{tag}_g95_test.log")
say("torch test chain done")
