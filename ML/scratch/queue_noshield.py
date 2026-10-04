"""No-shield test (2026-10-04, user: "let the model decide, as the LA does").
Headline torch + trees, seeds 0-2, test split:
  nsN   no rule layer, features nominal (as trained); --guard-q 0.99 only
        counts the picks the guard would have blocked
  g99srN  full shield, but features nominal instead of at the guard quantile
Runs 4 at a time.  Log: ML/logs/queue_noshield.log"""
import datetime
import os
import subprocess
import time

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_noshield.log")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

MODELS = [("torch", f"tmlp_F95_split_list_s{s}") for s in range(3)] + \
         [("gbt", f"gbt_F95_base_s{s}") for s in range(3)]
VARIANTS = {"nsN": ["--guard-q", "0.99", "--no-shield", "--nominal-features"],
            "g99srN": ["--guard-q", "0.99", "--spread-room", "--nominal-features"]}


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")


jobs = []
for v, extra in VARIANTS.items():
    for kind, tag in MODELS:
        out = f"eval_{tag}_{v}_test.json"
        args = [PY, "-u", "ML/code/evaluate.py", "--kind", kind, "--tag", tag,
                "--split", "test", "--out", out] + extra
        jobs.append((args, os.path.join(ROOT, "ML", "logs", out.replace(".json", ".log"))))

say(f"no-shield chain started (pid {os.getpid()}), {len(jobs)} jobs")
running = []
while jobs or running:
    while jobs and len(running) < 4:
        args, log = jobs.pop(0)
        fh = open(log, "w", encoding="utf-8")
        running.append((subprocess.Popen(args, cwd=ROOT, stdout=fh,
                                         stderr=subprocess.STDOUT, env=ENV), fh, args))
    for p, fh, args in list(running):
        if p.poll() is not None:
            fh.close()
            running.remove((p, fh, args))
            say(f"rc={p.returncode}  {' '.join(args[3:])}")
    time.sleep(2)
say("no-shield chain done")
