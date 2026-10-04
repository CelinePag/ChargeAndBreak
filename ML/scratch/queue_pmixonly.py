"""Is the uniform pool diluting the mixed data? (2026-10-04, laptop, detached)
1. mixed routes only, 3 seeds, with and without the DAgger labels (fast)
2. the full pool with the 121 mixed routes weighted x5 and x20, seed 0
3. every model on the mixed-power VALIDATION routes (choose there)
Log: ML/logs/queue_pmixonly.log"""
import datetime
import os
import subprocess

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_pmixonly.log")
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


def fit(tag, extra, threads=4):
    if os.path.exists(os.path.join(ROOT, "ML", "models", f"{tag}_meta.json")):
        say(f"{tag} exists, skipped")
        return
    run(["ML/code/torch_train.py", "--tag", tag, "--threads", str(threads)] + TC + extra,
        f"train_{tag}.log")


say(f"pmix-only chain started (pid {os.getpid()})")
for s in (0, 1, 2):
    fit(f"tmlp_T144_pmixonly_split_list_s{s}",
        ["--physics", "pmix", "--stop-seeds", "11,12", "--seed", str(s)])
    fit(f"tmlp_T144_pmixonlydg2_split_list_s{s}",
        ["--physics", "pmix", "--dagger", "dgpmix,dgpmix2", "--stop-seeds", "11,12",
         "--seed", str(s)])
# validation first for what exists (the first run died at 13:08 while
# training the x5 model, with no traceback), then the weighted pool, then
# validation again (the drive skips rows it already has)
run(["ML/code/mixed_eval.py", "drive", "--set", "val", "--variants", "pmix", "--jobs", "6"],
    "mixed_val_drive_pmixonly.log")
for k in (5, 20):
    fit(f"tmlp_T144_physpmix121w{k}_split_list_s0",
        ["--physics", f"{PHYS},pmix", "--weight-physics", f"pmix={k}", "--seed", "0"],
        threads=4)
run(["ML/code/mixed_eval.py", "drive", "--set", "val", "--variants", "pmix", "--jobs", "6"],
    "mixed_val_drive_pmixonly_w.log")
say("pmix-only chain done")
