"""Step 1 of the solver-call gate (2026-10-20, user: "ok start the plan").
Out-of-fold students: the fit seeds 1-19 are split into 5 folds by route seed;
each fold model is trained on the other four folds with the headline recipe,
so its picks on its own fold are honest, held-out picks.  Per fold:
  gbt_F95_base_f<k>_s0             trees, LA labels   (gbt_F95_base_s* recipe)
  gbt_F95_oracle_f<k>_s0           trees, oracle labels (the "teachers disagree" signal)
  tmlp_F95_split_list_f<k>_s0      PyTorch, LA labels (tmlp_F95_split_list_s* recipe)
Early stopping stays on seeds 20-21; test seeds 22-25 are never touched.
Two PyTorch jobs run side by side (4 threads each); trees run after them.
Log: ML/logs/queue_gate_folds.log"""
import datetime
import os
import subprocess

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "ML", "logs", "queue_gate_folds.log")
ORACLE = os.path.join(ROOT, "ML", "data", "dataset_oracle.npz")
FOLDS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 19)]


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def have(tag):
    return os.path.exists(os.path.join(ROOT, "ML", "models", f"{tag}_meta.json"))


def fit_seeds(k):
    return ",".join(f"{a}-{b}" for j, (a, b) in enumerate(FOLDS) if j != k)


def start(args, log):
    fh = open(os.path.join(ROOT, "ML", "logs", log), "w", encoding="utf-8")
    return subprocess.Popen([PY, "-u"] + args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT), fh


say(f"gate folds started (pid {os.getpid()})")
torch_jobs = []
for k in range(len(FOLDS)):
    tag = f"tmlp_F95_split_list_f{k}_s0"
    if not have(tag):
        torch_jobs.append((["ML/code/torch_train.py", "--tag", tag, "--fset", "F", "--arch", "split",
                            "--lambda-list", "1", "--tau", "0.25", "--seed", "0", "--threads", "4",
                            "--fit-seeds", fit_seeds(k)], f"train_{tag}.log", tag))
running = []
while torch_jobs or running:
    while torch_jobs and len(running) < 2:
        args, log, tag = torch_jobs.pop(0)
        p, fh = start(args, log)
        running.append((p, fh, tag))
    p, fh, tag = running[0]
    rc = p.wait()
    fh.close()
    running.pop(0)
    say(f"rc={rc}  {tag}")
for k in range(len(FOLDS)):
    for tag, extra in ((f"gbt_F95_base_f{k}_s0", []),
                       (f"gbt_F95_oracle_f{k}_s0", ["--data", ORACLE])):
        if have(tag):
            continue
        p, fh = start(["ML/code/gbt_train.py", "--fset", "F", "--tag", tag, "--seed", "0",
                       "--fit-seeds", fit_seeds(k)] + extra, f"train_{tag}.log")
        rc = p.wait()
        fh.close()
        say(f"rc={rc}  {tag}")
say("gate folds done")
