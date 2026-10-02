"""Detached (2026-10-02): re-extract every dataset with the charger tokens of
features.py section J.  Waits for the B0 chain (memory), backs the current
files up as *_v222.npz, extracts, then checks EVERY old column of EVERY row is
bit-identical; on a mismatch the backups are restored.  Log:
ML/logs/reextract_tokens.log"""
import datetime
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
DATA = os.path.join(ROOT, "ML", "data")
LOG = os.path.join(ROOT, "ML", "logs", "reextract_tokens.log")
TAGS = ["base", "kwh300", "kwh700", "kwh900", "kw150", "kw700", "kw1000", "cs30", "cs100"]
WAIT_PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def say(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {msg}\n")


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def fname(tag):
    return "dataset.npz" if tag == "base" else f"dataset_phys_{tag}.npz"


def same(tag):
    new = np.load(os.path.join(DATA, fname(tag)), allow_pickle=True)
    old = np.load(os.path.join(DATA, fname(tag).replace(".npz", "_v222.npz")), allow_pickle=True)
    nn, no = [str(x) for x in new["feature_names"]], [str(x) for x in old["feature_names"]]
    if len(new["stop"]) != len(old["stop"]):
        return f"row count {len(new['stop'])} vs {len(old['stop'])}"
    Xn = new["X"]
    Xo = old["X"]
    bad = [c for c in no if not np.array_equal(Xn[:, nn.index(c)], Xo[:, no.index(c)], equal_nan=True)]
    del Xn, Xo
    for k in ("action_ix", "regret", "cost", "clean", "chosen", "stop", "instance_ix", "seed"):
        if not np.array_equal(new[k], old[k], equal_nan=True):
            bad.append(k)
    return "ok" if not bad else f"MISMATCH {bad[:6]}"


if WAIT_PID:
    say(f"waiting for pid {WAIT_PID}")
    while alive(WAIT_PID):
        time.sleep(60)
say("backing up and re-extracting")
for t in TAGS:
    src = os.path.join(DATA, fname(t))
    bak = src.replace(".npz", "_v222.npz")
    if not os.path.exists(bak):
        shutil.copy2(src, bak)
with open(os.path.join(ROOT, "ML", "logs", "reextract_tokens_extract.log"), "w") as fh:
    rc = subprocess.call([PY, "-u", "ML/code/extract.py", "--jobs", "2", "--physics", ",".join(TAGS)],
                         cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
say(f"extract rc={rc}")
ok = rc == 0
for t in TAGS:
    try:
        r = same(t)
    except Exception as e:                       # noqa: BLE001
        r = f"error {e}"
    say(f"{t}: {r}")
    ok &= r == "ok"
if not ok:
    for t in TAGS:
        bak = os.path.join(DATA, fname(t).replace(".npz", "_v222.npz"))
        shutil.copy2(bak, os.path.join(DATA, fname(t)))
    say("RESTORED the v222 files: something differed")
else:
    say("all datasets re-extracted with tokens; old columns identical; backups kept as *_v222.npz")
