"""No shield, stricter feasibility threshold (0.9, 0.99): can the model be
made safe by itself?  Follow-up to queue_noshield.py, same layout."""
import os, subprocess, time
ROOT = r"C:\Users\celinep\Documents\GitHub\ChargeAndBreak"
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
jobs = []
for thr in ("0.9", "0.99"):
    for kind, pat in (("torch", "tmlp_F95_split_list_s{}"), ("gbt", "gbt_F95_base_s{}")):
        for s in range(3):
            tag = pat.format(s); v = f"nsN_f{thr[2:]}"
            jobs.append([PY, "-u", "ML/code/evaluate.py", "--kind", kind, "--tag", tag,
                         "--split", "test", "--guard-q", "0.99", "--no-shield",
                         "--nominal-features", "--feas-thr", thr,
                         "--out", f"eval_{tag}_{v}_test.json"])
running = []
while jobs or running:
    while jobs and len(running) < 4:
        a = jobs.pop(0)
        fh = open(os.path.join(ROOT, "ML", "logs", a[-1].replace(".json", ".log")), "w")
        running.append((subprocess.Popen(a, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=ENV), fh))
    for p, fh in list(running):
        if p.poll() is not None:
            fh.close(); running.remove((p, fh)); print("rc", p.returncode, flush=True)
    time.sleep(2)
