import glob, json, os, re, collections
root = "solutions"
rows = collections.defaultdict(lambda: collections.Counter())
keys = set()
for bucket in ("sensitivity", "basecase"):
    for p in glob.glob(os.path.join(root, bucket, "*_LA*.json")):
        b = os.path.basename(p)
        m = re.match(r"(.+?)_(LA[A-Z0-9_]*?)_(\d{8})_", b)
        if not m: continue
        inst, meth = m.group(1), m.group(2)
        tag = inst.split("__")[1] if "__" in inst else "base"
        with open(p) as fh: s = json.load(fh)
        keys |= set(s.keys())
        mode = s.get("solve_mode") or s.get("config", {}).get("solve_mode")
        pq = s.get("prune_quantile", "NA")
        inf = bool(s.get("metrics", {}).get("run_infeasible"))
        hz = s.get("horizon_h", s.get("H", "NA"))
        rows[(bucket, tag, meth)][(str(mode), str(pq), "inf" if inf else "ok")] += 1
for k in sorted(rows): print(k, dict(rows[k]))
print(sorted(keys))
