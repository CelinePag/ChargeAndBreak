import json, os, sys, numpy as np
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'test'
R = r"c:\Users\celinep\Documents\GitHub\ChargeAndBreak\ML\results"
def load(tag):
    p = os.path.join(R, f"eval_{tag}_g95_{SPLIT}.json")
    return {r["instance"]: r for r in json.load(open(p))} if os.path.exists(p) else None

groups = {
    "torch RowNet F95": ["tmlp_F95_base_s0", "tmlp_F95_sel_s0", "tmlp_F95_list_s0",
                         "tmlp_F95_list01_s0", "tmlp_F95_list3_s0"],
    "trees F95": [f"gbt_F95_base_s{i}" for i in range(3)],
    "sk MLP F95": [f"mlp_F95_base_s{i}" for i in range(3)],
    "sk MLP D77": [f"mlp_D77_base_s{i}" for i in range(3)],
}
ref = load("gbt_F95_base_s0")
print(f"{'model':22s} {'seed':18s} {'done':>5s} {'viol':>4s} {'mean%LA':>8s} {'se':>5s} {'med%LA':>7s} {'>5%':>4s} {'rests+':>6s} {'TWmiss':>6s}  {'vs trees s0 mean':>16s}")
for g, tags in groups.items():
    for t in tags:
        d = load(t)
        if d is None:
            print(g, t, "missing"); continue
        done = [r for r in d.values() if r["route_completed"]]
        viol = sum(r["n_violations"] for r in d.values())
        both = [r for r in done if r["LA_completed"] and r["LA"]]
        pct = np.array([100 * (r["duration_h"] / r["LA"] - 1) for r in both])
        rests = np.mean([r["n_rests"] - r["LA_rests"] for r in both])
        tw = sum(r["tw_misses"] for r in d.values())
        common = [k for k in d if k in ref and d[k]["route_completed"] and ref[k]["route_completed"]]
        dv = np.array([100 * (d[k]["duration_h"] / ref[k]["duration_h"] - 1) for k in common])
        print(f"{g:22s} {t:18s} {len(done):5d} {viol:4d} {pct.mean():+8.2f} {pct.std(ddof=1)/np.sqrt(len(pct)):5.2f} {np.median(pct):+7.2f} {(pct>5).sum():4d} {rests:+6.3f} {tw:6d}  {dv.mean():+8.2f} +- {dv.std(ddof=1)/np.sqrt(len(dv)):.2f}")

# where does torch lose? families
d = load("tmlp_F95_base_s0")
print("\nworst routes torch vs trees s0 (pct, rests torch/trees/LA)")
rows = []
for k in d:
    if k in ref and d[k]["route_completed"] and ref[k]["route_completed"]:
        rows.append((100 * (d[k]["duration_h"] / ref[k]["duration_h"] - 1), k, d[k]["n_rests"], ref[k]["n_rests"], d[k]["LA_rests"], d[k]["duration_h"], ref[k]["duration_h"], d[k]["LA"]))
for r in sorted(rows, reverse=True)[:10]:
    print(f"  {r[0]:+7.2f}  {r[1]:24s} rests {r[2]}/{r[3]}/{r[4]}  dur {r[5]:.1f}/{r[6]:.1f}/{r[7]:.1f} h")
print("incomplete torch:", [k for k, r in d.items() if not r["route_completed"]], [(r["halt_reason"]) for r in d.values() if not r["route_completed"]])

