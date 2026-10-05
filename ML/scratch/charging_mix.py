"""Where does each policy charge on the mixed test routes? (2026-10-05)
Share of charged energy at fast (>=700 kW) and slow (150 kW) chargers, routes
the LA has run, for the LA, the oracle and a few students (seeds pooled)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import mixed_eval as me
KEEP = ("tmlp_T144_phys_split_list_s0", "tmlp_T144_physpmix121_split_list",
        "tmlp_T144_physdg2_split_list", "tmlp_T144_physmix_split_list")
me.MODELS = [m for m in me.MODELS if m[1].startswith(KEEP)]
GROUP = {"torch split+list, T inputs, all phys": "no mixed data",
         "torch T, + 121 mixed routes": "pool + 121 pmix",
         "torch T, + DAgger x2": "pool + DAgger x2",
         "torch T, + pmix + mix routes": "pool + pmix + mix"}
for v in ("pmix", "mix"):
    E_by, _n, n_routes = me.energy_by_power(v, la_only=True)
    agg = {}
    for lab, c in E_by.items():
        g = GROUP.get(lab.split(" (s")[0], lab)
        a = agg.setdefault(g, {})
        for k, e in c.items():
            a[k] = a.get(k, 0.0) + e
    print(f"\n{v}: {n_routes} test routes the LA has run")
    print(f"{'policy':22s} {'>=700 kW':>9s} {'150 kW':>8s}")
    for g, a in agg.items():
        tot = sum(a.values()) or 1.0
        fast = sum(e for k, e in a.items() if k >= 700) / tot
        slow = sum(e for k, e in a.items() if k <= 150) / tot
        print(f"{g:22s} {100*fast:8.0f}% {100*slow:7.0f}%")
