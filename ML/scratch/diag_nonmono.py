"""Stop-by-stop comparison of two CORR2 runs (low vs high power) of one instance."""
import glob, json, sys
import src.paths as P
stem, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3]
def load(tag):
    name = f"{stem}__{tag}" if tag != "None" else stem
    f = sorted(P.glob_solutions(f"{name}_GREEDY_CORR2_*.json"))[-1]
    return json.load(open(f))
a, b = load(lo), load(hi)
print(stem, lo, a["duration_h"], "|", hi, b["duration_h"])
shown = 0
for i, (xa, xb) in enumerate(zip(a["actions"], b["actions"])):
    da, db = a["durations_list"][i], b["durations_list"][i]
    ta, tb = a["sim_trajectory"][i], b["sim_trajectory"][i]
    act_a = (xa.get("y"), xa.get("break_type"), xa.get("rest_type"))
    act_b = (xb.get("y"), xb.get("break_type"), xb.get("rest_type"))
    if act_a != act_b or shown < 0:
        print(f"stop {i:3d} t {ta['t_arr']:.2f}/{tb['t_arr']:.2f} soc {ta['e_arr']:.0f}/{tb['e_arr']:.0f}"
              f"  {lo}:{act_a} tauc={da['tauc']*60:.0f}m  {hi}:{act_b} tauc={db['tauc']*60:.0f}m")
        shown += 1
        if shown > 12: break
