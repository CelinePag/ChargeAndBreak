"""
extract_dataset.py — teacher trajectories (JSON) -> supervised learning table
=============================================================================
WHAT THIS DOES (easy mode)
--------------------------
Every teacher run in ML/data/miptail/ is a diary of one truck trip: at each
stop it wrote down the situation (battery, driver clocks, time) and what the
expensive MIP-tail lookahead decided to do there (charge? break? rest? how
long to charge?).  This script turns those diaries into one big table:

    one ROW  = one decision moment (a stop on some route)
    FEATURES = numbers describing the situation at that moment
    LABEL    = what the teacher chose there

The network will later be trained to reproduce LABEL from FEATURES.

Design decisions (and why):
  * We read the state DIRECTLY from `sim_trajectory` in the solution JSON —
    nothing is re-simulated, so extraction is exact and takes seconds.
  * Features that describe the future (next stops, energy to next charger,
    time-window slack) come from the INSTANCE file, joined by name.  They are
    always encoded RELATIVE to "now" (e.g. "window closes in 1.4 h", never
    "at 14:30") so the same feature means the same thing on every route.
  * Stop 0 is skipped (departure is a forced no-op, nothing to learn) and the
    final node has no action.  Runs flagged infeasible are skipped entirely.
  * The discrete label is the OBSERVED combo (y, break, rest).  We enumerate
    the combos that actually occur in the data rather than guessing the
    action space — the mapping is saved in meta so it is stable.
  * Output is .npz + meta.json (no pandas/pyarrow dependency).

Splits (written to splits.json, used verbatim by train.py):
  * IN-DISTRIBUTION pool = {Rshort,Rmedium} x {Tnone,Ttight} x all C.
      train = seeds 1..17, val = 18..21, test-ID = 22..25   (split BY SEED so
      no instance leaks between train and test — states within one route are
      highly correlated; splitting by row would inflate test scores).
  * OOD-route = Rlong x {Tnone,Ttight}          (never trained on: length)
  * OOD-tw    = {Rshort,Rmedium} x {Tmedium,Tlarge}  (never trained on: TWs)
  * OOD-both  = Rlong x {Tmedium,Tlarge}
Run:  python ML/code/extract_dataset.py       (from repo root)
"""
from __future__ import annotations
import csv, glob, json, os, re, sys
import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "ML", "data")
RUN_DIR  = os.path.join(DATA_DIR, "miptail")
INST_DIR = os.path.join(ROOT, "instances")
K_LOOK   = 10          # how many upcoming nodes get detailed per-node features
SLACK_CAP = 48.0       # clip "hours until latest arrival" at 48 h (beyond that
                       # the exact value carries no decision information)

# ── label helpers ────────────────────────────────────────────────────────────
def _norm(v) -> str:
    """The JSONs store 'no break' variously as None, 0, '0' or 'none'
    (artifacts of json round-trips) — collapse them to one token."""
    return "none" if v in (None, 0, "0", "none", "") else str(v)

def action_key(a: dict) -> tuple:
    """Discrete decision at a stop, as a hashable (y, break, rest) triple."""
    return (int(a.get("y", 0) or 0),
            _norm(a.get("break_type")), _norm(a.get("rest_type")))

# ── feature construction ─────────────────────────────────────────────────────
def build_instance_context(inst: dict):
    """Precompute per-instance arrays used by every row of that route.

    Everything keyed by node index 0..N.  JSON stores dict keys as strings,
    so we convert once here.
    """
    N     = int(inst["N"])
    D     = np.array([float(inst["D"][str(i)])  for i in range(N)])   # leg h
    E     = np.array([float(inst["E"][str(i)])  for i in range(N)])   # leg kWh
    km    = np.array([float(inst["km"][str(i)]) for i in range(N)])
    is_cs = np.zeros(N + 1); is_cs[[int(k) for k in inst["K"]]] = 1
    is_cust = np.zeros(N + 1); is_cust[[int(c) for c in inst["C"]]] = 1
    is_lay = np.zeros(N + 1); is_lay[[int(l) for l in inst["L"]]] = 1
    Q     = np.zeros(N + 1)
    for k, v in inst["Q"].items(): Q[int(k)] = float(v)
    ub_t  = np.array([float(inst["ub_t"][str(i)]) for i in range(N + 1)])

    # cumulative energy from node i to the NEXT charger at node j>i —
    # the single most safety-critical quantity (can we still reach a plug?)
    e_next_cs = np.zeros(N + 1); d_next_cs = np.zeros(N + 1)
    acc_e = acc_d = 0.0
    # walk backwards; reset accumulator when we stand on a charger
    for i in range(N, -1, -1):
        if is_cs[i] and i != N + 1:
            acc_e = acc_d = 0.0
        e_next_cs[i], d_next_cs[i] = acc_e, acc_d
        if i > 0:
            acc_e += E[i - 1]; acc_d += km[i - 1]
    # suffix sums for "what remains" aggregates
    suf_D  = np.concatenate([np.cumsum(D[::-1])[::-1], [0.0]])
    suf_E  = np.concatenate([np.cumsum(E[::-1])[::-1], [0.0]])
    suf_km = np.concatenate([np.cumsum(km[::-1])[::-1], [0.0]])
    return dict(N=N, D=D, E=E, km=km, is_cs=is_cs, is_cust=is_cust,
                is_lay=is_lay, Q=Q, ub_t=ub_t, e_next_cs=e_next_cs,
                d_next_cs=d_next_cs, suf_D=suf_D, suf_E=suf_E, suf_km=suf_km,
                Ecap=float(inst["Ecap"]), Emin=float(inst["Emin"]),
                allow_split=float(bool(inst.get("allow_split", True))))

def features_at(ctx: dict, st: dict) -> list:
    """One feature vector: current vehicle state + relative view of the future."""
    i, N = int(st["stop"]), ctx["N"]
    t    = float(st["t_arr"])
    f = [
        # -- vehicle state (the driver's dashboard) --
        float(st["e_arr"]) / ctx["Ecap"],            # state of charge, 0..1
        (float(st["e_arr"]) - ctx["Emin"]) / ctx["Ecap"],  # usable margin
        float(st["cd"]), float(st["sd"]), float(st["sw"]),  # HoS clocks (h)
        float(st["phi"]),                             # split-break state
        float(st["rho2_used"]),                       # reduced rests spent
        float(st["ext_shift_used"]),                  # 10h-driving days spent
        # -- where we stand --
        ctx["is_cs"][i], ctx["is_cust"][i], ctx["is_lay"][i], ctx["Q"][i],
        # -- reachability (safety-critical) --
        ctx["e_next_cs"][i] / ctx["Ecap"],            # energy to next charger
        ctx["d_next_cs"][i] / 100.0,                  # km to next charger
        # -- what remains of the route --
        (N - i) / 100.0,                              # stops left
        ctx["suf_D"][i] / 10.0,                       # nominal drive-h left
        ctx["suf_E"][i] / ctx["Ecap"],                # energy left / capacity
        ctx["suf_km"][i] / 100.0,
        float(ctx["is_cust"][i:].sum()),              # customers still ahead
        ctx["allow_split"],
    ]
    # -- tightest deadline ahead: min over remaining nodes of (ub_t - t) --
    slack = ctx["ub_t"][i:] - t
    f.append(min(float(slack.min()), SLACK_CAP) if len(slack) else SLACK_CAP)
    # -- detailed look at the next K nodes (zero-padded near route end);
    #    the network learns how far ahead actually matters (K is an ablation) --
    for j in range(1, K_LOOK + 1):
        n = i + j
        if n <= N:
            f += [ctx["D"][n - 1], ctx["E"][n - 1] / 100.0,
                  ctx["is_cs"][n], ctx["is_cust"][n], ctx["Q"][n],
                  min(float(ctx["ub_t"][n] - t), SLACK_CAP)]
        else:
            f += [0.0, 0.0, 0.0, 0.0, 0.0, SLACK_CAP]
    return f

FEATURE_NAMES = ([
    "soc", "soc_margin", "cd", "sd", "sw", "phi", "rho2_used", "ext_used",
    "at_cs", "at_cust", "at_lay", "queue_here", "e_to_next_cs", "km_to_next_cs",
    "stops_left", "driveh_left", "energy_left", "km_left", "cust_left",
    "allow_split", "min_slack"]
    + [f"n{j}_{x}" for j in range(1, K_LOOK + 1)
       for x in ("legD", "legE", "cs", "cust", "queue", "slack")])

# ── split assignment ─────────────────────────────────────────────────────────
def split_of(family: str, seed: int) -> str:
    m = re.match(r"(R[a-z]+)C([a-z]+)T([a-z]+)", family)
    r, tw = m.group(1), m.group(3)
    if r == "Rlong" and tw in ("none", "tight"):   return "ood_route"
    if r != "Rlong" and tw in ("medium", "large"): return "ood_tw"
    if r == "Rlong":                               return "ood_both"
    return "train" if seed <= 17 else ("val" if seed <= 21 else "test_id")

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    infeasible = set()
    with open(os.path.join(DATA_DIR, "manifest.csv")) as fh:
        for row in csv.DictReader(fh):
            if row["run_infeasible"] == "True":
                infeasible.add(row["instance"])

    X, y_cls, y_tauc, rows_meta = [], [], [], []
    combo_count: dict = {}
    files = sorted(glob.glob(os.path.join(RUN_DIR, "*.json")))
    for f in files:
        run = json.load(open(f))
        inst_name = run["instance"]
        if inst_name in infeasible:
            continue
        inst = json.load(open(os.path.join(INST_DIR, inst_name + ".json")))["instance"]
        ctx  = build_instance_context(inst)
        family, seed = inst_name.rsplit("_", 1)
        traj, acts, durs = run["sim_trajectory"], run["actions"], run["durations_list"]
        # stop 0 = forced no-op at departure -> not a decision, skip
        for i in range(1, len(acts)):
            st = traj[i]
            if int(st["stop"]) != i:      # trust but verify alignment
                raise RuntimeError(f"stop index mismatch in {f} @ {i}")
            key = action_key(acts[i])
            combo_count[key] = combo_count.get(key, 0) + 1
            X.append(features_at(ctx, st))
            y_cls.append(key)
            y_tauc.append(float(durs[i].get("tauc", 0.0)))
            rows_meta.append((inst_name, family, int(seed), i,
                              split_of(family, int(seed))))

    # stable class mapping: most frequent combo first (class 0 = usually "pass")
    combos = sorted(combo_count, key=lambda k: -combo_count[k])
    cls_of = {c: j for j, c in enumerate(combos)}
    y = np.array([cls_of[c] for c in y_cls], dtype=np.int64)

    X = np.asarray(X, dtype=np.float32)
    y_tauc = np.asarray(y_tauc, dtype=np.float32)
    splits = np.array([r[4] for r in rows_meta])
    np.savez_compressed(
        os.path.join(DATA_DIR, "dataset.npz"),
        X=X, y=y, tauc=y_tauc, split=splits,
        instance=np.array([r[0] for r in rows_meta]),
        family=np.array([r[1] for r in rows_meta]),
        stop=np.array([r[3] for r in rows_meta], dtype=np.int32))
    meta = dict(
        n_rows=int(len(y)), n_features=X.shape[1],
        feature_names=FEATURE_NAMES,
        classes=[list(c) for c in combos],
        class_counts={str(c): combo_count[c] for c in combos},
        split_sizes={s: int((splits == s).sum()) for s in np.unique(splits)},
        k_lookahead=K_LOOK, slack_cap_h=SLACK_CAP,
        n_runs_used=len(files) - len(infeasible))
    json.dump(meta, open(os.path.join(DATA_DIR, "meta.json"), "w"), indent=2)

    print(f"rows: {len(y)}   features: {X.shape[1]}   classes: {len(combos)}")
    for c in combos:
        print(f"  {str(c):28s} {combo_count[c]:7d}  ({100*combo_count[c]/len(y):.1f}%)")
    print("splits:", meta["split_sizes"])
    print("tauc>0 rows:", int((y_tauc > 0).sum()),
          " tauc range h: %.2f..%.2f" % (y_tauc[y_tauc > 0].min() if (y_tauc > 0).any() else 0,
                                          y_tauc.max()))

if __name__ == "__main__":
    sys.exit(main())
