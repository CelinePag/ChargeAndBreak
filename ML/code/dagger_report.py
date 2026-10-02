"""
dagger_report.py — what a DAgger label set says, before and after retraining
===========================================================================
Three readings of ML/data/dagger/<label>/labels/:

  ON-POLICY REGRET  at each queried stop the label holds the teacher's cost
     for every action, and the record holds what the student did there, so
     regret(student's action) is the student's error AT ITS OWN STATES -- the
     quantity behaviour cloning never measures.  An action the teacher found
     infeasible in some scenario has no regret; it is counted as `dirty`.

  --reference  the same model's offline regret on the TEACHER's states of the
     same split (argmin over the teacher-scored actions, feasibility head
     applied, as dataset.print_offline).  On-policy >> reference means errors
     compound and DAgger has something to fix; on-policy ~ reference means it
     does not.  Use a probe on the stop split for this: on fit routes the
     teacher's states were trained on, which flatters the reference.  The two
     are not identical rules (online the shield filters actions first), so
     read a ratio, not a small difference.

  --check  stops kept from BEFORE the student left the teacher's trajectory
     (dagger_rollout --check-stops): there the state IS the teacher's, so the
     fresh label must agree with the teacher's logged decision up to the LA's
     own scenario noise.  This is the test that the queried LA is the teacher.

    python ML/code/dagger_report.py --label probe --reference --check
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dagger_io import label_dir, teacher_run                      # noqa: E402
from parse_logs import parse_log                                  # noqa: E402


def read_label(label):
    """One dict per queried decision: model, route, stop, check flag, the
    student's action and its regret / cleanliness, the teacher's chosen action,
    and per-action (cost, std, clean)."""
    out = []
    for f in sorted(glob.glob(os.path.join(label_dir(label, "labels"), "*.npz"))):
        with np.load(f, allow_pickle=True) as z:
            vocab = [str(x) for x in z["action_vocab"]]
            keys = np.array(vocab)[z["action_ix"]]
            for s in np.unique(z["stop"]):
                m = z["stop"] == s
                stud = str(z["student_key"][m][0])
                per = {str(k): (float(c), float(sd), bool(cl), float(r))
                       for k, c, sd, cl, r in zip(keys[m], z["cost"][m], z["std"][m],
                                                  z["clean"][m], z["regret"][m])}
                hit = per.get(stud)
                out.append(dict(
                    model=str(z["model_tag"]), route=str(z["route"]),
                    physics=str(z["physics"]), stop=int(s),
                    check=bool(z["is_check"][m][0]), student=stud,
                    clean=bool(hit and hit[2]),
                    regret=hit[3] if hit and hit[2] else np.nan,
                    teacher=str(keys[m][z["chosen"][m] == 1][0])
                    if (z["chosen"][m] == 1).any() else None,
                    actions=per, length=str(z["family"])[1:].split("C")[0]))
    return out


def on_policy(decs):
    print(f"\nON-POLICY REGRET (student's executed action, teacher's costs)")
    print(f"{'model':28s} {'class':7s} {'n':>5s} {'regret min':>11s} {'wrong':>6s} "
          f"{'dirty':>6s} {'rest?':>6s}")
    for model in sorted({d["model"] for d in decs}):
        for cls in ("all", "short", "medium", "long"):
            sel = [d for d in decs if d["model"] == model and not d["check"]
                   and (cls == "all" or d["length"] == cls)]
            if not sel:
                continue
            r = np.array([d["regret"] for d in sel])
            ok = np.isfinite(r)
            wrong = np.mean([d["student"] != d["teacher"] for d in sel])
            rests = sum(1 for d in sel if d["student"].endswith(("_r1", "_r2"))
                        and not str(d["teacher"]).endswith(("_r1", "_r2")))
            print(f"{model:28s} {cls:7s} {len(sel):5d} {np.nanmean(r[ok]) * 60:9.2f}   "
                  f"{wrong * 100:5.1f}% {int((~ok).sum()):6d} {rests:6d}")


def reference(label, decs):
    """The model's offline regret on the teacher's states of the probe's split."""
    from dataset import argmin_policy_regret, load, split_masks
    from policy_core import load_policy
    q = sorted(glob.glob(os.path.join(label_dir(label, "queries"), "*.json")))
    kinds = {}
    for f in q:
        with open(f, encoding="utf-8") as fh:
            m = json.load(fh)["model"]
        kinds[m["tag"]] = m["kind"]
    with open(q[0], encoding="utf-8") as fh:
        split = json.load(fh)["split"]
    # the classifier scores a state, not (state, action) rows: no row reference
    kinds = {t: k for t, k in kinds.items() if k != "clf"}
    d = load()
    fit, stop, _ = split_masks(d)
    mask = {"fit": fit, "stop": stop}[split]
    names = [str(x) for x in d["feature_names"]]
    print(f"\nREFERENCE: offline regret on the teacher's {split}-split states")
    for tag, kind in sorted(kinds.items()):
        pol = load_policy(kind, tag)
        cols = [names.index(n) for n in pol.state_names + pol.action_names]
        ix = np.flatnonzero(mask)
        X = d["X"][np.ix_(ix, cols)]
        pc, pf = np.zeros(len(mask)), np.ones(len(mask))
        for s in range(0, len(ix), 65536):
            c, f = pol._predict(X[s:s + 65536].astype(np.float32 if kind == "torch"
                                                      else np.float64))
            pc[ix[s:s + 65536]], pf[ix[s:s + 65536]] = c, f
        r, n = argmin_policy_regret(d, mask, pc, pf)
        own = [x["regret"] for x in decs if x["model"] == tag and not x["check"]]
        print(f"  {tag:28s} teacher states {r * 60:6.2f} min ({n} decisions)   "
              f"own states {np.nanmean(own) * 60:6.2f} min ({len(own)})")


def check(decs):
    print("\nTEACHER CHECK (stops still on the teacher's trajectory)")
    sel = [d for d in decs if d["check"]]
    if not sel:
        print("  none (dagger_rollout --check-stops 0)")
        return
    same, z = 0, []
    cache = {}
    for d in sel:
        rs = "pmix" if d["physics"] == "pmix" else "base"
        key = (rs, d["route"])
        if key not in cache:
            run = teacher_run(rs, d["route"])
            cache[key] = {x.stop: x for x in parse_log(run[1])[0]} if run and run[1] else {}
        t = cache[key].get(d["stop"])
        if t is None:
            continue
        same += int(t.chosen == d["teacher"])
        for a in t.actions:
            b = d["actions"].get(a.key)
            if b and a.clean and b[2]:
                z.append(abs(a.cost_h - b[0]) / max(np.hypot(a.std_h, b[1]) / 5.0, 1e-6))
    n_cmp = sum(1 for d in sel if cache.get(("pmix" if d["physics"] == "pmix" else "base",
                                              d["route"]), {}).get(d["stop"]))
    if not z:
        print(f"  {len(sel)} checks, {n_cmp} with a logged teacher decision -- nothing to compare")
        return
    print(f"  {len(sel)} checks ({n_cmp} with a logged teacher decision): same chosen "
          f"action {same}/{n_cmp}; |cost difference| / scenario SEM: median "
          f"{np.median(z):.2f}, 95th pct {np.percentile(z, 95):.2f} over {len(z)} "
          f"actions (|N(0,1)|: median 0.67, 95th pct 1.96, if the queried LA is the teacher)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    decs = read_label(args.label)
    print(f"[report] {args.label}: {len(decs)} labelled decisions "
          f"({sum(d['check'] for d in decs)} teacher checks)")
    if not decs:
        return
    on_policy(decs)
    if args.reference:
        reference(args.label, decs)
    if args.check:
        check(decs)


if __name__ == "__main__":
    main()
