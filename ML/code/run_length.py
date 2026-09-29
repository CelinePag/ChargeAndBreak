"""
run_length.py — trained on short and medium routes, tested on long ones
======================================================================
The physics-shift experiment (ood_eval.py) showed the learned policies do not
transfer to a different battery or charger.  This asks the same question along
a gentler axis: the physics and the rules are unchanged, only the ROUTE LENGTH
is unseen.  Long routes are markedly harder -- median 162 stops, 101 h and 4
daily rests, against 96 stops, 56 h and 2 rests for medium -- so rest timing,
the thing that broke under the physics shifts, is exactly what gets stressed.

Models are trained with `--scope SM` (short + medium only; fit 1-19, stop
20-21) and named `<arm>_<SET><n>_base_SM_s<seed>`.  Each is scored three ways:

    in_dist      short + medium TEST routes (seeds 22-25, 95 routes): did
                 restricting the training data cost anything where it DID train?
    long_paired  the 30 long TEST routes (seeds 22-25), compared with the
                 models trained on ALL lengths on the very same routes -- the
                 cost of never having seen a long route
    long_all     every one of the 239 long routes: none was seen in training,
                 so all are fair, and 8x the routes of long_paired

Two feature sets per arm.  F is the best base-case set.  R ("route-local")
drops the seven features that measure position along the WHOLE route.  About
23% of long-route decisions put those features beyond anything seen on short
and medium routes -- e.g. `drive_left` reaches 50 h against a training maximum
of 31.5 h -- and a tree cannot extrapolate past its training range.  If those
features are what break the transfer, R should transfer better than F while
matching it in distribution.

    python ML/code/run_length.py --arms gbt,clf --fsets F,R --seeds 3

`--spread-room` re-scores the SAME models with policy_core.spread_room on
(the charge and stop overhead counted against the 15 h spread) and `--guard`
sets the drive-time quantile the checks use (0.95 by default).  Any setting
other than the default is a VARIANT, named in the files it writes --
g95sr, g99sr -- e.g. eval_<tag>_g99sr_longall.json into length_test_g99sr.json.
A variant never trains: a model that does not exist yet is skipped, so it can
run beside a training run without racing it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsets import label                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
STORE = os.path.join(RESULTS, "length_test.json")
PY = sys.executable
GUARD = "0.95"
BETA = 0.5

ARM = {
    "gbt": dict(trainer="gbt_train.py", kind="gbt", ck="{t}_cost.txt",
                flags=(), label="Trees", ref="gbt_F95_base_s{s}"),
    "mlp": dict(trainer="nn_train.py", kind="nn", ck="{t}_nn.joblib",
                flags=("--target-transform", "log1p"), label="MLP",
                ref="mlp_F95_base_s{s}"),
    "clf": dict(trainer="clf_train.py", kind="clf", ck="{t}_clf.joblib",
                flags=("--class-weight", "none"), label="Classifier",
                ref="clf_F77_base_s{s}"),
}


def length_of(inst):
    return inst[1:].split("C")[0]


def summarise_rows(rows):
    """Headline numbers for any subset of an evaluation's rows."""
    comp = [r for r in rows if r.get("route_completed")]

    def med(key):
        v = [100.0 * (r["duration_h"] - r[key]) / r[key] for r in comp
             if r.get(key) is not None and not r.get(f"{key}_infeasible")]
        return float(np.median(v)) if v else float("nan")

    return dict(med_la=med("LA"), med_greedy=med("GREEDY"),
                infeasible=len(rows) - len(comp), routes=len(rows),
                tw=sum(r.get("tw_misses", 0) for r in rows),
                tw_la=sum(r.get("LA_tw", 0) for r in rows))


def load_eval(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def run(cmd, what):
    print(f"   $ {what}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-1200:])
        print(p.stderr[-1200:])
        raise SystemExit(f"FAILED: {what}")


def evaluate(kind, tag, split, out, var=None):
    """var = (guard, spread_room); None is the default (0.95, off)."""
    guard, sr = var or (float(GUARD), False)
    if not os.path.exists(os.path.join(RESULTS, out)):
        run([PY, os.path.join(HERE, "evaluate.py"), "--kind", kind, "--tag", tag,
             "--split", split, "--guard-q", str(guard), "--out", out]
            + (["--spread-room"] if sr else []),
            f"evaluate {tag} on {split}")
    return load_eval(out)


def references(arms, seeds, var=None):
    """The models trained on ALL lengths, scored on the same test routes.

    Their existing test evaluations already contain the long routes of seeds
    22-25, so they are filtered rather than re-run.  They have no `long_all`
    figure: they trained on long seeds 1-19, so only 22-25 are unseen to them.
    """
    out = []
    for arm in arms:
        a = ARM[arm]
        for sd in range(seeds):
            tag = a["ref"].format(s=sd)
            f = f"eval_{tag}_{sfx(var)}_test.json"
            if var and not os.path.exists(os.path.join(RESULTS, f)):
                if not os.path.exists(os.path.join(MODELS, a["ck"].format(t=tag))):
                    continue
                evaluate(a["kind"], tag, "test", f, var)
            if not os.path.exists(os.path.join(RESULTS, f)):
                continue
            rows = load_eval(f)
            out.append(dict(
                arm=arm, arm_label=a["label"], scope="all", seed=sd, tag=tag,
                fset_label=tag.split("_")[1],
                in_dist=summarise_rows([r for r in rows
                                        if length_of(r["instance"]) != "long"]),
                long_paired=summarise_rows([r for r in rows
                                            if length_of(r["instance"]) == "long"]),
                long_all=None))
    return out


def sfx(var):
    """File-name fragment: g95 (the default), g95sr, g99sr, ..."""
    guard, sr = var or (float(GUARD), False)
    return f"g{round(100 * guard)}{'sr' if sr else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gbt,clf")
    ap.add_argument("--fsets", default="F,R")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--spread-room", action="store_true",
                    help="evaluate existing models with policy_core.spread_room "
                         "on; a variant, never trains")
    ap.add_argument("--guard", type=float, default=float(GUARD),
                    help="drive-time quantile of the checks; other than 0.95 "
                         "is a variant, never trains")
    args = ap.parse_args()
    var = ((args.guard, args.spread_room)
           if (args.spread_room or args.guard != float(GUARD)) else None)
    store = STORE.replace(".json", f"_{sfx(var)}.json") if var else STORE

    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    ns = int(d["n_state"])

    rows = []
    if os.path.exists(store):
        with open(store) as fh:
            rows = [r for r in json.load(fh) if r.get("scope") == "SM"]
    done = {r["tag"] for r in rows}

    for arm in args.arms.split(","):
        a = ARM[arm]
        for fs in args.fsets.split(","):
            lab = label(fs, arm, names, ns)
            for sd in range(args.seeds):
                tag = f"{arm}_{lab}_base_SM_s{sd}"
                if tag in done:
                    print(f"=== {tag}: done, skipped", flush=True)
                    continue
                print(f"\n=== {tag} ({a['label']}, set {lab}, trained on "
                      f"short+medium, training seed {sd}) ===", flush=True)
                t0 = time.time()
                if not os.path.exists(os.path.join(MODELS, a["ck"].format(t=tag))):
                    if var:
                        print("   (not trained yet -- skipped)", flush=True)
                        continue
                    run([PY, os.path.join(HERE, a["trainer"]), "--tag", tag,
                         "--fset", fs, "--scope", "SM", "--seed", str(sd)]
                        + list(a["flags"]), f"train {tag}")
                test = evaluate(a["kind"], tag, "test",
                                f"eval_{tag}_{sfx(var)}_test.json", var)
                lall = evaluate(a["kind"], tag, "long_all",
                                f"eval_{tag}_{sfx(var)}_longall.json", var)
                r = dict(
                    arm=arm, arm_label=a["label"], scope="SM", fset=fs,
                    fset_label=lab, seed=sd, tag=tag,
                    in_dist=summarise_rows([x for x in test
                                            if length_of(x["instance"]) != "long"]),
                    long_paired=summarise_rows([x for x in test
                                                if length_of(x["instance"]) == "long"]),
                    long_all=summarise_rows(lall),
                    seconds=round(time.time() - t0, 1))
                rows.append(r)
                done.add(tag)
                with open(store, "w") as fh:
                    json.dump(rows + references(ARM, 3, var), fh, indent=1)
                print(f"   -> in-dist {r['in_dist']['med_la']:+.2f}%  "
                      f"long(30) {r['long_paired']['med_la']:+.2f}%  "
                      f"long(239) {r['long_all']['med_la']:+.2f}%  "
                      f"infeasible on long(239) {r['long_all']['infeasible']}  "
                      f"({r['seconds']}s)", flush=True)

    refs = references(ARM, 3, var)
    with open(store, "w") as fh:
        json.dump(rows + refs, fh, indent=1)
    table(rows + refs, store)


def table(rows, store=STORE):
    print("\n" + "=" * 100)
    print("ROUTE-LENGTH EXTRAPOLATION — median vs teacher (%), mean ± sd over "
          "training seeds; [infeasible]")
    print("=" * 100)
    print(f"{'arm':11s} {'trained on':12s} {'set':5s} {'short+medium':>18s} "
          f"{'long, 30 paired':>20s} {'long, all 239':>20s}")
    print("-" * 100)

    def cell(sub, key):
        v = [r[key] for r in sub if r.get(key)]
        if not v:
            return f"{'—':>20s}"
        la = np.array([x["med_la"] for x in v])
        inf = np.mean([x["infeasible"] for x in v])
        return f"{la.mean():+7.2f} ± {la.std():4.2f} [{inf:4.1f}]"

    for arm in ("gbt", "clf", "mlp"):
        groups = sorted({(r["scope"], r["fset_label"]) for r in rows
                         if r["arm"] == arm}, key=lambda g: (g[0] != "all", g[1]))
        for sc, fl in groups:
            sub = [r for r in rows if r["arm"] == arm and r["scope"] == sc
                   and r["fset_label"] == fl]
            who = "all lengths" if sc == "all" else "short+medium"
            print(f"{sub[0]['arm_label']:11s} {who:12s} {fl:5s} "
                  f"{cell(sub, 'in_dist'):>18s} {cell(sub, 'long_paired'):>20s} "
                  f"{cell(sub, 'long_all'):>20s}")
        print()
    print(f"saved: {store}")


if __name__ == "__main__":
    main()
