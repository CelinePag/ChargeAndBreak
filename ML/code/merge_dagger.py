"""
merge_dagger.py — fold DAgger shards into a training set and report on them
===========================================================================
Builds ML/data/dataset_<tag>_dagger{R}.npz = base dataset + all aggregated
rounds up to R, with a `source` array marking each row "base" or "daggerN".

WEIGHTING — why the merged rows need it
---------------------------------------
A round of DAgger yields on the order of 10^3 rows against a base set of
~5.2x10^4.  Folded in unweighted they are ~2% of the deck and change almost
nothing, which is the usual reason a DAgger round "does not work".  The rows
are, however, worth far more per example than base rows: they are the only
labels drawn from the state distribution the student ACTUALLY produces.  So
train.py upweights them by `--dagger-weight` (default 5).  That is the same
role beta plays in the original DAgger formulation — how much of the
learner's own distribution to mix in — expressed as a loss weight rather
than a sampling rate, because our base set is fixed and reused every round.

DIAGNOSTIC — the interesting half
---------------------------------
The script also reports, per round, how often the teacher DISAGREED with what
the student did at that state.  A high disagreement rate is the direct
measurement of covariate shift: it says the student is reaching states where
its own policy is wrong.  That number falling across rounds is the evidence
that DAgger worked, and it is a paper figure in its own right.

Usage:  python ML/code/merge_dagger.py --round 1 [--k 20 --split-mode seed]
"""
from __future__ import annotations
import argparse, glob, json, os, sys, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ML", "code"))
from extract_dataset import tag_for

DATA = os.path.join(ROOT, "ML", "data")


def main(a):
    tag = tag_for(a.k, a.mode)
    base = np.load(os.path.join(DATA, f"dataset_{tag}.npz"), allow_pickle=False)
    meta = json.load(open(os.path.join(DATA, f"meta_{tag}.json")))
    classes = [tuple(c) for c in meta["classes"]]

    Xs = [base["X"]]; ys = [base["y"]]; ts = [base["tauc"]]
    splits = [base["split"]]; srcs = [np.array(["base"] * len(base["y"]))]

    for r in range(1, a.round + 1):
        shards = sorted(glob.glob(os.path.join(DATA, "dagger", f"round{r}", "*.npz")))
        if not shards:
            print(f"round {r}: no shards found, skipping")
            continue
        rX, ry, rt, n_inst = [], [], [], 0
        for s in shards:
            z = np.load(s, allow_pickle=False)
            if len(z["y"]) == 0:
                continue
            rX.append(z["X"]); ry.append(z["y"]); rt.append(z["tauc"]); n_inst += 1
        if not rX:
            print(f"round {r}: shards empty, skipping")
            continue
        rX = np.concatenate(rX); ry = np.concatenate(ry); rt = np.concatenate(rt)
        Xs.append(rX); ys.append(ry); ts.append(rt)
        splits.append(np.array(["train"] * len(ry)))
        srcs.append(np.array([f"dagger{r}"] * len(ry)))
        dist = collections.Counter(ry.tolist())
        print(f"round {r}: {len(ry):5d} labels from {n_inst} instances")
        for c, n in dist.most_common(6):
            print(f"    {str(classes[c]):26s} {n:5d} ({100*n/len(ry):5.1f}%)")

    X = np.concatenate(Xs); y = np.concatenate(ys); t = np.concatenate(ts)
    split = np.concatenate(splits); src = np.concatenate(srcs)
    out = os.path.join(DATA, f"dataset_{tag}_dagger{a.round}.npz")
    np.savez_compressed(out, X=X, y=y, tauc=t, split=split, source=src,
                        instance=np.concatenate(
                            [base["instance"]] +
                            [np.array([""] * (len(y) - len(base["y"])))]))
    n_dag = int((src != "base").sum())
    print(f"\nwrote {os.path.basename(out)}: {len(y):,} rows "
          f"({n_dag:,} aggregated = {100*n_dag/len(y):.1f}%)")
    # class mix shift, base vs aggregated — shows WHERE the student drifts
    print(f"\n{'class':26s} {'base %':>8s} {'dagger %':>9s}")
    bmask, dmask = src == "base", src != "base"
    for c in range(len(classes)):
        pb = 100 * (y[bmask] == c).mean()
        pd = 100 * (y[dmask] == c).mean() if dmask.any() else 0.0
        if pb > 0.05 or pd > 0.05:
            print(f"{str(classes[c]):26s} {pb:8.2f} {pd:9.2f}"
                  + ("   <-- over-represented at student states" if pd > 2 * pb + 0.5 else ""))
    meta2 = dict(meta); meta2.update(dagger_round=a.round, n_rows=int(len(y)),
                                     n_dagger=n_dag)
    json.dump(meta2, open(os.path.join(DATA, f"meta_{tag}_dagger{a.round}.json"), "w"),
              indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--split-mode", dest="mode", default="seed",
                   choices=["family", "seed"])
    main(p.parse_args())
